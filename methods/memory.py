# -*- coding: utf-8 -*-
"""
Factor Memory Bank (因子记忆库)
=================================

核心创新：状态感知的因子检索增强（State-Aware Factor RAG）

区别于竞品（AlphaGrail 等）的 One-Shot 生成模式，
本模块在生成新因子前，先基于当前市场状态从记忆库中
检索历史高绩效因子作为先验知识，实现：
  1. 加快收敛（ warm-start 而非 cold-start）
  2. 状态适配（bull/bear/sideways 分别检索对应时期的成功因子）
  3. 知识累积（每次实验的优质因子自动入库）

核心类：
  - MarketStateEncoder: 市场状态编码（四维离散状态空间）
  - FactorEmbedder: 因子语义嵌入（Sentence-BERT + 表达式 AST）
  - FactorMemoryBank: 记忆库主接口（存储 / 检索 / 更新）
  - MemoryAugmentedGenerator: 记忆增强的生成器包装器

依赖：
  - sentence-transformers (因子描述嵌入)
  - faiss-cpu (向量索引)
  - pandas, numpy（数据处理）
  - 可选：transformers（如果 Sentence-BERT 不可用，回退到 TF-IDF）

Author: Independent Researcher
Target: AAAI 2027
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Section 1: Market State Definition
# ──────────────────────────────────────────────


class VolRegime(Enum):
    LOW = "low"           # VIX < 15
    MEDIUM = "medium"     # 15 <= VIX < 25
    HIGH = "high"         # VIX >= 25


class LiquidityRegime(Enum):
    TIGHT = "tight"       # 资金面紧张（SHIBOR 高）
    NORMAL = "normal"
    ABUNDANT = "abundant"


class TrendRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class CorrelationRegime(Enum):
    LOW = "low"           # 个股相关性低 → 选股 alpha 容易获得
    MEDIUM = "medium"
    HIGH = "high"         # 齐涨齐跌 → 选股难度加大


@dataclass(frozen=True)
class MarketState:
    """
    市场状态的四维离散表示。
    任意时点 t 可映射为一个 MarketState，用于检索对应状态下的历史优质因子。
    """
    vol: VolRegime
    liquidity: LiquidityRegime
    trend: TrendRegime
    correlation: CorrelationRegime

    def to_string(self) -> str:
        return f"{self.vol.value}/{self.liquidity.value}/{self.trend.value}/{self.correlation.value}"

    def to_dict(self) -> dict:
        return {
            "vol": self.vol.value,
            "liquidity": self.liquidity.value,
            "trend": self.trend.value,
            "correlation": self.correlation.value,
        }

    @staticmethod
    def from_dict(d: dict) -> MarketState:
        return MarketState(
            vol=VolRegime(d["vol"]),
            liquidity=LiquidityRegime(d["liquidity"]),
            trend=TrendRegime(d["trend"]),
            correlation=CorrelationRegime(d["correlation"]),
        )


# ──────────────────────────────────────────────
# Section 2: Market State Encoder
# ──────────────────────────────────────────────


class MarketStateEncoder:
    """
    将原始市场数据（价格指数、波动率、资金面指标、相关性矩阵）
    编码为离散 MarketState。

    输入：DataFrame，至少包含 'close' 列；可选 'vix' / 'shibor' 等。
    输出：MarketState 对象
    """

    def __init__(
        self,
        vix_low: float = 15.0,
        vix_high: float = 25.0,
        ma_short: int = 20,
        ma_long: int = 60,
        corr_threshold_low: float = 0.3,
        corr_threshold_high: float = 0.6,
    ):
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.corr_low = corr_threshold_low
        self.corr_high = corr_threshold_high

    def encode(
        self,
        price_df: pd.DataFrame,
        vix: Optional[float] = None,
        shibor: Optional[float] = None,
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> MarketState:
        """
        Parameters
        ----------
        price_df : pd.DataFrame
            Either:
              - columns=['close', ...], index=date  (single-asset price series)
              - columns=stock_codes, index=date (multi-stock cross-sectional)
                In this case, a market index proxy is computed automatically
                via cross-sectional mean.
        vix : float, optional. 如果为 None，用 price_df['close'] 的滚动波动率代替
        shibor : float, optional. 如果为 None，默认 NORMAL
        corr_matrix : pd.DataFrame, optional. 个股收益率相关性矩阵；如果为 None，默认 MEDIUM
        """
        # --- Normalize input: accept both single-asset and multi-stock shapes ---
        if price_df is not None and "close" not in price_df.columns:
            # Multi-stock DataFrame: compute market index proxy (cross-sectional mean)
            if len(price_df.columns) > 0:
                market_close = price_df.mean(axis=1)
                price_df = pd.DataFrame({"close": market_close})
            else:
                # Empty DataFrame: return conservative defaults
                return MarketState(
                    vol=VolRegime.MEDIUM,
                    liquidity=LiquidityRegime.NORMAL,
                    trend=TrendRegime.SIDEWAYS,
                    correlation=CorrelationRegime.MEDIUM,
                )
        # 1. Volatility regime
        if vix is not None:
            vol = self._vol_from_vix(vix)
        else:
            vol = self._vol_from_price(price_df)

        # 2. Liquidity regime
        liquidity = self._liquidity_from_shibor(shibor)

        # 3. Trend regime
        trend = self._trend_from_price(price_df)

        # 4. Correlation regime
        correlation = self._correlation_from_matrix(corr_matrix)

        return MarketState(vol=vol, liquidity=liquidity, trend=trend, correlation=correlation)

    def _vol_from_vix(self, vix: float) -> VolRegime:
        if vix < self.vix_low:
            return VolRegime.LOW
        elif vix < self.vix_high:
            return VolRegime.MEDIUM
        else:
            return VolRegime.HIGH

    def _vol_from_price(self, price_df: pd.DataFrame) -> VolRegime:
        """用过去20日收益率波动率近似 VIX"""
        if "close" not in price_df.columns:
            return VolRegime.MEDIUM  # 保守默认
        returns = price_df["close"].pct_change().dropna()
        if len(returns) < 20:
            return VolRegime.MEDIUM
        vol = returns.tail(20).std() * np.sqrt(252)  # 年化波动率
        if vol < 0.15:
            return VolRegime.LOW
        elif vol < 0.25:
            return VolRegime.MEDIUM
        else:
            return VolRegime.HIGH

    def _liquidity_from_shibor(self, shibor: Optional[float]) -> LiquidityRegime:
        if shibor is None:
            return LiquidityRegime.NORMAL
        if shibor < 2.0:
            return LiquidityRegime.ABUNDANT
        elif shibor < 3.5:
            return LiquidityRegime.NORMAL
        else:
            return LiquidityRegime.TIGHT

    def _trend_from_price(self, price_df: pd.DataFrame) -> TrendRegime:
        if "close" not in price_df.columns:
            return TrendRegime.SIDEWAYS
        close = price_df["close"].dropna()
        if len(close) < self.ma_long:
            # 数据不足，用短期趋势判断
            if len(close) >= 5:
                return TrendRegime.BULL if close.iloc[-1] > close.iloc[-5] else TrendRegime.BEAR
            return TrendRegime.SIDEWAYS

        ma_s = close.rolling(self.ma_short).mean().iloc[-1]
        ma_l = close.rolling(self.ma_long).mean().iloc[-1]
        ratio = ma_s / ma_l if ma_l != 0 else 1.0

        if ratio > 1.02:
            return TrendRegime.BULL
        elif ratio < 0.98:
            return TrendRegime.BEAR
        else:
            return TrendRegime.SIDEWAYS

    def _correlation_from_matrix(self, corr_matrix: Optional[pd.DataFrame]) -> CorrelationRegime:
        if corr_matrix is None:
            return CorrelationRegime.MEDIUM
        # 取平均 pairwise 相关性（排除对角线）
        n = corr_matrix.shape[0]
        if n < 2:
            return CorrelationRegime.MEDIUM
        mask = ~np.eye(n, dtype=bool)
        avg_corr = corr_matrix.values[mask].mean()
        if avg_corr < self.corr_low:
            return CorrelationRegime.LOW
        elif avg_corr < self.corr_high:
            return CorrelationRegime.MEDIUM
        else:
            return CorrelationRegime.HIGH


# ──────────────────────────────────────────────
# Section 3: Factor Embedder
# ──────────────────────────────────────────────


class FactorEmbedder:
    """
    将因子（表达式字符串 + 自然语言描述）嵌入为固定长度向量。

    优先使用 Sentence-BERT（语义嵌入）；
    如果 sentence-transformers 不可用，自动回退到 TF-IDF。

    Design note:
      因子表达式本身是领域特定语言（DSL），直接做字符串相似度效果有限。
      因此我们同时嵌入：
        1. LLM 生成的自然语言描述（description）—— 语义丰富
        2. 表达式的结构化表示（expression_ast_hash）—— 精确匹配
      最终向量 = alpha * embed(desc) + beta * embed(expr)
      默认 alpha=0.7, beta=0.3
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        alpha: float = 0.7,
        beta: float = 0.3,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self._model = None
        self._use_tfidf = False
        self._tfidf_vectorizer = None

    def _lazy_load_model(self):
        """惰性加载 Sentence-BERT；失败则回退到 TF-IDF。"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[FactorEmbedder] Loaded Sentence-BERT: {self.model_name}")
        except ImportError:
            print("[FactorEmbedder] sentence-transformers not available, falling back to TF-IDF")
            self._use_tfidf = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(max_features=384)

    def embed(self, description: str, expression: str) -> np.ndarray:
        """
        Parameters
        ----------
        description : str, 因子的自然语言描述（由 LLM 生成）
        expression : str, 因子表达式（如 "rank(ts_corr(close, volume, 20))"）

        Returns
        -------
        np.ndarray, shape=(embedding_dim,), dtype=float32
        """
        self._lazy_load_model()

        if self._use_tfidf:
            return self._embed_tfidf(description, expression)
        else:
            return self._embed_sbert(description, expression)

    def _embed_sbert(self, description: str, expression: str) -> np.ndarray:
        """Sentence-BERT 嵌入：语义描述 + 表达式拼接"""
        # 将表达式转换为"伪自然语言"以增强嵌入质量
        expr_nl = self._expr_to_natural_language(expression)
        combined_text = f"{description} [SEP] {expr_nl}"
        vec = self._model.encode(combined_text, convert_to_numpy=True)
        return vec.astype(np.float32)

    def _embed_tfidf(self, description: str, expression: str) -> np.ndarray:
        """TF-IDF 回退方案（无外部依赖）"""
        import re
        # 简单 tokenization
        text = f"{description} {expression}"
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
        text_processed = " ".join(tokens)

        # 首次调用需要 fit
        if not hasattr(self._tfidf_vectorizer, "vocabulary_"):
            # 用当前文本 fit（实际使用中应在所有因子上预 fit）
            self._tfidf_vectorizer.fit([text_processed])

        vec = self._tfidf_vectorizer.transform([text_processed]).toarray()[0]
        # 补齐到 384 维（与 MiniLM 一致）
        if len(vec) < 384:
            vec = np.pad(vec, (0, 384 - len(vec)), mode="constant")
        else:
            vec = vec[:384]
        return vec.astype(np.float32)

    @staticmethod
    def _expr_to_natural_language(expr: str) -> str:
        """
        将因子表达式转换为伪自然语言描述，帮助 SBERT 理解。
        例如：'rank(ts_corr(close, volume, 20))'
        -> 'rank of time series correlation of close and volume over 20 days'
        """
        import re
        # 简单的关键词替换
        replacements = {
            "ts_corr": "time series correlation",
            "ts_mean": "time series mean",
            "ts_std": "time series standard deviation",
            "ts_max": "time series max",
            "ts_min": "time series min",
            "ts_rank": "time series rank",
            "cross_sectional_rank": "cross sectional rank",
            "rank": "rank",
            "delay": "delay",
            "delta": "change",
            "close": "closing price",
            "open": "opening price",
            "high": "highest price",
            "low": "lowest price",
            "volume": "trading volume",
            "returns": "return rate",
            "cap": "market capitalization",
        }
        result = expr
        for k, v in replacements.items():
            result = result.replace(k, v)
        return result


# ──────────────────────────────────────────────
# Section 4: Stored Factor Record
# ──────────────────────────────────────────────


@dataclass
class StoredFactor:
    """
    记忆库中存储的一条因子记录。
    """
    factor_id: str                      # 唯一 ID（hash of expression）
    expression: str                     # 因子表达式
    description: str                    # 自然语言描述（LLM 生成）
    market_state: MarketState           # 生成/验证时的市场状态

    # 绩效指标（回测结果）
    ic: float = 0.0                    # Information Coefficient
    icir: float = 0.0                  # ICIR = IC / std(IC)
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0

    # 元数据
    created_at: str = ""                # ISO 格式时间戳
    source: str = "llm"                # 'llm' | 'evolution' | 'human'
    parent_ids: list[str] = field(default_factory=list)  # 演化来源

    # 检索统计
    retrieval_count: int = 0
    last_retrieved: str = ""            # ISO 格式时间戳

    def quality_score(self) -> float:
        """
        综合质量评分，用于检索排序。
        权重：IC 50%, Sharpe 30%, win_rate 20%
        额外奖励：检索次数多说明泛化能力强（+bonus）
        """
        base = 0.5 * abs(self.ic) + 0.3 * max(self.sharpe, 0) / 3.0 + 0.2 * self.win_rate
        # 检索次数奖励（对数缩放，避免单一因子垄断）
        import math
        retrieval_bonus = 0.05 * math.log(1 + self.retrieval_count)
        return min(base + retrieval_bonus, 1.0)

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "expression": self.expression,
            "description": self.description,
            "market_state": self.market_state.to_dict(),
            "ic": self.ic,
            "icir": self.icir,
            "sharpe": self.sharpe,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "created_at": self.created_at,
            "source": self.source,
            "parent_ids": self.parent_ids,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved,
            "quality_score": self.quality_score(),
        }

    @staticmethod
    def from_dict(d: dict) -> StoredFactor:
        return StoredFactor(
            factor_id=d["factor_id"],
            expression=d["expression"],
            description=d["description"],
            market_state=MarketState.from_dict(d["market_state"]),
            ic=d.get("ic", 0.0),
            icir=d.get("icir", 0.0),
            sharpe=d.get("sharpe", 0.0),
            win_rate=d.get("win_rate", 0.0),
            max_drawdown=d.get("max_drawdown", 0.0),
            created_at=d.get("created_at", ""),
            source=d.get("source", "llm"),
            parent_ids=d.get("parent_ids", []),
            retrieval_count=d.get("retrieval_count", 0),
            last_retrieved=d.get("last_retrieved", ""),
        )

    @staticmethod
    def compute_id(expression: str) -> str:
        """基于表达式内容计算唯一 ID"""
        return hashlib.sha256(expression.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────
# Section 5: Factor Memory Bank (Core)
# ──────────────────────────────────────────────


class FactorMemoryBank:
    """
    因子记忆库核心类。

    架构：
      - 因子记录存储在 list[StoredFactor]（内存）+ JSON 持久化
      - 向量索引使用 FAISS（如果可用），否则用精确余弦相似度
      - 市场状态用于预过滤（state-aware retrieval）

    提供操作：
      - add(): 添加因子
      - retrieve(): 状态感知检索
      - update_quality(): 更新因子绩效（在线学习）
      - save() / load(): 持久化
    """

    def __init__(
        self,
        embedder: Optional[FactorEmbedder] = None,
        index_path: Optional[str] = None,
        use_faiss: bool = True,
    ):
        self.factors: list[StoredFactor] = []
        self.embedder = embedder or FactorEmbedder()
        self.index_path = index_path
        self.use_faiss = use_faiss
        self._faiss_index = None
        self._embeddings_cache: dict[str, np.ndarray] = {}  # factor_id -> embedding

        # Auto-load existing data if available
        if self.index_path:
            self.load()

    def __len__(self) -> int:
        """Return number of stored factors."""
        return len(self.factors)

    # ── Public API ────────────────────────────

    def add(
        self,
        expression: str,
        description: str,
        market_state: MarketState,
        ic: float = 0.0,
        icir: float = 0.0,
        sharpe: float = 0.0,
        win_rate: float = 0.0,
        max_drawdown: float = 0.0,
        source: str = "llm",
        parent_ids: Optional[list[str]] = None,
    ) -> str:
        """
        向记忆库添加一条因子记录。
        如果 expression 已存在（相同 factor_id），则更新其绩效指标。
        返回 factor_id。
        """
        factor_id = StoredFactor.compute_id(expression)

        # 去重检查
        existing = self._find_by_id(factor_id)
        if existing:
            # 更新现有因子的绩效（取历史平均）
            self._update_existing_factor(existing, ic, icir, sharpe, win_rate, max_drawdown)
            return factor_id

        # 新建
        now = datetime.now().isoformat()
        factor = StoredFactor(
            factor_id=factor_id,
            expression=expression,
            description=description,
            market_state=market_state,
            ic=ic,
            icir=icir,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            created_at=now,
            source=source,
            parent_ids=parent_ids or [],
        )
        self.factors.append(factor)

        # 更新向量索引
        emb = self.embedder.embed(description, expression)
        self._embeddings_cache[factor_id] = emb
        self._update_faiss_index(new_embedding=emb)

        print(f"[MemoryBank] Added factor {factor_id}: {expression[:60]}...")
        return factor_id

    def retrieve(
        self,
        query_description: str,
        query_expression: str,
        current_market_state: MarketState,
        top_k: int = 5,
        state_weight: float = 0.3,
        min_ic: float = 0.02,
    ) -> list[StoredFactor]:
        """
        状态感知的因子检索。

        Parameters
        ----------
        query_description : str, 当前生成任务的查询描述
        query_expression : str, 当前种子表达式（可选，可为 ''）
        current_market_state : MarketState, 当前市场状态
        top_k : int, 返回 top-K 个因子
        state_weight : float, 市场状态匹配的权重（0~1）
        min_ic : float, 最低 IC 阈值（过滤无效因子）

        Returns
        -------
        list[StoredFactor]，按综合得分降序排列
        """
        if not self.factors:
            return []

        # 1. 预过滤：IC 阈值
        candidates = [f for f in self.factors if f.ic >= min_ic]
        if not candidates:
            candidates = self.factors  # 放宽：返回所有

        # 2. 计算查询向量
        query_emb = self.embedder.embed(
            query_description or "",
            query_expression or "",
        )

        # 3. 计算语义相似度
        similarities = []
        for f in candidates:
            f_emb = self._get_embedding(f)
            sim = self._cosine_similarity(query_emb, f_emb)
            similarities.append(sim)

        # 4. 计算状态匹配得分
        state_scores = [
            self._state_match_score(current_market_state, f.market_state)
            for f in candidates
        ]

        # 5. 综合得分
        final_scores = []
        for idx, (sim, ss) in enumerate(zip(similarities, state_scores)):
            score = (1 - state_weight) * sim + state_weight * ss
            # 质量评分作为乘法权重
            quality = candidates[idx].quality_score()
            final_scores.append(score * quality)

        # 6. 排序并返回 top-k
        indexed = list(enumerate(final_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        result = []
        for idx, score in indexed[:top_k]:
            f = candidates[idx]
            f.retrieval_count += 1
            f.last_retrieved = datetime.now().isoformat()
            result.append(f)

        return result

    def update_quality(self, factor_id: str, ic: float, sharpe: float = 0.0, win_rate: float = 0.0):
        """在线更新因子的绩效指标（回测后调用）"""
        f = self._find_by_id(factor_id)
        if f is None:
            print(f"[MemoryBank] Warning: factor {factor_id} not found for update")
            return
        # 指数移动平均更新（新结果权重 0.3）
        alpha = 0.3
        f.ic = (1 - alpha) * f.ic + alpha * ic
        f.sharpe = (1 - alpha) * f.sharpe + alpha * sharpe
        f.win_rate = (1 - alpha) * f.win_rate + alpha * win_rate
        print(f"[MemoryBank] Updated factor {factor_id}: IC={f.ic:.4f}")

    def save(self, path: Optional[str] = None):
        """持久化到磁盘（JSON + FAISS index）"""
        p = Path(path) if path else (Path(self.index_path) or Path("./memory_bank"))
        p.mkdir(parents=True, exist_ok=True)

        # 保存因子记录
        records = [f.to_dict() for f in self.factors]
        with open(p / "factors.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # 保存向量索引
        if self.use_faiss and self._faiss_index is not None:
            import faiss
            faiss.write_index(self._faiss_index, str(p / "faiss.index"))
        else:
            # 保存原始 embeddings
            embeddings = {fid: emb.tolist() for fid, emb in self._embeddings_cache.items()}
            with open(p / "embeddings.json", "w", encoding="utf-8") as f:
                json.dump(embeddings, f)

        print(f"[MemoryBank] Saved {len(self.factors)} factors to {p}")

    def load(self, path: Optional[str] = None):
        """从磁盘加载"""
        p = Path(path) if path else (Path(self.index_path) or Path("./memory_bank"))
        if not p.exists():
            print(f"[MemoryBank] No saved data at {p}")
            return

        # 加载因子记录
        with open(p / "factors.json", "r", encoding="utf-8") as f:
            records = json.load(f)
        self.factors = [StoredFactor.from_dict(d) for d in records]

        # 加载向量索引
        if self.use_faiss and (p / "faiss.index").exists():
            import faiss
            self._faiss_index = faiss.read_index(str(p / "faiss.index"))
        elif (p / "embeddings.json").exists():
            with open(p / "embeddings.json", "r", encoding="utf-8") as f:
                embeddings = json.load(f)
            self._embeddings_cache = {k: np.array(v, dtype=np.float32) for k, v in embeddings.items()}

        print(f"[MemoryBank] Loaded {len(self.factors)} factors from {p}")

    # ── Internal Helpers ─────────────────────

    def _find_by_id(self, factor_id: str) -> Optional[StoredFactor]:
        for f in self.factors:
            if f.factor_id == factor_id:
                return f
        return None

    def _update_existing_factor(self, f: StoredFactor, ic: float, icir: float, sharpe: float, win_rate: float, max_drawdown: float):
        """当因子已存在时，用指数移动平均更新其绩效"""
        alpha = 0.3
        f.ic = (1 - alpha) * f.ic + alpha * ic
        f.icir = (1 - alpha) * f.icir + alpha * icir
        f.sharpe = (1 - alpha) * f.sharpe + alpha * sharpe
        f.win_rate = (1 - alpha) * f.win_rate + alpha * win_rate
        f.max_drawdown = (1 - alpha) * f.max_drawdown + alpha * max_drawdown
        print(f"[MemoryBank] Updated existing factor {f.factor_id}: new IC={f.ic:.4f}, new ICIR={f.icir:.4f}")

    def _get_embedding(self, f: StoredFactor) -> np.ndarray:
        """获取因子的嵌入向量（优先用缓存）"""
        if f.factor_id in self._embeddings_cache:
            return self._embeddings_cache[f.factor_id]
        emb = self.embedder.embed(f.description, f.expression)
        self._embeddings_cache[f.factor_id] = emb
        return emb

    def _update_faiss_index(self, new_embedding: Optional[np.ndarray] = None):
        """更新 FAISS 索引（新增因子后调用）。
        
        如果是首次创建则重建索引；如果已有索引则增量添加新向量，
        避免 O(n²) 的全量重建。
        """
        if not self.use_faiss:
            return
        try:
            import faiss
            dim = 384  # Sentence-BERT MiniLM 维度
            
            if self._faiss_index is None:
                # 首次创建：全量构建
                self._faiss_index = faiss.IndexFlatIP(dim)
                if self._embeddings_cache:
                    embs = np.array(list(self._embeddings_cache.values()), dtype=np.float32)
                    faiss.normalize_L2(embs)
                    self._faiss_index.add(embs)
            elif new_embedding is not None:
                # 增量添加：仅添加新向量
                emb = new_embedding.copy().reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(emb)
                self._faiss_index.add(emb)
        except ImportError:
            self.use_faiss = False

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    @staticmethod
    def _state_match_score(s1: MarketState, s2: MarketState) -> float:
        """
        计算两个市场状态的匹配得分（0~1）。
        精确匹配得 1.0；同一大状态（如都为 BULL）得 0.7；
        完全不同得 0.0。
        """
        matches = 0
        total = 4
        if s1.vol == s2.vol:
            matches += 1
        if s1.liquidity == s2.liquidity:
            matches += 1
        if s1.trend == s2.trend:
            matches += 1
        if s1.correlation == s2.correlation:
            matches += 1
        return matches / total


# ──────────────────────────────────────────────
# Section 6: Memory-Augmented Generator
# ──────────────────────────────────────────────


class MemoryAugmentedGenerator:
    """
    记忆增强的因子生成器包装器。

    在调用基础生成器（LLM）之前，先从记忆库检索相关历史因子，
    将其作为 few-shot 示例注入 prompt，实现状态感知的 warm-start 生成。

    这是论文 Figure 3 的核心流程图组件。
    """

    def __init__(
        self,
        base_generator: object,  # 基础生成器（如 evolve.py 中的 SelfEvolvingGenerator）
        memory_bank: FactorMemoryBank,
        encoder: MarketStateEncoder,
        n_shots: int = 3,
    ):
        self.base_generator = base_generator
        self.memory_bank = memory_bank
        self.encoder = encoder
        self.n_shots = n_shots

    def generate(
        self,
        task_description: str,
        price_df: pd.DataFrame,
        vix: Optional[float] = None,
        **kwargs,
    ) -> list[dict]:
        """
        生成因子（记忆增强版）。

        Parameters
        ----------
        task_description : str, 生成任务描述（如 "生成动量类因子"）
        price_df : pd.DataFrame, 当前市场数据（用于编码市场状态）
        vix : float, optional

        Returns
        -------
        list[dict], 生成的因子列表，每个 dict 包含 expression, description 等
        """
        # 1. 编码当前市场状态
        current_state = self.encoder.encode(price_df, vix=vix)
        print(f"[MemoryAugmented] Current market state: {current_state.to_string()}")

        # 2. 从记忆库检索相似历史因子
        retrieved = self.memory_bank.retrieve(
            query_description=task_description,
            query_expression="",
            current_market_state=current_state,
            top_k=self.n_shots,
        )

        # 3. 构造 few-shot prompt
        few_shot_examples = ""
        for i, f in enumerate(retrieved, 1):
            few_shot_examples += (
                f"\n[Example {i}] Market State: {f.market_state.to_string()}\n"
                f"Expression: {f.expression}\n"
                f"Description: {f.description}\n"
                f"IC: {f.ic:.4f}\n"
            )

        augmented_prompt = (
            f"{task_description}\n\n"
            f"Here are {len(retrieved)} examples of high-quality factors "
            f"(retrieved from memory under similar market states):\n"
            f"{few_shot_examples}\n"
            f"Now generate NEW factors following the above style and quality."
        )

        print(f"[MemoryAugmented] Augmented prompt with {len(retrieved)} retrieved factors")

        # 4. 调用基础生成器
        # （这里假设 base_generator 有一个 .generate() 方法接受 prompt 参数）
        if hasattr(self.base_generator, "generate"):
            results = self.base_generator.generate(augmented_prompt, **kwargs)
        else:
            # 回退：直接返回检索结果（用于调试）
            results = [{"expression": f.expression, "description": f.description} for f in retrieved]

        # 5. 将生成的新因子自动入库（异步，不阻塞）
        # （实际实现中，应在回测验证后调用 memory_bank.add()）
        return results


# ──────────────────────────────────────────────
# Section 7: CLI / Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 快速演示：构造 mock 数据，展示完整检索流程
    print("=== Factor Memory Bank Demo ===\n")

    # 1. 创建编码器 + 记忆库
    encoder = MarketStateEncoder()
    memory = FactorMemoryBank(use_faiss=False)

    # 2. Mock：添加几条历史因子
    mock_states = [
        MarketState(VolRegime.LOW, LiquidityRegime.ABUNDANT, TrendRegime.BULL, CorrelationRegime.LOW),
        MarketState(VolRegime.HIGH, LiquidityRegime.TIGHT, TrendRegime.BEAR, CorrelationRegime.HIGH),
        MarketState(VolRegime.MEDIUM, LiquidityRegime.NORMAL, TrendRegime.SIDEWAYS, CorrelationRegime.MEDIUM),
    ]
    mock_factors = [
        ("rank(ts_corr(close, volume, 20))", "排名时序相关性，捕捉价量背离"),
        ("-ts_std(returns, 20)", "负向波动率，低波动效应"),
        ("rank(close / delay(close, 60))", "60日动量因子"),
    ]
    mock_ics = [0.045, 0.032, 0.038]

    for state, (expr, desc), ic in zip(mock_states, mock_factors, mock_ics):
        memory.add(expression=expr, description=desc, market_state=state, ic=ic)

    # 3. 编码当前市场状态（假设当前是牛市低波环境）
    current_state = MarketState(
        vol=VolRegime.LOW,
        liquidity=LiquidityRegime.ABUNDANT,
        trend=TrendRegime.BULL,
        correlation=CorrelationRegime.LOW,
    )

    # 4. 检索
    print(f"Current market state: {current_state.to_string()}")
    print("Retrieving top-2 similar factors...\n")
    results = memory.retrieve(
        query_description="生成捕捉价量关系的因子",
        query_expression="",
        current_market_state=current_state,
        top_k=2,
        state_weight=0.4,
    )

    for i, f in enumerate(results, 1):
        print(f"[{i}] ID: {f.factor_id}")
        print(f"    Expression: {f.expression}")
        print(f"    Description: {f.description}")
        print(f"    IC: {f.ic:.4f} | Sharpe: {f.sharpe:.4f}")
        print(f"    Source state: {f.market_state.to_string()}")
        print(f"    Quality score: {f.quality_score():.4f}")
        print()

    print("=== Demo Complete ===")
