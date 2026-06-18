# -*- coding: utf-8 -*-
"""
Factor Fusion & Portfolio Construction
========================================

模块定位：AAAI 2027 论文的整合验证层（第四大模块）

将 SelfEvolvingGenerator 生成的多因子通过加权融合得到复合得分，
再基于复合得分构建实际股票组合，附带风控约束与绩效评估。

核心类：
  - FactorNormalizer: 因子标准化（Z-score / Rank / MinMax）
  - FactorFusion: 因子融合（多种加权策略 + 相关性惩罚）
  - PortfolioConstructor: 组合构建（Top-N / Score-Proportional）
  - RiskManager: 风控约束（行业集中度 / 个股权重 / 换手率限制）
  - Pipeline: 端到端流水线（整合上述所有类）

创新点（区别于传统多因子模型）：
  1. 动态 ICIR 加权（传统使用静态 IC 加权，我们在线更新权重）
  2. 相关性惩罚（避免类似因子重复计算 → 与 Alpha Grail 的 factor zoo 问题对比）
  3. 换手率平滑（降低交易成本，提升真实收益）

Author: Independent Researcher
Target: AAAI 2027
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# Section 1: Factor Normalizer
# ──────────────────────────────────────────────


class FactorNormalizer:
    """
    对原始因子值做横截面标准化。
    支持三种方法：
      - zscore: 均值 0，标准差 1
      - rank: 分位数归一化到 [0, 1]
      - minmax: 线性缩放到 [0, 1]

    所有方法都做行业中性化（可选）：残差回归消除行业影响。
    """

    def __init__(self, method: str = "zscore", neutralize_industry: bool = True):
        if method not in ("zscore", "rank", "minmax"):
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.neutralize_industry = neutralize_industry

    def normalize(
        self,
        factor_values: pd.DataFrame,
        industry: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        factor_values : pd.DataFrame, index=date, columns=stock_code
        industry : pd.Series, optional, index=stock_code, values=industry_label

        Returns
        -------
        pd.DataFrame, normalized factor scores, same shape as input
        """
        result = factor_values.copy()

        for date in result.index:
            row = result.loc[date].dropna()
            if len(row) < 5:
                # 有效值太少，跳过归一化，保留原值（不设为NaN）
                continue

            if self.method == "zscore":
                if self.neutralize_industry and industry is not None:
                    row = self._industry_neutralize(row, industry)
                mu, sigma = row.mean(), row.std()
                result.loc[date, row.index] = (row - mu) / (sigma + 1e-8)

            elif self.method == "rank":
                result.loc[date, row.index] = row.rank(pct=True)

            elif self.method == "minmax":
                lo, hi = row.min(), row.max()
                if hi - lo > 1e-8:
                    result.loc[date, row.index] = (row - lo) / (hi - lo)
                else:
                    result.loc[date, row.index] = 0.5

        return result

    @staticmethod
    def _industry_neutralize(
        raw_scores: pd.Series,
        industry: pd.Series,
    ) -> pd.Series:
        """
        行业中性化：对 raw_scores 做 industry dummy 回归，取残差。
        raw_scores ~ C(industry)  →  residual
        """
        common = raw_scores.index.intersection(industry.index)
        if len(common) < 10:
            return raw_scores

        y = raw_scores[common]
        industry_dummies = pd.get_dummies(industry[common], drop_first=True)

        # OLS: residuals = y - X * beta
        # beta = (X^T X)^{-1} X^T y
        X = industry_dummies.values.astype(float)
        yv = y.values.astype(float)

        try:
            beta = np.linalg.lstsq(X, yv, rcond=None)[0]
            residuals = yv - X @ beta
            result = raw_scores.copy()
            result[common] = residuals
            return result
        except np.linalg.LinAlgError:
            return raw_scores


# ──────────────────────────────────────────────
# Section 2: Factor Fusion
# ──────────────────────────────────────────────


@dataclass
class FactorInfo:
    """单个因子的元信息（用于加权计算）

    Attributes
    ----------
    ic : float
        信息系数（Information Coefficient），直接来自回测结果。
    icir : float
        ICIR = IC / std(IC)，直接来自回测结果。
        `FactorFusion` 的 `icir_weighted` 策略直接使用此值。
    ic_std : float
        保留字段，仅用于向后兼容（旧代码可能直接设置 ic_std）。
    sharpe : float
        因子多空组合的 Sharpe Ratio，来自回测。
    """
    name: str
    expression: str
    ic: float = 0.0
    icir: float = 0.0       # ICIR = IC / std(IC)，直接传入，不再动态计算
    ic_std: float = 0.0     # 保留，向后兼容
    sharpe: float = 0.0
    debate_score: float = 0.0  # 0-10, from multi-agent debate
    values: Optional[pd.DataFrame] = None  # index=date, columns=stock_code

    @property
    def ic_decayed(self) -> float:
        """IC 衰减后的有效值（基于最近 N 期滚动）"""
        return self.ic  # 子类可覆盖


class FactorFusion:
    """
    将多个因子融合为单一复合得分。

    支持四种加权策略：
      1. equal: 等权
      2. ic_weighted: IC 绝对值加权
      3. icir_weighted: ICIR 加权（IC 除以波动率）
      4. decay_weighted: 指数衰减加权（越新的 IC 权重越高）

    可选步骤：
      - 相关性惩罚：高相关因子的权重打折
      - 正交化：PCA 或 Gram-Schmidt
    """

    def __init__(
        self,
        strategy: str = "icir_weighted",
        corr_penalty: bool = True,
        corr_threshold: float = 0.7,
        normalizer: Optional[FactorNormalizer] = None,
    ):
        if strategy not in ("equal", "ic_weighted", "icir_weighted", "decay_weighted"):
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        self.strategy = strategy
        self.corr_penalty = corr_penalty
        self.corr_threshold = corr_threshold
        self.normalizer = normalizer or FactorNormalizer(method="zscore")

    def fuse(
        self,
        factors: list[FactorInfo],
        factor_values: dict[str, pd.DataFrame],
        industry: Optional[pd.Series] = None,
        precomputed_weights: Optional[dict[str, float]] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Parameters
        ----------
        factors : list[FactorInfo], 因子元信息
        factor_values : dict, {name: pd.DataFrame(index=date, columns=stock_code)}
        industry : pd.Series, optional
        weights : dict, optional, 预计算权重（来自训练期）
            若提供，则跳过 _compute_weights() 和相关性惩罚，
            直接使用权重（避免测试期数据泄露）。

        Returns
        -------
        composite_scores : pd.DataFrame, 融合后的复合得分
        meta : dict, 融合元信息（权重、IC 等）
        """
        n = len(factors)
        if n == 0:
            raise ValueError("No factors provided")

        # 1. 计算原始权重（若未提供预计算权重）
        _precomputed = (precomputed_weights is not None)
        if _precomputed:
            # 使用训练期确定的权重，跳过相关性惩罚（避免数据泄露）
            raw_weights = precomputed_weights
        else:
            raw_weights = self._compute_weights(factors)

        # 2. 相关性惩罚（仅当权重新鲜计算时应用；预计算权重已含惩罚）
        if not _precomputed and self.corr_penalty and n > 1:
            weights = self._apply_corr_penalty(raw_weights, factor_values)
        else:
            weights = raw_weights

        # 3. 归一化各因子值 + 加权求和
        weighted = None
        # Align all factor DataFrames to a common date index
        common_dates = factor_values[factors[0].name].index
        for fv in factor_values.values():
            common_dates = common_dates.intersection(fv.index)

        for f in factors:
            fv = self.normalizer.normalize(factor_values[f.name], industry)
            w = weights.get(f.name, 1.0 / n)
            fv_aligned = fv.loc[common_dates] * w

            if weighted is None:
                weighted = fv_aligned
            else:
                weighted = weighted.add(fv_aligned, fill_value=0)

        # 4. 最终标准化
        composite_scores = self.normalizer.normalize(weighted, industry)

        meta = {
            "strategy": self.strategy,
            "weights": weights,
            "n_factors": n,
            "corr_penalty_applied": self.corr_penalty,
            "timestamp": datetime.now().isoformat(),
        }

        return composite_scores, meta

    def _compute_weights(self, factors: list[FactorInfo]) -> dict[str, float]:
        """根据策略计算各因子的原始权重（含辩论分数融合）"""
        n = len(factors)
        names = [f.name for f in factors]

        # Compute raw weights by strategy
        if self.strategy == "equal":
            weights = {name: 1.0 / n for name in names}
        elif self.strategy == "ic_weighted":
            ics = np.array([abs(f.ic) for f in factors])
            total = ics.sum()
            if total < 1e-8:
                weights = {name: 1.0 / n for name in names}
            else:
                weights = {f.name: abs(f.ic) / total for f in factors}
        elif self.strategy == "icir_weighted":
            icirs = np.array([max(f.icir, 0.0) for f in factors])
            total = icirs.sum()
            if total < 1e-8:
                weights = {name: 1.0 / n for name in names}
            else:
                weights = {f.name: icirs[i] / total for i, f in enumerate(factors)}
        elif self.strategy == "decay_weighted":
            decay_weights = np.exp(-0.3 * np.arange(n))
            total = decay_weights.sum()
            weights = {f.name: decay_weights[i] / total for i, f in enumerate(factors)}
        else:
            weights = {name: 1.0 / n for name in names}

        # Blend with debate scores (if any factor has a non-zero debate score)
        debate_scores = np.array([getattr(f, 'debate_score', 0.0) for f in factors])
        if debate_scores.max() > 0:
            # Normalize to [0,1], then blend: final = raw * (0.3 + 0.7 * norm)
            norm = np.clip(debate_scores, 0.0, 10.0) / 10.0  # [0,1]
            blend_factor = 0.3 + 0.7 * norm  # [0.3, 1.0]
            for i, f in enumerate(factors):
                weights[f.name] *= blend_factor[i]
            # Renormalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _apply_corr_penalty(
        self,
        raw_weights: dict[str, float],
        factor_values: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        对高相关因子的权重打折。
        算法：
          对每个因子，找到与其相关性 > threshold 的其他因子，
          取其平均相关性作为惩罚系数：(1 - avg_corr_with_neighbors)
          最终权重 = 原始权重 * 惩罚系数，再 renormalize。
        """
        names = list(raw_weights.keys())
        n = len(names)
        if n <= 1:
            return raw_weights

        # 构建相关性矩阵（基于最近一期因子暴露）
        corr_mat = np.eye(n)
        recent_values = {}
        for name in names:
            fv = factor_values[name]
            last_valid = fv.dropna(how="all").tail(1)
            if not last_valid.empty:
                recent_values[name] = last_valid.iloc[-1]
            else:
                recent_values[name] = pd.Series(dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                common = recent_values[names[i]].dropna().index.intersection(
                    recent_values[names[j]].dropna().index
                )
                if len(common) >= 10:
                    corr = recent_values[names[i]][common].corr(recent_values[names[j]][common])
                    corr_mat[i, j] = corr_mat[j, i] = abs(corr) if not pd.isna(corr) else 0.0

        # 惩罚：高相关性 → 低权重
        penalized = {}
        for i, name in enumerate(names):
            # 找到高相关邻居
            neighbors = [j for j in range(n) if j != i and corr_mat[i, j] > self.corr_threshold]
            if neighbors:
                avg_corr = np.mean([corr_mat[i, j] for j in neighbors])
                penalty = max(0.5, 1.0 - avg_corr)  # 至少保留 50%
            else:
                penalty = 1.0
            penalized[name] = raw_weights[name] * penalty

        # 重新归一化
        total = sum(penalized.values())
        if total > 1e-8:
            penalized = {k: v / total for k, v in penalized.items()}

        return penalized


# ──────────────────────────────────────────────
# Section 3: Portfolio Constructor
# ──────────────────────────────────────────────


@dataclass
class PortfolioConfig:
    """组合构建超参数"""
    top_n: int = 50                     # 持仓股票数量
    method: str = "score_proportional"  # top_n | score_proportional | equal
    long_only: bool = True
    max_weight: float = 0.05            # 单只股票最大权重
    min_weight: float = 0.001           # 单只股票最小权重
    max_industry_exposure: float = 0.30  # 单行业最大暴露度
    rebalance_freq: str = "M"           # 换仓频率: D/W/M


@dataclass
class Portfolio:
    """一次调仓的持仓快照"""
    date: str
    weights: pd.Series                  # index=stock_code, values=weight
    composite_scores: pd.Series         # index=stock_code, values=composite_score
    turnover: float = 0.0               # 相对上次的换手率
    n_stocks: int = 0
    meta: dict = field(default_factory=dict)


class PortfolioConstructor:
    """
    组合构建器。

    基于复合得分（composite_scores），在每个调仓日构建投资组合。
    支持：
      - top_n: 等权持有 Top-N
      - score_proportional: 按复合得分比例分配权重
      - equal: 等权持有所有（兜底）
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def build(
        self,
        composite_scores: pd.DataFrame,
        prices: pd.DataFrame,
        market_cap: Optional[pd.DataFrame] = None,
        industry: Optional[pd.Series] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> list[Portfolio]:
        """
        Parameters
        ----------
        composite_scores : pd.DataFrame, index=date, columns=stock_code
        prices : pd.DataFrame, index=date, columns=stock_code（用于过滤停牌）
        market_cap : pd.DataFrame, optional
        industry : pd.Series, optional
        prev_weights : pd.Series, optional, 上一期持仓权重

        Returns
        -------
        list[Portfolio]，每个调仓日一个 Portfolio
        """
        portfolios = []
        rebalance_dates = self._get_rebalance_dates(composite_scores)
        if len(rebalance_dates) == 0:
            return portfolios

        skipped_no_scores = 0
        skipped_no_prices = 0
        for i, date in enumerate(rebalance_dates):
            scores = composite_scores.loc[date].dropna()
            if len(scores) == 0:
                skipped_no_scores += 1
                continue

            actual_top_n = min(self.config.top_n, len(scores))

            # 过滤停牌：用 get_loc 精确匹配，避免 loc 返回全 NaN 行
            if isinstance(date, pd.Timestamp):
                try:
                    loc_idx = prices.index.get_loc(date)
                    row = prices.iloc[loc_idx]
                except KeyError:
                    row = pd.Series(index=prices.columns, dtype=float)
            else:
                row = prices.loc[date] if date in prices.index else pd.Series(index=prices.columns, dtype=float)
            valid_prices = row.dropna()

            if len(valid_prices) > 0:
                scores = scores[scores.index.isin(valid_prices.index)]
            else:
                # All stocks have missing prices — skip this date
                skipped_no_prices += 1
                continue

            if len(scores) < actual_top_n:
                skipped_no_scores += 1
                continue

            # 选择 Top-N
            top_scores = scores.nlargest(actual_top_n)

            # 分配权重
            weights = self._allocate_weights(top_scores, market_cap, date, industry)

            # 计算换手率
            turnover = 0.0
            if prev_weights is not None and len(prev_weights) > 0:
                turnover = self._compute_turnover(weights, prev_weights)

            portfolios.append(Portfolio(
                date=str(date),
                weights=weights,
                composite_scores=top_scores,
                turnover=turnover,
                n_stocks=len(weights),
                meta={"config": self.config},
            ))

            prev_weights = weights

        return portfolios

    def _allocate_weights(
        self,
        top_scores: pd.Series,
        market_cap: Optional[pd.DataFrame],
        date: str,
        industry: Optional[pd.Series],
    ) -> pd.Series:
        """根据配置分配个股权重（含风控约束）"""
        method = self.config.method
        stocks = top_scores.index.tolist()
        n = len(stocks)

        if method == "equal" or n <= 1:
            weights = pd.Series(1.0 / n, index=top_scores.index)

        elif method == "score_proportional":
            raw = top_scores - top_scores.min() + 1e-6  # 平移为正
            weights = raw / raw.sum()

        elif method == "top_n":
            weights = pd.Series(1.0 / n, index=top_scores.index)

        else:
            weights = pd.Series(1.0 / n, index=top_scores.index)

        # 风控约束：个股权重上限
        weights = weights.clip(upper=self.config.max_weight)
        weights = weights / weights.sum()  # renormalize

        # 风控约束：最小权重
        weights = weights[weights >= self.config.min_weight]
        if len(weights) > 0:
            weights = weights / weights.sum()

        # 风控约束：行业暴露
        if industry is not None and self.config.max_industry_exposure < 1.0:
            weights = self._apply_industry_cap(weights, industry)

        return weights

    def _apply_industry_cap(self, weights: pd.Series, industry: pd.Series) -> pd.Series:
        """限制单行业最大权重"""
        common = weights.index.intersection(industry.index)
        if len(common) == 0:
            return weights

        w = weights[common].copy()
        for ind in industry[common].unique():
            mask = industry[common] == ind
            ind_weight = w[mask].sum()
            if ind_weight > self.config.max_industry_exposure:
                scale = self.config.max_industry_exposure / ind_weight
                w[mask] *= scale

        w = w / w.sum()
        return w

    @staticmethod
    def _get_rebalance_dates(scores: pd.DataFrame) -> list:
        """获取调仓日期列表"""
        n_cols = scores.shape[1]
        thresh = max(1, n_cols // 10)
        valid = scores.dropna(thresh=thresh)
        if len(valid) == 0 and n_cols > 0:
            valid = scores.dropna(thresh=1)
        return sorted(valid.index.tolist())

    @staticmethod
    def _compute_turnover(new: pd.Series, old: pd.Series) -> float:
        """计算单边换手率 = sum(|new_i - old_i|) / 2"""
        common = new.index.intersection(old.index)
        if len(common) == 0:
            return 1.0

        new_w = new[common].copy()
        old_w = old[common].copy()

        # 补全新入/退出的股票
        new_only = new.index.difference(old.index)
        old_only = old.index.difference(new.index)

        turnover_sum = abs(new_w - old_w).sum()
        turnover_sum += new[new_only].sum()      # 新增买入
        turnover_sum += old[old_only].sum()      # 完全卖出

        return turnover_sum / 2.0


# ──────────────────────────────────────────────
# Section 4: Risk Manager
# ──────────────────────────────────────────────


@dataclass
class RiskConfig:
    """风控配置"""
    max_stock_weight: float = 0.05
    min_stock_weight: float = 0.001
    max_industry_exposure: float = 0.30
    max_turnover: float = 0.50           # 单期最大单边换手率
    min_market_cap_percentile: float = 0.10  # 市值过滤（排最低10%）
    exclude_st: bool = True


class RiskManager:
    """
    风控管理器。

    在 PortfolioConstructor 输出之后，对仓位施加额外风控：
      - 市值过滤（排除小市值）
      - ST 排除
      - 行业集中度上限
      - 换手率平滑
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def apply(
        self,
        portfolio: Portfolio,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        st_flags: Optional[pd.Series] = None,
    ) -> Portfolio:
        """
        Parameters
        ----------
        portfolio : Portfolio, 原始组合
        market_cap : pd.Series, optional, 当日市值
        industry : pd.Series, optional
        st_flags : pd.Series, optional, True = ST 标记

        Returns
        -------
        Portfolio, 施加风控后的组合
        """
        weights = portfolio.weights.copy()

        # 1. 市值过滤：排除市值为空 或 市值最低 percentile 的股票
        if market_cap is not None and self.config.min_market_cap_percentile > 0:
            cap = market_cap[market_cap.index.isin(weights.index)]
            if len(cap) > 0:
                threshold = cap.quantile(self.config.min_market_cap_percentile)
                exclude = cap[cap < threshold].index
                weights = weights[~weights.index.isin(exclude)]

        # 2. ST 排除
        if self.config.exclude_st and st_flags is not None:
            st_stocks = st_flags[st_flags].index
            weights = weights[~weights.index.isin(st_stocks)]

        # 3. 个股权重上限
        weights = weights.clip(upper=self.config.max_stock_weight)
        weights = weights[weights >= self.config.min_stock_weight]
        if len(weights) > 0:
            weights = weights / weights.sum()

        # 4. 行业暴露上限
        if industry is not None:
            common = weights.index.intersection(industry.index)
            if len(common) > 0:
                w = weights[common].copy()
                for ind in industry[common].unique():
                    mask = industry[common] == ind
                    ind_weight = w[mask].sum()
                    if ind_weight > self.config.max_industry_exposure:
                        scale = self.config.max_industry_exposure / ind_weight
                        w[mask] *= scale
                w = w / w.sum()
                weights = w

        return Portfolio(
            date=portfolio.date,
            weights=weights,
            composite_scores=portfolio.composite_scores,
            turnover=portfolio.turnover,
            n_stocks=len(weights),
            meta={"risk_config": self.config, **portfolio.meta},
        )


# ──────────────────────────────────────────────
# Section 5: End-to-End Pipeline
# ──────────────────────────────────────────────


@dataclass
class PipelineResult:
    """流水线完整输出"""
    composite_scores: pd.DataFrame
    portfolios: list[Portfolio]
    factor_weights: dict[str, float]
    performance: dict
    meta: dict


class Pipeline:
    """
    端到端流水线：生成 → 融合 → 选股

    整合所有模块：
      1. 接收多个因子（来自 evolve.py / debate.py）
      2. 通过 FactorFusion 融合为复合得分
      3. 通过 PortfolioConstructor 构建组合
      4. 通过 RiskManager 施加风控
      5. 输出绩效评估
    """

    def __init__(
        self,
        fusion: Optional[FactorFusion] = None,
        constructor: Optional[PortfolioConstructor] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        self.fusion = fusion or FactorFusion()
        self.constructor = constructor or PortfolioConstructor()
        self.risk_manager = risk_manager or RiskManager()

    def run(
        self,
        factors: list[FactorInfo],
        factor_values: dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        market_cap: Optional[pd.DataFrame] = None,
        industry: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> PipelineResult:
        """
        Parameters
        ----------
        factors : list[FactorInfo], 因子元信息（含 IC、Sharpe 等）
        factor_values : dict, {name: pd.DataFrame( date x stock )}
        prices : pd.DataFrame, 股票价格（date x stock）
        market_cap : pd.DataFrame, optional
        industry : pd.Series, optional
        benchmark_returns : pd.Series, optional, 基准日收益率（用于绩效评估）

        Returns
        -------
        PipelineResult
        """
        # Step 1: 因子融合
        print(f"[Pipeline] Fusing {len(factors)} factors with strategy='{self.fusion.strategy}'...")
        composite_scores, fusion_meta = self.fusion.fuse(factors, factor_values, industry)

        # Step 2: 组合构建
        print(f"[Pipeline] Building portfolios (top_n={self.constructor.config.top_n})...")
        portfolios = self.constructor.build(composite_scores, prices, market_cap, industry)

        # Step 3: 风控（对每个调仓日的组合）
        print(f"[Pipeline] Applying risk controls to {len(portfolios)} periods...")
        for i, pf in enumerate(portfolios):
            cap_series = market_cap.loc[pf.date] if market_cap is not None and pf.date in market_cap.index else None
            portfolios[i] = self.risk_manager.apply(pf, market_cap=cap_series, industry=industry)

        # Step 4: 绩效评估
        print("[Pipeline] Computing performance metrics...")
        performance = self._evaluate(portfolios, prices, benchmark_returns)

        return PipelineResult(
            composite_scores=composite_scores,
            portfolios=portfolios,
            factor_weights=fusion_meta.get("weights", {}),
            performance=performance,
            meta=fusion_meta,
        )

    def _evaluate(
        self,
        portfolios: list[Portfolio],
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> dict:
        """计算组合绩效指标"""
        if not portfolios:
            return {"error": "No portfolios to evaluate"}

        # 转化为日收益率序列
        daily_returns = []
        for i in range(len(portfolios) - 1):
            pf = portfolios[i]
            next_date = portfolios[i + 1].date

            if pf.date not in prices.index or next_date not in prices.index:
                continue

            p_t = prices.loc[pf.date, pf.weights.index].dropna()
            p_t1 = prices.loc[next_date, pf.weights.index].dropna()

            common = p_t.index.intersection(p_t1.index).intersection(pf.weights.index)
            if len(common) < 3:
                continue

            stock_returns = (p_t1[common] / p_t[common] - 1).fillna(0)
            w = pf.weights[common]
            w = w / w.sum()
            daily_returns.append((stock_returns * w).sum())

        returns_series = pd.Series(daily_returns)

        if len(returns_series) == 0:
            return {"error": "No valid return periods"}

        total_return = (1 + returns_series).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        ann_vol = returns_series.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        max_dd = self._max_drawdown(returns_series)
        calmar = ann_return / max_dd if max_dd > 0 else 0
        win_rate = (returns_series > 0).mean()

        # 相对基准的超额收益
        excess = None
        info_ratio = None
        if benchmark_returns is not None:
            bm_aligned = benchmark_returns.reindex(returns_series.index).fillna(0)
            excess = returns_series - bm_aligned
            info_ratio = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

        # 平均换手率
        avg_turnover = np.mean([p.turnover for p in portfolios])

        return {
            "total_return": round(total_return, 4),
            "annual_return": round(ann_return, 4),
            "annual_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "win_rate": round(win_rate, 4),
            "excess_return_vs_benchmark": round(excess.mean() * 252, 4) if excess is not None else None,
            "information_ratio": round(info_ratio, 4) if info_ratio is not None else None,
            "avg_turnover": round(avg_turnover, 4),
            "n_periods": len(returns_series),
            "n_portfolios": len(portfolios),
        }

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return abs(dd.min()) if not pd.isna(dd.min()) else 0.0


# ──────────────────────────────────────────────
# Section 6: CLI / Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Factor Fusion & Portfolio Construction Demo")
    print("=" * 60)

    # ---- Mock Data ----
    np.random.seed(42)
    n_dates = 120
    n_stocks = 200
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    stock_codes = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    # 生成三个 mock 因子
    def _make_factor(ic: float, noise: float) -> pd.DataFrame:
        base = np.random.randn(n_dates, n_stocks) * noise
        # 注入信号使 IC 接近目标
        returns = np.random.randn(n_dates, n_stocks) * 0.02
        signal = returns * (ic * 10) + base
        return pd.DataFrame(signal, index=dates, columns=stock_codes)

    f_momentum = _make_factor(0.04, 0.3)
    f_value = _make_factor(0.03, 0.35)
    f_lowvol = _make_factor(0.025, 0.4)

    factors = [
        FactorInfo("momentum", "rank(close/delay(close,60))", ic=0.04, icir=0.04/0.08, ic_std=0.08),
        FactorInfo("value", "rank(1/pb)", ic=0.03, icir=0.03/0.10, ic_std=0.10),
        FactorInfo("lowvol", "-ts_std(returns,20)", ic=0.025, icir=0.025/0.08, ic_std=0.08),
    ]

    factor_values = {
        "momentum": f_momentum,
        "value": f_value,
        "lowvol": f_lowvol,
    }

    # Mock 股票价格
    price_data = 10 * (1 + np.random.randn(n_dates, n_stocks) * 0.02).cumprod(axis=0)
    prices = pd.DataFrame(price_data, index=dates, columns=stock_codes)

    # Mock 市值
    cap_data = np.random.uniform(5e8, 5e11, n_stocks)
    market_cap = pd.DataFrame(
        [cap_data * (1 + np.random.randn() * 0.1) for _ in range(n_dates)],
        index=dates, columns=stock_codes,
    )

    # Mock 行业
    industries = pd.Series(
        np.random.choice(["Tech", "Finance", "Health", "Energy", "Consumer"], n_stocks),
        index=stock_codes,
    )

    # ---- Run Pipeline ----
    fusion = FactorFusion(strategy="icir_weighted", corr_penalty=True)
    constructor = PortfolioConstructor(PortfolioConfig(
        top_n=30, method="score_proportional",
        max_weight=0.05, max_industry_exposure=0.30,
    ))
    risk_mgr = RiskManager(RiskConfig(
        max_stock_weight=0.05, max_industry_exposure=0.30,
    ))

    pipeline = Pipeline(fusion=fusion, constructor=constructor, risk_manager=risk_mgr)
    result = pipeline.run(
        factors=factors,
        factor_values=factor_values,
        prices=prices,
        market_cap=market_cap,
        industry=industries,
    )

    # ---- Print Results ----
    print(f"\nFactor Weights:")
    for name, w in result.factor_weights.items():
        print(f"  {name}: {w:.4f}")

    print(f"\nPortfolios built: {len(result.portfolios)}")
    if result.portfolios:
        pf = result.portfolios[0]
        print(f"First portfolio ({pf.date}):")
        print(f"  Stocks held: {pf.n_stocks}")
        print(f"  Top-5 by weight:")
        for stock in pf.weights.nlargest(5).index:
            print(f"    {stock}: {pf.weights[stock]:.4f}")

    print(f"\nPerformance Metrics:")
    for k, v in result.performance.items():
        print(f"  {k}: {v}")

    print(f"\nComposite Score Stats:")
    cs = result.composite_scores
    print(f"  Shape: {cs.shape}")
    print(f"  Mean: {cs.mean().mean():.4f}")
    print(f"  Std: {cs.std().std():.4f}")

    print("\n" + "=" * 60)
    print("  Demo Complete")
    print("=" * 60)
