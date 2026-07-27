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
  1. ICIR² 收缩加权（ICIR² + James-Stein shrinkage，理论最优 SNR² 组合）
  2. Sign-Aware 融合（负 IC 因子自动反转方向，不再丢弃）
  3. 全样本 IPR 相关性惩罚（替代单日横截面 + hard floor）
  4. Regime-Adaptive Tilt（基于市场状态动态调整因子权重）
  5. Debate 分数作为贝叶斯先验（替代乘法 hack）

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

from .portfolio_utils import (
    allocate_score_proportional,
    allocate_equal_weight,
    allocate_portfolio_weights,
    apply_caps,
    _apply_industry_cap as _apply_industry_cap_static,
)

# Safe import of MarketState types from memory module (no heavy deps)
try:
    from methods.memory import MarketState, TrendRegime, VolRegime
    _HAS_MEMORY = True
except Exception:
    _HAS_MEMORY = False
    MarketState = None
    TrendRegime = None
    VolRegime = None

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
        `FactorFusion` 的 `icir2_shrinkage` 策略使用 |ICIR|² 加权。
    ic_std : float
        保留字段，仅用于向后兼容（旧代码可能直接设置 ic_std）。
    sharpe : float
        因子多空组合的 Sharpe Ratio，来自回测。
    ic_sign : float
        IC 的符号（+1 或 -1）。负 IC 因子的方向会被自动反转，
        其 |ICIR| 仍参与权重计算，实现 sign-aware 融合。
    """
    name: str
    expression: str
    ic: float = 0.0
    icir: float = 0.0       # ICIR = IC / std(IC)，直接传入，不再动态计算
    val_icir: float = 0.0  # Validation-period ICIR (holdout). When >0/non-NaN,
                           # icir2_shrinkage uses |val_icir|² instead of |icir|²
                           # for honest out-of-sample weighting (val_mode only).
    ic_std: float = 0.0     # 保留，向后兼容
    sharpe: float = 0.0
    debate_score: float = 0.0  # 0-10, from multi-agent debate
    ic_sign: float = 1.0       # sign(IC): +1 or -1, for sign-aware fusion
    n_periods: int = 50        # 计算 IC/ICIR 所用的有效样本量 T，用于 Bayesian SNR 收缩
    values: Optional[pd.DataFrame] = None  # index=date, columns=stock_code

    @property
    def ic_decayed(self) -> float:
        """IC 衰减后的有效值（基于最近 N 期滚动）"""
        return self.ic  # 子类可覆盖


class FactorFusion:
    """
    将多个因子融合为单一复合得分。

    支持五种加权策略：
      1. equal: 等权
      2. ic_weighted: IC 绝对值加权（baseline）
      3. icir_weighted: 线性 ICIR 加权（baseline，保留旧行为用于 ablation）
      4. decay_weighted: 指数衰减加权（baseline）
      5. icir2_shrinkage: ICIR² 收缩 + Sign-Aware + Regime Tilt（创新策略）

    所有策略均支持：
      - Sign-Aware 融合（负 IC 因子自动反转方向）
      - 全样本 IPR 相关性惩罚（可选）
      - Debate 分数作为贝叶斯先验（icir2_shrinkage 策略）
      - Regime-Adaptive Tilt（icir2_shrinkage 策略，需传入 market_state）
    """

    def __init__(
        self,
        strategy: str = "icir2_shrinkage",
        corr_penalty: bool = True,
        corr_threshold: float = 0.7,
        normalizer: Optional[FactorNormalizer] = None,
        regime_tilt_strength: float = 0.2,
        shrinkage_kappa: float = 0.3,
        ipr_alpha: float = 2.0,
    ):
        if strategy not in (
            "equal", "ic_weighted", "icir_weighted",
            "decay_weighted", "icir2_shrinkage",
        ):
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        self.strategy = strategy
        self.corr_penalty = corr_penalty
        self.corr_threshold = corr_threshold
        self.normalizer = normalizer or FactorNormalizer(method="zscore")
        # ICIR² shrinkage hyperparameters
        self.regime_tilt_strength = regime_tilt_strength  # regime tilt magnitude
        self.shrinkage_kappa = shrinkage_kappa            # debate prior confidence [0, 1]
        self.ipr_alpha = ipr_alpha                        # IPR penalty sensitivity
        # Signs dict — populated by _compute_weights, used by fuse()
        self._signs: dict[str, float] = {}
        # Saved weights from last fuse() call — used by predict()
        self._saved_weights: dict[str, float] = {}

    def fuse(
        self,
        factors: list[FactorInfo],
        factor_values: dict[str, pd.DataFrame],
        industry: Optional[pd.Series] = None,
        market_state=None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Parameters
        ----------
        factors : list[FactorInfo], 因子元信息
        factor_values : dict, {name: pd.DataFrame(index=date, columns=stock_code)}
            用于权重计算（相关性惩罚使用此数据）以及生成 composite_scores。
            若需在测试期数据上生成 scores，请先调用 fuse() 在训练期算权重，
            再调用 predict() 在测试期数据上生成 scores。
        industry : pd.Series, optional, 行业分类（用于行业中性化）
        market_state : MarketState, optional, 当前市场状态（用于 regime-adaptive tilt）
            仅对 icir2_shrinkage 策略生效。

        Returns
        -------
        composite_scores : pd.DataFrame, 融合后的复合得分
        meta : dict, 融合元信息（权重、符号、IC 等）
        """
        n = len(factors)
        if n == 0:
            raise ValueError("No factors provided")

        # 1. 计算权重（含 sign-aware）— 使用 factor_values
        raw_weights = self._compute_weights(factors, market_state=market_state)

        # 2. 相关性惩罚 — 使用 factor_values
        if self.corr_penalty and n > 1:
            weights = self._apply_corr_penalty(raw_weights, factor_values)
        else:
            weights = raw_weights

        # 3. 保存权重和 signs 供 step7 直接使用
        self._saved_weights = dict(weights)
        self._saved_signs = dict(self._signs)

        # 4. Sign-Aware 归一化 + 加权求和
        #    负 IC 因子的符号 sign = -1，在加权时自动反转方向
        weighted = None
        common_dates = factor_values[factors[0].name].index
        for fv in factor_values.values():
            common_dates = common_dates.intersection(fv.index)

        for f in factors:
            fv = self.normalizer.normalize(factor_values[f.name], industry)
            w = weights.get(f.name, 1.0 / n)
            s = self._signs.get(f.name, 1.0)
            fv_aligned = fv.loc[common_dates] * w * s

            if weighted is None:
                weighted = fv_aligned
            else:
                weighted = weighted.add(fv_aligned, fill_value=0)

        # 5. 最终标准化
        composite_scores = self.normalizer.normalize(weighted, industry)

        meta = {
            "strategy": self.strategy,
            "weights": dict(weights),
            "signs": dict(self._signs),
            "n_factors": n,
            "corr_penalty_applied": self.corr_penalty,
            "regime_tilt_applied": (
                market_state is not None
                and self.strategy == "icir2_shrinkage"
                and _HAS_MEMORY
            ),
            "timestamp": datetime.now().isoformat(),
        }

        return composite_scores, meta

    def _compute_weights(
        self,
        factors: list[FactorInfo],
        market_state=None,
    ) -> dict[str, float]:
        """
        根据策略计算各因子的原始权重。

        所有策略均实现 sign-aware：从 IC 提取方向符号存入 self._signs，
        用 |IC| 或 |ICIR| 计算权重，在 fuse() 中乘以符号自动反转负 IC 因子。

        Parameters
        ----------
        factors : list[FactorInfo]
        market_state : MarketState, optional
            仅对 icir2_shrinkage 策略生效，用于 regime-adaptive tilt。

        Returns
        -------
        dict[str, float] : 因子名称 → 权重
        """
        n = len(factors)
        names = [f.name for f in factors]

        # ── Helper: extract IC sign for all factors ──
        def _extract_signs():
            signs = {}
            for f in factors:
                s = np.sign(f.ic) if abs(f.ic) > 1e-10 else 1.0
                signs[f.name] = float(s)
            return signs

        # ── Strategy dispatch ──
        if self.strategy == "equal":
            self._signs = _extract_signs()
            return {name: 1.0 / n for name in names}

        elif self.strategy == "ic_weighted":
            # Baseline: |IC| weighted
            self._signs = _extract_signs()
            ics = np.array([abs(f.ic) for f in factors])
            total = ics.sum()
            if total < 1e-8:
                return {name: 1.0 / n for name in names}
            return {f.name: abs(f.ic) / total for f in factors}

        elif self.strategy == "icir_weighted":
            # Baseline: linear |ICIR| weighted (fixed: no longer discards negative ICIR)
            self._signs = _extract_signs()
            icirs = np.array([abs(f.icir) for f in factors])
            total = icirs.sum()
            if total < 1e-8:
                return {name: 1.0 / n for name in names}
            return {f.name: icirs[i] / total for i, f in enumerate(factors)}

        elif self.strategy == "decay_weighted":
            # Baseline: exponential decay by list position
            self._signs = _extract_signs()
            decay_w = np.exp(-0.3 * np.arange(n))
            total = decay_w.sum()
            return {f.name: decay_w[i] / total for i, f in enumerate(factors)}

        elif self.strategy == "icir2_shrinkage":
            # ════════════════════════════════════════════════════════════
            # Innovation: ICIR² + James-Stein Shrinkage + Bayesian Prior
            #             + Regime-Adaptive Tilt
            # ════════════════════════════════════════════════════════════

            # 1. Sign-aware: extract IC sign, use |ICIR| for weighting.
            #    Prefer validation-period ICIR (holdout) when available, so the
            #    combining weight is estimated out-of-sample (honest, anti-overfit).
            #    Falls back to training ICIR when val_icir is missing/NaN/zero.
            self._signs = _extract_signs()

            def _effective_icir_abs(f: "FactorInfo") -> float:
                vic = getattr(f, "val_icir", None)
                if vic is not None and np.isfinite(vic) and abs(vic) > 1e-8:
                    return abs(vic)
                return abs(f.icir)

            icirs_abs = np.array([_effective_icir_abs(f) for f in factors])

            # Edge case: all ICIRs are zero → equal weight
            if icirs_abs.sum() < 1e-8:
                return {name: 1.0 / n for name in names}

            # 2. ICIR² weights (SNR² — optimal combining weight for
            #    independent signals is proportional to SNR²)
            raw_w = icirs_abs ** 2
            raw_w = raw_w / (raw_w.sum() + 1e-12)

            # 3. Bayesian SNR shrinkage (Efron & Morris 1975)
            #    ─────────────────────────────────────────────────────────
            #    Model:
            #      ICIR_i_obs ~ N(ICIR_i_true, SE_i²)
            #      ICIR_i_true ~ N(μ_0, τ²)          (empirical Bayes prior)
            #
            #    Optimal shrinkage toward prior mean:
            #      ICIR_i_shrunk = (1-δ)·ICIR_i_obs + δ·μ_0
            #      δ = SE² / (SE² + τ²)
            #
            #    Estimated from data (method of moments):
            #      var(ICIR_obs) = τ² + SE²    (in expectation)
            #      → τ²_hat = max(0, var(ICIR_obs) - SE²)
            #
            #    With SE² ≈ 1/T  (T = n_periods, effective sample size):
            #      δ_hat = 1 / (1 + max(0, var(ICIR_obs) × T - 1))
            #
            #    Interpretation:
            #      high var(ICIR)×T  →  ICIR estimates reliable  →  low δ
            #      low  var(ICIR)×T  →  ICIR estimates noisy     →  high δ
            #    ─────────────────────────────────────────────────────────
            n_periods_arr = np.array([max(2, f.n_periods) for f in factors])
            T = int(n_periods_arr.min())  # conservative: use min T across factors
            var_icir = float(np.var(icirs_abs))
            # SNR of ICIR estimates = estimated signal variance / noise variance
            # signal_var_hat = max(0, var(ICIR_obs) - 1/T)
            # snr_icir = signal_var_hat / (1/T) = max(0, var(ICIR_obs) × T - 1)
            snr_icir = max(0.0, var_icir * T - 1.0)
            delta = 1.0 / (1.0 + snr_icir)
            delta = float(np.clip(delta, 0.05, 0.5))

            equal_w = np.ones(n) / n
            w = (1.0 - delta) * raw_w + delta * equal_w

            # 4. Debate scores as Bayesian prior (not multiplicative hack)
            #    posterior = κ · prior + (1 - κ) · likelihood
            debate_scores = np.array(
                [getattr(f, 'debate_score', 0.0) for f in factors]
            )
            if debate_scores.max() > 0:
                prior = np.clip(debate_scores, 0.0, 10.0) / 10.0
                prior_sum = prior.sum()
                if prior_sum > 1e-8:
                    prior = prior / prior_sum
                    kappa = self.shrinkage_kappa
                    w = kappa * prior + (1.0 - kappa) * w

            # 5. Regime-adaptive tilt
            #    Boost momentum factors in bull markets, low-vol in bear/high-vol,
            #    value factors in sideways markets.
            if market_state is not None and _HAS_MEMORY:
                tilt = self._compute_regime_tilt(factors, market_state)
                w = w * (1.0 + tilt)

            # Normalize
            w = w / (w.sum() + 1e-12)
            return {names[i]: float(w[i]) for i in range(n)}

        else:
            self._signs = _extract_signs()
            return {name: 1.0 / n for name in names}

    def _apply_corr_penalty(
        self,
        raw_weights: dict[str, float],
        factor_values: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        基于 Inverse Participation Ratio (IPR) 的相关性惩罚。

        改进点（对比旧版）：
        1. 全样本 rank correlation 替代单日横截面 — 统计显著
        2. IPR 惩罚替代 hard floor 0.5 — 连续、无下限截断
        3. 加权相关性暴露 — 考虑其他因子的权重，而非简单阈值

        算法：
          1. 对所有因子值做横截面 rank，flatten 成长向量
          2. 计算 n×n 全样本 |correlation| 矩阵
          3. 对每个因子 i，计算加权相关性暴露：
             E_i = Σ_{j≠i} |corr(i,j)| × w_j / (1 - w_i)
          4. IPR 惩罚：penalty_i = 1 / (1 + α × E_i)
          5. 最终权重 = 原始权重 × penalty，再 renormalize
        """
        names = list(raw_weights.keys())
        n = len(names)
        if n <= 1:
            return raw_weights

        # ── Step 1: Build full-sample rank correlation matrix ──
        # Collect aligned dates
        common_dates = factor_values[names[0]].index
        for name in names:
            common_dates = common_dates.intersection(factor_values[name].index)

        if len(common_dates) < 5:
            # Not enough data for meaningful correlation → skip penalty
            return raw_weights

        # Rank each factor cross-sectionally, then flatten to 1D
        flat_data = {}
        for name in names:
            fv = factor_values[name].loc[common_dates]
            ranked = fv.rank(axis=1, pct=True)
            flat_data[name] = ranked.values.flatten()

        # Compute pairwise |correlation| using vectorized numpy
        corr_mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                xi = flat_data[names[i]]
                xj = flat_data[names[j]]
                mask = ~(np.isnan(xi) | np.isnan(xj))
                if mask.sum() >= 100:
                    c = np.corrcoef(xi[mask], xj[mask])[0, 1]
                    corr_mat[i, j] = corr_mat[j, i] = abs(c) if not np.isnan(c) else 0.0

        # ── Step 2: IPR (Inverse Participation Ratio) penalty ──
        w_vec = np.array([raw_weights[names[i]] for i in range(n)])

        # Weighted correlation exposure (excluding diagonal / self-correlation)
        off_diag = corr_mat - np.eye(n)
        # E_i = Σ_{j≠i} |corr(i,j)| × w_j  (normalized by total weight of others)
        weighted_exposure = off_diag @ w_vec  # shape: (n,)
        denom = (1.0 - w_vec)  # weight of all other factors
        denom = np.where(denom < 1e-10, 1.0, denom)
        E = weighted_exposure / denom

        # IPR penalty: high exposure → low multiplier
        # penalty_i = 1 / (1 + α × E_i)
        #   E_i = 0  → penalty = 1.0  (no correlation, no penalty)
        #   E_i = 0.5 → penalty = 1/(1+α·0.5)
        #   E_i = 1.0 → penalty = 1/(1+α)  (fully redundant)
        penalty = 1.0 / (1.0 + self.ipr_alpha * E)

        # Apply penalty
        penalized_w = w_vec * penalty

        # Renormalize
        total = penalized_w.sum()
        if total > 1e-12:
            penalized_w = penalized_w / total
        else:
            # Degenerate case → equal weight
            penalized_w = np.ones(n) / n

        return {names[i]: float(penalized_w[i]) for i in range(n)}

    # ── Regime-Adaptive Tilt helpers ──

    @staticmethod
    def _classify_factor(expression: str) -> set[str]:
        """
        Classify a factor into categories based on its DSL expression.

        Categories: momentum, lowvol, value, liquidity, other
        A factor can belong to multiple categories.

        This is a soft classification used only for regime tilt —
        the actual factor value computation is always done by _FactorExprEvaluator.
        """
        expr_lower = expression.lower()
        categories = set()

        if any(kw in expr_lower for kw in [
            'delta', 'ts_delta', 'delay', 'close/close', 'return', 'momentum',
            'close/delay',
        ]):
            categories.add('momentum')

        if any(kw in expr_lower for kw in [
            'ts_std', 'ts_stddev', 'std', 'volatility',
        ]):
            categories.add('lowvol')

        if any(kw in expr_lower for kw in [
            '1/pe', '1/pb', 'pb', 'pe', 'roe', 'book', 'eps',
        ]):
            categories.add('value')

        if any(kw in expr_lower for kw in [
            'volume', 'amount', 'turnover',
        ]):
            categories.add('liquidity')

        if not categories:
            categories.add('other')

        return categories

    def _compute_regime_tilt(
        self,
        factors: list[FactorInfo],
        market_state,
    ) -> np.ndarray:
        """
        Compute per-factor tilt based on market regime.

        Logic:
          - Bull market  → boost momentum factors
          - Bear market  → boost low-vol (defensive) factors
          - High vol     → boost low-vol factors
          - Sideways     → boost value (mean-reversion) factors

        Parameters
        ----------
        factors : list[FactorInfo]
        market_state : MarketState (from methods.memory)

        Returns
        -------
        np.ndarray of shape (n_factors,) — tilt multipliers (can be 0)
        """
        n = len(factors)
        tilt = np.zeros(n)
        strength = self.regime_tilt_strength

        if market_state is None or not _HAS_MEMORY:
            return tilt

        trend = market_state.trend
        volatility = market_state.volatility

        for i, f in enumerate(factors):
            cats = self._classify_factor(f.expression)

            # Bull → boost momentum
            if trend == TrendRegime.BULL and 'momentum' in cats:
                tilt[i] += strength

            # Bear → boost low-vol (defensive)
            if trend == TrendRegime.BEAR and 'lowvol' in cats:
                tilt[i] += strength

            # High volatility → boost low-vol
            if volatility == VolRegime.HIGH and 'lowvol' in cats:
                tilt[i] += strength * 0.75

            # Sideways → boost value (mean-reversion)
            if trend == TrendRegime.SIDEWAYS and 'value' in cats:
                tilt[i] += strength * 0.5

        return tilt


# ──────────────────────────────────────────────
# Section 3: Portfolio Constructor
# ──────────────────────────────────────────────


@dataclass
class PortfolioConfig:
    """组合构建超参数"""
    top_n: int = 50                     # 持仓股票数量
    method: str = "score_proportional"  # score_proportional | equal_weight | equal | top_n
    long_only: bool = True
    max_weight: float = 0.05            # 单只股票最大权重
    min_weight: float = 0.001           # 单只股票最小权重
    max_industry_exposure: float = 0.30  # 单行业最大暴露度


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
    支持（method，详见 methods.portfolio_utils.allocate_portfolio_weights）：
      - score_proportional: 按复合得分比例分配权重
      - equal_weight / equal / top_n: 1/n 等权持有（与 baselines 共用同一实现）
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
        """根据配置分配个股权重（含风控约束）

        权重方案与上限逻辑统一走 ``methods.portfolio_utils``，
        MASE 与 9 个 baseline 共用同一实现，保证组合构建逐位一致。
        ``method`` 取值见 ``allocate_portfolio_weights`` 的调度表
        （score_proportional / equal_weight / equal / top_n）。
        """
        method = self.config.method

        return allocate_portfolio_weights(
            top_scores,
            method=method,
            max_weight=self.config.max_weight,
            max_industry_exposure=self.config.max_industry_exposure,
            min_weight=self.config.min_weight,
            industry=industry,
        )

    def _apply_industry_cap(self, weights: pd.Series, industry: pd.Series) -> pd.Series:
        """限制单行业最大权重（委托给共享实现，保持向后兼容）"""
        return _apply_industry_cap_static(weights, industry, self.config.max_industry_exposure)

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
    min_market_cap_percentile: float = 0.0  # 市值过滤（disabled：放宽，不再剔除小市值）
    exclude_st: bool = False


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
    n_dates = 120
    n_stocks = 200
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    stock_codes = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    # 生成四个 mock 因子（含一个负 IC 因子以演示 sign-aware 融合）
    def _make_factor(ic: float, noise: float) -> pd.DataFrame:
        base = np.random.randn(n_dates, n_stocks) * noise
        # 注入信号使 IC 接近目标
        returns = np.random.randn(n_dates, n_stocks) * 0.02
        signal = returns * (ic * 10) + base
        return pd.DataFrame(signal, index=dates, columns=stock_codes)

    f_momentum = _make_factor(0.04, 0.3)
    f_value = _make_factor(0.03, 0.35)
    f_lowvol = _make_factor(0.025, 0.4)
    f_neg_ic = _make_factor(-0.035, 0.3)  # 负 IC 因子 — sign-aware 会自动反转

    factors = [
        FactorInfo("momentum", "rank(close/delay(close,60))", ic=0.04, icir=0.04/0.08, ic_std=0.08, debate_score=7.5),
        FactorInfo("value", "rank(1/pb)", ic=0.03, icir=0.03/0.10, ic_std=0.10, debate_score=6.0),
        FactorInfo("lowvol", "-ts_std(returns,20)", ic=0.025, icir=0.025/0.08, ic_std=0.08, debate_score=5.5),
        FactorInfo("neg_factor", "rank(ts_delta(close,5))", ic=-0.035, icir=-0.035/0.09, ic_std=0.09, debate_score=4.0),
    ]

    factor_values = {
        "momentum": f_momentum,
        "value": f_value,
        "lowvol": f_lowvol,
        "neg_factor": f_neg_ic,
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
    fusion = FactorFusion(
        strategy="icir2_shrinkage",
        corr_penalty=True,
        regime_tilt_strength=0.2,
        shrinkage_kappa=0.3,
        ipr_alpha=2.0,
    )
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
    print(f"\nFactor Weights & Signs:")
    signs = result.meta.get("signs", {})
    for name, w in result.factor_weights.items():
        s = signs.get(name, 1.0)
        print(f"  {name:15s}  weight={w:.4f}  sign={s:+.0f}")

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
