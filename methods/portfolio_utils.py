# -*- coding: utf-8 -*-
"""
Shared portfolio-weighting utilities
====================================

This module is the **single source of truth** for how portfolio weights are
allocated from a set of (already selected) top-N stock scores. Both MASE's
``PortfolioConstructor`` and the 9 baselines import from here, so the
portfolio-construction scheme stays *identical* across methods — a hard
requirement for a fair paper comparison.

The scheme mirrors ``config.yaml -> fusion.portfolio``:

    method                : "score_proportional" | "equal_weight"
    max_weight            : 0.50   (single-stock cap)
    max_industry_exposure : 0.80   (industry cap)
    min_weight            : 0.001  (single-stock floor)

If ``fusion.portfolio`` values change, update the ``DEFAULT_*`` constants below
as well (baselines that do not thread the full config dict through rely on them).

Supported ``method`` values (selectable via ``allocate_portfolio_weights`` or
``PortfolioConfig.method``):

    - "score_proportional" / "score_prop": weight ∝ shifted composite score.
    - "equal_weight" / "equal" / "top_n":  1/n equal weight across selected
      stocks (same caps applied for consistency; inert at typical settings).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Defaults mirror config.yaml -> fusion.portfolio. Kept here so baselines that do
# not pass the full config dict can still stay bit-for-bit consistent with MASE.
DEFAULT_MAX_WEIGHT = 0.20
DEFAULT_MAX_INDUSTRY_EXPOSURE = 0.50
DEFAULT_MIN_WEIGHT = 0.001
DEFAULT_TOP_N = 50

# Canonical method names -> internal dispatch keys.
METHOD_SCORE_PROPORTIONAL = "score_proportional"
METHOD_EQUAL_WEIGHT = "equal_weight"
_SCORE_ALIASES = ("score_proportional", "score_prop")
_EQUAL_ALIASES = ("equal_weight", "equal", "top_n")


def allocate_score_proportional(
    top_scores: pd.Series,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    max_industry_exposure: float = DEFAULT_MAX_INDUSTRY_EXPOSURE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    industry: Optional[pd.Series] = None,
) -> pd.Series:
    """Score-proportional weights with MASE-consistent caps.

    Given a Series of composite scores for the *already-selected* top-N stocks,
    produce long-only weights:

        1. shift scores to be non-negative:  raw = score - min(score) + 1e-6
        2. proportional weights:             raw / raw.sum()
        3. clip single-stock weight at ``max_weight`` and renormalize
        4. drop weights below ``min_weight`` and renormalize
        5. cap any single-industry exposure at ``max_industry_exposure``
           (only when an industry map is supplied and the cap < 1.0)

    This is the exact weighting step used by MASE's ``PortfolioConstructor``
    (method="score_proportional"). Baselines call it so their portfolio
    construction matches MASE.
    """
    raw = top_scores - top_scores.min() + 1e-6
    weights = raw / raw.sum()
    return apply_caps(
        weights, max_weight, max_industry_exposure, min_weight, industry
    )


def allocate_equal_weight(
    top_scores: pd.Series,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    max_industry_exposure: float = DEFAULT_MAX_INDUSTRY_EXPOSURE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    industry: Optional[pd.Series] = None,
) -> pd.Series:
    """Equal-weight (1/n) portfolio, with the same caps as score-proportional.

    Every selected stock receives weight ``1/n``. The cap logic (single-stock
    max, min-weight floor, industry exposure) is applied via :func:`apply_caps`
    so the scheme stays consistent with MASE's ``PortfolioConstructor`` and the
    score-proportional path. At typical settings (<=50 stocks, ``max_weight``
    >= 0.2) the caps are inert, so this is effectively pure 1/n.

    This mirrors the previously-inline ``equal`` branch in
    ``PortfolioConstructor._allocate_weights`` and is now the single source of
    truth for the equal-weight strategy.
    """
    n = len(top_scores)
    if n == 0:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / n, index=top_scores.index)
    return apply_caps(weights, max_weight, max_industry_exposure, min_weight, industry)


def apply_caps(
    weights: pd.Series,
    max_weight: float,
    max_industry_exposure: float,
    min_weight: float,
    industry: Optional[pd.Series] = None,
) -> pd.Series:
    """Apply the MASE weight caps to an already-computed weight vector.

    Shared by both the score-proportional and equal-weight paths so the cap
    logic never diverges between MASE and the baselines.
    """
    # single-stock cap
    weights = weights.clip(upper=max_weight)
    weights = weights / weights.sum()

    # minimum-weight floor
    weights = weights[weights >= min_weight]
    if len(weights) > 0:
        weights = weights / weights.sum()

    # industry exposure cap
    if industry is not None and max_industry_exposure < 1.0:
        weights = _apply_industry_cap(weights, industry, max_industry_exposure)

    return weights


def _apply_industry_cap(
    weights: pd.Series,
    industry: pd.Series,
    max_industry_exposure: float,
) -> pd.Series:
    """Scale down any industry whose total weight exceeds the cap."""
    common = weights.index.intersection(industry.index)
    if len(common) == 0:
        return weights

    w = weights[common].copy()
    for ind in industry[common].unique():
        mask = industry[common] == ind
        ind_weight = w[mask].sum()
        if ind_weight > max_industry_exposure:
            scale = max_industry_exposure / ind_weight
            w[mask] *= scale

    w = w / w.sum()
    return w


def allocate_portfolio_weights(
    top_scores: pd.Series,
    method: str = METHOD_SCORE_PROPORTIONAL,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    max_industry_exposure: float = DEFAULT_MAX_INDUSTRY_EXPOSURE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    industry: Optional[pd.Series] = None,
) -> pd.Series:
    """Single dispatch point for portfolio weight allocation.

    Routes ``method`` to the matching allocator so MASE and all 9 baselines
    share one implementation (the hard requirement for a fair paper comparison):

        - score_proportional / score_prop -> :func:`allocate_score_proportional`
        - equal_weight / equal / top_n      -> :func:`allocate_equal_weight`

    Unknown / unrecognised values fall back to ``score_proportional`` (this
    matches ``PortfolioConstructor``'s historical fallback-to-1/n behaviour,
    but logs nothing — callers should validate ``method`` upstream if strict).
    """
    if method in _SCORE_ALIASES:
        return allocate_score_proportional(
            top_scores, max_weight, max_industry_exposure, min_weight, industry
        )
    if method in _EQUAL_ALIASES:
        return allocate_equal_weight(
            top_scores, max_weight, max_industry_exposure, min_weight, industry
        )
    # Unknown method -> fall back to score-proportional for robustness.
    return allocate_score_proportional(
        top_scores, max_weight, max_industry_exposure, min_weight, industry
    )


# ──────────────────────────────────────────────
#  Unified portfolio construction (ML baselines)
# ──────────────────────────────────────────────


def build_portfolios_from_scores(
    composite_scores: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    method: str = METHOD_SCORE_PROPORTIONAL,
    max_weight: float = 0.1,
    max_industry_exposure: float = 0.80,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    industry: Optional[pd.Series] = None,
    test_start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build a date×stock portfolio-weight DataFrame from a raw score panel.

    This is the single, canonical portfolio-construction entry point shared by
    MASE's factor path (``main.py`` ``step7_construct_portfolio``) and the
    **end-to-end ML baselines** (LSTM / XGBoost / AlphaXGBoost).

    Those ML models are *not* factor methods — they never pass through
    ``FactorFusion``. Instead their model-predicted forward-return scores are
    fed here directly as ``composite_scores``. Reusing the exact same
    construction code as the factor baselines is what makes the paper
    comparison fair: the only difference between an ML baseline and a factor
    baseline is the *source* of the composite score, not how the portfolio is
    built.

    The construction is bit-for-bit identical to ``main.py`` ``step7``:

      1. ``PortfolioConstructor(PortfolioConfig(...)).build`` selects the top-N
         stocks per rebalance date (suspended-stock filtered) and allocates
         score-proportional weights with the **same caps** (``max_weight`` /
         ``max_industry_exposure`` / ``min_weight``) every baseline uses.
      2. The returned ``list[Portfolio]`` is reshaped into a date×stock
         DataFrame and aligned to the full stock universe + trading-day
         calendar, then forward-filled (hold positions on skipped rebalance
         dates) and cropped to the test period.

    Args:
        composite_scores: pd.DataFrame (date×stock) of raw scores. For ML
            baselines this is the model's predicted forward-return panel.
        prices: Full close-price panel (date×stock). Used to (a) filter
            suspended stocks, (b) align the output to the full stock universe
            and trading-day calendar.
        top_n: Number of stocks to hold.
        method: Weighting method ("score_proportional" | "equal_weight" | ...).
        max_weight: Single-stock weight cap (mirrors ``config.fusion.portfolio``).
        max_industry_exposure: Single-industry weight cap (mirrors config).
        min_weight: Single-stock weight floor (mirrors config).
        industry: Optional stock→industry Series for the industry cap.
        test_start_date: Optional first test date (YYYY-MM-DD). Portfolios
            dated before it (context-window rows) are cropped — they are
            invalid for backtest.

    Returns:
        pd.DataFrame, index = trading-day (DatetimeIndex), columns = all
        stocks, values = portfolio weights (each row sums to 1).
    """
    # Lazy import to avoid a circular dependency: fusion.py imports this module
    # (allocate_portfolio_weights), so importing it at module top would cycle.
    from methods.fusion import PortfolioConfig, PortfolioConstructor

    # Normalize the score-panel index to a DatetimeIndex so it interoperates
    # with the (DatetimeIndex) price panel inside PortfolioConstructor.build
    # regardless of whether the caller passed Timestamps or plain date strings.
    composite_scores = composite_scores.copy()
    composite_scores.index = pd.to_datetime(composite_scores.index)

    config = PortfolioConfig(
        top_n=top_n,
        method=method,
        max_weight=max_weight,
        max_industry_exposure=max_industry_exposure,
        min_weight=min_weight,
    )
    constructor = PortfolioConstructor(config)
    raw_portfolios = constructor.build(
        composite_scores=composite_scores,
        prices=prices,
        industry=industry,
    )

    if not raw_portfolios:
        raise ValueError("No portfolios generated. Check composite_scores and prices.")

    # Convert list[Portfolio] -> date×stock DataFrame (mirror main.py step7).
    weight_dict = {pf.date: pf.weights for pf in raw_portfolios}
    portfolios = pd.DataFrame(weight_dict).T
    portfolios.index = pd.to_datetime(portfolios.index)
    portfolios = portfolios.fillna(0.0)

    # Align to the full stock universe so backtest column shapes always match
    # the price panel (pf.weights only holds the per-day top-N subset).
    all_stocks = prices.columns
    portfolios = portfolios.reindex(columns=all_stocks, fill_value=0.0)

    # Align to the trading-day calendar; hold positions on skipped rebalance
    # dates via forward-fill.
    trading_days = prices.index
    portfolios = portfolios.reindex(trading_days.intersection(portfolios.index))
    portfolios = portfolios.ffill().fillna(0.0)

    # Crop context-window dates before the test period.
    if test_start_date is not None:
        test_start_ts = pd.Timestamp(test_start_date)
        portfolios = portfolios[portfolios.index >= test_start_ts]

    return portfolios
