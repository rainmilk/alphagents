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

    method                : "score_proportional"
    max_weight            : 0.50   (single-stock cap)
    max_industry_exposure : 0.80   (industry cap)
    min_weight            : 0.001  (single-stock floor)

If ``fusion.portfolio`` values change, update the ``DEFAULT_*`` constants below
as well (baselines that do not thread the full config dict through rely on them).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Defaults mirror config.yaml -> fusion.portfolio. Kept here so baselines that do
# not pass the full config dict can still stay bit-for-bit consistent with MASE.
DEFAULT_MAX_WEIGHT = 0.50
DEFAULT_MAX_INDUSTRY_EXPOSURE = 0.80
DEFAULT_MIN_WEIGHT = 0.001


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
