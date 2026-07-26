# -*- coding: utf-8 -*-
"""
Bridge: compute FAMA's 101 Alpha101 factor *exposures* with bit-for-bit
identical numerics inside the MASE pipeline.

Why this module exists
----------------------
MASE and FAMA "both select Alpha101 factors" but produced different total
returns. The fusion + portfolio layers are already unified; the divergence is
upstream, in the **factor set itself**:

  * FAMA evaluates 101 AlphaFAMA-DSL formulas per-ticker with a *time-series*
    ``rank()`` and a VWAP defined as ``(high + low + close) / 3``.
  * MASE evaluates 98 MASE-DSL formulas on a *cross-sectional* panel with a
    VWAP defined as ``amount / volume``.

Those two definitions are mathematically different (~40 of the 101 alphas
reference ``vwap``), so MASE can never reproduce FAMA's factors by re-evaluating
FAMA's formula *strings* in MASE's own evaluator.

The only correct way to get bit-for-bit identical results is to **reuse FAMA's
own code**: ``convert_price_data_to_alphafama`` (builds the exact panel,
including the correct VWAP/returns) and ``AlphaFactory.all_alphas`` / the
serial ``_compute_factors_chunk`` worker (the exact per-ticker math). This
module is a thin, deterministic adapter around those — it deliberately does
NOT re-implement any factor math.

Determinism note
----------------
We call ``_compute_factors_chunk`` **serially** (all tickers in one chunk)
rather than going through FAMA's ``_compute_factors`` parallel path. The per-
ticker computation is unchanged; only the scheduling differs, and the serial
row order is identical to FAMA's serial fallback. This sidesteps the
Windows-spawn ``__main__`` re-import hazard documented in
``alphafama_parallel.py`` while preserving numerical parity.
"""

from typing import Dict, Optional

import pandas as pd
import numpy as np

from baselines.AlphaFAMA.src.data_bridge import convert_price_data_to_alphafama
from baselines.AlphaFAMA.src.constants.formula_map import FORMULA_MAP
from baselines.alphafama_parallel import _compute_factors_chunk


# Stable id prefix so MASE keys the factors by a unique, expression-like name.
FAMA101_PREFIX = "fama101_"


def fama_id(alpha_name: str) -> str:
    """Map a FAMA AlphaFactory method name (``alpha001``) to a MASE factor id."""
    return f"{FAMA101_PREFIX}{alpha_name}"


def get_fama101_formula_map() -> Dict[str, str]:
    """Return FAMA's Alpha101 formula map (id -> AlphaFAMA-DSL formula text).

    Used to feed the LLM inspiration chains with the *real* FAMA formula text
    (rather than the opaque ``fama101_alpha001`` id), so the Stage-2 LLM mining
    still receives meaningful inspiration.
    """
    return {fama_id(k): v for k, v in FORMULA_MAP.items()}


def compute_fama101_exposures(
    price_data: Dict[str, pd.DataFrame],
    forward_period: int = 10,
) -> pd.DataFrame:
    """Compute FAMA's 101 Alpha101 factor exposures from MASE ``price_data``.

    Args:
        price_data: MASE price dict with keys
            ``open, high, low, close, volume, amount`` (each a wide DataFrame
            indexed by date with stock columns).
        forward_period: Forward-return horizon in trading days. MUST match
            ``config.evolution.forward_period`` (default 10) so the panel's
            ``forward_return`` column aligns with the rest of the project.

    Returns:
        A DataFrame with a MultiIndex ``(date, ticker)`` and 101 columns named
        ``fama101_alpha001`` … ``fama101_alpha101``. Each column is the exact
        per-ticker exposure FAMA would produce for that alpha on this data —
        no MASE-side recomputation, no DSL translation.

    Notes:
        * The result equals FAMA's ``_compute_factors(convert_price_data_to_
          alphafama(price_data))`` column-for-column (the golden-file test in
          ``tests/test_fama101_bridge_parity.py`` asserts this).
        * The ``forward_return`` column produced by the bridge is NOT returned
          here — only the 101 factor exposures. The Rank-IC target is supplied
          separately by ``FactorBacktester.forward_returns`` (which is
          mathematically equivalent: ``close.shift(-N)/close - 1``).
    """
    # 1. Build FAMA's exact panel (correct VWAP, correct backward returns).
    df = convert_price_data_to_alphafama(price_data, forward_period=forward_period)

    # 2. Run FAMA's exact per-ticker factor math (serial = bit-identical to
    #    FAMA's serial path; avoids the Windows-spawn __main__ hazard).
    groups = list(df.groupby("ticker"))
    ex_list = _compute_factors_chunk(groups)
    exposures = pd.concat(ex_list, axis=0)

    # 3. ``_compute_factors_chunk`` tags each per-ticker frame with a redundant
    #    ``ticker`` column — drop it; the (date, ticker) MultiIndex is enough.
    if "ticker" in exposures.columns:
        exposures = exposures.drop(columns=["ticker"])

    # 4. Rename ``alphaNNN`` -> ``fama101_alphaNNN`` so MASE can key the factor
    #    by a stable, expression-like name (used by _calculate_factor_values,
    #    step6's factor_meta_lookup, etc.).
    exposures = exposures.rename(
        columns=lambda c: fama_id(c) if c.startswith("alpha") else c
    )

    return exposures


def slice_exposures_to_dates(
    exposures: pd.DataFrame,
    dates,
) -> Dict[str, pd.DataFrame]:
    """Slice a full MultiIndex exposure frame down to a specific set of dates.

    Args:
        exposures: MultiIndex (date, ticker) frame from
            :func:`compute_fama101_exposures`.
        dates: iterable of dates (e.g. a training-period close.index) to keep.

    Returns:
        dict mapping each column name -> a date×ticker (unstacked) DataFrame,
        indexed by ``dates`` and reindexed on columns so missing tickers become
        NaN rather than raising. Used to (a) score IC on the train window and
        (b) feed ``_calculate_factor_values``.
    """
    date_set = pd.Index(pd.to_datetime(list(dates)))
    mask = exposures.index.get_level_values("date").isin(date_set)
    window = exposures.loc[mask]
    out: Dict[str, pd.DataFrame] = {}
    for col in exposures.columns:
        series = window[col]
        frame = series.unstack("ticker")
        frame = frame.reindex(index=date_set)
        out[col] = frame
    return out
