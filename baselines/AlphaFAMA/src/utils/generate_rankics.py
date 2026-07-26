# src/utils/generate_rankics.py

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..alpha_functions import AlphaFactory  # for calling alphaXXX methods
from .ts_functions import (
    ts_rank, delta, ts_sum, ts_min, ts_max, decay_linear, ts_argmax,
    ts_argmin, delay
)
from .stat_helpers import (
    stddev, correlation, rank, sma, scale, sign, covariance
)
from .math_helpers import product


def write_rankic_jsons(ic_df: pd.DataFrame, formula_map: dict, out_dir: Path):
    """
    - ic_df: DataFrame, rows = dates, cols = factor names (e.g. alpha001, alpha002, …)
    - formula_map: maps alphaXXX -> human formula
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) average IC per factor, drop NaNs and guard small samples
    avg_ic = ic_df.mean(axis=0).fillna(0)
    avg_ic_dict = avg_ic.to_dict()

    # 2) write the time-series JSON
    (out_dir / "time_series_rankic_all_factors.json").write_text(
        json.dumps(avg_ic_dict, indent=2), encoding='utf-8'
    )
    print("Wrote", out_dir / "time_series_rankic_all_factors.json")

    # 3) build formula->IC map for alpha factors only
    formula_ic = {}
    for f, ic in avg_ic_dict.items():
        if not f.startswith("alpha"):
            continue
        label = formula_map.get(f, f)
        formula_ic[label] = ic

    # 4) write the formulas JSON
    (out_dir / "formula_rankic.json").write_text(
        json.dumps(formula_ic, indent=2), encoding='utf-8'
    )
    print("Wrote", out_dir / "formula_rankic.json")


# Cache compiled expression code objects so repeated evaluation of the same
# expression across tickers (and across the Step-5 mining / Step-6 merge passes)
# doesn't re-parse the string thousands of times. eval() of a code object is
# dramatically cheaper than eval() of a raw string.
_EXPR_COMPILE_CACHE = {}


def _compile_expr(expr: str):
    code = _EXPR_COMPILE_CACHE.get(expr)
    if code is None:
        code = compile(expr, "<llm_expr>", "eval")
        _EXPR_COMPILE_CACHE[expr] = code
    return code


def factor_series_fn(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    Evaluate an Alpha101 expression or factory method name against df.
    """
    # 1) direct call for alphaXXX methods
    if hasattr(AlphaFactory, expr):
        return getattr(AlphaFactory, expr)(df)

    # 2) prepare safe namespace
    func_ns = {
        "ts_rank": ts_rank, "delta": delta, "ts_sum": ts_sum,
        "ts_min": ts_min, "ts_max": ts_max, "decay_linear": decay_linear,
        "ts_argmax": ts_argmax, "ts_argmin": ts_argmin, "delay": delay,
        "stddev": stddev, "correlation": correlation, "rank": rank,
        "sma": sma, "scale": scale, "sign": sign, "covariance": covariance,
        "product": product, "np": np
    }
    # inject all DataFrame columns
    local_ns = {col: df[col] for col in df.columns}

    # detect advXX patterns and add moving-average series
    adv_windows = {int(w) for w in
                   re.findall(r"adv(\d+)", expr)}
    for w in adv_windows:
        local_ns[f"adv{w}"] = df["volume"].rolling(w).mean()

    # 3) evaluate (using the cached compiled code object)
    try:
        result = eval(_compile_expr(expr), func_ns, local_ns)
    except NameError as e:
        raise ValueError(f"Unknown name in expression {expr!r}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to eval expression {expr!r}: {e}")

    # 4) ensure pd.Series
    if not isinstance(result, pd.Series):
        raise TypeError(f"Expression did not return a Series: got {type(result)}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Panel-level (cross-sectional) helper functions
# ═══════════════════════════════════════════════════════════════════════
# These operate on 2D DataFrames (date × ticker) and mirror the semantics
# of their 1D counterparts in stat_helpers.py / ts_functions.py, but with
# **cross-sectional** operations for rank() and scale() (operating across
# stocks on each date), and **per-column** (per-stock) time-series
# operations for ts_* functions (operating along the date axis).
#
# This fixes the semantic mismatch where the LLM prompt says
#   "# rank(x): cross-sectional rank into [0,1]"
# but the per-ticker factor_series_fn evaluates rank() on a 1D Series
# (time-series rank).  See factor_panel_fn below.


def _panel_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0~1) across stocks on each date.

    Matches MASE's ``_fn_rank``: ``x.rank(axis=1, pct=True, na_option='keep')``.
    NaN values are preserved (not ranked).
    """
    return x.rank(axis=1, pct=True, na_option='keep')


def _panel_scale(x: pd.DataFrame, a=1) -> pd.DataFrame:
    """Cross-sectional scale: rescale so sum(|x|) over stocks equals *a*.

    Matches MASE's ``_fn_scale`` and the WorldQuant Alpha101 spec
    ("scale so sum|x|=1").  The 1D ``stat_helpers.scale`` does z-score
    instead — a pre-existing discrepancy, but the panel version follows
    the spec because the LLM prompt advertises WorldQuant semantics.
    """
    aval = a if isinstance(a, (int, float)) else 1.0
    denom = x.abs().sum(axis=1, skipna=True).replace(0, np.nan)
    return x.div(denom, axis=0) * aval


def _panel_ts_rank(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Per-stock rolling rank (time-series, same as ts_rank but per column)."""
    return x.apply(lambda col: ts_rank(col, window))


def _panel_ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling argmax (0-based index of max within trailing window)."""
    return x.apply(lambda col: ts_argmax(col, window))


def _panel_ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling argmin (0-based index of min within trailing window)."""
    return x.apply(lambda col: ts_argmin(col, window))


def _panel_decay_linear(x: pd.DataFrame, window: int = None,
                        period: int = None) -> pd.DataFrame:
    """Per-stock linear-decay weighted moving average."""
    return x.apply(lambda col: decay_linear(col, window, period))


def _panel_sma(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock simple moving average."""
    return x.rolling(window, min_periods=1).mean()


def _panel_stddev(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling standard deviation (population, ddof=0)."""
    return x.rolling(window, min_periods=1).std(ddof=0)


def _panel_correlation(x: pd.DataFrame, y: pd.DataFrame,
                       window: int) -> pd.DataFrame:
    """Per-stock rolling Pearson correlation between *x* and *y*."""
    return x.rolling(window, min_periods=1).corr(y)


def _panel_covariance(x: pd.DataFrame, y: pd.DataFrame,
                      window: int) -> pd.DataFrame:
    """Per-stock rolling covariance between *x* and *y*."""
    return x.rolling(window, min_periods=1).cov(y)


def _panel_ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling sum."""
    return x.rolling(window, min_periods=1).sum()


def _panel_ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling minimum."""
    return x.rolling(window, min_periods=1).min()


def _panel_ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling maximum."""
    return x.rolling(window, min_periods=1).max()


def _panel_delta(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Per-stock difference over *period* lags."""
    return x.diff(period)


def _panel_delay(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Per-stock lag by *period* periods."""
    return x.shift(period)


def _panel_sign(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise sign (+1, 0, -1)."""
    return np.sign(x)


def _panel_product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-stock rolling product."""
    return x.rolling(window, min_periods=1).apply(np.prod, raw=True)


def factor_panel_fn(panel_df: pd.DataFrame, expr: str) -> pd.Series:
    """Evaluate an Alpha101 expression on a **panel** (MultiIndex date,
    ticker) DataFrame using **cross-sectional** semantics for ``rank()``
    and ``scale()``.

    This is the panel-level counterpart of :func:`factor_series_fn`.
    Instead of evaluating per-ticker (where ``rank()`` becomes a
    time-series rank on a 1D Series), it unstacks the panel to 2D
    (date × ticker) DataFrames so that:

    * ``rank(x)`` ranks **across stocks** on each date (cross-sectional,
      matching the LLM prompt's "cross-sectional rank into [0,1]").
    * ``scale(x)`` scales so ``sum(|x|)=1`` across stocks on each date
      (matching the WorldQuant Alpha101 spec).
    * All ``ts_*`` functions operate **per-stock** (per-column of the 2D
      frame), preserving their time-series semantics.

    Args:
        panel_df: MultiIndex ``(date, ticker)`` DataFrame with columns
            such as ``open, high, low, close, volume, vwap, returns, vol``.
        expr: Factor expression string (Alpha101 DSL).

    Returns:
        MultiIndex ``(date, ticker)`` Series of factor values.

    Raises:
        ValueError: If the expression references an unknown name or fails
            to evaluate.
        TypeError: If the expression does not return a 2D structure.
    """
    # 1) AlphaFactory method names — fall back to per-ticker evaluation.
    #    Factory methods expect single-ticker DataFrames and handle their
    #    own rank() internally; they cannot be evaluated on a panel.
    if hasattr(AlphaFactory, expr):
        result_list = []
        for _tkr, grp in panel_df.groupby('ticker'):
            series = factor_series_fn(grp, expr)
            result_list.append(
                pd.Series(series.values, index=grp.index, name=expr)
            )
        return pd.concat(result_list)

    # 2) Unstack each column to 2D (date × ticker)
    local_ns = {}
    for col in panel_df.columns:
        local_ns[col] = panel_df[col].unstack('ticker')

    # 3) Detect advXX patterns and add moving-average 2D frames
    adv_windows = {int(w) for w in re.findall(r"adv(\d+)", expr)}
    for w in adv_windows:
        local_ns[f"adv{w}"] = local_ns["volume"].rolling(
            w, min_periods=1
        ).mean()

    # 4) Build panel-aware function namespace
    func_ns = {
        "ts_rank": _panel_ts_rank,
        "delta": _panel_delta,
        "ts_sum": _panel_ts_sum,
        "ts_min": _panel_ts_min,
        "ts_max": _panel_ts_max,
        "decay_linear": _panel_decay_linear,
        "ts_argmax": _panel_ts_argmax,
        "ts_argmin": _panel_ts_argmin,
        "delay": _panel_delay,
        "stddev": _panel_stddev,
        "correlation": _panel_correlation,
        "rank": _panel_rank,
        "sma": _panel_sma,
        "scale": _panel_scale,
        "sign": _panel_sign,
        "covariance": _panel_covariance,
        "product": _panel_product,
        "np": np,
    }

    # 5) Evaluate (using the cached compiled code object)
    try:
        result = eval(_compile_expr(expr), func_ns, local_ns)
    except NameError as e:
        raise ValueError(f"Unknown name in expression {expr!r}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to eval expression {expr!r}: {e}")

    # 6) Stack the 2D result back to a MultiIndex (date, ticker) Series.
    #    stack() drops NaN by default; reindex restores all (date, ticker)
    #    pairs from the original panel (with NaN where the 2D result was
    #    NaN).  This avoids the pandas ≥2.1 FutureWarning on dropna=False.
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Expression did not return a DataFrame: got {type(result)}"
        )

    result_series = result.stack()
    result_series.name = expr
    # Reindex to match the original panel_df index (ensures consistent
    # ordering and preserves all (date, ticker) pairs, including NaNs).
    return result_series.reindex(panel_df.index)


def compute_rankic(series: pd.Series, returns: pd.Series) -> float:
    """
    Compute cross-sectional Rank-IC between a factor series and the return
    series, averaged across dates.

    Both inputs should share a MultiIndex (date, ticker). For each date, the
    factor values and returns are ranked across tickers (Spearman = Pearson of
    ranks), and the per-date IC is averaged over all dates.

    This mirrors the logic in ``factor_matrix.compute_ic_matrix`` but operates
    on a single factor Series rather than a DataFrame of exposures. The
    previous implementation pooled all (date, ticker) rows and computed a
    single Spearman correlation, which mixed cross-sectional and time-series
    information and systematically inflated IC.
    """
    # Align on common (date, ticker) pairs
    common = series.index.intersection(returns.index)
    s = series.loc[common]
    r = returns.loc[common]

    if len(s) < 2 or len(r) < 2:
        return 0.0

    # Cross-sectional Rank-IC (panel data with date level)
    if isinstance(s.index, pd.MultiIndex) and 'date' in s.index.names:
        # Rank within each date (cross-sectional), method='average' matches scipy
        sr = s.groupby(level="date", group_keys=False).rank()
        rr = r.groupby(level="date", group_keys=False).rank()

        # Center within each date
        sm = sr - sr.groupby(level="date", group_keys=False).transform("mean")
        rm = rr - rr.groupby(level="date", group_keys=False).transform("mean")

        # Per-date Pearson of ranks = Spearman per date
        num = (sm * rm).groupby(level="date", group_keys=False).sum()
        den_s = (sm ** 2).groupby(level="date", group_keys=False).sum()
        den_r = (rm ** 2).groupby(level="date", group_keys=False).sum()
        den = np.sqrt(den_s * den_r)

        ic_series = num / den
        ic_series = ic_series.replace([np.inf, -np.inf], np.nan)

        mean_ic = ic_series.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    # Fallback: simple Spearman for non-panel data (e.g. single-ticker Series)
    dfv = pd.concat([s, r], axis=1).dropna()
    if dfv.shape[0] < 2:
        return 0.0
    ic = spearmanr(dfv.iloc[:, 0], dfv.iloc[:, 1]).correlation
    return float(ic) if not np.isnan(ic) else 0.0
