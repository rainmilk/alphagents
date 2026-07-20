import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def ts_rank(df: pd.Series, window: int = 10) -> pd.Series:
    """Rolling rank of the LAST element within a trailing window (Alpha101 ts_rank).

    Equivalent to ``scipy.stats.rankdata(arr, method='min')[-1]`` for each
    trailing window, but fully vectorized. With ``method='min'``, the rank of
    the last value equals ``1 + (number of elements in the window strictly
    less than it)`` — so we count with a single broadcast instead of calling
    ``rankdata`` once per window (the old, ~100x slower path).
    """
    arr = np.asarray(df.to_numpy(dtype=float)).ravel()
    n = arr.size
    out = np.full(n, np.nan)
    if n >= window and window > 0:
        win = sliding_window_view(arr, window)            # (n-window+1, window)
        last = win[:, -1]
        counts = (win[:, :-1] < last[:, None]).sum(axis=1)  # strictly-less count
        ranks = 1.0 + counts
        # Match pandas Rolling.apply default min_periods=window: any NaN in the
        # window -> NaN (rankdata would return NaN too).
        nanmask = np.isnan(win).any(axis=1)
        ranks = np.where(nanmask, np.nan, ranks)
        out[window - 1:] = ranks
    return pd.Series(out, index=df.index, name=df.name)


def delta(df: pd.Series, period: int = 1) -> pd.Series:
    return df.diff(period)


def delay(df: pd.Series, period: int = 1) -> pd.Series:
    return df.shift(period)


def decay_linear(series: pd.Series, window: int = None, period: int = None) -> pd.Series:
    """
    Rolling linear-decay over the last `window` bars (alias `period` for legacy calls).
    Weights are 1/n, 2/n, …, n/n. Vectorized via a single matrix product for the
    full windows; the short warm-up tail (min_periods=1) uses a tiny explicit loop
    so behavior is identical to the old rolling().apply().
    """
    w = period if period is not None else window
    if w is None:
        w = 10
    arr = np.asarray(series.to_numpy(dtype=float)).ravel()
    n = arr.size
    out = np.full(n, np.nan)
    # warm-up: variable-length windows 1..w-1 (min_periods=1 semantics)
    for i in range(min(w - 1, n)):
        seg = arr[: i + 1]
        weights = np.arange(1, seg.size + 1) / seg.size
        out[i] = float(np.dot(seg, weights))
    # main: fixed-length windows -> one matmul
    if n >= w:
        win = sliding_window_view(arr, w)               # (n-w+1, w)
        weights = np.arange(1, w + 1) / w
        out[w - 1:] = win @ weights
    return pd.Series(out, index=series.index, name=series.name)


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).sum()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """0-based index of the maximum within the trailing window (min_periods=1).

    Replicates ``rolling(window, min_periods=1).apply(np.argmax, raw=True)``
    exactly (numpy's argmax is used directly, so NaN handling matches).
    """
    arr = np.asarray(series.to_numpy(dtype=float)).ravel()
    n = arr.size
    out = np.full(n, np.nan)
    for i in range(min(window - 1, n)):
        out[i] = np.argmax(arr[: i + 1])
    if n >= window:
        win = sliding_window_view(arr, window)
        out[window - 1:] = np.argmax(win, axis=1)
    return pd.Series(out, index=series.index, name=series.name)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """0-based index of the minimum within the trailing window (min_periods=1)."""
    arr = np.asarray(series.to_numpy(dtype=float)).ravel()
    n = arr.size
    out = np.full(n, np.nan)
    for i in range(min(window - 1, n)):
        out[i] = np.argmin(arr[: i + 1])
    if n >= window:
        win = sliding_window_view(arr, window)
        out[window - 1:] = np.argmin(win, axis=1)
    return pd.Series(out, index=series.index, name=series.name)
