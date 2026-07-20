import numpy as np
import pandas as pd


def compute_ic_matrix(exposures, returns, n_jobs=None):
    """
    Compute the Rank-IC matrix (information coefficient) between each factor's
    exposure and the forward return, per trading date.

    Args:
        exposures: MultiIndex [date, ticker] x factors
        returns:   MultiIndex [date, ticker] x ['returns']
        n_jobs:    Retained ONLY for API compatibility with callers. The
                   vectorized implementation below is a single pass over the
                   panel and is far faster than the old ``scipy.stats.spearmanr``
                   loop (and than process parallelism over dates), so the
                   argument is intentionally ignored.

    Returns:
        pd.DataFrame  index=dates, columns=factor names, values=Rank-IC.

    Why this is correct (and matches the old scipy output to ~1e-12)
    -----------------------------------------------------------------
    Rank-IC is, by definition, the Pearson correlation of the *ranks* of the
    factor and the return. ``scipy.stats.spearmanr(x, y)`` ranks ``x`` and ``y``
    (average ties) and then computes their Pearson correlation. We do exactly
    that, but vectorized across the whole (date x ticker) panel in one shot:
      rank within each date  ->  center within each date  ->  Pearson of ranks.
    The per-date Pearson is ``sum((x-mx)(y-my)) / sqrt(sum((x-mx)^2) sum((y-my)^2))``,
    computed simultaneously for every factor via broadcasting.
    """
    # Align on common (date, ticker) pairs so ranks are comparable.
    common = exposures.index.intersection(returns.index)
    X = exposures.loc[common]
    Y = returns.loc[common, "returns"]  # Series with MultiIndex (date, ticker)

    if len(X) == 0 or len(Y) == 0:
        return pd.DataFrame(
            columns=list(exposures.columns),
            index=pd.Index([], name="date",
                           dtype=X.index.get_level_values("date").dtype),
        )

    # Rank within each date across tickers. `method='average'` is the same tie
    # handling scipy uses internally for spearmanr.
    Xr = X.groupby(level="date", group_keys=False).rank()
    Yr = Y.groupby(level="date", group_keys=False).rank()

    # Center each date's ranks (per factor for X, shared for Y).
    Xm = Xr - Xr.groupby(level="date", group_keys=False).transform("mean")
    Ym = Yr - Yr.groupby(level="date", group_keys=False).transform("mean")

    # Numerator: sum over tickers of (Xm * Ym) per (date, factor).
    # Ym is a single Series broadcast across all factor columns (axis=0 align).
    num = (Xm.mul(Ym, axis=0)).groupby(level="date", group_keys=False).sum()

    # Denominator: sqrt( sum(Xm^2) * sum(Ym^2) ).
    den_x = (Xm ** 2).groupby(level="date", group_keys=False).sum()
    den_y = (Ym ** 2).groupby(level="date", group_keys=False).sum()  # Series over dates
    den = np.sqrt(den_x.mul(den_y, axis=0))  # broadcast den_y across factors

    corr = num.div(den)
    # Zero-variance dates (constant factor or return) -> den == 0 -> inf/nan;
    # scipy returns NaN there, so normalize inf to NaN for parity.
    corr = corr.replace([np.inf, -np.inf], np.nan)

    # Restore factor column order and a clean, date-sorted index.
    corr = corr[list(exposures.columns)]
    corr.index = pd.Index(corr.index, name="date")
    corr = corr.sort_index()
    return corr
