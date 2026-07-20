# -*- coding: utf-8 -*-
"""
AlphaGrail Baseline Runner - Integrated with Main DataLoader & BacktestEngine
=============================================================================

AlphaGrail: "Automate-Strategy-Finding-with-LLM-in-Quant-investment"

Core methodology (from baselines/AlphaGrail/):
  1. Seed Alpha Generation: 35 alpha factor formulas (from Seed Alpha.xlsx)
     - Uses real fundamentals (EPS, PB, GPM) when available
     - Falls back to clearly-marked price-based proxies otherwise
     - Duplicates from original Excel removed (PriceReturn14d_v2, SMA20_minus_Close)
  2. Factor Evaluation: comprehensive metrics matching original AlphaGrail's
     FactorAnalysisEngine output:
     - IC summary (mean_ic, ic_std, icir, ic_positive_ratio)
     - Quantile returns (monotonicity score, long-short spread)
     - Quantile turnover
     - Factor returns (Sharpe, volatility)
     - Max drawdown
     - Optional: industry + size neutralization
  3. LLM Tournament Selection: linear king-of-the-hill (sequential comparison)
     via LLM (GPT-4o) with full metric set, or quantitative fallback
  4. Portfolio Construction: top-N stocks by winning factor score, equal-weight
  5. Backtest: unified BacktestEngine (commission=0.0003, slippage=0.001)

Data: main project DataLoader (NOT rqfactor/rqdatac)
Backtest: unified BacktestEngine (same as all other baselines)

References:
  - baselines/AlphaGrail/main.py          (factor analysis engine)
  - baselines/AlphaGrail/AutoGPT/main.py   (LLM tournament selection)
  - baselines/AlphaGrail/data/Seed Alpha.xlsx (37 seed formulas)

Author: Code Review Expert
Date: 2026-07-02 (updated 2026-07-03)
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlphaGrailConfig:
    """Configuration for AlphaGrail baseline."""
    top_n_stocks: int = 50          # Portfolio size
    holding_period: int = 1         # Rebalance frequency (1=daily)
    n_seed_factors: int = 37        # Number of seed alpha formulas
    use_llm_tournament: bool = False # Use LLM for tournament (needs API key)
    tournament_metric: str = "icir" # Fallback metric: "icir", "sharpe", "composite"
    train_window: int = 250         # Training window (trading days)
    retrain_freq: int = 20          # Re-evaluate factors every N days
    min_stocks_per_day: int = 10    # Minimum stocks with valid factor values
    forward_period: int = 10        # Forward return period (must match MASE forward_period)
    n_quantiles: int = 5            # Number of quantile groups for group return analysis
    use_neutralization: bool = False # Apply industry+size neutralization (needs industry_data)
    # LLM config (read from config.yaml llm.generator section)
    llm_api_key: str = ""           # API key for LLM service
    llm_base_url: str = ""          # Base URL for LLM service
    llm_model: str = "gpt-4o"       # Model name for LLM comparison


# ═══════════════════════════════════════════════════════════════════════
#  Factor Engine — implements the 37 seed alpha formulas from Seed Alpha.xlsx
#  using pure pandas/numpy (no rqfactor dependency)
# ═══════════════════════════════════════════════════════════════════════

# ── Helper functions (rqfactor → pandas equivalents) ──────────────────

def _delay(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """DELAY(x, n) = x.shift(n)"""
    return series.shift(n)

def _sma(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """SMA / MA(x, n) = rolling mean"""
    return series.rolling(n, min_periods=1).mean()

def _ema(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """EMA(x, n) = exponential moving average"""
    return series.ewm(span=n, adjust=False).mean()

def _std(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """STD(x, n) = rolling std"""
    return series.rolling(n, min_periods=2).std()

def _rolling_max(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """MAX(x, n) = rolling max"""
    return series.rolling(n, min_periods=1).max()

def _rolling_min(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """MIN(x, n) = rolling min"""
    return series.rolling(n, min_periods=1).min()

def _rolling_sum(series: pd.DataFrame, n: int) -> pd.DataFrame:
    """SUM(x, n) = rolling sum"""
    return series.rolling(n, min_periods=1).sum()

def _compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR).

    TR = max(H - L, |H - prev_close|, |L - prev_close|)
    ATR = EMA(TR, n)
    """
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce([
            (high - low).values,
            (high - prev_close).fillna(0).values,
            (low - prev_close).fillna(0).values,
        ]),
        index=high.index, columns=high.columns
    )
    return _ema(tr, n)

def _compute_rsi(close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Compute RSI (Relative Strength Index)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = _ema(gain, n)
    avg_loss = _ema(loss, n)
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Factor definitions ────────────────────────────────────────────────
# Each entry: (name, function) where function takes price_data dict and
# returns a DataFrame (date × stock) of factor values.
# All 37 factors from Seed Alpha.xlsx are implemented.

def _build_factor_library(price_data: Dict[str, pd.DataFrame],
                          fundamental_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
    """
    Build all seed alpha factors from Seed Alpha.xlsx.

    Original AlphaGrail uses rqfactor with real fundamentals (EPS, PB, gross_profit_margin).
    When fundamental_data is provided, real fundamentals are used.
    Otherwise, price-based proxies are used and clearly marked with _proxy suffix.

    Duplicates from the original Seed Alpha.xlsx (e.g. PriceReturn14d == PriceReturn14d_v2,
    MeanReversion == SMA20_minus_Close) are removed to avoid biasing the tournament.

    Args:
        price_data: dict with keys 'open', 'high', 'low', 'close', 'volume', 'amount'
        fundamental_data: optional dict with keys like 'eps', 'pb_ratio_ttm',
                          'gross_profit_margin_ttm', etc.

    Returns:
        Dict mapping factor name to DataFrame (date x stock) of factor values.
    """
    close = price_data['close']
    high = price_data.get('high', close)
    low = price_data.get('low', close)
    volume = price_data.get('volume', pd.DataFrame(1, index=close.index, columns=close.columns))
    amount = price_data.get('amount', close * volume)

    # Precompute common indicators
    atr = _compute_atr(high, low, close, 14)
    rsi = _compute_rsi(close, 10)

    factors = {}

    # ── Momentum factors ──────────────────────────────────────────────

    # 1. PriceMomentum = CLOSE - DELAY(CLOSE, 14)
    factors['PriceMomentum'] = close - _delay(close, 14)

    # 2. VolumeMomentum = VOLUME - DELAY(VOLUME, 14)
    factors['VolumeMomentum'] = volume - _delay(volume, 14)

    # 3. RSIMomentum = RSI - DELAY(RSI, 14)
    factors['RSIMomentum'] = rsi - _delay(rsi, 14)

    # 4. PriceReturn14d = (CLOSE / DELAY(CLOSE, 14)) - 1
    #    NOTE: Original Seed Alpha has a duplicate (#5) with identical formula — removed.
    factors['PriceReturn14d'] = (close / _delay(close, 14)) - 1

    # 5. RSI_manual = (SUM(gain,14) - SUM(loss,14)) / (SUM(gain,14) + SUM(loss,14)) * 100
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    sum_gain = _rolling_sum(gain, 14)
    sum_loss = _rolling_sum(loss, 14)
    factors['RSI_manual'] = ((sum_gain - sum_loss) / (sum_gain + sum_loss + 1e-10)) * 100

    # 6. Stochastic_K = ((CLOSE - MIN(LOW,14)) - (MAX(HIGH,14) - CLOSE)) / (MAX(HIGH,14) - MIN(LOW,14))
    min_low_14 = _rolling_min(low, 14)
    max_high_14 = _rolling_max(high, 14)
    factors['Stochastic_K'] = ((close - min_low_14) - (max_high_14 - close)) / (max_high_14 - min_low_14 + 1e-10)

    # 7. ATRMomentum = ATR - DELAY(ATR, 14)
    factors['ATRMomentum'] = atr - _delay(atr, 14)

    # 8. Price_vs_SMA14_delay7 = CLOSE - DELAY(SMA(CLOSE,14), 7)
    factors['Price_vs_SMA14_delay7'] = close - _delay(_sma(close, 14), 7)

    # ── Mean reversion factors ────────────────────────────────────────

    # 9. MeanReversion = MA(CLOSE,20) - CLOSE
    #    NOTE: Original Seed Alpha has a duplicate (#13 SMA20_minus_Close) — removed.
    factors['MeanReversion'] = _sma(close, 20) - close

    # 10. ZScoreMeanReversion = (CLOSE - MA(CLOSE,20)) / STD(CLOSE,20)
    factors['ZScoreMeanReversion'] = (close - _sma(close, 20)) / (_std(close, 20) + 1e-10)

    # 11. BollingerBands = (CLOSE - MA(CLOSE,20)) / (2 * STD(CLOSE,20))
    factors['BollingerBands'] = (close - _sma(close, 20)) / (2 * _std(close, 20) + 1e-10)

    # 12. EMA20_minus_Close = EMA(CLOSE,20) - CLOSE
    factors['EMA20_minus_Close'] = _ema(close, 20) - close

    # 13. MaxHigh20_minus_Close = MAX(HIGH,20) - CLOSE
    factors['MaxHigh20_minus_Close'] = _rolling_max(high, 20) - close

    # 14. Close_minus_MinLow20 = CLOSE - MIN(LOW,20)
    factors['Close_minus_MinLow20'] = close - _rolling_min(low, 20)

    # 15. InverseRSI = 100 - RSI
    factors['InverseRSI'] = 100 - rsi

    # ── Volatility factors ────────────────────────────────────────────

    # 16. StandardDeviation = STD(CLOSE,20)
    factors['StandardDeviation'] = _std(close, 20)

    # 17. ATR
    factors['ATR'] = atr

    # 18. BollingerBandWidth = (BOLL_UP - BOLL_DOWN) / SMA(CLOSE,20)
    boll_up = _sma(close, 20) + 2 * _std(close, 20)
    boll_down = _sma(close, 20) - 2 * _std(close, 20)
    factors['BollingerBandWidth'] = (boll_up - boll_down) / (_sma(close, 20) + 1e-10)

    # 19. VolRatio_10_50 = STD(CLOSE,10) / STD(CLOSE,50)
    factors['VolRatio_10_50'] = _std(close, 10) / (_std(close, 50) + 1e-10)

    # 20. HL_EMA_ratio = (EMA(HIGH-LOW,10) / DELAY(EMA(HIGH-LOW,10),10)) - 1
    hl_ema = _ema(high - low, 10)
    factors['HL_EMA_ratio'] = (hl_ema / (_delay(hl_ema, 10) + 1e-10)) - 1

    # ── Fundamental factors (use real data if available, else proxy) ──

    if fundamental_data and 'eps' in fundamental_data:
        # 21. PE = CLOSE / EPS (real)
        factors['PE'] = close / (fundamental_data['eps'].reindex(index=close.index, columns=close.columns) + 1e-10)
    else:
        # 21. PE_proxy — price-based proxy (1/close). NOT equivalent to real PE.
        factors['PE_proxy'] = 1.0 / (close + 1e-10)

    if fundamental_data and 'pb_ratio_ttm' in fundamental_data:
        # 22. PB (real)
        factors['PB'] = fundamental_data['pb_ratio_ttm'].reindex(index=close.index, columns=close.columns)
    else:
        # 22. PB_proxy — price-based proxy. NOT equivalent to real PB.
        factors['PB_proxy'] = 1.0 / (close + 1e-10)

    if fundamental_data and 'gross_profit_margin_ttm' in fundamental_data:
        # 23. GrossProfitMargin (real)
        factors['GrossProfitMargin'] = fundamental_data['gross_profit_margin_ttm'].reindex(index=close.index, columns=close.columns)
    else:
        # 23. GrossProfitMargin_proxy — 60d return as proxy. NOT equivalent to real GPM.
        factors['GrossProfitMargin_proxy'] = (close / _delay(close, 60) - 1)

    if fundamental_data and 'profit_from_operation_to_revenue_ttm' in fundamental_data:
        # 24. OperatingProfitMargin (real)
        factors['OperatingProfitMargin'] = fundamental_data['profit_from_operation_to_revenue_ttm'].reindex(index=close.index, columns=close.columns)
    else:
        # 24. OperatingProfitMargin_proxy — 20d return as proxy. NOT equivalent.
        factors['OperatingProfitMargin_proxy'] = (close / _delay(close, 20) - 1)

    if fundamental_data and 'peg_ratio_ttm' in fundamental_data:
        # 25. EarningsGrowthRate (real, via PEG)
        factors['EarningsGrowthRate'] = fundamental_data['peg_ratio_ttm'].reindex(index=close.index, columns=close.columns)
    else:
        # 25. EarningsGrowthRate_proxy — 90d return as proxy. NOT equivalent.
        factors['EarningsGrowthRate_proxy'] = (close / _delay(close, 90) - 1)

    # 26. EBITDAGrowthRate = EBITDA / DELAY(EBITDA,1) - 1
    if fundamental_data and 'ebitda_ttm' in fundamental_data:
        ebitda = fundamental_data['ebitda_ttm'].reindex(index=close.index, columns=close.columns)
        factors['EBITDAGrowthRate'] = ebitda / (_delay(ebitda, 1) + 1e-10) - 1
    else:
        # 26. EBITDAGrowthRate_proxy — dollar volume growth as proxy. NOT equivalent.
        dollar_vol = volume * close
        factors['EBITDAGrowthRate_proxy'] = dollar_vol / (_delay(dollar_vol, 1) + 1e-10) - 1

    # ── Volume / liquidity factors ────────────────────────────────────

    # 27. Volume
    factors['Volume'] = volume

    # 28. VolumeReturn14d = (VOLUME - DELAY(VOLUME,14)) / DELAY(VOLUME,14)
    factors['VolumeReturn14d'] = (volume - _delay(volume, 14)) / (_delay(volume, 14) + 1e-10)

    # 29. Turnover_proxy = VOLUME / CLOSE (proxy for turnover since no shares_outstanding)
    factors['Turnover_proxy'] = volume / (close + 1e-10)

    # 30. AverageTradingVolume = MA(VOLUME,20)
    factors['AverageTradingVolume'] = _sma(volume, 20)

    # 31. DailyRange = (HIGH - LOW) / CLOSE
    factors['DailyRange'] = (high - low) / (close + 1e-10)

    # 32. DollarVolume = VOLUME * CLOSE
    factors['DollarVolume'] = volume * close

    # ── Moving average factors ────────────────────────────────────────

    # 33. ExponentialMovingAverage = EMA(CLOSE,20)
    factors['ExponentialMovingAverage'] = _ema(close, 20)

    # 34. Stochastic_pct_K = ((CLOSE - MIN(LOW,14)) / (MAX(HIGH,14) - MIN(LOW,14))) * 100
    factors['Stochastic_pct_K'] = ((close - min_low_14) / (max_high_14 - min_low_14 + 1e-10)) * 100

    # 35. Stochastic_pct_D_inv = ((MAX(HIGH,14) - CLOSE) / (MAX(HIGH,14) - MIN(LOW,14))) * -100
    factors['Stochastic_pct_D_inv'] = ((max_high_14 - close) / (max_high_14 - min_low_14 + 1e-10)) * -100

    return factors


# ═══════════════════════════════════════════════════════════════════════
#  Factor Neutralization (mirrors original AlphaGrail's Neutralization pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _neutralize_factor(
    factor_values: pd.DataFrame,
    industry_data: Optional[pd.DataFrame] = None,
    size_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply industry and size neutralization to factor values.

    Mirrors the original AlphaGrail's Neutralization(industry='citics_2019',
    style_factors=['size', 'beta', ...]) pipeline. This implementation does
    a simplified cross-sectional regression: factor ~ industry_dummies + log(size).

    Args:
        factor_values: Factor values (date x stock)
        industry_data: Industry membership (date x stock), values are industry codes
        size_data: Market cap or size proxy (date x stock)

    Returns:
        Neutralized factor values (residuals after regression)
    """
    result = factor_values.copy()

    for date in result.index:
        row = result.loc[date].dropna()
        if len(row) < 20:
            continue

        # Build regression matrix
        regressors = []

        # Industry dummies
        if industry_data is not None:
            if isinstance(industry_data, pd.DataFrame) and date in industry_data.index:
                # (date x stock) panel: pick this date's column of industry codes
                ind = industry_data.loc[date].reindex(row.index)
            elif isinstance(industry_data, pd.Series):
                # Static per-stock classification (index = stock codes): the
                # DataLoader returns this shape. Broadcast the same mapping
                # across all dates since industry membership is time-invariant
                # in this simplified setup.
                ind = industry_data.reindex(row.index)
            else:
                ind = None
            if ind is not None:
                dummies = pd.get_dummies(ind, drop_first=True)
                if dummies.shape[1] > 0:
                    regressors.append(dummies.values.astype(float))

        # Size factor (log market cap)
        if size_data is not None and date in size_data.index:
            sz = size_data.loc[date].reindex(row.index)
            log_sz = np.log(sz.clip(lower=1e-10))
            if log_sz.notna().all():
                regressors.append(log_sz.values.reshape(-1, 1))

        if not regressors:
            continue

        X = np.column_stack(regressors) if len(regressors) > 1 else regressors[0]
        X = np.column_stack([np.ones(len(row)), X])  # intercept

        y = row.values.astype(float)

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            result.loc[date, row.index] = residuals
        except np.linalg.LinAlgError:
            continue

    return result


def _neutralize_factor_library(
    factor_library: Dict[str, pd.DataFrame],
    industry_data: Optional[pd.DataFrame] = None,
    size_data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Apply neutralization to all factors in the library."""
    if industry_data is None and size_data is None:
        return factor_library

    neutralized = {}
    for name, values in factor_library.items():
        neutralized[name] = _neutralize_factor(values, industry_data, size_data)
    return neutralized


# ═══════════════════════════════════════════════════════════════════════
#  Factor Evaluation — computes the full metric set that the original
#  AlphaGrail feeds to GPT-4o (IC summary, quantile returns, turnover,
#  factor returns, max drawdown, volatility)
# ═══════════════════════════════════════════════════════════════════════

def _compute_forward_returns(close: pd.DataFrame, periods: List[int] = [1]) -> Dict[int, pd.DataFrame]:
    """
    Compute forward returns for given periods.

    Args:
        close: Close price DataFrame (date × stock)
        periods: List of holding periods (in days)

    Returns:
        Dict mapping period → returns DataFrame
    """
    returns = {}
    for p in periods:
        returns[p] = close.shift(-p) / close - 1
    return returns


def _compute_rank_ic(factor_values: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """
    Compute mean Rank IC (Spearman correlation) between factor and forward returns.

    Uses pure numpy (no scipy dependency).

    Args:
        factor_values: Factor values (date × stock)
        forward_returns: Forward returns (date × stock)

    Returns:
        Mean daily Rank IC
    """
    ic_list = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < 10:
            continue

        f_aligned = f[common].rank()
        r_aligned = r[common].rank()

        # Pearson correlation of ranks = Spearman
        f_mean = f_aligned.mean()
        r_mean = r_aligned.mean()
        f_std = f_aligned.std()
        r_std = r_aligned.std()

        if f_std > 0 and r_std > 0:
            ic = ((f_aligned - f_mean) * (r_aligned - r_mean)).mean() / (f_std * r_std)
            if np.isfinite(ic):
                ic_list.append(ic)

    return float(np.mean(ic_list)) if ic_list else 0.0


def _compute_rank_ic_series(factor_values: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily Rank IC time series.

    Args:
        factor_values: Factor values (date × stock)
        forward_returns: Forward returns (date × stock)

    Returns:
        Series of daily Rank IC values, indexed by date.
    """
    ic_series = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < 10:
            ic_series.append(np.nan)
            continue

        f_aligned = f[common].rank()
        r_aligned = r[common].rank()

        f_std = f_aligned.std()
        r_std = r_aligned.std()

        if f_std > 0 and r_std > 0:
            ic = ((f_aligned - f_aligned.mean()) * (r_aligned - r_aligned.mean())).mean() / (f_std * r_std)
            ic_series.append(ic if np.isfinite(ic) else np.nan)
        else:
            ic_series.append(np.nan)

    return pd.Series(ic_series, index=common_dates, dtype=float)


def _compute_factor_sharpe(factor_values: pd.DataFrame, forward_returns: pd.DataFrame, top_n: int = 50) -> float:
    """
    Compute factor-mimicking portfolio Sharpe ratio.
    Long top-N stocks, short bottom-N stocks, daily rebalance.

    Args:
        factor_values: Factor values (date x stock)
        forward_returns: Forward returns (date x stock)
        top_n: Number of stocks for long/short

    Returns:
        Annualized Sharpe ratio
    """
    daily_returns = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < 2 * top_n:
            continue

        f_aligned = f[common]
        r_aligned = r[common]

        long_stocks = f_aligned.nlargest(top_n).index
        short_stocks = f_aligned.nsmallest(top_n).index

        long_ret = r_aligned[long_stocks].mean()
        short_ret = r_aligned[short_stocks].mean()

        daily_returns.append(long_ret - short_ret)

    if len(daily_returns) < 20:
        return 0.0

    rets = np.array(daily_returns)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    if std_ret == 0:
        return 0.0

    # Annualized Sharpe (252 trading days)
    return float(np.sqrt(252) * mean_ret / std_ret)


def _compute_quantile_returns(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> Tuple[pd.Series, float]:
    """
    Compute quantile portfolio returns and monotonicity score.

    Mirrors original AlphaGrail's QuantileReturnAnalysis(quantile=5).
    A good factor should show monotonic returns across quantiles.

    Args:
        factor_values: Factor values (date x stock)
        forward_returns: Forward returns (date x stock)
        n_quantiles: Number of quantile groups

    Returns:
        (mean_returns_by_quantile, monotonicity_score)
        monotonicity_score: 1.0 = perfectly monotonic, 0.0 = random
    """
    quantile_daily_returns = {q: [] for q in range(n_quantiles)}
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < n_quantiles * 2:
            continue

        f_aligned = f[common]
        r_aligned = r[common]

        # Assign quantile labels (0 = lowest factor value, n-1 = highest)
        try:
            labels = pd.qcut(f_aligned, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            continue

        for q in range(n_quantiles):
            mask = labels == q
            if mask.sum() > 0:
                quantile_daily_returns[q].append(r_aligned[mask].mean())

    # Average returns per quantile
    mean_returns = []
    for q in range(n_quantiles):
        if quantile_daily_returns[q]:
            mean_returns.append(np.mean(quantile_daily_returns[q]))
        else:
            mean_returns.append(0.0)

    mean_returns = pd.Series(mean_returns, index=range(n_quantiles))

    # Monotonicity: fraction of adjacent pairs that are correctly ordered
    if len(mean_returns) < 2:
        return mean_returns, 0.0

    diffs = np.diff(mean_returns.values)
    n_positive = (diffs > 0).sum()
    n_negative = (diffs < 0).sum()
    monotonicity = max(n_positive, n_negative) / len(diffs)

    return mean_returns, float(monotonicity)


def _compute_quantile_turnover(
    factor_values: pd.DataFrame,
    n_quantiles: int = 5,
) -> float:
    """
    Compute average quantile portfolio turnover.

    Mirrors original AlphaGrail's quantile_turnover metric.
    Lower turnover = more stable factor.

    Args:
        factor_values: Factor values (date x stock)
        n_quantiles: Number of quantile groups

    Returns:
        Average daily turnover (0-1 scale)
    """
    turnovers = []
    dates = factor_values.index.tolist()

    for i in range(1, len(dates)):
        prev = factor_values.loc[dates[i-1]].dropna()
        curr = factor_values.loc[dates[i]].dropna()
        common = prev.index.intersection(curr.index)

        if len(common) < n_quantiles * 2:
            continue

        try:
            prev_labels = pd.qcut(prev[common], n_quantiles, labels=False, duplicates='drop')
            curr_labels = pd.qcut(curr[common], n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            continue

        # Turnover for top quantile (most relevant for trading)
        prev_top = set(prev_labels[prev_labels == n_quantiles - 1].index)
        curr_top = set(curr_labels[curr_labels == n_quantiles - 1].index)

        if prev_top or curr_top:
            overlap = len(prev_top & curr_top)
            total = len(prev_top | curr_top)
            turnover = 1.0 - (overlap / total) if total > 0 else 0.0
            turnovers.append(turnover)

    return float(np.mean(turnovers)) if turnovers else 0.0


def _compute_max_drawdown(factor_values: pd.DataFrame, forward_returns: pd.DataFrame, top_n: int = 50) -> float:
    """
    Compute max drawdown of the factor-mimicking long-short portfolio.

    Mirrors original AlphaGrail's max_drawdown() metric.

    Args:
        factor_values: Factor values (date x stock)
        forward_returns: Forward returns (date x stock)
        top_n: Number of stocks for long/short

    Returns:
        Maximum drawdown (positive number, e.g. 0.15 = 15% drawdown)
    """
    daily_returns = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < 2 * top_n:
            continue

        f_aligned = f[common]
        r_aligned = r[common]

        long_stocks = f_aligned.nlargest(top_n).index
        short_stocks = f_aligned.nsmallest(top_n).index

        daily_returns.append(r_aligned[long_stocks].mean() - r_aligned[short_stocks].mean())

    if len(daily_returns) < 20:
        return 0.0

    cumret = np.cumprod(1 + np.array(daily_returns))
    running_max = np.maximum.accumulate(cumret)
    drawdowns = (cumret - running_max) / running_max
    return float(abs(np.min(drawdowns)))


def _compute_factor_volatility(factor_values: pd.DataFrame, forward_returns: pd.DataFrame, top_n: int = 50) -> float:
    """
    Compute annualized volatility of the factor-mimicking long-short portfolio.

    Mirrors original AlphaGrail's std() metric.

    Args:
        factor_values: Factor values (date x stock)
        forward_returns: Forward returns (date x stock)
        top_n: Number of stocks for long/short

    Returns:
        Annualized volatility
    """
    daily_returns = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < 2 * top_n:
            continue

        f_aligned = f[common]
        r_aligned = r[common]

        long_stocks = f_aligned.nlargest(top_n).index
        short_stocks = f_aligned.nsmallest(top_n).index

        daily_returns.append(r_aligned[long_stocks].mean() - r_aligned[short_stocks].mean())

    if len(daily_returns) < 20:
        return 0.0

    return float(np.std(daily_returns, ddof=1) * np.sqrt(252))


def evaluate_factors(
    factor_library: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    top_n: int = 50,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Evaluate all factors and return a comprehensive summary DataFrame.

    Computes the full metric set that the original AlphaGrail feeds to GPT-4o:
    - IC summary: mean_ic, ic_std, icir, ic_positive_ratio
    - Quantile returns: monotonicity score
    - Quantile turnover: average top-quantile turnover
    - Factor returns: factor Sharpe
    - Max drawdown of factor-mimicking portfolio
    - Factor return volatility (annualized)

    Args:
        factor_library: Dict mapping factor name to values DataFrame
        forward_returns: Forward returns (date x stock)
        top_n: Number of stocks for factor-mimicking portfolio
        n_quantiles: Number of quantile groups

    Returns:
        DataFrame with comprehensive evaluation metrics per factor.
    """
    records = []

    for name, values in factor_library.items():
        # IC summary
        mean_ic = _compute_rank_ic(values, forward_returns)

        ic_series = _compute_rank_ic_series(values, forward_returns)
        ic_clean = ic_series.dropna()
        ic_std = float(ic_clean.std()) if len(ic_clean) > 1 else 0.0
        icir = float(ic_clean.mean() / ic_std) if ic_std > 0 else 0.0
        ic_pos_ratio = float((ic_clean > 0).mean()) if len(ic_clean) > 0 else 0.0

        # Factor-mimicking portfolio metrics
        factor_sharpe = _compute_factor_sharpe(values, forward_returns, top_n)
        max_drawdown = _compute_max_drawdown(values, forward_returns, top_n)
        factor_volatility = _compute_factor_volatility(values, forward_returns, top_n)

        # Quantile analysis
        quantile_returns, monotonicity = _compute_quantile_returns(values, forward_returns, n_quantiles)
        quantile_turnover = _compute_quantile_turnover(values, n_quantiles)

        # Long-short spread (Q_top - Q_bottom)
        if len(quantile_returns) >= 2:
            ls_spread = float(quantile_returns.iloc[-1] - quantile_returns.iloc[0])
        else:
            ls_spread = 0.0

        records.append({
            'factor_name': name,
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_pos_ratio,
            'factor_sharpe': factor_sharpe,
            'factor_volatility': factor_volatility,
            'max_drawdown': max_drawdown,
            'monotonicity': monotonicity,
            'quantile_turnover': quantile_turnover,
            'ls_spread': ls_spread,
            'abs_ic': abs(mean_ic),
            'abs_icir': abs(icir),
        })

    return pd.DataFrame(records).set_index('factor_name')


# ═══════════════════════════════════════════════════════════════════════
#  Tournament Selection (core AlphaGrail innovation)
# ═══════════════════════════════════════════════════════════════════════

def _llm_compare_factors(
    factor_a_name: str, factor_a_metrics: Dict,
    factor_b_name: str, factor_b_metrics: Dict,
    api_key: str = "",
    base_url: str = "",
    model: str = "gpt-4o",
) -> str:
    """
    Use LLM to compare two alpha factors and select the better one.

    Mirrors the AutoGPT tournament in baselines/AlphaGrail/AutoGPT/main.py,
    where GPT-4o receives a comprehensive factor analysis Excel with:
      - IC summary, quantile returns, quantile turnover,
        factor returns, max drawdown, volatility

    This implementation passes the same metric set as a structured prompt
    (since we don't have Code Interpreter / Assistant API access here).

    Args:
        factor_a_name: Name of factor A
        factor_a_metrics: Dict with full evaluation metrics
        factor_b_name: Name of factor B
        factor_b_metrics: Same structure
        api_key: API key for LLM service (falls back to env OPENAI_API_KEY)
        base_url: Base URL for LLM service (falls back to env OPENAI_BASE_URL)
        model: Model name for LLM comparison

    Returns:
        "A" or "B" indicating the winner
    """
    try:
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL", None)

        if not resolved_key:
            raise ValueError("No API key provided (param or OPENAI_API_KEY env)")

        client = OpenAI(api_key=resolved_key, base_url=resolved_url) if resolved_url else OpenAI(api_key=resolved_key)

        def _fmt(m, prefix=""):
            """Format all metrics for the prompt."""
            return f"""{prefix}Mean Rank IC:        {m.get('mean_ic', 0):+.4f}
{prefix}ICIR:                 {m.get('icir', 0):+.4f}
{prefix}IC Std:               {m.get('ic_std', 0):.4f}
{prefix}IC Positive Ratio:    {m.get('ic_positive_ratio', 0):.2%}
{prefix}Factor Sharpe:        {m.get('factor_sharpe', 0):+.4f}
{prefix}Factor Volatility:    {m.get('factor_volatility', 0):.4f}
{prefix}Max Drawdown:         {m.get('max_drawdown', 0):.4f}
{prefix}Monotonicity Score:   {m.get('monotonicity', 0):.2%}
{prefix}Quantile Turnover:    {m.get('quantile_turnover', 0):.4f}
{prefix}Long-Short Spread:    {m.get('ls_spread', 0):+.6f}"""

        prompt = f"""You are a quantitative finance expert. Compare two alpha factors and select the better one.

Consider ALL of the following dimensions:
1. IC magnitude and stability (ICIR = mean_ic / ic_std)
2. Factor-mimicking portfolio risk-adjusted return (Sharpe)
3. IC consistency (ic_positive_ratio — what fraction of days IC is positive)
4. Quantile monotonicity (higher = more monotonic returns across groups = better factor)
5. Drawdown and volatility (lower = more stable)
6. Turnover (lower = lower transaction costs)
7. Long-short spread (larger absolute = stronger signal)

=== Factor A: {factor_a_name} ===
{_fmt(factor_a_metrics, "  ")}

=== Factor B: {factor_b_name} ===
{_fmt(factor_b_metrics, "  ")}

Based on your expertise, which factor is better overall? Reply with ONLY "A" or "B"."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().upper()
        return "A" if "A" in answer else "B"

    except Exception as e:
        print(f"    LLM comparison failed ({e}), falling back to quantitative")
        return _quantitative_compare(factor_a_metrics, factor_b_metrics)


def _quantitative_compare(
    metrics_a: Dict,
    metrics_b: Dict,
) -> str:
    """
    Quantitative fallback for factor comparison.
    Uses a composite score incorporating all available metrics:
    IC stability (ICIR), risk-adjusted return (Sharpe), monotonicity,
    drawdown penalty, and turnover penalty.

    Args:
        metrics_a: Factor A metrics dict
        metrics_b: Factor B metrics dict

    Returns:
        "A" or "B"
    """
    def composite_score(m):
        icir = abs(m.get('icir', 0))
        sharpe = abs(m.get('factor_sharpe', 0))
        ic = abs(m.get('mean_ic', 0))
        ic_pos = m.get('ic_positive_ratio', 0.5)
        mono = m.get('monotonicity', 0.5)
        mdd = m.get('max_drawdown', 0.0)
        turnover = m.get('quantile_turnover', 0.5)
        ls_spread = abs(m.get('ls_spread', 0))

        # Higher is better: icir, sharpe, ic, ic_pos, mono, ls_spread
        # Lower is better: mdd, turnover
        return (
            0.25 * icir +
            0.20 * sharpe +
            0.10 * ic +
            0.10 * ic_pos +
            0.10 * mono +
            0.10 * ls_spread * 100 +  # scale up since ls_spread is tiny
            0.10 * (1 - min(mdd, 1.0)) +  # drawdown penalty
            0.05 * (1 - min(turnover, 1.0))  # turnover penalty
        )

    score_a = composite_score(metrics_a)
    score_b = composite_score(metrics_b)

    return "A" if score_a >= score_b else "B"


def tournament_selection(
    factor_eval: pd.DataFrame,
    use_llm: bool = False,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "gpt-4o",
) -> Tuple[str, Dict]:
    """
    Run tournament selection to find the best factor.

    Implements the original AlphaGrail's linear king-of-the-hill tournament
    (AutoGPT/main.py L186-200):
      - Start with the first factor as the current champion
      - Each subsequent factor challenges the champion
      - If the challenger wins, it becomes the new champion
      - After all challenges, the final champion is the winner

    This is NOT a bracket elimination tournament. The comparison order
    matters (first factor has a slight advantage as it only needs to defend,
    not attack), matching the original implementation's behavior.

    Args:
        factor_eval: Evaluation DataFrame from evaluate_factors()
        use_llm: If True, use LLM for comparison (requires API key)
        llm_api_key: API key for LLM service
        llm_base_url: Base URL for LLM service
        llm_model: Model name for LLM comparison

    Returns:
        Tuple of (winner_factor_name, winner_metrics_dict)
    """
    factors = list(factor_eval.index)
    n = len(factors)

    print(f"\n  Tournament: {n} factors, {'LLM' if use_llm else 'quantitative'} mode")
    print(f"    Mode: linear king-of-the-hill (sequential comparison)")
    if use_llm and llm_model:
        print(f"    LLM: {llm_model} @ {llm_base_url or 'default'}")

    # Sort by abs_icir descending so stronger factors are compared later
    # (gives a more meaningful tournament — weak factors are eliminated early)
    factors_sorted = factor_eval['abs_icir'].sort_values(ascending=True).index.tolist()

    champion = factors_sorted[0]
    champion_metrics = factor_eval.loc[champion].to_dict()
    n_defenses = 0

    print(f"    Round 1: Initial champion = {champion}")

    for i in range(1, len(factors_sorted)):
        challenger = factors_sorted[i]
        challenger_metrics = factor_eval.loc[challenger].to_dict()

        if use_llm:
            winner = _llm_compare_factors(
                champion, champion_metrics, challenger, challenger_metrics,
                api_key=llm_api_key,
                base_url=llm_base_url,
                model=llm_model,
            )
        else:
            winner = _quantitative_compare(champion_metrics, challenger_metrics)

        if winner == "B":
            # Challenger wins — new champion
            print(f"    Round {i+1}: {champion} vs {challenger} → {challenger} (NEW champion, defended {n_defenses}x)")
            champion = challenger
            champion_metrics = challenger_metrics
            n_defenses = 0
        else:
            # Champion defends
            n_defenses += 1
            if i <= 5 or i >= len(factors_sorted) - 3:
                # Print early and final rounds
                print(f"    Round {i+1}: {champion} vs {challenger} → {champion} (defended {n_defenses}x)")

    print(f"\n  Tournament Winner: {champion} (survived {n_defenses} final defenses)")
    print(f"    Mean IC:        {champion_metrics['mean_ic']:.4f}")
    print(f"    ICIR:           {champion_metrics['icir']:.4f}")
    print(f"    Sharpe:         {champion_metrics['factor_sharpe']:.4f}")
    print(f"    Monotonicity:   {champion_metrics.get('monotonicity', 0):.2%}")
    print(f"    Max Drawdown:   {champion_metrics.get('max_drawdown', 0):.4f}")
    print(f"    Turnover:       {champion_metrics.get('quantile_turnover', 0):.4f}")

    return champion, champion_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Portfolio Construction & Backtest
# ═══════════════════════════════════════════════════════════════════════

def _normalize_cross_section(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score normalization (per date).
    Removes extreme outliers via winsorization at ±3 std.

    Args:
        factor_values: Raw factor values (date × stock)

    Returns:
        Normalized factor values
    """
    result = factor_values.copy()

    for date in result.index:
        row = result.loc[date]
        valid = row.dropna()
        if len(valid) < 2:
            continue

        median = valid.median()
        mad = (valid - median).abs().median()
        if mad > 0:
            # MAD-winsorize at ±3
            capped = valid.clip(lower=median - 3 * 1.4826 * mad, upper=median + 3 * 1.4826 * mad)
        else:
            capped = valid

        mean = capped.mean()
        std = capped.std()
        if std > 0:
            result.loc[date, capped.index] = (capped - mean) / std
        else:
            result.loc[date, capped.index] = 0.0

    return result


def build_portfolios_from_factor(
    factor_values: pd.DataFrame,
    top_n: int = 50,
    test_start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build equal-weight long-only portfolios from factor scores.
    Selects top-N stocks by normalized factor value each day.

    Args:
        factor_values: Factor values (date × stock)
        top_n: Number of stocks to hold
        test_start_date: If provided, only build portfolios from this date onward

    Returns:
        Portfolio weights DataFrame (date × stock), each row sums to 1.0
    """
    # Filter to test period
    if test_start_date is not None:
        test_start_ts = pd.Timestamp(test_start_date)
        factor_values = factor_values[factor_values.index >= test_start_ts]

    if factor_values.empty:
        return pd.DataFrame()

    # Normalize cross-sectionally
    normalized = _normalize_cross_section(factor_values)

    portfolio_rows = []
    portfolio_dates = []

    for date in normalized.index:
        scores = normalized.loc[date].dropna()
        if len(scores) < top_n:
            n = len(scores)
        else:
            n = top_n

        if n == 0:
            continue

        # Select top-N stocks (highest factor value = most bullish)
        top_stocks = scores.nlargest(n)
        weights = pd.Series(1.0 / n, index=top_stocks.index)

        portfolio_rows.append(weights)
        portfolio_dates.append(date)

    if not portfolio_rows:
        return pd.DataFrame()

    # Build DataFrame
    all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
    portfolios = pd.DataFrame(
        index=pd.DatetimeIndex(portfolio_dates),
        columns=all_stocks,
        dtype=float,
    )
    for i, w in enumerate(portfolio_rows):
        portfolios.loc[portfolio_dates[i], w.index] = w.values

    portfolios = portfolios.fillna(0.0)
    # Ensure each row sums to 1.0
    row_sums = portfolios.sum(axis=1)
    portfolios = portfolios.div(row_sums, axis=0).fillna(0.0)

    return portfolios


# ═══════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def run_alphagrail_baseline(
    config_path: str = "config/config.yaml",
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    universe: Optional[str] = None,
    top_n_stocks: int = 50,
    holding_period: Optional[int] = None,
    use_llm_tournament: bool = False,
    forward_period: Optional[int] = None,
    n_quantiles: int = 5,
    use_neutralization: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run AlphaGrail baseline using the main project's DataLoader.

    Pipeline:
      1. Load A-share data via main DataLoader
      2. Build seed alpha factors (from Seed Alpha.xlsx)
         - Uses real fundamentals if available, else price-based proxies
         - Optional: industry + size neutralization
      3. Evaluate factors with comprehensive metrics:
         IC, ICIR, Sharpe, quantile returns, turnover, drawdown, volatility
      4. Tournament selection (LLM or quantitative) — linear king-of-the-hill
      5. Build portfolios from winning factor on test data
      6. Backtest with unified BacktestEngine

    Args:
        config_path: Path to main project config file.
        train_start_date: Data start date (YYYY-MM-DD).
        test_end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        top_n_stocks: Number of stocks in portfolio.
        holding_period: Rebalance frequency (1=daily, 5=weekly, 20=monthly).
        use_llm_tournament: If True, use LLM for tournament (needs OpenAI API key).
        forward_period: Forward return period in days (should match MASE forward_period).
        n_quantiles: Number of quantile groups for group return analysis.
        use_neutralization: If True, apply industry + size neutralization.
        output_dir: Directory for saving results.

    Returns:
        Dict of performance metrics.
    """
    from dataloader.loader import DataLoader
    from backtest.engine import BacktestEngine

    print("=" * 60)
    print("  AlphaGrail Baseline — LLM-Driven Alpha Selection (via Main DataLoader)")
    print("=" * 60)
    print(f"  Quantiles: {n_quantiles} | Neutralize: {use_neutralization}")

    # ── Step 1: Load data via main DataLoader ──────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)

    # ── Resolve forward_period / holding_period from config ──────────
    # explicit arg > config.yaml > default, so standalone runs also honor config.
    _ev_cfg = loader.config.get('evolution', {})
    _bt_cfg = loader.config.get('backtest', {}).get('trading', {})
    if not forward_period or forward_period <= 0:
        forward_period = _ev_cfg.get('forward_period', 10)
    if not holding_period or holding_period <= 0:
        holding_period = _bt_cfg.get('holding_period', 1)
    print(f"  Forward period: {forward_period}d | Holding period: {holding_period}d")
    train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')

    bundle = loader.load_data(universe=universe, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    train_price, train_fund, train_ind = bundle.train
    test_price, test_fund, test_ind = bundle.test
    # FULL span (close / fundamental / industry) — used for factor building,
    # neutralization size proxy, and backtest price alignment.
    price_data = bundle.full[0]
    fundamental_data = bundle.full[1]
    industry_data = bundle.full[2]

    close = price_data['close']
    n_dates = len(close.index)
    n_stocks = len(close.columns)
    print(f"  Loaded: {n_dates} trading days x {n_stocks} stocks")
    if fundamental_data:
        print(f"  Fundamental data available: {list(fundamental_data.keys())[:5]}...")
    else:
        print(f"  No fundamental data — will use price-based proxies for PE/PB/GPM/etc.")

    # ── Step 2: Train/test split ───────────────────────────────────────
    # The split is produced centrally by loader.load_data (bundle.train /
    # bundle.test); we use the slices' own date indices instead of re-deriving
    # boundaries from train_end/test_start timestamps.
    print(f"  Train end: {train_end}, Test start: {test_start}")
    train_dates = train_price['close'].index
    test_dates = test_price['close'].index

    # ── Step 3: Build seed alpha factors ───────────────────────────────
    print("\n[Step 2] Building seed alpha factors...")
    factor_library = _build_factor_library(price_data, fundamental_data)
    n_proxy = sum(1 for k in factor_library if '_proxy' in k)
    print(f"  Built {len(factor_library)} factors ({n_proxy} using price-based proxies)")

    # ── Step 3b: Optional neutralization ───────────────────────────────
    if use_neutralization:
        print("\n[Step 2b] Applying industry + size neutralization...")
        # Use close * volume as size proxy if no market_cap available
        size_data = None
        if fundamental_data and 'market_cap' in fundamental_data:
            size_data = fundamental_data['market_cap']
        else:
            size_data = close * price_data.get('volume', pd.DataFrame(1, index=close.index, columns=close.columns))

        ind_data = None
        if industry_data is not None:
            # DataLoader returns industry_data as a pd.Series indexed by stock
            # code (see loader.load_data: "industry_series: Series, index =
            # stock codes"). The old check only accepted pd.DataFrame and then
            # called .get('industry') on the Series, which returns None
            # (Series.get looks up a *label*, not a key) -> ind_data stayed
            # None and industry neutralization was silently skipped.
            if isinstance(industry_data, (pd.DataFrame, pd.Series)):
                ind_data = industry_data
            elif hasattr(industry_data, 'get'):
                ind_data = industry_data.get('industry')
            else:
                ind_data = industry_data

        if ind_data is not None or size_data is not None:
            factor_library = _neutralize_factor_library(factor_library, ind_data, size_data)
            print(f"  Neutralized {len(factor_library)} factors")
        else:
            print("  Warning: No industry or size data available, skipping neutralization")

    # ── Step 4: Compute forward returns ────────────────────────────────
    print(f"\n[Step 3] Computing {forward_period}d forward returns...")
    forward_returns = _compute_forward_returns(close, periods=[forward_period])[forward_period]

    # ── Step 5: Evaluate factors on training data ──────────────────────
    print("\n[Step 4] Evaluating factors on training data (comprehensive metrics)...")
    train_returns = forward_returns.loc[forward_returns.index.isin(train_dates)]

    train_factors = {
        name: vals.loc[vals.index.isin(train_dates)] for name, vals in factor_library.items()
    }

    factor_eval = evaluate_factors(train_factors, train_returns, top_n=top_n_stocks, n_quantiles=n_quantiles)
    print(f"\n  Factor evaluation (top 10 by |ICIR|):")
    top_eval = factor_eval.reindex(factor_eval['abs_icir'].nlargest(10).index)
    for name, row in top_eval.iterrows():
        print(f"    {name:35s}  IC={row['mean_ic']:+.4f}  ICIR={row['icir']:+.4f}  "
              f"Sharpe={row['factor_sharpe']:+.4f}  Mono={row['monotonicity']:.0%}  "
              f"MDD={row['max_drawdown']:.4f}")

    # ── Step 6: Tournament selection ───────────────────────────────────
    print("\n[Step 5] Tournament selection...")

    llm_api_key = ""
    llm_base_url = ""
    llm_model = "gpt-4o"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        llm_cfg = main_config.get('llm', {}).get('generator', {})
        llm_api_key = llm_cfg.get('api_key', '')
        llm_base_url = llm_cfg.get('base_url', '')
        llm_model = llm_cfg.get('model', 'gpt-4o')
    except Exception:
        pass
    if not llm_api_key:
        llm_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not llm_base_url:
        llm_base_url = os.environ.get("OPENAI_BASE_URL", "")

    if use_llm_tournament and not llm_api_key:
        print("  Warning: use_llm_tournament=True but no API key found, falling back to quantitative")
        use_llm_tournament = False

    winner, winner_metrics = tournament_selection(
        factor_eval,
        use_llm=use_llm_tournament,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )

    # ── Step 7: Build portfolios on test data ──────────────────────────
    print(f"\n[Step 6] Building portfolios using winning factor: {winner}...")
    test_factor_values = factor_library[winner]
    portfolios = build_portfolios_from_factor(
        test_factor_values,
        top_n=top_n_stocks,
        test_start_date=test_start,
    )

    if portfolios.empty:
        print("  Warning: No valid portfolios built")
        return {
            'method': 'AlphaGrail',
            'winning_factor': winner,
            'n_factors': len(factor_library),
            'mean_rank_ic': float(winner_metrics.get('mean_ic', 0)),
            'icir': float(winner_metrics.get('icir', 0)),
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'information_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'avg_turnover': 0.0,
            'train_end': train_end,
            'test_start': test_start,
            'forward_period': forward_period,
            'train_start': train_start,
            'test_end': test_end,
            'holding_period': holding_period,
        }

    print(f"  Portfolios: {portfolios.shape[0]} days x {portfolios.shape[1]} stocks")

    # ── Step 8: Backtest with unified BacktestEngine ───────────────────
    print("\n[Step 7] Running backtest (unified BacktestEngine)...")

    prices_aligned = close.reindex(portfolios.index)
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    run_dir = None
    method_name = "alphagrail"
    if output_dir:
        _u = universe or loader.data_config.get('universe', {}).get('index', 'hs300')
        _s = train_start_date or loader.data_config.get('train_start_date', 'na')
        _e = test_end_date or loader.data_config.get('test_end_date', 'na')
        _fp = forward_period if forward_period is not None else 10
        _hp = holding_period if holding_period is not None else 1
        param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
        run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
        os.makedirs(run_dir, exist_ok=True)
    metrics = engine.run(portfolios, prices_aligned, save_dir=run_dir)

    # ── Step 9: Compute test-period IC for the winning factor ──────────
    test_returns = forward_returns.loc[forward_returns.index.isin(test_dates)]
    test_factor = factor_library[winner].loc[factor_library[winner].index.isin(test_dates)]
    test_ic = _compute_rank_ic(test_factor, test_returns)
    # Also compute test ICIR from the daily Rank-IC series (reported in tables).
    test_ic_series = _compute_rank_ic_series(test_factor, test_returns)
    if len(test_ic_series) > 1 and test_ic_series.std(ddof=1) > 0:
        test_icir = float(test_ic_series.mean() / test_ic_series.std(ddof=1))
    else:
        test_icir = 0.0

    # ── Step 10: Compile results ───────────────────────────────────────
    results = {
        'method': 'AlphaGrail',
        'winning_factor': winner,
        'n_factors': len(factor_library),
        'n_proxy_factors': n_proxy,
        'forward_period': forward_period,
        'use_neutralization': use_neutralization,
        'mean_rank_ic_train': float(winner_metrics.get('mean_ic', 0)),
        'icir': float(winner_metrics.get('icir', 0)),
        'factor_sharpe_train': float(winner_metrics.get('factor_sharpe', 0)),
        'mean_rank_ic_test': float(test_ic),
        'icir_test': float(test_icir),
        'annual_return': metrics.get('annual_return', 0.0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'information_ratio': metrics.get('information_ratio', 0.0),
        'calmar_ratio': metrics.get('calmar_ratio', 0.0),
        'win_rate': metrics.get('win_rate', 0.0),
        'avg_turnover': metrics.get('avg_turnover', 0.0),
        'n_trading_days': metrics.get('n_trading_days', 0),
        'train_end': train_end,
        'test_start': test_start,
        'train_start': train_start,
        'test_end': test_end,
        'holding_period': holding_period,
        'use_llm_tournament': use_llm_tournament,
        'llm_model': llm_model if use_llm_tournament else None,
        'tournament_winner_metrics': {
            'mean_ic': float(winner_metrics.get('mean_ic', 0)),
            'icir': float(winner_metrics.get('icir', 0)),
            'factor_sharpe': float(winner_metrics.get('factor_sharpe', 0)),
            'ic_positive_ratio': float(winner_metrics.get('ic_positive_ratio', 0)),
            'monotonicity': float(winner_metrics.get('monotonicity', 0)),
            'max_drawdown': float(winner_metrics.get('max_drawdown', 0)),
            'quantile_turnover': float(winner_metrics.get('quantile_turnover', 0)),
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("  AlphaGrail Baseline Complete")
    print("=" * 60)
    print(f"  Winning Factor:   {winner}")
    print(f"  Forward Period:   {forward_period}d")
    print(f"  Mean Rank-IC (train): {results['mean_rank_ic_train']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
    print(f"  Mean Rank-IC (test):  {results['mean_rank_ic_test']:.4f}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")

    # ── Step 11: Save results ──────────────────────────────────────────
    if output_dir:
        result_path = os.path.join(run_dir, 'alphagrail_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

        eval_path = os.path.join(run_dir, 'factor_evaluation.csv')
        factor_eval.to_csv(eval_path)
        print(f"  Factor evaluation saved to {eval_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run AlphaGrail baseline with main DataLoader'
    )
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to main config')
    parser.add_argument('--train-start', default=None,
                        help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', default=None,
                        help='Data end date (YYYY-MM-DD)')
    parser.add_argument('--universe', default=None,
                        help='Stock universe (hs300, zz500, all_a)')
    parser.add_argument('--train-end', default=None,
                        help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', default=None,
                        help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of stocks in portfolio')
    parser.add_argument('--holding-period', type=int, default=1,
                        help='Rebalance frequency (1=daily, 5=weekly)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for tournament (needs OpenAI API key)')
    parser.add_argument('--forward-period', type=int, default=10,
                        help='Forward return period in days (should match MASE forward_period)')
    parser.add_argument('--n-quantiles', type=int, default=5,
                        help='Number of quantile groups for group return analysis')
    parser.add_argument('--neutralize', action='store_true',
                        help='Apply industry + size neutralization to factors')
    parser.add_argument('--output-dir', default='experiments/alphagrail',
                        help='Output directory')

    args = parser.parse_args()

    results = run_alphagrail_baseline(
        config_path=args.config,
        universe=args.universe,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        top_n_stocks=args.top_n,
        holding_period=args.holding_period,
        use_llm_tournament=args.use_llm,
        forward_period=args.forward_period,
        n_quantiles=args.n_quantiles,
        use_neutralization=args.neutralize,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Winning Factor:   {results['winning_factor']}")
    print(f"  Forward Period:   {results['forward_period']}d")
    print(f"  Mean Rank-IC (test): {results['mean_rank_ic_test']:.4f}")
    print(f"  ICIR (test):         {results['icir_test']:.4f}")
    print(f"  Turnover:            {results['avg_turnover']:.4f}")
