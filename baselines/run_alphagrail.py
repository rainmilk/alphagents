# -*- coding: utf-8 -*-
"""
AlphaGrail Baseline Runner - Integrated with Main DataLoader & BacktestEngine
=============================================================================

AlphaGrail: "Automate-Strategy-Finding-with-LLM-in-Quant-investment"

Core methodology (from baselines/AlphaGrail/):
  1. Seed Alpha Generation: 37 alpha factor formulas (from Seed Alpha.xlsx)
  2. Factor Evaluation: compute IC, Rank-IC, ICIR, factor-mimicking portfolio metrics
  3. LLM Tournament Selection: pairwise comparison via LLM (GPT-4o) or
     quantitative fallback (composite ICIR/Sharpe score)
  4. Portfolio Construction: top-N stocks by winning factor score, equal-weight
  5. Backtest: unified BacktestEngine (commission=0.0003, slippage=0.001)

Data: main project DataLoader (NOT rqfactor/rqdatac)
Backtest: unified BacktestEngine (same as all other baselines)

References:
  - baselines/AlphaGrail/main.py          (factor analysis engine)
  - baselines/AlphaGrail/AutoGPT/main.py   (LLM tournament selection)
  - baselines/AlphaGrail/data/Seed Alpha.xlsx (37 seed formulas)

Author: Code Review Expert (火眼眼)
Date: 2026-07-02
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

def _build_factor_library(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build all 37 seed alpha factors from Seed Alpha.xlsx.

    Args:
        price_data: dict with keys 'open', 'high', 'low', 'close', 'volume', 'amount'
                    Each value is a DataFrame (date × stock).

    Returns:
        Dict mapping factor name → DataFrame (date × stock) of factor values.
    """
    close = price_data['close']
    high = price_data.get('high', close)
    low = price_data.get('low', close)
    volume = price_data.get('volume', pd.DataFrame(1, index=close.index, columns=close.columns))
    amount = price_data.get('amount', close * volume)

    # Precompute common indicators
    atr = _compute_atr(high, low, close, 14)
    rsi = _compute_rsi(close, 10)
    vwap = amount / (volume + 1e-10)

    factors = {}

    # 1. PriceMomentum = CLOSE - DELAY(CLOSE, 14)
    factors['PriceMomentum'] = close - _delay(close, 14)

    # 2. VolumeMomentum = VOLUME - DELAY(VOLUME, 14)
    factors['VolumeMomentum'] = volume - _delay(volume, 14)

    # 3. RSIMomentum = RSI - DELAY(RSI, 14)
    factors['RSIMomentum'] = rsi - _delay(rsi, 14)

    # 4. ((CLOSE / DELAY(CLOSE, 14)) - 1)
    factors['PriceReturn14d'] = (close / _delay(close, 14)) - 1

    # 5. ((CLOSE - DELAY(CLOSE, 14)) / DELAY(CLOSE, 14))  — same as #4, keep for fidelity
    factors['PriceReturn14d_v2'] = (close - _delay(close, 14)) / (_delay(close, 14) + 1e-10)

    # 6. RSI-like: (SUM(gain, 14) - SUM(loss, 14)) / (SUM(gain, 14) + SUM(loss, 14)) * 100
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    sum_gain = _rolling_sum(gain, 14)
    sum_loss = _rolling_sum(loss, 14)
    factors['RSI_manual'] = ((sum_gain - sum_loss) / (sum_gain + sum_loss + 1e-10)) * 100

    # 7. Stochastic %K: ((CLOSE - MIN(LOW, 14)) - (MAX(HIGH, 14) - CLOSE)) / (MAX(HIGH, 14) - MIN(LOW, 14))
    min_low_14 = _rolling_min(low, 14)
    max_high_14 = _rolling_max(high, 14)
    factors['Stochastic_K'] = ((close - min_low_14) - (max_high_14 - close)) / (max_high_14 - min_low_14 + 1e-10)

    # 8. (ATR - DELAY(ATR, 14))
    factors['ATRMomentum'] = atr - _delay(atr, 14)

    # 9. (CLOSE - DELAY(SMA(CLOSE, 14), 7))
    factors['Price_vs_SMA14_delay7'] = close - _delay(_sma(close, 14), 7)

    # 10. MeanReversion = MA(CLOSE, 20) - CLOSE
    factors['MeanReversion'] = _sma(close, 20) - close

    # 11. ZScoreMeanReversion = (CLOSE - MA(CLOSE, 20)) / STD(CLOSE, 20)
    factors['ZScoreMeanReversion'] = (close - _sma(close, 20)) / (_std(close, 20) + 1e-10)

    # 12. BollingerBands = (CLOSE - MA(CLOSE, 20)) / (2 * STD(CLOSE, 20))
    factors['BollingerBands'] = (close - _sma(close, 20)) / (2 * _std(close, 20) + 1e-10)

    # 13. (SMA(CLOSE, 20) - CLOSE)  — same as MeanReversion, keep for fidelity
    factors['SMA20_minus_Close'] = _sma(close, 20) - close

    # 14. (EMA(CLOSE, 20) - CLOSE)
    factors['EMA20_minus_Close'] = _ema(close, 20) - close

    # 15. (MAX(HIGH, 20) - CLOSE)
    factors['MaxHigh20_minus_Close'] = _rolling_max(high, 20) - close

    # 16. (CLOSE - MIN(LOW, 20))
    factors['Close_minus_MinLow20'] = close - _rolling_min(low, 20)

    # 17. (100 - RSI)
    factors['InverseRSI'] = 100 - rsi

    # 18. StandardDeviation = STD(CLOSE, 20)
    factors['StandardDeviation'] = _std(close, 20)

    # 19. ATR
    factors['ATR'] = atr

    # 20. BollingerBandWidth = (BOLL_UP - BOLL_DOWN) / SMA(CLOSE, 20)
    #     BOLL_UP = MA(CLOSE, 20) + 2*STD(CLOSE, 20), BOLL_DOWN = MA(CLOSE, 20) - 2*STD(CLOSE, 20)
    boll_up = _sma(close, 20) + 2 * _std(close, 20)
    boll_down = _sma(close, 20) - 2 * _std(close, 20)
    factors['BollingerBandWidth'] = (boll_up - boll_down) / (_sma(close, 20) + 1e-10)

    # 21. STD(CLOSE, 10) / STD(CLOSE, 50)
    factors['VolRatio_10_50'] = _std(close, 10) / (_std(close, 50) + 1e-10)

    # 22. (EMA(HIGH - LOW, 10) / DELAY(EMA(HIGH - LOW, 10), 10)) - 1
    hl_ema = _ema(high - low, 10)
    factors['HL_EMA_ratio'] = (hl_ema / (_delay(hl_ema, 10) + 1e-10)) - 1

    # 23. PE = CLOSE / EPS — fundamental, use inverse of price as proxy if no fundamentals
    #     We use 1/close as a price-based proxy (cheaper stocks score higher)
    factors['PE_proxy'] = 1.0 / (close + 1e-10)

    # 24. PB — fundamental, use book-to-market proxy (1/close)
    factors['PB_proxy'] = 1.0 / (close + 1e-10)

    # 25. VOLUME
    factors['Volume'] = volume

    # 26. (VOLUME - DELAY(VOLUME, 14)) / DELAY(VOLUME, 14)
    factors['VolumeReturn14d'] = (volume - _delay(volume, 14)) / (_delay(volume, 14) + 1e-10)

    # 27. VOLUME / MARKET_CAP — use volume / (close * shares_outstanding) ≈ volume / amount * close
    #     Since we don't have shares_outstanding, use turnover proxy: volume / close
    factors['Turnover_proxy'] = volume / (close + 1e-10)

    # 28. AverageTradingVolume = MA(VOLUME, 20)
    factors['AverageTradingVolume'] = _sma(volume, 20)

    # 29. (HIGH - LOW) / CLOSE
    factors['DailyRange'] = (high - low) / (close + 1e-10)

    # 30. VOLUME * CLOSE (dollar volume)
    factors['DollarVolume'] = volume * close

    # 31. GrossProfitMargin — fundamental, not available; use price momentum as proxy
    factors['GrossProfitMargin_proxy'] = (close / _delay(close, 60) - 1)

    # 32. OperatingProfitMargin — fundamental, not available; use 20d return as proxy
    factors['OperatingProfitMargin_proxy'] = (close / _delay(close, 20) - 1)

    # 33. EarningsGrowthRate — fundamental, not available; use 90d return as proxy
    factors['EarningsGrowthRate_proxy'] = (close / _delay(close, 90) - 1)

    # 34. EBITDAGrowthRate = EBITDA / DELAY(EBITDA, 1) - 1
    #     Use dollar volume as EBITDA proxy
    factors['EBITDAGrowthRate_proxy'] = factors['DollarVolume'] / (_delay(factors['DollarVolume'], 1) + 1e-10) - 1

    # 35. ExponentialMovingAverage = EMA(CLOSE, 20)
    factors['ExponentialMovingAverage'] = _ema(close, 20)

    # 36. ((CLOSE - MIN(LOW, 14)) / (MAX(HIGH, 14) - MIN(LOW, 14))) * 100  (Stochastic %K)
    factors['Stochastic_pct_K'] = ((close - min_low_14) / (max_high_14 - min_low_14 + 1e-10)) * 100

    # 37. ((MAX(HIGH, 14) - CLOSE) / (MAX(HIGH, 14) - MIN(LOW, 14))) * -100  (Stochastic %D inverted)
    factors['Stochastic_pct_D_inv'] = ((max_high_14 - close) / (max_high_14 - min_low_14 + 1e-10)) * -100

    return factors


# ═══════════════════════════════════════════════════════════════════════
#  Factor Evaluation
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
        factor_values: Factor values (date × stock)
        forward_returns: Forward returns (date × stock)
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


def evaluate_factors(
    factor_library: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Evaluate all factors and return a summary DataFrame.

    Args:
        factor_library: Dict mapping factor name → values DataFrame
        forward_returns: Forward 1-day returns (date × stock)
        top_n: Number of stocks for factor-mimicking portfolio

    Returns:
        DataFrame with columns: factor_name, mean_ic, ic_std, icir,
            factor_sharpe, ic_positive_ratio
    """
    records = []

    for name, values in factor_library.items():
        # Mean Rank IC
        mean_ic = _compute_rank_ic(values, forward_returns)

        # IC time series for ICIR
        ic_series = _compute_rank_ic_series(values, forward_returns)
        ic_clean = ic_series.dropna()
        ic_std = float(ic_clean.std()) if len(ic_clean) > 1 else 0.0
        icir = float(ic_clean.mean() / ic_std) if ic_std > 0 else 0.0
        ic_pos_ratio = float((ic_clean > 0).mean()) if len(ic_clean) > 0 else 0.0

        # Factor-mimicking portfolio Sharpe
        factor_sharpe = _compute_factor_sharpe(values, forward_returns, top_n)

        records.append({
            'factor_name': name,
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'icir': icir,
            'factor_sharpe': factor_sharpe,
            'ic_positive_ratio': ic_pos_ratio,
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

    This mirrors the AutoGPT tournament in baselines/AlphaGrail/AutoGPT/main.py,
    where GPT-4o with Code Interpreter compares alpha performance data.

    Args:
        factor_a_name: Name of factor A
        factor_a_metrics: Dict with mean_ic, icir, factor_sharpe, ic_positive_ratio
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

        # Resolve credentials: explicit params → env vars
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL", None)

        if not resolved_key:
            raise ValueError("No API key provided (param or OPENAI_API_KEY env)")

        client = OpenAI(api_key=resolved_key, base_url=resolved_url) if resolved_url else OpenAI(api_key=resolved_key)

        prompt = f"""You are a quantitative finance expert. Compare two alpha factors and select the better one.

Factor A: {factor_a_name}
  - Mean Rank IC: {factor_a_metrics['mean_ic']:.4f}
  - ICIR (IC Information Ratio): {factor_a_metrics['icir']:.4f}
  - Factor-mimicking portfolio Sharpe: {factor_a_metrics['factor_sharpe']:.4f}
  - IC positive ratio: {factor_a_metrics['ic_positive_ratio']:.2%}

Factor B: {factor_b_name}
  - Mean Rank IC: {factor_b_metrics['mean_ic']:.4f}
  - ICIR (IC Information Ratio): {factor_b_metrics['icir']:.4f}
  - Factor-mimicking portfolio Sharpe: {factor_b_metrics['factor_sharpe']:.4f}
  - IC positive ratio: {factor_b_metrics['ic_positive_ratio']:.2%}

Consider: IC magnitude and stability (ICIR), factor Sharpe, and consistency (IC positive ratio).
Reply with ONLY "A" or "B"."""

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
    Uses a composite score: weighted |ICIR| + |Sharpe| + IC consistency.

    Args:
        metrics_a: Factor A metrics dict
        metrics_b: Factor B metrics dict

    Returns:
        "A" or "B"
    """
    def composite_score(m):
        return (
            0.4 * abs(m.get('icir', 0)) +
            0.3 * abs(m.get('factor_sharpe', 0)) +
            0.2 * abs(m.get('mean_ic', 0)) +
            0.1 * m.get('ic_positive_ratio', 0.5)
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

    Implements a single-elimination tournament bracket:
    - Pairs up all factors
    - Each pair is compared (LLM or quantitative)
    - Winners advance to next round
    - Continues until one winner remains

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
    if use_llm and llm_model:
        print(f"    LLM: {llm_model} @ {llm_base_url or 'default'}")

    # Shuffle for fair bracket (deterministic seed)
    rng = np.random.RandomState(42)
    rng.shuffle(factors)

    current_round = factors
    round_num = 0

    while len(current_round) > 1:
        round_num += 1
        next_round = []

        # Pair up
        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                # Odd number: auto-advance
                next_round.append(current_round[i])
                continue

            factor_a = current_round[i]
            factor_b = current_round[i + 1]

            metrics_a = factor_eval.loc[factor_a].to_dict()
            metrics_b = factor_eval.loc[factor_b].to_dict()

            if use_llm:
                winner = _llm_compare_factors(
                    factor_a, metrics_a, factor_b, metrics_b,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                    model=llm_model,
                )
            else:
                winner = _quantitative_compare(metrics_a, metrics_b)

            winner_name = factor_a if winner == "A" else factor_b
            next_round.append(winner_name)

            if round_num <= 3 or len(current_round) <= 8:
                # Print early rounds and final rounds
                print(f"    Round {round_num}: {factor_a} vs {factor_b} → {winner_name}")

        current_round = next_round

    winner = current_round[0]
    winner_metrics = factor_eval.loc[winner].to_dict()

    print(f"\n  Tournament Winner: {winner}")
    print(f"    Mean IC:  {winner_metrics['mean_ic']:.4f}")
    print(f"    ICIR:     {winner_metrics['icir']:.4f}")
    print(f"    Sharpe:   {winner_metrics['factor_sharpe']:.4f}")

    return winner, winner_metrics


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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    top_n_stocks: int = 50,
    holding_period: int = 1,
    use_llm_tournament: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run AlphaGrail baseline using the main project's DataLoader.

    Pipeline:
      1. Load A-share data via main DataLoader
      2. Build 37 seed alpha factors (from Seed Alpha.xlsx)
      3. Evaluate factors: IC, ICIR, factor Sharpe on training data
      4. Tournament selection (LLM or quantitative)
      5. Build portfolios from winning factor on test data
      6. Backtest with unified BacktestEngine

    Args:
        config_path: Path to main project config file.
        start_date: Data start date (YYYY-MM-DD).
        end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        top_n_stocks: Number of stocks in portfolio.
        holding_period: Rebalance frequency (1=daily, 5=weekly, 20=monthly).
        use_llm_tournament: If True, use LLM for tournament (needs OpenAI API key).
        output_dir: Directory for saving results.

    Returns:
        Dict of performance metrics:
            annual_return, sharpe_ratio, max_drawdown, information_ratio,
            calmar_ratio, win_rate, avg_turnover, mean_rank_ic, icir,
            winning_factor, n_factors
    """
    from dataloader.loader import DataLoader
    from backtest.engine import BacktestEngine

    print("=" * 60)
    print("  AlphaGrail Baseline — LLM-Driven Alpha Selection (via Main DataLoader)")
    print("=" * 60)

    # ── Step 1: Load data via main DataLoader ──────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )

    close = price_data['close']
    n_dates = len(close.index)
    n_stocks = len(close.columns)
    print(f"  Loaded: {n_dates} trading days x {n_stocks} stocks")

    # ── Step 2: Determine train/test split ─────────────────────────────
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    print(f"  Train end: {train_end}, Test start: {test_start}")

    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    # ── Step 3: Build 37 seed alpha factors ────────────────────────────
    print("\n[Step 2] Building 37 seed alpha factors...")
    factor_library = _build_factor_library(price_data)
    print(f"  Built {len(factor_library)} factors: {list(factor_library.keys())[:5]}...")

    # ── Step 4: Compute forward returns ────────────────────────────────
    print("\n[Step 3] Computing forward returns...")
    forward_returns = _compute_forward_returns(close, periods=[1])[1]

    # ── Step 5: Evaluate factors on training data ──────────────────────
    print("\n[Step 4] Evaluating factors on training data...")
    train_mask = forward_returns.index < test_start_ts
    train_returns = forward_returns.loc[train_mask]

    # Also need train period factor values
    train_factors = {
        name: vals.loc[train_mask] for name, vals in factor_library.items()
    }

    factor_eval = evaluate_factors(train_factors, train_returns, top_n=top_n_stocks)
    print(f"\n  Factor evaluation (top 10 by |ICIR|):")
    top_eval = factor_eval.reindex(factor_eval['abs_icir'].nlargest(10).index)
    for name, row in top_eval.iterrows():
        print(f"    {name:35s}  IC={row['mean_ic']:+.4f}  ICIR={row['icir']:+.4f}  Sharpe={row['factor_sharpe']:+.4f}")

    # ── Step 6: Tournament selection ───────────────────────────────────
    print("\n[Step 5] Tournament selection...")

    # Read LLM config from config.yaml (llm.generator section)
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
    # Fall back to env vars
    if not llm_api_key:
        llm_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not llm_base_url:
        llm_base_url = os.environ.get("OPENAI_BASE_URL", "")

    # Auto-enable LLM if we have a key and user requested it
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
        }

    print(f"  Portfolios: {portfolios.shape[0]} days x {portfolios.shape[1]} stocks")

    # ── Step 8: Backtest with unified BacktestEngine ───────────────────
    print("\n[Step 7] Running backtest (unified BacktestEngine)...")

    # Align prices to portfolio dates and columns
    prices_aligned = close.reindex(portfolios.index)
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    metrics = engine.run(portfolios, prices_aligned)

    # ── Step 9: Compute test-period IC for the winning factor ──────────
    test_mask = forward_returns.index >= test_start_ts
    test_returns = forward_returns.loc[test_mask]
    test_factor = factor_library[winner].loc[test_mask]
    test_ic = _compute_rank_ic(test_factor, test_returns)

    # ── Step 10: Compile results ───────────────────────────────────────
    results = {
        'method': 'AlphaGrail',
        'winning_factor': winner,
        'n_factors': len(factor_library),
        'mean_rank_ic_train': float(winner_metrics.get('mean_ic', 0)),
        'icir': float(winner_metrics.get('icir', 0)),
        'factor_sharpe_train': float(winner_metrics.get('factor_sharpe', 0)),
        'mean_rank_ic_test': float(test_ic),
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
        'use_llm_tournament': use_llm_tournament,
        'llm_model': llm_model if use_llm_tournament else None,
        'tournament_winner_metrics': {
            'mean_ic': float(winner_metrics.get('mean_ic', 0)),
            'icir': float(winner_metrics.get('icir', 0)),
            'factor_sharpe': float(winner_metrics.get('factor_sharpe', 0)),
            'ic_positive_ratio': float(winner_metrics.get('ic_positive_ratio', 0)),
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("  AlphaGrail Baseline Complete")
    print("=" * 60)
    print(f"  Winning Factor:   {winner}")
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
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics JSON
        result_path = os.path.join(output_dir, 'alphagrail_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

        # Save factor evaluation
        eval_path = os.path.join(output_dir, 'factor_evaluation.csv')
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
    parser.add_argument('--start', default=None,
                        help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None,
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
    parser.add_argument('--output-dir', default='experiments/alphagrail',
                        help='Output directory')

    args = parser.parse_args()

    results = run_alphagrail_baseline(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        top_n_stocks=args.top_n,
        holding_period=args.holding_period,
        use_llm_tournament=args.use_llm,
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
    print(f"  Mean Rank-IC:     {results['mean_rank_ic_train']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
