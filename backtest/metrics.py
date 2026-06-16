# -*- coding: utf-8 -*-
"""
Common financial metrics for factor evaluation and backtesting.

This module provides pure functions for computing financial metrics
shared between FactorBacktester (factor evaluation) and BacktestEngine
(portfolio backtesting).

All functions are pure: no side effects, no mutable state.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a return series.

    Returns a negative number (e.g., -0.15 = 15% drawdown).

    Args:
        returns: Series of periodic returns (typically daily).

    Returns:
        Minimum drawdown as a negative float; 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series, rf: float = 0.02) -> float:
    """
    Compute annualized Sharpe ratio from periodic returns.

    Uses the standard definition:
        Sharpe = (mean(excess) / std(excess)) * sqrt(252)

    Args:
        returns: Series of periodic returns (assumed daily).
        rf: Annual risk-free rate (default 2%).

    Returns:
        Annualized Sharpe ratio; 0.0 if std is zero or data insufficient.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / 252.0
    mean_ret = float(excess.mean())
    std_ret = float(excess.std(ddof=1))
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(252)


def total_return(portfolio_values: pd.Series) -> float:
    """
    Compute total return from a portfolio value series.

    Args:
        portfolio_values: Series of cumulative portfolio values.

    Returns:
        Total return as a float (e.g., 0.25 = 25% total return).
    """
    if len(portfolio_values) < 2:
        return 0.0
    return float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)


def annualized_return(total_return: float, n_years: float) -> float:
    """
    Compute annualized return from total return and time span.

    Args:
        total_return: Total return (e.g., 0.25 = 25%).
        n_years: Number of years (trading years = n_days / 252).

    Returns:
        Annualized return as a float.
    """
    if n_years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """
    Compute annualized volatility from daily returns.

    Args:
        daily_returns: Series of daily returns.

    Returns:
        Annualized volatility (daily std * sqrt(252)).
    """
    if len(daily_returns) < 2:
        return 0.0
    daily_vol = float(daily_returns.std())
    return daily_vol * np.sqrt(252)


def calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    Compute Calmar ratio: annual return / |max drawdown|.

    Args:
        annual_return: Annualized return.
        max_drawdown: Maximum drawdown (negative number).

    Returns:
        Calmar ratio; 0.0 if max_drawdown is zero.
    """
    if max_drawdown == 0:
        return 0.0
    return annual_return / abs(max_drawdown)


def win_rate(returns: pd.Series) -> float:
    """
    Compute win rate: fraction of positive returns.

    Args:
        returns: Series of returns.

    Returns:
        Fraction of positive returns; 0.5 if empty.
    """
    if len(returns) == 0:
        return 0.5
    return float((returns > 0).sum() / len(returns))


def information_ratio(excess_returns: pd.Series) -> float:
    """
    Compute annualized Information Ratio.

    Args:
        excess_returns: Series of excess returns vs. benchmark (or zero).

    Returns:
        Annualized Information Ratio.
    """
    if len(excess_returns) < 2:
        return 0.0
    mean_excess = float(excess_returns.mean())
    std_excess = float(excess_returns.std(ddof=1))
    if std_excess == 0:
        return 0.0
    return (mean_excess / std_excess) * np.sqrt(252)


def avg_turnover(turnovers: List[float]) -> float:
    """
    Compute average turnover from a list of turnover values.

    Args:
        turnovers: List of turnover values (one per rebalance).

    Returns:
        Mean turnover; 0.0 if list is empty.
    """
    if not turnovers:
        return 0.0
    return float(np.mean(turnovers))


def rank_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Compute Rank IC and ICIR (cross-sectional Spearman correlation).

    Vectorized implementation: computes ranks for all time periods at once,
    then computes Pearson correlation of ranks per period.

    Args:
        factor_values: DataFrame of factor values (dates × stocks).
        forward_returns: DataFrame of forward returns (dates × stocks).

    Returns:
        (mean_ic, ic_ir) where ic_ir = mean_ic / std_ic.
    """
    # Cross-sectional ranks for each time t
    fv_ranks = factor_values.rank(axis=1, numeric_only=True, pct=False)
    fr_ranks = forward_returns.rank(axis=1, numeric_only=True, pct=False)

    # Pearson correlation of ranks per row (time t)
    fv_mean = fv_ranks.mean(axis=1)
    fr_mean = fr_ranks.mean(axis=1)
    fv_centered = fv_ranks.subtract(fv_mean, axis=0)
    fr_centered = fr_ranks.subtract(fr_mean, axis=0)

    numerator = (fv_centered * fr_centered).sum(axis=1)
    denom = (
        np.sqrt((fv_centered ** 2).sum(axis=1))
        * np.sqrt((fr_centered ** 2).sum(axis=1))
    )
    ic_series = numerator / denom

    # Handle edge cases (constant rank → denominator = 0)
    ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()

    if ic_series.empty:
        return 0.0, 0.0

    ic_arr = ic_series.values
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0

    return mean_ic, ic_ir
