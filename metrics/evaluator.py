# -*- coding: utf-8 -*-
"""
Metrics Evaluation Module

This module provides comprehensive evaluation metrics for factor-based
stock selection strategies.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_ic(
    factor_values: pd.Series,
    future_returns: pd.Series,
) -> float:
    """
    Calculate Information Coefficient (IC).
    
    IC measures the correlation between factor values and future returns.
    
    Args:
        factor_values: Series of factor values
        future_returns: Series of future returns
        
    Returns:
        IC value (Pearson correlation)
    """
    # Align indices
    common_idx = factor_values.index.intersection(future_returns.index)
    if len(common_idx) == 0:
        return 0.0
    
    factor_aligned = factor_values.loc[common_idx]
    returns_aligned = future_returns.loc[common_idx]
    
    # Calculate Pearson correlation
    correlation = factor_aligned.corr(returns_aligned)
    
    return correlation if not np.isnan(correlation) else 0.0


def calculate_icir(
    factor_values: pd.Series,
    future_returns: pd.Series,
    window: int = 20,
) -> float:
    """
    Calculate Information Coefficient Information Ratio (ICIR).
    
    ICIR = mean(IC) / std(IC) over a rolling window.
    
    Args:
        factor_values: Series of factor values
        future_returns: Series of future returns
        window: Rolling window size
        
    Returns:
        ICIR value
    """
    # Calculate IC for each time point
    dates = factor_values.index
    ics = []
    
    for i in range(window, len(dates)):
        window_factor = factor_values.loc[dates[i-window:i]]
        window_returns = future_returns.loc[dates[i-window:i]]
        
        ic = calculate_ic(window_factor, window_returns)
        ics.append(ic)
    
    if len(ics) == 0:
        return 0.0
    
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    
    return mean_ic / (std_ic + 1e-8)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annual)
        annualize: Whether to annualize the result
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    
    if annualize:
        sharpe *= np.sqrt(252)
    
    return sharpe


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio.
    
    Calmar ratio = Annualized return / |Maximum drawdown|
    
    Args:
        returns: Series of returns
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Annualized return
    n_days = len(returns)
    n_years = n_days / 252
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Max drawdown
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / abs(max_dd)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate Information ratio.
    
    IR = mean(excess_returns) / std(excess_returns)
    
    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        
    Returns:
        Information ratio
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Align indices
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        return 0.0
    
    excess_returns = returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess * np.sqrt(252)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate.
    
    Args:
        returns: Series of returns
        
    Returns:
        Win rate (proportion of positive returns)
    """
    if len(returns) == 0:
        return 0.0
    
    return (returns > 0).sum() / len(returns)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Calculate Sortino ratio.
    
    Sortino ratio uses downside deviation instead of total deviation.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annual)
        annualize: Whether to annualize the result
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    mean_excess = excess_returns.mean()
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = mean_excess / downside_deviation
    
    if annualize:
        sortino *= np.sqrt(252)
    
    return sortino


def calculate_beta(
    returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Calculate beta (market exposure).
    
    Args:
        returns: Series of strategy returns
        market_returns: Series of market returns
        
    Returns:
        Beta value
    """
    if len(returns) == 0 or len(market_returns) == 0:
        return 0.0
    
    # Align indices
    common_idx = returns.index.intersection(market_returns.index)
    if len(common_idx) < 2:
        return 0.0
    
    x = market_returns.loc[common_idx]
    y = returns.loc[common_idx]
    
    # Beta = Cov(x, y) / Var(x)
    covariance = np.cov(x, y)[0, 1]
    variance = np.var(x)
    
    return covariance / (variance + 1e-8)


def calculate_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate alpha (Jensen's alpha).
    
    Args:
        returns: Series of strategy returns
        market_returns: Series of market returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Alpha value (annualized)
    """
    if len(returns) == 0 or len(market_returns) == 0:
        return 0.0
    
    # Calculate beta
    beta = calculate_beta(returns, market_returns)
    
    # Calculate alpha
    strategy_return = returns.mean() * 252
    market_return = market_returns.mean() * 252
    
    alpha = strategy_return - risk_free_rate - beta * (market_return - risk_free_rate)
    
    return alpha


def evaluate_factor_comprehensive(
    factor_values: pd.Series,
    future_returns: pd.Series,
    prices: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Comprehensive factor evaluation.
    
    Args:
        factor_values: Series of factor values
        future_returns: Series of future returns
        prices: DataFrame of prices (optional, for additional metrics)
        
    Returns:
        Dict of evaluation metrics
    """
    metrics = {}
    
    # IC
    metrics['ic'] = calculate_ic(factor_values, future_returns)
    
    # ICIR
    metrics['icir'] = calculate_icir(factor_values, future_returns)
    
    # Sharpe ratio (if prices provided)
    if prices is not None:
        # Simulate portfolio based on factor
        top_stocks = factor_values.nlargest(50).index
        if len(top_stocks) > 0:
            portfolio_returns = prices[top_stocks].pct_change().mean(axis=1)
            metrics['sharpe'] = calculate_sharpe_ratio(portfolio_returns)
            metrics['max_drawdown'] = calculate_max_drawdown(portfolio_returns)
            metrics['win_rate'] = calculate_win_rate(portfolio_returns)
    
    return metrics


def evaluate_portfolio_comprehensive(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    market_returns: Optional[pd.Series] = None,
) -> Dict:
    """
    Comprehensive portfolio evaluation.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns (optional)
        market_returns: Series of market returns (optional)
        
    Returns:
        Dict of evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['annual_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns)
    metrics['win_rate'] = calculate_win_rate(returns)
    
    # Risk-adjusted metrics
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
    
    if market_returns is not None:
        metrics['beta'] = calculate_beta(returns, market_returns)
        metrics['alpha'] = calculate_alpha(returns, market_returns)
    
    return metrics


class FactorEvaluator:
    """
    Comprehensive factor evaluator.
    
    Provides multiple evaluation metrics for factors.
    """
    
    def __init__(self):
        """Initialize factor evaluator."""
        pass
    
    def evaluate(
        self,
        factor_values: pd.DataFrame,
        future_returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Evaluate multiple factors.
        
        Args:
            factor_values: DataFrame of factor values (n_stocks x n_factors)
            future_returns: Series of future returns
            prices: DataFrame of prices (optional)
            
        Returns:
            DataFrame of evaluation results
        """
        results = []
        
        for factor_name in factor_values.columns:
            factor_series = factor_values[factor_name]
            metrics = evaluate_factor_comprehensive(factor_series, future_returns, prices)
            metrics['factor_name'] = factor_name
            results.append(metrics)
        
        return pd.DataFrame(results)


if __name__ == '__main__':
    # Demo
    print("=== Metrics Evaluation Demo ===\n")
    
    # Generate mock data
    np.random.seed(42)
    n_stocks = 100
    n_days = 100
    
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # Factor values
    factor_values = pd.Series(
        np.random.randn(n_stocks),
        index=stock_codes,
    )
    
    # Future returns
    future_returns = pd.Series(
        0.001 * factor_values + np.random.randn(n_stocks) * 0.01,
        index=stock_codes,
    )
    
    # Calculate IC
    ic = calculate_ic(factor_values, future_returns)
    print(f"IC: {ic:.4f}")
    
    # Calculate Sharpe ratio
    returns = pd.Series(np.random.randn(n_days) * 0.01 + 0.0005, index=dates)
    sharpe = calculate_sharpe_ratio(returns)
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(returns)
    print(f"Max drawdown: {max_dd:.4f}")
    
    # Comprehensive evaluation
    metrics = evaluate_portfolio_comprehensive(returns)
    print(f"\nComprehensive metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n=== Demo Complete ===")
