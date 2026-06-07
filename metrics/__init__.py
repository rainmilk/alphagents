"""
Metrics Package

This package contains evaluation metrics and analysis tools.
"""

from .evaluator import (
    calculate_ic,
    calculate_icir,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_win_rate,
    evaluate_portfolio_comprehensive,
    FactorEvaluator,
)

__all__ = [
    'calculate_ic',
    'calculate_icir',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_win_rate',
    'evaluate_portfolio_comprehensive',
    'FactorEvaluator',
]
