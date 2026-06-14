# -*- coding: utf-8 -*-
"""
Backtest Engine Module

This module implements a backtest engine for evaluating factor-based
stock selection strategies.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """
    Backtest engine for factor-based strategies.
    
    Simulates portfolio rebalancing, calculates returns,
    and tracks performance metrics.
    """
    
    def __init__(
        self,
        commission: float = 0.0003,
        slippage: float = 0.001,
        benchmark: str = 'hs300',
    ):
        """
        Initialize backtest engine.
        
        Args:
            commission: Trading commission rate (default: 0.03%)
            slippage: Slippage rate (default: 0.1%)
            benchmark: Benchmark index name
        """
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark
        
        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.positions = []
        self.turnovers = []
        
    def run(
        self,
        portfolios: pd.DataFrame,
        prices: pd.DataFrame,
        market_cap: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            portfolios: DataFrame of portfolio weights (n_dates x n_stocks)
            prices: DataFrame of stock prices (n_dates x n_stocks)
            market_cap: DataFrame of market capitalization (optional)
            
        Returns:
            Dict with performance metrics
        """
        print("Running backtest...")
        
        # Align dates
        common_dates = portfolios.index.intersection(prices.index)
        if len(common_dates) == 0:
            # 尝试转换 portfolios.index 为 DatetimeIndex 再试一次
            try:
                portfolios.index = pd.to_datetime(portfolios.index)
                common_dates = portfolios.index.intersection(prices.index)
            except Exception:
                pass
        portfolios = portfolios.loc[common_dates]
        prices = prices.loc[common_dates]
        
        n_dates = len(common_dates)
        portfolio_value = 1.0  # Start with $1
        portfolio_values = [portfolio_value]
        daily_returns = []
        positions_list = []
        turnovers = []
        
        prev_weights = None
        
        for i, date in enumerate(common_dates):
            # Current portfolio weights
            weights = portfolios.loc[date]
            weights = weights / weights.sum()  # Normalize to sum = 1
            
            # Calculate turnover (if not first period)
            if prev_weights is not None:
                turnover = np.sum(np.abs(weights - prev_weights)) / 2
                turnovers.append(turnover)
                
                # Apply transaction costs
                transaction_cost = turnover * (self.commission + self.slippage)
                portfolio_value *= (1 - transaction_cost)
            
            # Calculate daily return
            if i < n_dates - 1:
                next_date = common_dates[i + 1]
                daily_returns_pct = prices.loc[next_date] / prices.loc[date] - 1
                portfolio_return = np.sum(weights * daily_returns_pct)
                portfolio_value *= (1 + portfolio_return)
                
                daily_returns.append(portfolio_return)
                portfolio_values.append(portfolio_value)
                positions_list.append(weights)
            
            prev_weights = weights.copy()
        
        # Convert to Series
        # portfolio_values has n_dates entries (initial 1.0 + n_dates-1 daily updates)
        # daily_returns has n_dates-1 entries (one per transition)
        # We exclude the last date from portfolio_values since no forward return is available
        n_return_dates = n_dates - 1
        if n_return_dates <= 0:
            # No return dates available (empty portfolios or single-date data)
            return {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            }
        portfolio_values_series = pd.Series(portfolio_values[:n_return_dates], index=common_dates[:n_return_dates])
        daily_returns_series = pd.Series(daily_returns, index=common_dates[:n_return_dates])
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(
            portfolio_values_series,
            daily_returns_series,
            turnovers,
        )
        
        # Store results
        self.portfolio_values = portfolio_values_series
        self.returns = daily_returns_series
        self.positions = positions_list
        self.turnovers = turnovers
        
        return metrics
    
    def run_qlib(
        self,
        factor_values: pd.DataFrame,
        start_time: str,
        end_time: str,
        provider_uri: Optional[str] = None,
        topk: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        """
        Run backtest using Qlib's professional framework.

        This delegates to QlibBacktester for a full portfolio simulation
        with realistic signal delay, trading costs, and risk controls.

        Args:
            factor_values: DataFrame of factor values (dates × Qlib-format stock codes)
            start_time: Backtest start date (YYYY-MM-DD)
            end_time: Backtest end date (YYYY-MM-DD)
            provider_uri: Qlib data path (default from config or ~/.qlib/qlib_data/cn_data)
            topk: Number of stocks in portfolio (default 50)
            **kwargs: Passed to QlibBacktester constructor

        Returns:
            Dict of performance metrics compatible with run() output
        """
        from .qlib_backtester import QlibBacktester

        bt_kwargs = {
            'commission': self.commission,
            'slippage': self.slippage,
        }
        if provider_uri:
            bt_kwargs['provider_uri'] = provider_uri
        if topk:
            bt_kwargs['topk'] = topk
        bt_kwargs.update(kwargs)

        qlib_bt = QlibBacktester(**bt_kwargs)
        result = qlib_bt.run(factor_values, start_time, end_time)

        # Store results in engine state for compatibility
        self.qlib_metrics = result

        # Map to our standard metric names where possible
        mapped = {
            'total_return': result.get('total_return', 0),
            'annual_return': result.get('annual_return', 0),
            'annual_volatility': result.get('annual_volatility', 0),
            'sharpe_ratio': result.get('sharpe', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'calmar_ratio': result.get('calmar_ratio', 0),
            'information_ratio': result.get('information_ratio', 0),
            'win_rate': result.get('win_rate', 0.5),
            'avg_turnover': result.get('turnover', 0),
            'n_trading_days': 0,  # Qlib handles this internally
        }
        return mapped

    def _calculate_metrics(
        self,
        portfolio_values: pd.Series,
        daily_returns: pd.Series,
        turnovers: List[float],
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: Series of portfolio values
            daily_returns: Series of daily returns
            turnovers: List of turnover values
            
        Returns:
            Dict of performance metrics
        """
        # Basic metrics
        total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
        n_days = len(daily_returns)
        n_years = n_days / 252
        
        # Annualized return
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Annualized volatility
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        # Average turnover
        avg_turnover = np.mean(turnovers) if turnovers else 0
        
        # Information ratio (vs. benchmark)
        # For simplicity, assume benchmark return = 0
        excess_returns = daily_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'avg_turnover': avg_turnover,
            'n_trading_days': n_days,
        }
    
    def get_portfolio_values(self) -> pd.Series:
        """Get portfolio value series."""
        return self.portfolio_values
    
    def get_returns(self) -> pd.Series:
        """Get daily returns series."""
        return self.returns
    
    def get_positions(self) -> List[pd.Series]:
        """Get position history."""
        return self.positions
    
    def plot_performance(self, save_path: Optional[str] = None):
        """
        Plot portfolio performance.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Portfolio value
            axes[0].plot(self.portfolio_values.index, self.portfolio_values.values)
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Portfolio Value')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative returns
            cumulative_returns = (1 + self.returns).cumprod() - 1
            axes[1].plot(cumulative_returns.index, cumulative_returns.values)
            axes[1].set_title('Cumulative Returns')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Cumulative Return')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Cannot plot.")


class LightweightBacktester:
    """
    Lightweight backtester for factor evaluation.
    
    Used within the evolution module for quick factor evaluation.
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize backtester.
        
        Args:
            prices: DataFrame of stock prices (n_dates x n_stocks)
        """
        self.prices = prices
        
    def evaluate_factor(
        self,
        factor_values: pd.Series,
        top_n: int = 50,
        holding_period: int = 5,
    ) -> Dict:
        """
        Evaluate a single factor.
        
        Args:
            factor_values: Series of factor values (index = stock codes)
            top_n: Number of stocks to select
            holding_period: Holding period in days
            
        Returns:
            Dict of evaluation metrics
        """
        # Select top-N stocks
        top_stocks = factor_values.nlargest(top_n).index.tolist()
        
        # Calculate returns over holding period
        returns = []
        for stock in top_stocks:
            if stock in self.prices.columns:
                stock_prices = self.prices[stock]
                if len(stock_prices) > holding_period:
                    ret = stock_prices.iloc[holding_period] / stock_prices.iloc[0] - 1
                    returns.append(ret)
        
        if not returns:
            return {'ic': 0, 'sharpe': 0, 'win_rate': 0, 'max_drawdown': 0}
        
        returns = np.array(returns)
        
        # Calculate metrics
        ic = np.mean(returns)  # Simplified IC (correlation with future returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 / holding_period)
        win_rate = (returns > 0).sum() / len(returns)
        
        # Max drawdown (simplified)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'ic': ic,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
        }
    
    def evaluate_factors(
        self,
        factor_dict: Dict[str, pd.Series],
        top_n: int = 50,
        holding_period: int = 5,
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple factors.
        
        Args:
            factor_dict: Dict of factor values (factor_name -> factor_values)
            top_n: Number of stocks to select
            holding_period: Holding period in days
            
        Returns:
            Dict of evaluation results
        """
        results = {}
        for factor_name, factor_values in factor_dict.items():
            results[factor_name] = self.evaluate_factor(
                factor_values, top_n, holding_period
            )
        
        return results


if __name__ == '__main__':
    # Demo
    print("=== Backtest Engine Demo ===\n")
    
    # Generate mock data
    np.random.seed(42)
    n_dates = 100
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    # Prices
    prices = pd.DataFrame(
        10 + np.cumsum(np.random.randn(n_dates, n_stocks) * 0.02, axis=0),
        index=dates,
        columns=stock_codes,
    )
    
    # Portfolios (random weights)
    portfolios = pd.DataFrame(
        np.random.dirichlet(np.ones(n_stocks), size=n_dates),
        index=dates,
        columns=stock_codes,
    )
    
    # Run backtest
    engine = BacktestEngine()
    metrics = engine.run(portfolios, prices)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== Demo Complete ===")
