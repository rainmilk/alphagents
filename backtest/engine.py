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

from . import metrics
import warnings
warnings.filterwarnings('ignore')
import os


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
        risk_free_rate: float = 0.0,
        holding_period: int = 1,
    ):
        """
        Initialize backtest engine.
        
        Args:
            commission: Trading commission rate (default: 0.03%)
            slippage: Slippage rate (default: 0.1%)
            benchmark: Benchmark index name
            risk_free_rate: Annualized risk-free rate for Sharpe ratio (default: 0.0)
            holding_period: Number of trading days to hold positions before rebalancing.
                           1 = daily rebalance (T+1), 5 = weekly, 20 = monthly.
                           On non-rebalance days, positions drift with price changes.
        """
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        # Coerce None -> 1 so a standalone baseline that never went through
        # main.py (where global_config.holding_period gets populated) cannot
        # later crash engine.run() with `max(1, None)`.
        self.holding_period = holding_period if holding_period is not None else 1
        
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
        holding_period: Optional[int] = None,
        save_dir: Optional[str] = None,
        method_prefix: Optional[str] = None,
    ) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            portfolios: DataFrame of portfolio weights (n_dates x n_stocks)
            prices: DataFrame of stock prices (n_dates x n_stocks)
            market_cap: DataFrame of market capitalization (optional)
            holding_period: Number of days between rebalances.
                            None → use self.holding_period (default: 1 = daily).
                            5 = weekly, 20 = monthly.
                            On non-rebalance days, positions drift with price changes;
                            no transaction costs are incurred.
            
        Returns:
            Dict with performance metrics
        """
        # Resolve holding_period
        hp = holding_period if holding_period is not None else self.holding_period
        if hp is None:
            hp = 1  # fallback: daily rebalance
        hp = max(1, hp)  # Minimum 1 day
        
        print(f"Running backtest (holding_period={hp}d)...")
        
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
        n_rebalances = 0
        
        for i, date in enumerate(common_dates):
            # --- Determine if this is a rebalance date ---
            is_rebalance = (i % hp == 0)
            
            if is_rebalance:
                # Use the pre-computed portfolio weights
                target_weights = portfolios.loc[date]
                target_weights = target_weights / target_weights.sum()
                
                # Calculate turnover vs previous holdings
                if prev_weights is not None:
                    turnover = np.sum(np.abs(target_weights - prev_weights)) / 2
                    turnovers.append(turnover)
                    transaction_cost = turnover * (self.commission + self.slippage)
                    portfolio_value *= (1 - transaction_cost)
                
                current_weights = target_weights
                n_rebalances += 1
            else:
                # Hold: drift previous weights by today's price change
                if prev_weights is not None and i > 0:
                    prev_date = common_dates[i - 1]
                    price_ratio = prices.loc[date] / prices.loc[prev_date]
                    current_weights = prev_weights * price_ratio
                    current_weights = current_weights.fillna(0.0)
                    w_sum = current_weights.sum()
                    current_weights = current_weights / w_sum if w_sum > 0 else current_weights
                else:
                    # First date fallback (should not reach here with hp >= 1)
                    current_weights = portfolios.loc[date]
                    current_weights = current_weights / current_weights.sum()
            
            # Calculate next-day return (always computed, regardless of rebalance)
            if i < n_dates - 1:
                next_date = common_dates[i + 1]
                daily_returns_pct = prices.loc[next_date] / prices.loc[date] - 1
                portfolio_return = np.sum(current_weights * daily_returns_pct)
                portfolio_value *= (1 + portfolio_return)
                
                daily_returns.append(portfolio_return)
                portfolio_values.append(portfolio_value)
                positions_list.append(current_weights)
            
            prev_weights = current_weights.copy()
        
        print(f"  Rebalanced {n_rebalances} times over {n_dates} trading days")
        
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

        # ── Optional: persist daily returns & portfolio values ──
        # When save_dir is provided, dump the strategy daily-return series and the
        # equity curve to CSV. This gives every baseline a uniform, comparable
        # daily_returns.csv (referenced by our own MASE model's date-isolated layout).
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # Prefix filenames with the method name so multiple baselines can
            # share one parameter-isolated directory without clobbering each other.
            dr_name = f"{method_prefix}_daily_returns.csv" if method_prefix else "daily_returns.csv"
            pv_name = f"{method_prefix}_portfolio_values.csv" if method_prefix else "portfolio_values.csv"
            if isinstance(self.returns, pd.Series) and len(self.returns) > 0:
                self.returns.to_csv(
                    os.path.join(save_dir, dr_name),
                    header=["daily_return"],
                )
            if isinstance(self.portfolio_values, pd.Series) and len(self.portfolio_values) > 0:
                self.portfolio_values.to_csv(
                    os.path.join(save_dir, pv_name),
                    header=["portfolio_value"],
                )

        return metrics
    
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
        # Compute metrics using shared functions
        n_days = len(daily_returns)
        n_years = n_days / 252
        
        total_ret = metrics.total_return(portfolio_values)
        annual_ret = metrics.annualized_return(total_ret, n_years)
        annual_vol = metrics.annualized_volatility(daily_returns)
        sharpe_ratio = metrics.annualized_sharpe(daily_returns, rf=self.risk_free_rate)
        max_dd = metrics.max_drawdown(daily_returns)
        calmar = metrics.calmar_ratio(annual_ret, max_dd)
        wr = metrics.win_rate(daily_returns)
        avg_turn = metrics.avg_turnover(turnovers)
        # For simplicity, assume benchmark return = 0
        ir = metrics.information_ratio(daily_returns)
        
        return {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': wr,
            'information_ratio': ir,
            'avg_turnover': avg_turn,
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
    
    def get_turnovers(self) -> List[float]:
        """Get turnover history."""
        return self.turnovers
    
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
