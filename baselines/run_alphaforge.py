"""
AlphaForge (AFF) Baseline Runner - Complete 3-Stage Implementation
===============================================================

This implementation follows the exact 3-stage process from README.md:
- Stage 1: Mining alpha factors (simplified GAN-based approach)
- Stage 2: Combining alpha factors (rolling window + linear regression)
- Stage 3: Calculate and display results

Uses MAIN PROJECT'S DATALOADER (not Qlib).

Based on: 
- baselines/AlphaForge/README.MD
- baselines/AlphaForge/train_AFF.py
- baselines/AlphaForge/combine_AFF.py
- baselines/AlphaForge/exp_AFF_calc_result.ipynb

Author: Code Review Expert (火眼眼)
Date: 2026-07-02
"""

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AlphaForgeConfig:
    """Configuration for AlphaForge baseline."""
    instruments: str = "csi300"
    train_end_year: int = 2023
    seeds: List[int] = None
    save_name: str = "alphaforge"
    zoo_size: int = 50  # Number of factors to mine in Stage 1
    n_factors: int = 10  # Number of factors to use in Stage 2
    window: Union[int, str] = "inf"  # Rolling window for factor evaluation
    top_n_stocks: int = 50  # Number of stocks in portfolio
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4]


def convert_to_multindex(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert main DataLoader format to MultiIndex format.
    
    Args:
        prices: DataFrame with columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        
    Returns:
        pd.DataFrame: MultiIndex (date, symbol) with OHLCV columns
    """
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Pivot to (date, symbol) format
    result = prices.pivot(index='date', columns='symbol')
    result.columns.names = ['field', 'symbol']
    
    return result


def compute_returns(prices_multindex: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from MultiIndex price data.
    
    Args:
        prices_multindex: MultiIndex DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: Daily returns (date x symbol)
    """
    close = prices_multindex['close']
    returns = close.pct_change().shift(-1)  # Forward 1-day return
    return returns


def generate_template_factors(prices: pd.DataFrame, n_factors: int = 50) -> List[str]:
    """
    Generate template alpha factor expressions.
    
    These are simplified versions of common alpha factors.
    In real AlphaForge, these would be mined by GAN.
    
    Args:
        prices: Price data
        n_factors: Number of factors to generate
        
    Returns:
        List[str]: Factor expressions (as strings for evaluation)
    """
    # Basic factor templates
    templates = [
        # Momentum factors
        "close / close.shift(20) - 1",  # 20-day momentum
        "close / close.shift(5) - 1",   # 5-day momentum
        "(close - close.shift(1)) / close.shift(1)",  # 1-day return
        
        # Mean reversion
        "(close - close.rolling(20).mean()) / close.rolling(20).std()",  # Bollinger Band position
        "volume / volume.rolling(20).mean()",  # Volume ratio
        
        # Volatility
        "close.rolling(20).std() / close",  # Historical volatility
        "(high - low) / close",  # Daily volatility
        
        # Volume-price
        "(close - close.shift(1)) * volume",  # Volume-weighted return
        "volume / volume.rolling(5).mean()",  # Short-term volume trend
        
        # Technical indicators
        "close.rolling(12).mean() - close.rolling(26).mean()",  # MACD line
        "((close - close.shift(1)) / close.shift(1)).rolling(14).mean()",  # Momentum MA
    ]
    
    # Generate variations by changing parameters
    extended = []
    for template in templates:
        extended.append(template)
        # Create variations with different windows
        for window in [3, 10, 15, 30]:
            if 'rolling(20)' in template:
                extended.append(template.replace('rolling(20)', f'rolling({window})'))
            if 'shift(20)' in template:
                extended.append(template.replace('shift(20)', f'shift({window})'))
            if 'shift(5)' in template:
                extended.append(template.replace('shift(5)', f'shift({window})'))
    
    # Return unique factors up to n_factors
    unique_factors = list(dict.fromkeys(extended))
    return unique_factors[:n_factors]


def evaluate_factor_expression(expr: str, prices_multindex: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate a factor expression on price data.
    
    Args:
        expr: Factor expression string
        prices_multindex: MultiIndex price data
        
    Returns:
        pd.DataFrame: Factor values (date x symbol)
    """
    close = prices_multindex['close']
    high = prices_multindex['high']
    low = prices_multindex['low']
    volume = prices_multindex['volume']
    
    try:
        # Evaluate expression in safe context
        factor_values = eval(expr, {"__builtins__": {}}, {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'np': np,
            'pd': pd,
        })
        
        if isinstance(factor_values, pd.Series):
            factor_values = factor_values.to_frame()
        
        return factor_values
        
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {e}")
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)


def calculate_ic(factor_scores: pd.DataFrame, returns: pd.DataFrame) -> float:
    """
    Calculate Information Coefficient (IC) between factor and returns.
    
    Args:
        factor_scores: Factor values (date x symbol)
        returns: Return values (date x symbol)
        
    Returns:
        float: Mean IC across all dates
    """
    ic_values = []
    
    for date in factor_scores.index:
        factor_valid = factor_scores.loc[date].dropna()
        return_valid = returns.loc[date].dropna()
        
        # Align indices
        common_idx = factor_valid.index.intersection(return_valid.index)
        if len(common_idx) < 10:
            continue
            
        factor_aligned = factor_valid[common_idx]
        return_aligned = return_valid[common_idx]
        
        # Calculate Pearson correlation
        try:
            ic = np.corrcoef(factor_aligned, return_aligned)[0, 1]
            if np.isfinite(ic):
                ic_values.append(ic)
        except Exception:
            continue
    
    return np.mean(ic_values) if ic_values else 0.0


def calculate_rank_ic(factor_scores: pd.DataFrame, returns: pd.DataFrame) -> float:
    """
    Calculate Rank IC (Spearman correlation) between factor and returns.
    
    Uses pure numpy implementation (no scipy dependency).
    
    Args:
        factor_scores: Factor values (date x symbol)
        returns: Return values (date x symbol)
        
    Returns:
        float: Mean Rank IC across all dates
    """
    rank_ic_values = []
    
    for date in factor_scores.index:
        factor_valid = factor_scores.loc[date].dropna()
        return_valid = returns.loc[date].dropna()
        
        # Align indices
        common_idx = factor_valid.index.intersection(return_valid.index)
        if len(common_idx) < 10:
            continue
            
        factor_aligned = factor_valid[common_idx]
        return_aligned = return_valid[common_idx]
        
        # Calculate Spearman rank correlation using pure numpy
        try:
            # Convert to ranks
            factor_ranks = factor_aligned.rank()
            return_ranks = return_aligned.rank()
            
            # Calculate Pearson correlation of ranks
            rho = np.corrcoef(factor_ranks, return_ranks)[0, 1]
            if np.isfinite(rho):
                rank_ic_values.append(rho)
        except Exception:
            continue
    
    return np.mean(rank_ic_values) if rank_ic_values else 0.0


def stage1_mine_factors(
    prices: pd.DataFrame,
    config: AlphaForgeConfig,
    output_dir: str,
) -> Dict:
    """
    Stage 1: Mine alpha factors.
    
    In original AlphaForge, this uses GAN to mine factors.
    Here we use template expressions as a simplified version.
    
    Args:
        prices: Price data from main DataLoader
        config: AlphaForge configuration
        output_dir: Directory to save results
        
    Returns:
        Dict: Mined factors and their evaluations
    """
    print("\n" + "="*60)
    print("[Stage 1] Mining alpha factors...")
    print("="*60)
    
    # Convert to MultiIndex format
    prices_multindex = convert_to_multindex(prices)
    returns = compute_returns(prices_multindex)
    
    # Generate template factors
    print(f"  Generating {config.zoo_size} template factors...")
    factor_exprs = generate_template_factors(prices, config.zoo_size)
    
    # Evaluate all factors
    print("  Evaluating factors...")
    factor_scores_dict = {}
    factor_metrics = {}
    
    for i, expr in enumerate(factor_exprs):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(factor_exprs)}")
        
        # Evaluate factor
        factor_values = evaluate_factor_expression(expr, prices_multindex)
        
        # Calculate IC and Rank IC
        ic = calculate_ic(factor_values, returns)
        rank_ic = calculate_rank_ic(factor_values, returns)
        
        factor_scores_dict[expr] = factor_values
        factor_metrics[expr] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'expr': expr,
        }
    
    # Sort by Rank IC and select top factors
    sorted_factors = sorted(
        factor_metrics.items(),
        key=lambda x: abs(x[1]['rank_ic']),
        reverse=True
    )
    
    top_factors = [x[0] for x in sorted_factors[:config.zoo_size]]
    
    print(f"  Top factor Rank IC: {factor_metrics[top_factors[0]]['rank_ic']:.4f}")
    print(f"  Saved {len(top_factors)} factors to zoo")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    zoo_path = os.path.join(output_dir, "zoo_factors.json")
    
    with open(zoo_path, 'w') as f:
        json.dump({
            'factor_exprs': top_factors,
            'metrics': {expr: factor_metrics[expr] for expr in top_factors},
        }, f, indent=2)
    
    print(f"  Zoo saved to: {zoo_path}")
    
    return {
        'factor_exprs': top_factors,
        'factor_scores_dict': factor_scores_dict,
        'metrics': factor_metrics,
        'zoo_path': zoo_path,
    }


def stage2_combine_factors(
    prices: pd.DataFrame,
    stage1_results: Dict,
    config: AlphaForgeConfig,
    output_dir: str,
) -> Dict:
    """
    Stage 2: Combine alpha factors using rolling window.
    
    This follows the logic in combine_AFF.py:
    - Use rolling window to evaluate factors
    - Select top n_factors based on Rank IC IR
    - Use linear regression to combine factors
    
    Args:
        prices: Price data from main DataLoader
        stage1_results: Results from Stage 1
        config: AlphaForge configuration
        output_dir: Directory to save results
        
    Returns:
        Dict: Combined predictions and weights
    """
    print("\n" + "="*60)
    print("[Stage 2] Combining alpha factors...")
    print("="*60)
    
    # Convert to MultiIndex format
    prices_multindex = convert_to_multindex(prices)
    returns = compute_returns(prices_multindex)
    
    # Load factor expressions from Stage 1
    factor_exprs = stage1_results['factor_exprs']
    
    # Evaluate all factors on all data
    print(f"  Evaluating {len(factor_exprs)} factors...")
    all_factor_values = {}
    for expr in factor_exprs:
        all_factor_values[expr] = evaluate_factor_expression(expr, prices_multindex)
    
    # Calculate IC and Rank IC for each factor at each date
    print("  Calculating rolling IC...")
    n_dates = len(returns.index)
    n_factors = len(factor_exprs)
    
    # Store IC time series for each factor
    factor_ic_series = {expr: [] for expr in factor_exprs}
    factor_rankic_series = {expr: [] for expr in factor_exprs}
    
    for i, date in enumerate(returns.index):
        if i == 0:
            continue
            
        # Use past window to calculate IC
        if config.window == "inf":
            start_idx = 0
        else:
            start_idx = max(0, i - config.window)
        
        for expr in factor_exprs:
            factor_past = all_factor_values[expr].iloc[start_idx:i]
            returns_past = returns.iloc[start_idx:i]
            
            ic = calculate_ic(factor_past, returns_past)
            rank_ic = calculate_rank_ic(factor_past, returns_past)
            
            factor_ic_series[expr].append(ic)
            factor_rankic_series[expr].append(rank_ic)
    
    # Rolling combination
    print("  Rolling combination...")
    predictions = []
    selected_factors_history = []
    weights_history = []
    
    # Dynamic minimum start index (handle short data)
    if config.window == "inf":
        min_start = min(63, n_dates // 3)  # Use 1/3 of data for short periods
    else:
        min_start = min(config.window, n_dates // 3)
    
    for i in range(min_start, n_dates):
        if i >= len(returns.index):
            break
            
        date = returns.index[i]
        
        # Calculate factor metrics using past window
        if config.window == "inf":
            start_idx = 0
        else:
            start_idx = max(0, i - config.window)
        
        factor_metrics = {}
        for expr in factor_exprs:
            ic_series = factor_ic_series[expr][start_idx:i]
            rankic_series = factor_rankic_series[expr][start_idx:i]
            
            if len(ic_series) > 0:
                ic_mean = np.mean(ic_series)
                ic_std = np.std(ic_series)
                rankic_mean = np.mean(rankic_series)
                rankic_std = np.std(rankic_series)
                
                factor_metrics[expr] = {
                    'ic': ic_mean,
                    'ic_std': ic_std,
                    'icir': ic_mean / ic_std if ic_std > 0 else 0,
                    'rank_ic': rankic_mean,
                    'rank_ic_std': rankic_std,
                    'rank_icir': rankic_mean / rankic_std if rankic_std > 0 else 0,
                }
        
        # Select top factors
        sorted_factors = sorted(
            factor_metrics.items(),
            key=lambda x: abs(x[1].get('rank_icir', 0)),
            reverse=True
        )
        
        # Filter by thresholds (from combine_AFF.py)
        good_factors = [
            x[0] for x in sorted_factors
            if abs(x[1].get('rank_ic', 0)) > 0.02 and abs(x[1].get('rank_icir', 0)) > 0.2
        ]
        
        if len(good_factors) < 1:
            good_factors = [sorted_factors[0][0]]
        
        good_factors = good_factors[:config.n_factors]
        selected_factors_history.append(good_factors)
        
        # Prepare data for linear regression
        X = np.column_stack([
            all_factor_values[expr].loc[date].fillna(0)
            for expr in good_factors
        ])
        y = returns.loc[date].fillna(0).values
        
        # Fit linear regression using numpy lstsq
        valid_idx = np.isfinite(y)
        if valid_idx.sum() < 10:
            predictions.append(0)
            weights_history.append(np.zeros(len(good_factors)))
            continue
        
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        # Add bias term
        X_bias = np.column_stack([X_valid, np.ones(len(X_valid))])
        
        # Solve using least squares
        coef = np.linalg.lstsq(X_bias, y_valid, rcond=None)[0]
        
        # Predict
        X_pred_bias = np.column_stack([X, np.ones(len(X))])
        pred = X_pred_bias @ coef
        predictions.append(pred)
        weights_history.append(coef[:-1])  # Exclude bias term
    
    print(f"  Generated {len(predictions)} predictions")
    
    # Save results
    pred_path = os.path.join(output_dir, "predictions.npy")
    np.save(pred_path, np.array(predictions))
    
    weights_path = os.path.join(output_dir, "weights.npy")
    np.save(weights_path, np.array(weights_history))
    
    print(f"  Predictions saved to: {pred_path}")
    print(f"  Weights saved to: {weights_path}")
    
    return {
        'predictions': predictions,
        'selected_factors': selected_factors_history,
        'weights': weights_history,
        'pred_path': pred_path,
        'weights_path': weights_path,
    }


def stage3_evaluate_results(
    prices: pd.DataFrame,
    stage2_results: Dict,
    config: AlphaForgeConfig,
) -> Dict:
    """
    Stage 3: Evaluate results using the unified BacktestEngine.

    Constructs portfolio weights from Stage 2 predictions and runs the
    unified backtest engine for consistent metrics across all baselines.

    Args:
        prices: Price data from main DataLoader (date x stock)
        stage2_results: Results from Stage 2 (contains 'predictions' list)
        config: AlphaForge configuration

    Returns:
        Dict: Final performance metrics (from BacktestEngine)
    """
    print("\n" + "="*60)
    print("[Stage 3] Evaluating results (unified BacktestEngine)...")
    print("="*60)

    predictions = stage2_results['predictions']
    if not predictions:
        print("  ⚠️  No predictions from Stage 2, returning zero metrics")
        return {
            'metrics': {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            },
            'portfolio_returns': pd.Series(dtype=float),
        }

    # Build portfolios DataFrame from predictions
    # predictions: list of arrays (one per date), each array = stock scores
    print("  Building portfolios from predictions...")

    portfolio_rows = []
    date_index = []

    for i, pred in enumerate(predictions):
        if not isinstance(pred, np.ndarray):
            continue

        # pred is a 1D array of scores for all stocks
        # Need to map to stock names — use prices.columns
        n_stocks = min(len(pred), len(prices.columns))
        scores = pd.Series(pred[:n_stocks], index=prices.columns[:n_stocks])

        # Select top-N stocks and equal-weight
        top = scores.dropna().nlargest(config.top_n_stocks)
        if len(top) == 0:
            continue

        w = pd.Series(1.0 / len(top), index=top.index)
        portfolio_rows.append(w)
        # Use prices.index aligned to predictions (skip first date which has no return)
        if i + 1 < len(prices):
            date_index.append(prices.index[i + 1])

    if not portfolio_rows:
        print("  ⚠️  No valid portfolios built, returning zero metrics")
        return {
            'metrics': {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            },
            'portfolio_returns': pd.Series(dtype=float),
        }

    # Align to common dates and columns
    all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
    portfolios = pd.DataFrame(
        index=pd.DatetimeIndex(date_index),
        columns=all_stocks,
        dtype=float,
    )
    for i, w in enumerate(portfolio_rows):
        portfolios.loc[date_index[i], w.index] = w.values
    portfolios = portfolios.fillna(0.0)
    portfolios = portfolios.div(portfolios.sum(axis=1), axis=0).fillna(0.0)

    # Align prices to portfolio dates
    prices_aligned = prices.reindex(portfolios.index)
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    # Run unified backtest
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=1,
    )
    metrics = engine.run(portfolios, prices_aligned)

    print(f"\n  Results (BacktestEngine):")
    print(f"    Annual Return:    {metrics.get('annual_return', 0):.4f}")
    print(f"    Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"    Max Drawdown:     {metrics.get('max_drawdown', 0):.4f}")
    print(f"    Information Ratio:{metrics.get('information_ratio', 0):.4f}")
    print(f"    Win Rate:         {metrics.get('win_rate', 0):.4f}")
    print(f"    Calmar Ratio:     {metrics.get('calmar_ratio', 0):.4f}")

    return {
        'metrics': metrics,
        'portfolio_returns': engine.get_returns(),
    }


def run_alphaforge_baseline(
    config_path: str = "config/config.yaml",
    dataloader=None,
    prices: pd.DataFrame = None,
    start_date: str = None,
    end_date: str = None,
    instruments: str = None,
    top_n_stocks: int = None,
    n_factors: int = 10,
    zoo_size: int = 50,
    seeds: List[int] = None,
    output_dir: str = "results/alphaforge",
    verbose: bool = False,
) -> Dict:
    """
    Run complete AlphaForge baseline (all 3 stages).
    
    Data loading priority:
    1. If prices is provided, use it directly
    2. If dataloader is provided, call dataloader.get_prices()
    3. If config_path is provided (and dataloader/prices are None), load from config
    
    Args:
        config_path: Path to config YAML (used if dataloader/prices not provided)
        dataloader: Main project DataLoader (optional)
        prices: Price data DataFrame (optional)
        start_date: Start date (overrides config)
        end_date: End date (overrides config)
        instruments: Instrument list name (overrides config)
        top_n_stocks: Number of stocks in portfolio (overrides config)
        n_factors: Number of factors to combine
        zoo_size: Number of factors to mine
        seeds: Random seeds
        output_dir: Output directory
        verbose: Verbose output
        
    Returns:
        Dict: Results with metrics
    """
    # Load config if needed (for defaults)
    if config_path and (start_date is None or end_date is None or instruments is None or top_n_stocks is None):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if start_date is None:
                start_date = config['data']['universe'].get('start_date', '2019-01-01')
            if end_date is None:
                end_date = config['data']['universe'].get('end_date', '2025-12-31')
            if instruments is None:
                instruments = config['data']['universe'].get('name', 'csi300')
            if top_n_stocks is None:
                top_n_stocks = config.get('backtest', {}).get('top_n_stocks', 50)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            # Use defaults
            if start_date is None:
                start_date = "2023-01-01"
            if end_date is None:
                end_date = "2024-12-31"
            if instruments is None:
                instruments = "csi300"
            if top_n_stocks is None:
                top_n_stocks = 50
    
    # Load data
    if prices is None and dataloader is None and config_path:
        # Load from config
        print("Loading data from config...")
        try:
            from dataloader.loader import DataLoader as ProjectDataLoader
            loader = ProjectDataLoader(config_path=config_path)
            price_data, _, _ = loader.load_data(start_date=start_date, end_date=end_date)
            
            # Convert to DataFrame format expected by our functions
            rows = []
            for date in price_data['close'].index:
                for symbol in price_data['close'].columns:
                    rows.append({
                        'symbol': symbol,
                        'date': date,
                        'open': price_data['open'].loc[date, symbol] if 'open' in price_data else np.nan,
                        'high': price_data['high'].loc[date, symbol] if 'high' in price_data else np.nan,
                        'low': price_data['low'].loc[date, symbol] if 'low' in price_data else np.nan,
                        'close': price_data['close'].loc[date, symbol],
                        'volume': price_data['volume'].loc[date, symbol] if 'volume' in price_data else np.nan,
                    })
            prices = pd.DataFrame(rows)
            print(f"  Loaded {len(prices)} records")
        except Exception as e:
            raise ValueError(f"Failed to load data from config: {e}. Please provide dataloader or prices.")
    
    elif dataloader is not None:
        print("Loading data from DataLoader...")
        prices = dataloader.get_prices(start_date, end_date)
    
    if prices is None:
        raise ValueError("Must provide either dataloader, prices, or a valid config_path")
    
    # Create config
    config = AlphaForgeConfig(
        instruments=instruments,
        n_factors=n_factors,
        zoo_size=zoo_size,
        seeds=seeds or [0],
        top_n_stocks=top_n_stocks,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: Mine factors
    stage1_results = stage1_mine_factors(prices, config, output_dir)
    
    # Stage 2: Combine factors
    stage2_results = stage2_combine_factors(prices, stage1_results, config, output_dir)
    
    # Stage 3: Evaluate results
    stage3_results = stage3_evaluate_results(prices, stage2_results, config)
    
    return {
        'metrics': stage3_results['metrics'],
        'portfolio_returns': stage3_results['portfolio_returns'],
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AlphaForge baseline")
    parser.add_argument("--config-path", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--start", type=str, default=None, help="Start date (overrides config)")
    parser.add_argument("--end", type=str, default=None, help="End date (overrides config)")
    parser.add_argument("--instruments", type=str, default=None, help="Instruments list (overrides config)")
    parser.add_argument("--top-n", type=int, default=None, help="Number of stocks in portfolio (overrides config)")
    parser.add_argument("--n-factors", type=int, default=10, help="Number of factors to combine")
    parser.add_argument("--zoo-size", type=int, default=50, help="Number of factors to mine")
    parser.add_argument("--output-dir", type=str, default="results/alphaforge", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = run_alphaforge_baseline(
        config_path=args.config_path,
        start_date=args.start,
        end_date=args.end,
        instruments=args.instruments,
        top_n_stocks=args.top_n,
        n_factors=args.n_factors,
        zoo_size=args.zoo_size,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Annual Return: {results['metrics']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Information Ratio: {results['metrics']['information_ratio']:.4f}")
