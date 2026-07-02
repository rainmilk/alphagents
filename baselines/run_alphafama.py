# -*- coding: utf-8 -*-
"""
AlphaFAMA Baseline Runner — Integrated with Main Dataloader

This runner:
1. Loads A-share data via the main project's DataLoader
2. Converts to AlphaFAMA's expected format via data_bridge
3. Runs AlphaFAMA's 101-factor generation + IC computation
4. Returns performance metrics compatible with the main project

Usage:
    python baselines/run_alphafama.py
    python baselines/run_alphafama.py --start 2020-01-01 --end 2024-12-31 --universe hs300

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Path setup: add project root to sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.loader import DataLoader
from baselines.AlphaFAMA.src.data_bridge import (
    convert_price_data_to_alphafama,
    split_alphafama_data,
)
# NOTE: AlphaFAMA modules use relative imports; we import them via the
# full package path to avoid polluting sys.path with AlphaFAMA's src/
# (which would shadow the main project's config.py).
from baselines.AlphaFAMA.src.alpha_functions import AlphaFactory
from baselines.AlphaFAMA.src.factor_matrix import compute_ic_matrix
from scipy.stats import spearmanr


def run_alphafama_baseline(
    config_path: str = "config/config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    context_days: int = 30,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run AlphaFAMA baseline using the main project's DataLoader.

    Args:
        config_path: Path to the main project config file.
        start_date: Data start date (YYYY-MM-DD).
        end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        context_days: Context window for factor calculation.
        output_dir: Directory for saving results.

    Returns:
        Dict of performance metrics with keys:
            annual_return, sharpe_ratio, max_drawdown, information_ratio,
            mean_rank_ic, icir, top_ic_factors, n_factors
    """
    print("=" * 60)
    print("  AlphaFAMA Baseline — A-Share (via Main DataLoader)")
    print("=" * 60)

    # ── Step 1: Load data via main DataLoader ──────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )

    print(f"  Loaded: {len(price_data['close'].index)} trading days × "
          f"{len(price_data['close'].columns)} stocks")

    # ── Step 2: Convert to AlphaFAMA format ────────────────────────────
    print("\n[Step 2] Converting data to AlphaFAMA format...")
    af_df = convert_price_data_to_alphafama(price_data)
    print(f"  Converted: {len(af_df)} rows, MultiIndex (date, ticker)")

    # ── Step 3: Train/Test split ───────────────────────────────────────
    print("\n[Step 3] Splitting into train/test...")
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')

    train_df, test_df = split_alphafama_data(
        af_df,
        train_end_date=train_end,
        test_start_date=test_start,
        context_days=context_days,
    )

    # ── Step 4: Generate Alpha101 factors ──────────────────────────────
    print("\n[Step 4] Generating Alpha101 factors...")
    train_exposures, train_returns = _compute_factors(train_df)
    test_exposures, test_returns = _compute_factors(test_df)

    n_factors = len(train_exposures.columns)
    print(f"  Generated {n_factors} factors")

    # ── Step 5: Compute Rank-IC on training data ───────────────────────
    print("\n[Step 5] Computing Rank-IC on training data...")
    train_ic = compute_ic_matrix(train_exposures, train_returns)
    mean_train_ic = train_ic.mean()
    avg_ic = mean_train_ic.mean()
    ic_std = mean_train_ic.std()
    icir = avg_ic / ic_std if ic_std > 0 else 0.0

    print(f"  Mean Rank-IC (train): {avg_ic:.4f}, ICIR: {icir:.4f}")

    # ── Step 6: Evaluate on test data ──────────────────────────────────
    print("\n[Step 6] Evaluating on test data...")
    test_ic = compute_ic_matrix(test_exposures, test_returns)
    mean_test_ic = test_ic.mean()
    avg_test_ic = mean_test_ic.mean()

    # Filter to logical test period (exclude context window)
    test_start_ts = pd.Timestamp(test_start)
    logical_test_ic = test_ic[test_ic.index >= test_start_ts]
    if len(logical_test_ic) > 0:
        logical_avg_ic = logical_test_ic.mean().mean()
    else:
        logical_avg_ic = avg_test_ic

    print(f"  Mean Rank-IC (test): {avg_test_ic:.4f}")
    if len(logical_test_ic) > 0:
        print(f"  Mean Rank-IC (test, no context): {logical_avg_ic:.4f}")

    # ── Step 7: Top IC factors ─────────────────────────────────────────
    top_n = min(10, n_factors)
    top_factors = mean_train_ic.abs().nlargest(top_n)
    top_factors_dict = {k: float(v) for k, v in top_factors.items()}

    print(f"\n  Top-{top_n} factors by |IC| (train):")
    for f, ic_val in top_factors_dict.items():
        print(f"    {f}: {ic_val:.4f}")

    # ── Step 8: Simulate portfolio performance (unified BacktestEngine) ─
    print("\n[Step 8] Simulating portfolio performance (unified BacktestEngine)...")

    # Build prices DataFrame for BacktestEngine (close price, date x stock)
    prices = price_data.get('close')
    if prices is None:
        raise ValueError("Missing 'close' price data for backtest")

    simulated_metrics = _simulate_portfolio_from_ic(
        train_ic=train_ic,
        test_ic=test_ic,
        test_exposures=test_exposures,
        test_returns=test_returns,
        top_n_factors=top_factors,
        test_start_date=test_start,
        prices=prices,
        holding_period=1,  # Daily rebalance for fairest comparison
    )

    # ── Step 9: Compile results ────────────────────────────────────────
    results = {
        'method': 'AlphaFAMA',
        'n_factors': n_factors,
        'mean_rank_ic_train': float(avg_ic),
        'icir': float(icir),
        'mean_rank_ic_test': float(logical_avg_ic),
        'top_ic_factors': top_factors_dict,
        'annual_return': simulated_metrics.get('annual_return', 0.0),
        'sharpe_ratio': simulated_metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': simulated_metrics.get('max_drawdown', 0.0),
        'information_ratio': simulated_metrics.get('information_ratio', 0.0),
        'calmar_ratio': simulated_metrics.get('calmar_ratio', 0.0),
        'win_rate': simulated_metrics.get('win_rate', 0.0),
        'avg_turnover': simulated_metrics.get('avg_turnover', 0.0),
        'train_end': train_end,
        'test_start': test_start,
    }

    # ── Step 10: Save results ──────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, 'alphafama_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  AlphaFAMA Baseline Complete")
    print("=" * 60)

    return results


def _compute_factors(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Alpha101 factor exposures for each ticker.

    Args:
        df: AlphaFAMA-format DataFrame with MultiIndex (date, ticker).

    Returns:
        Tuple of (exposures_df, returns_df) with MultiIndex (date, ticker).
    """
    ex_list, ret_list = [], []
    for ticker, grp in df.groupby("ticker"):
        alphas = AlphaFactory.all_alphas(grp)
        ex_list.append(
            pd.DataFrame(alphas, index=grp.index).assign(ticker=ticker)
        )
        ret_list.append(
            grp[["returns"]].assign(ticker=ticker)
        )

    exposures = pd.concat(ex_list)
    returns = pd.concat(ret_list)
    return exposures, returns


def _simulate_portfolio_from_ic(
    train_ic: pd.DataFrame,
    test_ic: pd.DataFrame,
    test_exposures: pd.DataFrame,
    test_returns: pd.DataFrame,
    top_n_factors: pd.Series,
    test_start_date: str,
    prices: pd.DataFrame,
    holding_period: int = 1,
) -> Dict:
    """
    Simulate an IC-weighted portfolio using the unified BacktestEngine.

    Strategy:
    1. Select top-N factors by training |IC|
    2. At each rebalance date, compute weighted score as sum(|IC| * normalized_exposure)
    3. Go long top-50 stocks by score, equal-weight
    4. Use BacktestEngine for consistent metrics

    Args:
        train_ic: IC matrix on training data (dates x factors)
        test_ic: IC matrix on test data (dates x factors)
        test_exposures: Factor exposures on test data, MultiIndex (date, ticker) x factors
        test_returns: Returns on test data, MultiIndex (date, ticker) x ['returns']
        top_n_factors: Top factors by |IC| (Series, index=factor name, value=IC)
        test_start_date: Test start date (to filter out context window)
        prices: Close price DataFrame (date x stock) for BacktestEngine
        holding_period: Rebalance frequency (1=daily, 5=weekly, 20=monthly)

    Returns:
        Dict with metrics from BacktestEngine.
    """
    from backtest.engine import BacktestEngine

    # Get top factor names
    top_factor_names = list(top_n_factors.index[:min(10, len(top_n_factors))])
    factor_weights = train_ic[top_factor_names].abs().mean()
    factor_weights = factor_weights / factor_weights.sum()

    # Filter test exposures to logical test period (no context)
    test_start_ts = pd.Timestamp(test_start_date)
    dates_in_range = test_exposures.index.get_level_values('date')
    test_exposures_filtered = test_exposures[dates_in_range >= test_start_ts]

    if test_exposures_filtered.empty:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }

    unique_dates = test_exposures_filtered.index.get_level_values('date').unique().sort_values()

    # Build portfolios DataFrame: each row = one date, values = position weights
    portfolio_rows = []
    portfolio_dates = []

    # Rebalance at holding_period intervals
    rebalance_indices = list(range(0, len(unique_dates), holding_period))

    for idx_pos, i in enumerate(rebalance_indices):
        rebal_date = unique_dates[i]

        try:
            exp = test_exposures_filtered.xs(rebal_date, level='date')
        except KeyError:
            continue

        # Compute composite score = weighted sum of normalized factor exposures
        score = pd.Series(0.0, index=exp.index)
        for f in top_factor_names:
            if f in exp.columns:
                f_vals = exp[f].dropna()
                if len(f_vals) > 1:
                    f_norm = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-10)
                    score.loc[f_norm.index] += factor_weights.get(f, 0.0) * f_norm

        # Select top-50 stocks and equal-weight
        top_stocks = score.nlargest(min(50, len(score)))
        if len(top_stocks) == 0:
            # No stocks selected: emit a zero-weight row (BacktestEngine handles this)
            continue

        w = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        portfolio_rows.append(w)
        portfolio_dates.append(rebal_date)

    if not portfolio_rows:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }

    # Build portfolios DataFrame
    all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
    portfolios = pd.DataFrame(
        index=pd.DatetimeIndex(portfolio_dates),
        columns=all_stocks,
        dtype=float,
    )
    for i, w in enumerate(portfolio_rows):
        portfolios.loc[portfolio_dates[i], w.index] = w.values
    portfolios = portfolios.fillna(0.0)
    portfolios = portfolios.div(portfolios.sum(axis=1), axis=0).fillna(0.0)

    # Align prices to portfolio dates
    prices_aligned = prices.reindex(portfolios.index)
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    # Run unified backtest
    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    metrics = engine.run(portfolios, prices_aligned)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AlphaFAMA baseline with main DataLoader')
    parser.add_argument('--config', default='config/config.yaml', help='Path to main config')
    parser.add_argument('--start', default=None, help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='Data end date (YYYY-MM-DD)')
    parser.add_argument('--universe', default=None, help='Stock universe (hs300, zz500, all_a)')
    parser.add_argument('--train-end', default=None, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', default=None, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--context-days', type=int, default=30, help='Context window days')
    parser.add_argument('--output-dir', default='experiments/alphafama', help='Output directory')

    args = parser.parse_args()

    results = run_alphafama_baseline(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        context_days=args.context_days,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic_train']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
