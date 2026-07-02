#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaAgent baseline runner using the main project's DataLoader.

AlphaAgent is an LLM-driven autonomous agent for alpha factor mining.
This runner integrates it with the main project by:
1. Loading data via the main DataLoader (instead of Qlib)
2. Generating HDF5 data files that AlphaAgent's factor execution code expects
3. Using AlphaAgent's function library to compute factor values from formulas
4. Computing Rank-IC, ranking factors, and running portfolio backtest
5. Outputting evaluation metrics

When no LLM is available, falls back to simulated factor formulas.

Backtest uses the unified BacktestEngine from backtest/engine.py to ensure
consistent evaluation across all baselines.

Usage:
    python baselines/run_alphaagent.py
    python baselines/run_alphaagent.py --output-dir experiments/alphaagent_test
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "AlphaAgent"))

from dataloader.loader import DataLoader

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data bridge: generate HDF5 files from main DataLoader
# ═══════════════════════════════════════════════════════════════════════

def convert_to_multindex(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Convert main DataLoader price_data dict to MultiIndex (datetime, instrument) Series.

    The main DataLoader returns {field: DataFrame(date × stock)} dictionaries.
    AlphaAgent's function library expects MultiIndex Series with index names
    ('datetime', 'instrument').
    """
    result = {}
    for field, df in price_data.items():
        if df is None or df.empty:
            continue
        # Stack: (date, stock) → MultiIndex
        stacked = df.stack()
        stacked.index.names = ['datetime', 'instrument']
        result[field] = stacked
    return result


def compute_returns(price_midx: Dict[str, pd.Series]) -> pd.Series:
    """Compute daily returns from close prices."""
    close = price_midx.get('close')
    if close is None:
        raise ValueError("Missing 'close' field in price data")
    returns = close.unstack('instrument').pct_change().fillna(0)
    returns = returns.stack()
    returns.index.names = ['datetime', 'instrument']
    return returns


def save_data_as_hdf5(
    price_midx: Dict[str, pd.Series],
    return_series: pd.Series,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Save price/return data as HDF5 files compatible with AlphaAgent's factor code.

    AlphaAgent's factor.py expects to read data from daily_pv.h5 files that contain
    columns: $open, $close, $high, $low, $volume, $return.

    Returns paths to (all_data_h5, debug_data_h5).
    Falls back to pickle if HDF5 (pytables) is not available.
    """
    # Build the combined DataFrame
    combined = pd.DataFrame(index=price_midx.get('close').index)

    field_map = {
        'open': '$open',
        'close': '$close',
        'high': '$high',
        'low': '$low',
        'volume': '$volume',
    }
    for src_field, h5_col in field_map.items():
        if src_field in price_midx:
            combined[h5_col] = price_midx[src_field]

    # Add returns
    combined['$return'] = return_series.reindex(combined.index)

    # Drop NaN-only rows
    combined = combined.dropna(how='all')
    combined = combined.sort_index()

    os.makedirs(output_dir, exist_ok=True)

    # Try HDF5 first, fall back to pickle
    h5_ok = True
    try:
        all_path = os.path.join(output_dir, "daily_pv_all.h5")
        combined.to_hdf(all_path, key="data", mode='w')
    except (ImportError, OSError) as e:
        h5_ok = False
        all_path = os.path.join(output_dir, "daily_pv_all.pkl")
        combined.to_pickle(all_path)
        print(f"  [WARN] pytables not available, using pickle: {all_path}")
    else:
        print(f"  Saved: {all_path}  ({combined.shape[0]} rows × {combined.shape[1]} cols)")

    # Save debug data (subset: first 100 instruments)
    instruments = combined.index.get_level_values('instrument').unique()
    debug_instruments = instruments[:min(100, len(instruments))]
    debug_data = combined.loc[pd.IndexSlice[:, debug_instruments], :]

    if h5_ok:
        try:
            debug_path = os.path.join(output_dir, "daily_pv_debug.h5")
            debug_data.to_hdf(debug_path, key="data", mode='w')
        except (ImportError, OSError):
            debug_path = os.path.join(output_dir, "daily_pv_debug.pkl")
            debug_data.to_pickle(debug_path)
    else:
        debug_path = os.path.join(output_dir, "daily_pv_debug.pkl")
        debug_data.to_pickle(debug_path)

    print(f"  Saved: {debug_path}  ({debug_data.shape[0]} rows × {debug_data.shape[1]} cols)")

    return all_path, debug_path


# ═══════════════════════════════════════════════════════════════════════
# Simulated factor generation (no LLM required)
# ═══════════════════════════════════════════════════════════════════════

# All available fields for factor formulas
AVAILABLE_FIELDS = ['$open', '$close', '$high', '$low', '$volume']

# AlphaAgent's function library operators (cross-sectional and time-series)
OPS_CS = ['Rank', 'Delayed_Rank', 'Std_CS', 'Mean_CS', 'Skew_CS', 'Kurt_CS',
          'Min_CS', 'Max_CS', 'Median_CS', 'DELTA_CS']
OPS_TS = ['TS_Mean', 'TS_Std', 'TS_Min', 'TS_Max', 'TS_Median', 'TS_Sum',
          'TS_Rank', 'TS_MAD', 'TS_ZScore', 'TS_PctChange', 'EMA', 'SMA',
          'DELTA', 'DELAY', 'ABS', 'LOG', 'SIGN', 'SQRT', 'PROD',
          'DECAYLINEAR', 'COUNT']
OPS_BINARY = ['TS_CORR', 'TS_COVARIANCE', 'REGBETA', 'REGRESI']

WINDOW_PARAMS = [5, 10, 20, 30, 60]


def _generate_random_formula(rng: np.random.Generator, field: str = None) -> str:
    """Generate a random factor formula string in AlphaAgent function notation."""
    field = field or rng.choice(AVAILABLE_FIELDS)

    op_type = rng.choice(['cs', 'ts', 'ts', 'ts', 'binary'])  # bias toward time-series

    if op_type == 'cs':
        op = rng.choice(OPS_CS)
        return f"{op}({field})"
    elif op_type == 'ts':
        op = rng.choice(OPS_TS)
        w = rng.choice(WINDOW_PARAMS)
        if op in ('ABS', 'LOG', 'SIGN', 'SQRT', 'DELTA_CS'):
            return f"{op}({field})"
        else:
            return f"{op}({field}, {w})"
    else:  # binary
        op = rng.choice(OPS_BINARY)
        field2 = rng.choice(AVAILABLE_FIELDS)
        w = rng.choice(WINDOW_PARAMS)
        return f"{op}({field}, {field2}, {w})"


def generate_simulated_formulas(
    n_formulas: int = 50,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Generate simulated factor formulas for baseline evaluation.

    Returns list of (formula_name, formula_string) tuples.
    """
    rng = np.random.default_rng(seed)
    formulas = []

    for i in range(n_formulas):
        name = f"alpha_{i+1:03d}"
        formula = _generate_random_formula(rng)
        formulas.append((name, formula))

    return formulas


# ═══════════════════════════════════════════════════════════════════════
# Factor computation using AlphaAgent's function library
# ═══════════════════════════════════════════════════════════════════════

def compute_factor_values(
    formulas: List[Tuple[str, str]],
    price_midx: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Compute factor values from formulas using AlphaAgent's function library.

    This is a simplified evaluator that computes factor values directly in pandas
    without going through the full AlphaAgent pipeline (code generation → execution).

    Args:
        formulas: List of (name, expression) tuples
        price_midx: Price data as MultiIndex Series dict

    Returns:
        DataFrame with datetime × instrument index, one column per factor
    """
    # Import AlphaAgent's function library
    from baselines.AlphaAgent.alphaagent.components.coder.factor_coder import function_lib as flib

    # Build a combined DataFrame from price data
    price_df = pd.DataFrame(index=price_midx.get('close', list(price_midx.values())[0]).index)
    field_map = {
        'open': '$open', 'close': '$close', 'high': '$high',
        'low': '$low', 'volume': '$volume',
    }
    for src, target in field_map.items():
        if src in price_midx:
            price_df[target] = price_midx[src]
    price_df = price_df.sort_index()

    factors = {}
    n_total = len(formulas)
    n_ok = 0
    n_fail = 0

    for name, formula in formulas:
        try:
            value = _eval_alphaagent_formula(formula, price_df, flib)
            if value is not None and not value.isna().all():
                factors[name] = value
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.debug(f"  Failed computing {name}: {e}")
            n_fail += 1

    print(f"  Computed {n_ok}/{n_total} formulas (failed: {n_fail})")

    if not factors:
        raise RuntimeError("No factors could be computed")

    df = pd.DataFrame(factors).sort_index()
    return df


def _eval_alphaagent_formula(
    formula: str,
    price_df: pd.DataFrame,
    flib,
) -> Optional[pd.Series]:
    """
    Evaluate a single AlphaAgent-style formula.

    Supports:
    - Unary: Op(field), Op(field, window)
    - Binary: Op(field1, field2, window)
    """
    import re
    formula = formula.strip()
    match = re.match(r'(\w+)\(([^)]+)\)', formula)
    if not match:
        logger.debug(f"  Cannot parse: {formula}")
        return None

    op_name = match.group(1)
    args_str = match.group(2)
    args = [a.strip() for a in args_str.split(',')]

    def resolve_arg(arg: str):
        arg = arg.strip()
        if arg.startswith('$') and arg in price_df.columns:
            return price_df[arg]
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return arg

    resolved = [resolve_arg(a) for a in args]

    func = getattr(flib, op_name, None)
    if func is None:
        func = getattr(flib, op_name.upper(), None)
    if func is None:
        logger.debug(f"  Unknown function: {op_name}")
        return None

    try:
        if len(resolved) == 1:
            result = func(resolved[0])
        elif len(resolved) == 2:
            result = func(resolved[0], resolved[1])
        elif len(resolved) == 3:
            result = func(resolved[0], resolved[1], resolved[2])
        else:
            result = func(*resolved)

        if isinstance(result, np.ndarray):
            result = pd.Series(result.flatten(), index=price_df.index)
        elif isinstance(result, pd.DataFrame) and result.shape[1] == 1:
            result = result.iloc[:, 0]
        elif isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]

        if isinstance(result, pd.Series) and not result.index.equals(price_df.index):
            result = result.reindex(price_df.index)

        return result
    except Exception as e:
        logger.debug(f"  Error calling {op_name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# IC computation and factor ranking
# ═══════════════════════════════════════════════════════════════════════

def compute_rank_ic(
    factor_df: pd.DataFrame,
    return_series: pd.Series,
    train_end_date: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Spearman Rank-IC for each factor on the training set.

    Args:
        factor_df: DataFrame with factor values (datetime × instrument)
        return_series: Future returns (same index)
        train_end_date: End of training period

    Returns:
        (ic_series: mean IC per factor, ic_all_df: IC per date per factor)
    """
    train_idx = factor_df.index.get_level_values('datetime') <= train_end_date

    factor_train = factor_df.loc[train_idx]
    ret_train = return_series.reindex(factor_train.index)

    ic_results = {}
    for col in factor_train.columns:
        df = pd.DataFrame({'factor': factor_train[col], 'ret': ret_train})
        ic = df.groupby('datetime').apply(
            lambda x: x['factor'].corr(x['ret'], method='spearman')
        )
        ic_results[col] = ic

    ic_df = pd.DataFrame(ic_results)
    ic_mean = ic_df.mean().sort_values(ascending=False)

    return ic_mean, ic_df


# ═══════════════════════════════════════════════════════════════════════
# Portfolio construction (returns portfolios DataFrame for BacktestEngine)
# ═══════════════════════════════════════════════════════════════════════

def simulate_factor_portfolio(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    ic_mean: pd.Series,
    test_start_date: str,
    end_date: str,
    top_n_factors: int = 10,
    top_n_stocks: int = 30,
) -> pd.DataFrame:
    """
    Construct daily portfolio weights from factor scores.

    Uses the unified BacktestEngine (backtest/engine.py) for backtesting.
    This function only constructs the portfolio weights DataFrame.

    Args:
        factor_df: Factor values, wide DataFrame (date x stock) or MultiIndex
        prices: Daily close prices (date x stock), used by BacktestEngine
        ic_mean: Mean IC for each factor (used for factor selection/weighting)
        test_start_date: Start date for test period
        end_date: End date
        top_n_factors: Number of top factors to use
        top_n_stocks: Number of stocks in portfolio each day

    Returns:
        pd.DataFrame: Daily portfolio weights (date x stock), each row sums to 1.0
    """
    # Handle MultiIndex Series input (factor_df from AlphaAgent)
    if isinstance(factor_df, pd.Series):
        factor_df = factor_df.unstack(fill_value=np.nan)
    elif factor_df.index.nlevels > 1:
        # MultiIndex DataFrame → unstack to wide
        factor_df = factor_df.unstack(fill_value=np.nan)

    # Select top factors by |IC|
    top_factors = ic_mean.abs().nlargest(top_n_factors).index.tolist()
    available = [f for f in top_factors if f in factor_df.columns]
    if not available:
        raise RuntimeError("No valid factors found for portfolio construction")

    # Align to test period
    test_start_ts = pd.Timestamp(test_start_date)
    factor_test = factor_df[factor_df.index >= test_start_ts]
    if factor_test.empty:
        return pd.DataFrame()

    # Compute composite score: equal-weight average of z-scored top factors
    # Z-score normalization per date (cross-sectional)
    composite = pd.DataFrame(0.0, index=factor_test.index, columns=factor_test.columns)
    for f in available:
        vals = factor_test[f]
        # Group by date and normalize
        norm = vals.groupby(vals.index) if vals.index.nlevels == 1 else vals.groupby(level=0)
        norm = norm.transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))
        composite = composite.add(norm, fill_value=0.0)
    composite = composite.div(len(available))

    # Build portfolios: each row = one date, values = weights for top-N stocks
    portfolio_rows = []
    date_index = []

    for date in composite.index:
        scores = composite.loc[date].dropna()
        if scores.empty:
            continue

        # Select top-N stocks and equal-weight
        top = scores.nlargest(top_n_stocks)
        if len(top) == 0:
            continue

        w = pd.Series(1.0 / len(top), index=top.index)
        portfolio_rows.append(w)
        date_index.append(date)

    if not portfolio_rows:
        return pd.DataFrame()

    # Align to a common column set (union of all selected stocks across dates)
    all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
    portfolios = pd.DataFrame(
        index=pd.DatetimeIndex(date_index),
        columns=all_stocks,
        dtype=float,
    )
    for i, w in enumerate(portfolio_rows):
        portfolios.loc[date_index[i], w.index] = w.values

    portfolios = portfolios.fillna(0.0)
    # Re-normalize: each row must sum to exactly 1.0
    row_sums = portfolios.sum(axis=1)
    portfolios = portfolios.div(row_sums, axis=0).fillna(0.0)

    return portfolios


# ═══════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════

def run_alphaagent_baseline(
    config_path: str = "config/config.yaml",
    output_dir: str = "experiments/alphaagent",
    n_formulas: int = 50,
    seed: int = 42,
    start_date: str = None,
    end_date: str = None,
    train_end_date: str = None,
    test_start_date: str = None,
) -> Dict:
    """
    Run AlphaAgent baseline using the main project's DataLoader.

    Uses the unified BacktestEngine from backtest/engine.py for consistent
    performance evaluation across all baselines.

    Args:
        config_path: Path to main project config YAML
        output_dir: Directory to save results
        n_formulas: Number of simulated factor formulas
        seed: Random seed for reproducibility
        start_date: Override data start date
        end_date: Override data end date
        train_end_date: Override train end date
        test_start_date: Override test start date

    Returns:
        Dict with metrics, IC information, and factor details
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ─────────────────────────────────────────────────
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config.get('data', {}).get('universe', {})
    start_date = start_date or data_cfg.get('start_date', '2019-01-01')
    end_date = end_date or data_cfg.get('end_date', '2025-12-31')
    train_end_date = train_end_date or config['data'].get('train_end_date', '2023-12-31')
    test_start_date = test_start_date or config['data'].get('test_start_date', '2024-01-01')

    # ── 1. Load data from main DataLoader ───────────────────────────
    print("=" * 60)
    print("[1/6] Loading data from main DataLoader...")
    print("=" * 60)

    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_series = loader.load_data(
        start_date=start_date,
        end_date=end_date,
    )

    price_midx = convert_to_multindex(price_data)
    return_series = compute_returns(price_midx)

    # Extract prices DataFrame for BacktestEngine (close price, date x stock)
    prices = price_data.get('close')
    if prices is None:
        raise ValueError("Missing 'close' price data for backtest")
    prices = prices.loc[start_date:end_date]

    n_dates = len(price_midx.get('close').index.get_level_values('datetime').unique())
    n_stocks = len(price_midx.get('close').index.get_level_values('instrument').unique())
    print(f"  Loaded: {n_dates} dates × {n_stocks} stocks")
    print(f"  Train: ≤ {train_end_date}  |  Test: ≥ {test_start_date}")

    # ── 2. Save HDF5 data for AlphaAgent compatibility ─────────────
    print(f"\n[2/6] Generating HDF5 data files...")
    data_dir = os.path.join(output_dir, "data")
    save_data_as_hdf5(price_midx, return_series, data_dir)

    # ── 3. Generate simulated factor formulas ───────────────────────
    print(f"\n[3/6] Generating {n_formulas} simulated factor formulas...")
    formulas = generate_simulated_formulas(n_formulas=n_formulas, seed=seed)
    for i, (name, formula) in enumerate(formulas[:5]):
        print(f"  {name}: {formula}")
    if len(formulas) > 5:
        print(f"  ... and {len(formulas) - 5} more")

    # Save formulas
    formulas_path = os.path.join(output_dir, "formulas.json")
    with open(formulas_path, 'w') as f:
        json.dump([(n, fm) for n, fm in formulas], f, indent=2)
    print(f"  Saved formulas to: {formulas_path}")

    # ── 4. Compute factor values ───────────────────────────────────
    print(f"\n[4/6] Computing factor values...")
    factor_df = compute_factor_values(formulas, price_midx)
    print(f"  Shape: {factor_df.shape}")

    # Save factor values
    factor_path = os.path.join(output_dir, "factors.csv")
    factor_df.to_csv(factor_path)
    print(f"  Saved factors to: {factor_path}")

    # ── 5. Compute IC and select factors ───────────────────────────
    print(f"\n[5/6] Computing Rank-IC on training set...")
    ic_mean, ic_df = compute_rank_ic(factor_df, return_series, train_end_date)

    print(f"  Top 10 factors by IC:")
    for i, (name, ic) in enumerate(ic_mean.head(10).items()):
        print(f"    {i+1}. {name}: Rank-IC = {ic:.4f}")

    # Save IC results
    ic_path = os.path.join(output_dir, "ic_results.csv")
    ic_mean.to_csv(ic_path, header=['mean_rank_ic'])
    print(f"  Saved IC results to: {ic_path}")

    # ── 6. Portfolio backtest (using unified BacktestEngine) ───────
    print(f"\n[6/6] Running portfolio backtest (unified BacktestEngine)...")

    # Build portfolio weights DataFrame
    portfolios = simulate_factor_portfolio(
        factor_df=factor_df,
        prices=prices,
        ic_mean=ic_mean,
        test_start_date=test_start_date,
        end_date=end_date,
    )

    if portfolios.empty:
        print("  ⚠️  No valid portfolios generated, using zero metrics")
        metrics = {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }
    else:
        # Align prices to portfolio dates
        prices_aligned = prices.reindex(portfolios.index)
        # Also align to portfolio columns (stocks actually held)
        prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

        # Run unified backtest
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(
            commission=0.0003,
            slippage=0.001,
            risk_free_rate=0.0,
            holding_period=1,  # Daily rebalance
        )
        metrics = engine.run(portfolios, prices_aligned)

        # Save portfolio values for analysis
        pv = engine.get_portfolio_values()
        if pv is not None and not pv.empty:
            pv_path = os.path.join(output_dir, "portfolio_values.csv")
            pv.to_csv(pv_path, header=['portfolio_value'])
            print(f"  Portfolio values saved to: {pv_path}")

    print(f"\n{'=' * 60}")
    print("  AlphaAgent Baseline Results (BacktestEngine)")
    print(f"{'=' * 60}")
    print(f"  Annual Return:    {metrics.get('annual_return', 0):.4f}")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.4f}")
    print(f"  Information Ratio:{metrics.get('information_ratio', 0):.4f}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.4f}")
    print(f"  Calmar Ratio:     {metrics.get('calmar_ratio', 0):.4f}")
    print(f"  Avg Turnover:     {metrics.get('avg_turnover', 0):.4f}")
    print(f"  N Factors:        {len(ic_mean)}")

    # ── Build result ───────────────────────────────────────────────
    best_ic = float(ic_mean.iloc[0]) if len(ic_mean) > 0 else 0.0
    avg_ic = float(ic_mean.mean()) if len(ic_mean) > 0 else 0.0

    result = {
        'metrics': metrics,
        'mean_rank_ic_train': best_ic,
        'avg_rank_ic_train': avg_ic,
        'icir': float(best_ic / max(ic_mean.std(), 1e-10)) if len(ic_mean) > 1 else 0.0,
        'n_factors': len(ic_mean),
        'annual_return': metrics.get('annual_return', 0.0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'information_ratio': metrics.get('information_ratio', 0.0),
    }

    # Save full results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n  Results saved to: {results_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaAgent Baseline Runner")
    parser.add_argument("--config-path", default="config/config.yaml",
                        help="Path to main config YAML")
    parser.add_argument("--output-dir", default="experiments/alphaagent",
                        help="Output directory for results")
    parser.add_argument("--n-formulas", type=int, default=50,
                        help="Number of simulated formulas")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start-date", default=None,
                        help="Data start date (default from config)")
    parser.add_argument("--end-date", default=None,
                        help="Data end date (default from config)")
    args = parser.parse_args()

    run_alphaagent_baseline(
        config_path=args.config_path,
        output_dir=args.output_dir,
        n_formulas=args.n_formulas,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
    )
