# -*- coding: utf-8 -*-
"""
XGBoost Baseline Runner -- Close-Price-Only

This runner mirrors the simplified LSTM baseline:
1. Loads A-share data via the main project's DataLoader
2. Uses daily close-price returns as the sole feature (no factor engineering)
3. Trains a gradient-boosted-tree model to predict forward-period returns
4. Trains once on the training period (one-shot, consistent with other baselines)
5. Ranks stocks by predicted forward return, selects top-N for portfolio
6. Backtests via unified BacktestEngine

Usage:
    python baselines/run_xgboost_simple.py
    python baselines/run_xgboost_simple.py --train-start 2020-01-01 --test-end 2024-12-31 --universe hs300

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# -- Path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.loader import DataLoader
from backtest.engine import BacktestEngine
from methods.portfolio_utils import allocate_score_proportional, allocate_portfolio_weights

# -- Model backend: prefer XGBoost, fall back to sklearn --
_MODEL_BACKEND = "unknown"
try:
    import xgboost as xgb
    _MODEL_BACKEND = "xgboost"
    print("[XGBoost-Simple] Using XGBoost backend.")
except ImportError:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        _MODEL_BACKEND = "sklearn"
        print("[XGBoost-Simple] XGBoost not installed, falling back to "
              "sklearn GradientBoostingRegressor.")
    except ImportError:
        raise ImportError(
            "Neither xgboost nor scikit-learn is available. "
            "Install one of them: pip install xgboost  OR  pip install scikit-learn"
        )


# ===========================================================================
#  Feature & Target
# ===========================================================================

def _build_features(close: pd.DataFrame) -> pd.DataFrame:
    """
    Build the single feature: daily close-price return.

    No technical indicators or factors are used -- the model must learn
    the relationship between today's return and the forward-period return
    directly from raw price changes.

    Args:
        close: Close price DataFrame (date x stock)

    Returns:
        DataFrame, MultiIndex (date, stock), single column 'daily_return'
    """
    daily_ret = close.pct_change()
    # dropna=False: keep the full (date, stock) grid so features and targets
    # share an identical MultiIndex. With the default dropna=True the first
    # row (pct_change NaN) is dropped, shifting the index and breaking
    # alignment/merging with targets downstream.
    panel = daily_ret.stack(dropna=False).to_frame('daily_return')
    panel.index.names = ['date', 'stock']
    return panel


def _build_targets(close: pd.DataFrame, forward_period: int = 10) -> pd.DataFrame:
    """
    Build forward-period returns as the prediction target.

    Cross-sectional winsorization at 3 std prevents outliers from
    dominating the loss, consistent with the LSTM baseline.

    Args:
        close: Close price DataFrame (date x stock)
        forward_period: Number of days ahead for return calculation (default 10).

    Returns:
        DataFrame, MultiIndex (date, stock), column 'forward_return'
    """
    forward_ret = close.shift(-forward_period) / close - 1

    # Cross-sectional winsorize: clip at 3 std per date
    row_mean = forward_ret.mean(axis=1)
    row_std = forward_ret.std(axis=1)
    lower = row_mean - 3 * row_std
    upper = row_mean + 3 * row_std
    forward_ret = forward_ret.clip(lower=lower, upper=upper, axis=0)

    # dropna=False: keep the full (date, stock) grid. The last forward_period
    # rows are NaN (no future data for the shift); dropping them with the
    # default dropna=True shifts the index and breaks alignment with features.
    panel = forward_ret.stack(dropna=False).to_frame('forward_return')
    panel.index.names = ['date', 'stock']
    return panel


# ===========================================================================
#  Model Training & Prediction
# ===========================================================================

def _create_model(n_estimators: int = 200,
                  max_depth: int = 5,
                  learning_rate: float = 0.05,
                  random_state: int = 42):
    """Create a gradient boosting regressor (XGBoost or sklearn fallback)."""
    if _MODEL_BACKEND == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            # n_jobs=1 (not -1) for bit-reproducible results: under multiple
            # threads XGBoost's histogram aggregation order differs run-to-run,
            # and XGBoost does not guarantee determinism under parallelism.
            n_jobs=1,
            tree_method='hist',
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=random_state,
        )


def _train_predict_oneshot(
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    test_features: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a single model on the training slice and predict on the test slice.

    One-shot approach, consistent with all other baselines. The train/test
    split is centralized in the DataLoader (bundle.train / bundle.test); this
    function consumes the pre-sliced panels directly instead of masking a
    full panel by date.

    Args:
        train_features: MultiIndex (date, stock) feature panel (train slice)
        train_targets: MultiIndex (date, stock) target panel ('forward_return')
        test_features: MultiIndex (date, stock) feature panel (test slice)
        n_estimators, max_depth, learning_rate: Model hyperparameters

    Returns:
        DataFrame (date x stock) of predicted forward returns.
    """
    train_dates = train_features.index.get_level_values('date').unique().sort_values()
    test_dates = test_features.index.get_level_values('date').unique().sort_values()

    if len(test_dates) == 0:
        raise ValueError("No test dates found in the test slice.")
    if len(train_dates) == 0:
        raise ValueError("No train dates found in the train slice.")

    print(f"  Train period: {train_dates[0].date()} to {train_dates[-1].date()} "
          f"({len(train_dates)} days)")
    print(f"  Test period:  {test_dates[0].date()} to {test_dates[-1].date()} "
          f"({len(test_dates)} days)")

    # Merge train features and targets
    train_merged = train_features.join(train_targets, how='inner')
    feature_cols = [c for c in train_merged.columns if c != 'forward_return']

    train_df = train_merged.dropna(subset=['forward_return'])
    train_feature_vals = train_df[feature_cols].fillna(0.0)
    train_target_vals = train_df['forward_return'].values

    if len(train_df) < 100:
        raise ValueError(
            f"Too few valid training rows ({len(train_df)}). "
            f"Check data availability in the train slice."
        )

    print(f"  Training samples: {len(train_df)} rows, {len(feature_cols)} feature(s)")

    model = _create_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(train_feature_vals.values, train_target_vals)
    print(f"  Model trained (one-shot on {len(train_dates)} days)")

    # --- Predict on test period ---
    predictions_list = []
    for d in test_dates:
        try:
            day_features = test_features.xs(d, level='date')
        except KeyError:
            continue
        day_features = day_features.fillna(0.0)
        if len(day_features) == 0:
            continue
        preds = model.predict(day_features[feature_cols].values)
        pred_series = pd.Series(preds, index=day_features.index, name=d)
        predictions_list.append(pred_series)

    if not predictions_list:
        raise ValueError("No predictions generated. Check data availability.")

    predictions = pd.DataFrame(predictions_list)
    predictions.index.name = 'date'
    return predictions


# ===========================================================================
#  Portfolio Construction
# ===========================================================================

def _build_portfolios(predictions: pd.DataFrame,
                      prices: pd.DataFrame,
                      top_n: int = 50,
                      industry: Optional[pd.Series] = None,
                      portfolio_method: str = "score_proportional") -> pd.DataFrame:
    """
    Build long-only equal-weighted portfolios from model predictions.

    Each day, select the top-N stocks by predicted forward return and
    equal-weight them.
    """
    common_stocks = predictions.columns.intersection(prices.columns)
    predictions = predictions[common_stocks]
    prices_aligned = prices.loc[predictions.index, common_stocks]

    portfolio_rows = []
    portfolio_dates = []

    for date in predictions.index:
        scores = predictions.loc[date].dropna()
        if len(scores) == 0:
            continue

        valid_prices = prices_aligned.loc[date].dropna()
        valid_stocks = scores.index.intersection(valid_prices.index)
        scores = scores.loc[valid_stocks]

        if len(scores) == 0:
            continue

        n_select = min(top_n, len(scores))
        top_stocks = scores.nlargest(n_select)

        # MASE-consistent: score-proportional weights (caps applied in helper)
        weights = allocate_portfolio_weights(top_stocks, industry=industry, method=portfolio_method)
        portfolio_rows.append(weights)
        portfolio_dates.append(date)

    if not portfolio_rows:
        raise ValueError("No portfolio rows generated. Check predictions and prices.")

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

    return portfolios


# ===========================================================================
#  Main Entry Point
# ===========================================================================

def run_xgboost_simple(
    config_path: str = "config/config.yaml",
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    universe: Optional[str] = None,
    context_days: int = 30,
    top_n_stocks: int = 50,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    holding_period: int = 1,
    forward_period: int = 10,
    seed: int = 42,
    output_dir: Optional[str] = None,
    portfolio_method: str = "score_proportional",
) -> Dict:
    """
    Run simplified XGBoost baseline (close-price-only) via main DataLoader.

    Pipeline:
    1. Load OHLCV data via DataLoader (only close price is used)
    2. Build single feature: daily close-price return
    3. Build target = forward-period return (winsorized)
    4. Train one model on the entire training period (one-shot, consistent
       with other baselines)
    5. Predict forward returns on test period, rank stocks, select top-N
    6. Backtest with unified BacktestEngine
    """
    print("=" * 60)
    print("  XGBoost Baseline -- Close-Price-Only (via Main DataLoader)")
    print(f"  Backend: {_MODEL_BACKEND}")
    print("=" * 60)

    # -- Step 1: Load data --
    # The full data span is now provided by the DataLoader/DatasetBundle
    # (config-backed); no manual end-date extension is needed here.
    loader = DataLoader(config_path=config_path)

    # -- Resolve forward_period: explicit arg > config.yaml > default 10 --
    if not forward_period or forward_period <= 0:
        forward_period = loader.config.get('evolution', {}).get('forward_period', 10)

    # -- Step 2: Determine train/test split (config-backed) --
    train_start = train_start_date or loader.data_config.get(
        'train_start_date', '2023-01-01')
    train_end = train_end_date or loader.data_config.get(
        'train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get(
        'test_start_date', '2024-01-01')
    test_end = test_end_date or loader.data_config.get(
        'test_end_date', '2025-06-30')

    # DatasetBundle: full span + pre-sliced train/test (split centralized in loader)
    bundle = loader.load_data(universe=universe, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    train_price, train_fund, train_ind = bundle.train
    test_price, test_fund, test_ind = bundle.test
    price_data = bundle.full[0]

    close_extended = price_data['close']
    # Trim close to original test_end_date for backtesting
    if test_end_date:
        close = close_extended[close_extended.index <= pd.Timestamp(test_end_date)]
    else:
        close = close_extended
    print(f"  Loaded (full): {len(close_extended.index)} trading days x "
          f"{len(close_extended.columns)} stocks")
    print(f"  Train: {len(train_price['close'].index)} days, "
          f"Test: {len(test_price['close'].index)} days")
    print(f"  Train end: {train_end}, Test start: {test_start}")

    # -- Step 3: Build features (train & test slices, daily return only) --
    # Built per-slice; the first day's pct_change is NaN and handled by
    # dropna=False inside _build_features + dropna at training time.
    print("\n[Step 3] Building features (daily close-price return)...")
    train_features = _build_features(train_price['close'])
    test_features = _build_features(test_price['close'])
    n_features = len(train_features.columns)
    print(f"  Feature(s): {list(train_features.columns)}")
    print(f"  Train feature panel: {len(train_features)} rows, "
          f"Test: {len(test_features)} rows (date, stock)")

    # -- Step 4: Build targets (train & test slices) --
    print(f"\n[Step 4] Building targets ({forward_period}d forward return)...")
    train_targets = _build_targets(train_price['close'], forward_period=forward_period)
    test_targets = _build_targets(test_price['close'], forward_period=forward_period)
    print(f"  Train target panel: {len(train_targets)} rows, "
          f"Test: {len(test_targets)} rows")

    # -- Step 5: One-shot training and prediction --
    print("\n[Step 5] Training model (one-shot on train period)...")
    predictions = _train_predict_oneshot(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    print(f"  Predictions: {predictions.shape[0]} days x "
          f"{predictions.shape[1]} stocks")

    # -- Step 6: Build portfolios --
    print(f"\n[Step 6] Building portfolios (top-{top_n_stocks} long, score-proportional)...")
    portfolios = _build_portfolios(
        predictions=predictions,
        prices=close,
        top_n=top_n_stocks,
        industry=test_ind,
        portfolio_method=portfolio_method,
    )
    print(f"  Portfolios: {portfolios.shape[0]} days x "
          f"{portfolios.shape[1]} stocks")

    # -- Step 7: Backtest --
    print("\n[Step 7] Running backtest (unified BacktestEngine)...")
    prices_aligned = close.loc[portfolios.index]
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    engine = BacktestEngine(
        commission=0.001,
        slippage=0.0,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    # ── Parameter-tagged run directory ──
    # Layout: experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{method}/
    method_name = "xgboost_simple"
    _u = universe or loader.data_config.get('universe', {}).get('index', 'hs300')
    _s = train_start_date or loader.data_config.get('train_start_date', 'na')
    _e = test_end_date or loader.data_config.get('test_end_date', 'na')
    _fp = forward_period if forward_period is not None else loader.config.get('evolution', {}).get('forward_period', 10)
    _hp = holding_period if holding_period is not None else 1
    param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
    run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
    os.makedirs(run_dir, exist_ok=True)
    _bm = prices_aligned.pct_change().shift(-1).mean(axis=1).dropna()
    _bm.name = 'benchmark_return'
    bt_metrics = engine.run(portfolios, prices_aligned, benchmark_returns=_bm, save_dir=run_dir)

    # -- Step 8: Compute Rank-IC on test set --
    print("\n[Step 8] Computing Rank-IC on test set...")
    # test_targets already holds the test slice (no date masking needed)

    pred_stacked = predictions.stack()
    pred_stacked.index.names = ['date', 'stock']

    daily_ics = []
    for d in predictions.index:
        if d in test_targets.index.get_level_values('date'):
            try:
                pred_d = pred_stacked.xs(d, level='date')
            except KeyError:
                continue
            try:
                actual_d = test_targets.xs(d, level='date')['forward_return']
            except KeyError:
                continue
            common = pred_d.index.intersection(actual_d.index)
            if len(common) > 5:
                pred_rank = pred_d.loc[common].rank()
                actual_rank = actual_d.loc[common].rank()
                ic = pred_rank.corr(actual_rank)
                if not np.isnan(ic):
                    daily_ics.append(ic)

    if daily_ics:
        mean_ic = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics))
        icir = mean_ic / ic_std if ic_std > 0 else 0.0
    else:
        mean_ic = 0.0
        ic_std = 0.0
        icir = 0.0

    print(f"  Mean Rank-IC: {mean_ic:.4f}, ICIR: {icir:.4f}")

    # -- Step 9: Compile results --
    results = {
        'method': 'XGBoost-Simple',
        'backend': _MODEL_BACKEND,
        'feature': 'daily_close_return',
        'n_features': n_features,
        'n_stocks_universe': len(close.columns),
        'top_n_stocks': top_n_stocks,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'mean_rank_ic': mean_ic,
        'icir': icir,
        'annual_return': bt_metrics.get('annual_return', 0.0),
        'sharpe_ratio': bt_metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': bt_metrics.get('max_drawdown', 0.0),
        'information_ratio': bt_metrics.get('information_ratio', 0.0),
        'calmar_ratio': bt_metrics.get('calmar_ratio', 0.0),
        'win_rate': bt_metrics.get('win_rate', 0.0),
        'avg_turnover': bt_metrics.get('avg_turnover', 0.0),
        'annual_volatility': bt_metrics.get('annual_volatility', 0.0),
        'total_return': bt_metrics.get('total_return', 0.0),
        'n_trading_days': bt_metrics.get('n_trading_days', 0),
        'train_end': train_end,
        'test_start': test_start,
        'forward_period': forward_period,
        'train_start': train_start,
        'test_end': test_end,
        'holding_period': holding_period,
    }

    # -- Step 10: Save results --
    if output_dir:
        result_path = os.path.join(run_dir, 'xgboost_simple_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  XGBoost-Simple Baseline Complete")
    print("=" * 60)

    return results


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == '__main__':
    # Seed CLI defaults from config.yaml so --forward-period / --holding-period
    # track evolution.forward_period / backtest.trading.holding_period unless
    # explicitly overridden on the command line.
    _cli_cfg = {}
    try:
        import yaml
        with open('config/config.yaml', encoding='utf-8') as _f:
            _cli_cfg = yaml.safe_load(_f) or {}
    except Exception:
        pass
    _ev = _cli_cfg.get('evolution', {})
    _bt = _cli_cfg.get('backtest', {}).get('trading', {})
    _seed = _cli_cfg.get('seed', 42)

    parser = argparse.ArgumentParser(
        description='Run simplified XGBoost baseline (close-price-only) with main DataLoader')
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
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of boosting rounds')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--holding-period', type=int,
                        default=_bt.get('holding_period', 1),
                        help='Holding period (config: backtest.trading.holding_period)')
    parser.add_argument('--forward-period', type=int,
                        default=_ev.get('forward_period', 10),
                        help='Forward return period in days (config: evolution.forward_period)')
    parser.add_argument('--seed', type=int, default=_seed,
                        help='Random seed for reproducibility (config: seed)')
    parser.add_argument('--output-dir', default='experiments/xgboost_simple',
                        help='Output directory')

    args = parser.parse_args()

    results = run_xgboost_simple(
        config_path=args.config,
        universe=args.universe,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        top_n_stocks=args.top_n,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        holding_period=args.holding_period,
        forward_period=args.forward_period,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Backend:          {results['backend']}")
    print(f"  Feature:          {results['feature']}")
    print(f"  N Features:       {results['n_features']}")
    print(f"  N Estimators:     {results['n_estimators']}")
    print(f"  Max Depth:        {results['max_depth']}")
    print(f"  Total Return:     {results['total_return']:.4f}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
