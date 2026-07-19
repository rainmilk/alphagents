# -*- coding: utf-8 -*-
"""
XGBoost Baseline Runner -- Integrated with Main Dataloader

This runner:
1. Loads A-share data via the main project's DataLoader
2. Constructs technical features from OHLCV + fundamental data
3. Trains a gradient-boosted-tree model (XGBoost or sklearn fallback)
    to predict forward-period cross-sectional return ranks
4. Trains once on the training period (one-shot, consistent with other baselines)
5. Builds long-only top-N portfolios and backtests via BacktestEngine

Usage:
    python baselines/run_xgboost_baseline.py
    python baselines/run_xgboost_baseline.py --start 2020-01-01 --end 2024-12-31 --universe hs300

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# -- Path setup: add project root to sys.path --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.loader import DataLoader
from backtest.engine import BacktestEngine

# -- Model backend: prefer XGBoost, fall back to sklearn --
_MODEL_BACKEND = "unknown"
try:
    import xgboost as xgb
    _MODEL_BACKEND = "xgboost"
    print("[XGBoost Baseline] Using XGBoost backend.")
except ImportError:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        _MODEL_BACKEND = "sklearn"
        print("[XGBoost Baseline] XGBoost not installed, falling back to "
              "sklearn GradientBoostingRegressor.")
    except ImportError:
        raise ImportError(
            "Neither xgboost nor scikit-learn is available. "
            "Install one of them: pip install xgboost  OR  pip install scikit-learn"
        )


# ===========================================================================
#  Feature Engineering
# ===========================================================================

def _build_features(price_data: Dict[str, pd.DataFrame],
                    fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                    ) -> pd.DataFrame:
    """
    Construct cross-sectional technical features from OHLCV data.

    Returns a panel DataFrame with MultiIndex (date, stock) and feature columns.
    All features are computed per-stock (cross-sectional) and aligned to a
    common date index.

    Features (20+):
        - ret_1d, ret_5d, ret_10d, ret_20d       (multi-horizon returns)
        - vol_5d, vol_10d, vol_20d               (rolling volatility)
        - turnover_5d, turnover_20d              (rolling turnover proxy)
        - volume_ratio_5_20                      (short/long volume ratio)
        - rsi_14                                  (relative strength index)
        - ma_5, ma_10, ma_20                      (moving averages)
        - ma_diff_5_20                            (MA crossover signal)
        - price_above_ma5, price_above_ma20       (binary signals)
        - high_low_range_10d                      (price range)
        - close_to_high_20d, close_to_low_20d     (position in range)
        - vwap_dev_5d                             (VWAP deviation)
        - pe, pb, roe (if fundamental data available)

    Args:
        price_data: Dict with keys 'open', 'high', 'low', 'close', 'volume', 'amount'
        fundamental_data: Optional dict with keys like 'pe', 'pb', 'roe'

    Returns:
        DataFrame, MultiIndex (date, stock), columns = feature names
    """
    close = price_data['close']
    open_ = price_data.get('open', close)
    high = price_data.get('high', close)
    low = price_data.get('low', close)
    volume = price_data.get('volume')
    amount = price_data.get('amount')

    # Common aligned date index
    dates = close.index
    stocks = close.columns

    # Stack into MultiIndex (date, stock)
    features = {}

    # -- Returns --
    for horizon in [1, 5, 10, 20]:
        ret = close.pct_change(horizon)
        features[f'ret_{horizon}d'] = ret.stack()

    # -- Volatility (rolling std of daily returns) --
    daily_ret = close.pct_change()
    for window in [5, 10, 20]:
        vol = daily_ret.rolling(window, min_periods=max(3, window // 2)).std()
        features[f'vol_{window}d'] = vol.stack()

    # -- Turnover proxy (volume / rolling mean volume) --
    if volume is not None:
        vol_5 = volume.rolling(5, min_periods=3).mean()
        vol_20 = volume.rolling(20, min_periods=10).mean()
        features['turnover_5d'] = (volume / vol_5).replace(
            [np.inf, -np.inf], np.nan).stack()
        features['turnover_20d'] = (volume / vol_20).replace(
            [np.inf, -np.inf], np.nan).stack()
        features['volume_ratio_5_20'] = (vol_5 / vol_20).replace(
            [np.inf, -np.inf], np.nan).stack()

    # -- RSI (14-day) --
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features['rsi_14'] = rsi.stack()

    # -- Moving averages and crossovers --
    ma_5 = close.rolling(5, min_periods=3).mean()
    ma_10 = close.rolling(10, min_periods=5).mean()
    ma_20 = close.rolling(20, min_periods=10).mean()
    features['ma_5'] = ma_5.stack()
    features['ma_10'] = ma_10.stack()
    features['ma_20'] = ma_20.stack()
    features['ma_diff_5_20'] = ((ma_5 - ma_20) / (ma_20 + 1e-10)).stack()
    features['price_above_ma5'] = (close > ma_5).astype(float).stack()
    features['price_above_ma20'] = (close > ma_20).astype(float).stack()

    # -- Price range features --
    high_20 = high.rolling(20, min_periods=10).max()
    low_20 = low.rolling(20, min_periods=10).min()
    features['high_low_range_10d'] = (
        (high_20 - low_20) / (low_20 + 1e-10)
    ).stack()
    features['close_to_high_20d'] = (
        (high_20 - close) / (high_20 - low_20 + 1e-10)
    ).stack()
    features['close_to_low_20d'] = (
        (close - low_20) / (high_20 - low_20 + 1e-10)
    ).stack()

    # -- VWAP deviation --
    if amount is not None and volume is not None:
        vwap = amount / (volume + 1e-10)
        vwap_ma5 = vwap.rolling(5, min_periods=3).mean()
        features['vwap_dev_5d'] = (
            (close - vwap_ma5) / (vwap_ma5 + 1e-10)
        ).replace([np.inf, -np.inf], np.nan).stack()

    # -- Fundamental features (if available) --
    if fundamental_data:
        for fname in ['pe', 'pb', 'roe', 'market_cap']:
            fund_df = fundamental_data.get(fname)
            if fund_df is not None:
                # Clip extreme values for valuation ratios
                if fname in ('pe', 'pb'):
                    fund_df = fund_df.clip(lower=-500, upper=500)
                features[fname] = fund_df.stack()

    # Combine into single DataFrame
    panel = pd.DataFrame(features)
    panel.index.names = ['date', 'stock']

    # Cross-sectional winsorize each feature per date (clip at 3 sigma)
    # Done in batch for efficiency
    for col in panel.columns:
        grouped = panel.groupby('date')[col]
        med = grouped.transform('median')
        mad = (panel[col] - med).abs().groupby('date').transform('median')
        # MAD-based clipping: values beyond median +/- 3*1.4826*MAD
        clip_bound = 3 * 1.4826 * mad
        panel[col] = panel[col].clip(
            lower=med - clip_bound, upper=med + clip_bound
        )

    # Cross-sectional z-score per date
    for col in panel.columns:
        grouped = panel.groupby('date')[col]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        panel[col] = (panel[col] - mean) / (std + 1e-10)

    return panel


def _build_targets(close: pd.DataFrame, forward_period: int = 10) -> pd.DataFrame:
    """
    Build forward-period cross-sectional rank of returns as the prediction target.

    Using rank (0 to N-1, normalized to [0, 1]) instead of raw returns makes
    the model more robust to outliers and focuses on cross-sectional
    ordering, which is what matters for portfolio construction.

    Args:
        close: Close price DataFrame (date x stock)
        forward_period: Number of days ahead for return calculation (default 10).
            Must match the forward_period used by other baselines for fair comparison.

    Returns:
        DataFrame, MultiIndex (date, stock), column 'target_rank'
    """
    forward_ret = close.shift(-forward_period) / close - 1
    # Cross-sectional rank, normalized to [0, 1]
    target_rank = forward_ret.rank(axis=1, pct=True)
    # dropna=False: keep the full (date, stock) grid. The last forward_period
    # rows are NaN (no future data); dropping them with the default dropna=True
    # shifts the index and breaks alignment with features on merge.
    panel = target_rank.stack(dropna=False).to_frame('target_rank')
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
            n_jobs=-1,
            tree_method='hist',   # Fast histogram-based
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
    features: pd.DataFrame,
    targets: pd.DataFrame,
    test_start_date: str,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Train a single model on all training data and predict on the test period.

    This one-shot approach is consistent with other baselines (AlphaAgent,
    AlphaGrail, AlphaFAMA, MCTS-LLM-Alpha) which also train/select factors
    once on the training period and use them throughout the test period.

    Strategy:
    - Split features/targets at test_start_date
    - Train one model on all pre-test data
    - Predict scores for every day in the test period

    Args:
        features: MultiIndex (date, stock) feature panel
        targets: MultiIndex (date, stock) target panel
        test_start_date: First test date (YYYY-MM-DD)
        n_estimators, max_depth, learning_rate: Model hyperparameters

    Returns:
        DataFrame (date x stock) of predicted scores. Higher = predicted to
        have higher returns.
    """
    test_start_ts = pd.Timestamp(test_start_date)
    all_dates = features.index.get_level_values('date').unique().sort_values()
    test_dates = all_dates[all_dates >= test_start_ts]
    train_dates = all_dates[all_dates < test_start_ts]

    if len(test_dates) == 0:
        raise ValueError(f"No test dates found after {test_start_date}")
    if len(train_dates) == 0:
        raise ValueError(f"No train dates found before {test_start_date}")

    print(f"  Train period: {train_dates[0].date()} to {train_dates[-1].date()} "
          f"({len(train_dates)} days)")
    print(f"  Test period:  {test_dates[0].date()} to {test_dates[-1].date()} "
          f"({len(test_dates)} days)")

    # Merge features and targets
    merged = features.join(targets, how='inner')
    feature_cols = [c for c in merged.columns if c != 'target_rank']

    # --- Train on all pre-test data ---
    train_mask = merged.index.get_level_values('date') < test_start_ts
    train_df = merged[train_mask].dropna(subset=['target_rank'])

    train_feature_vals = train_df[feature_cols].fillna(0.0)
    train_targets = train_df['target_rank'].values

    if len(train_df) < 100:
        raise ValueError(
            f"Too few valid training rows ({len(train_df)}). "
            f"Check data availability before {test_start_date}."
        )

    print(f"  Training samples: {len(train_df)} rows, {len(feature_cols)} features")

    model = _create_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(train_feature_vals.values, train_targets)
    print(f"  Model trained (one-shot on {len(train_dates)} days)")

    # --- Predict on test period ---
    predictions_list = []
    for d in test_dates:
        try:
            day_features = features.xs(d, level='date')
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
                      top_n: int = 50) -> pd.DataFrame:
    """
    Build long-only equal-weighted portfolios from model predictions.

    Each day, select the top-N stocks by predicted score and equal-weight.

    Args:
        predictions: DataFrame (date x stock) of predicted scores
        prices: Close price DataFrame (date x stock)
        top_n: Number of stocks to hold (default 50)

    Returns:
        DataFrame (date x stock) of portfolio weights, each row sums to 1.
    """
    # Align columns
    common_stocks = predictions.columns.intersection(prices.columns)
    predictions = predictions[common_stocks]
    prices_aligned = prices.loc[predictions.index, common_stocks]

    portfolio_rows = []
    portfolio_dates = []

    for date in predictions.index:
        scores = predictions.loc[date].dropna()
        if len(scores) == 0:
            continue

        # Filter to stocks that have valid prices on this date
        valid_prices = prices_aligned.loc[date].dropna()
        valid_stocks = scores.index.intersection(valid_prices.index)
        scores = scores.loc[valid_stocks]

        if len(scores) == 0:
            continue

        n_select = min(top_n, len(scores))
        top_stocks = scores.nlargest(n_select)

        # Equal weight
        weights = pd.Series(1.0 / n_select, index=top_stocks.index)
        portfolio_rows.append(weights)
        portfolio_dates.append(date)

    if not portfolio_rows:
        raise ValueError("No portfolio rows generated. Check predictions and prices.")

    # Build full DataFrame
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

def run_xgboost_baseline(
    config_path: str = "config/config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    context_days: int = 30,
    top_n_stocks: int = 50,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    holding_period: int = 1,
    forward_period: int = 10,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run XGBoost baseline using the main project's DataLoader and BacktestEngine.

    Pipeline:
    1. Load OHLCV + fundamental data via DataLoader
    2. Build 20+ technical features per (date, stock)
    3. Build target = forward-period cross-sectional return rank
    4. Train one model on the entire training period (one-shot, consistent
       with other baselines)
    5. Predict scores on test period, select top-N stocks
    6. Backtest with unified BacktestEngine

    Args:
        config_path: Path to the main project config file.
        start_date: Data start date (YYYY-MM-DD).
        end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        context_days: Ignored (kept for API compatibility with other baselines).
        top_n_stocks: Number of stocks in long portfolio (default 50).
        n_estimators: Number of boosting rounds.
        max_depth: Max tree depth.
        learning_rate: Boosting learning rate.
        holding_period: Rebalance frequency for backtest (1=daily).
        forward_period: Forward return period in days (default 10). Must align
            with other baselines for fair comparison.
        output_dir: Directory for saving results.

    Returns:
        Dict of performance metrics.
    """
    print("=" * 60)
    print("  XGBoost Baseline -- A-Share (via Main DataLoader)")
    print(f"  Backend: {_MODEL_BACKEND}")
    print("=" * 60)

    # -- Step 1: Load data via main DataLoader --
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )

    close = price_data['close']
    print(f"  Loaded: {len(close.index)} trading days x "
          f"{len(close.columns)} stocks")

    # -- Step 2: Determine train/test split --
    print("\n[Step 2] Determining train/test split...")
    train_end = train_end_date or loader.data_config.get(
        'train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get(
        'test_start_date', '2024-01-01')
    print(f"  Train end: {train_end}, Test start: {test_start}")

    # -- Step 3: Build features --
    print("\n[Step 3] Building technical features...")
    features = _build_features(price_data, fundamental_data)
    n_features = len(features.columns)
    print(f"  Built {n_features} features: {list(features.columns)}")
    print(f"  Feature panel: {len(features)} rows (date, stock)")

    # -- Step 4: Build targets --
    print(f"\n[Step 4] Building prediction targets ({forward_period}d forward return rank)...")
    targets = _build_targets(close, forward_period=forward_period)
    print(f"  Target panel: {len(targets)} rows")

    # -- Step 5: One-shot training and prediction --
    print("\n[Step 5] Training model (one-shot on train period)...")
    predictions = _train_predict_oneshot(
        features=features,
        targets=targets,
        test_start_date=test_start,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    print(f"  Predictions: {predictions.shape[0]} days x "
          f"{predictions.shape[1]} stocks")

    # -- Step 6: Build portfolios --
    print(f"\n[Step 6] Building portfolios (top-{top_n_stocks} long, equal-weight)...")
    portfolios = _build_portfolios(
        predictions=predictions,
        prices=close,
        top_n=top_n_stocks,
    )
    print(f"  Portfolios: {portfolios.shape[0]} days x "
          f"{portfolios.shape[1]} stocks")

    # -- Step 7: Backtest with unified BacktestEngine --
    print("\n[Step 7] Running backtest (unified BacktestEngine)...")
    prices_aligned = close.loc[portfolios.index]
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    # ── Parameter-tagged, date-isolated run directory ──
    # Layout: experiments/{method}/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/{YYYYMMDD}/
    method_name = "alpha_xgboost"
    _u = universe or loader.data_config.get('universe', {}).get('index', 'hs300')
    _s = start_date or loader.data_config.get('universe', {}).get('start_date', 'na')
    _e = end_date or loader.data_config.get('universe', {}).get('end_date', 'na')
    _fp = forward_period if forward_period is not None else 10
    _hp = holding_period if holding_period is not None else 1
    param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
    run_dir = os.path.join(os.path.dirname(output_dir), param_dir)
    os.makedirs(run_dir, exist_ok=True)
    bt_metrics = engine.run(portfolios, prices_aligned, save_dir=run_dir, method_prefix=method_name)

    # -- Step 8: Compute Rank-IC on test set --
    print("\n[Step 8] Computing Rank-IC on test set...")
    test_start_ts = pd.Timestamp(test_start)
    test_features = features[features.index.get_level_values('date') >= test_start_ts]
    test_targets = targets[targets.index.get_level_values('date') >= test_start_ts]

    # Align predictions with targets for IC computation
    pred_stacked = predictions.stack()
    pred_stacked.index.names = ['date', 'stock']

    # Compute daily Rank-IC (Spearman correlation between predicted score
    # and actual forward-period return rank)
    daily_ics = []
    for d in predictions.index:
        if d in test_targets.index.get_level_values('date'):
            pred_d = pred_stacked.xs(d, level='date') if isinstance(
                pred_stacked.index, pd.MultiIndex) else pred_stacked.loc[d]
            actual_d = test_targets.xs(d, level='date')['target_rank']
            common = pred_d.index.intersection(actual_d.index)
            if len(common) > 5:
                pred_rank = pred_d.loc[common].rank()
                actual_rank = actual_d.loc[common].rank()
                # Spearman = Pearson of ranks
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
        'method': 'XGBoost',
        'backend': _MODEL_BACKEND,
        'n_features': n_features,
        'feature_names': list(features.columns),
        'n_stocks_universe': len(close.columns),
        'top_n_stocks': top_n_stocks,
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
    }

    # -- Step 10: Save results --
    if output_dir:
        result_path = os.path.join(run_dir, 'xgboost_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  XGBoost Baseline Complete")
    print("=" * 60)

    return results


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run XGBoost baseline with main DataLoader')
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
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of boosting rounds')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--holding-period', type=int, default=1,
                        help='Holding period (1=daily, 5=weekly)')
    parser.add_argument('--forward-period', type=int, default=10,
                        help='Forward return period in days (must align with other baselines)')
    parser.add_argument('--output-dir', default='experiments/xgboost',
                        help='Output directory')

    args = parser.parse_args()

    results = run_xgboost_baseline(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        top_n_stocks=args.top_n,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        holding_period=args.holding_period,
        forward_period=args.forward_period,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Backend:          {results['backend']}")
    print(f"  N Features:       {results['n_features']}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
