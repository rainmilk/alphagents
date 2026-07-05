# -*- coding: utf-8 -*-
"""
LSTM Baseline Runner -- Close-Price-Only

This runner:
1. Loads A-share data via the main project's DataLoader
2. Uses daily close-price returns as the sole feature (no factor engineering)
3. Uses an LSTM model to predict forward-period returns
4. Trains once on the training period (one-shot, consistent with other baselines)
5. Ranks stocks by predicted forward return, selects top-N for portfolio
6. Backtests via unified BacktestEngine

Usage:
    python baselines/run_lstm_baseline.py
    python baselines/run_lstm_baseline.py --start 2020-01-01 --end 2024-12-31 --universe hs300

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

# -- Path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.loader import DataLoader
from backtest.engine import BacktestEngine

# -- PyTorch --
import torch
import torch.nn as nn

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[LSTM Baseline] Using device: {_DEVICE}")


# ===========================================================================
#  Feature & Target
# ===========================================================================

def _build_features(close: pd.DataFrame) -> pd.DataFrame:
    """
    Build the single feature: daily close-price return.

    The LSTM consumes a sequence of daily returns (seq_len days) to predict
    the forward-period return. No technical indicators or factors are used --
    the model must learn price-momentum patterns directly from raw returns.

    Args:
        close: Close price DataFrame (date x stock)

    Returns:
        DataFrame, MultiIndex (date, stock), single column 'daily_return'
    """
    daily_ret = close.pct_change()
    panel = daily_ret.stack().to_frame('daily_return')
    panel.index.names = ['date', 'stock']
    return panel


def _build_targets(close: pd.DataFrame, forward_period: int = 10) -> pd.DataFrame:
    """
    Build forward-period returns as the prediction target.

    Cross-sectional winsorization at 3 std prevents outliers from
    dominating the MSE loss.

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

    panel = forward_ret.stack().to_frame('forward_return')
    panel.index.names = ['date', 'stock']
    return panel


# ===========================================================================
#  LSTM Data Preparation
# ===========================================================================

def _prepare_lstm_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    all_stocks: pd.Index,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert feature/target panels into LSTM-ready sequences.

    For each (date_t, stock_s) pair where t >= seq_len-1:
      X = features[date_{t-seq_len+1} : date_t, stock_s, :]
      y = target[date_t, stock_s]

    Missing values within a sequence are forward-filled per stock, then
    filled with 0.0.

    Args:
        features: MultiIndex (date, stock) feature panel (single column)
        targets: MultiIndex (date, stock) target panel with 'forward_return'
        all_dates: sorted unique dates
        all_stocks: sorted unique stocks
        seq_len: lookback window length

    Returns:
        X: (n_samples, seq_len, n_features) float32
        y: (n_samples,) float32
        sample_dates: (n_samples,) array of Timestamps
        sample_stocks: (n_samples,) array of stock labels
    """
    feature_cols = features.columns.tolist()
    n_features = len(feature_cols)
    n_dates = len(all_dates)
    n_stocks = len(all_stocks)

    print(f"  Preparing LSTM data: {n_dates} dates x {n_stocks} stocks "
          f"x {n_features} feature(s), seq_len={seq_len}")

    # -- Build 3D feature array: (date, stock, feature) --
    features_3d = np.zeros((n_dates, n_stocks, n_features), dtype=np.float32)

    for f_idx, col in enumerate(feature_cols):
        wide = features[col].unstack(level='stock')
        wide = wide.reindex(index=all_dates, columns=all_stocks)
        wide = wide.ffill().fillna(0.0)
        features_3d[:, :, f_idx] = wide.values

    # -- Build 2D target array: (date, stock) --
    target_wide = targets['forward_return'].unstack(level='stock')
    target_wide = target_wide.reindex(index=all_dates, columns=all_stocks)
    targets_2d = target_wide.values  # (n_dates, n_stocks)

    # -- Create sequences --
    X_list = []
    y_list = []
    date_list = []
    stock_list = []
    stocks_arr = np.array(all_stocks)

    for t in range(seq_len - 1, n_dates):
        # Sequence for all stocks: (seq_len, n_stocks, n_features)
        # -> transpose to (n_stocks, seq_len, n_features)
        X_t = features_3d[t - seq_len + 1: t + 1, :, :].transpose(1, 0, 2)
        y_t = targets_2d[t, :]  # (n_stocks,)

        # Filter: only keep samples with valid (non-NaN) targets
        valid_mask = ~np.isnan(y_t)
        if valid_mask.sum() == 0:
            continue

        X_list.append(X_t[valid_mask])
        y_list.append(y_t[valid_mask])
        date_list.append(np.full(valid_mask.sum(), all_dates[t]))
        stock_list.append(stocks_arr[valid_mask])

    if not X_list:
        raise ValueError(
            "No valid sequences generated. Check data availability and seq_len."
        )

    X = np.concatenate(X_list, axis=0)      # (n_samples, seq_len, n_features)
    y = np.concatenate(y_list, axis=0)      # (n_samples,)
    sample_dates = np.concatenate(date_list)
    sample_stocks = np.concatenate(stock_list)

    print(f"  Prepared {len(y)} samples "
          f"({len(y) // n_stocks if n_stocks > 0 else 0} dates x ~{n_stocks} stocks)")

    return X, y, sample_dates, sample_stocks


# ===========================================================================
#  LSTM Model
# ===========================================================================

class LSTMModel(nn.Module):
    """
    LSTM-based forward return predictor.

    Architecture:
        Input (batch, seq_len, n_features)
        -> LSTM (num_layers, hidden_size, dropout)
        -> Take last time step
        -> FC (hidden -> hidden/2) + ReLU + Dropout
        -> FC (hidden/2 -> 1)
        -> Squeeze -> (batch,)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last time step: (batch, hidden_size)
        out = self.head(out)      # (batch, 1)
        return out.squeeze(-1)    # (batch,)


# ===========================================================================
#  Training & Prediction
# ===========================================================================

def _train_predict_oneshot_lstm(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    test_start_date: str,
    seq_len: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 0.001,
) -> pd.DataFrame:
    """
    Train a single LSTM on all training data and predict on the test period.

    One-shot approach, consistent with other baselines.

    Args:
        features: MultiIndex (date, stock) feature panel
        targets: MultiIndex (date, stock) target panel
        test_start_date: First test date (YYYY-MM-DD)
        seq_len: Lookback window in trading days
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Adam learning rate

    Returns:
        DataFrame (date x stock) of predicted forward returns.
    """
    test_start_ts = pd.Timestamp(test_start_date)
    all_dates = features.index.get_level_values('date').unique().sort_values()
    all_stocks = features.index.get_level_values('stock').unique().sort_values()

    train_dates = all_dates[all_dates < test_start_ts]
    test_dates = all_dates[all_dates >= test_start_ts]

    if len(test_dates) == 0:
        raise ValueError(f"No test dates found after {test_start_date}")
    if len(train_dates) == 0:
        raise ValueError(f"No train dates found before {test_start_date}")

    print(f"  Train period: {train_dates[0].date()} to {train_dates[-1].date()} "
          f"({len(train_dates)} days)")
    print(f"  Test period:  {test_dates[0].date()} to {test_dates[-1].date()} "
          f"({len(test_dates)} days)")

    # -- Prepare sequences for train and test --
    print("\n  Preparing training sequences...")
    train_features = features[features.index.get_level_values('date') < test_start_ts]
    train_targets = targets[targets.index.get_level_values('date') < test_start_ts]
    X_train, y_train, _, _ = _prepare_lstm_data(
        train_features, train_targets, train_dates, all_stocks, seq_len,
    )

    # Drop any remaining NaN in targets
    valid = ~np.isnan(y_train)
    X_train = X_train[valid]
    y_train = y_train[valid]

    if len(X_train) < 100:
        raise ValueError(
            f"Too few valid training samples ({len(X_train)}). "
            f"Check data availability and seq_len."
        )

    n_features = X_train.shape[2]
    print(f"  Training samples: {len(X_train)}, features: {n_features}")

    # -- Train LSTM --
    print(f"\n  Training LSTM (epochs={epochs}, batch_size={batch_size}, "
          f"lr={learning_rate}, hidden={hidden_size}, layers={num_layers})...")

    model = LSTMModel(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_train).to(_DEVICE)
    y_tensor = torch.FloatTensor(y_train).to(_DEVICE)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tensor))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            batch_idx = perm[i:i + batch_size]
            X_batch = X_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs}, "
                  f"Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("  LSTM training complete.")

    # -- Predict on test period --
    # Include seq_len days of training data before test_start so the LSTM
    # can form sequences for ALL test dates (including the first one).
    # Without this context, the first seq_len-1 test days would be skipped,
    # causing incomplete test coverage.
    print("\n  Preparing test sequences (with context from train period)...")
    test_start_pos = all_dates.get_loc(test_dates[0])
    context_start_pos = max(0, test_start_pos - seq_len + 1)
    context_dates = all_dates[context_start_pos:]

    context_features = features[features.index.get_level_values('date').isin(context_dates)]
    context_targets = targets[targets.index.get_level_values('date').isin(context_dates)]
    X_test, _, test_sample_dates, test_sample_stocks = _prepare_lstm_data(
        context_features, context_targets, context_dates, all_stocks, seq_len,
    )

    # Filter to only test dates (drop context days from predictions)
    test_mask = np.array([d >= test_start_ts for d in test_sample_dates])
    X_test = X_test[test_mask]
    test_sample_dates = test_sample_dates[test_mask]
    test_sample_stocks = test_sample_stocks[test_mask]

    print(f"  Predicting on {len(X_test)} test samples...")
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i:i + batch_size]).to(_DEVICE)
            pred = model(X_batch)
            predictions_list.append(pred.cpu().numpy())

    all_preds = np.concatenate(predictions_list)

    # -- Map predictions back to (date, stock) DataFrame --
    pred_df = pd.DataFrame({
        'date': test_sample_dates,
        'stock': test_sample_stocks,
        'pred': all_preds,
    })
    predictions = pred_df.pivot_table(
        index='date', columns='stock', values='pred'
    )
    predictions.index.name = 'date'

    print(f"  Predictions: {predictions.shape[0]} days x "
          f"{predictions.shape[1]} stocks")

    return predictions


# ===========================================================================
#  Portfolio Construction
# ===========================================================================

def _build_portfolios(predictions: pd.DataFrame,
                      prices: pd.DataFrame,
                      top_n: int = 50) -> pd.DataFrame:
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

        weights = pd.Series(1.0 / n_select, index=top_stocks.index)
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

def run_lstm_baseline(
    config_path: str = "config/config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    context_days: int = 30,
    top_n_stocks: int = 50,
    seq_len: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 0.001,
    holding_period: int = 1,
    forward_period: int = 10,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run LSTM baseline using the main project's DataLoader and BacktestEngine.

    Pipeline:
    1. Load OHLCV data via DataLoader (only close price is used)
    2. Build single feature: daily close-price return
    3. Build target = forward-period return (winsorized)
    4. Train one LSTM on the entire training period (one-shot, consistent
       with other baselines)
    5. Predict forward returns on test period, rank stocks, select top-N
    6. Backtest with unified BacktestEngine
    """
    print("=" * 60)
    print("  LSTM Baseline -- Close-Price-Only (via Main DataLoader)")
    print(f"  Device: {_DEVICE}")
    print("=" * 60)

    # -- Step 1: Load data --
    # Extend data loading by context_days to ensure forward return targets
    # are valid for ALL test dates. Without the extension, the last
    # forward_period days of test data have NaN targets because
    # shift(-forward_period) goes beyond the loaded data.
    from datetime import timedelta
    buffer_calendar_days = int(max(context_days, forward_period) * 2) + 10
    if end_date:
        extended_end = (pd.Timestamp(end_date) + timedelta(days=buffer_calendar_days)).strftime('%Y-%m-%d')
    else:
        extended_end = None

    print("\n[Step 1] Loading data via main DataLoader...")
    if extended_end and extended_end != end_date:
        print(f"  Extending end_date by {context_days}d context: "
              f"{end_date} -> {extended_end}")
    loader = DataLoader(config_path=config_path)
    price_data, _, _ = loader.load_data(
        start_date=start_date,
        end_date=extended_end,
        universe=universe,
    )

    close_extended = price_data['close']
    # Trim close to original end_date for backtesting
    if end_date:
        close = close_extended[close_extended.index <= pd.Timestamp(end_date)]
    else:
        close = close_extended
    print(f"  Loaded: {len(close_extended.index)} trading days (extended) x "
          f"{len(close_extended.columns)} stocks")
    print(f"  Test/backtest period: {len(close.index)} trading days")

    # -- Step 2: Determine train/test split --
    print("\n[Step 2] Determining train/test split...")
    train_end = train_end_date or loader.data_config.get(
        'train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get(
        'test_start_date', '2024-01-01')
    print(f"  Train end: {train_end}, Test start: {test_start}")

    # -- Step 3: Build features (daily return only) --
    # Compute on extended close (so first day's pct_change is valid),
    # then trim to original period for train/test
    print("\n[Step 3] Building features (daily close-price return)...")
    features = _build_features(close_extended)
    if end_date:
        end_ts = pd.Timestamp(end_date)
        features = features[features.index.get_level_values('date') <= end_ts]
    n_features = len(features.columns)
    print(f"  Feature(s): {list(features.columns)}")
    print(f"  Feature panel: {len(features)} rows (date, stock)")

    # -- Step 4: Build targets --
    # Compute on extended close so forward returns are valid for all test
    # dates up to end_date, then trim
    print(f"\n[Step 4] Building targets ({forward_period}d forward return)...")
    targets = _build_targets(close_extended, forward_period=forward_period)
    if end_date:
        targets = targets[targets.index.get_level_values('date') <= end_ts]
    print(f"  Target panel: {len(targets)} rows")

    # -- Step 5: One-shot LSTM training and prediction --
    print("\n[Step 5] Training LSTM (one-shot on train period)...")
    predictions = _train_predict_oneshot_lstm(
        features=features,
        targets=targets,
        test_start_date=test_start,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
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

    # -- Step 7: Backtest --
    print("\n[Step 7] Running backtest (unified BacktestEngine)...")
    prices_aligned = close.loc[portfolios.index]
    prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    bt_metrics = engine.run(portfolios, prices_aligned)

    # -- Step 8: Compute Rank-IC on test set --
    print("\n[Step 8] Computing Rank-IC on test set...")
    test_start_ts = pd.Timestamp(test_start)
    test_targets = targets[targets.index.get_level_values('date') >= test_start_ts]

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
        'method': 'LSTM',
        'device': str(_DEVICE),
        'feature': 'daily_close_return',
        'n_features': n_features,
        'n_stocks_universe': len(close.columns),
        'top_n_stocks': top_n_stocks,
        'seq_len': seq_len,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'epochs': epochs,
        'batch_size': batch_size,
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
    }

    # -- Step 10: Save results --
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, 'lstm_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  LSTM Baseline Complete")
    print("=" * 60)

    return results


# ===========================================================================
#  CLI
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run LSTM baseline (close-price-only) with main DataLoader')
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
    parser.add_argument('--seq-len', type=int, default=20,
                        help='LSTM lookback window (trading days)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--holding-period', type=int, default=1,
                        help='Holding period (1=daily, 5=weekly)')
    parser.add_argument('--forward-period', type=int, default=10,
                        help='Forward return period in days (must align with other baselines)')
    parser.add_argument('--output-dir', default='experiments/lstm',
                        help='Output directory')

    args = parser.parse_args()

    results = run_lstm_baseline(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        top_n_stocks=args.top_n,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        holding_period=args.holding_period,
        forward_period=args.forward_period,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Device:           {results['device']}")
    print(f"  Feature:          {results['feature']}")
    print(f"  N Features:       {results['n_features']}")
    print(f"  Seq Length:       {results['seq_len']}")
    print(f"  Hidden Size:      {results['hidden_size']}")
    print(f"  Num Layers:       {results['num_layers']}")
    print(f"  Epochs:           {results['epochs']}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
