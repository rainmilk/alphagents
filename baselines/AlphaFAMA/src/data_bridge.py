# -*- coding: utf-8 -*-
"""
Data bridge between the main project's DataLoader and AlphaFAMA's expected format.

The main DataLoader returns:
    price_data:  Dict[str, DataFrame]  — keys: open, high, low, close, volume, amount
                 Each DataFrame: (n_dates, n_stocks) — date index, stock columns
    fundamental_data: Dict[str, DataFrame]
    industry_series: Series, index=stock codes

AlphaFAMA expects:
    A single DataFrame with MultiIndex (date, ticker) and columns:
        open, high, low, close, volume, vwap, returns, vol
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np


def convert_price_data_to_alphafama(
    price_data: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert the main project's price_data dict to AlphaFAMA's MultiIndex DataFrame.

    Args:
        price_data: Dict with keys ['open', 'high', 'low', 'close', 'volume', 'amount'].
                    Each value is a DataFrame with date index and stock columns.
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).

    Returns:
        A DataFrame with MultiIndex (date, ticker) and columns:
            open, high, low, close, volume, vwap, returns, vol
    """
    close_df = price_data['close']

    # --- Convert each OHLCV field from wide to long format ---
    parts = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        if field in price_data:
            df = price_data[field].copy()
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            # stack: (date, ticker) MultiIndex -> Series with field name
            stacked = df.stack()
            stacked.name = field
            parts[field] = stacked

    # Combine into a single DataFrame
    result = pd.DataFrame(parts)
    # Explicitly name the MultiIndex levels: (date, ticker)
    result.index.names = ['date', 'ticker']

    # --- Drop rows where close is NaN (no trading) ---
    result = result.dropna(subset=['close'])

    # --- Compute VWAP (volume-weighted average price) ---
    if 'vwap' not in result.columns:
        result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0

    # --- Compute vol (alias for volume, as AlphaFAMA expects) ---
    result['vol'] = result['volume']

    # --- Compute daily returns per ticker ---
    result['returns'] = result.groupby('ticker', group_keys=False)['close'].pct_change()

    # --- Drop rows with NaN returns (first row of each ticker) ---
    result = result.dropna(subset=['returns'])

    # --- Forward/back fill remaining gaps ---
    result = result.ffill().bfill()

    # --- Select and order columns as AlphaFAMA expects ---
    expected_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'returns', 'vol']
    result = result[expected_cols]

    return result


def split_alphafama_data(
    df: pd.DataFrame,
    train_end_date: str,
    test_start_date: str,
    context_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split AlphaFAMA-format data into train and test sets.

    Args:
        df: AlphaFAMA-format DataFrame with MultiIndex (date, ticker).
        train_end_date: Last date of training period (YYYY-MM-DD).
        test_start_date: First date of test period (YYYY-MM-DD).
        context_days: Number of days to include in test set for rolling window context.

    Returns:
        Tuple of (train_df, test_df) in AlphaFAMA format.
    """
    train_end = pd.Timestamp(train_end_date)
    test_start = pd.Timestamp(test_start_date)

    dates = df.index.get_level_values('date').unique().sort_values()

    # Find split index
    train_dates = dates[dates <= train_end]

    # Test data: from context window before test_start to end
    context_start_idx = max(0, len(train_dates) - context_days)
    if context_start_idx < len(dates):
        test_dates = dates[context_start_idx:]
    else:
        test_dates = dates[len(train_dates):]

    train_df = df.loc[df.index.get_level_values('date').isin(train_dates)]
    test_df = df.loc[df.index.get_level_values('date').isin(test_dates)]

    print(f"  [bridge] Train: {train_dates[0].strftime('%Y-%m-%d')} → "
          f"{train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days)")
    print(f"  [bridge] Test:  {test_dates[0].strftime('%Y-%m-%d')} → "
          f"{test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} days, "
          f"includes ~{context_days}-day context)")

    return train_df, test_df
