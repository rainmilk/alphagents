# src/data_fetch.py
import pandas as pd
from pandas_datareader.data import DataReader
from tqdm import tqdm
from pathlib import Path

# Offline guard: block live data access outside the unified load_datasets.py
# flow. See dataloader/_offline_guard.py. Opt in with MASE_ALLOW_LEGACY_FETCH=1.
import os
import sys


def _mase_repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "load_datasets.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot locate MASE repo root (load_datasets.py).")


sys.path.insert(0, _mase_repo_root())
from dataloader._offline_guard import assert_offline_or_optin as _mase_offline_guard


def download_spx_spy(start_date: str, end_date: str, output_path: str):
    """
    Download S&P 500 index (^SPX) and SPY ETF OHLCV from Stooq,
    then write a single Parquet at output_path.
    """
    _mase_offline_guard("baselines/AlphaFAMA/src/data_fetch.py::download_spx_spy (Stooq via pandas_datareader)")
    tickers = {
        "^SPX": "^SPX",  # Stooq uses ^SPX for the S&P 500 index
        "SPY": "SPY"
    }
    parts = []

    for label, symbol in tqdm(tickers.items(), desc="Fetching from Stooq"):
        try:
            df = DataReader(symbol, "stooq", start=start_date, end=end_date)
        except Exception as e:
            tqdm.write(f"⚠️  Failed to fetch {label} ({symbol}): {e}")
            continue

        if df.empty:
            tqdm.write(f"⚠️  No data for {label}, skipping.")
            continue

        # Stooq returns descending dates—sort ascending
        df = df.sort_index()

        # Keep & lowercase the columns you need
        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        df["ticker"] = label
        df.index.name = "date"
        parts.append(df)

    if not parts:
        raise RuntimeError(
            "❌ No market data was fetched for SPX or SPY from Stooq."
        )

    # Concatenate, re-index, write out
    out = (
        pd.concat(parts)
          .reset_index()
          .set_index(["date", "ticker"])
    )
    # ensure parent dirs exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    print(f"✅ Market data saved to {output_path}")
