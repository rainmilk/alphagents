# -*- coding: utf-8 -*-
"""
Data Loader Module

This module handles:
1. Loading stock data from various sources (Tushare, AkShare, local files, westock)
2. Preprocessing (missing values, outliers, normalization)
3. Feature engineering (technical indicators, fundamental features)
4. Train/test splitting

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
import warnings

from config import config_path
import config as global_config

warnings.filterwarnings('ignore')


def _save_cache(cache_file: str, price_data, fundamental_data, industry_data):
    """Persist loaded data to a pickle cache. Skips if price data is all-NaN."""
    close = price_data.get('close')
    if close is not None and close.isna().all().all():
        print(f"  [cache] Refusing to cache all-NaN price data")
        return
    with open(cache_file, 'wb') as f:
        pd.to_pickle((price_data, fundamental_data, industry_data), f)


def _load_cache(cache_file: str):
    """Load data from a pickle cache. Returns None if cache doesn't exist."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pd.read_pickle(f)
    return None


# ---------------------------------------------------------------------------
# Local dataset store (pre-fetched, queryable by universe + date range)
# ---------------------------------------------------------------------------
# Per the project workflow, raw market data is fetched ONCE by the standalone
# `load_datasets.py` CLI (which calls load_real_data + preprocess and persists
# the result here). Every other entry point (MASE main.py, the 9 baselines)
# then RETRIEVES the requested (universe, start_date, end_date) slice from this
# local store via DataLoader.load_data() — no network access at experiment time.
DATASETS_DIR = "datasets"


def dataset_path(universe: str, start_date: str = None, end_date: str = None) -> str:
    """Canonical path of a pre-fetched dataset archive.

    The filename encodes the exact (universe, start_date, end_date) triple so
    that retrieval is an exact-key lookup — no date slicing needed.
    """
    file_name = f"{universe}.pkl" if end_date is None else f"{universe}_{start_date}_{end_date}.pkl"
    return os.path.join(DATASETS_DIR, file_name)


def _slice_dataset(triple, start: str, end: str):
    """Slice a (price_data, fundamental_data, industry_data) triple to [start, end].

    price_data / fundamental_data are dicts of date-indexed DataFrames; they are
    cropped with pandas label-based `.loc[start:end]` (inclusive on both ends).
    industry_data is time-invariant (indexed by stock code), so it is returned
    unchanged.
    """
    price_data, fundamental_data, industry_data = triple
    sliced_price = {k: v.loc[start:end] for k, v in price_data.items()}
    sliced_fund = (
        {k: v.loc[start:end] for k, v in fundamental_data.items()}
        if fundamental_data else {}
    )
    return sliced_price, sliced_fund, industry_data


@dataclass
class DatasetBundle:
    """Returned by retrieve_dataset() / DataLoader.load_data().

    Holds the full pre-fetched triple plus train/test slices cut from it.
    Each component is a (price_data, fundamental_data, industry_data) triple:
      - price_data:       dict of DataFrames [open, high, low, close, volume, amount]
      - fundamental_data: dict of DataFrames (empty dict if unavailable)
      - industry_data:    Series indexed by stock code (time-invariant)

    Convenience properties (.price_data / .fundamental_data / .industry_data)
    expose the FULL triple's elements so legacy code that only needs the full
    span keeps working.
    """
    full: Tuple
    train: Tuple
    test: Tuple

    @property
    def price_data(self):
        return self.full[0]

    @property
    def fundamental_data(self):
        return self.full[1]

    @property
    def industry_data(self):
        return self.full[2]


def retrieve_dataset(
    universe: str,
    train_start: str = None,
    train_end: str = None,
    test_start: str = None,
    test_end: str = None,
    full_start: str = None,
    full_end: str = None,
):
    """Load a pre-fetched dataset from the local store and slice train/test.

    The archive (datasets/{universe}_{full_start}_{full_end}.pkl) is produced
    once by `python load_datasets.py --universe ... --start-date ... --end-date ...`
    and contains the FULL data span. This function retrieves it and cuts out the
    (train_start, train_end) and (test_start, test_end) windows on demand, so
    every experiment shares one pre-fetched archive and a single split location.

    Args:
        universe, full_start, full_end: identify the pre-fetched archive
            (must match what `load_datasets.py` stored).
        train_start, train_end: training window (YYYY-MM-DD, inclusive).
            Defaults to the full span when omitted.
        test_start, test_end: test window (YYYY-MM-DD, inclusive).
            Defaults to the full span when omitted.
        (Pass the project's train_end_date / test_start_date from config to get
         a real train/test split; omit them to get the full span in both.)

    Returns:
        DatasetBundle with .full / .train / .test, each a
        (price_data, fundamental_data, industry_data) triple.

    Raises:
        FileNotFoundError: if no matching archive exists, with the exact
            `load_datasets.py` command to run.
    """
    path = dataset_path(universe, full_start, full_end)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Local dataset not found: {path}\n"
            f"  → Pre-fetch the FULL span once with:\n"
            f"      python load_datasets.py --universe {universe} "
            f"--start-date {full_start} --end-date {full_end}\n"
            f"  (retrieve_dataset() only slices from this local store and "
            f"never hits the network)"
        )
    with open(path, 'rb') as f:
        full = pd.read_pickle(f)

    # Resolve train/test windows. When a bound is omitted we fall back to the
    # full span so callers that don't care about splitting still get a usable
    # bundle (train/test == full).
    train_start = train_start or full_start
    train_end = train_end or full_end
    test_start = test_start or full_start
    test_end = test_end or full_end

    train = _slice_dataset(full, train_start, train_end)
    test = _slice_dataset(full, test_start, test_end)
    return DatasetBundle(full=full, train=train, test=test)


# ---------------------------------------------------------------------------
# Raw data persistence helpers
# ---------------------------------------------------------------------------
def _save_raw_csv(df: pd.DataFrame, relpath: str, index: bool = True):
    """Save a DataFrame as CSV to data/raw/... with utf-8-sig encoding."""
    raw_data_dir = config_path('data', 'raw')
    full_path = os.path.join(raw_data_dir, relpath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_csv(full_path, index=index, encoding='utf-8-sig')
    n_rows, n_cols = df.shape
    print(f"  [raw] Saved {full_path}  ({n_rows} rows x {n_cols} cols)")

def _generate_column_mapping_json(output_path: str = "data/column_mapping.json"):
    """
    Generate a comprehensive JSON file documenting the field mapping
    between each data source's raw column names and the project's
    standardised field names.
    """
    mapping = {
        "description": (
            "A-share multi-source data field mapping. "
            "Documents how each data provider's raw column names map to "
            "the project's canonical field names used in price_data / "
            "fundamental_data / industry_series."
        ),
        "version": "1.0",
        "canonical_fields": {
            "price": {
                "open":   {"type": "float64", "unit": "CNY",     "desc": "Open price"},
                "high":   {"type": "float64", "unit": "CNY",     "desc": "Highest price"},
                "low":    {"type": "float64", "unit": "CNY",     "desc": "Lowest price"},
                "close":  {"type": "float64", "unit": "CNY",     "desc": "Close price"},
                "volume": {"type": "float64", "unit": "shares",  "desc": "Trading volume (shares)"},
                "amount": {"type": "float64", "unit": "CNY",     "desc": "Trading amount"},
            },
            "fundamental": {
                "pe":         {"type": "float64", "unit": "倍",    "desc": "P/E ratio (TTM)"},
                "pb":         {"type": "float64", "unit": "倍",    "desc": "P/B ratio"},
                "ps":         {"type": "float64", "unit": "倍",    "desc": "P/S ratio (TTM)"},
                "roe":        {"type": "float64", "unit": "ratio", "desc": "Return on Equity"},
                "market_cap": {"type": "float64", "unit": "亿元",  "desc": "Total market cap"},
            },
            "industry": {
                "code":          {"type": "str",     "desc": "Stock code"},
                "industry_name": {"type": "str",     "desc": "Industry classification"},
            },
        },
        "sources": {
            "akshare": {
                "description": "AkShare open-source Python library (free, no token).",
                "endpoints": {
                    "stock_zh_a_hist": {
                        "description": "Individual stock daily K-line (unadjusted).",
                        "url_hint": "ak.stock_zh_a_hist(symbol, period='daily', start_date=..., end_date=..., adjust='')",
                        "columns": {
                            "日期":   {"canonical": "date",      "type": "datetime", "nullable": False},
                            "开盘":   {"canonical": "open",      "type": "float64",  "nullable": True},
                            "收盘":   {"canonical": "close",     "type": "float64",  "nullable": True},
                            "最高":   {"canonical": "high",      "type": "float64",  "nullable": True},
                            "最低":   {"canonical": "low",       "type": "float64",  "nullable": True},
                            "成交量": {"canonical": "volume",    "type": "int64",    "nullable": True,   "note": "unit: shares"},
                            "成交额": {"canonical": "amount",    "type": "float64",  "nullable": True,   "note": "unit: CNY"},
                            "振幅":   {"canonical": "amplitude", "type": "float64",  "nullable": True,   "note": "%"},
                            "涨跌幅": {"canonical": "pct_change","type": "float64",  "nullable": True,   "note": "%"},
                            "涨跌额": {"canonical": "change",    "type": "float64",  "nullable": True,   "note": "CNY"},
                            "换手率": {"canonical": "turnover",  "type": "float64",  "nullable": True,   "note": "%"},
                        },
                    },
                    "stock_zh_a_spot_em": {
                        "description": "Real-time A-share snapshot (used for fundamental data fill).",
                        "url_hint": "ak.stock_zh_a_spot_em()",
                        "columns": {
                            "代码":         {"canonical": "code",                 "type": "str"},
                            "名称":         {"canonical": "name",                 "type": "str"},
                            "最新价":       {"canonical": "latest_price",         "type": "float64"},
                            "今开":         {"canonical": "open",                 "type": "float64"},
                            "最高":         {"canonical": "high",                 "type": "float64"},
                            "最低":         {"canonical": "low",                  "type": "float64"},
                            "昨收":         {"canonical": "prev_close",           "type": "float64"},
                            "市盈率-动态":   {"canonical": "pe",                   "type": "float64",  "note": "TTM P/E"},
                            "市净率":       {"canonical": "pb",                   "type": "float64"},
                            "市销率":       {"canonical": "ps",                   "type": "float64",  "note": "TTM P/S"},
                            "总市值":       {"canonical": "market_cap_raw",       "type": "float64",  "note": "unit: CNY; divided by 1e8 → canonical(亿元)"},
                            "流通市值":     {"canonical": "circulating_market_cap","type": "float64", "note": "unit: CNY"},
                            "成交量":       {"canonical": "volume",               "type": "int64"},
                            "成交额":       {"canonical": "amount",               "type": "float64"},
                            "换手率":       {"canonical": "turnover",             "type": "float64",  "note": "%"},
                            "涨跌幅":       {"canonical": "pct_change",           "type": "float64",  "note": "%"},
                            "量比":         {"canonical": "volume_ratio",         "type": "float64"},
                            "60日涨跌幅":   {"canonical": "pct_change_60d",       "type": "float64"},
                            "年初至今涨跌幅":{"canonical": "pct_change_ytd",      "type": "float64"},
                        },
                    },
                    "stock_industry_clf_hist_sw": {
                        "description": "Shenwan industry classification (broken: SSL issues with swsresearch.com).",
                        "url_hint": "ak.stock_industry_clf_hist_sw() — currently broken, use Tushare stock_basic instead",
                        "columns": {
                            "代码": {"canonical": "code", "type": "str"},
                            "名称": {"canonical": "name", "type": "str"},
                            "行业": {"canonical": "industry_name", "type": "str"},
                        },
                    },
                },
            },
            "tushare": {
                "description": "Tushare Pro API (requires token).",
                "endpoints": {
                    "daily": {
                        "description": "Daily K-line (already adjusted).",
                        "url_hint": "pro.daily(ts_code=..., start_date=..., end_date=...)",
                        "columns": {
                            "ts_code":    {"canonical": "code",       "type": "str"},
                            "trade_date": {"canonical": "date",       "type": "datetime"},
                            "open":       {"canonical": "open",       "type": "float64"},
                            "high":       {"canonical": "high",       "type": "float64"},
                            "low":        {"canonical": "low",        "type": "float64"},
                            "close":      {"canonical": "close",      "type": "float64"},
                            "pre_close":  {"canonical": "prev_close", "type": "float64"},
                            "change":     {"canonical": "change",     "type": "float64"},
                            "pct_chg":    {"canonical": "pct_change", "type": "float64", "note": "%"},
                            "vol":        {"canonical": "volume",     "type": "float64", "note": "unit: lots (=100 shares); loader uses as-is"},
                            "amount":     {"canonical": "amount",     "type": "float64", "note": "unit: 千元; loader uses as-is"},
                        },
                    },
                    "daily_basic": {
                        "description": "Daily fundamental indicators.",
                        "url_hint": "pro.daily_basic(ts_code=..., trade_date=..., fields='ts_code,pe_ttm,pb,ps_ttm,roe,total_mv')",
                        "columns": {
                            "ts_code":    {"canonical": "code",       "type": "str"},
                            "trade_date": {"canonical": "date",       "type": "datetime"},
                            "pe_ttm":     {"canonical": "pe",         "type": "float64", "note": "TTM P/E"},
                            "pb":         {"canonical": "pb",         "type": "float64"},
                            "ps_ttm":     {"canonical": "ps",         "type": "float64", "note": "TTM P/S"},
                            "roe":        {"canonical": "roe",        "type": "float64", "note": "raw is %; loader divides by 100"},
                            "total_mv":   {"canonical": "market_cap", "type": "float64", "note": "unit: 万元; loader divides by 10000 → 亿元"},
                        },
                    },
                    "stock_basic": {
                        "description": "Stock basic info + industry classification.",
                        "url_hint": "pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')",
                        "columns": {
                            "ts_code":  {"canonical": "code",          "type": "str"},
                            "industry": {"canonical": "industry_name", "type": "str"},
                        },
                    },
                },
            },
            "westock": {
                "description": "WorkBuddy built-in westock-data skill.",
                "endpoints": {
                    "get_daily_kline": {
                        "description": "Per-stock daily K-line.",
                        "url_hint": "westock.get_daily_kline(code, start, end, fields=[...])",
                        "columns": {
                            "open":   {"canonical": "open",   "type": "float64"},
                            "high":   {"canonical": "high",   "type": "float64"},
                            "low":    {"canonical": "low",    "type": "float64"},
                            "close":  {"canonical": "close",  "type": "float64"},
                            "volume": {"canonical": "volume", "type": "float64"},
                            "amount": {"canonical": "amount", "type": "float64"},
                        },
                    },
                    "get_fundamentals": {
                        "description": "Batch fundamental data.",
                        "url_hint": "westock.get_fundamentals(codes, factor, start, end)",
                        "columns": {
                            "pe":         {"canonical": "pe",         "type": "float64"},
                            "pb":         {"canonical": "pb",         "type": "float64"},
                            "roe":        {"canonical": "roe",        "type": "float64"},
                            "market_cap": {"canonical": "market_cap", "type": "float64"},
                        },
                    },
                    "get_industry_classification": {
                        "description": "Industry classification dict.",
                        "url_hint": "westock.get_industry_classification(codes)",
                        "columns": {
                            "code":          {"canonical": "code",          "type": "str"},
                            "industry_name": {"canonical": "industry_name", "type": "str"},
                        },
                    },
                },
            },
        },
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "data", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  [raw] Column mapping saved to {output_path}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tushare token helper
# ---------------------------------------------------------------------------

def _get_tushare_token(cfg: Optional[dict] = None) -> str:
    """
    Get Tushare token.

    Priority:
    1. cfg['data']['tushare_token']   (if cfg is provided)
    2. config/config.yaml  →  data.tushare_token
    3. TUSHARE_TOKEN env var

    Args:
        cfg: Optional pre-loaded config dict. If None, will try to load
             config/config.yaml automatically.
    """
    # If cfg not provided, try to load from default path
    if cfg is None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "config", "config.yaml"
        )
        cfg_path = os.path.abspath(cfg_path)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        if cfg is None:
            cfg = {}

    if cfg:
        token = cfg.get("data", {}).get("tushare_token", "")
        if token:
            return token

    return os.environ.get("TUSHARE_TOKEN", "")


# ---------------------------------------------------------------------------
# Real data loader (westock / AkShare / Tushare)


# ---------------------------------------------------------------------------
# Code format converters (shared helpers)
# ---------------------------------------------------------------------------


def _ts_code(code: str) -> str:
    """
    Convert plain stock code to Tushare ts_code format.

    Examples:
        "000001"   -> "000001.SZ"
        "600000"   -> "600000.SH"
        "000001.SZ" -> "000001.SZ"  (passthrough)
    """
    c = str(code).strip()
    if "." in c:
        return c  # already ts_code format
    if c.startswith("6"):
        return c + ".SH"
    elif c.startswith("0") or c.startswith("3"):
        return c + ".SZ"
    elif c.startswith("8") or c.startswith("4"):
        return c + ".BJ"
    return c + ".SZ"  # fallback


def _ak_daily_symbol(code: str) -> str:
    """
    Convert plain stock code to AkShare stock_zh_a_daily() symbol (Sina source).

    Examples:
        "600000" -> "sh600000"
        "000001" -> "sz000001"
        "830839" -> "bj830839"
    """
    c = str(code).strip()
    if c.startswith("6"):
        return "sh" + c
    elif c.startswith("0") or c.startswith("3"):
        return "sz" + c
    elif c.startswith("8") or c.startswith("4"):
        return "bj" + c
    return "sh" + c  # fallback


# ---------------------------------------------------------------------------

def load_real_data(
    universe: str = "hs300",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    source: str = "westock",
    force_refresh: bool = False,
    config: Optional[dict] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.Series]:
    """
    Load real A-share market data from the configured source.

    This function tries sources in order:
      1. Local pickle cache (fastest)
      2. westock (WorkBuddy built-in A-share data)
      3. Tushare (requires token)
      4. AkShare (open-source Python library)
      5. Qlib (high-performance .bin format, local — lowest priority, heavy dependency)
      6. Synthetic fallback (generated data)

    Args:
        universe: Stock universe ('hs300', 'zz500', 'all_a')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Preferred source ('westock', 'akshare', 'tushare', 'auto')
        force_refresh: Skip cache and re-download
        config: Optional config dict (to read data.tushare_token).
                If None, will try to load from config/config.yaml.

    Returns:
        Tuple of (price_data, fundamental_data, industry_series)
        - price_data: Dict of DataFrames with keys [open, high, low, close, volume, amount]
        - fundamental_data: Dict of DataFrames with keys [pe, pb, ps, roe, market_cap]
        - industry_series: Series, index=stock_code, value=industry_name
    """

    global_config.universe = universe
    global_config.start_date = start_date
    global_config.end_date = end_date

    # Load config if not provided (try default path)
    if config is None:
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "config", "config.yaml")
        _cfg_path = os.path.abspath(_cfg_path)
        if os.path.exists(_cfg_path):
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                config = yaml.safe_load(_f)
                if config is None:
                    config = {}

    # Generate column mapping JSON for documentation (idempotent)
    _generate_column_mapping_json()

    cache_file = config_path('data','cache.pkl')

    # 1. Try cache first
    if not force_refresh:
        cached = _load_cache(cache_file)
        if cached is not None:
            price_data, _, _ = cached
            close = price_data.get('close')
            if close is not None and close.isna().all().all():
                # Cache is corrupted (all NaN) — discard and re-fetch
                print(f"  [cache] Corrupted (all NaN), discarding {cache_file}")
                os.remove(cache_file)
            else:
                print(f"  [cache] Loaded from {cache_file}")
                return cached

    # 2. Try westock (WorkBuddy native A-share data)
    if source in ("westock", "auto"):
        try:
            result = _load_from_westock(universe, start_date, end_date)
            if result is not None:
                print(f"  [westock] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [westock] Failed: {e}")

    # 3. Try Tushare
    if source in ("tushare", "auto"):
        try:
            result = _load_from_tushare(universe, start_date, end_date)
            if result is not None:
                print(f"  [tushare] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [tushare] Failed: {e}")

    # 4. Try AkShare
    if source in ("akshare", "auto"):
        try:
            result = _load_from_akshare(universe, start_date, end_date)
            if result is not None:
                print(f"  [akshare] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [akshare] Failed: {e}")

    # 6. Fallback: generate synthetic data (same shape as real data would have)
    print(f"  [fallback] No real data source available, generating synthetic data")
    # 智能解析 n_stocks 从 universe 字符串
    import re
    m = re.search(r'(\d+)', universe)
    if m:
        n_stocks = int(m.group(1))
    else:
        n_stocks = {'hs300': 300, 'zz500': 500, 'all_a': 1000}.get(universe, 100)
    print(f"  [fallback] universe='{universe}' → n_stocks={n_stocks}")
    return _generate_synthetic_data(n_stocks, start_date, end_date)


def _load_from_westock(
    universe: str,
    start_date: str,
    end_date: str,
) -> Optional[Tuple]:
    """
    Load data via westock-data skill (WorkBuddy built-in).
    
    westock-data provides: daily kline, fundamentals, industry classification
    for A-share stocks. This function uses the westock-data skill's
    Python API to batch-fetch data.
    
    Note: westock-data is available as a skill in WorkBuddy. This function
    is designed to be called from a WorkBuddy session where the skill is loaded.
    When running standalone, it falls back to synthetic data.
    """
    try:
        # Attempt to use westock's Python API
        # The westock-data skill exposes a Python SDK at runtime
        import importlib
        try:
            westock = importlib.import_module('westock')
        except ImportError:
            # westock SDK not installed — this is expected in standalone runs
            return None
        
        # Get universe stock list
        if universe == 'hs300':
            stocks = westock.get_index_constituents('000300.SH')
        elif universe == 'zz500':
            stocks = westock.get_index_constituents('000905.SH')
        else:
            stocks = westock.get_all_stocks()
        
        stock_codes = [s['code'] for s in stocks[:300]]  # cap at 300 for performance
        n_stocks = len(stock_codes)
        
        if n_stocks == 0:
            return None

        # Save stock list as raw data
        _save_raw_csv(pd.DataFrame({'code': stock_codes}), 'westock/stock_list.csv', index=False)

        dates = pd.date_range(start_date, end_date, freq='B')
        n_days = len(dates)
        
        # Build price data
        price_data = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            price_data[field] = pd.DataFrame(
                np.full((n_days, n_stocks), np.nan),
                index=dates, columns=stock_codes
            )
        
        # Fetch daily kline for each stock (batched)
        batch_size = 50
        for i in range(0, n_stocks, batch_size):
            batch = stock_codes[i:i + batch_size]
            for code in batch:
                try:
                    kline = westock.get_daily_kline(
                        code, start_date, end_date,
                        fields=['open', 'high', 'low', 'close', 'volume', 'amount']
                    )
                    if kline is not None and not kline.empty:
                        kline.index = pd.to_datetime(kline.index)
                        # Save raw kline per stock
                        try:
                            _save_raw_csv(kline.sort_index(), f'westock/kline/{code}.csv', index=True)
                        except Exception:
                            pass
                        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                            if field in kline.columns:
                                common_dates = dates.intersection(kline.index)
                                price_data[field].loc[common_dates, code] = \
                                    kline.loc[common_dates, field].values
                except Exception:
                    continue
        
        # Fill missing price data: ffill only (no bfill — avoids look-ahead bias)
        # volume: fill NaN with 0 (no trading on suspension days)
        for field in price_data:
            if field == 'volume':
                price_data[field] = price_data[field].fillna(0)
            else:
                price_data[field] = price_data[field].ffill()

        # Validate: if close data is still all NaN, return None to fall through
        if price_data['close'].isna().all().all():
            print(f"  [westock] All price data is NaN — falling back to next source")
            return None
        
        # Fundamental data — try westock fundamentals API
        fundamental_data = {}
        # Initialize with NaN; real values will be filled in below
        for factor in ['pe', 'pb', 'ps', 'roe', 'market_cap']:
            fundamental_data[factor] = pd.DataFrame(
                np.nan, index=dates, columns=stock_codes
            )
        try:
            for factor in ['pe', 'pb', 'ps', 'roe', 'market_cap']:
                fund_df = westock.get_fundamentals(stock_codes, factor, start_date, end_date)
                if fund_df is not None and not fund_df.empty:
                    fund_df.index = pd.to_datetime(fund_df.index)
                    fundamental_data[factor] = fund_df.reindex(index=dates, columns=stock_codes)
                    # Save raw fundamentals per factor
                    try:
                        _save_raw_csv(fund_df, f'westock/fundamentals/{factor}.csv', index=True)
                    except Exception:
                        pass
        except Exception:
            pass  # Keep NaN-initialized DataFrames; ffill applied below

        # Forward fill missing fundamentals (use most recent available value; no bfill to avoid look-ahead bias)
        for factor in fundamental_data:
            fundamental_data[factor] = fundamental_data[factor].ffill()
        
        # Industry classification
        try:
            industry_map = westock.get_industry_classification(stock_codes)
            industry_series = pd.Series(industry_map)
            # Save raw industry classification
            try:
                _save_raw_csv(
                    pd.DataFrame({'code': stock_codes, 'industry': [industry_map.get(c, '') for c in stock_codes]}),
                    'westock/industry.csv', index=False
                )
            except Exception:
                pass
        except Exception:
            industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Materials']
            np.random.seed(42)
            industry_series = pd.Series(
                np.random.choice(industries, size=n_stocks),
                index=stock_codes
            )
        
        return price_data, fundamental_data, industry_series

    except Exception as e:
        print(f"  [westock] Exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Tushare real data loader
# ---------------------------------------------------------------------------

def _load_from_tushare(
    universe: str,
    start_date: str,
    end_date: str,
) -> Optional[Tuple]:
    """
    Load real A-share data via Tushare Pro API.

    Requirements:
      pip install tushare
      export TUSHARE_TOKEN="your_token"   # or set in config.yaml

    Tushare returns data at stock-level; this function loops over constituents
    and assembles the project's canonical dict-of-DataFrames format.

    Returns (price_data, fundamental_data, industry_series) on success,
    None on any failure / missing token.
    """
    try:
        import tushare as ts
        import os
        import time

        token = _get_tushare_token()
        if not token:
            print("  [tushare] TUSHARE_TOKEN not set (checked config.yaml + env var), skipping")
            return None

        ts.set_token(token)
        pro = ts.pro_api()
        print("  [tushare] API initialized")
    except Exception as e:
        print(f"  [tushare] Init failed: {e}")
        return None

    try:
        # ---- 1. Get stock list for the universe ----
        if universe == 'hs300':
            try:
                import akshare as ak
                df_cons = ak.index_stock_cons_csindex(symbol="000300")
                raw_codes = df_cons['成分券代码'].tolist()
            except Exception:
                df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
                raw_codes = df_basic['ts_code'].tolist()[:300]
        elif universe == 'zz500':
            try:
                import akshare as ak
                df_cons = ak.index_stock_cons_csindex(symbol="000905")
                raw_codes = df_cons['成分券代码'].tolist()
            except Exception:
                df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
                raw_codes = df_basic['ts_code'].tolist()[:500]
        else:
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
            raw_codes = df_basic['ts_code'].tolist()

        if not raw_codes:
            print("  [tushare] Empty stock list")
            return None

        stock_codes = [c for c in raw_codes if isinstance(c, str) and len(c) > 0][:300]
        n_stocks = len(stock_codes)
        print(f"  [tushare] {n_stocks} stocks to fetch")

        # Save stock list as raw data
        _save_raw_csv(pd.DataFrame({'code': stock_codes}), 'tushare/stock_list.csv', index=False)

        # ---- 2. Date range ----
        dates = pd.date_range(start_date, end_date, freq='B')
        n_days = len(dates)
        if n_days == 0:
            return None

        # ---- 3. Initialize price_data DataFrames ----
        price_data = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            price_data[field] = pd.DataFrame(np.nan, index=dates, columns=stock_codes)

        # ---- 4. Fetch daily kline per stock ----
        print(f"  [tushare] Fetching daily kline ({n_stocks} stocks)...")
        sd = start_date.replace('-', '')
        ed = end_date.replace('-', '')

        # Helper: convert plain code to Tushare ts_code format
        # AkShare returns '000001' → Tushare needs '000001.SZ'
        # Tushare stock_basic already returns '000001.SZ' — passthrough
        fail_count = 0
        for i, code in enumerate(stock_codes):
            try:
                ts_code = _ts_code(code)
                df = pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)
                if df is None or df.empty:
                    fail_count += 1
                    continue
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.sort_values('trade_date').set_index('trade_date')
                # Save raw kline per stock (Tushare English column names preserved)
                try:
                    _save_raw_csv(df.sort_index(), f'tushare/kline/{code}.csv', index=True)
                except Exception:
                    pass
                col_map = {'open': 'open', 'high': 'high', 'low': 'low',
                           'close': 'close', 'volume': 'vol', 'amount': 'amount'}
                for field, src in col_map.items():
                    if src in df.columns:
                        common = dates.intersection(df.index)
                        if len(common) > 0:
                            price_data[field].loc[common, code] = df.loc[common, src].values
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    print(f"  [tushare] Fetch error for {code}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  [tushare] Kline progress: {i + 1}/{n_stocks} (errors: {fail_count})")
            time.sleep(0.2)  # rate limit

        # Fill missing price data: ffill only (no bfill — avoids look-ahead bias)
        # volume: fill NaN with 0 (no trading on suspension days)
        for field in price_data:
            if field == 'volume':
                price_data[field] = price_data[field].fillna(0)
            else:
                price_data[field] = price_data[field].ffill()

        # Validate: if close data is still all NaN, return None to fall through
        if price_data['close'].isna().all().all():
            print(f"  [tushare] All price data is NaN — falling back to next source")
            return None

        # ---- 6. Fundamental data ----
        print("  [tushare] Fetching fundamentals...")
        fundamental_data = {}
        for field in ['pe', 'pb', 'ps', 'roe', 'market_cap']:
            fundamental_data[field] = pd.DataFrame(np.nan, index=dates, columns=stock_codes)

        # daily_basic: iterate ONE STOCK AT A TIME with full date range.
        # Tushare daily_basic does NOT support comma-separated ts_code
        # with start_date/end_date range queries (returns empty).
        # Single stock + date range is the only reliable pattern.
        daily_basic_all = []
        try:
            ts_codes_fund = [_ts_code(c) for c in stock_codes]
            n_stocks = len(ts_codes_fund)
            sd_fund = start_date.replace('-', '')
            ed_fund = end_date.replace('-', '')

            for si, tsc in enumerate(ts_codes_fund):
                plain = stock_codes[si]  # original plain code (e.g. '000001')
                try:
                    df_b = pro.daily_basic(
                        ts_code=tsc,
                        start_date=sd_fund,
                        end_date=ed_fund,
                        fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv'
                    )
                except Exception:
                    df_b = None

                if df_b is None or df_b.empty:
                    time.sleep(0.05)
                    continue

                daily_basic_all.append(df_b)
                for _, row in df_b.iterrows():
                    try:
                        dt_val = row['trade_date']
                        if isinstance(dt_val, str):
                            dt = pd.Timestamp(dt_val[:4] + '-' + dt_val[4:6] + '-' + dt_val[6:])
                        else:
                            dt = pd.Timestamp(dt_val)
                        if dt not in fundamental_data['pe'].index:
                            continue
                        if 'pe_ttm' in row and pd.notna(row['pe_ttm']):
                            fundamental_data['pe'].loc[dt, plain] = float(row['pe_ttm'])
                        if 'pb' in row and pd.notna(row['pb']):
                            fundamental_data['pb'].loc[dt, plain] = float(row['pb'])
                        if 'ps_ttm' in row and pd.notna(row['ps_ttm']):
                            fundamental_data['ps'].loc[dt, plain] = float(row['ps_ttm'])
                        if 'total_mv' in row and pd.notna(row['total_mv']):
                            fundamental_data['market_cap'].loc[dt, plain] = float(row['total_mv']) / 10000.0
                    except Exception:
                        continue

                if (si + 1) % 50 == 0 or si == n_stocks - 1:
                    n_filled = fundamental_data['pe'].notna().any(axis=1).sum()
                    print(f"  [tushare] Fundamentals: stock {si+1}/{n_stocks}, filled dates: {n_filled}/{n_days}")
                time.sleep(0.05)

        except Exception as e:
            print(f"  [tushare] Fundamentals fetch error: {e}")

        # Save combined daily_basic raw data
        if daily_basic_all:
            try:
                df_daily_basic_combined = pd.concat(daily_basic_all, ignore_index=True)
                _save_raw_csv(df_daily_basic_combined, 'tushare/daily_basic.csv', index=False)
            except Exception:
                pass

        # ---- 6b. ROE from fina_indicator (quarterly) ----
        # daily_basic doesn't have ROE; use fina_indicator which provides
        # quarterly financial ratios. Forward-fill to daily dates.
        #
        # NOTE: We look back 2 years before start_date when fetching ROE.
        # This ensures that even if a stock's first report within [start_date, end_date]
        # is published late, we still capture the prior report and can ffill into
        # the early part of our date range (no need for bfill / look-ahead bias).
        roe_loaded = 0
        try:
            print("  [tushare] Fetching ROE (fina_indicator)...")
            roe_start_dt = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime('%Y%m%d')
            for si, tsc in enumerate(ts_codes_fund):
                plain = stock_codes[si]
                try:
                    df_roe = pro.fina_indicator(
                        ts_code=tsc,
                        start_date=roe_start_dt,
                        end_date=ed_fund,
                        fields='ts_code,end_date,roe'
                    )
                except Exception:
                    df_roe = None

                if df_roe is None or df_roe.empty:
                    time.sleep(0.05)
                    continue

                # Sort by end_date ascending so earlier reports are filled first;
                # otherwise a later (older) report can overwrite a newer one.
                df_roe = df_roe.sort_values('end_date')

                for _, row in df_roe.iterrows():
                    try:
                        dt_val = row['end_date']
                        if isinstance(dt_val, str):
                            dt = pd.Timestamp(dt_val[:4] + '-' + dt_val[4:6] + '-' + dt_val[6:])
                        else:
                            dt = pd.Timestamp(dt_val)
                        if 'roe' in row and pd.notna(row['roe']):
                            roe_val = float(row['roe']) / 100.0  # Tushare roe is in %
                            # Assign to the nearest date in our index that's >= report_date
                            match_dates = fundamental_data['roe'].index[
                                fundamental_data['roe'].index >= dt
                            ]
                            if len(match_dates) > 0:
                                fundamental_data['roe'].loc[match_dates[0]:, plain] = roe_val
                            roe_loaded += 1
                    except Exception:
                        continue

                if (si + 1) % 100 == 0 or si == n_stocks - 1:
                    n_roe = fundamental_data['roe'].notna().any(axis=0).sum()
                    print(f"  [tushare] ROE: stock {si+1}/{n_stocks}, covered: {n_roe}/{n_stocks}, data points: {roe_loaded}")
                time.sleep(0.05)
        except Exception as e:
            print(f"  [tushare] ROE fetch error: {e}")

        # Fill missing fundamentals (forward fill only; no bfill to avoid look-ahead bias)
        for field in fundamental_data:
            fundamental_data[field] = fundamental_data[field].ffill()

        # ---- 7. Industry classification ----
        print("  [tushare] Fetching industry classification...")
        industry_dict = {}
        try:
            ts_codes = [_ts_code(c) for c in stock_codes]
            df_ind = pro.stock_basic(ts_code=','.join(ts_codes),
                                      fields='ts_code,industry')
            # Save raw industry classification
            try:
                _save_raw_csv(df_ind, 'tushare/industry_raw.csv', index=False)
            except Exception:
                pass
            for _, row in df_ind.iterrows():
                if pd.notna(row.get('industry', None)):
                    ts_code = str(row['ts_code']).strip()
                    ind = str(row['industry']).strip()
                    # Convert ts_code → plain code to match stock_codes keys
                    plain = ts_code.split('.')[0]
                    industry_dict[plain] = ind
        except Exception:
            pass

        # Fallback for missing industries (only fill keys NOT already present)
        # 申万一级行业（中文，和 Tushare stock_basic(industry=...) 返回格式一致）
        all_industries = ['银行', '房地产', '医药生物', '电子', '计算机',
                          '传媒', '通信', '电力设备', '基础化工', '机械设备',
                          '汽车', '食品饮料', '家用电器', '建筑材料', '建筑装饰',
                          '有色金属', '钢铁', '国防军工', '农林牧渔', '纺织服饰',
                          '轻工制造', '商贸零售', '社会服务', '综合', '公用事业']
        np.random.seed(42)
        for code in stock_codes:
            if code not in industry_dict:
                industry_dict[code] = np.random.choice(all_industries)

        industry_series = pd.Series(industry_dict)

        print(f"  [tushare] Data loaded: {n_stocks} stocks x {n_days} days")

        # Save fundamental data to CSV
        for factor in fundamental_data:
            _save_raw_csv(fundamental_data[factor], f'tushare/fundamentals/{factor}.csv', index=True)

        return price_data, fundamental_data, industry_series

    except Exception as e:
        print(f"  [tushare] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Qlib real data loader
# ---------------------------------------------------------------------------

def _load_from_akshare(
    universe: str,
    start_date: str,
    end_date: str,
) -> Optional[Tuple]:
    """
    Load real A-share data via AkShare (free, no token required).

    Install: pip install akshare

    Returns (price_data, fundamental_data, industry_series) on success,
    None on any failure.
    """
    try:
        import akshare as ak
        import time
    except ImportError:
        print("  [akshare] Not installed. Run: pip install akshare")
        return None

    try:
        # ---- 1. Get stock list ----
        if universe == 'hs300':
            df_cons = ak.index_stock_cons_csindex(symbol="000300")
            raw_codes = df_cons['成分券代码'].tolist()
        elif universe == 'zz500':
            df_cons = ak.index_stock_cons_csindex(symbol="000905")
            raw_codes = df_cons['成分券代码'].tolist()
        else:
            # all_a: get the full A-share stock list
            # Try multiple sources; EastMoney (stock_zh_a_spot_em) is often unreachable.
            raw_codes = []
            # 1) Sina: code + name only (lightweight, most reliable)
            try:
                df_info = ak.stock_info_a_code_name()
                if df_info is not None and not df_info.empty:
                    col = 'code' if 'code' in df_info.columns else df_info.columns[0]
                    raw_codes = df_info[col].tolist()
            except Exception as e:
                print(f"  [akshare] stock_info_a_code_name() failed: {e}")

            # 2) Fallback: Sina spot quote (heavier but has 代码 col)
            if not raw_codes:
                try:
                    df_spot = ak.stock_zh_a_spot()
                    raw_codes = df_spot['代码'].tolist()
                except Exception as e:
                    print(f"  [akshare] stock_zh_a_spot() failed: {e}")

            # 3) Fallback: hs300 + zz500 as approximation for all_a
            if not raw_codes:
                print("  [akshare] Cannot fetch all_a stock list from AkShare, "
                      "falling back to hs300+zz500")
                try:
                    df_cons_1 = ak.index_stock_cons_csindex(symbol="000300")
                    df_cons_2 = ak.index_stock_cons_csindex(symbol="000905")
                    raw_codes = (df_cons_1['成分券代码'].tolist() +
                                 df_cons_2['成分券代码'].tolist())
                except Exception as e:
                    print(f"  [akshare] Fallback hs300+zz500 also failed: {e}")

        if not raw_codes:
            print("  [akshare] Empty stock list")
            return None

        stock_codes = [str(c).strip() for c in raw_codes if str(c).strip()][:300]
        n_stocks = len(stock_codes)
        if n_stocks == 0:
            return None

        print(f"  [akshare] {n_stocks} stocks to fetch")

        # Save stock list as raw data
        _save_raw_csv(pd.DataFrame({'code': stock_codes}), 'akshare/stock_list.csv', index=False)

        # ---- 2. Date range ----
        dates = pd.date_range(start_date, end_date, freq='B')
        n_days = len(dates)
        if n_days == 0:
            return None

        # ---- 3. Initialize DataFrames ----
        price_data = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            price_data[field] = pd.DataFrame(np.nan, index=dates, columns=stock_codes)

        # ---- 4. Fetch daily kline per stock ----
        print(f"  [akshare] Fetching daily kline...")

        # Helper: detect exchange prefix for stock_zh_a_daily (新浪源)
        # Shanghai: 6xxxxx, Shenzhen: 0xxxxx/3xxxxx, Beijing: 8xxxxx/4xxxxx
        # stock_zh_a_daily uses English column names (新浪源, works when EastMoney is blocked)
        col_map_en = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount',
        }

        fail_count = 0
        for i, code in enumerate(stock_codes):
            try:
                sym = _ak_daily_symbol(code)
                df = ak.stock_zh_a_daily(symbol=sym, adjust="")
                if df is None or df.empty:
                    fail_count += 1
                    continue
                date_col = 'date' if 'date' in df.columns else df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col).set_index(date_col)
                # Filter to requested date range
                df = df.loc[start_date:end_date]
                if df.empty:
                    fail_count += 1
                    continue
                # Save raw kline per stock
                try:
                    _save_raw_csv(df.sort_index(), f'akshare/kline/{code}.csv', index=True)
                except Exception:
                    pass
                for field, en_col in col_map_en.items():
                    if en_col in df.columns:
                        common = dates.intersection(df.index)
                        if len(common) > 0:
                            price_data[field].loc[common, code] = df.loc[common, en_col].values
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    print(f"  [akshare] Fetch error for {code}: {e}")

            if (i + 1) % 100 == 0:
                print(f"  [akshare] Kline progress: {i + 1}/{n_stocks} (errors: {fail_count})")

        # ---- 5. Fill missing ----
        # Price data: ffill only (no bfill — avoids look-ahead bias)
        # volume: fill NaN with 0 (no trading on suspension days)
        for field in price_data:
            if field == 'volume':
                price_data[field] = price_data[field].fillna(0)
            else:
                price_data[field] = price_data[field].ffill()

        # Validate: if close data is still all NaN, return None to fall through
        if price_data['close'].isna().all().all():
            print(f"  [akshare] All price data is NaN — falling back to next source")
            return None

        # ---- 6. Fundamental data ----
        # Initialize with NaN; try multiple sources.
        # NOTE: Using a spot snapshot and broadcasting to all dates is a poor
        # approximation for backtesting. We now try Tushare daily_basic first
        # (real historical data), then fall back to AkShare spot.
        print("  [akshare] Building fundamental data...")
        fundamental_data = {}
        for field in ['pe', 'pb', 'ps', 'roe', 'market_cap']:
            fundamental_data[field] = pd.DataFrame(
                np.nan, index=dates, columns=stock_codes
            )

        fundamental_loaded = False

        # 1) Try Tushare daily_basic — real daily PE/PB/PS/market_cap history
        # Tushare daily_basic does NOT support comma-separated ts_code with
        # start_date/end_date range queries (returns empty). Iterate one stock
        # at a time — the only reliable pattern.
        ts_codes_fund = []
        n_stocks_fund = 0
        sd_fund = ''
        ed_fund = ''
        try:
            import tushare as ts
            token = _get_tushare_token()
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                ts_codes_fund = [_ts_code(c) for c in stock_codes]
                n_stocks_fund = len(ts_codes_fund)
                sd_fund = start_date.replace('-', '')
                ed_fund = end_date.replace('-', '')
                pe_loaded = pb_loaded = ps_loaded = mc_loaded = 0
                for si, tsc in enumerate(ts_codes_fund):
                    plain = tsc.split('.')[0]
                    # Build Qlib-style code (SH000001 / SZ000001) for stock_codes lookup
                    if tsc.endswith('.SH'):
                        qc = 'SH' + plain
                    elif tsc.endswith('.SZ'):
                        qc = 'SZ' + plain
                    else:
                        qc = plain
                    if qc not in stock_codes:
                        continue
                    try:
                        df_fund = pro.daily_basic(
                            ts_code=tsc,
                            start_date=sd_fund,
                            end_date=ed_fund,
                            fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,total_mv'
                        )
                    except Exception:
                        df_fund = None

                    if df_fund is None or df_fund.empty:
                        time.sleep(0.05)
                        continue

                    for _, row in df_fund.iterrows():
                        try:
                            dt_val = row['trade_date']
                            if isinstance(dt_val, str):
                                dt = pd.Timestamp(dt_val[:4] + '-' + dt_val[4:6] + '-' + dt_val[6:])
                            else:
                                dt = pd.Timestamp(dt_val)
                            if dt not in fundamental_data['pe'].index:
                                continue
                            if pd.notna(row.get('pe_ttm')):
                                fundamental_data['pe'].loc[dt, qc] = float(row['pe_ttm'])
                                pe_loaded += 1
                            if pd.notna(row.get('pb')):
                                fundamental_data['pb'].loc[dt, qc] = float(row['pb'])
                                pb_loaded += 1
                            if pd.notna(row.get('ps_ttm')):
                                fundamental_data['ps'].loc[dt, qc] = float(row['ps_ttm'])
                                ps_loaded += 1
                            if pd.notna(row.get('total_mv')):
                                fundamental_data['market_cap'].loc[dt, qc] = float(row['total_mv']) / 10000.0
                                mc_loaded += 1
                        except Exception:
                            pass

                    if (si + 1) % 50 == 0 or si == n_stocks_fund - 1:
                        n_covered = fundamental_data['pe'].notna().any(axis=0).sum()
                        print(f"  [akshare] Tushare daily_basic: stock {si+1}/{n_stocks_fund}, "
                              f"covered: {n_covered}/{n_stocks}")
                    time.sleep(0.05)

                n_covered = fundamental_data['pe'].notna().any(axis=0).sum()
                if n_covered > 0:
                    fundamental_loaded = True
                    print(f"  [akshare] Tushare daily_basic coverage: "
                          f"pe={n_covered}/{n_stocks}, "
                          f"rows_pe={pe_loaded}, rows_pb={pb_loaded}, rows_ps={ps_loaded}, rows_mc={mc_loaded}")
        except Exception as e:
            print(f"  [akshare] Tushare daily_basic failed: {e}")

        # 1b) Try Tushare fina_indicator for ROE (quarterly, forward-fill to daily)
        try:
            token = _get_tushare_token()
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                roe_loaded = 0
                print("  [akshare] Fetching ROE (fina_indicator)...")
                roe_start_dt = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime('%Y%m%d')
                for si, tsc in enumerate(ts_codes_fund):
                    plain = tsc.split('.')[0]
                    if tsc.endswith('.SH'):
                        qc = 'SH' + plain
                    elif tsc.endswith('.SZ'):
                        qc = 'SZ' + plain
                    else:
                        qc = plain
                    if qc not in stock_codes:
                        continue
                    try:
                        df_roe = pro.fina_indicator(
                            ts_code=tsc,
                            start_date=roe_start_dt,
                            end_date=ed_fund,
                            fields='ts_code,end_date,roe'
                        )
                    except Exception:
                        df_roe = None

                    if df_roe is None or df_roe.empty:
                        time.sleep(0.05)
                        continue

                    # Sort by end_date ascending so earlier reports are filled first
                    df_roe = df_roe.sort_values('end_date')

                    for _, row in df_roe.iterrows():
                        try:
                            dt_val = row['end_date']
                            if isinstance(dt_val, str):
                                dt = pd.Timestamp(dt_val[:4] + '-' + dt_val[4:6] + '-' + dt_val[6:])
                            else:
                                dt = pd.Timestamp(dt_val)
                            if 'roe' in row and pd.notna(row['roe']):
                                roe_val = float(row['roe']) / 100.0
                                match_dates = fundamental_data['roe'].index[
                                    fundamental_data['roe'].index >= dt
                                ]
                                if len(match_dates) > 0:
                                    fundamental_data['roe'].loc[match_dates[0]:, qc] = roe_val
                                roe_loaded += 1
                        except Exception:
                            pass

                    if (si + 1) % 100 == 0 or si == n_stocks_fund - 1:
                        n_roe = fundamental_data['roe'].notna().any(axis=0).sum()
                        print(f"  [akshare] ROE: stock {si+1}/{n_stocks_fund}, covered: {n_roe}/{n_stocks}, data points: {roe_loaded}")
                    time.sleep(0.05)
        except Exception as e:
            print(f"  [akshare] ROE fetch error: {e}")

        # 2) Fallback: AkShare spot snapshot (latest data only, broadcast to all dates)
        if not fundamental_loaded:
            try:
                df_spot = None
                # EastMoney source (has PE/PB cols but often unreachable)
                try:
                    df_spot = ak.stock_zh_a_spot_em()
                except Exception as e:
                    print(f"  [akshare] stock_zh_a_spot_em() failed: {e}")
                # Sina source (more reliable but may lack PE/PB cols)
                if df_spot is None:
                    try:
                        df_spot = ak.stock_zh_a_spot()
                    except Exception as e:
                        print(f"  [akshare] stock_zh_a_spot() failed: {e}")

                if df_spot is not None and not df_spot.empty:
                    try:
                        _save_raw_csv(df_spot, 'akshare/spot_snapshot.csv', index=False)
                    except Exception:
                        pass
                    code_col = '代码' if '代码' in df_spot.columns else df_spot.columns[0]
                    pe_col   = '市盈率-动态' if '市盈率-动态' in df_spot.columns else None
                    pb_col   = '市净率'      if '市净率'      in df_spot.columns else None
                    ps_col   = '市销率'      if '市销率'      in df_spot.columns else None
                    mc_col   = '总市值'      if '总市值'      in df_spot.columns else None

                    if code_col in df_spot.columns:
                        for _, row in df_spot.iterrows():
                            code = str(row[code_col]).strip()
                            if code not in stock_codes:
                                continue
                            if pe_col and pd.notna(row.get(pe_col)):
                                try:
                                    fundamental_data['pe'].loc[:, code] = float(row[pe_col])
                                except Exception:
                                    pass
                            if pb_col and pd.notna(row.get(pb_col)):
                                try:
                                    fundamental_data['pb'].loc[:, code] = float(row[pb_col])
                                except Exception:
                                    pass
                            if ps_col and pd.notna(row.get(ps_col)):
                                try:
                                    fundamental_data['ps'].loc[:, code] = float(row[ps_col])
                                except Exception:
                                    pass
                            if mc_col and pd.notna(row.get(mc_col)):
                                try:
                                    fundamental_data['market_cap'].loc[:, code] = float(row[mc_col]) / 1e8
                                except Exception:
                                    pass
                    n_covered = fundamental_data['pe'].notna().any(axis=0).sum()
                    if n_covered > 0:
                        fundamental_loaded = True
                        print(f"  [akshare] Spot fundamental coverage: "
                              f"pe={n_covered}/{n_stocks} stocks")
            except Exception as e:
                print(f"  [akshare] Spot fundamental fetch failed: {e}")

        if not fundamental_loaded:
            print("  [akshare] Fundamental data unavailable from all sources — "
                  "proceeding with NaN (pipeline will handle)")

        # Forward fill only (no bfill to avoid look-ahead bias)
        for field in fundamental_data:
            fundamental_data[field] = fundamental_data[field].ffill()

        # ---- 7. Industry classification ----
        print("  [akshare] Fetching industry classification...")
        industry_dict = {}

        # Try Tushare for industry data (if available)
        try:
            import tushare as ts

            token = _get_tushare_token()
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                df_ind = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
                if df_ind is not None and not df_ind.empty:
                    for _, row in df_ind.iterrows():
                        ts_code = str(row['ts_code']).strip()
                        ind = str(row['industry']).strip()
                        if ind and ind != 'nan':
                            plain = ts_code.split('.')[0]
                            if plain in stock_codes:
                                industry_dict[plain] = ind
                    print(f"  [akshare] Industry from Tushare: {len(industry_dict)} stocks")
        except Exception as e:
            print(f"  [akshare] Industry from Tushare failed: {e}")

        # Fallback for missing industries (only fill keys NOT already present)
        # 申万一级行业（中文，和 Tushare stock_basic(industry=...) 返回格式一致）
        all_industries = ['银行', '房地产', '医药生物', '电子', '计算机',
                          '传媒', '通信', '电力设备', '基础化工', '机械设备',
                          '汽车', '食品饮料', '家用电器', '建筑材料', '建筑装饰',
                          '有色金属', '钢铁', '国防军工', '农林牧渔', '纺织服饰',
                          '轻工制造', '商贸零售', '社会服务', '综合', '公用事业']
        np.random.seed(42)
        for code in stock_codes:
            if code not in industry_dict:
                industry_dict[code] = np.random.choice(all_industries)

        industry_series = pd.Series(industry_dict)

        print(f"  [akshare] Data loaded: {n_stocks} stocks x {n_days} days")

        # Save fundamental data to CSV
        for factor in fundamental_data:
            _save_raw_csv(fundamental_data[factor], f'akshare/fundamentals/{factor}.csv', index=True)

        return price_data, fundamental_data, industry_series

    except Exception as e:
        print(f"  [akshare] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_stocks: int,
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.Series]:
    """
    Generate realistic synthetic data when no real data source is available.
    
    This is a fallback for when westock/AkShare/Tushare are all unavailable.
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq='B')
    n_days = len(dates)
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    # Price data with realistic random walk
    price_data = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in ['open', 'high', 'low', 'close']:
            base_price = 10 + np.random.randn(n_stocks) * 5
            returns = np.random.randn(n_days, n_stocks) * 0.02
            prices = base_price * np.exp(np.cumsum(returns, axis=0))
            price_data[col] = pd.DataFrame(prices, index=dates, columns=stock_codes)
        else:
            volume = np.abs(np.random.randn(n_days, n_stocks) * 1e6)
            price_data[col] = pd.DataFrame(volume, index=dates, columns=stock_codes)
    
    # Synthetic fundamentals with realistic A-share distributions
    # PE(TTM): log-normal, median ~20, always positive
    fundamental_data = {
        'pe': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.6 + np.log(20)),
            index=dates, columns=stock_codes
        ),
        # PB: log-normal, median ~2, always positive
        'pb': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.5 + np.log(2)),
            index=dates, columns=stock_codes
        ),
        # PS(TTM): log-normal, median ~2, always positive (similar to PB)
        'ps': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.7 + np.log(2)),
            index=dates, columns=stock_codes
        ),
        # ROE: clipped normal, median ~10%, range -50%~80%
        'roe': pd.DataFrame(
            np.clip(np.random.randn(n_days, n_stocks) * 0.08 + 0.10, -0.5, 0.8),
            index=dates, columns=stock_codes
        ),
        # Market cap (亿元): log-normal, median ~50亿, always positive
        'market_cap': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 1.2 + np.log(50)),
            index=dates, columns=stock_codes
        ),
    }

    industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Materials', 'Industrial']
    industry_series = pd.Series(
        np.random.choice(industries, size=n_stocks),
        index=stock_codes
    )
    
    return price_data, fundamental_data, industry_series


class DataLoader:
    """
    Data loader for stock market data.
    
    Supports multiple data sources and handles preprocessing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.preprocessing_config = self.data_config['preprocessing']
        
        # Data cache
        self.price_data = None
        self.fundamental_data = None
        self.industry_data = None
        
    def load_data(
        self,
        universe: str = None,
        train_start: str = None,
        train_end: str = None,
        test_start: str = None,
        test_end: str = None,
        full_start: str = None,
        full_end: str = None
    ) -> DatasetBundle:
        """
        Retrieve the pre-fetched dataset for (universe, full_start, full_end)
        from the local store and return train/test slices.

        This method NEVER downloads data. Populate the store once with
        `python load_datasets.py --universe ... --start-date ... --end-date ...`
        (the FULL span). If the archive is missing, retrieve_dataset raises a
        FileNotFoundError with run instructions.

        Args:
            full_start, full_end: the FULL pre-fetch span (YYYY-MM-DD); fall
                back to config data.universe.* if None.
            universe: Stock universe (hs300, zz500, all_a); falls back to config.
            train_start, train_end, test_start, test_end: train/test windows
                (YYYY-MM-DD); fall back to config split dates
                (train_start_date / train_end_date / test_start_date /
                test_end_date) if None. Omit all four to get the full span in
                both .train and .test.
            force_refresh: accepted for signature compatibility, ignored here —
                re-fetching is done via `load_datasets.py --force-refresh`.

        Returns:
            DatasetBundle with .full / .train / .test, each a
            (price_data, fundamental_data, industry_data) triple. The loader's
            .price_data / .fundamental_data / .industry_data are set to the
            FULL triple; .train_data / .test_data hold the sliced tuples.
        """
        # full_start = full_start or self.data_config['universe']['start_date']
        # full_end = full_end or self.data_config['universe']['end_date']
        universe = universe or self.data_config['universe']['index']
        d = self.data_config
        train_start = train_start or d.get('train_start_date', full_start)
        train_end = train_end or d.get('train_end_date', full_end)
        test_start = test_start or d.get('test_start_date', full_start)
        test_end = test_end or d.get('test_end_date', full_end)

        print(f"[loader] Retrieving local dataset: {universe}")

        bundle = retrieve_dataset(
            universe, train_start, train_end, test_start, test_end, full_start, full_end
        )

        self.price_data, self.fundamental_data, self.industry_data = bundle.full
        self.train_data = bundle.train
        self.test_data = bundle.test

        return bundle

    def fetch_and_store_dataset(
        self,
        start_date: str = None,
        end_date: str = None,
        universe: str = None,
        source: str = None,
        force_refresh: bool = False,
    ) -> DatasetBundle:
        """
        Fetch real data from the network source, preprocess it, and persist it
        to the local dataset store (datasets/{universe}_{start}_{end}.pkl).

        This is the ONLY method that downloads data. It is intended to be called
        by the standalone `load_datasets.py` CLI. Because the already-processed
        triple is stored, subsequent load_data() calls are pure in-memory
        retrieval (no re-preprocessing, no network).

        Args:
            start_date, end_date, universe: the FULL pre-fetch span (config
                fallback applies). NOTE: these name the full archive span, not a
                train/test split — splitting happens at retrieve time.
            source: data source override ('westock'/'tushare'/'akshare'/'auto')
            force_refresh: if False and the archive already exists, skip the
                fetch and return the stored bundle; if True, always re-fetch.

        Returns:
            DatasetBundle (see retrieve_dataset) — .full holds the fetched
            triple; .train/.test equal .full when no split bounds are given.
            The (price_data, fundamental_data, industry_series) triple.
        """
        start_date = start_date or self.data_config['universe']['start_date']
        end_date = end_date or self.data_config['universe']['end_date']
        universe = universe or self.data_config['universe']['index']
        source = source or self.data_config.get('source', 'auto')

        path = dataset_path(universe, start_date, end_date)
        if os.path.exists(path) and not force_refresh:
            print(f"[loader] Dataset already exists, skipping fetch: {path}")
            return retrieve_dataset(universe, start_date, end_date)

        print(f"[loader] Fetching data from {start_date} to {end_date}, universe: {universe} (source={source})")

        # Centralized real-data loader. force_refresh=True bypasses the old
        # date-blind data/cache.pkl so we always pull the requested range
        # straight from the source rather than a possibly-wrong cached slice.
        price_data, fundamental_data, industry_data = load_real_data(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            force_refresh=True,
            config=self.config,
        )

        # Filter to actual trading days only (drop holiday weekdays that are
        # all-NaN in close). Mirrors the historical load_data() preprocessing.
        if price_data and 'close' in price_data:
            actual_trading_days = price_data['close'].dropna(how='all').index
            n_removed = len(price_data['close'].index) - len(actual_trading_days)
            if n_removed > 0:
                print(f"  [loader] Removed {n_removed} non-trading days (holidays) from index")
            price_data = {k: v.reindex(actual_trading_days) for k, v in price_data.items()}
            if fundamental_data:
                fundamental_data = {
                    k: v.reindex(actual_trading_days) for k, v in fundamental_data.items()
                }
            print(f"  [loader] Final trading-day index: {len(actual_trading_days)} days")

        # Preprocess
        price_data = self._preprocess_price_data(price_data)
        fundamental_data = self._preprocess_fundamental_data(fundamental_data)

        # Persist the fully-processed triple to the local store
        os.makedirs(DATASETS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pd.to_pickle((price_data, fundamental_data, industry_data), f)
        print(f"[loader] Saved dataset -> {path}")

        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.industry_data = industry_data

        return price_data, fundamental_data, industry_data

    def _preprocess_price_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess price data.
        
        Args:
            price_data: Dict of DataFrames (open, high, low, close, volume, amount)
            
        Returns:
            Preprocessed price data
        """
        processed = {}
        
        for col, df in price_data.items():
            # Fill missing values
            if self.preprocessing_config['fill_method'] == 'forward':
                df = df.ffill()
            elif self.preprocessing_config['fill_method'] == 'backward':
                # DEPRECATED: backward fill introduces look-ahead bias; treat as forward fill
                df = df.ffill()
            elif self.preprocessing_config['fill_method'] == 'interpolate':
                df = df.interpolate(method='linear')
            
            # Handle outliers
            if self.preprocessing_config['outlier_method'] == 'winsorize':
                df = self._winsorize(df, self.preprocessing_config['outlier_threshold'])
            elif self.preprocessing_config['outlier_method'] == 'clip':
                df = self._clip_outliers(df, self.preprocessing_config['outlier_threshold'])
            
            processed[col] = df
        
        return processed
    
    def _preprocess_fundamental_data(self, fundamental_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess fundamental data.
        
        Args:
            fundamental_data: Dict of DataFrames
            
        Returns:
            Preprocessed fundamental data
        """
        processed = {}
        
        for col, df in fundamental_data.items():
            # By this point the index has already been aligned to actual trading days
            # (holidays removed) in load_data() via dropna(how='all') on price_data['close'].
            # So we only need to forward-fill within each column.
            # We deliberately do NOT bfill: bfill would use future report values to fill
            # the period before the first report, introducing look-ahead bias.
            # Pre-first-report NaN values are left as NaN and handled downstream.
            df = df.ffill()
            processed[col] = df
        
        return processed
    
    def _winsorize(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Winsorize data (clip at percentiles).
        
        Args:
            df: DataFrame to winsorize
            threshold: Percentile threshold (e.g., 3.0 means clip at 0.5% and 99.5%)
            
        Returns:
            Winsorized DataFrame
        """
        lower = threshold / 100
        upper = 1 - threshold / 100
        
        result = df.copy()
        for col in df.columns:
            q_low = df[col].quantile(lower)
            q_high = df[col].quantile(upper)
            result[col] = df[col].clip(lower=q_low, upper=q_high)
        
        return result
    
    def _clip_outliers(self, df: pd.DataFrame, sigma: float) -> pd.DataFrame:
        """
        Clip outliers using sigma rule.
        
        Args:
            df: DataFrame to clip
            sigma: Number of standard deviations
            
        Returns:
            Clipped DataFrame
        """
        result = df.copy()
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            result[col] = df[col].clip(lower=mean - sigma * std, upper=mean + sigma * std)
        
        return result
    
    def get_returns(
        self,
        price_col: str = 'close',
        period: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate stock returns.
        
        Args:
            price_col: Price column to use ('open', 'high', 'low', 'close')
            period: Return period (1 = daily, 5 = weekly, etc.)
            
        Returns:
            DataFrame of returns
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_data() first.")
        
        close_prices = self.price_data[price_col]
        returns = close_prices.pct_change(period).shift(-period)
        
        return returns
    
    def get_market_cap(self) -> pd.DataFrame:
        """
        Get market capitalization data.
        
        Returns:
            DataFrame of market cap
        """
        if self.fundamental_data is None:
            raise ValueError("Fundamental data not loaded. Call load_data() first.")
        
        return self.fundamental_data.get('market_cap', pd.DataFrame())
    
    def get_industry(self) -> pd.Series:
        """
        Get industry classification.
        
        Returns:
            Series of industry labels
        """
        if self.industry_data is None:
            raise ValueError("Industry data not loaded. Call load_data() first.")
        
        return self.industry_data
    
    def split_data(
        self,
        test_ratio: float = 0.2,
        train_start_date: str = None,
        train_end_date: str = None,
        test_start_date: str = None,
        test_end_date: str = None,
        context_days: int = 30,
    ) -> Tuple[Dict, Dict]:
        """
        Split ALL loaded data into train and test sets.

        This splits the full price_data dict (all columns: open/high/low/close/volume/amount),
        fundamental_data, and industry_data — NOT just close prices.

        **Context window**: To compute factor values (e.g. rolling means) on the first
        few test dates, test_data is prepended with `context_days` of training data.
        The logical test boundary is unchanged; the context window is metadata only.
        Callers should filter results to dates >= test_start_date.

        Args:
            test_ratio: Ratio of test data (used only when train_end_date is not given)
            method: Split method ('chronological', 'rolling')
            train_end_date: Explicit train/test split date (YYYY-MM-DD).
                         If given, overrides test_ratio.
            test_start_date: Explicit test start date. If given (and train_end_date is
                         None), it defines the train/test boundary — training ends the
                         day before test_start_date. If None, the boundary comes from
                         train_end_date (or test_ratio). Recorded in _meta for downstream
                         result cropping; the returned test_data still prepends context_days
                         of training history, so its first row precedes test_start_date.
            train_start_date: Explicit first day of training (YYYY-MM-DD). If None, training
                         starts at the first available date. Clamps the train window's start.
            test_end_date: Explicit last day of testing (YYYY-MM-DD, inclusive). If None,
                         testing runs to the last available date. Clamps the test window's end.
            context_days: Number of trading days from end of training to prepend as context
                         for factor calculation. Default 30 (covers typical rolling windows).

        Returns:
            Tuple of (train_data_dict, test_data_dict)
            Each dict has keys: 'price_data', 'fundamental_data', 'industry_data',
            and a '_meta' key with context info.
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_data() first.")

        close_prices = self.price_data['close']
        n_days = len(close_prices)
        dates = close_prices.index

        # --- Determine split index (train/test boundary) ---
        if train_end_date is not None:
            # Use explicit train-end boundary
            train_end = pd.Timestamp(train_end_date)
            split_idx = int(np.searchsorted(dates, train_end))
            if split_idx >= n_days:
                raise ValueError(f"train_end_date {train_end_date} is at or after the last date {dates[-1]}")
            if split_idx <= 0:
                raise ValueError(f"train_end_date {train_end_date} is before the first date {dates[0]}")
            # Guard against a contradictory explicit test_start_date
            if test_start_date is not None and pd.Timestamp(test_start_date) <= train_end:
                raise ValueError(
                    f"test_start_date {test_start_date} must be strictly after train_end_date {train_end_date}"
                )
        elif test_start_date is not None:
            # Derive the boundary from the explicit test start: training ends the
            # day right before test_start_date, so split_idx points AT test_start_date
            # (which then becomes the first real test day). This makes test_start_date
            # actually define the split when train_end_date is not supplied.
            test_start = pd.Timestamp(test_start_date)
            split_idx = int(np.searchsorted(dates, test_start))
            if split_idx <= 0:
                raise ValueError(f"test_start_date {test_start_date} is at or before the first date {dates[0]}")
            if split_idx >= n_days:
                raise ValueError(f"test_start_date {test_start_date} is at or after the last date {dates[-1]}")
        else:
            split_idx = int(n_days * (1 - test_ratio))

        # --- Determine train start index ---
        # train_start_date is the FIRST day of training (inclusive). Defaults to
        # the first available date when omitted.
        if train_start_date is not None:
            train_start = pd.Timestamp(train_start_date)
            train_start_idx = int(np.searchsorted(dates, train_start))
            if train_start_idx < 0:
                train_start_idx = 0
            if train_start_idx >= split_idx:
                raise ValueError(
                    f"train_start_date {train_start_date} is at or after train_end_date {train_end_date}"
                )
        else:
            train_start_idx = 0

        # --- Determine test end index ---
        # test_end_date is the LAST day of testing (inclusive). Defaults to the
        # last available date when omitted. +1 makes the endpoint inclusive to
        # match the user's intent ("up to and including test_end_date").
        if test_end_date is not None:
            test_end = pd.Timestamp(test_end_date)
            test_end_idx = int(np.searchsorted(dates, test_end)) + 1
            if test_end_idx <= split_idx:
                raise ValueError(
                    f"test_end_date {test_end_date} is at or before the train/test split"
                )
            if test_end_idx > n_days:
                test_end_idx = n_days
        else:
            test_end_idx = n_days

        # --- Determine context start index ---
        # test_data includes context_days from end of training so factor
        # expressions with rolling/ts_* functions have enough history
        context_start = max(0, split_idx - context_days)
        if context_start < split_idx:
            context_dates = dates[context_start:split_idx]
        else:
            context_dates = pd.DatetimeIndex([])

        # --- Split price_data (ALL columns) ---
        train_price = {}
        test_price = {}
        for col in self.price_data:
            train_price[col] = self.price_data[col].iloc[train_start_idx:split_idx]
            test_price[col] = self.price_data[col].iloc[context_start:test_end_idx]  # includes context

        # --- Split fundamental_data (all keys) ---
        train_fund = {}
        test_fund = {}
        if self.fundamental_data:
            for key in self.fundamental_data:
                df = self.fundamental_data[key]
                train_fund[key] = df.iloc[train_start_idx:split_idx]
                test_fund[key] = df.iloc[context_start:test_end_idx]  # includes context

        # --- Split industry_data (Series) ---
        train_industry = None
        test_industry = None
        if self.industry_data is not None:
            # industry_data is typically a Series or DataFrame with stock codes as index
            # It's cross-sectional, not time-series — use the latest available for both
            train_industry = self.industry_data
            test_industry = self.industry_data

        train_data = {
            'price_data': train_price,
            'fundamental_data': train_fund,
            'industry_data': train_industry,
        }
        test_data = {
            'price_data': test_price,
            'fundamental_data': test_fund,
            'industry_data': test_industry,
            # Metadata: the REAL test period boundary, used to crop results
            '_meta': {
                'context_days': context_days,
                'context_start_idx': int(context_start),
                'train_start_idx': int(train_start_idx),
                'test_end_idx': int(test_end_idx),
                'test_start_idx': int(split_idx),
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'test_start_date': test_start_date if test_start_date is not None
                                  else (str(dates[split_idx].date()) if split_idx < n_days else None),
                'test_end_date': test_end_date,
            },
        }

        train_n = len(train_price['close'])
        test_n = len(test_price['close'])
        ctx_n = len(context_dates)
        print(f"  [split] Train: {train_price['close'].index[0].date()} ~ {train_price['close'].index[-1].date()} ({train_n} days)")
        if ctx_n > 0:
            print(f"  [split] Test:  {test_price['close'].index[0].date()} ~ {test_price['close'].index[-1].date()} ({test_n} days, incl. {ctx_n}d context)")
        else:
            print(f"  [split] Test:  {test_price['close'].index[0].date()} ~ {test_price['close'].index[-1].date()} ({test_n} days, no context)")
        print(f"  [split] Real test boundary: {test_data['_meta']['test_start_date']}")

        return train_data, test_data


def load_sample_data(n_stocks: int = 100, n_days: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load sample data for quick testing.
    
    Args:
        n_stocks: Number of stocks
        n_days: Number of trading days
        
    Returns:
        Tuple of (price_data, fundamental_data, industry_series)
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    # Price data
    price_data = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in ['open', 'high', 'low', 'close']:
            base_price = 10 + np.random.randn(n_stocks) * 5
            returns = np.random.randn(n_days, n_stocks) * 0.02
            prices = base_price * np.exp(np.cumsum(returns, axis=0))
            price_data[col] = pd.DataFrame(prices, index=dates, columns=stock_codes)
        else:
            volume = np.abs(np.random.randn(n_days, n_stocks) * 1e6)
            price_data[col] = pd.DataFrame(volume, index=dates, columns=stock_codes)
    
    # Fundamental data — same realistic distributions as _generate_synthetic_data
    fundamental_data = {
        'pe': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.6 + np.log(20)),
            index=dates, columns=stock_codes
        ),
        'pb': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.5 + np.log(2)),
            index=dates, columns=stock_codes
        ),
        'ps': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 0.7 + np.log(2)),
            index=dates, columns=stock_codes
        ),
        'roe': pd.DataFrame(
            np.clip(np.random.randn(n_days, n_stocks) * 0.08 + 0.10, -0.5, 0.8),
            index=dates, columns=stock_codes
        ),
        'market_cap': pd.DataFrame(
            np.exp(np.random.randn(n_days, n_stocks) * 1.2 + np.log(50)),
            index=dates, columns=stock_codes
        ),
    }
    
    # Industry data
    industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy']
    industry_series = pd.Series(
        np.random.choice(industries, size=n_stocks),
        index=stock_codes
    )
    
    return price_data, fundamental_data, industry_series


if __name__ == '__main__':
    # Demo
    print("=== Data Loader Demo ===\n")
    
    loader = DataLoader()
    bundle = loader.load_data(
        full_start='2022-01-01',
        full_end='2024-12-31',
        universe='hs300',
    )
    price_data, fundamental_data, industry_data = bundle.full
    
    print(f"Price data keys: {list(price_data.keys())}")
    print(f"Close prices shape: {price_data['close'].shape}")
    print(f"Fundamental data keys: {list(fundamental_data.keys())}")
    print(f"Number of stocks: {len(industry_data)}")
    print(f"Industries: {industry_data.value_counts().to_dict()}")
    
    # Get returns
    returns = loader.get_returns()
    print(f"\nReturns shape: {returns.shape}")
    print(f"Mean return: {returns.mean().mean():.6f}")
    print(f"Return std: {returns.std().mean():.6f}")
    
    print("\n=== Demo Complete ===")
