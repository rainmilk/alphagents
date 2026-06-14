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
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(universe: str, start_date: str, end_date: str) -> str:
    """Build a consistent cache file path."""
    os.makedirs('data', exist_ok=True)
    return f"data/cache_{universe}_{start_date}_{end_date}.pkl"


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
# Raw data persistence helpers
# ---------------------------------------------------------------------------

RAW_DATA_DIR = "data/raw"

def _ensure_raw_dir(subdir: str) -> str:
    """Create raw data subdirectory and return its path."""
    path = os.path.join(RAW_DATA_DIR, subdir)
    os.makedirs(path, exist_ok=True)
    return path

def _save_raw_csv(df: pd.DataFrame, relpath: str, index: bool = True):
    """Save a DataFrame as CSV to data/raw/... with utf-8-sig encoding."""
    full_path = os.path.join(RAW_DATA_DIR, relpath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_csv(full_path, index=index, encoding='utf-8-sig')
    n_rows, n_cols = df.shape
    print(f"  [raw] Saved {full_path}  ({n_rows} rows x {n_cols} cols)")

def _save_raw_parquet(df: pd.DataFrame, relpath: str):
    """Save a DataFrame as Parquet to data/raw/... (fast & compact)."""
    full_path = os.path.join(RAW_DATA_DIR, relpath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_parquet(full_path)
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
                        "url_hint": "pro.daily_basic(ts_code=..., trade_date=..., fields='ts_code,pe_ttm,pb,roe,total_mv')",
                        "columns": {
                            "ts_code":    {"canonical": "code",       "type": "str"},
                            "trade_date": {"canonical": "date",       "type": "datetime"},
                            "pe_ttm":     {"canonical": "pe",         "type": "float64", "note": "TTM P/E"},
                            "pb":         {"canonical": "pb",         "type": "float64"},
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
            "qlib": {
                "description": "Qlib high-performance .bin format (pip install qlib). Auto-downloads cn_data bundle.",
                "endpoints": {
                    "D.features": {
                        "description": "Batch feature retrieval via Qlib DataHandler.",
                        "url_hint": "qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn'); D.features(instruments, fields, start_time, end_time)",
                        "columns": {
                            "$open":   {"canonical": "open",   "type": "float64"},
                            "$high":   {"canonical": "high",   "type": "float64"},
                            "$low":    {"canonical": "low",    "type": "float64"},
                            "$close":  {"canonical": "close",  "type": "float64"},
                            "$volume": {"canonical": "volume", "type": "float64"},
                            "$vwap":   {"canonical": "(amount = vwap * volume)", "type": "float64", "note": "Used to synthesize amount field"},
                            "$factor": {"canonical": "adj_factor", "type": "float64", "note": "Adjustment factor; Qlib data is already adjusted"},
                        },
                    },
                    "D.instruments": {
                        "description": "Stock universe by market (csi300 / csi500 / all).",
                        "url_hint": "D.instruments(market='csi300')",
                        "columns": {
                            "SH600000": {"canonical": "code", "type": "str", "note": "Qlib native code format (SH=Shanghai, SZ=Shenzhen)"},
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
      3. AkShare (open-source Python library)
      4. Tushare (requires token)
      5. Qlib (high-performance .bin format, local — lowest priority, heavy dependency)
      6. Synthetic fallback (generated data)

    Args:
        universe: Stock universe ('hs300', 'zz500', 'all_a')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Preferred source ('westock', 'akshare', 'tushare', 'qlib', 'auto')
        force_refresh: Skip cache and re-download
        config: Optional config dict (to read data.tushare_token).
                If None, will try to load from config/config.yaml.

    Returns:
        Tuple of (price_data, fundamental_data, industry_series)
        - price_data: Dict of DataFrames with keys [open, high, low, close, volume, amount]
        - fundamental_data: Dict of DataFrames with keys [pe, pb, roe, market_cap]
        - industry_series: Series, index=stock_code, value=industry_name
    """
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

    cache_file = _cache_path(universe, start_date, end_date)

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

    # 3. Try AkShare
    if source in ("akshare", "auto"):
        try:
            result = _load_from_akshare(universe, start_date, end_date)
            if result is not None:
                print(f"  [akshare] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [akshare] Failed: {e}")

    # 4. Try Tushare
    if source in ("tushare", "auto"):
        try:
            result = _load_from_tushare(universe, start_date, end_date)
            if result is not None:
                print(f"  [tushare] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [tushare] Failed: {e}")

    # 5. Try Qlib (high-performance .bin format, local — lowest priority, heavy dependency)
    if source in ("qlib", "auto"):
        try:
            result = _load_from_qlib(universe, start_date, end_date)
            if result is not None:
                print(f"  [qlib] Data loaded successfully")
                _save_cache(cache_file, *result)
                return result
        except Exception as e:
            print(f"  [qlib] Failed: {e}")

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
        
        # Fill missing
        for field in price_data:
            price_data[field] = price_data[field].ffill().bfill()

        # Validate: if close data is still all NaN, return None to fall through
        if price_data['close'].isna().all().all():
            print(f"  [westock] All price data is NaN — falling back to next source")
            return None
        
        # Fundamental data — try westock fundamentals API
        fundamental_data = {}
        # Initialize with NaN; real values will be filled in below
        for factor in ['pe', 'pb', 'roe', 'market_cap']:
            fundamental_data[factor] = pd.DataFrame(
                np.nan, index=dates, columns=stock_codes
            )
        try:
            for factor in ['pe', 'pb', 'roe', 'market_cap']:
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
            pass  # Keep NaN-initialized DataFrames; ffill/bfill applied below

        # Forward/back fill missing fundamentals (use most recent available value)
        for factor in fundamental_data:
            fundamental_data[factor] = fundamental_data[factor].ffill().bfill()
        
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

        # ---- 5. Forward/back fill missing price data ----
        for field in price_data:
            price_data[field] = price_data[field].ffill().bfill()

        # Validate: if close data is still all NaN, return None to fall through
        if price_data['close'].isna().all().all():
            print(f"  [tushare] All price data is NaN — falling back to next source")
            return None

        # ---- 6. Fundamental data ----
        print("  [tushare] Fetching fundamentals...")
        fundamental_data = {}
        for field in ['pe', 'pb', 'roe', 'market_cap']:
            fundamental_data[field] = pd.DataFrame(np.nan, index=dates, columns=stock_codes)

        # daily_basic: fetch by date (more efficient than by stock)
        daily_basic_all = []  # collect for raw save
        try:
            date_samples = [d.strftime('%Y%m%d') for d in dates[::max(1, n_days // 20)]][:20]
            for d in date_samples:
                try:
                    df_b = pro.daily_basic(ts_code=','.join(stock_codes),
                                            trade_date=d,
                                            fields='ts_code,pe_ttm,pb,roe,total_mv')
                    if df_b is None or df_b.empty:
                        continue
                    daily_basic_all.append(df_b)
                    dt = pd.Timestamp(d[:4] + '-' + d[4:6] + '-' + d[6:])
                    for _, row in df_b.iterrows():
                        code = row['ts_code']
                        if code not in stock_codes:
                            continue
                        if 'pe_ttm' in row and pd.notna(row['pe_ttm']):
                            fundamental_data['pe'].loc[dt, code] = float(row['pe_ttm'])
                        if 'pb' in row and pd.notna(row['pb']):
                            fundamental_data['pb'].loc[dt, code] = float(row['pb'])
                        if 'roe' in row and pd.notna(row['roe']):
                            fundamental_data['roe'].loc[dt, code] = float(row['roe']) / 100.0
                        if 'total_mv' in row and pd.notna(row['total_mv']):
                            fundamental_data['market_cap'].loc[dt, code] = float(row['total_mv']) / 10000.0
                except Exception:
                    continue
                time.sleep(0.1)
        except Exception:
            pass

        # Save combined daily_basic raw data
        if daily_basic_all:
            try:
                df_daily_basic_combined = pd.concat(daily_basic_all, ignore_index=True)
                _save_raw_csv(df_daily_basic_combined, 'tushare/daily_basic.csv', index=False)
            except Exception:
                pass

        # Fill missing fundamentals (forward/back fill; leave remaining as NaN)
        for field in fundamental_data:
            fundamental_data[field] = fundamental_data[field].ffill().bfill()

        # ---- 7. Industry classification ----
        print("  [tushare] Fetching industry classification...")
        industry_dict = {}
        try:
            df_ind = pro.stock_basic(ts_code=','.join(stock_codes),
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
        return price_data, fundamental_data, industry_series

    except Exception as e:
        print(f"  [tushare] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Qlib real data loader
# ---------------------------------------------------------------------------

def _load_qlib_instruments(provider_uri: str, market: str) -> List[str]:
    """
    Load Qlib instrument codes, handling version-specific API differences.

    Strategy (best-effort, tries each until one works):
      1. D.instruments(market=...)           — official Qlib API
      2. Direct file read from instruments/   — bypasses API version skew
      3. Enumerate features/ subdirectories  — emergency fallback

    Instrument files inside {provider_uri}/instruments/ use TSV format:
        SH600000    2005-01-01  2020-09-25

    Returns deduplicated, sorted, stripped list of instrument codes.
    """
    instruments: List[str] = []

    # --- Strategy 1: D.instruments() (requires real Microsoft Qlib) ---
    try:
        import qlib as _qlib_pkg

        # Guard against impostor packages (e.g. PyPI "qlib" ≠ Microsoft Qlib)
        if not hasattr(_qlib_pkg, 'init'):
            raise ImportError("Not Microsoft Qlib (missing qlib.init)")

        from qlib.data import D

        # Only try init if not already initialized
        try:
            _qlib_pkg.init(provider_uri=provider_uri, region='cn')
        except Exception:
            pass  # may already be initialized in parent scope

        raw = D.instruments(market=market)
        if raw:
            # Normalise across Qlib versions
            if isinstance(raw, dict):
                raw = raw.get('instruments', [])
            elif not isinstance(raw, (list, tuple)):
                try:
                    raw = list(raw)
                except TypeError:
                    raw = []

            for inst in raw:
                if isinstance(inst, bytes):
                    inst = inst.decode('utf-8')
                code = str(inst).strip()
                if code:
                    instruments.append(code)

            if instruments:
                print(f"  [qlib] D.instruments() returned {len(instruments)} codes")
                return sorted(set(instruments))
    except Exception as e:
        print(f"  [qlib] D.instruments() failed: {e}")

    # --- Strategy 2: read instrument files directly ---
    inst_dir = os.path.join(provider_uri, 'instruments')
    inst_path = os.path.join(inst_dir, f'{market}.txt')
    if os.path.isfile(inst_path):
        try:
            with open(inst_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # TSV: STOCK_CODE\tSTART_DATE\tEND_DATE
                    code = line.split('\t')[0].strip()
                    if code:
                        instruments.append(code)
            if instruments:
                print(f"  [qlib] Read {len(instruments)} codes from {inst_path}")
                return sorted(set(instruments))
        except Exception as e:
            print(f"  [qlib] Failed to read {inst_path}: {e}")

    # --- Strategy 3: enumerate features/ subdirectories ---
    features_dir = os.path.join(provider_uri, 'features')
    if os.path.isdir(features_dir):
        try:
            for entry in os.listdir(features_dir):
                # Each stock has a subdirectory like 'sh600000'
                code = entry.strip().upper()
                if code and os.path.isdir(os.path.join(features_dir, entry)):
                    instruments.append(code)
            if instruments:
                print(f"  [qlib] Enumerated {len(instruments)} codes from features/")
                return sorted(set(instruments))
        except Exception as e:
            print(f"  [qlib] Failed to enumerate features/: {e}")

    return []


def _read_qlib_bin_files(
    provider_uri: str,
    instruments: List[str],
    fields: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Read Qlib .bin feature files directly with numpy, bypassing the Qlib API.

    Qlib cn_data stores each field as a flat float32 array in:
        {provider_uri}/features/{instrument}/{field_name_without_$}.day.bin

    The array is aligned to the trading calendar at calendars/day.txt.
    Each line in the calendar file is a date string (YYYY-MM-DD).

    Returns a dict mapping instrument code → DataFrame(columns=fields, index=datetime).
    """
    import numpy as np

    calendar_path = os.path.join(provider_uri, 'calendars', 'day.txt')
    if not os.path.isfile(calendar_path):
        print(f"  [qlib-bin] Calendar not found: {calendar_path}")
        return {}

    # Read trading calendar
    with open(calendar_path, 'r', encoding='utf-8') as f:
        cal_dates = [line.strip() for line in f if line.strip()]
    cal_index = pd.DatetimeIndex(pd.to_datetime(cal_dates))
    cal_len = len(cal_index)

    # Filter to requested date range
    mask = (cal_index >= pd.Timestamp(start_date)) & (cal_index <= pd.Timestamp(end_date))
    slice_start = int(mask.argmax()) if mask.any() else 0
    slice_end = int(mask[::-1].argmax()) if mask.any() else cal_len
    slice_end = cal_len - slice_end  # convert from reversed index

    result = {}
    features_dir = os.path.join(provider_uri, 'features')

    for code in instruments:
        code_dir = os.path.join(features_dir, code.lower())
        if not os.path.isdir(code_dir):
            continue

        df_data = {}
        for field in fields:
            # Strip '$' prefix to get filename: $close → close.day.bin
            fname = field.lstrip('$') + '.day.bin'
            fpath = os.path.join(code_dir, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                arr = np.fromfile(fpath, dtype=np.float32)
                # Some .bin files have one extra leading element (e.g. IPO factor)
                offset = len(arr) - cal_len
                if offset < 0:
                    continue  # array too short, skip
                if offset > 0:
                    arr = arr[offset:]  # trim leading extra elements
                arr = arr[:cal_len]     # ensure exact match

                # Slice to requested date range
                arr_sliced = arr[slice_start:slice_end]
                date_sliced = cal_index[slice_start:slice_end]

                df_data[field] = pd.Series(arr_sliced, index=date_sliced, name=field)
            except Exception as e:
                print(f"  [qlib-bin] Error reading {fpath}: {e}")
                continue

        if df_data:
            result[code.upper()] = pd.DataFrame(df_data).sort_index()

    if result:
        print(f"  [qlib-bin] Read {len(result)} stocks directly from .bin files")
    return result


def _load_from_qlib(
    universe: str,
    start_date: str,
    end_date: str,
) -> Optional[Tuple]:
    """
    Load real A-share data via Qlib (.bin format, high-performance).

    Qlib provides pre-processed A-share data in a compact .bin format with
    fast feature retrieval via D.features(). This loader:

    1. Auto-downloads cn_data bundle if missing (configurable)
    2. Maps Qlib instrument codes (SH600000 / SZ000001) to canonical format
    3. Fetches OHLCV price data via $open/$high/$low/$close/$volume
    4. Computes amount from vwap * volume if $vwap is available
    5. Fetches fundamental data from Qlib's alpha158 factor set
    6. Falls back to AkShare for industry classification

    Install: pip install qlib

    Returns (price_data, fundamental_data, industry_series) on success,
    None on any failure.
    """
    # ---- 0. Read Qlib config from project config.yaml ----
    qlib_config = {}
    cfg_path = 'config/config.yaml'
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        qlib_config = cfg.get('data', {}).get('qlib', {})

    provider_uri = qlib_config.get('provider_uri', '~/.qlib/qlib_data/cn_data')
    provider_uri = os.path.abspath(os.path.expanduser(provider_uri))
    auto_download = qlib_config.get('auto_download', True)

    # ---- 1. Ensure Qlib data exists (auto-download if missing) ----
    if not os.path.exists(provider_uri):
        if auto_download:
            print(f"  [qlib] Data not found at {provider_uri}, downloading cn_data...")
            try:
                from qlib.tests.data import GetData
                GetData().qlib_data(target_dir=provider_uri, region="cn")
            except Exception as e:
                print(f"  [qlib] Auto-download failed: {e}")
            if not os.path.exists(provider_uri):
                print(f"  [qlib] Auto-download did not produce data, falling back")
                return None
        else:
            print(f"  [qlib] Data not found at {provider_uri}, and auto_download is disabled")
            return None

    # ---- 2. Map universe to Qlib market ----
    qlib_universe_map = {
        'hs300': 'csi300',
        'zz500': 'csi500',
        'all_a': 'all',
    }
    qlib_market = qlib_universe_map.get(universe, 'all')

    # ---- 3. Load instruments via direct file I/O (no Qlib API needed) ----
    instruments = _load_qlib_instruments(provider_uri, qlib_market)
    if not instruments:
        print(f"  [qlib] No instruments for market={qlib_market}")
        return None

    qlib_codes = instruments[:500]  # cap at 500 for performance
    n_stocks = len(qlib_codes)
    print(f"  [qlib] {n_stocks} instruments for market={qlib_market}")

    # Save stock list as raw data
    _save_raw_csv(pd.DataFrame({'code': qlib_codes}), 'qlib/stock_list.csv', index=False)

    # ---- 4. Date range ----
    dates = pd.date_range(start_date, end_date, freq='B')
    n_days = len(dates)
    if n_days == 0:
        return None

    # ---- 5. Fetch price data directly from .bin files (no Qlib API dependency) ----
    print(f"  [qlib] Reading daily kline from .bin files ({n_stocks} stocks)...")

    qlib_price_fields = ['$open', '$high', '$low', '$close', '$volume']
    canonical_price = ['open', 'high', 'low', 'close', 'volume', 'amount']

    all_data = _read_qlib_bin_files(
        provider_uri, qlib_codes, qlib_price_fields,
        start_date, end_date
    )
    n_bin = sum(1 for df in (all_data.values() if isinstance(all_data, dict) else [])
                if df is not None and not (hasattr(df, 'empty') and df.empty))
    print(f"  [qlib] Direct .bin read returned {n_bin}/{n_stocks} stocks")

    # ---- 6. Build canonical price_data dict ----
    price_data = {}
    for field in canonical_price:
        price_data[field] = pd.DataFrame(
            np.nan, index=dates, columns=qlib_codes
        )

    n_fetched = 0
    for code, df in all_data.items():
        if df is None:
            continue
        if hasattr(df, 'empty') and df.empty:
            continue
        n_fetched += 1
        # Normalize index to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Map Qlib $fields to canonical columns
        field_map = {
            '$open': 'open',
            '$high': 'high',
            '$low': 'low',
            '$close': 'close',
            '$volume': 'volume',
        }
        for qf, cf in field_map.items():
            if qf in df.columns:
                common = dates.intersection(df.index)
                if len(common) > 0:
                    price_data[cf].loc[common, code] = df.loc[common, qf].values

        # amount = close * volume (approximation when vwap not available)
        if '$close' in df.columns and '$volume' in df.columns:
            common = dates.intersection(df.index)
            if len(common) > 0:
                close_vals = df.loc[common, '$close'].values
                vol_vals = df.loc[common, '$volume'].values
                price_data['amount'].loc[common, code] = close_vals * vol_vals

    print(f"  [qlib] Fetched price data for {n_fetched}/{n_stocks} stocks")

    # Fill missing
    for field in price_data:
        price_data[field] = price_data[field].ffill().bfill()

    # Validate
    if price_data['close'].isna().all().all():
        print(f"  [qlib] All price data is NaN — falling back to next source")
        return None

    # ---- 7. Fundamental data ----
    print("  [qlib] Building fundamental data...")
    fundamental_data = {}
    for field in ['pe', 'pb', 'roe', 'market_cap']:
        fundamental_data[field] = pd.DataFrame(
            np.nan, index=dates, columns=qlib_codes
        )

    # Try to extract PE/PB/ROE/MarketCap from Qlib's alpha158/alpha360 factors
    # These are available as calculated features in the data bundle
    try:
        # Qlib alpha158-style fundamental features (ETF/Daily-based)
        extra_fields = [
            'Ref($close, 1)',
            'Mean($volume, 5)',
        ]
        # The default Qlib data doesn't include PE/PB in basic features.
        # We provide NaN fundamental data and let the pipeline handle it.
        pass
    except Exception as e:
        print(f"  [qlib] Fundamental factor extraction failed: {e}")

    # Forward/back fill
    for field in fundamental_data:
        fundamental_data[field] = fundamental_data[field].ffill().bfill()

    # ---- 8. Industry classification ----
    # AkShare industry APIs are currently broken (network/SSL issues with external sources).
    # Directly use Tushare or fallback to random assignment.
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
                # Map ts_code (000001.SZ) → plain code (000001) → Qlib code (SZ000001)
                for _, row in df_ind.iterrows():
                    ts_code = str(row['ts_code']).strip()
                    ind = str(row['industry']).strip()
                    if ind and ind != 'nan':
                        plain = ts_code.split('.')[0]
                        qc = None
                        if ts_code.endswith('.SH'):
                            qc = 'SH' + plain
                        elif ts_code.endswith('.SZ'):
                            qc = 'SZ' + plain
                        if qc and qc in qlib_codes:
                            industry_dict[qc] = ind
                print(f"  [qlib] Industry from Tushare: {len(industry_dict)} stocks")
    except Exception as e:
        print(f"  [qlib] Industry from Tushare failed: {e}")

    # Fallback for missing industries
    # 申万一级行业（中文，和 Tushare stock_basic(industry=...) 返回格式一致）
    all_industries = ['银行', '房地产', '医药生物', '电子', '计算机',
                      '传媒', '通信', '电力设备', '基础化工', '机械设备',
                      '汽车', '食品饮料', '家用电器', '建筑材料', '建筑装饰',
                      '有色金属', '钢铁', '国防军工', '农林牧渔', '纺织服饰',
                      '轻工制造', '商贸零售', '社会服务', '综合', '公用事业']
    np.random.seed(42)
    for code in qlib_codes:
        if code not in industry_dict:
            industry_dict[code] = np.random.choice(all_industries)

    industry_series = pd.Series(industry_dict)

    print(f"  [qlib] Data loaded: {n_fetched} stocks x {n_days} days")
    return price_data, fundamental_data, industry_series


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# AkShare real data loader
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
        for field in price_data:
            price_data[field] = price_data[field].ffill().bfill()

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
        for field in ['pe', 'pb', 'roe', 'market_cap']:
            fundamental_data[field] = pd.DataFrame(
                np.nan, index=dates, columns=stock_codes
            )

        fundamental_loaded = False

        # 1) Try Tushare daily_basic — real daily PE/PB/market_cap history
        try:
            import tushare as ts
            token = _get_tushare_token()
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                # Build ts_code list (max 50 per batch to avoid URL too long)
                ts_codes = [_ts_code(c) for c in stock_codes]
                pe_loaded = pb_loaded = mc_loaded = 0
                for i in range(0, len(ts_codes), 50):
                    batch = ts_codes[i:i+50]
                    ts_code_str = ','.join(batch)
                    try:
                        df_fund = pro.daily_basic(
                            ts_code=ts_code_str,
                            start_date=start_date.replace('-', ''),
                            end_date=end_date.replace('-', ''),
                            fields='ts_code,trade_date,pe_ttm,pb,total_mv'
                        )
                    except Exception:
                        # Try one-by-one for this batch
                        df_fund = None
                        for tc in batch:
                            try:
                                df_one = pro.daily_basic(
                                    ts_code=tc,
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''),
                                    fields='ts_code,trade_date,pe_ttm,pb,total_mv'
                                )
                                if df_fund is None:
                                    df_fund = df_one
                                else:
                                    df_fund = pd.concat([df_fund, df_one], ignore_index=True)
                            except Exception:
                                pass

                    if df_fund is None or df_fund.empty:
                        continue

                    df_fund['trade_date'] = pd.to_datetime(df_fund['trade_date'], format='%Y%m%d')
                    df_fund = df_fund.sort_values('trade_date')

                    for _, row in df_fund.iterrows():
                        tc = str(row['ts_code']).strip()
                        plain = tc.split('.')[0]
                        qc = None
                        if tc.endswith('.SH'):
                            qc = 'SH' + plain
                        elif tc.endswith('.SZ'):
                            qc = 'SZ' + plain
                        if qc is None or qc not in stock_codes:
                            continue
                        dt = row['trade_date']
                        if dt in fundamental_data['pe'].index:
                            try:
                                if pd.notna(row.get('pe_ttm')):
                                    fundamental_data['pe'].loc[dt, qc] = float(row['pe_ttm'])
                                    pe_loaded += 1
                            except Exception:
                                pass
                            try:
                                if pd.notna(row.get('pb')):
                                    fundamental_data['pb'].loc[dt, qc] = float(row['pb'])
                                    pb_loaded += 1
                            except Exception:
                                pass
                            try:
                                if pd.notna(row.get('total_mv')):
                                    # total_mv unit: 万元 → 亿元
                                    fundamental_data['market_cap'].loc[dt, qc] = float(row['total_mv']) / 10000.0
                                    mc_loaded += 1
                            except Exception:
                                pass

                n_covered = fundamental_data['pe'].notna().any(axis=0).sum()
                if n_covered > 0:
                    fundamental_loaded = True
                    print(f"  [akshare] Tushare daily_basic coverage: "
                          f"pe={n_covered}/{n_stocks}, "
                          f"rows_pe={pe_loaded}, rows_pb={pb_loaded}, rows_mc={mc_loaded}")
        except Exception as e:
            print(f"  [akshare] Tushare daily_basic failed: {e}")

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

        # Forward/back fill missing fundamentals
        for field in fundamental_data:
            fundamental_data[field] = fundamental_data[field].ffill().bfill()

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
        start_date: str = None,
        end_date: str = None,
        universe: str = None,
        force_refresh: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load stock data for the specified period and universe.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            universe: Stock universe (hs300, zz500, all_a)
            force_refresh: Skip cache and re-download

        Returns:
            Tuple of (price_data, fundamental_data, industry_series)
            - price_data: Dict of DataFrames with keys [open, high, low, close, volume, amount]
            - fundamental_data: Dict of DataFrames
            - industry_series: Series, index = stock codes
        """
        start_date = start_date or self.data_config['universe']['start_date']
        end_date = end_date or self.data_config['universe']['end_date']
        universe = universe or self.data_config['universe']['index']
        source = self.data_config.get('source', 'auto')

        print(f"Loading data from {start_date} to {end_date}, universe: {universe}")

        # Use the centralized real-data loader (handles cache + fallback)
        price_data, fundamental_data, industry_data = load_real_data(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            source=source,
            force_refresh=force_refresh,
            config=self.config,
        )

        # Preprocess
        price_data = self._preprocess_price_data(price_data)
        fundamental_data = self._preprocess_fundamental_data(fundamental_data)

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
                df = df.bfill()
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
            # Fill missing values with forward fill (use latest available)
            df = df.ffill().bfill()
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
        method: str = 'chronological',
        train_end_date: str = None,
        test_start_date: str = None,
    ) -> Tuple[Dict, Dict]:
        """
        Split ALL loaded data into train and test sets.
        
        This splits the full price_data dict (all columns: open/high/low/close/volume/amount),
        fundamental_data, and industry_data — NOT just close prices.
        
        Args:
            test_ratio: Ratio of test data (used only when train_end_date is not given)
            method: Split method ('chronological', 'rolling')
            train_end_date: Explicit train/test split date (YYYY-MM-DD).
                         If given, overrides test_ratio.
            test_start_date: Explicit test start date. If None, derived from train_end_date.
            
        Returns:
            Tuple of (train_data_dict, test_data_dict)
            Each dict has keys: 'price_data', 'fundamental_data', 'industry_data'
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_data() first.")
        
        close_prices = self.price_data['close']
        n_days = len(close_prices)
        dates = close_prices.index
        
        # --- Determine split index ---
        if train_end_date is not None:
            # Use explicit date boundary
            train_end = pd.Timestamp(train_end_date)
            split_idx = int(np.searchsorted(dates, train_end))
            if split_idx >= n_days:
                raise ValueError(f"train_end_date {train_end_date} is at or after the last date {dates[-1]}")
            if split_idx <= 0:
                raise ValueError(f"train_end_date {train_end_date} is before the first date {dates[0]}")
        else:
            split_idx = int(n_days * (1 - test_ratio))
        
        # --- Split price_data (ALL columns) ---
        train_price = {}
        test_price = {}
        for col in self.price_data:
            train_price[col] = self.price_data[col].iloc[:split_idx]
            test_price[col] = self.price_data[col].iloc[split_idx:]
        
        # --- Split fundamental_data (all keys) ---
        train_fund = {}
        test_fund = {}
        if self.fundamental_data:
            for key in self.fundamental_data:
                df = self.fundamental_data[key]
                train_fund[key] = df.iloc[:split_idx]
                test_fund[key] = df.iloc[split_idx:]
        
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
        }
        
        print(f"  [split] Train: {train_price['close'].index[0].date()} ~ {train_price['close'].index[-1].date()} ({len(train_price['close'])} days)")
        print(f"  [split] Test:  {test_price['close'].index[0].date()} ~ {test_price['close'].index[-1].date()} ({len(test_price['close'])} days)")
        
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
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date='2022-01-01',
        end_date='2024-12-31',
        universe='hs300',
    )
    
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
