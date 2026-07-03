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

When LLM is available (configured in config.yaml llm.generator section):
  Stage 1: LLM generates market hypotheses (mirrors AlphaAgentHypothesisGen)
  Stage 2: LLM converts hypotheses to factor expressions (mirrors AlphaAgentHypothesis2FactorExpression)
When LLM is not available, falls back to random formula generation.

Backtest uses the unified BacktestEngine from backtest/engine.py to ensure
consistent evaluation across all baselines.

Usage:
    python baselines/run_alphaagent.py
    python baselines/run_alphaagent.py --output-dir experiments/alphaagent_test
    python baselines/run_alphaagent.py --use-llm --n-formulas 30
"""

import sys
import os
import json
import argparse
import logging
import importlib.util
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
# Direct loading of AlphaAgent's function_lib (bypasses __init__.py chain)
# ═══════════════════════════════════════════════════════════════════════

def _load_function_lib():
    """
    Load AlphaAgent's function_lib.py directly via importlib.

    This bypasses the package's __init__.py which triggers a chain of imports
    requiring pydantic_settings and other AlphaAgent framework dependencies.
    function_lib.py itself only needs numpy, pandas, and joblib.
    """
    flib_path = PROJECT_ROOT / "baselines" / "AlphaAgent" / "alphaagent" / "components" / "coder" / "factor_coder" / "function_lib.py"
    spec = importlib.util.spec_from_file_location("alphaagent_function_lib", str(flib_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════
# Data bridge: generate HDF5 files from main DataLoader
# ═══════════════════════════════════════════════════════════════════════

def convert_to_multindex(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Convert main DataLoader price_data dict to MultiIndex (datetime, instrument) Series.

    The main DataLoader returns {field: DataFrame(date x stock)} dictionaries.
    AlphaAgent's function library expects MultiIndex Series with index names
    ('datetime', 'instrument').
    """
    result = {}
    for field, df in price_data.items():
        if df is None or df.empty:
            continue
        # Stack: (date, stock) -> MultiIndex
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
        print(f"  Saved: {all_path}  ({combined.shape[0]} rows x {combined.shape[1]} cols)")

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

    print(f"  Saved: {debug_path}  ({debug_data.shape[0]} rows x {debug_data.shape[1]} cols)")

    return all_path, debug_path


# ═══════════════════════════════════════════════════════════════════════
# Simulated factor generation (fallback when LLM is not available)
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
# LLM-based factor generation (AlphaAgent core pipeline)
# ═══════════════════════════════════════════════════════════════════════

# Full function library description -- from AlphaAgent's prompts_alphaagent.yaml
FUNCTION_LIB_DESCRIPTION = """Only the following operations are allowed in expressions:
### Cross-sectional Functions
- RANK(A): Ranking of each element in the cross-sectional dimension of A.
- ZSCORE(A): Z-score of each element in the cross-sectional dimension of A.
- MEAN(A): Mean value of each element in the cross-sectional dimension of A.
- STD(A): Standard deviation in the cross-sectional dimension of A.
- SKEW(A): Skewness in the cross-sectional dimension of A.
- KURT(A): Kurtosis in the cross-sectional dimension of A.
- MAX(A): Maximum value in the cross-sectional dimension of A.
- MIN(A): Minimum value in the cross-sectional dimension of A.
- MEDIAN(A): Median value in the cross-sectional dimension of A

### Time-Series Functions
- DELTA(A, n): Change in value of A over n periods.
- DELAY(A, n): Value of A delayed by n periods.
- TS_MEAN(A, n): Mean value of sequence A over the past n days.
- TS_SUM(A, n): Sum of sequence A over the past n days.
- TS_RANK(A, n): Time-series rank of the last value of A in the past n days.
- TS_ZSCORE(A, n): Z-score for each sequence in A over the past n days.
- TS_MEDIAN(A, n): Median value of sequence A over the past n days.
- TS_PCTCHANGE(A, p): Percentage change in the value of sequence A over p periods.
- TS_MIN(A, n): Minimum value of A in the past n days.
- TS_MAX(A, n): Maximum value of A in the past n days.
- TS_ARGMAX(A, n): The index (relative to the current time) of the maximum value of A over the past n days.
- TS_ARGMIN(A, n): The index (relative to the current time) of the minimum value of A over the past n days.
- TS_QUANTILE(A, p, q): Rolling quantile of sequence A over the past p periods, where q is the quantile value between 0 and 1.
- TS_STD(A, n): Standard deviation of sequence A over the past n days.
- TS_VAR(A, p): Rolling variance of sequence A over the past p periods.
- TS_CORR(A, B, n): Correlation coefficient between sequences A and B over the past n days.
- TS_COVARIANCE(A, B, n): Covariance between sequences A and B over the past n days.
- TS_MAD(A, n): Rolling Median Absolute Deviation of sequence A over the past n days.
- HIGHDAY(A, n): Number of days since the highest value of A in the past n days.
- LOWDAY(A, n): Number of days since the lowest value of A in the past n days.
- SUMAC(A, n): Cumulative sum of A over the past n days.

### Moving Averages and Smoothing Functions
- SMA(A, n, m): Simple moving average of A over n periods with modifier m.
- WMA(A, n): Weighted moving average of A over n periods.
- EMA(A, n): Exponential moving average of A over n periods.
- DECAYLINEAR(A, d): Linearly weighted moving average of A over d periods.

### Mathematical Operations
- PROD(A, n): Product of values in A over the past n days. Use * for general multiplication.
- LOG(A): Natural logarithm of each element in A.
- SQRT(A): Square root of each element in A.
- POW(A, n): Raise each element in A to the power of n.
- SIGN(A): Sign of each element in A, one of 1, 0, or -1.
- EXP(A): Exponential of each element in A.
- ABS(A): Absolute value of A.
- MAX(A, B): Maximum value between A and B.
- MIN(A, B): Minimum value between A and B.
- INV(A): Reciprocal (1/x) of each element in sequence A.
- FLOOR(A): Floor of each element in sequence A.

### Conditional and Logical Functions
- COUNT(C, n): Count of samples satisfying condition C in the past n periods.
- SUMIF(A, n, C): Sum of A over the past n periods if condition C is met.
- FILTER(A, C): Filtering multi-column sequence A based on condition C.
- (C1)&&(C2): Logical "and".
- (C1)||(C2): Logical "or".
- (C1)?(A):(B): If condition C1 holds, then A, otherwise B.

### Regression and Residual Functions
- SEQUENCE(n): A single-column sequence of length n, ranging from 1 to n.
- REGBETA(A, B, n): Regression coefficient of A on B using the past n samples.
- REGRESI(A, B, n): Residual of regression of A on B using the past n samples.

### Technical Indicators
- RSI(A, n): Relative Strength Index of sequence A over n periods.
- MACD(A, short_window, long_window): Moving Average Convergence Divergence.
- BB_MIDDLE(A, n): Middle Bollinger Band.
- BB_UPPER(A, n): Upper Bollinger Band.
- BB_LOWER(A, n): Lower Bollinger Band.

Note:
- Only the variables provided in data (e.g., $open), arithmetic operators (+, -, *, /), logical operators (&&, ||), and the operations above are allowed.
- Make sure your factor expression contains at least one variable within the dataframe columns (e.g. $open).
- Pay attention to the distinction between TS prefix (e.g., TS_STD()) and those without (e.g., STD())."""

# Market hypothesis directions for LLM inspiration
HYPOTHESIS_DIRECTIONS = [
    "momentum and mean reversion effects in stock prices",
    "volume-price divergence patterns and their predictive power",
    "volatility clustering and risk-adjusted momentum",
    "intraday price range patterns (high-low spread) as volatility signals",
    "liquidity shocks and their impact on short-term returns",
    "cross-sectional relative strength and weakness",
    "time-series trend acceleration and deceleration",
    "overnight vs intraday return decomposition",
    "trading volume concentration and price persistence",
    "price acceleration patterns and reversal tendencies",
]


def _read_llm_config(config_path: str) -> Tuple[str, str, str]:
    """
    Read LLM configuration from config.yaml.

    Returns:
        (api_key, base_url, model) -- falls back to env vars if config missing.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        llm_cfg = config.get('llm', {}).get('generator', {})
        api_key = llm_cfg.get('api_key') or os.environ.get('OPENAI_API_KEY', '')
        base_url = llm_cfg.get('base_url') or os.environ.get('OPENAI_BASE_URL', '')
        model = llm_cfg.get('model', 'gpt-4o')
        return api_key, base_url, model
    except Exception:
        return '', '', 'gpt-4o'


def _llm_generate_hypothesis(
    api_key: str,
    base_url: str,
    model: str,
    direction: str,
    round_idx: int = 0,
    prev_hypotheses: List[str] = None,
) -> Optional[str]:
    """
    Stage 1: Use LLM to generate a market hypothesis for factor mining.

    This mirrors AlphaAgent's AlphaAgentHypothesisGen.gen() -- the LLM proposes
    a testable financial hypothesis that guides factor expression construction.

    Args:
        direction: Market direction theme to inspire the hypothesis
        round_idx: Current round (0 = first round)
        prev_hypotheses: Previously generated hypotheses for context

    Returns:
        Hypothesis text string, or None if LLM call fails.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        system_prompt = """You are a quantitative finance expert generating hypotheses for alpha factor mining.
Your task is to propose a clear, actionable, and testable market hypothesis that can be translated into quantitative factor expressions.

The hypothesis should:
1. Be grounded in financial theory or observed market patterns
2. Suggest a clear path for factor construction using price/volume data
3. Be specific enough to guide mathematical expression design
4. Focus on relationships between price, volume, and returns

Respond with ONLY the hypothesis text (2-4 sentences). No JSON, no formatting."""

        user_parts = [f"Market direction theme: {direction}"]
        if prev_hypotheses:
            user_parts.append("\nPreviously explored hypotheses (avoid repeating):\n" +
                             "\n".join(f"  {i+1}. {h}" for i, h in enumerate(prev_hypotheses[-5:])))
            user_parts.append("\nGenerate a NEW hypothesis that explores a different angle.")

        user_prompt = "\n".join(user_parts)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        hypothesis = response.choices[0].message.content.strip()
        return hypothesis

    except Exception as e:
        logger.warning(f"  LLM hypothesis generation failed: {e}")
        return None


def _llm_generate_factors(
    api_key: str,
    base_url: str,
    model: str,
    hypothesis: str,
    prev_factors: List[Tuple[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    Stage 2: Use LLM to convert a hypothesis into 2-3 factor expressions.

    This mirrors AlphaAgent's AlphaAgentHypothesis2FactorExpression.convert() --
    the LLM generates JSON with factor name, description, and expression using
    the function library.

    Args:
        hypothesis: Market hypothesis from Stage 1
        prev_factors: Previously generated factors (to avoid duplication)

    Returns:
        List of (factor_name, expression_string) tuples.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        system_prompt = f"""You are a quantitative researcher constructing alpha factor expressions.

The user will provide a market hypothesis. Your task is to generate 2-3 factor expressions that capture the hypothesis.

When constructing factor expressions, you are restricted to utilizing only the following daily-level variables:
- $open: open price of the stock on that day.
- $close: close price of the stock on that day.
- $high: high price of the stock on that day.
- $low: low price of the stock on that day.
- $volume: volume of the stock on that day.
- $return: daily return of the stock on that day.

{FUNCTION_LIB_DESCRIPTION}

Key considerations:
- Avoid using raw prices directly due to scale differences; use relative changes or standardized data
- Add small constants (e.g., 1e-8) to denominators to prevent division by zero
- Apply RANK() or ZSCORE() for cross-sectional comparability
- Choose suitable window sizes (5, 10, 20, 30, 60 days) for moving averages

The output should follow JSON format without other content. The schema is:
{{
    "factor_name_1": {{
        "description": "description of factor 1",
        "expression": "expression using $close, $open, etc. and functions like RANK(), TS_MEAN()"
    }},
    "factor_name_2": {{
        "description": "description of factor 2",
        "expression": "expression"
    }}
}}

Example:
{{
    "Normalized_Intraday_Range": {{
        "description": "Candlestick body normalized by volatility",
        "expression": "ABS($close - $open) / (TS_STD($close, 10) + 1e-8)"
    }},
    "Volume_Price_Correlation": {{
        "description": "Correlation between price range and volume",
        "expression": "TS_CORR($high - $low, $volume, 20)"
    }}
}}

Strictly adhere to the syntax. Do NOT use undeclared variables or functions."""

        user_parts = [f"Target hypothesis:\n{hypothesis}"]
        if prev_factors:
            user_parts.append("\nPreviously generated factors (avoid similar expressions):")
            for name, expr in prev_factors[-10:]:
                user_parts.append(f"  - {name}: {expr}")

        user_prompt = "\n".join(user_parts)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.5,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        result = json.loads(raw)
        factors = []
        for name, info in result.items():
            expr = info.get('expression', '').strip()
            if expr:
                # Sanitize factor name
                safe_name = name.replace(' ', '_').replace('-', '_')
                factors.append((safe_name, expr))

        return factors

    except Exception as e:
        logger.warning(f"  LLM factor generation failed: {e}")
        return []


def generate_llm_factors(
    n_formulas: int = 50,
    config_path: str = "config/config.yaml",
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Generate factor formulas using LLM (AlphaAgent's core pipeline).

    Implements a simplified version of AlphaAgent's two-stage loop:
    Stage 1: LLM generates market hypotheses
    Stage 2: LLM converts hypotheses to factor expressions (2-3 per call)

    Falls back to random generation if LLM is unavailable.

    Args:
        n_formulas: Target number of factor formulas
        config_path: Path to config.yaml for LLM settings
        seed: Random seed (used for direction selection and fallback)

    Returns:
        (formulas, used_llm) -- list of (name, expression) tuples and whether LLM was used.
    """
    api_key, base_url, model = _read_llm_config(config_path)

    if not api_key:
        print("  [WARN] No LLM API key found, falling back to random generation")
        return generate_simulated_formulas(n_formulas=n_formulas, seed=seed), False

    print(f"  LLM backend: model={model}, base_url={base_url[:40]}...")

    rng = np.random.default_rng(seed)
    formulas = []
    prev_hypotheses = []

    # Each LLM call generates 2-3 factors. Calculate rounds needed.
    n_rounds = max(1, (n_formulas + 2) // 3)

    for round_idx in range(n_rounds):
        if len(formulas) >= n_formulas:
            break

        # Stage 1: Generate hypothesis
        direction = str(rng.choice(HYPOTHESIS_DIRECTIONS))
        print(f"  [Round {round_idx+1}/{n_rounds}] Hypothesis: {direction}...")

        hypothesis = _llm_generate_hypothesis(
            api_key, base_url, model, direction,
            round_idx=round_idx, prev_hypotheses=prev_hypotheses,
        )

        if hypothesis is None:
            print(f"    Hypothesis generation failed, using fallback direction")
            hypothesis = f"Factor based on {direction}"

        prev_hypotheses.append(hypothesis)
        print(f"    Hypothesis: {hypothesis[:100]}...")

        # Stage 2: Generate factor expressions from hypothesis
        new_factors = _llm_generate_factors(
            api_key, base_url, model, hypothesis,
            prev_factors=formulas,
        )

        if not new_factors:
            print(f"    Factor generation failed, skipping this round")
            continue

        for name, expr in new_factors:
            # Avoid duplicate names
            base_name = name
            suffix = 1
            while any(f[0] == name for f in formulas):
                suffix += 1
                name = f"{base_name}_{suffix}"

            formulas.append((name, expr))
            print(f"    -> {name}: {expr[:80]}")

        print(f"    Total: {len(formulas)}/{n_formulas}")

    if len(formulas) < n_formulas:
        # Supplement with random formulas if LLM didn't generate enough
        remaining = n_formulas - len(formulas)
        print(f"  Supplementing with {remaining} random formulas...")
        random_formulas = generate_simulated_formulas(n_formulas=remaining, seed=seed + 1)
        formulas.extend(random_formulas)

    return formulas[:n_formulas], True


# ═══════════════════════════════════════════════════════════════════════
# Factor computation using AlphaAgent's function library
# ═══════════════════════════════════════════════════════════════════════

def compute_factor_values(
    formulas: List[Tuple[str, str]],
    price_midx: Dict[str, pd.Series],
    return_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute factor values from formulas using AlphaAgent's function library.

    Uses eval() with a restricted namespace containing all function_lib
    functions and price data columns. Supports arithmetic operators (+, -, *, /)
    natively, since pandas Series/DataFrame support them.

    Args:
        formulas: List of (name, expression) tuples
        price_midx: Price data as MultiIndex Series dict
        return_series: Daily returns (for $return in factor expressions)

    Returns:
        DataFrame with datetime x instrument index, one column per factor
    """
    # Import AlphaAgent's function library (bypasses __init__.py chain)
    flib = _load_function_lib()

    # Build a combined DataFrame from price data
    price_df = pd.DataFrame(index=price_midx.get('close', list(price_midx.values())[0]).index)
    field_map = {
        'open': '$open', 'close': '$close', 'high': '$high',
        'low': '$low', 'volume': '$volume',
    }
    for src, target in field_map.items():
        if src in price_midx:
            price_df[target] = price_midx[src]

    # Add $return column if available (LLM expressions may use it)
    if return_series is not None:
        price_df['$return'] = return_series.reindex(price_df.index)

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
    Evaluate a single AlphaAgent-style factor expression.

    Uses eval() with a restricted namespace containing all function_lib
    functions and price data columns. Supports arithmetic operators (+, -, *, /)
    natively, since pandas Series/DataFrame support them.

    This handles expressions like:
      ABS($close - $open) / (TS_STD($close, 10) + 1e-8)
      RANK(DELTA($close, 5) / $close)
      TS_CORR($high - $low, $volume, 20)
    """
    try:
        # Step 1: Remove $ from variable names (not valid Python identifiers)
        expr = formula.replace('$', '')

        # Step 2: Build namespace with all function_lib functions
        namespace = {}
        for name in dir(flib):
            obj = getattr(flib, name)
            if callable(obj) and not name.startswith('_'):
                namespace[name] = obj

        # Step 3: Add price data columns (without $ prefix) as variables
        for col in price_df.columns:
            var_name = col.replace('$', '')
            namespace[var_name] = price_df[col]

        # Step 4: Add numeric/math utilities
        namespace['np'] = np
        namespace['pd'] = pd

        # Step 5: Restrict builtins for safety
        namespace['__builtins__'] = {}

        # Step 6: Evaluate expression
        result = eval(expr, namespace)

        # Step 7: Ensure result is a Series with correct index
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        elif isinstance(result, np.ndarray):
            result = pd.Series(result.flatten(), index=price_df.index)

        if isinstance(result, pd.Series) and not result.index.equals(price_df.index):
            result = result.reindex(price_df.index)

        return result

    except Exception as e:
        logger.debug(f"  Error evaluating '{formula}': {e}")
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
        factor_df: DataFrame with factor values (datetime x instrument)
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
        # MultiIndex DataFrame -> unstack to wide
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
    use_llm: bool = True,
) -> Dict:
    """
    Run AlphaAgent baseline using the main project's DataLoader.

    When use_llm=True (default), uses LLM to generate factor formulas via:
      Stage 1: LLM generates market hypotheses
      Stage 2: LLM converts hypotheses to factor expressions
    When LLM is unavailable or use_llm=False, falls back to random generation.

    Uses the unified BacktestEngine from backtest/engine.py for consistent
    performance evaluation across all baselines.

    Args:
        config_path: Path to main project config YAML
        output_dir: Directory to save results
        n_formulas: Number of factor formulas to generate
        seed: Random seed for reproducibility
        start_date: Override data start date
        end_date: Override data end date
        train_end_date: Override train end date
        test_start_date: Override test start date
        use_llm: If True, use LLM to generate factors (default True).
                 Falls back to random if LLM is unavailable.

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
    print(f"  Loaded: {n_dates} dates x {n_stocks} stocks")
    print(f"  Train: <= {train_end_date}  |  Test: >= {test_start_date}")

    # ── 2. Save HDF5 data for AlphaAgent compatibility ─────────────
    print(f"\n[2/6] Generating HDF5 data files...")
    data_dir = os.path.join(output_dir, "data")
    save_data_as_hdf5(price_midx, return_series, data_dir)

    # ── 3. Generate factor formulas ─────────────────────────────────
    if use_llm:
        print(f"\n[3/6] Generating {n_formulas} factor formulas via LLM...")
        formulas, llm_used = generate_llm_factors(
            n_formulas=n_formulas,
            config_path=config_path,
            seed=seed,
        )
    else:
        print(f"\n[3/6] Generating {n_formulas} random factor formulas...")
        formulas = generate_simulated_formulas(n_formulas=n_formulas, seed=seed)
        llm_used = False

    mode_label = "LLM-generated" if llm_used else "random (fallback)"
    print(f"  Factor generation mode: {mode_label}")
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
    factor_df = compute_factor_values(formulas, price_midx, return_series)
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
        print("  WARNING: No valid portfolios generated, using zero metrics")
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
    print(f"  AlphaAgent Baseline Results ({'LLM' if llm_used else 'Random'})")
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
        'used_llm': llm_used,
        'llm_model': _read_llm_config(config_path)[2] if llm_used else None,
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
                        help="Number of factor formulas to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start-date", default=None,
                        help="Data start date (default from config)")
    parser.add_argument("--end-date", default=None,
                        help="Data end date (default from config)")
    parser.add_argument("--use-llm", action="store_true", default=True,
                        help="Use LLM to generate factors (default: True)")
    parser.add_argument("--no-llm", action="store_true", default=False,
                        help="Disable LLM, use random factor generation")
    args = parser.parse_args()

    run_alphaagent_baseline(
        config_path=args.config_path,
        output_dir=args.output_dir,
        n_formulas=args.n_formulas,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
        use_llm=not args.no_llm,
    )
