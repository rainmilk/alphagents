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
import re
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
# Factor Originality Regulator (AST-based, from original AlphaAgent)
# ═══════════════════════════════════════════════════════════════════════

# Valid data variables allowed in factor expressions
VALID_DATA_VARS = {'$open', '$close', '$high', '$low', '$volume', '$return'}

# Cache for valid function names from function_lib
_valid_functions_cache = None


def _get_valid_functions() -> set:
    """Get the set of valid function names from AlphaAgent's function library."""
    global _valid_functions_cache
    if _valid_functions_cache is None:
        flib = _load_function_lib()
        _valid_functions_cache = {
            name for name in dir(flib)
            if callable(getattr(flib, name)) and not name.startswith('_')
        }
    return _valid_functions_cache


def _load_factor_ast():
    """
    Load AlphaAgent's factor_ast.py via importlib.

    This module provides pyparsing-based AST parsing, maximum common subtree
    matching with commutative operator handling, and originality metrics
    (count_free_args, count_unique_vars, count_all_nodes).

    Falls back to None if pyparsing or the module is unavailable.
    """
    try:
        ast_path = (PROJECT_ROOT / "baselines" / "AlphaAgent" / "alphaagent" /
                    "components" / "coder" / "factor_coder" / "factor_ast.py")
        spec = importlib.util.spec_from_file_location("alphaagent_factor_ast", str(ast_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning(f"  Failed to load factor_ast module: {e}")
        logger.warning(f"  AST originality constraint will be disabled")
        return None


class FactorRegulator:
    """
    AST-based factor originality regulator, mirroring AlphaAgent's FactorRegulator.

    Three-stage validation pipeline (matching original AlphaAgent):
    1. is_parsable()        — pyparsing syntax check (catches malformed expressions)
    2. validate_semantics() — function/variable name check (catches unknown functions/vars)
    3. is_expression_acceptable() — originality check with 3 criteria:
       - cond1: max common subtree with existing factors <= 8 nodes
       - cond2: numeric constants / total nodes < 50%
       - cond3: unique variables / total nodes < 50%

    The factor zoo starts empty and accumulates accepted factors, so later
    factors in the same run are checked against earlier ones for structural
    duplication. Commutative operators (+, *) are handled: A+B matches B+A.
    """

    def __init__(self, duplication_threshold: int = 8):
        self.ast_mod = _load_factor_ast()
        self.duplication_threshold = duplication_threshold
        self.factor_zoo = []  # List of expression strings (for AST comparison)

    def is_available(self) -> bool:
        """Check if the AST module loaded successfully."""
        return self.ast_mod is not None

    def is_parsable(self, expr: str) -> bool:
        """
        Stage 1: Check if expression can be parsed by pyparsing.

        This catches syntax errors like mismatched parentheses, invalid tokens,
        or malformed function calls — before sending to eval().
        """
        if not self.is_available():
            return True  # Skip if AST module unavailable
        try:
            self.ast_mod.parse_expression(expr)
            return True
        except Exception:
            return False

    def validate_semantics(self, expr: str) -> Tuple[bool, str]:
        """
        Stage 2: Check that all function names and variable names are valid.

        Walks the AST and verifies:
        - All VarNode names starting with '$' are in VALID_DATA_VARS
        - All FunctionNode names exist in AlphaAgent's function library

        Returns (ok, error_message).
        """
        if not self.is_available():
            return True, ""
        try:
            tree = self.ast_mod.parse_expression(expr)
            errors = []
            self._check_node_semantics(tree, errors)
            if errors:
                return False, "; ".join(errors[:3])  # Limit error messages
            return True, ""
        except Exception as e:
            return False, f"parse error: {e}"

    def _check_node_semantics(self, node, errors: list):
        """Recursively check AST nodes for invalid function/variable names."""
        ast_mod = self.ast_mod

        if isinstance(node, ast_mod.VarNode):
            # VarNode.name is always a string (from pyparsing Combine)
            var_name = node.name if isinstance(node.name, str) else str(node.name)
            if var_name.startswith('$') and var_name not in VALID_DATA_VARS:
                errors.append(f"unknown variable '{var_name}'")

        elif isinstance(node, ast_mod.FunctionNode):
            # FunctionNode.name is stored as a VarNode (from pyparsing grammar:
            # function_call = var + "(" + ...), so we extract the string
            if isinstance(node.name, ast_mod.VarNode):
                func_name = node.name.name if isinstance(node.name.name, str) else str(node.name.name)
            else:
                func_name = str(node.name)
            valid_funcs = _get_valid_functions()
            if func_name not in valid_funcs:
                errors.append(f"unknown function '{func_name}()'")
            for arg in node.args:
                self._check_node_semantics(arg, errors)

        elif isinstance(node, ast_mod.BinaryOpNode):
            self._check_node_semantics(node.left, errors)
            self._check_node_semantics(node.right, errors)

        elif isinstance(node, ast_mod.ConditionalNode):
            self._check_node_semantics(node.condition, errors)
            self._check_node_semantics(node.true_expr, errors)
            self._check_node_semantics(node.false_expr, errors)

    def evaluate(self, expr: str) -> dict:
        """
        Compute originality metrics for an expression.

        Returns dict with:
        - duplicated_subtree_size: size of largest common subtree with factor zoo
        - duplicated_subtree: the matched subtree (for feedback)
        - matched_alpha: the zoo expression it duplicated
        - num_free_args: count of numeric constants
        - num_unique_vars: count of unique $-prefixed variables
        - num_all_nodes: total AST node count
        """
        if not self.is_available():
            return {
                'duplicated_subtree_size': 0,
                'duplicated_subtree': None,
                'matched_alpha': None,
                'num_free_args': 0,
                'num_unique_vars': 0,
                'num_all_nodes': 1,
            }

        # Compare against factor zoo
        max_size = 0
        matched_subtree = None
        matched_alpha = None

        for zoo_expr in self.factor_zoo:
            try:
                match = self.ast_mod.compare_expressions(expr, zoo_expr)
                if match is not None and match.size > max_size:
                    max_size = match.size
                    matched_subtree = match.root1
                    matched_alpha = zoo_expr
            except Exception:
                pass  # Skip comparison errors

        num_free_args = self.ast_mod.count_free_args(expr)
        num_unique_vars = self.ast_mod.count_unique_vars(expr)
        num_all_nodes = self.ast_mod.count_all_nodes(expr)

        return {
            'duplicated_subtree_size': max_size,
            'duplicated_subtree': matched_subtree,
            'matched_alpha': matched_alpha,
            'num_free_args': num_free_args,
            'num_unique_vars': num_unique_vars,
            'num_all_nodes': num_all_nodes,
        }

    def is_expression_acceptable(self, eval_dict: dict) -> bool:
        """
        Stage 3: Check three originality acceptance criteria.

        cond1: duplicated_subtree_size <= duplication_threshold (default 8)
            Prevents factors that are structurally near-identical to existing ones.

        cond2: -ln(1 - free_args_ratio) < 0.693
            Equivalent to: num_free_args / num_all_nodes < 0.5
            Prevents factors that are mostly numeric constants (too simple).

        cond3: -ln(1 - unique_vars_ratio) < 0.693
            Equivalent to: num_unique_vars / num_all_nodes < 0.5
            Prevents factors that are just variable stacking without operations.
        """
        num_all_nodes = eval_dict['num_all_nodes']
        if num_all_nodes == 0:
            return False

        # cond1: structural originality
        cond1 = eval_dict['duplicated_subtree_size'] <= self.duplication_threshold

        # cond2: numeric constant ratio < 50%
        free_args_ratio = float(eval_dict['num_free_args']) / float(num_all_nodes)
        if free_args_ratio >= 1.0:
            return False
        cond2 = -np.log(1.0 - free_args_ratio) < 0.693

        # cond3: variable ratio < 50%
        unique_vars_ratio = float(eval_dict['num_unique_vars']) / float(num_all_nodes)
        if unique_vars_ratio >= 1.0:
            return False
        cond3 = -np.log(1.0 - unique_vars_ratio) < 0.693

        return cond1 and cond2 and cond3

    def add_factor(self, name: str, expr: str):
        """Add an accepted factor to the zoo for future originality checks."""
        self.factor_zoo.append(expr)

    def validate_factor(self, name: str, expr: str) -> Tuple[bool, str, Optional[dict]]:
        """
        Full validation pipeline: parsable → semantics → originality.

        Returns (accepted, reason, eval_dict).
        - accepted=True: factor passed all checks
        - accepted=False: factor rejected, reason explains why
        """
        # Stage 1: Syntax
        if not self.is_parsable(expr):
            return False, "unparseable expression", None

        # Stage 2: Semantics
        ok, err = self.validate_semantics(expr)
        if not ok:
            return False, f"semantic error: {err}", None

        # Stage 3: Originality
        eval_dict = self.evaluate(expr)
        if not self.is_expression_acceptable(eval_dict):
            dup_size = eval_dict['duplicated_subtree_size']
            if dup_size > self.duplication_threshold:
                matched = eval_dict.get('matched_alpha', '')
                return False, (
                    f"structural duplication (subtree_size={dup_size} > {self.duplication_threshold})"
                    + (f", similar to: {matched[:60]}..." if matched else "")
                ), eval_dict

            num_all = eval_dict['num_all_nodes']
            free_ratio = eval_dict['num_free_args'] / max(num_all, 1)
            var_ratio = eval_dict['num_unique_vars'] / max(num_all, 1)
            if free_ratio >= 0.5:
                return False, f"too many constants (ratio={free_ratio:.2f})", eval_dict
            if var_ratio >= 0.5:
                return False, f"too few operations (var_ratio={var_ratio:.2f})", eval_dict

            return False, "originality check failed", eval_dict

        return True, "accepted", eval_dict


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
    """Compute daily returns from close prices.

    Used as $return in factor expressions (LLM-generated formulas may
    reference $return). This is NOT the same as forward returns used
    for IC computation — see compute_forward_returns().
    """
    close = price_midx.get('close')
    if close is None:
        raise ValueError("Missing 'close' field in price data")
    returns = close.unstack('instrument').pct_change().fillna(0)
    returns = returns.stack()
    returns.index.names = ['datetime', 'instrument']
    return returns


def compute_forward_returns(
    price_midx: Dict[str, pd.Series],
    forward_period: int = 10,
) -> pd.Series:
    """Compute forward N-day returns for IC-based factor ranking.

    For forward_period=N:  return[t] = close[t+N] / close[t] - 1

    This aligns AlphaAgent's IC evaluation with other baselines that use
    forward_period-day forward returns (MCTS-LLM-Alpha, AlphaGrail, XGBoost,
    LSTM, XGBoost-Simple all default to forward_period=10).

    Note: compute_returns() produces 1-day daily returns for $return in
    factor expressions. This function produces N-day forward returns used
    for IC-based factor selection — the two serve different purposes.

    Args:
        price_midx: MultiIndex price data dict (must contain 'close')
        forward_period: Number of trading days to look ahead (default 10,
            matching other baselines for fair comparison)

    Returns:
        pd.Series of forward returns, aligned to current date.
        Last forward_period days will be NaN (no future data available).
    """
    close = price_midx.get('close')
    if close is None:
        raise ValueError("Missing 'close' field in price data")
    future_close = close.groupby(level='instrument').shift(-forward_period)
    forward_ret = future_close / close - 1
    forward_ret.name = 'forward_return'
    return forward_ret


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


def _extract_message_text(message) -> str:
    """Extract usable text from an OpenAI chat completion message.

    Reasoning models (o1, DeepSeek-R1 / deepseek-reasoner, QwQ, ...) frequently
    emit chain-of-thought in a separate ``reasoning_content`` field while leaving
    ``content`` empty or ``None``. If we only read ``content``, callers silently
    receive an empty string — which (before the Stage-1 guard) poisoned Stage 2
    with an empty hypothesis and now still produces a fake fallback hypothesis.

    Strategy: prefer ``content``; if it is empty/None, fall back to
    ``reasoning_content`` (exposed directly on some SDKs, or under
    ``model_extra`` on pydantic-based ones). Returns '' when nothing usable.
    """
    content = getattr(message, "content", None)
    if content:
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning
    # openai>=1.x pydantic models stash unknown fields in model_extra
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        rc = extra.get("reasoning_content")
        if rc:
            return rc
    return ""


def _parse_factors_json(raw: str):
    """Best-effort parse of a factor-spec JSON object from LLM output.

    Handles the common failure modes seen in practice:
      * reasoning-model chain-of-thought wrapped in <think>...</think>
      * markdown code fences (```json ... ```, any casing / language tag)
      * a JSON object embedded in / wrapped by surrounding prose
      * output truncated before the JSON could be completed

    Uses bracket-balanced substring extraction (not a greedy regex) so that
    explanatory prose containing stray braces can't swallow the real payload.

    Returns the dict, or None if no valid JSON object can be recovered.
    """
    if not raw:
        return None
    # 1) drop any <think>...</think> CoT block, then strip code fences
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    fence = re.match(r"```[a-zA-Z]*\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    candidate = fence.group(1).strip() if fence else cleaned
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2) bracket-balanced extraction: try every '{' as a candidate start and
    #    match it to its *corresponding* '}', returning the first span that
    #    parses to a dict. (The old greedy regex grabbed first '{'..last '}',
    #    which broke whenever prose itself contained braces.)
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        matched = -1
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    matched = i
                    break
        if matched != -1:
            cand = cleaned[start:matched + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        start = cleaned.find("{", start + 1)
    return None


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

        call_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        thinking = True
        while True:
            if thinking:
                # Reasoning endpoints may emit CoT in content; disable thinking so
                # the hypothesis is the actual answer, not the chain-of-thought.
                call_kwargs["extra_body"] = {"enable_thinking": False}
            try:
                response = client.chat.completions.create(**call_kwargs)
                break
            except Exception as e:
                if thinking:
                    thinking = False
                    logger.warning(
                        "  Provider rejected extra_body (enable_thinking); "
                        "retrying without thinking control. (%s)",
                        e,
                    )
                    continue
                raise
        content = _extract_message_text(response.choices[0].message)
        hypothesis = content.strip() if content else ""
        if not hypothesis:
            # Empty/whitespace response (e.g. reasoning model left `content`
            # empty and had no usable `reasoning_content`). Treat as failure so
            # the orchestration fallback fires instead of poisoning Stage 2.
            return None
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
    feedback: str = None,
) -> List[Tuple[str, str]]:
    """
    Stage 2: Use LLM to convert a hypothesis into 2-3 factor expressions.

    This mirrors AlphaAgent's AlphaAgentHypothesis2FactorExpression.convert() --
    the LLM generates JSON with factor name, description, and expression using
    the function library.

    Args:
        hypothesis: Market hypothesis from Stage 1
        prev_factors: Previously generated factors (to avoid duplication)
        feedback: Originality rejection feedback (from FactorRegulator) to
            inject into the prompt, guiding the LLM to avoid duplicated
            sub-expressions. None for first attempt.

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

Use relative changes or standardized data (not raw prices), add 1e-8 to denominators, prefer RANK()/ZSCORE(), and window sizes 5/10/20/30/60. Do NOT think step-by-step and do NOT explain — output the JSON object immediately.

You MUST output ONLY a single JSON object and nothing else — no explanations, no reasoning, no markdown, no code fences. Your entire response must be valid JSON parseable as an object. The schema is:
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

Example (one factor, terse):
{{
    "Normalized_Intraday_Range": {{
        "description": "Candlestick body normalized by volatility",
        "expression": "ABS($close - $open) / (TS_STD($close, 10) + 1e-8)"
    }}
}}

Strictly adhere to the syntax. Do NOT use undeclared variables or functions."""

        user_parts = [f"Target hypothesis:\n{hypothesis}"]
        if prev_factors:
            user_parts.append("\nPreviously generated factors (avoid similar expressions):")
            for name, expr in prev_factors[-10:]:
                user_parts.append(f"  - {name}: {expr}")

        # Inject originality feedback if provided (from FactorRegulator rejection)
        if feedback:
            user_parts.append(feedback)

        user_prompt = "\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = None
        response = None
        json_mode = True  # try API-level JSON enforcement; self-heal if unsupported
        thinking = True   # try disabling thinking; self-heal if unsupported
        # Up to 2 attempts: the normal call, then one correction pass if the
        # model emitted non-JSON (verbose prose / truncated before the JSON).
        # Temperature drops on the correction pass to nudge determinism.
        idx = 0
        while idx < 2:
            call_kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=0.1 if idx > 0 else 0.2,
            )
            if json_mode:
                call_kwargs["response_format"] = {"type": "json_object"}
            if thinking:
                # Reasoning endpoints (bailian/deepseek-v4-flash, …) may write
                # CoT into content; disable thinking for clean JSON.
                call_kwargs["extra_body"] = {"enable_thinking": False}
            try:
                response = client.chat.completions.create(**call_kwargs)
            except Exception as e:
                # Some self-hosted / third-party OpenAI-compatible endpoints
                # reject `extra_body` / `response_format`. Self-heal rather than
                # failing the whole mining run.
                if thinking:
                    thinking = False
                    logger.warning(
                        "  Provider rejected extra_body (enable_thinking); retrying "
                        "without thinking control. (%s)",
                        e,
                    )
                    continue
                if json_mode and "response_format" in str(e).lower():
                    json_mode = False
                    logger.warning(
                        "  Provider rejected response_format; retrying without "
                        "JSON mode. (%s)",
                        e,
                    )
                    continue
                raise
            raw = _extract_message_text(response.choices[0].message).strip()
            result = _parse_factors_json(raw)
            if result is not None:
                break
            logger.warning(
                "  LLM factor generation returned non-JSON output (attempt %d); "
                "retrying with strict JSON-only instruction. raw=%r",
                idx + 1, raw,
            )
            messages = messages + [
                {"role": "user", "content":
                 "Your previous reply was not valid JSON. Respond with ONLY the "
                 "JSON object (no explanations, no markdown, no code fences). "
                 "Nothing else."},
            ]
            idx += 1

        if not isinstance(result, dict):
            logger.warning(
                "  LLM factor generation returned non-JSON output; skipping. raw=%r",
                _extract_message_text(response.choices[0].message) if response else "",
            )
            return []

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
    max_retries: int = 3,
) -> Tuple[List[Tuple[str, str]], bool]:
    """
    Generate factor formulas using LLM (AlphaAgent's core pipeline).

    Implements AlphaAgent's two-stage loop with AST originality constraint:
    Stage 1: LLM generates market hypotheses
    Stage 2: LLM converts hypotheses to factor expressions (2-3 per call)

    Each generated factor undergoes three-stage validation:
    1. is_parsable()        — pyparsing syntax check
    2. validate_semantics() — function/variable name check
    3. is_expression_acceptable() — AST originality (3 criteria)

    If a factor fails originality, a feedback prompt is constructed and the
    LLM is retried (up to max_retries times per round), mirroring the original
    AlphaAgent's while-True retry loop in factor_proposal.py.

    Falls back to random generation if LLM is unavailable.

    Args:
        n_formulas: Target number of factor formulas
        config_path: Path to config.yaml for LLM settings
        seed: Random seed (used for direction selection and fallback)
        max_retries: Max LLM retry attempts per round when factors are rejected
            by the originality regulator (default 3)

    Returns:
        (formulas, used_llm) -- list of (name, expression) tuples and whether LLM was used.
    """
    api_key, base_url, model = _read_llm_config(config_path)

    if not api_key:
        print("  [WARN] No LLM API key found, falling back to random generation")
        return generate_simulated_formulas(n_formulas=n_formulas, seed=seed), False

    print(f"  LLM backend: model={model}, base_url={base_url[:40]}...")

    # Initialize AST originality regulator
    regulator = FactorRegulator()
    if regulator.is_available():
        print(f"  AST originality regulator: enabled (threshold={regulator.duplication_threshold})")
    else:
        print(f"  AST originality regulator: disabled (factor_ast module unavailable)")

    rng = np.random.default_rng(seed)
    formulas = []
    prev_hypotheses = []
    n_rejected = 0
    n_retried = 0

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

        if not hypothesis or not hypothesis.strip():
            print(f"    Hypothesis generation failed/empty, using fallback direction")
            hypothesis = f"Factor based on {direction}"

        prev_hypotheses.append(hypothesis)
        print(f"    Hypothesis: {hypothesis[:100]}...")

        # Stage 2: Generate factor expressions from hypothesis (with retry loop)
        feedback = None
        for retry_idx in range(max_retries + 1):  # +1 for initial attempt
            if len(formulas) >= n_formulas:
                break

            new_factors = _llm_generate_factors(
                api_key, base_url, model, hypothesis,
                prev_factors=formulas,
                feedback=feedback,
            )

            if not new_factors:
                print(f"    Factor generation failed, skipping this round")
                break

            # Validate each factor through the regulator
            accepted_factors = []
            rejection_feedbacks = []

            for name, expr in new_factors:
                accepted, reason, eval_dict = regulator.validate_factor(name, expr)

                if accepted:
                    # Avoid duplicate names
                    base_name = name
                    suffix = 1
                    while any(f[0] == name for f in formulas):
                        suffix += 1
                        name = f"{base_name}_{suffix}"

                    formulas.append((name, expr))
                    regulator.add_factor(name, expr)
                    accepted_factors.append((name, expr))
                    print(f"    -> {name}: {expr[:80]}")
                else:
                    n_rejected += 1
                    print(f"    [REJECT] {name}: {reason}")

                    # Build feedback for rejected factors (originality rejections only)
                    if eval_dict is not None:
                        dup_size = eval_dict.get('duplicated_subtree_size', 0)
                        if dup_size > 0:
                            rejection_feedbacks.append(
                                f"- Proposed Expression: {expr}\n"
                                f"  Duplicated Sub-expression Size: {dup_size}\n"
                                f"  Please avoid this pattern and generate a structurally novel factor."
                            )

            if accepted_factors:
                print(f"    Accepted: {len(accepted_factors)}, Total: {len(formulas)}/{n_formulas}")

            # If all factors accepted or no originality rejections, move to next round
            if not rejection_feedbacks or len(formulas) >= n_formulas:
                break

            # Build feedback for retry
            if retry_idx < max_retries:
                n_retried += 1
                feedback = (
                    "\n**Alert: Duplication Detected in Previous Factor Expressions**\n"
                    + "\n".join(rejection_feedbacks) +
                    "\nRecommendations:\n"
                    "- Avoid the duplicated sub-expressions above\n"
                    "- Generate novel factors by uniquely combining data variables and operations\n"
                    "- Experiment with different function combinations (e.g., TS_CORR, REGBETA, TS_MAD)\n"
                    "- Replace raw variables with transformed variants to enhance expressiveness\n"
                )
                print(f"    Retrying with originality feedback (attempt {retry_idx+2}/{max_retries+1})...")
            else:
                print(f"    Max retries reached, accepting {len(accepted_factors)} factors from this round")

    if len(formulas) < n_formulas:
        # Supplement with random formulas if LLM didn't generate enough
        remaining = n_formulas - len(formulas)
        print(f"  Supplementing with {remaining} random formulas...")
        random_formulas = generate_simulated_formulas(n_formulas=remaining, seed=seed + 1)
        formulas.extend(random_formulas)

    # Report validation statistics
    if regulator.is_available() and n_rejected > 0:
        print(f"  AST regulator: rejected {n_rejected} factors, retried {n_retried} times")

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
    forward_return_series: pd.Series,
    train_end_date: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Spearman Rank-IC for each factor on the training set.

    IC measures the cross-sectional rank correlation between factor values
    at date t and forward returns over forward_period days starting at t.
    This ensures factor ranking is consistent with the prediction horizon
    used by all other baselines (forward_period=10 by default).

    Args:
        factor_df: DataFrame with factor values (datetime x instrument)
        forward_return_series: Forward returns over forward_period days
            (from compute_forward_returns, NOT daily returns)
        train_end_date: End of training period

    Returns:
        (ic_series: mean IC per factor, ic_all_df: IC per date per factor)
    """
    train_idx = factor_df.index.get_level_values('datetime') <= train_end_date

    factor_train = factor_df.loc[train_idx]
    ret_train = forward_return_series.reindex(factor_train.index)

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


def compute_rank_ic_test(
    factor_df: pd.DataFrame,
    forward_return_series: pd.Series,
    test_start_date: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Spearman Rank-IC for each factor on the TEST set (>= test_start_date).

    Mirrors :func:`compute_rank_ic` but evaluates on the out-of-sample period so
    the reported IC/ICIR reflect test performance, not in-sample fit. This is the
    number that should appear in the paper tables.
    """
    test_idx = factor_df.index.get_level_values('datetime') >= test_start_date

    factor_test = factor_df.loc[test_idx]
    ret_test = forward_return_series.reindex(factor_test.index)

    ic_results = {}
    for col in factor_test.columns:
        df = pd.DataFrame({'factor': factor_test[col], 'ret': ret_test})
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
    # Cross-sectional z-score normalization: for each date, normalize across stocks
    # factor_test[f] returns a DataFrame (date x stock) for factor f
    sample = factor_test[available[0]]
    composite = pd.DataFrame(0.0, index=factor_test.index, columns=sample.columns)
    for f in available:
        vals = factor_test[f]  # DataFrame: date x stock
        # Z-score across stocks (axis=1) for each date (row)
        row_mean = vals.mean(axis=1)
        row_std = vals.std(axis=1)
        norm = vals.sub(row_mean, axis=0).div(row_std + 1e-10, axis=0)
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
    train_start_date: str = None,
    train_end_date: str = None,
    test_start_date: str = None,
    test_end_date: str = None,
    use_llm: bool = True,
    forward_period: Optional[int] = None,  # None -> config['evolution']['forward_period'] (10)
    holding_period: Optional[int] = None,  # None -> config['backtest']['holding_period'] or 1
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
        train_start_date: Override data start date
        test_end_date: Override data end date
        train_end_date: Override train end date
        test_start_date: Override test start date
        use_llm: If True, use LLM to generate factors (default True).
                 Falls back to random if LLM is unavailable.
        forward_period: Forward return horizon in trading days (default 10).
            Must match other baselines for fair IC-based factor comparison.
            Used to compute forward returns for Rank-IC evaluation.

    Returns:
        Dict with metrics, IC information, and factor details
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ─────────────────────────────────────────────────
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config.get('data', {}).get('universe', {})
    train_start_date = train_start_date or config['data'].get('train_start_date', '2019-01-01')
    test_end_date = test_end_date or config['data'].get('test_end_date', '2025-12-31')
    train_end_date = train_end_date or config['data'].get('train_end_date', '2023-12-31')
    test_start_date = test_start_date or config['data'].get('test_start_date', '2024-01-01')

    # ── Unified result directory (computed early so data/ + formula artifacts ──
    # ── land in the SAME param-tagged dir as factors/IC/portfolio/results) ──
    # Layout: {output_dir_parent}/hs300_{start}_{end}_forward-{fp}_holding-{hp}/alphaagent/
    # This matches every other baseline's run_dir convention (dash separators +
    # {method} subdir), so cross-method comparison globs keep working.
    method_name = "alphaagent"
    _u = data_cfg.get('index', 'hs300')
    _s = train_start_date
    _e = test_end_date
    # Resolve forward/holding periods: honor explicit args, else fall back to
    # config so BOTH the orchestrator path and the standalone CLI follow
    # config['evolution']['forward_period'] / config['backtest']['trading']['holding_period'].
    # IMPORTANT: reassign the resolved value back into forward_period/holding_period
    # so it flows into compute_forward_returns() below (the raw parameter must NOT
    # be used directly — otherwise a None/empty arg silently breaks IC computation).
    if not forward_period or forward_period <= 0:
        forward_period = config.get('evolution', {}).get('forward_period', 10)
    if not holding_period or holding_period <= 0:
        holding_period = config.get('backtest', {}).get('trading', {}).get('holding_period', 1)
    _fp = forward_period
    _hp = holding_period
    param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
    run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── 1. Load data from main DataLoader ───────────────────────────
    print("=" * 60)
    print("[1/6] Loading data from main DataLoader...")
    print("=" * 60)

    loader = DataLoader(config_path=config_path)
    train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')
    bundle = loader.load_data(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    price_data, fundamental_data, industry_series = bundle.full

    price_midx = convert_to_multindex(price_data)
    return_series = compute_returns(price_midx)
    forward_return_series = compute_forward_returns(price_midx, forward_period=forward_period)

    # Extract prices DataFrame for BacktestEngine (close price, date x stock)
    prices = price_data.get('close')
    if prices is None:
        raise ValueError("Missing 'close' price data for backtest")
    prices = prices.loc[train_start:test_end]

    n_dates = len(price_midx.get('close').index.get_level_values('datetime').unique())
    n_stocks = len(price_midx.get('close').index.get_level_values('instrument').unique())
    print(f"  Loaded: {n_dates} dates x {n_stocks} stocks")
    print(f"  Forward period: {forward_period}d (for IC-based factor ranking)")
    print(f"  Train: <= {train_end_date}  |  Test: >= {test_start_date}")

    # ── 2. Save HDF5 data for AlphaAgent compatibility ─────────────
    print(f"\n[2/6] Generating HDF5 data files...")
    data_dir = os.path.join(run_dir, "data")
    # NOTE: $return = DAILY returns, used as a FACTOR FEATURE only.
    # The IC target (forward_return_series) is passed separately to compute_rank_ic below.
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
    formulas_path = os.path.join(run_dir, "formulas.json")
    with open(formulas_path, 'w') as f:
        json.dump([(n, fm) for n, fm in formulas], f, indent=2)
    print(f"  Saved formulas to: {formulas_path}")

    # ── 4. Compute factor values ───────────────────────────────────
    print(f"\n[4/6] Computing factor values...")
    # NOTE: pass DAILY returns as the $return FEATURE for factor formulas.
    # Do NOT pass forward_return_series here — that would leak future returns into the signal.
    factor_df = compute_factor_values(formulas, price_midx, return_series)
    print(f"  Shape: {factor_df.shape}")

    # ── Factor values are saved into run_dir (the unified result dir) ──

    # Save factor values
    factor_path = os.path.join(run_dir, "factors.csv")
    factor_df.to_csv(factor_path)
    print(f"  Saved factors to: {factor_path}")

    # ── 5. Compute IC and select factors ───────────────────────────
    print(f"\n[5/6] Computing Rank-IC on training set (forward_period={forward_period}d)...")
    ic_mean, ic_df = compute_rank_ic(factor_df, forward_return_series, train_end_date)

    print(f"  Top 10 factors by IC:")
    for i, (name, ic) in enumerate(ic_mean.head(10).items()):
        print(f"    {i+1}. {name}: Rank-IC = {ic:.4f}")

    # ── 5b. Compute Rank-IC on TEST set (the number reported in tables) ──
    print(f"\n[5b/6] Computing Rank-IC on TEST set (>= {test_start_date})...")
    ic_mean_test, ic_df_test = compute_rank_ic_test(
        factor_df, forward_return_series, test_start_date
    )
    print(f"  Top 10 factors by test IC:")
    for i, (name, ic) in enumerate(ic_mean_test.head(10).items()):
        print(f"    {i+1}. {name}: Rank-IC = {ic:.4f}")
    best_ic_test = float(ic_mean_test.iloc[0]) if len(ic_mean_test) > 0 else 0.0
    icir_test = (
        float(best_ic_test / max(ic_mean_test.std(), 1e-10))
        if len(ic_mean_test) > 1 else 0.0
    )

    # Save IC results
    ic_path = os.path.join(run_dir, "ic_results.csv")
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
        end_date=test_end_date,
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
            commission=0.001,
            slippage=0.0,
            risk_free_rate=0.0,
            holding_period=holding_period if holding_period is not None
            else config.get('backtest', {}).get('holding_period', 1),  # Daily rebalance
        )
        metrics = engine.run(portfolios, prices_aligned, save_dir=run_dir)

        # Save portfolio values for analysis
        pv = engine.get_portfolio_values()
        if pv is not None and not pv.empty:
            pv_path = os.path.join(run_dir, "portfolio_values.csv")
            pv.to_csv(pv_path, header=['portfolio_value'])
            print(f"  Portfolio values saved to: {pv_path}")

    # Train IC summary values, needed by the results print block below
    best_ic = float(ic_mean.iloc[0]) if len(ic_mean) > 0 else 0.0
    avg_ic = float(ic_mean.mean()) if len(ic_mean) > 0 else 0.0

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
    print(f"  Mean IC (train):  {best_ic:.4f}")
    print(f"  Mean IC (test):   {best_ic_test:.4f}")
    print(f"  ICIR (test):      {icir_test:.4f}")
    print(f"  N Factors:        {len(ic_mean)}")

    # ── Build result ───────────────────────────────────────────────
    result = {
        'metrics': metrics,
        'mean_rank_ic_train': best_ic,
        'avg_rank_ic_train': avg_ic,
        'mean_rank_ic_test': best_ic_test,
        'icir': float(best_ic / max(ic_mean.std(), 1e-10)) if len(ic_mean) > 1 else 0.0,
        'icir_test': icir_test,
        'n_factors': len(ic_mean),
        'forward_period': forward_period,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'holding_period': holding_period,
        'annual_return': metrics.get('annual_return', 0.0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'information_ratio': metrics.get('information_ratio', 0.0),
        'used_llm': llm_used,
        'llm_model': _read_llm_config(config_path)[2] if llm_used else None,
    }

    # Save full results
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n  Results saved to: {results_path}")

    # ── Save consolidated final_result (mirrors the console summary so the
    #    run is reproducible from disk, not just from terminal scrollback) ──
    # Includes the metrics/ICs AND the expressions of the factors actually
    # used (factor_df columns), which results.json alone does not carry.
    chosen = set(factor_df.columns)
    final_result = dict(result)
    final_result['factors'] = {
        name: expr for name, expr in formulas if name in chosen
    }
    final_path = os.path.join(run_dir, "final_result.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, default=float, ensure_ascii=False)
    print(f"  Final result saved to: {final_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaAgent Baseline Runner")
    parser.add_argument("--config-path", default="config/config.yaml",
                        help="Path to main config YAML")
    parser.add_argument("--output-dir", default="experiments/alphaagent",
                        help="Base directory; actual results land in "
                             "{parent}/hs300_{start}_{end}_forward-{fp}_holding-{hp}/alphaagent/")
    parser.add_argument("--n-formulas", type=int, default=50,
                        help="Number of factor formulas to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--train-start", default=None,
                        help="Data start date (default from config)")
    parser.add_argument("--test-end", default=None,
                        help="Data end date (default from config)")
    parser.add_argument("--train-end", default=None,
                        help="Train end date (default from config)")
    parser.add_argument("--test-start", default=None,
                        help="Test start date (default from config)")
    parser.add_argument("--use-llm", action="store_true", default=True,
                        help="Use LLM to generate factors (default: True)")
    parser.add_argument("--no-llm", action="store_true", default=False,
                        help="Disable LLM, use random factor generation")
    parser.add_argument("--forward-period", type=int, default=None,
                        help="Forward return period in days for IC evaluation "
                             "(default: config['evolution']['forward_period'], 10)")
    parser.add_argument("--holding-period", type=int, default=None,
                        help="Portfolio holding period in days for backtest "
                             "(default: config['backtest']['holding_period'], 1 = daily rebalance)")
    args = parser.parse_args()

    run_alphaagent_baseline(
        config_path=args.config_path,
        output_dir=args.output_dir,
        n_formulas=args.n_formulas,
        seed=args.seed,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        use_llm=not args.no_llm,
        forward_period=args.forward_period,
        holding_period=args.holding_period,
    )
