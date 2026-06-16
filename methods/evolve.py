# -*- coding: utf-8 -*-
"""
Self-Evolving Factor Generator Module

This module implements a self-evolving factor generation system that
iteratively improves factors through generation → backtest → reflection → re-generation.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import re
import warnings
import os
import json
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Try to import scipy for Spearman rank correlation; fall back to pandas-only
try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Try to import openai for LLM calls
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local imports
from backtest.metrics import annualized_sharpe, max_drawdown
from backtest.metrics import rank_ic


@dataclass
class CandidateFactor:
    """Candidate factor for evolution."""
    id: str
    expression: str
    description: str
    parent_id: Optional[str] = None
    generation: int = 0
    ic: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    
    
@dataclass
class EvolutionRound:
    """Record of an evolution round."""
    round_id: int
    seed_factors: List[CandidateFactor]
    improved_factors: List[CandidateFactor]
    best_ic: float
    avg_ic: float
    
    
@dataclass
class EvolutionResult:
    """Result of evolution process."""
    best_factors: List[CandidateFactor]
    evolution_history: List[EvolutionRound]
    best_ic: float
    total_rounds: int
    
    
# ---------------------------------------------------------------------------
# Factor expression evaluator (recursive-descent parser)
# ---------------------------------------------------------------------------

# Token types
_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?)'  # number
    r'|(?P<ident>[a-zA-Z_]\w*)'                   # identifier / function name
    r'|(?P<cmp>[<>!=]=|[<>!=])'                   # comparison operators: >=, <=, ==, !=, >, <, !
    r'|(?P<op>[+\-*/^])'                          # arithmetic operator
    r'|(?P<paren>[()])'                            # parentheses
    r'|(?P<comma>,)'                               # comma
    r'|(?P<invalid>\S+)'                           # catch-all
    r')'
)

def _tokenize(expr: str) -> List[tuple]:
    """Tokenize a factor expression string into (type, value) pairs."""
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if m is None:
            # Skip whitespace
            pos += 1
            continue
        kind = m.lastgroup
        value = m.group(kind)
        if kind == 'number':
            tokens.append(('NUMBER', float(value)))
        elif kind == 'ident':
            tokens.append(('IDENT', value))
        elif kind == 'cmp':
            tokens.append(('CMP', value))
        elif kind == 'op':
            tokens.append(('OP', value))
        elif kind == 'paren':
            tokens.append(('PAREN', value))
        elif kind == 'comma':
            tokens.append(('COMMA', ','))
        elif kind == 'invalid':
            raise ValueError(f"Unexpected token '{value}' in factor expression")
        pos = m.end()
    return tokens


class _FactorExprEvaluator:
    """
    Recursive-descent evaluator for WorldQuant-style factor expressions.

    Supported data sources:
        open, high, low, close, volume, amount  (price_data dict keys)
        pe, pb, roe, market_cap                  (fundamental_data dict keys)

    Supported functions:
        rank(X)          — cross-sectional percentile rank [0, 1]
        ts_rank(X, w)    — rolling time-series rank [0, 1]
        ts_corr(X, Y, w) — rolling Pearson correlation
        ts_mean(X, w)    — rolling mean
        ts_std(X, w)     — rolling std (ddof=0)
        ts_min(X, w)     — rolling minimum
        ts_max(X, w)     — rolling maximum
        ts_sum(X, w)     — rolling sum
        ts_delta(X, w)   — X - delay(X, w)
        ts_zscore(X, w)  — rolling z-score
        ts_decay(X, w)   — exponential weighted moving average (EWMA)
        delay(X, d)      — lag by d periods
        sign(X)          — element-wise sign
        abs(X)           — element-wise absolute value
        log(X)           — element-wise natural log
        sqrt(X)          — element-wise square root (clamped >= 0)
        forward_returns  — 1-day forward return (precomputed)

    Operators: +, -, *, /, ^ (power)
    """

    def __init__(self, data_map: Dict[str, pd.DataFrame]):
        """
        Args:
            data_map: dict mapping field name → pd.DataFrame (dates × stocks)
        """
        self._data = data_map
        self._tokens = []
        self._pos = 0

    def evaluate(self, expr: str) -> pd.DataFrame:
        """Parse and evaluate a factor expression."""
        # --- Input validation ---
        if not expr or not expr.strip():
            raise ValueError("Empty factor expression")

        self._tokens = _tokenize(expr)
        self._pos = 0

        if not self._tokens:
            raise ValueError(f"No valid tokens in expression: {repr(expr)}")

        try:
            result = self._parse_expression()
        except ValueError as e:
            raise ValueError(
                f"Parse error in expression '{expr}': {e}"
            ) from e

        # Ensure result is a DataFrame (not a scalar or Series)
        if isinstance(result, (int, float)):
            # Broadcast scalar to same shape as first data source
            template = next(iter(self._data.values()))
            result = pd.DataFrame(result, index=template.index, columns=template.columns)
        return result

    # ---- Parser ----

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else ('EOF', None)

    def _advance(self):
        tok = self._peek()
        self._pos += 1
        return tok

    def _expect(self, kind, value=None):
        tok = self._advance()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise ValueError(f"Expected {kind}({value}), got {tok}")
        return tok

    def _parse_expression(self) -> pd.DataFrame:
        """expression := comparison (('+' | '-') comparison)*"""
        left = self._parse_comparison()
        while self._peek()[0] == 'OP' and self._peek()[1] in ('+', '-'):
            op = self._advance()[1]
            right = self._parse_comparison()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left


    def _parse_comparison(self) -> pd.DataFrame:
        """comparison := term (CMP term)*"""
        left = self._parse_term()
        while self._peek()[0] == "CMP":
            op = self._advance()[1]
            right = self._parse_term()
            if op == ">":
                left = (left > right).astype(float)
            elif op == "<":
                left = (left < right).astype(float)
            elif op == ">=":
                left = (left >= right).astype(float)
            elif op == "<=":
                left = (left <= right).astype(float)
            elif op == "==":
                left = (left == right).astype(float)
            elif op == "!=":
                left = (left != right).astype(float)
        return left
    def _parse_term(self) -> pd.DataFrame:
        """term := factor (('*' | '/') factor)*"""
        left = self._parse_unary()
        while self._peek()[0] == 'OP' and self._peek()[1] in ('*', '/'):
            op = self._advance()[1]
            right = self._parse_unary()
            if op == '*':
                left = left * right
            else:
                left = left / right.replace(0, np.nan)
        return left

    def _parse_unary(self) -> pd.DataFrame:
        """unary := ('+' | '-')? atom"""
        sign = 1
        while self._peek()[0] == 'OP' and self._peek()[1] in ('+', '-'):
            if self._advance()[1] == '-':
                sign = -sign
        result = self._parse_atom()
        if sign == -1:
            return -result
        return result

    def _parse_atom(self) -> pd.DataFrame:
        """atom := NUMBER | IDENT ['(' args ')'] | '(' expression ')' | POWER"""
        tok = self._peek()

        if tok[0] == 'NUMBER':
            self._advance()
            # Broadcast scalar
            template = next(iter(self._data.values()))
            return pd.DataFrame(tok[1], index=template.index, columns=template.columns)

        if tok[0] == 'PAREN' and tok[1] == '(':
            self._advance()
            result = self._parse_expression()
            self._expect('PAREN', ')')
            return result

        if tok[0] == 'IDENT':
            name = self._advance()[1]
            # Check if it's a function call
            if self._peek()[0] == 'PAREN' and self._peek()[1] == '(':
                return self._parse_func_call(name)
            # Otherwise it's a data column

            # Handle special literals: NaN / Inf / -Inf
            uname = name.upper()
            if uname == "NAN":
                template = next(iter(self._data.values()))
                return pd.DataFrame(np.nan, index=template.index, columns=template.columns)
            if uname == "INF":
                template = next(iter(self._data.values()))
                return pd.DataFrame(np.inf, index=template.index, columns=template.columns)
            if uname == "-INF":
                template = next(iter(self._data.values()))
                return pd.DataFrame(-np.inf, index=template.index, columns=template.columns)

            return self._lookup_data(name)

        raise ValueError(f"Unexpected token {tok}")

    def _parse_func_call(self, name: str) -> pd.DataFrame:
        """Parse function arguments: '(' expression (',' expression)* ')'"""
        self._expect('PAREN', '(')
        args = []
        if self._peek()[0] == 'PAREN' and self._peek()[1] == ')':
            # Empty args — not valid for our functions, but handle gracefully
            pass
        else:
            args.append(self._parse_expression())
            while self._peek()[0] == 'COMMA':
                self._advance()
                args.append(self._parse_expression())
        self._expect('PAREN', ')')
        return self._dispatch_func(name, args)

    def _lookup_data(self, name: str) -> pd.DataFrame:
        """Look up a named data field."""
        if name not in self._data:
            raise ValueError(
                f"Unknown identifier '{name}'. "
                f"Available: {list(self._data.keys())}"
            )
        return self._data[name]

    # ---- Function dispatch ----

    def _dispatch_func(self, name: str, args: List) -> pd.DataFrame:
        """Route function call to implementation."""
        arity = len(args)
        # Check arity
        arity_map = {
            'rank':       (1, 1),
            'ts_rank':    (2, 2),
            'ts_corr':    (3, 3),
            'ts_mean':    (2, 2),
            'ts_std':     (2, 2),
            'ts_min':     (2, 2),
            'ts_max':     (2, 2),
            'ts_sum':     (2, 2),
            'ts_delta':   (2, 2),
            'ts_zscore':  (2, 2),
            'ts_decay':   (2, 2),
            'delay':      (2, 2),
            'sign':       (1, 1),
            'abs':        (1, 1),
            'log':        (1, 1),
            'sqrt':       (1, 1),
            'if':         (3, 3),
        }

        if name in arity_map:
            lo, hi = arity_map[name]
            if not (lo <= arity <= hi):
                raise ValueError(
                    f"Function '{name}' expects {lo}-{hi} args, got {arity}"
                )

        impl = getattr(self, f'_fn_{name}', None)
        if impl is None:
            raise ValueError(f"Unknown function '{name}'")
        return impl(*args)

    # ---- Data access ----

    @staticmethod
    def _safe_int(val, default: int = 5) -> int:
        """
        Convert window/horizon argument to int, NaN-safe.

        Windows are usually parsed from NUMBER tokens → scalar DataFrames.
        NaN can appear if thread contention corrupts the token stream;
        this guard prevents a hard crash.
        """
        if isinstance(val, pd.DataFrame):
            raw = val.iloc[0, 0] if val.values.size > 0 else default
        elif isinstance(val, pd.Series):
            raw = val.iloc[0] if len(val) > 0 else default
        elif isinstance(val, (int, float)):
            raw = val
        else:
            raw = default

        if isinstance(raw, float) and np.isnan(raw):
            return default
        return int(raw)

    # ---- Function implementations ----

    @staticmethod
    def _fn_rank(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank (0~1). NaN remains NaN."""
        return x.rank(axis=1, pct=True, na_option='keep')

    @staticmethod
    def _fn_ts_rank(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling time-series rank within each stock."""
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(3, w // 2)).rank(pct=True) / w

    @staticmethod
    def _fn_ts_corr(x: pd.DataFrame, y: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling Pearson correlation between x and y (per stock)."""
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(10, w)).corr(y)

    @staticmethod
    def _fn_ts_mean(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(1, w // 2)).mean()

    @staticmethod
    def _fn_ts_std(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(5, w * 2 // 3)).std(ddof=0)

    @staticmethod
    def _fn_ts_min(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(1, w // 2)).min()

    @staticmethod
    def _fn_ts_max(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(1, w // 2)).max()

    @staticmethod
    def _fn_ts_sum(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(1, w // 2)).sum()

    @staticmethod
    def _fn_ts_delta(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x - x.shift(w)

    @staticmethod
    def _fn_ts_zscore(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        mean = x.rolling(window=w, min_periods=max(5, w * 2 // 3)).mean()
        std = x.rolling(window=w, min_periods=max(5, w * 2 // 3)).std(ddof=0)
        return (x - mean) / std.replace(0, np.nan)

    @staticmethod
    def _fn_delay(x: pd.DataFrame, d) -> pd.DataFrame:
        periods = _FactorExprEvaluator._safe_int(d)
        return x.shift(periods)

    @staticmethod
    def _fn_sign(x: pd.DataFrame) -> pd.DataFrame:
        return np.sign(x)

    @staticmethod
    def _fn_abs(x: pd.DataFrame) -> pd.DataFrame:
        return x.abs()

    @staticmethod
    def _fn_log(x: pd.DataFrame) -> pd.DataFrame:
        return np.log(x.clip(lower=1e-10))

    @staticmethod
    def _fn_sqrt(x: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(x.clip(lower=0))

    @staticmethod
    def _fn_ts_decay(x: pd.DataFrame, window) -> pd.DataFrame:
        """Exponential weighted moving average (EWMA / decay)."""
        w = _FactorExprEvaluator._safe_int(window)
        return x.ewm(span=w, adjust=False, min_periods=max(3, w // 2)).mean()



    @staticmethod
    def _fn_if(cond: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """if(cond, x, y) — element-wise ternary: return x where cond!=0, else y."""
        return x.where(cond != 0, y)

# ---------------------------------------------------------------------------
# FactorBacktester — lightweight factor backtesting engine
# ---------------------------------------------------------------------------

class FactorBacktester:
    """
    Lightweight backtester for factor evaluation.

    For each factor, this engine:
      1. Parses and evaluates the factor expression on the full data panel
      2. Computes Rank IC (Spearman) at each cross-section
      3. Builds a long-short quantile portfolio and computes Sharpe / win_rate / max_dd

    Constructor supports two calling conventions for backward compatibility:

        Legacy:  FactorBacktester(close_df)
                 → close_df is treated as price_data['close'].

        Full:    FactorBacktester(price_df, volume_df, fundamentals_dict,
                                  forward_period=20)
                 → price_df = close prices (dates × stocks)
                 → volume_df = volume (dates × stocks)
                 → fundamentals_dict = {'pe': df, 'pb': df, ...}
                 → forward_period = forward return horizon in trading days

    If a dict is passed as the first argument, it is treated as price_data
    with keys: open, high, low, close, volume, amount.
    """

    _DEFAULT_FORWARD_PERIOD = 20

    def __init__(
        self,
        prices,
        volume: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
        forward_period: int = _DEFAULT_FORWARD_PERIOD,
        use_qlib: bool = False,
        qlib_provider_uri: Optional[str] = None,
        qlib_topk: int = 50,
    ):
        """
        Initialize the backtester.

        Args:
            prices: If dict → price_data dict (keys: open/high/low/close/volume/amount).
                    If DataFrame → close prices only (legacy API, volume=None required).
            volume: Volume DataFrame, used only when prices is a DataFrame (legacy API).
            fundamentals: Optional dict of fundamental DataFrames (pe, pb, roe, market_cap).
            forward_period: Forward return horizon in trading days.
            use_qlib: If True, use Qlib's professional backtesting for IC/metrics.
            qlib_provider_uri: Qlib data path (only used when use_qlib=True).
            qlib_topk: Portfolio size for Qlib backtest (only used when use_qlib=True).
        """
        # ---- Normalise price_data ----
        if isinstance(prices, dict):
            self.price_data = {k: v.copy() for k, v in prices.items()}
        elif isinstance(prices, pd.DataFrame):
            self.price_data = {'close': prices.copy()}
            if volume is not None:
                self.price_data['volume'] = volume.copy()
        else:
            raise TypeError(f"prices must be dict or DataFrame, got {type(prices)}")

        # Qlib integration flags
        self.use_qlib = use_qlib
        self.qlib_provider_uri = qlib_provider_uri
        self.qlib_topk = qlib_topk

        # Pad missing price fields with empty DataFrames
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if field not in self.price_data:
                # Create a NaN DataFrame of same shape as 'close' (or whatever we have)
                template = next(iter(self.price_data.values()))
                self.price_data[field] = pd.DataFrame(
                    np.nan, index=template.index, columns=template.columns
                )

        self.fundamental_data = fundamentals or {}
        self.forward_period = forward_period

        # ---- Build the unified data map for the expression evaluator ----
        self._data_map = dict(self.price_data)
        self._data_map.update(self.fundamental_data)

        # ---- Compute forward returns ----
        self._forward_returns = self._compute_forward_returns()

        # ---- Expression evaluator (lazy init) ----
        self._evaluator = None
        self._metric_cache = {}      # expression → metrics dict

    @property
    def evaluator(self) -> _FactorExprEvaluator:
        """
        Return a FRESH expression evaluator every time.
        
        The evaluator has mutable parsing state (self._tokens, self._pos),
        so it MUST NOT be shared across threads. Creating a new instance
        on every access is cheap — it only stores a reference to the data.
        """
        return _FactorExprEvaluator(self._data_map)

    # ------------------------------------------------------------------
    # Forward returns
    # ------------------------------------------------------------------

    def _compute_forward_returns(self) -> pd.DataFrame:
        """Compute forward returns for the configured period."""
        close = self.price_data['close']
        fwd = close.pct_change(self.forward_period).shift(-self.forward_period)
        return fwd

    @property
    def forward_returns(self) -> pd.DataFrame:
        return self._forward_returns

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(self, factor: 'CandidateFactor') -> Dict:
        """
        Evaluate a candidate factor.

        If self.use_qlib is True, delegates to evaluate_qlib() for
        professional backtesting via Microsoft Qlib.

        Args:
            factor: CandidateFactor with .expression set.

        Returns:
            Dict with keys: ic, ic_ir, sharpe, win_rate, max_drawdown,
                            long_short_ret, factor_values.
        """
        # Delegate to Qlib if enabled
        if self.use_qlib:
            return self.evaluate_qlib(
                factor,
                provider_uri=self.qlib_provider_uri,
                topk=self.qlib_topk,
            )

        expr = factor.expression

        # Check cache
        if expr in self._metric_cache:
            cached = self._metric_cache[expr].copy()
            # Update the factor object
            factor.ic = cached['ic']
            factor.sharpe = cached['sharpe']
            factor.win_rate = cached['win_rate']
            return cached

        try:
            # Step 1: Compute factor values
            factor_values = self.evaluator.evaluate(expr)
            if factor_values.isna().all().all():
                return self._empty_metrics()

            # Step 2: Align factor values with forward returns
            fwd = self._forward_returns
            common_dates = factor_values.index.intersection(fwd.index)
            common_codes = factor_values.columns.intersection(fwd.columns)
            if len(common_dates) < 10 or len(common_codes) < 5:
                return self._empty_metrics()

            fv_aligned = factor_values.loc[common_dates, common_codes]
            fr_aligned = fwd.loc[common_dates, common_codes]

            # Step 3: Rank IC
            ic, ic_ir = rank_ic(fv_aligned, fr_aligned)

            # Step 4: Quantile portfolio metrics
            sharpe, win_rate, max_dd, long_short_ret = self._compute_quantile_metrics(
                fv_aligned, fr_aligned
            )

            metrics = {
                'ic': float(ic) if not np.isnan(ic) else 0.0,
                'ic_ir': float(ic_ir) if not np.isnan(ic_ir) else 0.0,
                'sharpe': float(sharpe) if not np.isnan(sharpe) else 0.0,
                'win_rate': float(win_rate) if not np.isnan(win_rate) else 0.5,
                'max_drawdown': float(max_dd) if not np.isnan(max_dd) else 0.0,
                'long_short_ret': long_short_ret,
                'factor_values': factor_values,
            }

            # Update the factor object
            factor.ic = metrics['ic']
            factor.sharpe = metrics['sharpe']
            factor.win_rate = metrics['win_rate']

            # Cache
            self._metric_cache[expr] = {k: v for k, v in metrics.items()
                                         if k not in ('long_short_ret', 'factor_values')}

            return metrics

        except Exception as e:
            import traceback
            logger.warning("Factor evaluation failed for '%s': %s", expr, e)
            traceback.print_exc()
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict:
        """Return a safe empty-result dict."""
        return {
            'ic': 0.0, 'ic_ir': 0.0, 'sharpe': 0.0,
            'win_rate': 0.5, 'max_drawdown': 0.0,
            'long_short_ret': pd.Series(dtype=float),
            'factor_values': pd.DataFrame(),
        }

    def evaluate_qlib(
        self,
        factor: 'CandidateFactor',
        factor_values: Optional[pd.DataFrame] = None,
        provider_uri: Optional[str] = None,
        topk: int = 50,
    ) -> Dict:
        """
        Evaluate a candidate factor using Qlib's professional backtesting.

        Uses Qlib's TopkDropoutStrategy + BacktestExecutor for realistic
        portfolio simulation with signal delay and trading costs.

        Args:
            factor: CandidateFactor with .expression set.
            factor_values: Pre-computed factor values (optional; computed if None).
            provider_uri: Qlib data path (default: ~/.qlib/qlib_data/cn_data).
            topk: Number of stocks in portfolio.

        Returns:
            Dict with keys: ic, ic_ir, rank_ic, rank_icir,
                            long_short_sharpe, long_short_return,
                            win_rate, max_drawdown
        """
        from backtest.qlib_backtester import QlibBacktester

        expr = factor.expression

        # Compute factor values if not provided
        if factor_values is None:
            try:
                factor_values = self.compute_factor_values(expr)
            except Exception as e:
                logger.warning("Qlib eval: failed to compute factor values for '%s': %s", expr, e)
                return self._empty_metrics()

        if factor_values.empty or factor_values.isna().all().all():
            return self._empty_metrics()

        # Get date bounds from data
        close_df = self.price_data.get('close')
        if close_df is None or close_df.empty:
            return self._empty_metrics()

        start_time = str(close_df.index[0].date())
        end_time = str(close_df.index[-1].date())

        # Convert stock codes to Qlib format (SH600000 / SZ000001)
        factor_values_qlib = self._to_qlib_codes(factor_values)

        # Run Qlib evaluation
        bt = QlibBacktester(
            provider_uri=provider_uri or '~/.qlib/qlib_data/cn_data',
            topk=topk,
        )
        qlib_metrics = bt.evaluate_factor(
            factor_values_qlib,
            start_time=start_time,
            end_time=end_time,
        )

        # Update factor object with Qlib metrics
        factor.ic = qlib_metrics.get('rank_ic', qlib_metrics.get('ic', 0.0))
        # Map long_short_sharpe to sharpe for compatibility
        factor.sharpe = qlib_metrics.get('long_short_sharpe', 0.0)
        factor.win_rate = qlib_metrics.get('win_rate', 0.5)

        # Normalize output to match standard evaluate() return format
        return {
            'ic': qlib_metrics.get('rank_ic', qlib_metrics.get('ic', 0.0)),
            'ic_ir': qlib_metrics.get('rank_icir', qlib_metrics.get('ic_ir', 0.0)),
            'sharpe': qlib_metrics.get('long_short_sharpe', 0.0),
            'win_rate': qlib_metrics.get('win_rate', 0.5),
            'max_drawdown': qlib_metrics.get('max_drawdown', 0.0),
            'long_short_ret': pd.Series(dtype=float),  # Qlib handles internally
            'factor_values': factor_values,             # Computed locally
        }

    def _to_qlib_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert stock codes to Qlib format (SH600000 / SZ000001).

        Recognizes common A-share code formats:
          - 600000.SH, 000001.SZ → SH600000, SZ000001
          - SH600000, SZ000001   → unchanged (already Qlib format)
          - 600000, 000001       → SH600000, SZ000001 (inferred from first digit)
        """
        new_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            # Already Qlib format?
            if col_str.startswith(('SH', 'SZ', 'BJ')) and len(col_str) == 8:
                new_cols.append(col_str)
                continue

            # Remove .SH/.SZ/.BJ suffix
            for suffix in ('.SH', '.SZ', '.BJ', '.sh', '.sz', '.bj'):
                if col_str.endswith(suffix):
                    code = col_str[:-len(suffix)]
                    market = suffix.upper().replace('.', '')
                    new_cols.append(f'{market}{code}')
                    break
            else:
                # Bare number — infer market from first digit
                if col_str.startswith('6'):
                    new_cols.append(f'SH{col_str}')
                elif col_str.startswith(('0', '3')):
                    new_cols.append(f'SZ{col_str}')
                elif col_str.startswith(('8', '4')):
                    new_cols.append(f'BJ{col_str}')
                else:
                    # Unknown format, keep as-is
                    new_cols.append(col_str)

        result = df.copy()
        result.columns = new_cols
        return result

    def clear_cache(self):
        """Clear cached metrics."""
        self._metric_cache.clear()

    def evaluate_batch(
        self,
        factors: List['CandidateFactor'],
        max_workers: int = 4,
        parallel: bool = True,
    ) -> List[Dict]:
        """
        Evaluate multiple factors.

        When parallel=True, uses ThreadPoolExecutor for speed.
        When parallel=False, runs sequentially — much easier to debug
        because tracebacks are clear and print() output is ordered.

        Args:
            factors: List of CandidateFactor objects to evaluate.
            max_workers: Number of parallel workers (ignored when parallel=False).
            parallel: If False, evaluate factors one by one in series.

        Returns:
            List of metric dicts, same order as `factors`.
        """
        # --- Serial path (debug-friendly) ---
        if not parallel:
            results = []
            for i, factor in enumerate(factors):
                try:
                    results.append(self.evaluate(factor))
                except Exception as e:
                    logger.warning("Eval failed for '%s': %s", factor.expression, e)
                    results.append(self._empty_metrics())
            return results

        # --- Parallel path ---
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        cache_lock = threading.Lock()

        def _evaluate_one(factor):
            expr = factor.expression

            with cache_lock:
                if expr in self._metric_cache:
                    cached = self._metric_cache[expr].copy()
                    factor.ic = cached['ic']
                    factor.sharpe = cached['sharpe']
                    factor.win_rate = cached['win_rate']
                    return cached

            try:
                metrics = self.evaluate(factor)
                with cache_lock:
                    if expr not in self._metric_cache:
                        self._metric_cache[expr] = {
                            k: v for k, v in metrics.items()
                            if k not in ('long_short_ret', 'factor_values')
                        }
                return metrics
            except Exception as e:
                logger.warning("Batch eval failed for '%s': %s", expr, e)
                return self._empty_metrics()

        results = [None] * len(factors)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_evaluate_one, factors[i]): i
                for i in range(len(factors))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = self._empty_metrics()

        return results

    # ------------------------------------------------------------------
    # Quantile portfolio metrics
    # ------------------------------------------------------------------

    def _compute_quantile_metrics(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> Tuple[float, float, float, pd.Series]:
        """
        Build a long-short quantile portfolio and compute performance metrics.

        For each period:
          1. Sort stocks by factor value and assign to n_quantiles
          2. Top quantile = long, bottom quantile = short
          3. Long-short return = equal-weighted top return - equal-weighted bottom return

        Returns:
            (annualized_sharpe, win_rate, max_drawdown, long_short_returns_series)
        """
        long_short_rets = []

        for t in range(len(factor_values)):
            fv = factor_values.iloc[t]
            fr = forward_returns.iloc[t]

            mask = ~(fv.isna() | fr.isna())
            n_valid = mask.sum()
            if n_valid < n_quantiles * 3:
                continue

            fv_valid = fv[mask]
            fr_valid = fr[mask]

            # Assign quantile labels (1 = lowest, n_quantiles = highest)
            try:
                labels = pd.qcut(fv_valid, q=n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                continue

            # Compute equal-weighted return per quantile
            top_mask = labels == labels.max()
            bot_mask = labels == labels.min()

            top_ret = fr_valid[top_mask].mean()
            bot_ret = fr_valid[bot_mask].mean()

            long_short_rets.append(top_ret - bot_ret)

        if not long_short_rets:
            return 0.0, 0.5, 0.0, pd.Series(dtype=float)

        ls_series = pd.Series(long_short_rets, name='long_short')
        sharpe = annualized_sharpe(ls_series)
        win_rate = float((ls_series > 0).mean())
        max_dd = max_drawdown(ls_series)

        return sharpe, win_rate, max_dd, ls_series

    # ------------------------------------------------------------------
    # Factor computation helpers
    # ------------------------------------------------------------------

    def compute_factor_values(self, expression: str) -> pd.DataFrame:
        """Compute raw factor values for a given expression."""
        try:
            return self.evaluator.evaluate(expression)
        except ValueError as e:
            raise ValueError(
                f"Failed to compute factor values for '{expression}': {e}"
            ) from e


class SelfEvolvingGenerator:
    """
    Self-evolving factor generator.
    
    Generates factors through iterative evolution:
    1. Generate seed factors (using LLM)
    2. Backtest and evaluate
    3. Reflect on failures (using LLM)
    4. Generate improvements (using LLM)
    5. Repeat until convergence
    """
    
    def __init__(
        self,
        llm_model: str = "deepseek-ai/DeepSeek-V4-Pro",
        n_seeds: int = 20,
        n_best_factors: int = 10,
        parallel: bool = True,
        api_key: str = "",
        base_url: str = "http://180.163.156.38:53000/v1",
    ):
        """
        Initialize self-evolving generator.
        
        Args:
            llm_model: LLM model to use
            n_seeds: Number of seed factors to generate
            n_best_factors: Number of top factors to select per round and final output
            parallel: If False, evaluate factors serially (easier to debug)
            api_key: API key for LLM service. If empty, reads from config or env.
            base_url: API base URL
        """
        self.llm_model = llm_model
        self.n_seeds = n_seeds
        self.n_best_factors = n_best_factors
        self.parallel = parallel
        
        # Initialize OpenAI client for LLM calls
        self.client = None
        self.use_llm = False
        
        if OPENAI_AVAILABLE:
            # Try to get API key from config file if not provided
            if not api_key:
                try:
                    import yaml
                    with open("config/config.yaml", "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        api_key = config.get("llm", {}).get("generator", {}).get("api_key", "")
                        if not base_url or base_url == "http://180.163.156.38:53000/v1":
                            base_url = config.get("llm", {}).get("generator", {}).get("base_url", base_url)
                            llm_model = config.get("llm", {}).get("generator", {}).get("model", llm_model)
                except Exception:
                    pass
            
            # Fallback to environment variable
            if not api_key:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    # Test connection
                    test_resp = self.client.chat.completions.create(
                        model=llm_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
                    if not hasattr(test_resp, "choices"):
                        raise RuntimeError(f"API test failed: unexpected response type {type(test_resp).__name__}")
                    self.use_llm = True
                    print(f"  [evolve] LLM client initialized: {llm_model} @ {base_url}")
                    print(f"  [evolve] Connection test passed.")
                except Exception as e:
                    print(f"  [evolve] Warning: API connection test failed: {e}")
                    print(f"  [evolve] Falling back to rule-based mode.")
                    self.client = None
                    self.use_llm = False
            else:
                print(f"  [evolve] Warning: No API key found. Set api_key in config or DEEPSEEK_API_KEY env var.")
                print(f"  [evolve] Falling back to rule-based mode.")
                self.use_llm = False
        else:
            print(f"  [evolve] openai package not installed. Running in rule-based mode.")
            self.use_llm = False
        
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.7, expect_json: bool = True) -> str:
        """
        Call LLM and return response content.
        
        Args:
            system_prompt: System prompt for LLM
            user_prompt: User prompt for LLM
            temperature: Sampling temperature
            expect_json: Whether to expect JSON response
            
        Returns:
            LLM response content as string
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Check API key.")
        
        kwargs = dict(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            timeout=180,   # ← 防止 LLM 无限挂起，3 分钟超时
        )
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            resp = self.client.chat.completions.create(**kwargs)
            
            # Validate response shape
            if not hasattr(resp, "choices"):
                resp_type = type(resp).__name__
                resp_preview = str(resp)[:200]
                raise RuntimeError(
                    f"Unexpected LLM response type '{resp_type}'. "
                    f"Response preview: {resp_preview}"
                )
            
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned empty content (None).")
            return content
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  [evolve] LLM call failed ({type(e).__name__}): {e}")
            print(f"  [evolve] Traceback (last frame): {tb.splitlines()[-1] if tb else 'N/A'}")
            raise
    
    @staticmethod
    def _parse_llm_json(raw, expected_keys=("expression", "description")):
        """
        Robustly parse an LLM JSON response into a list of dicts.

        Handles:
        - Markdown code fences (```json ... ```)
        - Leading/trailing explanatory text
        - Response wrapped in an object: {"factors": [...]}
        - Bare JSON array: [...]
        """
        # If already parsed (e.g. future _call_llm change), normalize
        if isinstance(raw, list):
            return [item if isinstance(item, dict) else {} for item in raw]
        if isinstance(raw, dict):
            return SelfEvolvingGenerator._unwrap_if_object(raw, expected_keys)

        if not isinstance(raw, str):
            raise ValueError(f"Unexpected raw type: {type(raw).__name__}")

        text = raw.strip()

        # Strip markdown code fences
        if "```" in text:
            parts = text.split("```", 2)
            if len(parts) >= 3:
                inner = parts[1].strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].strip()
                text = inner

        # Try to extract the first [...] or {...} JSON block
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            idx = text.find(start_char)
            if idx != -1:
                depth = 0
                in_str = False
                escape_next = False
                for j in range(idx, len(text)):
                    ch = text[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == "\\" and in_str:
                        escape_next = True
                        continue
                    if ch == '"' and not escape_next:
                        in_str = not in_str
                    if in_str:
                        continue
                    if ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            candidate = text[idx:j+1]
                            try:
                                parsed = json.loads(candidate)
                                return SelfEvolvingGenerator._unwrap_if_object(parsed, expected_keys)
                            except json.JSONDecodeError:
                                break

        # Last resort: try to parse the whole cleaned text
        try:
            parsed = json.loads(text)
            return SelfEvolvingGenerator._unwrap_if_object(parsed, expected_keys)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Cannot parse LLM response as JSON. "
                f"Cleaned text (first 300 chars): {text[:300]!r}. "
                f"Error: {e}"
            )

    @staticmethod
    def _unwrap_if_object(parsed, expected_keys):
        """If parsed is a dict, try to extract a list from known keys."""
        if isinstance(parsed, list):
            return [item if isinstance(item, dict) else {} for item in parsed]
        if isinstance(parsed, dict):
            for key in ("factors", "factor", "results", "result", "data", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    return [item if isinstance(item, dict) else {} for item in parsed[key]]
            for v in parsed.values():
                if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
                    return [item if isinstance(item, dict) else {} for item in v]
            return [parsed]  # Single-object response
        raise ValueError(f"Parsed JSON is neither list nor dict: {type(parsed).__name__}")

    @staticmethod
    def _fix_parentheses(expr: str) -> str:
        """
        Auto-fix unbalanced parentheses in factor expressions.

        LLMs often generate expressions with wrong paren counts (e.g. one too few
        closing parens). This method counts '(' vs ')' and appends the missing
        closing parens to the end of the expression.

        Returns the original expression if parentheses are already balanced.
        """
        n_open = expr.count('(')
        n_close = expr.count(')')
        if n_open > n_close:
            fixed = expr + ')' * (n_open - n_close)
            print(f"  [evolve] Auto-fixed parentheses: {expr!r} → {fixed!r}")
            return fixed
        if n_close > n_open:
            # More closing than opening — strip trailing excess ')'
            stripped = expr.rstrip(')')
            excess = n_close - n_open
            if excess > 0:
                fixed = stripped + ')' * (n_close - excess)
                print(f"  [evolve] Auto-fixed parentheses: {expr!r} → {fixed!r}")
                return fixed
        return expr

    def _save_factors_to_file(self, factors: List['CandidateFactor'], round_id: int, subdir: str = "self_evolve", filename: str = "generated_factors.json"):
        """
        Save factors to experiments/{yyyymmdd}/{subdir}/round_{round_id}/{filename}
        """
        import os
        import json
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("experiments", date_str, subdir, f"round_{round_id}")
        os.makedirs(save_dir, exist_ok=True)

        factors_dicts = [asdict(f) for f in factors]
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(factors_dicts, f, indent=2, ensure_ascii=False)

        print(f"  [evolve] Saved {len(factors)} factors to {save_path}")

    def _save_metrics_to_csv(self, factors: List['CandidateFactor'], metrics_list: List[Dict], round_id: int):
        """
        Save backtest metrics to experiments/{yyyymmdd}/self_evolve/round_{round_id}/backtest_factor_metrics.csv
        """
        import os
        import csv
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("experiments", date_str, "self_evolve", f"round_{round_id}")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "backtest_factor_metrics.csv")
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["factor_id", "expression", "ic", "ic_ir", "sharpe", "win_rate", "max_drawdown"])
            for factor, metrics in zip(factors, metrics_list):
                writer.writerow([
                    factor.id,
                    factor.expression,
                    f"{metrics.get('ic', 0.0):.6f}",
                    f"{metrics.get('ic_ir', 0.0):.6f}",
                    f"{metrics.get('sharpe', 0.0):.6f}",
                    f"{metrics.get('win_rate', 0.5):.6f}",
                    f"{metrics.get('max_drawdown', 0.0):.6f}",
                ])

        print(f"  [evolve] Saved {len(factors)} factor metrics to {save_path}")

    def _generate_reflection(self, evaluated_factors: List['CandidateFactor'], top_factors: List['CandidateFactor'], round_id: int = 0) -> str:
        """
        Generate reflection text based on backtest results.
        
        Analyzes successful and failed factors to produce strategic reflection notes,
        which will be injected into the next round's LLM prompt to guide factor generation.
        
        Args:
            evaluated_factors: All factors evaluated in this round
            top_factors: Top-performing factors (IC > 0)
            round_id: Current evolution round number
            
        Returns:
            Reflection text (multi-line string)
        """
        if not evaluated_factors:
            return "No factors evaluated in this round."
        
        # --- 1. Categorize factors by performance ---
        successful = [f for f in evaluated_factors if f.ic is not None and f.ic > 0.02]
        moderate = [f for f in evaluated_factors if f.ic is not None and 0 <= f.ic <= 0.02]
        failed = [f for f in evaluated_factors if f.ic is not None and f.ic < 0]
        
        lines = []
        lines.append(f"=== Reflection for Round {round_id + 1} ===")
        lines.append(f"Total factors evaluated: {len(evaluated_factors)}")
        lines.append(f"Successful (IC > 0.02): {len(successful)}")
        lines.append(f"Moderate (0 <= IC <= 0.02): {len(moderate)}")
        lines.append(f"Failed (IC < 0): {len(failed)}")
        lines.append("")
        
        # --- 2. Analyze successful factors: common patterns ---
        if successful:
            lines.append("--- Successful Factor Patterns ---")
            
            # Extract expression features
            momentum_count = 0
            value_count = 0
            quality_count = 0
            volatility_count = 0
            growth_count = 0
            liquidity_count = 0
            combo_count = 0  # Factors with 2+ operator types
            
            for f in successful:
                expr = f.expression.lower()
                features = []
                if any(k in expr for k in ['ts_delta', 'ts_rank', 'momentum', 'return_']):
                    momentum_count += 1
                    features.append('momentum')
                if any(k in expr for k in ['pe', 'pb', 'ps', 'value', 'roe', 'roa']):
                    value_count += 1
                    features.append('value')
                if any(k in expr for k in ['roe', 'roa', 'margin', 'quality', 'debt']):
                    quality_count += 1
                    features.append('quality')
                if any(k in expr for k in ['std', 'vol', 'var', 'risk']):
                    volatility_count += 1
                    features.append('volatility')
                if any(k in expr for k in ['growth', 'revenue', 'eps']):
                    growth_count += 1
                    features.append('growth')
                if any(k in expr for k in ['turnover', 'volume', 'liq']):
                    liquidity_count += 1
                    features.append('liquidity')
                if len(features) >= 2:
                    combo_count += 1
            
            lines.append(f"Feature distribution in successful factors:")
            if momentum_count > 0:
                lines.append(f"  - Momentum-related: {momentum_count}")
            if value_count > 0:
                lines.append(f"  - Value-related: {value_count}")
            if quality_count > 0:
                lines.append(f"  - Quality-related: {quality_count}")
            if volatility_count > 0:
                lines.append(f"  - Volatility-related: {volatility_count}")
            if growth_count > 0:
                lines.append(f"  - Growth-related: {growth_count}")
            if liquidity_count > 0:
                lines.append(f"  - Liquidity-related: {liquidity_count}")
            if combo_count > 0:
                lines.append(f"  - Combined (2+ features): {combo_count} → Suggest mixing factor types in next generation")
            
            # Top 3 successful factor expressions
            top3 = sorted(successful, key=lambda x: x.ic, reverse=True)[:3]
            lines.append("")
            lines.append("Top successful factor expressions:")
            for f in top3:
                sharpe_str = f"{f.sharpe:.2f}" if f.sharpe else "N/A"
                lines.append(f"  - IC={f.ic:.4f}, Sharpe={sharpe_str}: {f.expression}")
            
            lines.append("")
        
        # --- 3. Analyze failed factors: root causes ---
        if failed:
            lines.append("--- Failed Factor Analysis ---")
            
            # Check common issues
            neg_momentum = [f for f in failed if any(k in f.expression.lower() for k in ['ts_delta', 'momentum'])]
            neg_value = [f for f in failed if any(k in f.expression.lower() for k in ['pe', 'pb', 'value'])]
            neg_volatile = [f for f in failed if 'std' in f.expression.lower() or 'vol' in f.expression.lower()]
            too_complex = [f for f in failed if len(re.findall(r'[+\-*/()]', f.expression)) > 10]
            
            if neg_momentum:
                lines.append(f"  - {len(neg_momentum)} momentum factors have negative IC → Consider reversing sign or adjusting window")
            if neg_value:
                lines.append(f"  - {len(neg_value)} value factors have negative IC → Current market may not favor value style")
            if neg_volatile:
                lines.append(f"  - {len(neg_volatile)} volatility factors have negative IC → Volatility may be mis-priced")
            if too_complex:
                lines.append(f"  - {len(too_complex)} factors are overly complex (too many operators) → Simplify expressions")
            
            lines.append("")
        
        # --- 4. Strategic suggestions for next round ---
        lines.append("--- Suggestions for Next Generation ---")
        if successful:
            best = max(successful, key=lambda x: x.ic or 0)
            lines.append(f"  1. Try extending the successful pattern: {best.expression[:50]}...")
            if combo_count > 0:
                lines.append("  2. Increase factor combinations (momentum + value, quality + reversal)")
            if len(successful) < 5:
                lines.append("  3. Current successful factors are few → Increase generation diversity (higher temperature)")
            else:
                lines.append("  3. Sufficient successful factors → Try fine-tuning window sizes")
        else:
            lines.append("  1. No successful factors this round → Try completely different factor templates")
            lines.append("  2. Consider lowering LLM temperature to get more conservative factors")
        
        if failed and len(failed) > len(successful):
            lines.append("  4. High failure rate → Add more fundamental filters (pe > 0, roe > 0)")
        
        lines.append("")
        lines.append("=== End of Reflection ===")
        
        return "\n".join(lines)

    def generate_seed_factors(self) -> List['CandidateFactor']:
        """
        Generate seed factors using LLM.
        
        Uses self.n_seeds (set at construction time) as the target count.
            
        Returns:
            List of seed factors
        """
        n_factors = self.n_seeds
        print(f"Generating {n_factors} seed factors...")
        
        # Try to use LLM to generate factors
        if self.use_llm and self.client:
            try:
                seed_factors = self._generate_factors_via_llm(n_factors)
                if seed_factors and len(seed_factors) >= 1:
                    result = seed_factors[:n_factors]
                    print(f"Generated {len(result)} seed factors via LLM")
                    self._save_factors_to_file(result, round_id=0)
                    return result
            except Exception as e:
                print(f"  [evolve] LLM factor generation failed: {e}")
                print(f"  [evolve] Falling back to rule-based generation.")
        
        # Fallback: rule-based generation
        print(f"  [evolve] Using rule-based factor generation.")
        fallback_factors = self._generate_factors_rule_based(n_factors)
        self._save_factors_to_file(fallback_factors, round_id=0)
        return fallback_factors
    
    def _generate_factors_via_llm(self, n_factors: int) -> List[CandidateFactor]:
        """
        Generate factors using LLM.
        
        Args:
            n_factors: Number of factors to generate
            
        Returns:
            List of candidate factors
        """
        system_prompt = """You are a quantitative factor research expert specializing in A-share stock selection.
Your task is to generate diverse, economically meaningful factor expressions for stock selection.

Supported data sources:
- open, high, low, close, volume, amount (price and volume data)
- pe, pb, roe, market_cap (fundamental data)

Supported functions (WorldQuant style):
- rank(X): cross-sectional percentile rank [0, 1]
- ts_rank(X, w): rolling time-series rank within window w
- ts_corr(X, Y, w): rolling correlation between X and Y over window w
- ts_mean(X, w): rolling mean over window w
- ts_std(X, w): rolling standard deviation over window w
- ts_min(X, w): rolling minimum over window w
- ts_max(X, w): rolling maximum over window w
- ts_sum(X, w): rolling sum over window w
- ts_delta(X, w): X - delay(X, w)
- ts_zscore(X, w): rolling z-score over window w
- delay(X, d): lag by d periods
- sign(X): element-wise sign
- abs(X): element-wise absolute value
- log(X): element-wise natural log
- sqrt(X): element-wise square root

Supported operators: +, -, *, /, ^

Factor categories to cover:
1. Momentum: price trend, volume confirmation, reversal
2. Value: valuation ratios, mean reversion
3. Quality: profitability, earnings quality, balance sheet strength
4. Liquidity: trading volume, turnover, bid-ask spread
5. Growth: earnings growth, revenue growth, analyst revisions

Return a JSON array of objects, each with "expression" and "description" keys.
Ensure expressions are valid and can be evaluated by the factor engine."""

        # Build user_prompt with valid JSON example (avoid f-string escaping issues)
        json_example = json.dumps({
            "factors": [
                {"expression": "rank(ts_corr(close, volume, 20))", "description": "Price-volume correlation momentum"},
                {"expression": "-rank(pe)", "description": "Value factor based on P/E ratio"},
                {"expression": "rank(ts_zscore(roe, 60))", "description": "Quality factor based on ROE z-score"}
            ]
        }, ensure_ascii=False)

        user_prompt = (
            f"Please generate {n_factors} diverse factor expressions for A-share stock selection.\n"
            + "Requirements:\n"
            + "1. Factors should be diverse and cover different categories (momentum, value, quality, liquidity, growth)\n"
            + "2. Expressions must be valid and use only supported functions and data sources\n"
            + "3. Avoid trivial factors (e.g., just \"close\" or \"volume\")\n"
            + "4. Each factor should have economic intuition\n"
            + "5. You MUST return exactly " + str(n_factors) + " factors\n"
            + "6. Return a JSON object with key \"factors\" (array of factor objects), no other text, no markdown fences\n"
            + "\n"
            + "Example (valid JSON object with \"factors\" key):\n"
            + json_example + "\n\n"
            + f"Please generate {n_factors} factors now. Return only the JSON object."
        )

        try:
            raw = self._call_llm(system_prompt, user_prompt, temperature=0.7, expect_json=True)
            factors_json = self._parse_llm_json(raw)



            seed_factors = []
            for i, f in enumerate(factors_json):
                if not isinstance(f, dict) or "expression" not in f:
                    continue
                expr = self._fix_parentheses(f["expression"])
                factor = CandidateFactor(
                    id=f"seed_llm_{i}",
                    expression=expr,
                    description=f.get("description", f"LLM-generated factor {i}"),
                    generation=0,
                )
                seed_factors.append(factor)
            
            return seed_factors
            
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [evolve] Failed to parse LLM response: {e}")
            print(f"  [evolve] Raw response: {raw[:300] if 'raw' in locals() else 'N/A'}")
            raise
        except Exception as e:
            raise
    
    def _generate_factors_rule_based(self, n_factors: int) -> List[CandidateFactor]:
        """
        Generate factors using rule-based method (fallback when LLM is unavailable).
        
        Args:
            n_factors: Number of factors to generate
            
        Returns:
            List of candidate factors
        """
        # Template expressions for different factor categories
        templates = [
            # Momentum factors
            ("rank(ts_corr(close, volume, 20))", "Price-volume correlation momentum"),
            ("rank(ts_delta(close, 5))", "Short-term price momentum"),
            ("rank(ts_delta(close, 20))", "Medium-term price momentum"),
            ("rank(ts_zscore(close, 20))", "Mean-reversion based on z-score"),
            ("rank(ts_rank(close, 10))", "Short-term ranking momentum"),
            # Value factors
            ("-rank(pe)", "Value factor based on P/E ratio"),
            ("-rank(pb)", "Value factor based on P/B ratio"),
            ("rank(1 / pe)", "Earnings yield factor"),
            ("rank(1 / pb)", "Book-to-market factor"),
            # Quality factors
            ("rank(ts_zscore(roe, 60))", "ROE quality factor"),
            ("rank(roe)", "Direct ROE factor"),
            ("rank(market_cap)", "Size factor (large-cap)"),
            ("-rank(market_cap)", "Size factor (small-cap)"),
            # Liquidity factors
            ("rank(ts_mean(volume, 20))", "Average trading volume"),
            ("rank(volume / delay(volume, 1))", "Volume change rate"),
            ("rank(abs(ts_delta(close, 1)) / volume)", "Price impact (illiquidity)"),
            # Combined factors
            ("rank(ts_corr(close, volume, 20)) * -rank(pe)", "Combined momentum-value factor"),
            ("rank(ts_zscore(close, 20)) * rank(roe)", "Combined momentum-quality factor"),
            ("rank(ts_delta(close, 10)) * -rank(market_cap)", "Combined momentum-size factor"),
        ]
        
        seed_factors = []
        for i in range(n_factors):
            template_idx = i % len(templates)
            expr, desc = templates[template_idx]
            
            # Add slight variation to avoid exact duplicates
            if i >= len(templates):
                # Modify window size for time-series functions
                import re
                expr = re.sub(r'(\d+)', lambda m: str(max(5, int(m.group(1)) + (i % 10 - 5))), expr)
            
            factor = CandidateFactor(
                id=f"seed_rule_{i}",
                expression=expr,
                description=desc,
                generation=0,
            )
            seed_factors.append(factor)
        
        return seed_factors
    
    def evolve(
        self,
        seed_factors: List[CandidateFactor],
        backtester: FactorBacktester,
        n_rounds: int = 10,
    ) -> EvolutionResult:
        """
        Evolve factors through iterative improvement.
        
        Args:
            seed_factors: Seed factors to evolve
            backtester: Backtester for evaluation
            n_rounds: Number of evolution rounds
            
        Returns:
            Evolution result
        """
        print(f"\nStarting evolution ({n_rounds} rounds)...")
        
        evolution_history = []
        current_factors = seed_factors
        
        best_ic = 0.0
        
        for round_id in range(n_rounds):
            print(f"\nRound {round_id + 1}/{n_rounds}")
            
            # Evaluate current factors
            mode = "parallel" if self.parallel else "serial"
            print(f"  Evaluating {len(current_factors)} factors ({mode})...")
            metrics_list = backtester.evaluate_batch(current_factors, max_workers=4, parallel=self.parallel)
            
            evaluated_factors = []
            for i, factor in enumerate(current_factors):
                metrics = metrics_list[i]
                factor.ic = metrics.get('ic', 0.0)
                factor.sharpe = metrics.get('sharpe', 0.0)
                factor.win_rate = metrics.get('win_rate', 0.5)
                evaluated_factors.append(factor)
                
                if factor.ic > best_ic:
                    best_ic = factor.ic

            # Save backtest metrics for this round
            self._save_metrics_to_csv(current_factors, metrics_list, round_id)
            
            # Select top factors
            top_factors = sorted(evaluated_factors, key=lambda x: x.ic, reverse=True)[:self.n_best_factors]
            
            # Generate reflection based on backtest results
            reflection_notes = self._generate_reflection(evaluated_factors, top_factors, round_id)
            print(f"\n  [evolve] Reflection notes for round {round_id + 1}:\n{reflection_notes}\n")
            
            # Generate improvements (try LLM first, fall back to rule-based)
            improved_factors = None
            if self.use_llm and self.client:
                try:
                    llm_factors = self._generate_improvements_via_llm(top_factors, round_id, reflection_notes)
                    if llm_factors and len(llm_factors) >= len(top_factors) // 2:
                        improved_factors = llm_factors
                        print(f"  [evolve] Generated {len(improved_factors)} improved factors via LLM")
                except Exception as e:
                    print(f"  [evolve] LLM improvement generation failed: {e}")

            if improved_factors is None:
                print(f"  [evolve] Using rule-based factor improvement.")
                improved_factors = self._generate_improvements_rule_based(top_factors)

            # Save improved factors for this round
            self._save_factors_to_file(improved_factors, round_id + 1, filename="improved_factors.json")
            
            # Record evolution round
            round_record = EvolutionRound(
                round_id=round_id,
                seed_factors=current_factors,
                improved_factors=improved_factors,
                best_ic=best_ic,
                avg_ic=np.mean([f.ic for f in top_factors]),
            )
            evolution_history.append(round_record)
            
            # Update current factors
            current_factors = improved_factors
            
            print(f"  Best IC: {best_ic:.4f}")
            print(f"  Avg IC (top {self.n_best_factors}): {round_record.avg_ic:.4f}")
            
            # Check convergence
            if self._check_convergence(evolution_history):
                print("\nConvergence reached!")
                break
        
        # Select best factors
        all_factors = []
        for round_record in evolution_history:
            all_factors.extend(round_record.improved_factors)
        
        best_factors = sorted(all_factors, key=lambda x: x.ic, reverse=True)[:self.n_best_factors]
        
        return EvolutionResult(
            best_factors=best_factors,
            evolution_history=evolution_history,
            best_ic=best_ic,
            total_rounds=len(evolution_history),
        )
    
    def _generate_improvements_via_llm(self, top_factors: List['CandidateFactor'], round_id: int = 0, reflection_notes: str = "") -> List['CandidateFactor']:
        """
        Generate improved factors using LLM.
        
        Args:
            top_factors: Top-performing factors
            
        Returns:
            List of improved factors
        """
        system_prompt = """You are a quantitative factor research expert specializing in factor improvement.
Your task is to analyze top-performing factors and generate improved versions.

Supported operations for factor improvement:
1. Parameter tuning: Adjust window sizes (e.g., change 20 to 10 or 40)
2. Combination: Combine multiple good factors using arithmetic operations
3. Nonlinear transformation: Apply log, sqrt, abs, sign to enhance signal
4. Cross-sectional ranking: Apply rank() to normalize signals
5. Time-series operations: Use ts_zscore, ts_rank, ts_mean for robustness
6. Fundamental integration: Combine price signals with pe, pb, roe, market_cap

Guidelines:
- Improvements should be non-trivial (not just changing window size by 1)
- Avoid overfitting: don't create overly complex expressions
- Ensure expressions are valid and can be evaluated
- Return a JSON object with key "factors" (a list of objects with "expression" and "description" keys)"""

        # Build prompt with top factors and reflection notes
        factors_str = "\n".join([
            f"  {f.id}: {f.expression} (IC={f.ic:.4f}, Sharpe={f.sharpe:.2f}) - {f.description}"
            for f in top_factors[:self.n_best_factors]  # Only show top N to avoid prompt overflow
        ])
        
        # Build reflection section
        reflection_section = ""
        if reflection_notes:
            # Truncate to avoid prompt overflow (keep last 1500 chars)
            notes_truncated = reflection_notes[-1500:] if len(reflection_notes) > 1500 else reflection_notes
            reflection_section = f"""
=== Reflection Notes from Previous Round ===
{notes_truncated}

Use these reflection notes to guide your improvements:
- Avoid the failure patterns mentioned above
- Extend the successful factor patterns
- Consider the strategic suggestions when generating new factors
=== End of Reflection Notes ===
"""
        
        user_prompt = f"""Based on the following top-performing factors, generate improved versions.

Top factors:
{factors_str}
{reflection_section}
Requirements:
1. Generate {len(top_factors)} improved factors
2. Each improvement should be based on one or more of the top factors
3. Improvements can include:
   - Parameter tuning (change window sizes)
   - Factor combination (combine 2-3 good factors)
   - Adding fundamental data (pe, pb, roe, market_cap)
   - Applying nonlinear transformations
4. Ensure expressions are valid and diverse
5. Return strictly as a JSON array
5. Return a JSON object with key "factors" (array of factor objects)

Example format:
{{
  "factors": [
    {{"expression": "rank(ts_corr(close, volume, 10))", "description": "Improved momentum factor with shorter window"}},
    {{"expression": "rank(ts_corr(close, volume, 20)) * -rank(pe)", "description": "Combined momentum-value factor"}}
  ]
}}

Please generate improved factors now. Return only the JSON object, no other text."""

        try:
            raw = self._call_llm(system_prompt, user_prompt, temperature=0.8, expect_json=True)
            factors_json = self._parse_llm_json(raw)

            improved_factors = []
            for i, f in enumerate(factors_json):
                if not isinstance(f, dict) or "expression" not in f:
                    continue
                expr = self._fix_parentheses(f["expression"])
                # Find parent factor
                parent_id = top_factors[i % len(top_factors)].id if top_factors else None
                
                factor = CandidateFactor(
                    id=f"improved_llm_{parent_id}_{i}" if parent_id else f"improved_llm_{i}",
                    expression=expr,
                    description=f.get("description", f"LLM-improved factor {i}"),
                    parent_id=parent_id,
                    generation=(top_factors[0].generation + 1) if top_factors else 1,
                )
                improved_factors.append(factor)
            
            return improved_factors
            
        except json.JSONDecodeError as e:
            print(f"  [evolve] Failed to parse LLM response as JSON: {e}")
            print(f"  [evolve] Raw response: {raw[:200] if 'raw' in locals() else 'N/A'}")
            raise
        except Exception as e:
            raise
    
    def _generate_improvements_rule_based(self, top_factors: List['CandidateFactor']) -> List['CandidateFactor']:
        """Generate improved factors using four rule-based strategies (fallback when LLM unavailable)."""
        import random
        random.seed(42)

        improved_factors = []

        for i, factor in enumerate(top_factors):
            expr = factor.expression
            other_expr = top_factors[(i + 1) % len(top_factors)].expression if len(top_factors) > 1 else expr

            # Strategy 1: Adjust window size in time-series functions
            win_parts = expr.split('(')
            win_result = []
            for k, part in enumerate(win_parts):
                if k == 0:
                    win_result.append(part)
                    continue
                sub_parts = part.split(',')
                for m in range(1, len(sub_parts)):
                    nm = re.match(r'\s*(\d+)', sub_parts[m])
                    if nm:
                        n = int(nm.group(1))
                        if 5 <= n <= 60:
                            sub_parts[m] = sub_parts[m].replace(str(n), str(max(5, n + random.randint(-10, 10))), 1)
                win_result.append(','.join(sub_parts))
            adjusted_expr = '('.join(win_result)

            # Strategy 2: Add fundamental signal
            fund = random.choice(['pe', 'pb', 'roe', 'market_cap'])
            if random.random() < 0.5:
                fund_expr = f"rank({expr}) * -rank({fund})"
            else:
                fund_expr = f"rank({expr}) + rank(-{fund})"

            # Strategy 3: Apply nonlinear transformation
            transform = random.choice(['log', 'sqrt', 'abs', 'sign'])
            nonlinear_expr = f"{transform}({expr})"

            # Strategy 4: Combine with another top factor
            op = random.choice(['*', '+'])
            combined_expr = f"rank({expr}) {op} rank({other_expr})"

            improvements = [adjusted_expr, fund_expr, nonlinear_expr, combined_expr]

            for j, improved_expr in enumerate(improvements):
                if improved_expr and improved_expr != expr:
                    improved = CandidateFactor(
                        id=f"improved_rule_{factor.id}_{i}_{j}",
                        expression=improved_expr,
                        description=f"Rule-based improvement of {factor.description} (strategy {j+1})",
                        parent_id=factor.id,
                        generation=factor.generation + 1,
                    )
                    improved_factors.append(improved)

        return improved_factors
    
    def _check_convergence(self, history: List[EvolutionRound]) -> bool:
        """
        Check if evolution has converged.
        
        Args:
            history: Evolution history
            
        Returns:
            True if converged
        """
        if len(history) < 2:
            return False
        
        # Check if IC improvement is below threshold
        recent_ics = [h.best_ic for h in history[-2:]]
        improvement = recent_ics[-1] - recent_ics[-2]
        
        return improvement < 0.003  # convergence_delta


if __name__ == '__main__':
    """Quick smoke test for expression evaluator and backtester."""
    print("=== Factor Evaluator Smoke Test ===\n")

    np.random.seed(42)
    n_dates, n_stocks = 252, 50
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]

    close_prices = pd.DataFrame(
        10 + np.cumsum(np.random.randn(n_dates, n_stocks) * 0.02 + 0.0005, axis=0),
        index=dates, columns=codes,
    )
    volume_data = pd.DataFrame(
        np.abs(np.random.randn(n_dates, n_stocks) * 1e6 + 5e6),
        index=dates, columns=codes,
    )

    backtester = FactorBacktester(prices=close_prices, volume=volume_data, forward_period=20)

    test_expressions = [
        "rank(ts_corr(close, volume, 20))",
        "rank(ts_delta(close, 5))",
        "ts_zscore(close, 20) * -1",
    ]

    for expr in test_expressions:
        fv = backtester.compute_factor_values(expr)
        print(f"  {expr}: shape={fv.shape}, nan={fv.isna().sum().sum()}")

    print("\nFactor evaluation metrics:")
    for expr in test_expressions:
        f = CandidateFactor(id="test", expression=expr, description=f"Test: {expr}")
        m = backtester.evaluate(f)
        print(f"  {expr}: IC={m['ic']:.4f}, Sharpe={m['sharpe']:.2f}, "
              f"WinRate={m['win_rate']:.2%}, MaxDD={m['max_drawdown']:.2%}")

    print("\n=== Smoke Test Complete ===")
