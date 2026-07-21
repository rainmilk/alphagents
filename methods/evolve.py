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
import ast
import difflib
import warnings
import os
import json
import logging

from config import config_path

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
    icir: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    val_ic: float = 0.0            # Validation-period Rank-IC (set only when a val_backtester is supplied). Used for final selection + early-stop to combat train overfitting.
    val_icir: float = 0.0
    is_valid: bool = True          # False when evaluation raises ValueError (parse error)
    parse_error: str = ""          # stores error message for debugging / reflection
    originality_ok: bool = True    # False when the AST originality gate rejects the factor
    gate_reason: str = ""          # reason recorded by the originality gate
    
    
@dataclass
class EvolutionRound:
    """Record of an evolution round — stores evaluated factors for this round."""
    round_id: int
    factors: List[CandidateFactor]   # evaluated factors in this round
    best_ic: float
    avg_ic: float


@dataclass
class EvolutionResult:
    """Result of evolution process."""
    best_factors: List[CandidateFactor]
    evolution_history: List[EvolutionRound]
    total_rounds: int
    
    
# ---------------------------------------------------------------------------
# AST-based originality gate (ported spirit of AlphaAgent's OriginalityRegulator)
# ---------------------------------------------------------------------------

# MASE factor DSL: canonical function names accepted by the evaluator.
_GATE_ALLOWED_FUNCS = {
    'rank', 'ts_rank', 'ts_corr', 'ts_cov', 'ts_mean', 'ts_std', 'ts_var',
    'ts_skew', 'ts_kurt', 'ts_min', 'ts_max', 'ts_sum', 'ts_delta',
    'ts_zscore', 'ts_decay', 'delay', 'sign', 'abs', 'log', 'sqrt', 'if',
    'ts_pct_change',
}
# LLM-frequent aliases → canonical (mirrors _FactorExprEvaluator._FUNC_ALIASES)
_GATE_FUNC_ALIASES = {
    'ts_stddev': 'ts_std', 'ts_std_dev': 'ts_std', 'stddev': 'ts_std',
    'ts_average': 'ts_mean', 'ts_avg': 'ts_mean', 'ts_median': 'ts_mean',
    'ts_diff': 'ts_delta', 'ts_delay': 'delay', 'ts_lag': 'delay', 'lag': 'delay',
    'cov': 'ts_cov', 'skew': 'ts_skew', 'kurt': 'ts_kurt', 'kurtosis': 'ts_kurt',
    'max': 'ts_max', 'min': 'ts_min', 'mean': 'ts_mean', 'sum': 'ts_sum',
    'ts_pctchange': 'ts_pct_change', 'pct_change': 'ts_pct_change',
    'ts_pctchg': 'ts_pct_change',
    'ts_roc':       'ts_pct_change',   # rate of change = pct change
    'ts_skewness': 'ts_skew', 'ts_kurtosis': 'ts_kurt',
}
# AST node types that count toward "structural size" (excludes Expression/Load
# and bare operator singletons like ast.Add).
_GATE_MEANINGFUL = (ast.Name, ast.Call, ast.Constant, ast.BinOp,
                    ast.Compare, ast.IfExp, ast.UnaryOp)


class FactorOriginalityGate:
    """
    Static AST gate that runs independently of (and cheaper than) backtesting.

    Three checks — mirrors the intent of AlphaAgent's OriginalityRegulator:
      1. Syntax     — expression must parse as a Python-evaluable factor.
      2. Semantics  — every called function must be in MASE's function library.
      3. Originality— reject (a) structural near-duplicates of an already
         accepted factor, and (b) degenerate factors (no data dependency,
         single token, or constant-dominated).

    CALIBRATION NOTE: AlphaAgent computes constant/var RATIOS against its
    pyparsing node count. Replicating that exactly on MASE's AST would reject
    legitimate factors such as `rank(close - open)` (vars would be >= half the
    nodes). We therefore use MASE-calibrated heuristics (see _is_trivial) that
    keep genuine factors while still killing `close`, `5`, `rank(5)`, and
    structurally near-identical variants. Variable NAMES are preserved in the
    canonical signature (only constants are normalized to '#') so economically
    distinct but structurally-identical factors (e.g. `close-open` vs `high-low`)
    are NOT falsely collapsed.
    """

    def __init__(self, enabled: bool = True, duplication_threshold: int = 8,
                 dedup_similarity: float = 0.90):
        self.enabled = enabled
        # duplication_threshold retained for API parity / future subtree sizing
        self.duplication_threshold = duplication_threshold
        self.dedup_similarity = dedup_similarity
        self.zoo: List[str] = []        # canonical signatures of accepted factors
        self.zoo_exprs: List[str] = []  # original expressions (for reporting)

    # -- preprocessing so the MASE DSL parses as a Python AST ---------------
    @staticmethod
    def _preprocess(expr: str) -> str:
        # MASE uses '^' for power; Python uses '**'
        e = expr.replace('^', '**')
        # MASE 'if(cond,a,b)' is a function; 'if' is a Python keyword → rename
        e = re.sub(r'\bif\s*\(', '_if(', e, flags=re.IGNORECASE)
        return e

    def _parse(self, expr: str):
        return ast.parse(self._preprocess(expr), mode='eval')

    @staticmethod
    def _canon_name(name: str) -> str:
        return 'if' if name == '_if' else name

    def _canonical(self, node) -> str:
        """Canonical signature: keeps variable names, normalizes constants to '#'."""
        if isinstance(node, ast.Expression):
            return self._canonical(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return '#'
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = '-' if isinstance(node.op, ast.USub) else '+'
            return f'{sign}{self._canonical(node.operand)}'
        if isinstance(node, ast.Name):
            return self._canon_name(node.id)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                fname = _GATE_FUNC_ALIASES.get(func.id, func.id)
                fname = self._canon_name(fname)
            else:
                fname = '?'
            args = [self._canonical(a) for a in node.args]
            if fname in ('+', '*') and len(args) == 2:
                args = sorted(args)
            return f'{fname}({",".join(args)})'
        if isinstance(node, ast.BinOp):
            opmap = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
                     ast.Div: '/', ast.Pow: '^'}
            sym = opmap.get(type(node.op), type(node.op).__name__)
            l, r = self._canonical(node.left), self._canonical(node.right)
            if sym in ('+', '*'):
                l, r = sorted([l, r])
            return f'({l}{sym}{r})'
        if isinstance(node, ast.Compare):
            parts = [self._canonical(node.left)]
            for op, comp in zip(node.ops, node.comparators):
                parts.append(type(op).__name__)
                parts.append(self._canonical(comp))
            return f'cmp({",".join(parts)})'
        if isinstance(node, ast.IfExp):
            return (f'if({self._canonical(node.body)},'
                    f'{self._canonical(node.test)},{self._canonical(node.orelse)})')
        return type(node).__name__

    @staticmethod
    def _count_constants(node) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return FactorOriginalityGate._count_constants(node.operand)
        return 0

    def _analyze(self, expr: str):
        tree = self._parse(expr)
        num_all = sum(1 for n in ast.walk(tree) if isinstance(n, _GATE_MEANINGFUL))
        num_const = sum(self._count_constants(n) for n in ast.walk(tree))
        # Collect all function-call names so we can exclude them from the
        # variable set (otherwise `rank(5)` would count `rank` as a "variable").
        call_names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                call_names.add(self._canon_name(n.func.id))
        varset = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                canon = self._canon_name(n.id)
                if canon not in call_names:
                    varset.add(canon)
        canon = self._canonical(tree)
        return canon, num_const, len(varset), num_all

    @staticmethod
    def _is_trivial(num_const: int, num_vars: int, num_all: int) -> Optional[str]:
        if num_vars < 1:
            return "no data dependency (constant factor)"
        if num_all < 2:
            return "single-token expression (too trivial)"
        if num_const > num_vars * 3:
            return f"constant-dominated (const={num_const}, vars={num_vars})"
        return None

    def validate(self, expr: str) -> Tuple[bool, str]:
        """
        Returns (accepted, reason). On accept, the expression is recorded in the
        zoo so subsequent near-duplicates are rejected.
        """
        if not self.enabled:
            return True, ""
        if not expr or not expr.strip():
            return False, "empty expression"
        # 1) syntax
        try:
            tree = self._parse(expr)
        except SyntaxError as e:
            return False, f"syntax error: {e}"
        except Exception as e:  # pragma: no cover - defensive
            return False, f"unparseable: {e}"
        # 2) semantics: unknown function
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                func = n.func
                if isinstance(func, ast.Name):
                    raw = func.id
                    fname = _GATE_FUNC_ALIASES.get(raw, raw)
                    fname = self._canon_name(fname)   # _if → if (post-preprocessing)
                    if fname not in _GATE_ALLOWED_FUNCS:
                        return False, f"unknown function '{raw}()'"
        # 3) originality: dedup + triviality
        try:
            canon, num_const, num_vars, num_all = self._analyze(expr)
        except Exception as e:  # pragma: no cover - defensive
            return False, f"analysis error: {e}"
        for zexpr, zcanon in zip(self.zoo_exprs, self.zoo):
            if zcanon == canon:
                return False, f"duplicate of existing factor: {zexpr[:60]}"
            sim = difflib.SequenceMatcher(None, canon, zcanon).ratio()
            if sim >= self.dedup_similarity:
                return False, f"near-duplicate (similarity={sim:.2f}) of: {zexpr[:60]}"
        trivial = self._is_trivial(num_const, num_vars, num_all)
        if trivial:
            return False, trivial
        # accept → record in zoo
        self.zoo.append(canon)
        self.zoo_exprs.append(expr)
        return True, "accepted"


# ---------------------------------------------------------------------------
# Factor expression evaluator (recursive-descent parser)
# ---------------------------------------------------------------------------

# Token types
_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?)'  # number
    r'|(?P<ident>[a-zA-Z_]\w*)'                   # identifier / function name
    r'|(?P<cmp>[<>!=]=|[<>!=])'                   # comparison operators: >=, <=, ==, !=, >, <, !
    r'|(?P<pow>\*\*)'                             # power operator (**) — must precede `op`
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
        elif kind == 'pow':
            tokens.append(('POWER', '**'))
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
        pe, pb, ps, roe, market_cap             (fundamental_data dict keys)
        eps, return, returns, vwap  (derived fields)
        — eps = close / pe
        — return / returns = 1-day daily return (close.pct_change(1))
        — vwap = amount / volume
        — NOTE: forward_returns is intentionally NOT an expression field (look-ahead bias)

    Supported functions:
        rank(X)          — cross-sectional percentile rank [0, 1]
        ts_rank(X, w)    — rolling time-series rank [0, 1]
        ts_corr(X, Y, w) — rolling Pearson correlation
        ts_cov(X, Y, w)  — rolling covariance
        ts_mean(X, w)    — rolling mean
        ts_std(X, w)     — rolling std (ddof=0)  [alias: ts_stddev]
        ts_var(X, w)     — rolling variance (ddof=0)
        ts_skew(X, w)    — rolling skewness
        ts_kurt(X, w)    — rolling kurtosis
        ts_min(X, w)     — rolling minimum
        ts_max(X, w)     — rolling maximum
        ts_sum(X, w)     — rolling sum
        ts_delta(X, w)   — X - delay(X, w)
        ts_pct_change(X, w) — (X - delay(X, w)) / delay(X, w)  [alias: ts_pctchange, pct_change]
        ts_zscore(X, w)  — rolling z-score
        ts_decay(X, w)   — exponential weighted moving average (EWMA)
        delay(X, d)      — lag by d periods
        sign(X)          — element-wise sign
        abs(X)           — element-wise absolute value
        log(X)           — element-wise natural log
        sqrt(X)          — element-wise square root (clamped >= 0)

    Operators: +, -, *, /, ^ and ** (power)
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
        """term := power (('*' | '/') power)*"""
        left = self._parse_power()
        while self._peek()[0] == 'OP' and self._peek()[1] in ('*', '/'):
            op = self._advance()[1]
            right = self._parse_power()
            if op == '*':
                left = left * right
            else:
                left = left / right.replace(0, np.nan)
        return left

    def _parse_power(self) -> pd.DataFrame:
        """power := unary (('^' | '**') unary)*   (left-associative exponentiation)

        Both `^` and `**` are the power operator (documented in the class
        docstring). It binds tighter than `*` / `/`, so `a * b ^ 2` parses as
        `a * (b ^ 2)` — which is what LLM-generated expressions like
        `sqrt(1 - ts_corr(x, y, w)^2)` need. Previously the `^` token was
        recognized by the tokenizer but never consumed by any parse rule, so it
        leaked up and raised "Expected PAREN()), got ('OP', '^')" at the
        enclosing `)`. `**` was likewise mis-lexed as two `*` operators.
        """
        left = self._parse_unary()
        while (self._peek()[0] == 'OP' and self._peek()[1] == '^') or \
              (self._peek()[0] == 'POWER' and self._peek()[1] == '**'):
            self._advance()
            right = self._parse_unary()
            left = left ** right
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
        """Look up a named data field. Case-insensitive: LLMs often emit
        uppercase (ROE/PE) while the data map uses lowercase keys."""
        if name not in self._data:
            name_lower = name.lower()
            if name_lower in self._data:
                return self._data[name_lower]
            raise ValueError(
                f"Unknown identifier '{name}'. "
                f"Available: {list(self._data.keys())}"
            )
        return self._data[name]

    # ---- Function dispatch ----

    # Common WorldQuant/quant DSL aliases → canonical name
    # This lets the evaluator accept function names the LLM frequently generates
    # even when they differ from the prompt's canonical spelling.
    _FUNC_ALIASES = {
        'ts_stddev':  'ts_std',      # WorldQuant Alpha 101 standard name
        'ts_std_dev': 'ts_std',
        'stddev':     'ts_std',
        'ts_average': 'ts_mean',      # common alias
        'ts_avg':     'ts_mean',
        'ts_median':  'ts_mean',      # approximated by mean (rare alias)
        'ts_diff':    'ts_delta',     # NumPy naming convention
        'ts_delay':   'delay',
        'ts_lag':     'delay',
        'lag':        'delay',
        'cov':        'ts_cov',
        'skew':       'ts_skew',
        'kurt':       'ts_kurt',
        'kurtosis':   'ts_kurt',
        'max':        'ts_max',       # bare max → ts_max (ambiguous but LLM uses it)
        'min':        'ts_min',
        'mean':       'ts_mean',
        'sum':        'ts_sum',
        # percentage-change family
        'ts_pctchange': 'ts_pct_change',   # AlphaForge canonical spelling
        'pct_change':   'ts_pct_change',   # pandas naming
        'ts_pctchg':    'ts_pct_change',
        # longer "statistics" spellings the LLM sometimes emits
        'ts_skewness':  'ts_skew',
        'ts_kurtosis':  'ts_kurt',
        'ts_roc':       'ts_pct_change',   # rate of change = pct change
    }

    def _dispatch_func(self, name: str, args: List) -> pd.DataFrame:
        """Route function call to implementation."""
        # Normalize aliases before lookup
        name = self._FUNC_ALIASES.get(name, name)

        arity = len(args)
        # Check arity
        arity_map = {
            'rank':       (1, 1),
            'ts_rank':    (2, 2),
            'ts_corr':    (3, 3),
            'ts_cov':     (3, 3),
            'ts_mean':    (2, 2),
            'ts_std':     (2, 2),
            'ts_var':     (2, 2),
            'ts_skew':    (2, 2),
            'ts_kurt':    (2, 2),
            'ts_min':     (2, 2),
            'ts_max':     (2, 2),
            'ts_sum':     (2, 2),
            'ts_delta':   (2, 2),
            'ts_pct_change': (2, 2),
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
        min_p = min(max(2, w // 2), w)  # never exceed w
        return x.rolling(window=w, min_periods=min_p).rank(pct=True) / w

    @staticmethod
    def _fn_ts_corr(x: pd.DataFrame, y: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling Pearson correlation between x and y (per stock)."""
        w = _FactorExprEvaluator._safe_int(window)
        # min_periods must be <= window; require at least 3 points but never exceed w
        min_p = min(max(3, w // 2), w)
        return x.rolling(window=w, min_periods=min_p).corr(y)

    @staticmethod
    def _fn_ts_cov(x: pd.DataFrame, y: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling covariance between x and y (per stock)."""
        w = _FactorExprEvaluator._safe_int(window)
        min_p = min(max(3, w // 2), w)
        return x.rolling(window=w, min_periods=min_p).cov(y)

    @staticmethod
    def _fn_ts_mean(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        return x.rolling(window=w, min_periods=max(1, w // 2)).mean()

    @staticmethod
    def _fn_ts_std(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        # min_periods must be <= window; never exceed w
        min_p = min(max(2, w * 2 // 3), w)
        return x.rolling(window=w, min_periods=min_p).std(ddof=0)

    @staticmethod
    def _fn_ts_var(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling variance (ddof=0)."""
        w = _FactorExprEvaluator._safe_int(window)
        min_p = min(max(2, w * 2 // 3), w)
        return x.rolling(window=w, min_periods=min_p).var(ddof=0)

    @staticmethod
    def _fn_ts_skew(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling skewness."""
        w = _FactorExprEvaluator._safe_int(window)
        min_p = min(max(3, w * 2 // 3), w)
        return x.rolling(window=w, min_periods=min_p).skew()

    @staticmethod
    def _fn_ts_kurt(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling kurtosis (Fisher's definition, normal → 0)."""
        w = _FactorExprEvaluator._safe_int(window)
        min_p = min(max(4, w * 2 // 3), w)
        return x.rolling(window=w, min_periods=min_p).kurt()

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
    def _fn_ts_pct_change(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling percentage change over window w: (X - X.shift(w)) / X.shift(w).
        Mirrors AlphaForge's ts_pctchange. Zero/NaN denominators → NaN (meaningful)."""
        w = _FactorExprEvaluator._safe_int(window)
        prev = x.shift(w)
        prev = prev.replace(0, np.nan)   # avoid div-by-zero noise; 0 → NaN is informative
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = (x - prev) / prev
        return pct

    @staticmethod
    def _fn_ts_zscore(x: pd.DataFrame, window) -> pd.DataFrame:
        w = _FactorExprEvaluator._safe_int(window)
        mean = x.rolling(window=w, min_periods=max(5, w * 2 // 3)).mean()
        std = x.rolling(window=w, min_periods=max(5, w * 2 // 3)).std(ddof=0)
        # When std == 0 (e.g. ROE updates quarterly, values constant in window),
        # all values equal the mean, so z-score should be 0, not NaN.
        z = (x - mean) / std
        z[std == 0] = 0.0
        return z

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
                                  forward_period=10)
                 → price_df = close prices (dates × stocks)
                 → volume_df = volume (dates × stocks)
                 → fundamentals_dict = {'pe': df, 'pb': df, ...}
                 → forward_period = forward return horizon in trading days

    If a dict is passed as the first argument, it is treated as price_data
    with keys: open, high, low, close, volume, amount.
    """

    _DEFAULT_FORWARD_PERIOD = 10

    def __init__(
        self,
        prices,
        volume: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
        forward_period: int = _DEFAULT_FORWARD_PERIOD,
    ):
        """
        Initialize the backtester.

        Args:
            prices: If dict → price_data dict (keys: open/high/low/close/volume/amount).
                    If DataFrame → close prices only (legacy API, volume=None required).
            volume: Volume DataFrame, used only when prices is a DataFrame (legacy API).
            fundamentals: Optional dict of fundamental DataFrames (pe, pb, roe, market_cap).
            forward_period: Forward return horizon in trading days.
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

        # ---- Derived fields ----
        if 'close' in self._data_map and 'pe' in self._data_map:
            pe_safe = self._data_map['pe'].replace(0, np.nan)
            self._data_map['eps'] = self._data_map['close'] / pe_safe

        # Daily returns (1-day pct_change) — 'return' and 'returns' are aliases
        if 'close' in self._data_map:
            _ret = self._data_map['close'].pct_change(1)
            self._data_map['return'] = _ret
            self._data_map['returns'] = _ret

        # VWAP (volume-weighted average price) = amount / volume
        if 'amount' in self._data_map and 'volume' in self._data_map:
            vol_safe = self._data_map['volume'].replace(0, np.nan)
            self._data_map['vwap'] = self._data_map['amount'] / vol_safe

        # ---- Compute forward returns (used ONLY as the IC / portfolio target) ----
        # NOTE: forward_returns is deliberately NOT exposed to factor expressions.
        # It holds the N-day FUTURE return; letting a factor reference it would be
        # look-ahead bias (the factor would use future prices to predict the future).
        # It remains available via self.forward_returns for IC scoring (see evaluate()).
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

        Args:
            factor: CandidateFactor with .expression set.

        Returns:
            Dict with keys: ic, ic_ir, sharpe, win_rate, max_drawdown,
                            long_short_ret, factor_values.
        """
        expr = factor.expression

        # Check cache
        if expr in self._metric_cache:
            cached = self._metric_cache[expr].copy()
            # Update the factor object
            factor.ic = cached['ic']
            factor.icir = cached.get('icir', 0.0)
            factor.sharpe = cached['sharpe']
            factor.win_rate = cached['win_rate']
            factor.max_drawdown = cached.get('max_drawdown', 0.0)
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
                'icir': float(ic_ir) if not np.isnan(ic_ir) else 0.0,
                'sharpe': float(sharpe) if not np.isnan(sharpe) else 0.0,
                'win_rate': float(win_rate) if not np.isnan(win_rate) else 0.5,
                'max_drawdown': float(max_dd) if not np.isnan(max_dd) else 0.0,
                'long_short_ret': long_short_ret,
                'factor_values': factor_values,
            }

            # Update the factor object
            factor.ic = metrics['ic']
            factor.icir = metrics['icir']
            factor.sharpe = metrics['sharpe']
            factor.win_rate = metrics['win_rate']
            factor.max_drawdown = metrics['max_drawdown']

            # Cache
            self._metric_cache[expr] = {k: v for k, v in metrics.items()
                                         if k not in ('long_short_ret', 'factor_values')}

            return metrics

        except Exception as e:
            import traceback
            err_msg = str(e)
            logger.warning("Factor evaluation failed for '%s': %s", expr, err_msg)
            traceback.print_exc()
            # Mark the factor as invalid so the evolution loop can exclude it
            factor.is_valid = False
            factor.ic = float('nan')
            factor.icir = float('nan')
            factor.parse_error = err_msg
            return self._empty_metrics(error=err_msg)

    def _empty_metrics(self, error: str = "") -> Dict:
        """Return a safe empty-result dict for failed / invalid factors.

        ic / icir are set to NaN (not 0.0) so that downstream sort / filter
        logic can correctly identify parse-failed factors rather than treating
        them as factors with a neutral IC of 0.
        """
        return {
            'ic': float('nan'),
            'icir': float('nan'),
            'sharpe': 0.0,
            'win_rate': 0.5,
            'max_drawdown': 0.0,
            'long_short_ret': pd.Series(dtype=float),
            'factor_values': pd.DataFrame(),
            'is_valid': False,
            'parse_error': error,
        }

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
                    factor.icir = cached.get('icir', 0.0)
                    factor.sharpe = cached['sharpe']
                    factor.win_rate = cached['win_rate']
                    factor.max_drawdown = cached.get('max_drawdown', 0.0)
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
        # Annualize using the ACTUAL holding period, not daily. ls_series
        # entries are forward_period-day returns (sampled daily), so the
        # annualization factor must be sqrt(252 / forward_period). Using the
        # daily factor sqrt(252) overstated Sharpe by ~sqrt(forward_period).
        periods_per_year = 252.0 / self.forward_period
        sharpe = annualized_sharpe(ls_series, periods_per_year=periods_per_year)
        win_rate = float((ls_series > 0).mean())
        # Use simple (non-compounding) drawdown — ls_series contains
        # forward_period-day long-short returns sampled daily. Compounding
        # these overlapping observations via cumprod falsely amplifies
        # drawdown by ~√forward_period (same over-counting bug that the
        # Sharpe annualization fix addressed).
        max_dd = max_drawdown(ls_series, method='simple')

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
        n_seeds_hypothesis: int = 0,
        n_seeds_single_stage: int = 20,
        n_seeds_memory_augment: int = 0,
        n_best_factors: int = 10,
        n_improve: int = 10,
        n_mutate: int = 5,
        convergence_delta: float = 0.003,
        convergence_window: int = 2,
        patience: int = 3,
        min_ic: float = 0.02,
        min_sharpe: float = 0.0,
        max_drawdown: float = -1.0,
        min_val_ic: float = 0.01,
        originality_gate: bool = True,
        dedup_similarity: float = 0.90,
        improve_temperature: float = 0.3,
        elitism_carry: int = 2,
        parallel: bool = True,
        api_key: str = "",
        base_url: str = "http://180.163.156.38:53000/v1",
    ):
        """
        Initialize self-evolving generator.
        
        Args:
            llm_model: LLM model to use
            n_seeds_hypothesis: Target number of **hypothesis-driven** seed factors
                (Stage1→Stage2: LLM proposes market hypotheses, then factors
                expressing them). Produced whenever ``n_seeds_hypothesis > 0``
                (no separate master flag needed). 0 = none.
            n_seeds_single_stage: Target number of **plain single-stage** seed
                factors (the legacy 撒网式 coverage prompt, no hypothesis). 0 = none.
            n_seeds_memory_augment: Target number of **memory-augmented** seed
                factors. Consumed by ``MemoryAugmentedGenerator`` (few-shot
                examples from the memory bank). 0 = memory augmentation off.
                Splitting the old single ``n_seeds`` into these three lets you
                control each seed *source* independently.
            n_best_factors: Number of top factors to select per round and final output
            n_improve: Target number of improved factors to generate per round
            n_mutate: Number of mutation variations per top factor (rule-based fallback)
            convergence_delta: Min IC improvement to continue evolution
            convergence_window: Number of recent rounds to check for convergence
            patience: Max consecutive non-improving rounds before early stop
            min_ic: Minimum IC threshold for quality filtering
            min_sharpe: Soft floor on Sharpe ratio (default 0.0 = drop only
                clearly negative Sharpe). NOTE: f.sharpe is distorted by
                overlapping-window annualization, so keep this a soft guard.
            max_drawdown: Maximum drawdown threshold (negative, e.g. -1.0)
            min_val_ic: Minimum validation-IC threshold (val-mode only). The honest
                overfit gate: drop factors with a negative signal on the holdout split.
            originality_gate: Enable the AST-based originality gate (syntax check +
                function whitelist + structural dedup + trivial detection).
            dedup_similarity: Canonical-signature similarity threshold for
                near-duplicate rejection (0.0 = off, 1.0 = only exact).
            improve_temperature: LLM sampling temperature during factor IMPROVEMENT
                (NOT seed generation). Low (0.3) = focused refinement of input;
                high (>0.7) = near-random re-sampling that collapses diversity.
            elitism_carry: Number of top factors carried forward unmutated from the
                previous round. Prevents full-replacement collapse where the
                entire pool re-converges to a single direction each round.
            parallel: If False, evaluate factors serially (easier to debug)
            originality_gate: Enable the static AST originality gate (structural
                dedup + triviality/semantics checks) before factors enter the pool.
            dedup_similarity: Canonical-signature similarity (0-1) above which a
                factor is treated as a near-duplicate of an accepted one.
            parallel: If False, evaluate factors serially (easier to debug)
            api_key: API key for LLM service. If empty, reads from config or env.
            base_url: API base URL
        """
        self.llm_model = llm_model
        self.n_seeds_hypothesis = n_seeds_hypothesis
        self.n_seeds_single_stage = n_seeds_single_stage
        self.n_seeds_memory_augment = n_seeds_memory_augment
        self.n_best_factors = n_best_factors
        self.n_improve = n_improve
        self.n_mutate = n_mutate
        self.convergence_delta = convergence_delta
        self.convergence_window = convergence_window
        self.patience = patience
        self.min_ic = min_ic
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_val_ic = min_val_ic
        self.improve_temperature = improve_temperature
        self.elitism_carry = elitism_carry
        self.parallel = parallel

        # Static AST originality gate (structural dedup + triviality + semantics)
        self.gate = FactorOriginalityGate(
            enabled=originality_gate,
            dedup_similarity=dedup_similarity,
        )
        
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
        save_dir = os.path.join(date_str, subdir, f"round_{round_id}")
        save_dir = config_path("experiments", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Filter out parse-failed (is_valid=False) factors before saving
        valid_factors = [f for f in factors if getattr(f, 'is_valid', True)]
        skipped = len(factors) - len(valid_factors)
        if skipped:
            logger.info("[evolve] Skipping %d invalid factor(s) when saving to %s", skipped, filename)

        def _sanitize(d):
            """Replace nan/inf float values with None so json.dump produces valid JSON."""
            import math
            for k, v in d.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    d[k] = None
            return d

        factors_dicts = [_sanitize(asdict(f)) for f in valid_factors]
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(factors_dicts, f, indent=2, ensure_ascii=False)

        print(f"  [evolve] Saved {len(valid_factors)} factors to {save_path}"
              + (f" ({skipped} invalid skipped)" if skipped else ""))

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

    def _llm_generate_hypotheses(self, n_hypotheses: int = 4) -> List[str]:
        """
        Stage 1 of hypothesis-driven factor generation.

        Ask the LLM to propose N distinct, testable market hypotheses for A-share
        stock selection. Each hypothesis should cover a different angle (momentum,
        value, quality, liquidity, growth, behavioural, etc.).

        Returns a list of hypothesis strings; empty list on failure.
        """
        system_prompt = (
            "You are a quantitative finance researcher generating market hypotheses "
            "for alpha factor mining in the Chinese A-share market.\n\n"
            "Each hypothesis should:\n"
            "1. Be grounded in financial theory or observed market behaviour\n"
            "2. Be specific enough to guide mathematical factor construction\n"
            "3. Cover a different angle from the others (e.g. momentum vs value vs quality)\n"
            "4. Mention concrete data fields (price, volume, fundamentals) when possible\n\n"
            "Return ONLY a JSON array of strings. No explanations, no markdown, no code fences."
        )

        user_prompt = (
            f"Generate {n_hypotheses} distinct, testable market hypotheses for A-share "
            f"stock selection. Each should be a complete sentence (2-4 sentences). "
            f"Cover different investment angles. Return as JSON array of strings. "
            f"Example format: [\"Hypothesis 1 text...\", \"Hypothesis 2 text...\", ...]"
        )

        try:
            raw = self._call_llm(system_prompt, user_prompt,
                                 temperature=0.6, expect_json=True)
            parsed = json.loads(raw)
            # Robustness: LLMs sometimes return {"hypotheses": [...]} or even
            # {"factors": ["H1...", "H2..."]} (confusing Stage1 with Stage2 output format).
            if isinstance(parsed, dict):
                candidates = parsed.get("hypotheses", None)
                if candidates is None:
                    candidates = parsed.get("factors", None)
                if isinstance(candidates, list):
                    parsed = candidates
            if isinstance(parsed, list) and all(isinstance(h, str) for h in parsed):
                hypotheses = [h.strip() for h in parsed if h.strip()]
                if hypotheses:
                    print(f"  [hypothesis] Generated {len(hypotheses)} market hypotheses:")
                    for i, h in enumerate(hypotheses):
                        print(f"    {i+1}. {h[:120]}{'...' if len(h) > 120 else ''}")
                    return hypotheses
            print(f"  [hypothesis] Unexpected LLM response format: {type(parsed).__name__}")
            return []
        except Exception as e:
            print(f"  [hypothesis] LLM hypothesis generation failed: {e}")
            return []

    def generate_seed_factors(self) -> List['CandidateFactor']:
        """
        Generate seed factors from the three explicit sources.

        The three counts set in ``__init__`` are independent and control each
        seed *source* directly (no implicit ratio / cross-fill):

        * ``n_seeds_hypothesis``      → Stage1→Stage2 hypothesis-driven factors.
          Produced whenever ``n_seeds_hypothesis > 0``.
        * ``n_seeds_single_stage``   → plain single-stage (撒网式) factors.
        * ``n_seeds_memory_augment``  → memory-augmented factors. These are NOT
          generated here; ``main.py`` feeds this count to
          ``MemoryAugmentedGenerator`` and concatenates the result. (We only
          report the target here for logging.)

        Because the counts are explicit, if the hypothesis pipeline under-
        delivers we do NOT silently re-fill those slots from single-stage — the
        produced numbers reflect exactly what was requested. Each subset still
        best-effort tops up its OWN count if the LLM under-delivers a single call.
        Falls back to rule-based generation (filling the requested total) only
        when the LLM is entirely unavailable.
        """
        n_hyp_target = self.n_seeds_hypothesis
        n_plain_target = self.n_seeds_single_stage
        n_mem_target = self.n_seeds_memory_augment
        print(f"Generating seed factors — hypothesis: {n_hyp_target}, "
              f"single-stage: {n_plain_target}, memory-augment(target): {n_mem_target}")

        if not (self.use_llm and self.client):
            # LLM unavailable → rule-based fills the local (non-memory) total.
            total = n_hyp_target + n_plain_target
            print(f"  [evolve] LLM unavailable -> rule-based generation of {total}.")
            return self._generate_factors_rule_based(total)

        # ── Hypothesis-driven subset ──
        hyp_factors: List[CandidateFactor] = []
        if n_hyp_target > 0:
            hyp_factors = self._generate_hypothesis_factors(n_hyp_target)
            print(f"  Generated {len(hyp_factors)}/{n_hyp_target} "
                  f"hypothesis-driven seed factors")
            if not hyp_factors:
                print("  [hypothesis] pipeline yielded nothing (those seeds dropped "
                      "— not backfilled from single-stage to keep counts explicit)")

        # ── Single-stage subset ──
        plain_factors: List[CandidateFactor] = []
        if n_plain_target > 0:
            plain_factors = self._generate_single_stage_factors(n_plain_target)
            print(f"  Generated {len(plain_factors)}/{n_plain_target} "
                  f"single-stage seed factors")

        # Hypothesis first, then single-stage. No trimming to a shared total —
        # the two counts are independent by design.
        result = hyp_factors + plain_factors
        print(f"Generated {len(result)} seed factors locally "
              f"({len(hyp_factors)} hypothesis-driven, {len(plain_factors)} single-stage). "
              f"Memory-augmented seeds (target {n_mem_target}) are added by the caller.")
        return result

    def _generate_hypothesis_factors(self, n_hyp: int) -> List['CandidateFactor']:
        """
        Stage1→Stage2 hypothesis-driven generation, targeting ``n_hyp`` factors.

        Stage 1: LLM proposes 3-5 distinct market hypotheses.
        Stage 2: LLM generates factors expressing each hypothesis (thesis-driven
        construction, replacing the old category-coverage prompt). Each factor is
        tagged ``[H*]`` in its description for traceability.
        """
        if n_hyp <= 0:
            return []
        hypotheses = self._llm_generate_hypotheses(
            n_hypotheses=min(5, max(3, n_hyp // 3)),
        )
        if not hypotheses:
            return []
        factors_per_hypothesis = max(1, n_hyp // len(hypotheses))
        all_factors: List[CandidateFactor] = []
        for i, h in enumerate(hypotheses):
            try:
                batch = self._generate_factors_via_llm(
                    factors_per_hypothesis, hypothesis=h,
                )
                if batch:
                    for f in batch:
                        if not f.description.startswith(f"[H{i+1}]"):
                            f.description = f"[H{i+1}] {f.description}"
                    all_factors.extend(batch)
            except Exception as e:
                print(f"  [hypothesis] Factor generation for H{i+1} failed: {e}")
        return all_factors[:n_hyp]

    def _generate_single_stage_factors(self, n_plain: int) -> List[CandidateFactor]:
        """
        Plain single-stage (撒网式) generation, targeting ``n_plain`` factors.

        Best-effort: makes a primary LLM call for ``n_plain`` factors, then a
        single top-up call if the first under-delivered. Returns whatever was
        produced (may be fewer than ``n_plain`` only if the LLM keeps failing).
        """
        if n_plain <= 0:
            return []
        try:
            seed_factors = self._generate_factors_via_llm(n_plain)
            if seed_factors and len(seed_factors) >= 1:
                return seed_factors[:n_plain]
        except Exception as e:
            print(f"  [evolve] single-stage generation failed: {e}")
        return []

    def _generate_factors_via_llm(self, n_factors: int, hypothesis: Optional[str] = None) -> List[CandidateFactor]:
        """
        Generate factors using LLM.

        When *hypothesis* is provided (Stage 2 of hypothesis-driven generation),
        the prompt directs the LLM to construct factors that EXPRESS that
        specific market hypothesis, rather than just covering generic categories.

        Args:
            n_factors: Number of factors to generate
            hypothesis: Optional market hypothesis to guide factor construction

        Returns:
            List of candidate factors
        """
        system_prompt = """You are a quantitative factor research expert specializing in A-share stock selection.
Your task is to generate diverse, economically meaningful factor expressions for stock selection.

Supported data sources:
- open, high, low, close, volume, amount (price and volume data)
- pe, pb, ps, roe, market_cap, eps (fundamental data; eps = close / pe)
- return / returns (1-day daily return; alias: close.pct_change(1))
- vwap (volume-weighted average price = amount / volume)

Supported functions (WorldQuant style):
- rank(X): cross-sectional percentile rank [0, 1]
- ts_rank(X, w): rolling time-series rank within window w
- ts_corr(X, Y, w): rolling correlation between X and Y over window w
- ts_cov(X, Y, w): rolling covariance between X and Y over window w
- ts_mean(X, w): rolling mean over window w
- ts_std(X, w): rolling standard deviation over window w (alias: ts_stddev)
- ts_var(X, w): rolling variance over window w
- ts_skew(X, w): rolling skewness over window w
- ts_kurt(X, w): rolling kurtosis over window w
- ts_min(X, w): rolling minimum over window w
- ts_max(X, w): rolling maximum over window w
- ts_sum(X, w): rolling sum over window w
- ts_delta(X, w): X - delay(X, w) (alias: ts_diff)
- ts_pct_change(X, w): (X - delay(X, w)) / delay(X, w) (alias: ts_pctchange, pct_change)
- ts_zscore(X, w): rolling z-score over window w
- delay(X, d): lag by d periods (alias: ts_lag, lag)
- sign(X): element-wise sign
- abs(X): element-wise absolute value
- log(X): element-wise natural log
- sqrt(X): element-wise square root

Note: ts_stddev, ts_average/ts_avg, ts_diff, ts_lag/lag, cov, skew, kurt/kurtosis, ts_pctchange/pct_change are accepted as aliases.
Supported operators: +, -, *, /, ^"""

        # Build JSON example once (shared by both branches)
        json_example = json.dumps({
            "factors": [
                {"expression": "rank(ts_corr(close, volume, 20))", "description": "Price-volume correlation momentum"},
                {"expression": "-rank(pe)", "description": "Value factor based on P/E ratio"},
                {"expression": "rank(ts_zscore(roe, 60))", "description": "Quality factor based on ROE z-score"}
            ]
        }, ensure_ascii=False)

        if hypothesis:
            user_prompt = (
                f"Generate {n_factors} factor expressions that capture the following market hypothesis:\n\n"
                f"  \"{hypothesis}\"\n\n"
                + "Requirements:\n"
                + "1. Each factor should directly or indirectly express the hypothesis above\n"
                + "2. Expressions must be valid and use only supported functions and data sources\n"
                + "3. Avoid trivial factors (e.g., just \"close\" or \"volume\")\n"
                + "4. Each factor should have economic intuition tied to the hypothesis\n"
                + "5. You MUST return exactly " + str(n_factors) + " factors\n"
                + "6. Return a JSON object with key \"factors\" (array of factor objects), no other text, no markdown fences\n"
                + "\n"
                + "Example (valid JSON object with \"factors\" key):\n"
                + json_example + "\n\n"
                + f"Please generate {n_factors} factors now. Return only the JSON object."
            )
        else:
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

    def generate(
        self,
        prompt: str,
        n_factors: int = None,
    ) -> list[dict]:
        """
        Generate factors using a custom prompt (called by MemoryAugmentedGenerator).

        Parameters
        ----------
        prompt : str, augmented prompt with few-shot examples from memory bank
        n_factors : int, number of factors to generate
            (default: self.n_seeds_memory_augment)

        Returns
        -------
        list[dict], each with "expression" and "description" keys
        """
        n = n_factors if n_factors is not None else self.n_seeds_memory_augment

        if self.use_llm and self.client:
            try:
                return self._generate_via_llm_with_prompt(prompt, n)
            except Exception as e:
                print(f"  [SelfEvolvingGenerator.generate] LLM failed: {e}, falling back to rule-based")
                return self._generate_fallback_dict(n)

        # LLM not available — rule-based fallback
        return self._generate_fallback_dict(n)

    def _generate_via_llm_with_prompt(self, user_prompt: str, n_factors: int) -> list[dict]:
        """Call LLM with a custom user_prompt (from MemoryAugmentedGenerator)."""
        system_prompt = """You are a quantitative factor research expert specializing in A-share stock selection.
Your task is to generate diverse, economically meaningful factor expressions for stock selection.

Supported data sources:
- open, high, low, close, volume, amount (price and volume data)
- pe, pb, ps, roe, market_cap, eps (fundamental data; eps = close / pe)
- return / returns (1-day daily return; alias: close.pct_change(1))
- vwap (volume-weighted average price = amount / volume)

Supported functions (WorldQuant style):
- rank(X): cross-sectional percentile rank [0, 1]
- ts_rank(X, w): rolling time-series rank within window w
- ts_corr(X, Y, w): rolling correlation between X and Y over window w
- ts_mean(X, w): rolling mean over window w (alias: ts_average, ts_avg)
- ts_std(X, w): rolling standard deviation over window w (alias: ts_stddev)
- ts_min(X, w): rolling minimum over window w
- ts_max(X, w): rolling maximum over window w
- ts_sum(X, w): rolling sum over window w
- ts_delta(X, w): X - delay(X, w) (alias: ts_diff)
- ts_pct_change(X, w): (X - delay(X, w)) / delay(X, w) (alias: ts_pctchange, pct_change)
- ts_zscore(X, w): rolling z-score over window w
- delay(X, d): lag by d periods (alias: ts_lag, lag)
- sign(X): element-wise sign
- abs(X): element-wise absolute value
- log(X): element-wise natural log
- sqrt(X): element-wise square root

Note: ts_stddev, ts_average/ts_avg, ts_diff, ts_lag/lag, ts_pctchange/pct_change are accepted as aliases.

Supported operators: +, -, *, /, ^

Return a JSON object with a "factors" key, which is an array of objects.
Each object must have "expression" and "description" keys.
Ensure expressions are valid and can be evaluated by the factor engine."""

        resp = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"Generate {n_factors} factors. {user_prompt}",
            temperature=0.8,
            expect_json=True,
        )

        import json
        try:
            parsed = json.loads(resp)
            raw_factors = parsed.get("factors", parsed) if isinstance(parsed, dict) else parsed
            results = []
            for f in raw_factors[:n_factors]:
                if isinstance(f, dict) and "expression" in f:
                    results.append({
                        "expression": f["expression"],
                        "description": f.get("description", ""),
                    })
            print(f"  [SelfEvolvingGenerator] LLM generated {len(results)} factors from augmented prompt")
            return results
        except Exception as e:
            print(f"  [SelfEvolvingGenerator] Failed to parse LLM response: {e}")
            return []

    def _generate_fallback_dict(self, n_factors: int) -> list[dict]:
        """Rule-based fallback returning list[dict] format."""
        factors = self._generate_factors_rule_based(n_factors)
        return [{"expression": f.expression, "description": f.description} for f in factors]

    def _apply_originality_gate(self, factor: 'CandidateFactor') -> None:
        """
        Run the static AST originality gate on an already-backtested factor.

        Only runs when the factor passed parse + backtest (factor.is_valid). On
        rejection we set factor.originality_ok=False and record the reason in
        factor.gate_reason. We deliberately leave factor.is_valid=True so the
        existing selection fallback can still recover a survivor set if the gate
        (over-)rejects the entire pool.
        """
        gate = getattr(self, 'gate', None)
        if gate is None or not gate.enabled:
            return
        ok, reason = gate.validate(factor.expression)
        if not ok:
            factor.originality_ok = False
            factor.gate_reason = reason
            logger.info(
                "Originality gate rejected '%s': %s",
                factor.expression, reason,
            )

    def evolve(
        self,
        seed_factors: List[CandidateFactor],
        backtester: FactorBacktester,
        n_rounds: int = 10,
        val_backtester: Optional['FactorBacktester'] = None,
    ) -> EvolutionResult:
        """
        Evolve factors through iterative improvement.

        Args:
            seed_factors: Seed factors to evolve
            backtester: Backtester for evaluation (TRAIN period)
            n_rounds: Number of evolution rounds
            val_backtester: Optional backtester on a held-out VALIDATION period.
                When supplied, evolution is *driven* by train IC (so the LLM
                keeps exploiting the training signal) but factors are finally
                *ranked* and *early-stopped* on validation IC — the standard
                anti-overfitting recipe. When None, behaviour is unchanged
                (selection/early-stop use train IC).

        Returns:
            Evolution result
        """
        val_mode = val_backtester is not None

        def _ic_key(f: CandidateFactor) -> float:
            """Sort key: validation IC when available, else train IC."""
            if val_mode and f.val_ic is not None and not np.isnan(f.val_ic):
                return f.val_ic
            return f.ic if f.ic is not None and not np.isnan(f.ic) else float('-inf')

        print(f"\nStarting evolution ({n_rounds} rounds)..."
              f"{' [train-driven, val-selected]' if val_mode else ''}")

        evolution_history = []
        current_factors = seed_factors

        best_ic = 0.0
        best_val_ic = 0.0
        all_evaluated_factors: List[CandidateFactor] = []  # accumulates IC-evaluated factors from every round
        no_improvement_count = 0  # for patience-based early stop
        
        for round_id in range(n_rounds):
            print(f"\nRound {round_id + 1}/{n_rounds}")
            
            # Evaluate current factors
            mode = "parallel" if self.parallel else "serial"
            print(f"  Evaluating {len(current_factors)} factors ({mode})...")
            metrics_list = backtester.evaluate_batch(current_factors, max_workers=4, parallel=self.parallel)

            # Validation evaluation. We evaluate on *independent probe* factors
            # (same expression, fresh objects) so the validation pass cannot
            # clobber the train IC/Sharpe already written onto the originals by
            # the train backtester. We only read the returned metrics dict.
            val_metrics_list = None
            if val_mode:
                val_probes = [CandidateFactor(id=f.id, expression=f.expression,
                                          description=f.description or "")
                              for f in current_factors]
                val_metrics_list = val_backtester.evaluate_batch(
                    val_probes, max_workers=4, parallel=self.parallel)

            evaluated_factors = []
            for i, factor in enumerate(current_factors):
                metrics = metrics_list[i]
                factor.ic = metrics.get('ic', float('nan'))
                factor.icir = metrics.get('icir', float('nan'))
                factor.sharpe = metrics.get('sharpe', 0.0)
                factor.win_rate = metrics.get('win_rate', 0.5)
                factor.max_drawdown = metrics.get('max_drawdown', 0.0)
                # Sync is_valid / parse_error from metrics dict
                factor.is_valid = metrics.get('is_valid', True)
                if not factor.is_valid:
                    factor.parse_error = metrics.get('parse_error', '')
                    logger.info(
                        "Excluding invalid factor '%s' from candidate set: %s",
                        factor.expression, factor.parse_error,
                    )
                else:
                    # Originality gate: structural dedup + triviality + semantics.
                    # Runs only on parse/backtest-valid factors; on rejection it
                    # sets originality_ok=False (is_valid stays True for fallback).
                    self._apply_originality_gate(factor)
                # Attach validation-period IC (read from the probe's returned metrics)
                if val_mode and val_metrics_list is not None:
                    vmetrics = val_metrics_list[i]
                    if vmetrics.get('is_valid', True):
                        factor.val_ic = vmetrics.get('ic', float('nan'))
                        factor.val_icir = vmetrics.get('icir', float('nan'))
                    else:
                        factor.val_ic = float('nan')
                evaluated_factors.append(factor)

                # Only update best_ic with valid factors that have a real IC value
                if factor.is_valid and not np.isnan(factor.ic) and factor.ic > best_ic:
                    best_ic = factor.ic

            self._save_factors_to_file(current_factors, round_id, filename="improved_factors.json")

            # Select top factors — exclude parse-failed AND gate-rejected factors
            valid_evaluated = [f for f in evaluated_factors
                               if f.is_valid and f.originality_ok and not np.isnan(f.ic)]
            top_factors = sorted(valid_evaluated, key=lambda x: x.ic, reverse=True)[:self.n_best_factors]
            
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

            # Record evolution round (store evaluated factors only)
            round_best_ic = max((f.ic for f in evaluated_factors if not np.isnan(f.ic)),
                                default=0.0)
            if val_mode:
                round_best_val_ic = max(
                    (f.val_ic for f in evaluated_factors
                     if f.is_valid and f.val_ic is not None and not np.isnan(f.val_ic)),
                    default=0.0,
                )
            else:
                round_best_val_ic = 0.0
            round_record = EvolutionRound(
                round_id=round_id,
                factors=evaluated_factors,
                best_ic=round_best_ic,
                avg_ic=np.mean([f.ic for f in top_factors]),
            )
            evolution_history.append(round_record)
            
            # Elitism: carry top-N unmutated factors into the next round to
            # prevent diversity collapse. Without this, the entire pool is
            # replaced every round and can converge to a single direction.
            elite_factors: List[CandidateFactor] = []
            if self.elitism_carry > 0 and top_factors:
                elite_factors = sorted(
                    top_factors, key=_ic_key, reverse=True
                )[:self.elitism_carry]
                # Deep-copy the expressions so the elite originals stay intact
                # even if improved_factors accidentally mutate the same objects.
                elite_factors = [
                    CandidateFactor(
                        id=f"elite_r{round_id}_{f.id}",
                        expression=f.expression,
                        description=f.description,
                        parent_id=f.id,
                        generation=f.generation,
                    )
                    for f in elite_factors
                ]
                if elite_factors:
                    print(f"  [elitism] Carried {len(elite_factors)} elite factors "
                          f"(IC: {[f'{f.id[-8:]}:{_ic_key(f):.3f}' for f in elite_factors]})")

            # Update current factors for next round (elites + improved)
            current_factors = elite_factors + improved_factors
            
            # Keep a running list of all evaluated factors (with real IC values)
            # evaluated_factors holds the assessed batch for this round
            all_evaluated_factors.extend(evaluated_factors)

            print(f"  Best IC: {best_ic:.4f}")
            print(f"  Avg IC (top {self.n_best_factors}): {round_record.avg_ic:.4f}")
            if val_mode:
                print(f"  Best val IC: {best_val_ic:.4f} (round best: {round_best_val_ic:.4f})")

            # Track patience: consecutive rounds without improvement.
            # In val mode the early-stop signal is VALIDATION IC — this is the
            # actual guard against train overfitting (a train-IC plateau that
            # keeps climbing while val IC collapses must stop the search).
            if val_mode:
                ref_signal = round_best_val_ic
                ref_best = best_val_ic
            else:
                ref_signal = round_best_ic
                ref_best = best_ic

            if ref_signal >= ref_best:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Update best trackers
            if round_best_ic > best_ic:
                best_ic = round_best_ic
            if val_mode and round_best_val_ic > best_val_ic:
                best_val_ic = round_best_val_ic

            # Check convergence
            if self._check_convergence(evolution_history):
                print("\nConvergence reached!")
                break

            # Check patience: early stop if no improvement for N consecutive rounds
            _early_msg = ("no val-IC improvement" if val_mode else "no IC improvement")
            if no_improvement_count >= self.patience:
                print(f"\nEarly stop: {_early_msg} for {self.patience} consecutive rounds (patience={self.patience})")
                break
        
        # Evaluate the final round's improved_factors — they were generated but never assessed
        if current_factors:
            print(f"\n[evolve] Evaluating final improved factors ({len(current_factors)})...")
            final_metrics = backtester.evaluate_batch(current_factors, max_workers=4, parallel=self.parallel)
            val_final_metrics = None
            if val_mode:
                val_probes = [CandidateFactor(id=f.id, expression=f.expression,
                                          description=f.description or "")
                              for f in current_factors]
                val_final_metrics = val_backtester.evaluate_batch(
                    val_probes, max_workers=4, parallel=self.parallel)
            for i, factor in enumerate(current_factors):
                metrics = final_metrics[i]
                factor.ic = metrics.get('ic', float('nan'))
                factor.icir = metrics.get('icir', float('nan'))
                factor.sharpe = metrics.get('sharpe', 0.0)
                factor.win_rate = metrics.get('win_rate', 0.5)
                factor.max_drawdown = metrics.get('max_drawdown', 0.0)
                factor.is_valid = metrics.get('is_valid', True)
                if not factor.is_valid:
                    factor.parse_error = metrics.get('parse_error', '')
                else:
                    # Originality gate (see per-round eval block for rationale)
                    self._apply_originality_gate(factor)
                if val_mode and val_final_metrics is not None:
                    vmetrics = val_final_metrics[i]
                    if vmetrics.get('is_valid', True):
                        factor.val_ic = vmetrics.get('ic', float('nan'))
                        factor.val_icir = vmetrics.get('icir', float('nan'))
                    else:
                        factor.val_ic = float('nan')
                if factor.is_valid and not np.isnan(factor.ic) and factor.ic > best_ic:
                    best_ic = factor.ic
            # Save improved factors with real IC/Sharpe/win_rate after backtest evaluation
            self._save_factors_to_file(current_factors, len(evolution_history), filename="improved_factors.json")
            all_evaluated_factors.extend(current_factors)

        # Select best factors from all evaluated factors across all rounds.
        # In val mode we rank by VALIDATION IC (the out-of-sample signal) so the
        # survivor set is robust to train overfitting; otherwise by train IC.
        # Factors rejected by the originality gate are excluded here.
        valid_all = [f for f in all_evaluated_factors
                     if f.is_valid and f.originality_ok and not np.isnan(f.ic)
                     and (not val_mode or (f.val_ic is not None and not np.isnan(f.val_ic)))]

        # Safety net: if the originality gate (over-)rejected the entire pool,
        # fall back to ignoring it so the pipeline still yields a survivor set
        # instead of returning an empty result. The mis-tuning is logged loudly.
        if not valid_all:
            parse_valid = [f for f in all_evaluated_factors
                           if f.is_valid and not np.isnan(f.ic)
                           and (not val_mode or (f.val_ic is not None and not np.isnan(f.val_ic)))]
            if parse_valid:
                print(f"  [originality] Warning: gate rejected ALL candidates; "
                      f"falling back to ignoring originality (keeping parse-valid factors).")
                valid_all = parse_valid

        best_factors = sorted(valid_all, key=_ic_key, reverse=True)[:self.n_best_factors]

        # Quality filter: remove factors that don't meet minimum thresholds.
        # In val_mode the PRIMARY gate is validation IC (the honest, holdout
        # signal introduced for anti-overfitting): a factor with a negative val IC
        # is exactly the overfit signature we want to keep out of the pool.
        # min_ic / max_drawdown / min_sharpe are secondary guards. NOTE: f.sharpe
        # is distorted by overlapping-window annualization (daily-sampled 10d
        # forward returns * sqrt(252)), so it is a soft floor, not a hard gate.
        # The degenerate fallback must NOT silently return a wall of junk factors:
        # if every factor fails the gate we keep only the single best by the
        # honest ranking key, so the pipeline survives and the mis-tuning is loud.
        quality_filtered = []
        for f in best_factors:
            if not f.is_valid or not f.originality_ok or np.isnan(f.ic):
                continue
            if val_mode:
                # Honest gate: require a non-negative signal on the validation split.
                if f.val_ic is None or np.isnan(f.val_ic) or f.val_ic < self.min_val_ic:
                    continue
            if f.ic < self.min_ic:
                continue
            if f.sharpe < self.min_sharpe:
                continue
            if f.max_drawdown < self.max_drawdown:
                continue
            quality_filtered.append(f)

        n_before = len(best_factors)
        n_after = len(quality_filtered)
        if n_after < n_before:
            print(f"\n  [quality] Filtered {n_before - n_after} factors below threshold "
                  f"(min_ic={self.min_ic}, min_sharpe={self.min_sharpe}, "
                  f"max_drawdown={self.max_drawdown}, min_val_ic={self.min_val_ic})")

        # Degenerate fallback: thresholds too strict (or genuinely no usable
        # factor). Keep only the single best by the honest ranking key instead of
        # returning the full unfiltered list (which would re-admit junk factors).
        if not quality_filtered:
            print(f"  [quality] Warning: ALL factors failed quality filter "
                  f"(min_ic={self.min_ic}, min_sharpe={self.min_sharpe}, "
                  f"max_drawdown={self.max_drawdown}, min_val_ic={self.min_val_ic}). "
                  f"Keeping the single best by {'val_' if val_mode else ''}IC.")
            quality_filtered = [best_factors[0]] if best_factors else []
        
        return EvolutionResult(
            best_factors=quality_filtered,
            evolution_history=evolution_history,
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
            # Truncate to avoid prompt overflow (keep last 3000 chars)
            notes_truncated = reflection_notes[-3000:] if len(reflection_notes) > 3000 else reflection_notes
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
1. Generate {self.n_improve} improved factors
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
            raw = self._call_llm(system_prompt, user_prompt, temperature=self.improve_temperature, expect_json=True)
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
        """Generate improved factors using rule-based strategies (fallback when LLM unavailable).
        
        Generates up to `n_mutate` variations per top factor using diverse strategies.
        """
        import random
        random.seed(42)

        improved_factors = []

        for i, factor in enumerate(top_factors):
            expr = factor.expression
            other_expr = top_factors[(i + 1) % len(top_factors)].expression if len(top_factors) > 1 else expr

            # All available mutation strategies
            all_strategies = []

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
            all_strategies.append(adjusted_expr)

            # Strategy 2: Add fundamental signal
            fund = random.choice(['pe', 'pb', 'roe', 'market_cap'])
            if random.random() < 0.5:
                fund_expr = f"rank({expr}) * -rank({fund})"
            else:
                fund_expr = f"rank({expr}) + rank(-{fund})"
            all_strategies.append(fund_expr)

            # Strategy 3: Apply nonlinear transformation
            transform = random.choice(['log', 'sqrt', 'abs', 'sign'])
            nonlinear_expr = f"{transform}({expr})"
            all_strategies.append(nonlinear_expr)

            # Strategy 4: Combine with another top factor
            op = random.choice(['*', '+'])
            combined_expr = f"rank({expr}) {op} rank({other_expr})"
            all_strategies.append(combined_expr)

            # Strategy 5: Time-series z-score normalization
            ts_expr = f"ts_zscore({expr}, 20)"
            all_strategies.append(ts_expr)

            # Strategy 6: Rank of rank
            rank_expr = f"rank(rank({expr}))"
            all_strategies.append(rank_expr)

            # Select up to n_mutate strategies (capped by available strategies)
            n_strategies = min(self.n_mutate, len(all_strategies))
            improvements = random.sample(all_strategies, n_strategies) if len(all_strategies) > n_strategies else all_strategies

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
        
        Convergence is reached when IC improvement over the last `convergence_window`
        rounds is below `convergence_delta`.
        
        Args:
            history: Evolution history
            
        Returns:
            True if converged
        """
        if len(history) < self.convergence_window + 1:
            return False
        
        # Check if IC improvement is below threshold over the convergence window
        recent_ics = [h.best_ic for h in history[-(self.convergence_window + 1):]]
        improvement = recent_ics[-1] - recent_ics[0]
        
        return improvement < self.convergence_delta


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

    backtester = FactorBacktester(prices=close_prices, volume=volume_data, forward_period=10)

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
