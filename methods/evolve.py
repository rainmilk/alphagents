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
from functools import lru_cache
import threading
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
    val_sharpe: float = 0.0        # Validation-period long-short Sharpe (holdout window). Same-window counterpart to val_ic, so val_IC vs val_Sharpe comparisons are apples-to-apples (factor.sharpe is on the full TRAIN window). Direction-aligned by sign(val_IC) — see _compute_quantile_metrics.
    is_valid: bool = True          # False when evaluation raises ValueError (parse error)
    parse_error: str = ""          # stores error message for debugging / reflection
    originality_ok: bool = True    # False when the AST originality gate rejects the factor
    gate_reason: str = ""
    family: str = ""           # inferred PRIMARY factor family (e.g. 'Momentum',
                               # 'Value/Quality', 'Liquidity', 'Volatility',
                               # 'Mean-reversion', 'Growth', 'Other'). Used by
                               # diversity-aware selection (family-balanced elitism
                               # + improve input) to combat niche mode-collapse.          # reason recorded by the originality gate
    
    
def _infer_family(expr: str) -> str:
    """Infer a factor's PRIMARY family from its expression, for diversity bucketing.

    Priority order deliberately pulls price/volume OPERATORS out of the
    Value/Quality bucket, so a combined factor like `ts_cov(roe, returns, 20) *
    -rank(pb)` is bucketed as Momentum, not Value. This is what lets
    diversity-aware selection rescue non-value niches from collapse.
    """
    e = (expr or "").lower()
    if any(k in e for k in ("volume", "amt", "turnover", "illiquid")):
        return "Liquidity"
    if any(k in e for k in ("ts_std", "ts_max", "ts_min", "ts_skew", "ts_kurt")):
        return "Volatility"
    if any(k in e for k in ("ts_corr", "ts_cov", "ts_pct_change",
                            "ts_delta", "ts_rank", "returns", "momentum")):
        return "Momentum"
    if any(k in e for k in ("ts_zscore", "reversal", "mean_reversion")):
        return "Mean-reversion"
    if any(k in e for k in ("growth", "eps")):
        return "Growth"
    if any(k in e for k in ("pe", "pb", "ps", "roe", "roa",
                            "market_cap", "value", "margin", "debt", "quality")):
        return "Value/Quality"
    return "Other"


# Canonical factor families — shared by reflection + improve so the two stay in
# lock-step (reflection NAMES the gap; improve is steered toward it).
_ALL_FAMILIES = ["Momentum", "Mean-reversion", "Value/Quality",
                "Volatility", "Liquidity", "Growth"]

# Representative operator/field token per family, used by the rule-based improve
# fallback to SEED a missing family when no LLM is available. Each token is
# chosen so _infer_family() would classify it into the intended family (no
# higher-priority keyword leaks in), but the bridge factors set family EXPLICITLY
# anyway — the composite may contain the parent's keywords that would otherwise
# mis-classify it.
_FAMILY_BRIDGE_TOKEN = {
    "Momentum": "ts_pct_change(close, 20)",
    "Mean-reversion": "ts_zscore(close, 20)",
    "Value/Quality": "rank(-pb)",
    "Volatility": "ts_std(returns, 20)",
    "Liquidity": "ts_mean(volume, 20)",
    "Growth": "ts_pct_change(eps, 60)",
}


def _family_balanced_top(factors, n, key):
    """Select the top-`n` factors by `key` while guaranteeing family diversity.

    Round 1 takes the best factor of each distinct family (so a minority niche
    like Liquidity survives even if its IC is lower). Round 2 fills the
    remaining slots purely by `key`. `key` must return a sortable metric
    (lower = worse); NaN should map to -inf by the caller (e.g. `_ic_key`).
    """
    if not factors or n <= 0:
        return []
    ordered = sorted(factors, key=key, reverse=True)
    seen = set()
    selected = []
    selected_ids = set()
    # Round 1: one representative per family (highest key within that family).
    for f in ordered:
        fam = getattr(f, "family", "") or "Other"
        if fam not in seen:
            selected.append(f)
            selected_ids.add(id(f))
            seen.add(fam)
        if len(selected) >= n:
            break
    # Round 2: fill remaining slots by raw key.
    if len(selected) < n:
        for f in ordered:
            if id(f) not in selected_ids:
                selected.append(f)
                selected_ids.add(id(f))
            if len(selected) >= n:
                break
    return selected[:n]


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
    'ts_slope',
    # --- Alpha101 primitives (extend DSL so MASE can express WorldQuant alphas) ---
    'ts_argmax', 'ts_argmin', 'signedpower', 'scale', 'decay_linear',
    'ts_product', 'correlation',
    # --- elementwise min/max (Alpha101 Min(x,y)/Max(x,y), distinct from rolling ts_min/ts_max) ---
    'ele_min', 'ele_max',
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
    'ts_trend':     'ts_slope',        # alias for rolling linear-regression slope
    'ts_skewness': 'ts_skew', 'ts_kurtosis': 'ts_kurt',
    # --- Alpha101 primitive aliases ---
    'correlation':  'ts_corr',         # Alpha101 cross-sectional corr == per-stock ts_corr
    'ts_arg_max':   'ts_argmax',
    'argmax':       'ts_argmax',
    'ts_arg_min':   'ts_argmin',
    'argmin':       'ts_argmin',
    'signed_power': 'signedpower',
    'spower':       'signedpower',
    'ts_decay_linear': 'decay_linear',  # WorldQuant canonical name for decay_linear
    'decay':        'decay_linear',
    'product':      'ts_product',
    # --- elementwise min/max (Alpha101 Min(x,y)/Max(x,y)) ---
    'Min':          'ele_min',
    'Max':          'ele_max',
    'elementwise_min': 'ele_min',
    'elementwise_max': 'ele_max',
}

# --- Allowed DATA FIELDS (the ONLY identifiers the evaluator accepts) ---
# Anything else (revenue, assets, sales, roa, book, debt, cash, equity, ...)
# raises "Unknown identifier" at backtest. The LLM repeatedly HALLUCINATES these
# fundamentals, so we (a) inject this allowlist into every generation prompt and
# (b) pre-validate generated expressions to DROP bad ones instead of letting them
# waste a backtest slot or spam the logs. This is the single source of truth for
# valid field names — keep it in sync with the evaluator's `self._data` keys.
_ALLOWED_FIELDS = {
    "open", "high", "low", "close", "volume", "amount",
    "pe", "pb", "ps", "roe", "market_cap", "eps",
    "return", "returns", "vwap",
}
_ALLOWED_FIELDS_STR = (
    "ALLOWED DATA FIELDS (use ONLY these exact identifiers; any other field name "
    "— e.g. revenue, assets, sales, roa, book, debt, cash, equity, liability, "
    "margin — does NOT exist in the dataset and will be rejected): "
    + ", ".join(sorted(_ALLOWED_FIELDS)) + "."
)
_KNOWN_FUNCS = _GATE_ALLOWED_FUNCS | set(_GATE_FUNC_ALIASES.keys())

# Function arity (min, max positional args) — SINGLE SOURCE OF TRUTH shared by
# the validation gate (_validate_factor_expr) and the backtester's runtime
# evaluator (_FactorExprEvaluator._call_func). Keeping one copy avoids drift:
# if a function's signature changes here, both the pre-filter and the runtime
# check stay in sync. Mirrors the evaluator's runtime arity table so malformed
# LLM expressions (e.g. ts_zscore(x) with 1 arg instead of 2) are dropped at
# the validation gate instead of crashing during backtest.
_FUNCTION_ARITY = {
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
    'ts_slope':   (2, 2),
    'delay':      (2, 2),
    'sign':       (1, 1),
    'abs':        (1, 1),
    'log':        (1, 1),
    'sqrt':       (1, 1),
    'if':         (3, 3),
    # --- Alpha101 primitives ---
    'ts_argmax':  (2, 2),
    'ts_argmin':  (2, 2),
    'signedpower':(1, 2),
    'scale':      (1, 2),
    'decay_linear':(2, 2),
    'ts_product': (2, 2),
    'correlation':(3, 3),
    # --- elementwise min/max (Alpha101 Min(x,y)/Max(x,y)) ---
    'ele_min':    (2, 2),
    'ele_max':    (2, 2),
}


def _validate_factor_expr(expr: str):
    """Return (is_valid, bad_info).

    Cheap, data-independent pre-filter with two checks:

    1. **Identifier whitelist** — every bare identifier must be either a known
       DSL function or an allowed data field. Catches LLM-hallucinated
       fundamentals (revenue, assets, ...) BEFORE they reach the backtester.
    2. **Arity (argument count)** — every function *call* must respect its
       parameter count. Catches e.g. ``ts_zscore(x)`` with 1 arg when it needs
       2 (``ts_zscore(x, w)``), which would otherwise crash the backtester's
       evaluator with a ``ValueError`` mid-run.

    We only ever DROP expressions we are SURE are invalid. A syntax error (or a
    ``return`` keyword the custom parser tolerates) is NOT flagged here — the
    backtester makes the final authoritative parse. Likewise, an expression we
    cannot AST-parse cleanly is deferred to the backtester rather than dropped.
    """
    pre = FactorOriginalityGate._preprocess(expr)
    ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pre))
    bad = []
    for i in ids:
        name = "if" if i == "_if" else i
        if name in _KNOWN_FUNCS or name in _ALLOWED_FIELDS:
            continue
        bad.append(i)
    if bad:
        return (False, sorted(set(bad)))

    # --- Arity check: walk the AST and validate function-call argument counts ---
    try:
        tree = ast.parse(pre, mode='eval')
    except (SyntaxError, ValueError):
        # Can't parse cleanly — defer to the backtester (don't drop on uncertain
        # syntax). The gate only flags what it's CERTAIN is wrong.
        return (True, [])

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            # Normalize alias → canonical (mirrors the evaluator / gate canon)
            fname = _GATE_FUNC_ALIASES.get(fname, fname)
            if fname == '_if':
                fname = 'if'
            if fname in _FUNCTION_ARITY:
                lo, hi = _FUNCTION_ARITY[fname]
                n_args = len(node.args)
                if not (lo <= n_args <= hi):
                    bad.append(
                        f"{fname} expects {lo}-{hi} args, got {n_args}"
                    )
    if bad:
        return (False, sorted(set(bad)))

    return (True, [])


# Canonical factor-DSL function whitelist injected into every LLM factor-generation
# prompt (seed / memory-augmented / improve). This is the SINGLE SOURCE OF TRUTH:
# the originality gate only accepts these names (plus the aliases in
# _GATE_FUNC_ALIASES), so keeping prompts in sync with this set stops the LLM from
# inventing unregistered names like ts_roc / ts_skewness / ts_kurtosis.
_FUNCTION_WHITELIST_STR = (
    "ALLOWED FUNCTIONS (use ONLY these exact names; any other function name is "
    "rejected by the engine): "
    + ", ".join(sorted(_GATE_ALLOWED_FUNCS))
    + ". A few common aliases (e.g. ts_stddev, ts_average, ts_diff, ts_lag, "
    "ts_pctchange) are auto-normalized, but prefer the exact names listed above."
)

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

@lru_cache(maxsize=512)
def _tokenize(expr: str) -> List[tuple]:
    """Tokenize a factor expression string into (type, value) pairs.

    Memoized: identical expressions reuse the token stream instead of
    re-scanning the string on every ``evaluate()`` call (e.g. train vs val
    backtester, or repeated evolution rounds over the same seeds). The
    returned list is never mutated by the parser — only the evaluator's
    ``_pos`` index advances — so sharing it across calls is safe.
    """
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


# =====================================================================
# Compiled-expression cache
# ---------------------------------------------------------------------
# Alpha101 (and any other fixed/recurring) factor expressions are parsed
# from a DSL *string* on every ``evaluate()`` call. The recursive-descent
# parser re-runs even though the tokenizer itself is memoized. Since the
# Alpha101 library is fixed and known ahead of time, we pre-compile each
# expression into a Python code object ONCE (module-level cache) and execute
# the compiled bytecode on later evaluations — skipping the parse/dispatch
# overhead entirely. The generated code calls the SAME ``_fn_*`` primitives
# as the interpreter, so results are bit-for-bit identical to the slow path.
# Steps 4c/5/6 all evaluate the Alpha101 expressions; per-run this removes
# the repeated parse cost and (because the cache is process-global) any
# later backtester that re-evaluates the same formula reuses the bytecode.
# =====================================================================

_COMPILED: Dict = {}          # expr string -> compiled code object (or _SLOW)
_SLOW = object()              # sentinel: this expr could not be compiled


def _div(a, b):
    """Robust division that mirrors the slow parser's zero-guard.

    The DSL divides DataFrames by DataFrames (replace 0 -> NaN) but also by
    bare numeric literals in rare cases; handle both safely.
    """
    if isinstance(b, (pd.DataFrame, pd.Series)):
        return a / b.replace(0, np.nan)
    return a / (np.nan if b == 0 else b)


class _CIDict(dict):
    """Case-insensitive data lookup so compiled code tolerates PE/pe etc."""

    def __getitem__(self, k):
        if k in self:
            return super().__getitem__(k)
        lk = k.lower()
        if lk in self:
            return super().__getitem__(lk)
        raise KeyError(k)


class _ExprCompiler:
    """Generate Python *source* for a factor expression, mirroring the grammar
    of ``_FactorExprEvaluator`` but emitting code instead of eager DataFrames.

    The emitted source references ``data['field']`` (a case-insensitive view of
    the evaluator's ``_data``), the ``_fn_*`` primitive implementations, ``_div``
    for zero-safe division, and ``np``. It is compiled once via ``compile()``
    and cached; subsequent evaluations just ``eval`` the bytecode.
    """

    def __init__(self, tokens):
        self._tokens = tokens
        self._pos = 0

    def compile(self) -> str:
        src = self._parse_expression()
        if self._peek()[0] != 'EOF':
            raise ValueError(f"Trailing tokens after: {src}")
        return src

    # ---- scanner ----
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

    # ---- grammar (parallel to _FactorExprEvaluator's parser) ----
    def _parse_expression(self) -> str:
        left = self._parse_comparison()
        while self._peek()[0] == 'OP' and self._peek()[1] in ('+', '-'):
            op = self._advance()[1]
            right = self._parse_comparison()
            left = f"({left} {op} {right})"
        return left

    def _parse_comparison(self) -> str:
        left = self._parse_term()
        while self._peek()[0] == "CMP":
            op = self._advance()[1]
            right = self._parse_term()
            left = f"({left} {op} {right})"
        return left

    def _parse_term(self) -> str:
        left = self._parse_power()
        while self._peek()[0] == 'OP' and self._peek()[1] in ('*', '/'):
            op = self._advance()[1]
            right = self._parse_power()
            if op == '*':
                left = f"({left} * {right})"
            else:
                left = f"_div({left}, {right})"
        return left

    def _parse_power(self) -> str:
        left = self._parse_unary()
        while (self._peek()[0] == 'OP' and self._peek()[1] == '^') or \
              (self._peek()[0] == 'POWER' and self._peek()[1] == '**'):
            self._advance()
            right = self._parse_unary()
            left = f"({left} ** {right})"
        return left

    def _parse_unary(self) -> str:
        sign = 1
        while self._peek()[0] == 'OP' and self._peek()[1] in ('+', '-'):
            if self._advance()[1] == '-':
                sign = -sign
        result = self._parse_atom()
        if sign == -1:
            return f"(-{result})"
        return result

    def _parse_atom(self) -> str:
        tok = self._peek()
        if tok[0] == 'NUMBER':
            self._advance()
            return repr(tok[1])
        if tok[0] == 'PAREN' and tok[1] == '(':
            self._advance()
            result = self._parse_expression()
            self._expect('PAREN', ')')
            return f"({result})"
        if tok[0] == 'IDENT':
            name = self._advance()[1]
            if self._peek()[0] == 'PAREN' and self._peek()[1] == '(':
                return self._parse_func_call(name)
            uname = name.upper()
            if uname == "NAN":
                return "np.nan"
            if uname == "INF":
                return "np.inf"
            if uname == "-INF":
                return "(-np.inf)"
            return f"data[{name!r}]"
        raise ValueError(f"Unexpected token {tok}")

    def _parse_func_call(self, name: str) -> str:
        self._expect('PAREN', '(')
        args = []
        if self._peek()[0] == 'PAREN' and self._peek()[1] == ')':
            pass
        else:
            args.append(self._parse_expression())
            while self._peek()[0] == 'COMMA':
                self._advance()
                args.append(self._parse_expression())
        self._expect('PAREN', ')')
        canon = _FactorExprEvaluator._FUNC_ALIASES.get(name, name)
        fn = '_fn_if' if canon == 'if' else f"_fn_{canon}"
        return f"{fn}({', '.join(args)})"


_EVAL_NS = None  # populated lazily (needs _FactorExprEvaluator to exist)


def _ensure_eval_ns():
    """Build the global namespace used to ``eval`` compiled expressions.

    Holds numpy/pandas, the zero-safe ``_div`` helper, and every ``_fn_*``
    primitive (static methods of ``_FactorExprEvaluator``). Built once.
    """
    global _EVAL_NS
    if _EVAL_NS is not None:
        return
    ns = {'np': np, 'pd': pd, '_div': _div}
    for _n in dir(_FactorExprEvaluator):
        if _n.startswith('_fn_'):
            ns[_n] = getattr(_FactorExprEvaluator, _n)
    _EVAL_NS = ns


def _get_compiled(expr: str):
    """Return a compiled ``f(data) -> DataFrame`` callable for ``expr``.

    The callable is cached process-globally (so the fixed Alpha101 library is
    effectively pre-coded once and reused by every later evaluation). Returns
    ``None`` if the expression cannot be compiled — the caller then falls back
    to the original interpreter. Failed compiles are remembered via ``_SLOW``
    so we never retry them.
    """
    cached = _COMPILED.get(expr)
    if cached is _SLOW:
        return None
    if cached is not None:
        return cached
    try:
        tokens = _tokenize(expr)
        if not tokens:
            raise ValueError("no tokens")
        src = "def __alpha_eval__(data):\n    return " + _ExprCompiler(tokens).compile()
        _ensure_eval_ns()
        local = dict(_EVAL_NS)  # exec defines __alpha_eval__ inside this ns
        exec(compile(src, '<alpha101>', 'exec'), local)  # noqa: S102 trusted DSL
        fn = local['__alpha_eval__']
    except Exception:
        _COMPILED[expr] = _SLOW
        return None
    _COMPILED[expr] = fn
    return fn


def precompile_alpha101():
    """Pre-compile every Alpha101 DSL formula into a cached callable.

    Called (guarded) at ``methods.alpha101`` import time so the fixed library
    is "pre-coded" once and Step 4c's scoring calls the bytecode directly.
    No-op if the alpha101 module is unavailable.
    """
    try:
        from methods.alpha101 import get_alpha101_formulas
    except Exception:
        return
    for _expr in get_alpha101_formulas().values():
        try:
            _get_compiled(_expr)
        except Exception:
            pass


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
        # --- Alpha101 primitives ---
        ts_argmax(X, w)  — periods since rolling max (0=today, w-1=w days ago)
        ts_argmin(X, w)  — periods since rolling min
        signedpower(X, p)— sign(X) * |X|^p
        scale(X, a)      — cross-sectional rescale so sum(|X|)=a (default 1)
        decay_linear(X, w) — linearly-weighted moving avg (newest weighted highest)
        ts_product(X, w) — rolling product over trailing w days
        correlation(X, Y, w) — alias of ts_corr (per-stock rolling correlation)
        ele_min(X, Y)     — element-wise min of two series (Alpha101 Min(x,y))
        ele_max(X, Y)     — element-wise max of two series (Alpha101 Max(x,y))

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
        """Parse and evaluate a factor expression.

        Fast path: if ``expr`` has been compiled to a ``f(data)`` callable
        (e.g. the fixed Alpha101 library, or any expression seen before), call
        it directly — skipping the recursive-descent parse entirely. The
        callable's globals are the shared, read-only primitive namespace, and
        ``data`` is passed per call, so concurrent workers are safe. Any
        failure (or an un-compilable expression) transparently falls back to
        the original interpreter in ``_evaluate_slow`` so behaviour is
        identical and nothing regresses.
        """
        if not expr or not expr.strip():
            raise ValueError("Empty factor expression")

        fn = _get_compiled(expr)
        if fn is not None:
            try:
                result = fn(_CIDict(self._data))
            except Exception:
                # Compiled path unexpectedly failed — use the safe interpreter.
                return self._evaluate_slow(expr)
            # Ensure result is a DataFrame (not a scalar or Series)
            if isinstance(result, (int, float)):
                template = next(iter(self._data.values()))
                result = pd.DataFrame(
                    result, index=template.index, columns=template.columns
                )
            return result

        return self._evaluate_slow(expr)

    def _evaluate_slow(self, expr: str) -> pd.DataFrame:
        """Original recursive-descent interpreter (fallback / un-compilable)."""
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
        'ts_trend':     'ts_slope',        # common alias for rolling slope
        # --- Alpha101 primitive aliases ---
        'correlation':  'ts_corr',         # Alpha101 cross-sectional corr == per-stock ts_corr
        'ts_arg_max':   'ts_argmax',
        'argmax':       'ts_argmax',
        'ts_arg_min':   'ts_argmin',
        'argmin':       'ts_argmin',
        'signed_power': 'signedpower',
        'spower':       'signedpower',
        'ts_decay_linear': 'decay_linear',  # WorldQuant canonical name for decay_linear
    'decay':        'decay_linear',
    'product':      'ts_product',
    # --- elementwise min/max (Alpha101 Min(x,y)/Max(x,y)) ---
    'Min':          'ele_min',
    'Max':          'ele_max',
    'elementwise_min': 'ele_min',
    'elementwise_max': 'ele_max',
}

    def _dispatch_func(self, name: str, args: List) -> pd.DataFrame:
        """Route function call to implementation."""
        # Normalize aliases before lookup
        name = self._FUNC_ALIASES.get(name, name)

        arity = len(args)
        # Check arity against the shared gate table (single source of truth).
        if name in _FUNCTION_ARITY:
            lo, hi = _FUNCTION_ARITY[name]
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

    @staticmethod
    def _safe_float(val, default: float = 1.0) -> float:
        """Convert a window/scale argument (scalar DataFrame from a NUMBER token,
        a Series, or a plain number) to float, NaN-safe. Mirrors _safe_int."""
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
        return float(raw)

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
    def _fn_ts_slope(x: pd.DataFrame, window) -> pd.DataFrame:
        """Rolling linear-regression slope (trend strength) of x over window w,
        computed per stock. Equals the OLS slope of x on a local time index —
        the signed trend magnitude per period. A genuinely 'advanced' primitive:
        captures trend *direction & strength* (a flat series → 0, a steep uptrend
        → large positive), unlike a single delta which is scale-blind.

        Closed-form OLS slope with the shift-invariant absolute-time trick:
        slope = (Σ(t·x) − t̄·Σx) / C, where t = absolute row position and
        C = Σ(t−t̄)² is constant over any fixed window. The row-scaling
        arithmetic is done in numpy (not Series×DataFrame) to avoid a pandas
        DatetimeIndex-alignment quirk that would explode the column count."""
        w = _FactorExprEvaluator._safe_int(window)
        if w < 2:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        min_p = min(max(3, w * 2 // 3), w)
        xv = x.values
        n = xv.shape[0]
        P = np.arange(n, dtype=float)          # absolute row positions (shift-invariant)
        t_mean = P - (w - 1) / 2.0             # mean time within the trailing window
        C = w * (w * w - 1) / 12.0             # Σ(t−t̄)² for t=0..w-1 (constant)
        tx = xv * P[:, None]                   # t·x, row-wise numpy broadcast
        tx_df = pd.DataFrame(tx, index=x.index, columns=x.columns)
        sum_tx = tx_df.rolling(window=w, min_periods=min_p).sum().values
        sum_x = x.rolling(window=w, min_periods=min_p).sum().values
        slope = (sum_tx - t_mean[:, None] * sum_x) / C
        return pd.DataFrame(slope, index=x.index, columns=x.columns)



    @staticmethod
    def _fn_if(cond: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """if(cond, x, y) — element-wise ternary: return x where cond!=0, else y."""
        return x.where(cond != 0, y)

    # ---- Alpha101 primitive operators (added to extend the DSL) -----------

    @staticmethod
    def _fn_ts_argmax_slow(x: pd.DataFrame, d) -> pd.DataFrame:
        """Reference (slow) implementation — kept for verification / fallback.

        See :meth:`_fn_ts_argmax` for semantics. Uses ``rolling().apply()`` with a
        Python callback; this is the original (pre-vectorization) hot path.
        """
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)

        def _amax(arr):
            a = np.asarray(arr, dtype=float)
            return np.nan if np.isnan(a).all() else float(np.nanargmax(a))

        return (w - 1) - x.rolling(window=w, min_periods=min_p).apply(_amax, raw=True)

    @staticmethod
    def _fn_ts_argmax(x: pd.DataFrame, d) -> pd.DataFrame:
        """ts_argmax(x, d) — position (in periods) of the max within the trailing
        d-day window, counting BACK from the most recent day. Mirrors WorldQuant's
        ts_argmax: (d-1) - argmax. Returns 0 if today is the max, d-1 if the max was
        d days ago. NaN-aware: all-NaN windows yield NaN.

        Vectorized equivalent of :meth:`_fn_ts_argmax_slow` — **bit-identical**, but
        uses ``numpy.lib.stride_tricks.sliding_window_view`` + ``np.nanargmax`` so it
        runs at C speed instead of crossing the Python boundary once per
        (window, stock). The warm-up / partial-window region (rows ``0..w-2``) is
        handled explicitly so the ``min_periods`` gate and the ``(w-1) - argmax``
        offset match pandas exactly.
        """
        if not isinstance(x, pd.DataFrame):
            return _FactorExprEvaluator._fn_ts_argmax_slow(x, d)
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)
        vals = x.to_numpy(dtype=float)
        T, N = vals.shape
        out = np.full((T, N), np.nan)
        last_lead = min(w - 1, T)
        for i in range(last_lead):                       # warm-up partial windows
            L = i + 1
            seg = vals[:L, :]
            valid = (L - np.isnan(seg).sum(axis=0)) >= min_p
            if valid.any():
                am = np.argmax(np.nan_to_num(seg, nan=-np.inf), axis=0)
                out[i, :] = np.where(valid, (w - 1) - am, np.nan)
        if T >= w:                                       # full windows
            sw = np.lib.stride_tricks.sliding_window_view(vals, w, axis=0)
            n_nan = np.isnan(sw).sum(axis=2)
            valid = ((w - n_nan) >= min_p) & (n_nan < w)
            am = np.argmax(np.nan_to_num(sw, nan=-np.inf), axis=2)
            out[w - 1:, :] = np.where(valid, (w - 1) - am, np.nan)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    @staticmethod
    def _fn_ts_argmin_slow(x: pd.DataFrame, d) -> pd.DataFrame:
        """Reference (slow) implementation — kept for verification / fallback.

        See :meth:`_fn_ts_argmin` for semantics.
        """
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)

        def _amin(arr):
            a = np.asarray(arr, dtype=float)
            return np.nan if np.isnan(a).all() else float(np.nanargmin(a))

        return (w - 1) - x.rolling(window=w, min_periods=min_p).apply(_amin, raw=True)

    @staticmethod
    def _fn_ts_argmin(x: pd.DataFrame, d) -> pd.DataFrame:
        """ts_argmin(x, d) — position (in periods) of the min within the trailing
        d-day window, counting BACK from the most recent day. Mirrors WorldQuant's
        ts_argmin: (d-1) - argmin. Vectorized equivalent of :meth:`_fn_ts_argmin_slow`
        — bit-identical (see :meth:`_fn_ts_argmax` for the vectorization strategy).
        """
        if not isinstance(x, pd.DataFrame):
            return _FactorExprEvaluator._fn_ts_argmin_slow(x, d)
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)
        vals = x.to_numpy(dtype=float)
        T, N = vals.shape
        out = np.full((T, N), np.nan)
        last_lead = min(w - 1, T)
        for i in range(last_lead):                       # warm-up partial windows
            L = i + 1
            seg = vals[:L, :]
            valid = (L - np.isnan(seg).sum(axis=0)) >= min_p
            if valid.any():
                am = np.argmin(np.nan_to_num(seg, nan=np.inf), axis=0)
                out[i, :] = np.where(valid, (w - 1) - am, np.nan)
        if T >= w:                                       # full windows
            sw = np.lib.stride_tricks.sliding_window_view(vals, w, axis=0)
            n_nan = np.isnan(sw).sum(axis=2)
            valid = ((w - n_nan) >= min_p) & (n_nan < w)
            am = np.argmin(np.nan_to_num(sw, nan=np.inf), axis=2)
            out[w - 1:, :] = np.where(valid, (w - 1) - am, np.nan)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    @staticmethod
    def _fn_signedpower(x: pd.DataFrame, p=2) -> pd.DataFrame:
        """signedpower(x, p) = sign(x) * |x|^p. WorldQuant Alpha101 primitive used
        to shape a factor's distribution (e.g. alpha001, alpha013, alpha033, ...)."""
        pval = p if isinstance(p, (int, float)) else _FactorExprEvaluator._safe_int(p)
        return np.sign(x) * (x.abs() ** pval)

    @staticmethod
    def _fn_scale(x: pd.DataFrame, a=1) -> pd.DataFrame:
        """scale(x, a) — rescale *cross-sectionally* (per date, across stocks) so
        that sum(|x|) over stocks equals a (default 1). Matches WorldQuant's scale
        semantics; used by many Alpha101 formulas to normalize magnitude.
        """
        aval = a if isinstance(a, (int, float)) else _FactorExprEvaluator._safe_float(a, default=1.0)
        denom = x.abs().sum(axis=1, skipna=True).replace(0, np.nan)
        return x.div(denom, axis=0) * aval

    @staticmethod
    def _fn_decay_linear_slow(x: pd.DataFrame, d) -> pd.DataFrame:
        """Reference (slow) implementation — kept for verification / fallback.

        See :meth:`_fn_decay_linear` for semantics.
        """
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)
        weights = np.arange(1, w + 1, dtype=float)  # older=1 ... newest=w

        def _decay(arr):
            a = np.asarray(arr, dtype=float)
            n = a.shape[0]
            if n == 0 or np.isnan(a).all():
                return np.nan
            wsub = weights[-n:]  # align weights to the actual (warmup) window length
            return float(np.nansum(a * wsub) / np.nansum(wsub))

        return x.rolling(window=w, min_periods=min_p).apply(_decay, raw=True)

    @staticmethod
    def _fn_decay_linear(x: pd.DataFrame, d) -> pd.DataFrame:
        """decay_linear(x, d) — linearly-weighted moving average over the trailing
        d days, where the MOST RECENT day has the highest weight (1..d). WorldQuant
        Alpha101 primitive (e.g. alpha019, alpha028, alpha047).

        Vectorized equivalent of :meth:`_fn_decay_linear_slow` — **bit-identical**,
        using a broadcast weight matrix over ``sliding_window_view`` so the
        per-window Python callback is eliminated. All-NaN windows yield NaN
        (matching the reference guard).
        """
        if not isinstance(x, pd.DataFrame):
            return _FactorExprEvaluator._fn_decay_linear_slow(x, d)
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)
        weights = np.arange(1, w + 1, dtype=float)
        vals = x.to_numpy(dtype=float)
        T, N = vals.shape
        out = np.full((T, N), np.nan)
        last_lead = min(w - 1, T)
        for i in range(last_lead):                       # warm-up partial windows
            L = i + 1
            seg = vals[:L, :]
            n_nan = np.isnan(seg).sum(axis=0)
            valid = (L - n_nan) >= min_p
            if valid.any():
                wsub = weights[w - L:]                   # length-L tail of weights
                wmat = np.broadcast_to(wsub.reshape(L, 1), (L, N))
                num = np.nansum(seg * wmat, axis=0)
                den = np.nansum(wmat, axis=0)               # full weight-sum (wmat has no NaN)
                out[i, :] = np.where(valid & (den > 0), num / den, np.nan)
        if T >= w:                                       # full windows
            sw = np.lib.stride_tricks.sliding_window_view(vals, w, axis=0)
            wmat = np.broadcast_to(weights.reshape(1, 1, w), sw.shape)
            num = np.nansum(sw * wmat, axis=2)
            den = np.nansum(wmat, axis=2)               # full weight-sum (wmat has no NaN)
            n_nan = np.isnan(sw).sum(axis=2)
            valid = ((w - n_nan) >= min_p) & (n_nan < w) & (den > 0)
            out[w - 1:, :] = np.where(valid, num / den, np.nan)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    @staticmethod
    def _fn_ts_product_slow(x: pd.DataFrame, d) -> pd.DataFrame:
        """Reference (slow) implementation — kept for verification / fallback.

        See :meth:`_fn_ts_product` for semantics.
        """
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)

        def _prod(arr):
            a = np.asarray(arr, dtype=float)
            return np.nan if np.isnan(a).all() else float(np.nanprod(a))

        return x.rolling(window=w, min_periods=min_p).apply(_prod, raw=True)

    @staticmethod
    def _fn_ts_product(x: pd.DataFrame, d) -> pd.DataFrame:
        """ts_product(x, d) — rolling product over the trailing d days, per stock.
        WorldQuant Alpha101 primitive (e.g. alpha009, alpha022). All-NaN windows
        yield NaN; partially-NaN windows multiply the valid entries.

        Vectorized equivalent of :meth:`_fn_ts_product_slow` — **bit-identical**.
        """
        if not isinstance(x, pd.DataFrame):
            return _FactorExprEvaluator._fn_ts_product_slow(x, d)
        w = _FactorExprEvaluator._safe_int(d)
        min_p = min(max(2, w // 2), w)
        vals = x.to_numpy(dtype=float)
        T, N = vals.shape
        out = np.full((T, N), np.nan)
        last_lead = min(w - 1, T)
        for i in range(last_lead):                       # warm-up partial windows
            L = i + 1
            seg = vals[:L, :]
            n_nan = np.isnan(seg).sum(axis=0)
            all_nan = n_nan >= L
            valid = ((L - n_nan) >= min_p) & (~all_nan)
            if valid.any():
                prod = np.prod(np.nan_to_num(seg, nan=1.0), axis=0)
                out[i, :] = np.where(valid, prod, np.nan)
        if T >= w:                                       # full windows
            sw = np.lib.stride_tricks.sliding_window_view(vals, w, axis=0)
            n_nan = np.isnan(sw).sum(axis=2)
            valid = ((w - n_nan) >= min_p) & (n_nan < w)
            prod = np.prod(np.nan_to_num(sw, nan=1.0), axis=2)
            out[w - 1:, :] = np.where(valid, prod, np.nan)
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    @staticmethod
    def _fn_ele_min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """ele_min(x, y) — element-wise minimum of two series (Alpha101 Min(x,y)).

        Note: this is DISTINCT from the rolling ``ts_min``. Both arguments are
        DataFrames (numeric literals are broadcast to scalar DataFrames by the
        tokenizer), so clipping `x` by an upper bound of `y` yields min(x, y)
        element-wise. NaN propagates (np.clip semantics).
        """
        return x.clip(upper=y)

    @staticmethod
    def _fn_ele_max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """ele_max(x, y) — element-wise maximum of two series (Alpha101 Max(x,y)).

        See ``_fn_ele_min`` for the DataFrame/constant broadcast note.
        """
        return x.clip(lower=y)

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
        # Caches. A SINGLE lock guards both dicts; the (expensive) factor
        # computation happens OUTSIDE the lock, so worker threads overlap that
        # pandas/numpy C-level work rather than serialising on the lock.
        self._metric_cache = {}      # expression → metrics dict (no factor_values)
        self._factor_cache = {}      # expression → factor_values DataFrame
        self._cache_lock = threading.Lock()

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

        # Fast path: full metrics already cached for this expression.
        with self._cache_lock:
            cached_metrics = self._metric_cache.get(expr)
        if cached_metrics is not None:
            out = cached_metrics.copy()
            # Re-attach the (cached) factor_values so callers that need the raw
            # panel still get it on a cache hit.
            out['factor_values'] = self._get_factor_values(expr)
            factor.ic = out['ic']
            factor.icir = out.get('icir', 0.0)
            factor.sharpe = out['sharpe']
            factor.win_rate = out['win_rate']
            factor.max_drawdown = out.get('max_drawdown', 0.0)
            return out

        try:
            # Step 1: Compute factor values (cached + thread-safe).
            factor_values = self._get_factor_values(expr)
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

            # Step 4: Quantile portfolio metrics (direction-aligned to IC sign)
            sharpe, win_rate, max_dd, long_short_ret = self._compute_quantile_metrics(
                fv_aligned, fr_aligned, ic=ic
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

            # Cache metrics (without the heavy factor_values / long_short_ret).
            with self._cache_lock:
                self._metric_cache[expr] = {
                    k: v for k, v in metrics.items()
                    if k not in ('long_short_ret', 'factor_values')
                }

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
        """Clear cached metrics AND cached factor values."""
        with self._cache_lock:
            self._metric_cache.clear()
            self._factor_cache.clear()

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

        def _evaluate_one(factor):
            # All caching (metric + factor_values, thread-safe) lives inside
            # self.evaluate(), which guards both dicts with self._cache_lock.
            # Delegating here keeps a single source of truth for the cache logic.
            try:
                return self.evaluate(factor)
            except Exception as e:
                logger.warning("Batch eval failed for '%s': %s", factor.expression, e)
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
                except Exception:
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
        ic: float = 0.0,
    ) -> Tuple[float, float, float, pd.Series]:
        """
        Build a long-short quantile portfolio and compute performance metrics.

        For each period:
          1. Sort stocks by factor value and assign to n_quantiles
          2. Rank the quantiles by factor value (highest = top, lowest = bottom)
          3. Long-short return = sign(IC) * (top return - bottom return)

        Direction is aligned to the factor's Rank-IC sign: a factor whose IC is
        negative is only predictive when traded REVERSED (long the bottom
        quantile / short the top). Flipping here means the reported Sharpe,
        win_rate and max_drawdown reflect the factor's *usable* direction rather
        than a hardcoded long-high/short-low convention — this rescues genuinely
        predictive reverse factors that would otherwise be mis-scored as losers.
        Convention (matches README §Sign Extraction and fusion.py): when |IC| is
        near zero the direction is unreliable, so we default to +1 (no flip) —
        near-zero-IC factors already carry ~zero weight downstream.

        Returns:
            (annualized_sharpe, win_rate, max_drawdown, long_short_returns_series)
        """
        # Align portfolio direction to the factor's predictive sign.
        ic_sign = 1.0 if (ic is None or np.isnan(ic) or abs(ic) <= 1e-10) else float(np.sign(ic))

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

            # Assign quantile labels (0 = lowest, n_quantiles-1 = highest)
            try:
                labels = pd.qcut(fv_valid, q=n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                continue

            # Compute equal-weighted return per quantile
            top_mask = labels == labels.max()
            bot_mask = labels == labels.min()

            top_ret = fr_valid[top_mask].mean()
            bot_ret = fr_valid[bot_mask].mean()

            long_short_rets.append(ic_sign * (top_ret - bot_ret))

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

    def _get_factor_values(self, expr: str) -> pd.DataFrame:
        """
        Thread-safe, cached raw factor computation.

        ``factor_values`` depends ONLY on the price/fundamental data
        (``self._data_map``) — never on ``forward_returns`` — so it is safe to
        memoize for the lifetime of the backtester. This is the dominant cost
        of evaluation (parse + pandas panel math), so caching pays off when the
        same expression is evaluated repeatedly (seeds / survivors across
        evolution rounds, or duplicate expressions inside one batch).

        The underlying ``_FactorExprEvaluator`` comes from the ``self.evaluator``
        property, which returns a FRESH instance per call (mutable parse state,
        see its docstring), so concurrent workers never share parser state. The
        cached DataFrame is returned by reference and MUST be treated as
        read-only by callers (downstream code only slices/reads it).

        Uses double-checked locking: the expensive compute happens OUTSIDE the
        lock, and a second check on store prevents two racing threads from both
        computing the same expression.
        """
        with self._cache_lock:
            cached = self._factor_cache.get(expr)
            if cached is not None:
                return cached

        # Compute outside the lock so worker threads overlap this work.
        computed = self.evaluator.evaluate(expr)

        with self._cache_lock:
            # Second check: if another thread already stored this expr while we
            # were computing, keep the first result and discard ours.
            existing = self._factor_cache.get(expr)
            if existing is None:
                self._factor_cache[expr] = computed
                return computed
            return existing

    def compute_factor_values(self, expression: str) -> pd.DataFrame:
        """Compute raw factor values for a given expression (cached)."""
        try:
            return self._get_factor_values(expression)
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
        improve_temperature: float = 0.7,
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
                (NOT seed generation). Low (~0.3) = focused refinement; higher
                (~0.7) = more exploration. Diversity is now primarily enforced by
                the improve-prompt's family-rotation + operator-diversity rules
                (not by temperature alone), so 0.7 is safe here; the old 0.8 with
                NO such constraints caused "fresh-lottery" collapse.
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
        combo_count = 0  # populated in section 2; referenced by Suggestions (section 4)
        
        lines = []
        lines.append(f"=== Reflection for Round {round_id + 1} ===")
        lines.append(f"Total factors evaluated: {len(evaluated_factors)}")
        lines.append(f"Successful (IC > 0.02): {len(successful)}")
        lines.append(f"Moderate (0 <= IC <= 0.02): {len(moderate)}")
        lines.append(f"Failed (IC < 0): {len(failed)}")
        lines.append("")
        
        # --- 2. Analyze successful factors: family & pattern distribution ---
        # Reuse the persisted `family` field (falling back to _infer_family for
        # any factor lacking it, e.g. loaded from an old run) so the reported
        # taxonomy stays consistent with the family-balanced SELECTION in evolve().
        def _fam(f):
            return getattr(f, "family", "") or _infer_family(f.expression)

        if successful:
            lines.append("--- Successful Factor Patterns ---")

            fam_counts = {}
            for f in successful:
                fam = _fam(f)
                fam_counts[fam] = fam_counts.get(fam, 0) + 1

            lines.append("Family distribution in successful factors:")
            for fam, cnt in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {fam}: {cnt}")

            # Combo = expression uses 2+ distinct FUNCTIONS (mixes factor types)
            combo_count = sum(
                1 for f in successful
                if len(set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', f.expression))) >= 2
            )
            if combo_count > 0:
                lines.append(f"  - Combined (2+ distinct functions): {combo_count} → mixing factor types is working")

            # Top 3 successful factor expressions (rank by val_IC when available)
            def _rank_key(x):
                vic = x.val_ic
                return vic if (vic is not None and not np.isnan(vic)) else (x.ic or 0.0)

            top3 = sorted(successful, key=_rank_key, reverse=True)[:3]
            lines.append("")
            lines.append("Top successful factor expressions:")
            for f in top3:
                sharpe_str = f"{f.sharpe:.2f}" if f.sharpe else "N/A"
                vic = f.val_ic
                vic_str = f"{vic:.4f}" if (vic is not None and not np.isnan(vic)) else "N/A"
                vsharpe = getattr(f, 'val_sharpe', None)
                vsharpe_str = (f"{vsharpe:.2f}"
                               if (vsharpe is not None and not np.isnan(vsharpe)) else "N/A")
                lines.append(f"  - IC={f.ic:.4f}, val_IC={vic_str}, "
                              f"Sharpe(train)={sharpe_str}, val_Sharpe={vsharpe_str}: {f.expression}")

            lines.append("")
        
        # --- 3. Analyze failed factors: root causes (family-aware) ---
        if failed:
            lines.append("--- Failed Factor Analysis ---")

            fail_counts = {}
            for f in failed:
                fam = _fam(f)
                fail_counts[fam] = fail_counts.get(fam, 0) + 1
            for fam, cnt in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {cnt} {fam} factor(s) had NEGATIVE IC → consider reversing sign or dropping this style")

            too_complex = [f for f in failed if len(re.findall(r'[+\-*/()]', f.expression)) > 10]
            if too_complex:
                lines.append(f"  - {len(too_complex)} factors are overly complex (too many operators) → Simplify expressions")

            lines.append("")
        
        # --- 4. Strategic suggestions for next round (data-driven, family-aware) ---
        # Bridge from MEASUREMENT -> next round's LLM improve prompt. Name the
        # concrete family GAP so the prompt's "spread across families" constraint
        # has something specific to target (otherwise it stays a generic nudge).
        lines.append("--- Suggestions for Next Generation ---")

        ALL_FAMS = _ALL_FAMILIES
        strong = top_factors if top_factors else successful
        strong_fam = {}
        for f in strong:
            fam = _fam(f)
            strong_fam[fam] = strong_fam.get(fam, 0) + 1

        lines.append("Family coverage of the STRONGEST factors (these feed next round):")
        for fam in ALL_FAMS:
            n = strong_fam.get(fam, 0)
            tag = "   <- MISSING: prioritize next round" if n == 0 else ""
            lines.append(f"  - {fam}: {n}{tag}")

        present = {fam: strong_fam[fam] for fam in ALL_FAMS if strong_fam.get(fam, 0) > 0}
        missing = [fam for fam in ALL_FAMS if strong_fam.get(fam, 0) == 0]
        if present:
            dom_fam = max(present, key=lambda k: present[k])
            lines.append(
                f"  Dominant family: {dom_fam} ({present[dom_fam]}/{len(strong)}). "
                f"Do NOT generate more {dom_fam}-only variants."
            )

        if strong:
            def _rank_key2(x):
                vic = x.val_ic
                return vic if (vic is not None and not np.isnan(vic)) else (x.ic or 0.0)
            best = max(strong, key=_rank_key2)
            lines.append(
                f"  1. ANCHOR: keep the best factor (IC={best.ic:.4f}) as-is; "
                f"do not mutate it into a different style."
            )
            if missing:
                lines.append(
                    f"  2. PRIORITY: generate 2+ NEW factors from each MISSING family "
                    f"-> {', '.join(missing)}."
                )
                lines.append(
                    "     The pipeline now carries one representative per family into the"
                )
                lines.append(
                    "     next round, so target these GAPS rather than re-deriving the dominant style."
                )
            elif len(present) > 1:
                minority = [k for k in present if k != dom_fam]
                lines.append(
                    f"  2. All families present but {dom_fam} dominates -> generate 2+ more from"
                )
                lines.append(f"     the minority families: {', '.join(minority)}.")
            if combo_count > 0:
                lines.append(
                    "  3. Keep mixing factor types (e.g. value+quality, momentum+liquidity)."
                )
        else:
            lines.append("  1. No successful factors this round -> try completely different templates")
            lines.append("     (e.g. switch from value to momentum/liquidity).")

        if failed and len(failed) > len(successful):
            lines.append(
                "  4. High failure rate -> favor robust, well-understood operators and"
            )
            lines.append("     reduce expression complexity; double-check sign direction.")

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
        * ``n_seeds_memory_augment``  → memory-augmented factors. These are NOT
          generated here; ``main.py`` feeds this count to
          ``MemoryAugmentedGenerator`` and concatenates the result. (We only
          report the target here for logging.)

        Because the counts are explicit, if the hypothesis pipeline under-
        delivers we do NOT silently re-fill those slots from alpha101 — the
        produced numbers reflect exactly what was requested. Each subset still
        best-effort tops up its OWN count if the LLM under-delivers a single call.
        Falls back to rule-based generation (filling the requested total) only
        when the LLM is entirely unavailable.
        """
        n_hyp_target = self.n_seeds_hypothesis
        n_mem_target = self.n_seeds_memory_augment
        print(f"Generating seed factors — hypothesis: {n_hyp_target}, "
              f"memory-augment(target): {n_mem_target}")

        if not (self.use_llm and self.client):
            # LLM unavailable → rule-based fills the local (non-memory) total.
            total = n_hyp_target
            print(f"  [evolve] LLM unavailable -> rule-based generation of {total}.")
            return self._generate_factors_rule_based(total)

        # ── Hypothesis-driven subset ──
        hyp_factors: List[CandidateFactor] = []
        if n_hyp_target > 0:
            hyp_factors = self._generate_hypothesis_factors(n_hyp_target)
            print(f"  Generated {len(hyp_factors)}/{n_hyp_target} "
                  f"hypothesis-driven seed factors")
            if not hyp_factors:
                print("  [hypothesis] pipeline yielded nothing.")

        # Alpha101 factors are NO LONGER generated as seeds here — Step 4c
        # (main.step4c_retrieve_alpha101) now does LLM mining: it scores the
        # Alpha101 library on TRAIN data, builds inspiration chains, and uses
        # the LLM to evolve novel expressions. Only LLM-generated factors
        # enter the candidate pool.
        result = hyp_factors
        print(f"Generated {len(result)} seed factors locally "
              f"({len(hyp_factors)} hypothesis-driven). "
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

""" + _FUNCTION_WHITELIST_STR + "\n\n" + _ALLOWED_FIELDS_STR + """

Supported operators: +, -, *, /, ^

PRIMARY OPERATOR MENU (pick a DIFFERENT one for each factor you emit):
  - ts_corr(x, y, w)        cross-sectional/time-series correlation
  - ts_cov(x, y, w)         covariance
  - ts_pct_change(x, w)     period-over-period change
  - ts_rank(x, w)           cross-sectional rank over a window
  - ts_zscore(x, w)         z-score over a window
  - ts_std(x, w)            rolling volatility
  - ts_delay(x, w)          lagged value
  - ts_min(x, w) / ts_max(x, w)
  - ts_slope(x, w)          rolling linear-regression slope (trend strength)
  - ts_decay(x, w)          linearly-decaying weighted average
  - ts_mean(x, w) / ts_sum(x, w)
  - rank(x) / -rank(x)      cross-sectional rank
  - ts_argmax(x, w) / ts_argmin(x, w)   rolling position of max/min (Alpha101)
  - signedpower(x, p)       sign(x)*|x|^p — shape a factor's distribution
  - scale(x, a)             cross-sectional rescale (sum|rows|=a) for magnitude control
  - decay_linear(x, w)      linearly-decaying weighted average (recency-weighted)
  - ts_product(x, w)        rolling product
  - correlation(x, y, w)    cross-sectional correlation (alias of ts_corr)
  - ele_min(x, y) / ele_max(x, y)   element-wise min/max (Alpha101 Min/Max; distinct from rolling ts_min/ts_max)

DIVERSITY MANDATE (critical — factors that violate this are低 quality):
  1. SPREAD across families: Momentum, Mean-reversion, Value/Quality, Volatility, Liquidity, Growth.
  2. Each factor MUST use a DIFFERENT primary operator from the menu above.
  3. NO near-duplicates: two factors must not be algebraically equivalent or differ only by a
     constant multiplier, a sign flip, or swapping pe<->pb<->ps<->roe within the same family.
  4. Every factor needs a clear one-line economic intuition."""

        # Build JSON example once (shared by both branches) — showcases a
        # DIFFERENT family AND a DIFFERENT primary operator per factor so the
        # few-shot demonstration practises what the prompt preaches.
        json_example = json.dumps({
            "factors": [
                {"expression": "rank(ts_corr(close, volume, 20))",
                 "description": "Price-volume correlation momentum", "family": "Momentum"},
                {"expression": "-rank(ts_zscore(close, 20))",
                 "description": "Short-term mean-reversion", "family": "Mean-reversion"},
                {"expression": "-rank(pb) * rank(roe)",
                 "description": "Cheap + profitable quality-value", "family": "Value/Quality"},
                {"expression": "-rank(ts_std(returns, 20))",
                 "description": "Low realized volatility", "family": "Volatility"},
                {"expression": "rank(ts_mean(volume, 20) / ts_mean(volume, 60))",
                 "description": "Liquidity surge vs its own baseline", "family": "Liquidity"},
                {"expression": "rank(ts_pct_change(eps, 60))",
                 "description": "EPS (earnings) growth acceleration", "family": "Growth"}
            ]
        }, ensure_ascii=False)

        # Per-call diversity scaffolding (n_factors varies, so compute spread here)
        _fam_targets = list(_ALL_FAMILIES)
        _per = max(1, n_factors // len(_fam_targets))
        _spread_instr = (
            f"Assign approximately {_per} factors to EACH of these families "
            f"({', '.join(_fam_targets)}) and distribute any remainder across them; "
            f"do NOT over-concentrate in Value/Quality."
        )

        if hypothesis:
            user_prompt = (
                f"Generate {n_factors} factor expressions that capture the following market hypothesis:\n\n"
                f"  \"{hypothesis}\"\n\n"
                + "Requirements:\n"
                + "1. Each factor should directly or indirectly express the hypothesis above\n"
                + "2. Expressions must be valid and use only supported functions and data sources\n"
                + "3. Avoid trivial factors (e.g., just \"close\" or \"volume\")\n"
                + "4. Each factor should have economic intuition tied to the hypothesis\n"
                + "5. Even though they share a theme, DIVERSIFY: use a different primary operator "
                "and, where possible, a different family for each\n"
                + "6. NO near-duplicates: two factors must not be algebraically equivalent or differ "
                "only by a sign flip, constant multiplier, or swapping pe<->pb<->ps<->roe in the same family\n"
                + "7. Each factor object MUST include a \"family\" field (one of: "
                + ", ".join(_fam_targets) + ", or Other)\n"
                + "8. You MUST return exactly " + str(n_factors) + " factors\n"
                + "9. Return a JSON object with key \"factors\" (array of factor objects), no other text, no markdown fences\n"
                + "\n"
                + "Example (valid JSON object with \"factors\" key):\n"
                + json_example + "\n\n"
                + f"Please generate {n_factors} factors now. Return only the JSON object."
            )
        else:
            user_prompt = (
                f"Please generate {n_factors} diverse factor expressions for A-share stock selection.\n"
                + "Requirements:\n"
                + "1. " + _spread_instr + "\n"
                + "2. Expressions must be valid and use only supported functions and data sources\n"
                + "3. Avoid trivial factors (e.g., just \"close\" or \"volume\")\n"
                + "4. Each factor should have clear economic intuition\n"
                + "5. Each factor MUST use a DIFFERENT primary operator (see the menu in the system prompt)\n"
                + "6. NO near-duplicates: two factors must not be algebraically equivalent or differ "
                "only by a sign flip, constant multiplier, or swapping pe<->pb<->ps<->roe in the same family\n"
                + "7. Each factor object MUST include a \"family\" field (one of: "
                + ", ".join(_fam_targets) + ", or Other)\n"
                + "8. You MUST return exactly " + str(n_factors) + " factors\n"
                + "9. Return a JSON object with key \"factors\" (array of factor objects), no other text, no markdown fences\n"
                + "\n"
                + "Example (valid JSON object with \"factors\" key):\n"
                + json_example + "\n\n"
                + f"Please generate {n_factors} factors now. Return only the JSON object."
            )

        try:
            raw = self._call_llm(system_prompt, user_prompt, temperature=0.7, expect_json=True)
            factors_json = self._parse_llm_json(raw)



            seed_factors = []
            _valid_fams = set(_ALL_FAMILIES) | {"Other"}
            for i, f in enumerate(factors_json):
                if not isinstance(f, dict) or "expression" not in f:
                    continue
                expr = self._fix_parentheses(f["expression"])
                # Drop expressions that reference non-existent data fields
                # (e.g. revenue/assets/sales the LLM hallucinates) BEFORE they
                # waste a backtest slot. The backtester also guards this, but
                # skipping early keeps the factor budget on valid candidates.
                _ok, _bad = _validate_factor_expr(expr)
                if not _ok:
                    print(f"  [evolve] Skipping seed factor — invalid: "
                          f"{_bad}: {expr}")
                    continue
                # Prefer the LLM's explicit family label (it often knows a
                # combined factor's intent better than keyword inference), but
                # validate it — fall back to inference if the label is missing
                # or unknown.
                fam = f.get("family") or ""
                fam = fam if fam in _valid_fams else _infer_family(expr)
                factor = CandidateFactor(
                    id=f"seed_llm_{i}",
                    expression=expr,
                    description=f.get("description", f"LLM-generated factor {i}"),
                    generation=0,
                    family=fam,
                )
                seed_factors.append(factor)

            return seed_factors
            
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [evolve] Failed to parse LLM response: {e}")
            print(f"  [evolve] Raw response: {raw[:300] if 'raw' in locals() else 'N/A'}")
            raise
        except Exception as e:
            raise

    # ═══════════════════════════════════════════════════════════════════
    # Alpha101-inspired LLM mining (ported from FAMA _run_llm_mining)
    # ═══════════════════════════════════════════════════════════════════

    def llm_mine_alpha101_inspired(
        self,
        backtester: 'FactorBacktester',
        max_workers: int = 0,
        n_iters: int = 3,
        n_per_iter: int = 5,
        max_chain_len: int = 5,
        temperature: float = 0.7,
        alpha101_ratio: float = 1.0,
        alpha101_top_k: int = 5,
    ) -> List[CandidateFactor]:
        """Generate NEW factor expressions via LLM chain evolution.

        Mirrors FAMA's ``_run_llm_mining``: builds per-family "inspiration
        chains" from the Alpha101 formula library, then uses LLM to *evolve*
        novel expressions.

        Both the raw Alpha101 library factors (Stage 1, top-k by |IC|) and the
        LLM-evolved factors (Stage 2) are returned for **joint evaluation** in
        the candidate pool.  Set ``alpha101_top_k <= 0`` to return LLM-mined
        factors only (legacy behavior — library factors used purely as
        inspiration, never entered the pool).

        The Alpha101 formula library is **always** used as the inspiration
        source (matching FAMA's ``formula_map``).  ``alpha101_ratio``
        controls whether the formulas are *scored* before chain building:

        * **alpha101_ratio > 0** (default): loads the Alpha101 library (98
          formulas), scores all on the TRAIN backtester, and builds chains
          ordered by |IC|.  This is the expensive path (98 parallel
          evaluations) but provides IC-ranked inspiration.

        * **alpha101_ratio == 0**: loads the Alpha101 library but **skips
          scoring** — all formulas get IC=0.0 placeholder (matching FAMA's
          ``_seed_clusters_from_formula_map``).  Chains are built in
          arbitrary order (uniform IC), but the LLM still receives real
          formula expressions to evolve.  The expensive 98-factor
          evaluation is avoided entirely.

        Args:
            backtester: TRAIN FactorBacktester for scoring (when
                alpha101_ratio > 0) and for evaluating LLM-generated factors
                (always).
            max_workers: Parallel workers for Alpha101 scoring (0 = auto).
                Ignored when alpha101_ratio == 0.
            n_iters: Mining iterations (each walks every family chain).
            n_per_iter: Factors requested per LLM call.
            max_chain_len: Max chain length per family (top-N by |IC|).
            temperature: LLM sampling temperature.
            alpha101_ratio: 0 = load formulas without scoring (IC=0.0
                placeholder); >0 = load + score Alpha101 library.
            alpha101_top_k: Number of Stage-1 Alpha101 library factors
                (top by |IC|) to return alongside the LLM-mined factors.
                0 = return LLM-mined factors only (legacy behavior).

        Returns:
            List of CandidateFactor objects — top-k Alpha101 library factors
            (Stage 1) plus all LLM-mined factors (Stage 2), deduped by
            expression.
        """
        if not self.use_llm or not self.client:
            print("  [alpha101-mining] LLM not available; skipping mining.")
            return []

        # --- 0. Load Alpha101 library + optionally score it ---
        inspiration_source = "Alpha101 library"
        try:
            from methods.alpha101 import get_alpha101_formulas
        except Exception as e:
            print(f"  [alpha101-mining] Cannot import Alpha101 "
                  f"library ({e}).")
            return []

        formulas = get_alpha101_formulas()
        if not formulas:
            print("  [alpha101-mining] Alpha101 library is empty; "
                  "skipping.")
            return []

        if alpha101_ratio > 0:
            # Score all formulas on TRAIN data for IC-ranked chains
            a101_factors = [
                CandidateFactor(
                    id=f"a101_{fid}",
                    expression=expr,
                    description=f"Alpha101 {fid}",
                    generation=0,
                    family=_infer_family(expr),
                )
                for fid, expr in formulas.items()
            ]
            print(f"  [alpha101-mining] Loaded {len(a101_factors)} Alpha101 "
                  f"expressions; scoring on TRAIN data...")

            _mw = (max_workers if max_workers > 0
                   else max(1, min(32, (os.cpu_count() or 4) + 4)))
            metrics_list = backtester.evaluate_batch(
                a101_factors, max_workers=_mw, parallel=True)

            scored_expressions: List[Tuple[str, float]] = []
            for f, m in zip(a101_factors, metrics_list):
                ic = m.get('ic', float('nan'))
                if not np.isnan(ic):
                    scored_expressions.append((f.expression, ic))

            # Stage-1 library factors (raw Alpha101 formulas) — scored and kept
            # for joint evaluation with the LLM-mined Stage-2 factors.
            lib_factors = []
            for f, m in zip(a101_factors, metrics_list):
                ic = m.get('ic', float('nan'))
                if not np.isnan(ic):
                    f.ic = ic
                    lib_factors.append(f)

            if not scored_expressions:
                print("  [alpha101-mining] No valid Alpha101 factors "
                      "scored; skipping mining.")
                return []

            print(f"  [alpha101-mining] {len(scored_expressions)} valid "
                  f"scored expressions → building inspiration chains...")
        else:
            # Skip scoring — use all formulas with IC=0.0 placeholder
            # (matches FAMA's _seed_clusters_from_formula_map)
            scored_expressions = [
                (expr, 0.0) for expr in formulas.values()
            ]
            # Stage-1 library factors (raw Alpha101 formulas) — IC=0.0
            # placeholder; returned for joint eval if alpha101_top_k > 0.
            lib_factors = [
                CandidateFactor(
                    id=f"a101_{fid}",
                    expression=expr,
                    description=f"Alpha101 {fid}",
                    generation=0,
                    family=_infer_family(expr),
                )
                for fid, expr in formulas.items()
            ]
            print(f"  [alpha101-mining] alpha101_ratio=0: loaded "
                  f"{len(scored_expressions)} Alpha101 formulas "
                  f"(scoring skipped, IC=0.0 placeholder) → building "
                  f"inspiration chains...")

        # --- Select top-k Stage-1 Alpha101 library factors to return ---
        # Raw Alpha101 formulas returned for joint evaluation with the
        # LLM-mined Stage-2 factors. alpha101_top_k <= 0 disables returning
        # library factors (LLM-mined only, as before).
        _top_k = int(alpha101_top_k)
        if _top_k > 0 and lib_factors:
            lib_factors.sort(
                key=lambda x: abs(getattr(x, 'ic', 0.0) or 0.0), reverse=True)
            top_k_lib = lib_factors[:_top_k]
        else:
            top_k_lib = []
        for f in top_k_lib:
            f._from_alpha101 = True
            f._from_alpha101_lib = True

        # --- 1. Group by family and build initial chains ---
        from collections import defaultdict
        family_buckets: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for expr, ic in scored_expressions:
            if np.isnan(ic):
                continue
            fam = _infer_family(expr)
            family_buckets[fam].append((expr, ic))

        # Sort each family by |IC| desc, keep top max_chain_len
        chains: Dict[str, List[Tuple[str, float]]] = {}
        for fam, items in family_buckets.items():
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            chains[fam] = items[:max_chain_len]

        if not chains:
            print("  [alpha101-mining] No valid family chains built.")
            return []

        print(f"\n  [alpha101-mining] {len(chains)} family chains, "
              f"{n_iters} iterations, {n_per_iter} factors/call")
        print(f"  [alpha101-mining] Families: {', '.join(chains.keys())}")

        # Track all known expressions to avoid re-evaluating duplicates
        known = {expr for expr, _ in scored_expressions}
        mined_factors: List[CandidateFactor] = []
        mine_counter = 0

        for iteration in range(1, n_iters + 1):
            new_this_iter = 0
            for fam, chain in chains.items():
                if not chain:
                    continue

                # Build the LLM prompt with this family's chain as improve_path
                chain_exprs = [expr for expr, _ in chain[:max_chain_len]]
                chain_ics = [f"{ic:+.4f}" for _, ic in chain[:max_chain_len]]

                system_prompt = (
                    "You are an alpha-mining agent specializing in A-share stock "
                    "selection. You are given a chain of top-performing "
                    f"{inspiration_source} "
                    "factor expressions (ordered by improving performance) and "
                    "must generate NEW, different factor expressions inspired by "
                    "them.\n\n"
                    + _FUNCTION_WHITELIST_STR + "\n\n"
                    + _ALLOWED_FIELDS_STR + "\n\n"
                    "Supported operators: +, -, *, /, ^\n"
                    "Supported comparisons: <, >, <=, >=, ==, !=\n"
                    "Ternary: if(condition, value_if_true, value_if_false)\n\n"
                    "RULES:\n"
                    "1. Each new factor MUST be different from every input factor.\n"
                    "2. Do NOT just wrap an input in rank() or negate it — combine "
                    "ideas from multiple chain factors or use a different operator.\n"
                    "3. Use ONLY the functions and data fields listed above.\n"
                    "4. Each factor should have clear economic intuition.\n"
                    "5. Return a JSON object: {\"factors\": [{\"expression\": \"...\", "
                    "\"description\": \"...\"}, ...]}\n"
                    "6. Return ONLY the JSON object — no markdown, no explanation."
                )

                user_prompt = (
                    f"Generate {n_per_iter} NEW factor expressions inspired by "
                    f"the following {inspiration_source} factor chain "
                    f"(family: {fam}).\n\n"
                    f"The chain shows factors ordered by improving performance "
                    f"(IC values in parentheses). Use these as inspiration to "
                    f"create novel expressions:\n\n"
                )
                for i, (expr, ic) in enumerate(chain[:max_chain_len]):
                    user_prompt += f"  Factor {i+1} (IC={ic:+.4f}): {expr}\n"

                user_prompt += (
                    f"\nGenerate {n_per_iter} NEW expressions. Each must be "
                    f"algebraically distinct from the inputs above and from "
                    f"each other. Return JSON: "
                    f'{{"factors": [{{"expression": "...", "description": "..."}}]}}'
                )

                # --- LLM call with per-iteration error isolation ---
                try:
                    raw = self._call_llm(
                        system_prompt, user_prompt,
                        temperature=temperature, expect_json=True,
                    )
                    factors_json = self._parse_llm_json(raw)
                except Exception as e:
                    print(f"  [alpha101-mining iter {iteration} {fam}] "
                          f"LLM call failed: {e}")
                    continue

                if not factors_json:
                    print(f"  [alpha101-mining iter {iteration} {fam}] "
                          f"No factors parsed from LLM response.")
                    continue

                # --- Validate + evaluate each new expression ---
                for f_data in factors_json:
                    if not isinstance(f_data, dict) or "expression" not in f_data:
                        continue
                    expr = self._fix_parentheses(f_data["expression"])

                    # Skip duplicates of known expressions
                    if expr in known:
                        continue

                    # Skip expressions with invalid data fields
                    _ok, _bad = _validate_factor_expr(expr)
                    if not _ok:
                        print(f"  [alpha101-mining] Skipping invalid expr "
                              f"({_bad}): {expr[:80]}")
                        continue

                    known.add(expr)

                    # Evaluate on the TRAIN backtester
                    factor = CandidateFactor(
                        id=f"a101_mined_{mine_counter}",
                        expression=expr,
                        description=f_data.get("description",
                                               f"{inspiration_source}-inspired ({fam})"),
                        generation=iteration,
                        family=_infer_family(expr),
                    )
                    mine_counter += 1

                    try:
                        metrics = backtester.evaluate(factor)
                    except Exception as e:
                        print(f"  [alpha101-mining] Eval error for "
                              f"{expr[:60]}: {e}")
                        continue

                    ic = metrics.get('ic', float('nan'))
                    if np.isnan(ic):
                        continue

                    factor.ic = ic
                    factor.icir = metrics.get('icir', 0.0)
                    factor.sharpe = metrics.get('sharpe', 0.0)
                    factor.is_valid = metrics.get('is_valid', True)
                    mined_factors.append(factor)
                    new_this_iter += 1

                    # Update chain: add new factor, keep top by |IC|
                    updated = chain + [(expr, ic)]
                    updated.sort(key=lambda x: abs(x[1]), reverse=True)
                    chains[fam] = updated[:max_chain_len]

                    print(f"  [alpha101-mining iter {iteration} {fam}] "
                          f"IC={ic:+.4f} | {expr[:80]}")

            if new_this_iter == 0:
                print(f"  [alpha101-mining iter {iteration}] "
                      f"No new valid factors generated.")
            else:
                print(f"  [alpha101-mining iter {iteration}] "
                      f"Generated {new_this_iter} new valid factors.")

        # Deduplicate mined factors by expression
        seen_exprs = set()
        unique_factors = []
        for f in mined_factors:
            if f.expression not in seen_exprs:
                seen_exprs.add(f.expression)
                unique_factors.append(f)

        # Combine Stage-1 (Alpha101 library top-k) + Stage-2 (LLM-mined),
        # deduping across both so a library formula that was also mined is
        # not doubled.
        combined = list(top_k_lib)
        combined_seen = {f.expression for f in combined}
        for f in unique_factors:
            if f.expression not in combined_seen:
                combined_seen.add(f.expression)
                combined.append(f)

        print(f"\n  [alpha101-mining] Complete: {len(top_k_lib)} Stage-1 "
              f"library factors + {len(unique_factors)} Stage-2 LLM-mined "
              f"= {len(combined)} total returned "
              f"(from {n_iters} iterations).")
        return combined

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
                family=_infer_family(expr),
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
        system_prompt = (
            """You are a quantitative factor research expert specializing in A-share stock selection.
Your task is to generate diverse, economically meaningful factor expressions for stock selection.

Supported data sources:
- open, high, low, close, volume, amount (price and volume data)
- pe, pb, ps, roe, market_cap, eps (fundamental data; eps = close / pe)
- return / returns (1-day daily return; alias: close.pct_change(1))
- vwap (volume-weighted average price = amount / volume)

"""
            + _FUNCTION_WHITELIST_STR
            + """

Supported operators: +, -, *, /, ^

Return a JSON object with a "factors" key, which is an array of objects.
Each object must have "expression" and "description" keys.
Ensure expressions are valid and can be evaluated by the factor engine."""
        )

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
        max_workers: int = 0,
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
            max_workers: Parallel worker count for backtesting factors each
                round (passed to ``FactorBacktester.evaluate_batch``). 0 (default)
                = auto-scale to the machine: ``min(32, os.cpu_count() + 4)``.

        Returns:
            Evolution result
        """
        # Resolve parallel worker count (0 => auto). Same rationale as Step 4c:
        # pandas C-level ops release the GIL, so threads parallelise the heavy
        # backtest; we just stop hardcoding 4 and leaving cores idle.
        n_workers = (max_workers
                     if max_workers and max_workers > 0
                     else max(1, min(32, (os.cpu_count() or 4) + 4)))

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
            metrics_list = backtester.evaluate_batch(current_factors, max_workers=n_workers, parallel=self.parallel)

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
                    val_probes, max_workers=n_workers, parallel=self.parallel)

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
                        factor.val_sharpe = vmetrics.get('sharpe', 0.0)
                    else:
                        factor.val_ic = float('nan')
                        factor.val_sharpe = float('nan')
                evaluated_factors.append(factor)

                # Only update best_ic with valid factors that have a real IC value
                if factor.is_valid and not np.isnan(factor.ic) and factor.ic > best_ic:
                    best_ic = factor.ic

            self._save_factors_to_file(current_factors, round_id, filename="improved_factors.json")

            # Select top factors — exclude parse-failed AND gate-rejected factors
            valid_evaluated = [f for f in evaluated_factors
                               if f.is_valid and f.originality_ok and not np.isnan(f.ic)]
            top_factors = _family_balanced_top(
                valid_evaluated, self.n_best_factors, key=lambda x: x.ic
            )
            
            # Generate reflection based on backtest results
            reflection_notes = self._generate_reflection(evaluated_factors, top_factors, round_id)
            print(f"\n  [evolve] Reflection notes for round {round_id + 1}:\n{reflection_notes}\n")
            
            # Generate improvements (try LLM first, fall back to rule-based)
            improved_factors = None
            if self.use_llm and self.client:
                try:
                    llm_factors = self._generate_improvements_via_llm(top_factors, round_id, reflection_notes)
                    # Acceptance floor: the LLM must return at least the
                    # configured budget (n_improve), but never more than that
                    # floor derived from half the top set. Using
                    # min(n_improve, len(top_factors)//2) keeps n_improve
                    # authoritative while still guarding against a near-empty
                    # LLM response silently degrading to rule-based.
                    _min_llm = min(self.n_improve, len(top_factors) // 2) if top_factors else 0
                    if llm_factors and len(llm_factors) >= _min_llm:
                        improved_factors = llm_factors
                        print(f"  [evolve] Generated {len(improved_factors)} improved factors via LLM")
                except Exception as e:
                    print(f"  [evolve] LLM improvement generation failed: {e}")

            if improved_factors is None:
                print(f"  [evolve] Using rule-based factor improvement.")
                improved_factors = self._generate_improvements_rule_based(top_factors)

            # Enforce the configured per-round budget. Both the LLM path (capped
            # internally at L3222) and the rule-based fallback (which emits
            # n_mutate * len(top_factors) candidates + depth/family-gap variants,
            # i.e. potentially 100+ for 20 top factors) must yield at most
            # `n_improve` improved factors per round. Without this single
            # authority, the next round evaluates an unbounded number of factors
            # and the cost compounds every iteration. This matches the config
            # contract: evolution.n_improve = "target number of improved factors
            # per round".
            if self.n_improve and len(improved_factors) > self.n_improve:
                improved_factors = improved_factors[: self.n_improve]

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
            
            # Elitism: carry top-N unmutated factors into the next round, with
            # one representative per FAMILY (see _family_balanced_top) so a
            # minority niche (e.g. Liquidity, Momentum) survives even when its
            # IC is lower than the dominant Value/Quality cluster. Without this,
            # the pool converges to a single niche and the fused portfolio
            # collapses to one effective style.
            elite_factors: List[CandidateFactor] = []
            if self.elitism_carry > 0 and top_factors:
                elite_factors = _family_balanced_top(
                    top_factors, self.elitism_carry, key=_ic_key
                )
                # Deep-copy the expressions so the elite originals stay intact
                # even if improved_factors accidentally mutate the same objects.
                elite_factors = [
                    CandidateFactor(
                        id=f"elite_r{round_id}_{f.id}",
                        expression=f.expression,
                        description=f.description,
                        parent_id=f.id,
                        generation=f.generation,
                        family=f.family,
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
            final_metrics = backtester.evaluate_batch(current_factors, max_workers=n_workers, parallel=self.parallel)
            val_final_metrics = None
            if val_mode:
                val_probes = [CandidateFactor(id=f.id, expression=f.expression,
                                          description=f.description or "")
                              for f in current_factors]
                val_final_metrics = val_backtester.evaluate_batch(
                    val_probes, max_workers=n_workers, parallel=self.parallel)
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
                        factor.val_sharpe = vmetrics.get('sharpe', 0.0)
                    else:
                        factor.val_ic = float('nan')
                        factor.val_sharpe = float('nan')
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
        # In val_mode the min_sharpe soft floor reads f.val_sharpe (the holdout
        # realized P&L) instead of the distorted train sharpe — keeping the
        # "trust the sample-out-of signal" principle consistent across the gate.
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
            # Soft Sharpe floor. In val_mode prefer the holdout val_sharpe (the
            # honest, sample-out-of realized P&L) over the distorted train sharpe.
            # Only switch when a valid val_sharpe exists — when the val split was
            # invalid it stays NaN, so we correctly fall back to train sharpe.
            # (val_mode=False leaves val_sharpe at its 0.0 default, so we must NOT
            # switch there, or every factor would be filtered at min_sharpe>0.)
            _sharpe_for_filter = f.sharpe
            if val_mode and f.val_sharpe is not None and not np.isnan(f.val_sharpe):
                _sharpe_for_filter = f.val_sharpe
            if _sharpe_for_filter < self.min_sharpe:
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
Your task is to analyze top-performing factors and generate improved, DIVERSE AND MORE ADVANCED versions.

Improvement operates on TWO tracks — generate BOTH:

[A] BREADTH track — spread coverage across factor families (keeps the population diverse):
1. Parameter tuning: Adjust window sizes (e.g., change 20 to 10 or 40)
2. Family fill: generate factors for families MISSING from your inputs
   (Momentum, Mean-reversion, Value/Quality, Volatility, Liquidity, Growth)

[B] DEPTH track (PRIORITY) — make the STRONGEST input factors MORE ADVANCED.
   "Advanced" means higher economic expressiveness, NOT merely a longer string.
   For the strongest 2-3 input factors, produce DEEPER versions that keep the same
   intent but increase sophistication. Use at least one of these techniques per
   depth factor:
   1. Crossover: combine TWO strong inputs into ONE factor, e.g.
      rank(A) * rank(B),  rank(A) - rank(B),  rank(A) + rank(-B)
   2. Nesting: stack transforms, e.g.
      rank(ts_zscore(A, w)),  ts_zscore(rank(A), w),
      rank(ts_corr(A, B, w) * ts_std(A, w))
   3. Regime conditioning: switch behavior by market state with if(cond, A, B), e.g.
      if(ts_std(returns,20) > ts_mean(ts_std(returns,60),20), A, B)
   4. Multi-horizon ensemble: blend several window lengths, e.g.
      0.5*rank(ts_delta(close,5)) + 0.5*rank(ts_delta(close,20))
   5. Fundamental + signal fusion: rank(price_signal) * rank(-roe), etc.

Supported operators (use the WHITELIST below; reuse operators FREELY when composing
deeper factors — novelty comes from COMBINATION, not from avoiding known functions):
rank, ts_rank, ts_corr, ts_cov, ts_mean, ts_std, ts_var, ts_skew, ts_kurt,
ts_min, ts_max, ts_sum, ts_delta, ts_pct_change, ts_zscore, ts_decay, delay,
sign, abs, log, sqrt, if

Factor families (for the breadth track):
- Momentum: ts_corr, ts_cov, ts_pct_change, returns-based signals
- Mean-reversion: -ts_zscore, -ts_rank, reversal of short-term moves
- Value / Quality: rank(-pe), rank(-pb), rank(roe), rank(-market_cap)
- Volatility: ts_std(returns), ts_std(close)/ts_mean(close), ts_max(returns)
- Liquidity: volume, amount, vwap-based ratios
- Growth: ts_pct_change(eps, w)

""" + _FUNCTION_WHITELIST_STR + "\n\n" + _ALLOWED_FIELDS_STR + """

Rules:
- Do NOT generate factors that are near-duplicates of each other.
- You MAY reuse operators already present in the inputs when building DEEPER composites.
  For BREADTH factors, PREFER (but do not require) at least one operator not in the
  input list.
- Complexity is WELCOMED when it increases economic expressiveness — do NOT artificially
  keep expressions simple. Avoid ONLY degenerate complexity (e.g. wrapping a constant,
  or `rank(close - open)` with no signal).
- Ensure expressions are valid and can be evaluated.
- Return a JSON object with key "factors" (a list of objects, each with
  "expression", "description", and "family" keys)."""

        # Build prompt with top factors and reflection notes
        factors_str = "\n".join([
            f"  {f.id}: {f.expression} (IC={f.ic:.4f}, Sharpe={f.sharpe:.2f}) - {f.description}"
            for f in top_factors[:self.n_best_factors]  # Only show top N to avoid prompt overflow
        ])
        # Extract functions already used by top factors, to steer the LLM toward
        # UNUSED operators (this is the main lever against "all factors look the same").
        _used_funcs = set()
        for _f in top_factors:
            _used_funcs.update(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', _f.expression))
        used_functions = ", ".join(sorted(_used_funcs)) if _used_funcs else "(none detected)"

        # --- Family-gap targeting (bridge measurement -> generation) ---
        # Recompute directly from the input factors (don't parse reflection_notes,
        # which is fragile to prompt-format drift). Mirrors the family coverage
        # the reflection reports, so the LLM is steered toward the SPECIFIC
        # families currently missing from the strongest set.
        _strong_fam = {}
        for _f in top_factors:
            _ff = getattr(_f, "family", "") or _infer_family(_f.expression)
            _strong_fam[_ff] = _strong_fam.get(_ff, 0) + 1
        _missing_fams = [fam for fam in _ALL_FAMILIES if _strong_fam.get(fam, 0) == 0]
        _present_fams = {fam: c for fam, c in _strong_fam.items() if c > 0}
        if _missing_fams:
            family_focus = (
                "PRIORITY FAMILIES to generate (currently MISSING from your inputs): "
                f"{', '.join(_missing_fams)}. Generate at least 2 NEW factors from these."
            )
        else:
            family_focus = "All families are already represented — still aim to spread evenly."
        family_avoid = ""
        if _present_fams:
            _dom = max(_present_fams, key=lambda k: _present_fams[k])
            if _present_fams[_dom] >= 2:
                family_avoid = (
                    f"AVOID piling on the dominant family '{_dom}' "
                    f"({_present_fams[_dom]}/{len(top_factors)} of your inputs) — "
                    f"do NOT generate more {_dom}-only variants."
                )
        family_targeting = ""
        if family_focus or family_avoid:
            family_targeting = (
                "Family targeting (data-driven, from your input factors' coverage):\n"
                f"{family_focus}\n"
                f"{family_avoid}\n"
            )
        
        # Build reflection section
        reflection_section = ""
        if reflection_notes:
            # Truncate to avoid prompt overflow (keep last 8192 chars)
            notes_truncated = reflection_notes[-8192:] if len(reflection_notes) > 8192 else reflection_notes
            reflection_section = f"""
=== Reflection Notes from Previous Round ===
{notes_truncated}

Use these reflection notes to guide your improvements:
- Avoid the failure patterns mentioned above
- Extend the successful factor patterns
- Consider the strategic suggestions when generating new factors
=== End of Reflection Notes ===
"""
        
        # --- Depth budget: reserve a slice of n_improve for DEEPENING the
        # strongest factors (the advancement track). The rest serves breadth.
        # This is the direct fix for "improve generates only shallow factors":
        # without an explicit depth quota, the breadth/diversity rules crowd out
        # any attempt to make the strong factors more sophisticated.
        _depth_budget = max(2, int(self.n_improve * 0.4))

        user_prompt = f"""Based on the following top-performing factors, generate {self.n_improve} IMPROVED, DIVERSE, and MORE ADVANCED factors.

Top factors (use as a starting point — especially the strongest one):
{factors_str}
{reflection_section}
Functions ALREADY used by the top factors: {used_functions}
{family_targeting}
Requirements:
1. Return a JSON object: {{"factors": [...]}} with exactly {self.n_improve} entries.
2. Each entry MUST have keys: "expression", "description", and "family"
   (one of: {', '.join(_ALL_FAMILIES)}).
3. DEPTH track (PRIORITY): at least {_depth_budget} of the factors must be DEEPER
   refinements of your single STRONGEST input factor (or the 2-3 strongest). Keep the
   same economic intent but increase expressiveness via crossover / nesting /
   if-condition / multi-horizon ensemble (see system prompt [B]). These MAY share the
   strongest factor's family and MAY reuse operators already in the inputs.
   NOTE: your strongest input may itself be a PRIOR-ROUND improvement — deepen THAT one
   specifically rather than starting from scratch.
4. BREADTH track: the remaining factors should SPREAD across families — prioritize the
   MISSING families named in the targeting block above; cover >=3 families (ideally all
   missing ones). PREFER (do not require) at least one operator NOT in the "already used"
   list for these breadth factors.
5. Vary the APPROACH across all factors: parameter tuning, crossover of two strong
   factors, fundamental fusion, nonlinear transforms, regime conditioning.
6. Ensure expressions are valid and can be evaluated.
7. Return ONLY the JSON object, no other text.

Example format (note the DEEPER composites that combine signals / nest transforms):
{{
  "factors": [
    {{"expression": "rank(ts_corr(close, volume, 20)) * rank(-ts_zscore(roe, 60))", "description": "Volume-momentum crossed with quality z-score", "family": "Momentum"}},
    {{"expression": "if(ts_std(returns,20) > ts_mean(ts_std(returns,60),20), rank(ts_delta(close,5)), -rank(ts_delta(close,5)))", "description": "Regime-conditioned short-term reversal", "family": "Mean-reversion"}},
    {{"expression": "rank(-pb) * rank(ts_mean(roe, 60))", "description": "Value-quality composite", "family": "Value/Quality"}},
    {{"expression": "rank(ts_mean(volume, 5) / ts_mean(volume, 20)) * rank(-ts_std(returns, 20))", "description": "Liquidity surge tempered by volatility", "family": "Liquidity"}}
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
                # Drop expressions referencing non-existent data fields
                # (revenue/assets/sales the LLM hallucinates) so they don't
                # waste a backtest slot.
                _ok, _bad = _validate_factor_expr(expr)
                if not _ok:
                    print(f"  [evolve] Skipping improved factor — invalid: "
                          f"{_bad}: {expr}")
                    continue
                # Find parent factor
                parent_id = top_factors[i % len(top_factors)].id if top_factors else None
                
                factor = CandidateFactor(
                    id=f"improved_llm_{parent_id}_{i}" if parent_id else f"improved_llm_{i}",
                    expression=expr,
                    description=f.get("description", f"LLM-improved factor {i}"),
                    parent_id=parent_id,
                    generation=(top_factors[0].generation + 1) if top_factors else 1,
                    family=f.get("family") or _infer_family(expr),
                )
                improved_factors.append(factor)
            
            # Lightweight diversity guard: drop factors that reuse the exact same
            # operator signature (same set of functions) as an earlier one. This is
            # the final backstop against "all improved factors look identical".
            _seen_sigs = set()
            _diverse = []
            for _f in improved_factors:
                _sig = tuple(sorted(set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', _f.expression))))
                if _sig in _seen_sigs:
                    continue
                _seen_sigs.add(_sig)
                _diverse.append(_f)
            improved_factors = _diverse

            # --- Advancement observability (makes "is improve getting deeper?"
            # measurable instead of vibes). Compare structural size of the
            # improved factors against the inputs they were derived from.
            try:
                _gate = FactorOriginalityGate(enabled=True)
                _in_nodes = [ _gate._analyze(f.expression)[3] for f in top_factors
                              if f.expression ]
                _out_nodes = [ _gate._analyze(f.expression)[3] for f in improved_factors
                               if f.expression ]
                if _in_nodes and _out_nodes:
                    _avg_in = sum(_in_nodes) / len(_in_nodes)
                    _avg_out = sum(_out_nodes) / len(_out_nodes)
                    print(f"  [evolve] Advancement: improved factors avg "
                          f"{_avg_out:.1f} AST-nodes vs inputs {_avg_in:.1f} "
                          f"(Δ={_avg_out - _avg_in:+.1f}); "
                          f"depth-quota={_depth_budget}/{self.n_improve}")
            except Exception:
                # Observability must never break generation.
                pass

            # Enforce the configured budget. The LLM prompt only *asks* for
            # n_improve factors ("exactly N entries"); some models ignore this
            # and emit many more. Without this cap the next round evaluates
            # every factor the model returned — an unbounded, recurring
            # backtest cost. Capping AFTER the diversity guard keeps n_improve
            # meaningful and the per-round factor count predictable.
            if self.n_improve and len(improved_factors) > self.n_improve:
                improved_factors = improved_factors[: self.n_improve]

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

        # --- Family gap (mirror the LLM improve path) ---
        # When the strongest set is missing entire families, we still want the
        # next round to recover those niches even with NO LLM. Computed directly
        # from the inputs (not parsed from reflection text) for robustness.
        _strong_fam = {}
        for _f in top_factors:
            _ff = getattr(_f, "family", "") or _infer_family(_f.expression)
            _strong_fam[_ff] = _strong_fam.get(_ff, 0) + 1
        _missing_fams = [fam for fam in _ALL_FAMILIES if _strong_fam.get(fam, 0) == 0]

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

            # Strategy 7: Deep crossover — parent x sibling, regulated by volatility
            deep_cross = f"rank({expr}) * rank({other_expr}) * rank(-ts_std(returns, 20))"
            all_strategies.append(deep_cross)

            # Strategy 8: Nested rank-of-ts_zscore (stacking transforms)
            nested = f"rank(ts_zscore({expr}, 20))"
            all_strategies.append(nested)

            # Strategy 9: Regime-conditioned version of the parent signal
            regime = (f"if(ts_std(returns,20) > ts_mean(ts_std(returns,60),20), "
                      f"rank({expr}), -rank({expr}))")
            all_strategies.append(regime)

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
                        family=_infer_family(improved_expr),
                    )
                    improved_factors.append(improved)

        # --- Depth track (mirror the LLM improvement): explicitly DEEPEN the
        # strongest input factor(s) so the rule-based path also produces
        # "advanced" factors, not just shallow mutations / breadth seeds. Without
        # this, the fallback only ever produced parameter tweaks and 1-op combos,
        # which is exactly the "no more advanced factors" failure mode.
        if top_factors:
            _strong = sorted(
                top_factors,
                key=lambda f: (f.ic if (f.ic is not None
                                        and not (isinstance(f.ic, float) and np.isnan(f.ic)))
                               else float('-inf')),
                reverse=True,
            )
            _s = _strong[0]
            _s_expr = _s.expression
            _s2_expr = _strong[1].expression if len(_strong) > 1 else _s_expr
            _gen = _s.generation + 1
            _depth_variants = [
                (f"rank(ts_corr({_s_expr}, {_s2_expr}, 20)) * rank(-ts_zscore(roe, 60))",
                 "Crossover of two strongest, fused with quality z-score"),
                (f"if(ts_std(returns,20) > ts_mean(ts_std(returns,60),20), "
                 f"rank({_s_expr}), -rank({_s_expr}))",
                 "Regime-conditioned version of the strongest signal"),
                (f"rank(ts_delta(close,5)) + rank(ts_delta(close,20)) + rank(ts_delta(close,60))",
                 "Multi-horizon ensemble (short + medium + long momentum)"),
                (f"rank({_s_expr}) * rank(-pb) * rank(ts_mean(roe, 60))",
                 "Strongest signal deepened with value-quality fusion"),
            ]
            for _k, (_dexpr, _ddesc) in enumerate(_depth_variants):
                improved_factors.append(CandidateFactor(
                    id=f"improved_rule_depth_{_s.id}_{_k}",
                    expression=_dexpr,
                    description=f"Depth refinement of strongest (rule-based): {_ddesc}",
                    parent_id=_s.id,
                    generation=_gen,
                    family=_infer_family(_dexpr),
                ))

        # --- Family-gap bridging pass (LLM-unavailable diversity backstop) ---
        # Seed 1 composite + 1 standalone per missing family so the next round is
        # not a monoculture. Family labels are set EXPLICITLY (not inferred)
        # because the composite may contain the parent's keywords that would
        # mis-classify it; we KNOW we are targeting `fam`, so label it as such.
        if _missing_fams and top_factors:
            _base = top_factors[0].expression
            _gen = top_factors[0].generation + 1
            _pid = top_factors[0].id
            for _fam in _missing_fams:
                _token = _FAMILY_BRIDGE_TOKEN[_fam]
                _composite = f"rank({_base}) * rank({_token})"
                improved_factors.append(CandidateFactor(
                    id=f"improved_rule_bridge_{_fam.replace('/', '_')}",
                    expression=_composite,
                    description=f"Family-gap bridge into {_fam} (rule-based)",
                    parent_id=_pid,
                    generation=_gen,
                    family=_fam,
                ))
                _standalone = f"rank({_token})"
                improved_factors.append(CandidateFactor(
                    id=f"improved_rule_seed_{_fam.replace('/', '_')}",
                    expression=_standalone,
                    description=f"Missing-family seed: {_fam} (rule-based)",
                    parent_id=None,
                    generation=_gen,
                    family=_fam,
                ))

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
