# -*- coding: utf-8 -*-
"""
AlphaGen Baseline Runner — RL-Based Token-Based Factor Generation
====================================================================

AlphaGen implements automatic alpha factor generation via token-based
expression trees with three generation strategies:

  1. 'random'    — Random sampling with grammar constraints (no dependencies)
  2. 'reinforce' — REINFORCE + MLP policy (Option B, default, needs torch)
  3. 'ppo'       — MaskablePPO + LSTM policy (Option C, needs torch + sb3-contrib)

Core methodology:
  1. Token vocabulary — 63 discrete actions: Feature(6) + Operator(31)
     + Constant(15) + DeltaTime(10) + SEP(1)
  2. Expression Builder — Inverse Polish Notation (postfix) expression
     tree with grammar-rules-driven action masking
  3. Factor Generation —
     [random]    Random sampling of valid token sequences
     [reinforce] REINFORCE policy gradient with MLP (embedding→MLP→logits)
     [ppo]       MaskablePPO with LSTMSharedNet (embedding→LSTM→mean pool)
  4. Factor Evaluation — Cross-sectional Rank IC / ICIR on training data
  5. AlphaPool — Factor pool with mutual-IC dedup (>0.99 threshold),
     ensemble weight optimization via gradient descent (L1-regularized
     maximize-IC-minimize-correlation objective)
  6. Portfolio Construction — Top-N stocks by ensemble factor value,
     equal-weight long-only
  7. Backtest — Unified BacktestEngine (commission=0.001, slippage=0.0)

For RL methods, the AlphaPool is populated during training: the RL agent
generates expressions, each is evaluated and potentially added to the pool,
and the ensemble IC serves as the reward signal.

References:
  - baselines/AlphaForge/train_RL.py               (PPO training loop)
  - baselines/AlphaForge/exp_RL_calc_result.ipynb    (result evaluation)
  - baselines/AlphaForge/alphagen/data/expression.py (operators & features)
  - baselines/AlphaForge/alphagen/data/tree.py       (ExpressionBuilder)
  - baselines/AlphaForge/alphagen/models/alpha_pool.py (AlphaPool)
  - baselines/AlphaForge/alphagen/rl/env/core.py     (AlphaEnvCore)
  - baselines/AlphaForge/alphagen/rl/policy.py       (LSTMSharedNet)
  - baselines/rl_alphagen.py                         (RL training modules)

Author: Code Review Expert (火眼眼)
Date: 2026-07-03
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods.portfolio_utils import allocate_score_proportional, allocate_portfolio_weights


# ═══════════════════════════════════════════════════════════════════════
#  Section 1: Token Vocabulary & Expression Tree
#  (adapted from alphagen/data/tokens.py, expression.py, tree.py)
# ═══════════════════════════════════════════════════════════════════════

# ── 1.1 Feature definitions ───────────────────────────────────────────

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'vwap']

# ── 1.2 Operator definitions ──────────────────────────────────────────

class ExprNode:
    """Base class for all expression tree nodes."""
    _id_counter = itertools.count()

    def __init__(self):
        self._id = next(ExprNode._id_counter)
        self.is_featured = False

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate this expression node on data dictionaries.

        data: dict of {feature_name: np.ndarray of shape (n_dates, n_stocks)}
        Returns: np.ndarray of shape (n_dates, n_stocks)
        """
        raise NotImplementedError

    def n_args(self) -> int:
        """Number of child operands this node takes (0 for terminals)."""
        return 0

    def __repr__(self):
        return self.__class__.__name__

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id == getattr(other, '_id', None)


class Feature(ExprNode):
    """Terminal node: raw feature like $close, $volume."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.is_featured = True

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return data[self.name].copy()

    def __repr__(self):
        return f"Feature({self.name})"


class Constant(ExprNode):
    """Terminal node: numeric constant."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        return np.full_like(data['close'], self.value)

    def __repr__(self):
        return f"Const({self.value})"


class DeltaTime(ExprNode):
    """Auxiliary node: time offset (consumed by RollingOperator)."""
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        raise RuntimeError("DeltaTime should not be evaluated directly")

    def __repr__(self):
        return f"Δt({self.d})"


# ── 1.3 Operator nodes ────────────────────────────────────────────────

class UnaryOperator(ExprNode):
    """Operator taking 1 operand."""
    def __init__(self, child: ExprNode):
        super().__init__()
        self.child = child
        self.is_featured = child.is_featured

    def n_args(self): return 1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.child!r})"


class BinaryOperator(ExprNode):
    """Operator taking 2 operands."""
    def __init__(self, left: ExprNode, right: ExprNode):
        super().__init__()
        self.left = left
        self.right = right
        self.is_featured = left.is_featured or right.is_featured

    def n_args(self): return 2

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left!r}, {self.right!r})"


class RollingOperator(ExprNode):
    """Operator taking (child_expression, DeltaTime) — e.g. ts_mean($close, 20).

    In the stack: [featured_expr, DeltaTime] → pop → reversed → [featured_expr, DeltaTime]
    child = featured_expr, dt = DeltaTime
    """
    def __init__(self, child: ExprNode, dt: DeltaTime):
        super().__init__()
        self.child = child
        self.dt = dt
        self.is_featured = child.is_featured

    def n_args(self): return 2

    def __repr__(self):
        return f"{self.__class__.__name__}({self.child!r}, {self.dt!r})"


class PairRollingOperator(ExprNode):
    """Operator taking (left, right, DeltaTime) — e.g. ts_corr($close, $volume, 20).

    In the stack: [featured_left, featured_right, DeltaTime] → pop → reversed
    → [featured_left, featured_right, DeltaTime]
    """
    def __init__(self, left: ExprNode, right: ExprNode, dt: DeltaTime):
        super().__init__()
        self.left = left
        self.right = right
        self.dt = dt
        self.is_featured = left.is_featured or right.is_featured

    def n_args(self): return 3

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left!r}, {self.right!r}, {self.dt!r})"


# Concrete operators
EPS = 1e-8

class OpAbs(UnaryOperator):
    def evaluate(self, data): return np.abs(self.child.evaluate(data))

class OpSign(UnaryOperator):
    def evaluate(self, data): return np.sign(self.child.evaluate(data))

class OpLog(UnaryOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        return np.log(np.maximum(x, EPS))

class OpInv(UnaryOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        return np.where(np.abs(x) > EPS, 1.0 / x, 0.0)

class OpSLog1p(UnaryOperator):
    def evaluate(self, data): return np.sign(self.child.evaluate(data)) * np.log1p(np.abs(self.child.evaluate(data)))

class OpCSRank(UnaryOperator):
    """Cross-sectional rank (0~1)."""
    def evaluate(self, data):
        x = self.child.evaluate(data)
        # Rank each day's cross-section, converting to [0,1]
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            day_vals = x[i]
            valid = ~np.isnan(day_vals)
            if valid.sum() > 1:
                result[i, valid] = (pd.Series(day_vals[valid]).rank() - 1) / (valid.sum() - 1)
        return result


class OpAdd(BinaryOperator):
    def evaluate(self, data): return self.left.evaluate(data) + self.right.evaluate(data)

class OpSub(BinaryOperator):
    def evaluate(self, data): return self.left.evaluate(data) - self.right.evaluate(data)

class OpMul(BinaryOperator):
    def evaluate(self, data): return self.left.evaluate(data) * self.right.evaluate(data)

class OpDiv(BinaryOperator):
    def evaluate(self, data):
        b = self.right.evaluate(data)
        return self.left.evaluate(data) * np.where(np.abs(b) > EPS, 1.0 / b, 0.0)

class OpPow(BinaryOperator):
    def evaluate(self, data):
        a = self.left.evaluate(data)
        b = self.right.evaluate(data)
        # Limit exponent to avoid overflow
        b_clipped = np.clip(b, -5, 5)
        return np.sign(a) * np.power(np.abs(a) + EPS, b_clipped)

class OpGreater(BinaryOperator):
    def evaluate(self, data):
        return (self.left.evaluate(data) > self.right.evaluate(data)).astype(float)

class OpLess(BinaryOperator):
    def evaluate(self, data):
        return (self.left.evaluate(data) < self.right.evaluate(data)).astype(float)


# ── Rolling operators ─────────────────────────────────────────────────

def _rolling(x: np.ndarray, d: int, fn: Callable) -> np.ndarray:
    """Apply fn over a rolling window of size d across axis=0 (time)."""
    result = np.full_like(x, np.nan)
    if d <= 0 or d > x.shape[0]:
        return result
    for t in range(d - 1, x.shape[0]):
        window = x[t - d + 1: t + 1]
        result[t] = fn(window, axis=0)
    return result


class OpRef(RollingOperator):
    """Reference: value d days ago."""
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        result[d:] = x[:-d]
        return result


class OpTsMean(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanmean(w, axis=axis))


class OpTsSum(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nansum(w, axis=axis))


class OpTsStd(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanstd(w, axis=axis))


class OpTsVar(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanvar(w, axis=axis))


class OpTsMax(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanmax(w, axis=axis))


class OpTsMin(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanmin(w, axis=axis))


class OpTsMed(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanmedian(w, axis=axis))


class OpTsMad(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        return _rolling(x, d, lambda w, axis: np.nanmedian(np.abs(w - np.nanmedian(w, axis=axis, keepdims=True)), axis=axis))


class OpTsDelta(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        result[d:] = x[d:] - x[:-d]
        return result


class OpTsDiv(RollingOperator):
    def evaluate(self, data):
        # x[t] / x[t-d]
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        shifted = np.full_like(x, np.nan)
        shifted[d:] = x[:-d]
        denom = np.maximum(np.abs(shifted), EPS)
        result[d:] = x[d:] / denom[d:]
        return result


class OpTsPctChange(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        shifted = np.full_like(x, np.nan)
        shifted[d:] = x[:-d]
        denom = np.maximum(np.abs(shifted), EPS)
        result[d:] = (x[d:] - shifted[d:]) / denom[d:]
        return result


class OpTsWma(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        result = np.full_like(x, np.nan)
        for t in range(d - 1, x.shape[0]):
            window = x[t - d + 1: t + 1]
            result[t] = np.average(window, weights=weights, axis=0)
        return result


class OpTsEma(RollingOperator):
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        alpha = 2.0 / (d + 1.0)
        result = np.full_like(x, np.nan)
        # Initialize with mean of first d days
        if d > 0 and x.shape[0] > 0:
            result[d - 1] = np.nanmean(x[:d], axis=0)
            for t in range(d, x.shape[0]):
                prev = result[t - 1]
                curr = x[t]
                mask = ~np.isnan(curr)
                prev_ok = ~np.isnan(prev)
                result[t] = np.where(mask & prev_ok, alpha * curr + (1 - alpha) * prev,
                                     np.where(mask, curr, prev))
        return result


class OpTsRank(RollingOperator):
    """Rolling cross-sectional rank."""
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        for t in range(d - 1, x.shape[0]):
            day_vals = x[t]
            valid = ~np.isnan(day_vals)
            if valid.sum() > 1:
                result[t, valid] = (pd.Series(day_vals[valid]).rank() - 1) / (valid.sum() - 1)
        return result


class OpTsIR(RollingOperator):
    """Rolling Information Ratio = mean / std over window."""
    def evaluate(self, data):
        x = self.child.evaluate(data)
        d = self.dt.d
        result = np.full_like(x, np.nan)
        for t in range(d - 1, x.shape[0]):
            w = x[t - d + 1: t + 1]
            mu = np.nanmean(w, axis=0)
            sd = np.nanstd(w, axis=0)
            safe_sd = np.maximum(sd, EPS)
            result[t] = mu / safe_sd
        return result


# ── Pair rolling operators ────────────────────────────────────────────

class OpTsCov(PairRollingOperator):
    def evaluate(self, data):
        lv = self.left.evaluate(data)
        rv = self.right.evaluate(data)
        d = self.dt.d
        result = np.full_like(lv, np.nan)
        for t in range(d - 1, lv.shape[0]):
            lw = lv[t - d + 1: t + 1]
            rw = rv[t - d + 1: t + 1]
            lm = np.nanmean(lw, axis=0)
            rm = np.nanmean(rw, axis=0)
            result[t] = np.nanmean((lw - lm) * (rw - rm), axis=0)
        return result


class OpTsCorr(PairRollingOperator):
    def evaluate(self, data):
        lv = self.left.evaluate(data)
        rv = self.right.evaluate(data)
        d = self.dt.d
        result = np.full_like(lv, np.nan)
        for t in range(d - 1, lv.shape[0]):
            lw = lv[t - d + 1: t + 1]
            rw = rv[t - d + 1: t + 1]
            lm = np.nanmean(lw, axis=0)
            rm = np.nanmean(rw, axis=0)
            ld = lw - lm
            rd = rw - rm
            cov = np.nanmean(ld * rd, axis=0)
            ls = np.sqrt(np.maximum(np.nanmean(ld * ld, axis=0), EPS))
            rs = np.sqrt(np.maximum(np.nanmean(rd * rd, axis=0), EPS))
            result[t] = cov / np.maximum(ls * rs, EPS)
        return result


# ── 1.4 Operator registry ─────────────────────────────────────────────

# All operators available for RL action space
OPERATOR_CLASSES = {
    # Unary
    'Abs': (OpAbs, 1, 'unary'),
    'Sign': (OpSign, 1, 'unary'),
    'Log': (OpLog, 1, 'unary'),
    'Inv': (OpInv, 1, 'unary'),
    'S_log1p': (OpSLog1p, 1, 'unary'),
    'CSRank': (OpCSRank, 1, 'unary'),
    # Binary
    'Add': (OpAdd, 2, 'binary'),
    'Sub': (OpSub, 2, 'binary'),
    'Mul': (OpMul, 2, 'binary'),
    'Div': (OpDiv, 2, 'binary'),
    'Pow': (OpPow, 2, 'binary'),
    'Greater': (OpGreater, 2, 'binary'),
    'Less': (OpLess, 2, 'binary'),
    # Rolling
    'Ref': (OpRef, 2, 'rolling'),
    'ts_mean': (OpTsMean, 2, 'rolling'),
    'ts_sum': (OpTsSum, 2, 'rolling'),
    'ts_std': (OpTsStd, 2, 'rolling'),
    'ts_var': (OpTsVar, 2, 'rolling'),
    'ts_max': (OpTsMax, 2, 'rolling'),
    'ts_min': (OpTsMin, 2, 'rolling'),
    'ts_med': (OpTsMed, 2, 'rolling'),
    'ts_mad': (OpTsMad, 2, 'rolling'),
    'ts_delta': (OpTsDelta, 2, 'rolling'),
    'ts_div': (OpTsDiv, 2, 'rolling'),
    'ts_pctchange': (OpTsPctChange, 2, 'rolling'),
    'ts_wma': (OpTsWma, 2, 'rolling'),
    'ts_ema': (OpTsEma, 2, 'rolling'),
    'ts_rank': (OpTsRank, 2, 'rolling'),
    'ts_ir': (OpTsIR, 2, 'rolling'),
    # Pair rolling
    'ts_cov': (OpTsCov, 3, 'pair_rolling'),
    'ts_corr': (OpTsCorr, 3, 'pair_rolling'),
}

OP_NAMES = list(OPERATOR_CLASSES.keys())

# ── 1.5 Constants & DeltaTimes ────────────────────────────────────────

CONSTANTS = [-30, -20, -10, -5, -2, -1, 0, 0.01, 0.1, 1, 2, 5, 10, 20, 30]
DELTA_TIMES = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60]

# ── 1.6 Token IDs (for logic, not for neural network) ─────────────────

# We don't use integer action IDs like the RL version.
# Instead we use a token object system with explicit types.

class TokenType:
    NULL = 'NULL'
    FEATURE = 'FEATURE'
    OPERATOR = 'OPERATOR'
    CONSTANT = 'CONSTANT'
    DELTA_TIME = 'DELTA_TIME'
    SEP = 'SEP'  # End-of-expression marker

# Max expression length
MAX_EXPR_LENGTH = 20

# ── 1.7 Expression Builder ────────────────────────────────────────────

class ExpressionBuilder:
    """
    Build expression trees from token sequences using Inverse Polish Notation.

    This mirrors alphagen/data/tree.py ExpressionBuilder but simplified.
    The stack holds (ExprNode, has_delta_time_flag) pairs for grammar validation.
    """

    def __init__(self):
        self.stack: List[ExprNode] = []
        self.tokens_used = 0
        self._last_token_type = None

    def reset(self):
        self.stack = []
        self.tokens_used = 0
        self._last_token_type = None

    def add_feature(self, name: str) -> bool:
        """Add a feature token. Returns True if valid.

        Grammar (matching original AlphaGen validate_feature):
        Feature cannot follow a DeltaTime on the stack.
        DeltaTime should only be consumed as the last arg of a rolling op.
        """
        if self.tokens_used >= MAX_EXPR_LENGTH:
            return False
        if self.stack and isinstance(self.stack[-1], DeltaTime):
            return False
        self.stack.append(Feature(name))
        self.tokens_used += 1
        self._last_token_type = TokenType.FEATURE
        return True

    def add_constant(self, value: float) -> bool:
        """Add a constant token. Returns True if valid.

        Grammar (matching original AlphaGen validate_const):
        Constant is only valid when:
        - Stack is empty (first token), OR
        - Top of stack is featured (preparing second operand for binary op)

        This prevents Constant chains (Constant after Constant) which
        create dead-end states where the stack can never be reduced.
        """
        if self.tokens_used >= MAX_EXPR_LENGTH:
            return False
        if self.stack and not self.stack[-1].is_featured:
            return False
        self.stack.append(Constant(value))
        self.tokens_used += 1
        self._last_token_type = TokenType.CONSTANT
        return True

    def add_operator(self, op_name: str) -> bool:
        """Add an operator token. Pops operands from stack. Returns True if valid.

        Grammar rules (matching original AlphaGen validate_op):
        - Unary:   single child must be featured (not Constant, not DeltaTime)
        - Binary:  at least ONE child must be featured; neither can be DeltaTime
                   (allows Constant as one operand, e.g. Add($close, 1))
        - Rolling: [featured_expr, DeltaTime] — top=DeltaTime, below=featured
        - PairRoll: [featured_left, featured_right, DeltaTime] — both must be featured

        IMPORTANT: Validation happens BEFORE popping, matching original AlphaGen.
        If validation fails, the stack is left unmodified.
        """
        if self.tokens_used >= MAX_EXPR_LENGTH:
            return False

        op_cls, n_args, op_type = OPERATOR_CLASSES[op_name]

        if len(self.stack) < n_args:
            return False

        # ── Validate BEFORE popping (matching original AlphaGen validate_op) ──
        # This prevents stack corruption when validation fails.
        if op_type == 'unary':
            # n_args == 1; top must be featured and not DeltaTime
            child = self.stack[-1]
            if not child.is_featured or isinstance(child, DeltaTime):
                return False
        elif op_type == 'binary':
            # n_args == 2; at least one must be featured, neither can be DeltaTime
            a, b = self.stack[-2], self.stack[-1]
            if isinstance(a, DeltaTime) or isinstance(b, DeltaTime):
                return False
            if not (a.is_featured or b.is_featured):
                return False
        elif op_type == 'rolling':
            # n_args == 2; [featured_expr, DeltaTime]
            # top must be DeltaTime, below must be featured
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif op_type == 'pair_rolling':
            # n_args == 3; [featured_left, featured_right, DeltaTime]
            # top must be DeltaTime, both below must be featured
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            return False

        # ── Validation passed — now safe to pop and construct ──
        children = [self.stack.pop() for _ in range(n_args)]
        children = list(reversed(children))  # left-to-right order

        # Construct the node
        if n_args == 1:
            node = op_cls(children[0])
        elif n_args == 2:
            node = op_cls(children[0], children[1])
        elif n_args == 3:
            node = op_cls(children[0], children[1], children[2])
        else:
            return False

        self.stack.append(node)
        self.tokens_used += 1
        self._last_token_type = TokenType.OPERATOR
        return True

    def add_delta_time(self, d: int) -> bool:
        """Add a DeltaTime token. Top of stack must be featured.

        Grammar (matching original AlphaGen validate_dt):
        DeltaTime can only follow a featured expression on top of the stack.
        After adding DeltaTime, the top becomes non-featured, preventing
        consecutive DeltaTime additions naturally.
        """
        if self.tokens_used >= MAX_EXPR_LENGTH:
            return False
        if not self.stack or not self.stack[-1].is_featured:
            return False
        self.stack.append(DeltaTime(d))
        self.tokens_used += 1
        self._last_token_type = TokenType.DELTA_TIME
        return True

    def is_finished(self) -> bool:
        """Expression is valid and complete (single node, is_featured)."""
        return len(self.stack) == 1 and self.stack[0].is_featured

    def get_tree(self) -> Optional[ExprNode]:
        """Get the final expression tree if valid."""
        if self.is_finished():
            return self.stack[0]
        return None

    def valid_action_types(self) -> Dict[str, bool]:
        """
        Return which token types are grammatically valid for the next step.
        Mirrors the action masking logic in alphagen/rl/env/wrapper.py.

        Grammar rules (matching original AlphaGen):
        - Feature:  valid unless top is DeltaTime
        - Constant: valid when stack empty OR top is featured (prevents Constant chains)
        - Operator: unary needs 1 featured on top; binary needs ≥1 featured in top-2
        - DeltaTime: valid after a featured expression (no DeltaTime already on stack)
        - SEP:       valid when exactly 1 featured node on stack
        """
        n = len(self.stack)
        result = {
            'FEATURE': False, 'OPERATOR': False, 'CONSTANT': False,
            'DELTA_TIME': False, 'SEP': False,
        }

        # SEP: valid when we have exactly 1 featured node on stack
        if n == 1 and self.stack[0].is_featured:
            result['SEP'] = True

        if self.tokens_used >= MAX_EXPR_LENGTH:
            return result

        if n == 0:
            # Start of expression: only Feature or Constant
            result['FEATURE'] = True
            result['CONSTANT'] = True
            return result

        top = self.stack[-1]

        # Feature: cannot follow DeltaTime
        if not isinstance(top, DeltaTime):
            result['FEATURE'] = True

        # Constant: valid when stack empty or top is featured
        # (prevents Constant-after-Constant dead-end chains)
        if top.is_featured:
            result['CONSTANT'] = True

        # Unary operators: need 1 featured child on top (not DeltaTime)
        if top.is_featured and not isinstance(top, DeltaTime):
            result['OPERATOR'] = True

        # Binary operators: need ≥1 featured in top-2, neither can be DeltaTime
        if n >= 2:
            a, b = self.stack[-2], self.stack[-1]
            if not isinstance(a, DeltaTime) and not isinstance(b, DeltaTime):
                if a.is_featured or b.is_featured:
                    result['OPERATOR'] = True

        # Rolling operators: need [featured_expr, DeltaTime] on stack
        if n >= 2:
            child, dt = self.stack[-2], self.stack[-1]
            if child.is_featured and isinstance(dt, DeltaTime):
                result['OPERATOR'] = True

        # Pair rolling operators: need [featured_left, featured_right, DeltaTime]
        if n >= 3:
            a, b, dt = self.stack[-3], self.stack[-2], self.stack[-1]
            if a.is_featured and b.is_featured and isinstance(dt, DeltaTime):
                result['OPERATOR'] = True

        # DeltaTime: can add when top is featured (matching original validate_dt)
        # After adding, top becomes DeltaTime (not featured), preventing chains
        if top.is_featured:
            result['DELTA_TIME'] = True

        return result

    def get_valid_ops(self) -> List[str]:
        """Get list of operator names valid for current state."""
        if not self.valid_action_types().get('OPERATOR', False):
            return []
        n = len(self.stack)
        valid = []

        # Check unary: top must be featured and not DeltaTime
        if n >= 1:
            top = self.stack[-1]
            if top.is_featured and not isinstance(top, DeltaTime):
                valid.extend(name for name, (_, n_args, op_type) in OPERATOR_CLASSES.items()
                             if op_type == 'unary')

        # Check binary: ≥1 featured in top-2, neither can be DeltaTime
        if n >= 2:
            a, b = self.stack[-2], self.stack[-1]
            if not isinstance(a, DeltaTime) and not isinstance(b, DeltaTime):
                if a.is_featured or b.is_featured:
                    valid.extend(name for name, (_, n_args, op_type) in OPERATOR_CLASSES.items()
                                 if op_type == 'binary')

        # Check rolling operators: need [featured_expr, DeltaTime] on stack
        # (top = DeltaTime, below = featured child)
        if n >= 2:
            child, dt = self.stack[-2], self.stack[-1]
            if child.is_featured and isinstance(dt, DeltaTime):
                valid.extend(name for name, (_, n_args, op_type) in OPERATOR_CLASSES.items()
                             if op_type == 'rolling' and n_args == 2)

        # Check pair rolling: need [featured_left, featured_right, DeltaTime]
        # (top = DeltaTime, below = 2 featured children)
        if n >= 3:
            a, b, dt = self.stack[-3], self.stack[-2], self.stack[-1]
            if (a.is_featured and b.is_featured and isinstance(dt, DeltaTime)):
                valid.extend(name for name, (_, n_args, op_type) in OPERATOR_CLASSES.items()
                             if op_type == 'pair_rolling')

        return valid


# ═══════════════════════════════════════════════════════════════════════
#  Section 2: Factor Generation & Evaluation
# ═══════════════════════════════════════════════════════════════════════

def _normalize_cross_section(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score normalization (e.g., MAD-winsorize then standardize)."""
    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        row = x[i].copy()
        # MAD winsorize: cap at ±3*MAD
        med = np.nanmedian(row)
        mad = np.nanmedian(np.abs(row - med))
        if mad > EPS:
            upper = med + 3 * mad * 1.4826  # 1.4826 = scaling to std
            lower = med - 3 * mad * 1.4826
            row = np.clip(row, lower, upper)
        # Z-score
        mu = np.nanmean(row)
        sig = np.nanstd(row)
        if sig > EPS:
            result[i] = (row - mu) / sig
    return result


def _compute_forward_returns(close: pd.DataFrame, periods: List[int] = None)    -> Dict[int, pd.DataFrame]:
    """Compute forward returns for different holding periods."""
    if periods is None:
        periods = [1, 5, 20]
    result = {}
    for p in periods:
        fwd = close.shift(-p) / close - 1.0
        result[p] = fwd
    return result


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Rank IC (Spearman correlation) between two 1D arrays.
    
    Zero-computation failure is possible with synthetic data having degenerate ranks
    (e.g. when stocks simply trend upward). Returns NaN when computation fails,
    which callers must handle with `np.nan_to_num` or similar.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return np.nan
    try:
        a_valid = pd.Series(a[mask]).rank()
        b_valid = pd.Series(b[mask]).rank()
        return a_valid.corr(b_valid)  # pd.Series.corr handles zero-variance properly
    except Exception:
        return np.nan


def evaluate_factor(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    debug: bool = False,
) -> Dict:
    """
    Evaluate a single factor on training data.

    Computes:
    - Mean Rank IC (average daily cross-sectional Spearman correlation)
    - ICIR = mean(IC) / std(IC)
    - Factor Sharpe = IC mean / IC std  (same as ICIR for daily IC)
    - IC positive ratio

    Args:
        factor_values: (n_dates, n_stocks) factor values
        forward_returns: (n_dates, n_stocks) forward returns (N-day, typically 10)
        debug: If True, print evaluation details

    Returns:
        Dict with mean_ic, std_ic, icir, factor_sharpe, ic_positive_ratio, nb_points
    """
    ics = []
    for t in range(factor_values.shape[0]):
        ic = _rank_ic(factor_values[t], forward_returns[t])
        if not np.isnan(ic):
            ics.append(ic)

    if len(ics) < 5:
        return {'mean_ic': 0.0, 'std_ic': 0.0, 'icir': 0.0,
                'factor_sharpe': 0.0, 'ic_positive_ratio': 0.0,
                'nb_points': len(ics)}

    ics = np.array(ics)
    mean_ic = float(np.nanmean(ics))
    std_ic = float(np.nanstd(ics))
    icir = mean_ic / std_ic if std_ic > EPS else 0.0
    ic_pos = float((ics > 0).mean())

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir,
        'factor_sharpe': icir,
        'ic_positive_ratio': ic_pos,
        'nb_points': len(ics),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 3: AlphaPool — Factor Pool & Ensemble Optimization
#  (adapted from alphagen/models/alpha_pool.py)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlphaPool:
    """
    Factor pool that stores generated expressions and optimizes ensemble weights.

    Mirrors AlphaForge's AlphaPoolBase / AlphaPool, but uses numpy instead of
    PyTorch tensors for the baseline implementation.

    Key behaviors:
    - Mutual IC check: reject factors too similar to existing ones (>0.99)
    - Gradient descent ensemble weight optimization
    - L1 regularization for sparse weights
    - Capacity cap: remove lowest-weight factor when full
    """
    capacity: int = 20
    mutual_ic_threshold: float = 0.99
    alpha_l1: float = 5e-3
    lr: float = 5e-4
    n_iter: int = 500
    max_backtrack_history: int = 100  # so we can compute rolling-ops up to a window

    # Internal state
    exprs: List[ExprNode] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)    # (n_dates, n_stocks) per factor
    single_ics: List[Dict] = field(default_factory=list)
    mutual_ics: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    best_ic_ret: float = 0.0
    _size: int = 0

    @property
    def size(self) -> int:
        return self._size

    def _compute_mutual_ic(self, new_value: np.ndarray) -> Optional[np.ndarray]:
        """Compute mutual IC between new factor and all existing factors.
        
        Returns None only when computation fails (zero-variance edge-case).
        A degenerate value_repr that yields all-NaN ICs will produce an empty list;
        we treat that as "no meaningful correlation" and return an all-zero vector.
        """
        mutuals = []
        for exist_val in self.values:
            ics = []
            for t in range(new_value.shape[0]):
                ic = _rank_ic(new_value[t], exist_val[t])
                if not np.isnan(ic):
                    ics.append(ic)
            if len(ics) > 5:
                mutuals.append(np.nanmean(ics))
            else:
                # Not enough valid ICs to assess — assume independent
                mutuals.append(0.0)
        if not mutuals:
            return np.array([0.0], dtype=float)
        return np.array(mutuals)

    def try_add_factor(
        self, expr: ExprNode, value: np.ndarray, metrics: Dict,
    ) -> Tuple[bool, float]:
        """
        Try to add a new factor to the pool.

        Returns (accepted, ensemble_ic).
        ensemble_ic = 0.0 if rejected (mutual IC too high).
        """
        # ── Mutual IC check ──
        if self.size > 0:
            mut_ic = self._compute_mutual_ic(value)
            if mut_ic is not None and np.any(np.abs(mut_ic) > self.mutual_ic_threshold):
                return False, 0.0

        # ── Add factor ──
        self.exprs.append(expr)
        self.values.append(value)
        self.single_ics.append(metrics)
        self._size += 1

        # Resize mutual_ics matrix
        old_mut = self.mutual_ics
        n = self._size
        self.mutual_ics = np.zeros((n, n))
        if n > 1:
            self.mutual_ics[:n-1, :n-1] = old_mut
            # Last column/row
            if mut_ic is not None:
                self.mutual_ics[:n-1, n-1] = mut_ic
                self.mutual_ics[n-1, :n-1] = mut_ic

        # ── Optimize ensemble weights ──
        self._optimize_weights()

        # ── Pop if over capacity ──
        if self._size > self.capacity:
            # Remove factor with smallest weight
            min_idx = np.argmin(np.abs(self.weights))
            self._remove_factor(min_idx)

        # ── Return ensemble IC ──
        ensemble_ic = self._evaluate_ensemble_ic()
        self.best_ic_ret = max(self.best_ic_ret, ensemble_ic)
        return True, ensemble_ic

    def _remove_factor(self, idx: int):
        """Remove factor at index."""
        self.exprs.pop(idx)
        self.values.pop(idx)
        self.single_ics.pop(idx)
        self._size -= 1
        # Shrink mutual_ics
        mask = np.ones(self.mutual_ics.shape[0], dtype=bool)
        mask[idx] = False
        self.mutual_ics = self.mutual_ics[mask][:, mask]
        self.weights = np.delete(self.weights, idx)

    def _optimize_weights(self):
        """
        Optimize ensemble weights via gradient descent.

        Objective: minimize mutual_ic_sum - 2 * ret_ic_sum + 1 + alpha * L1
        Equivalent to maximize IC of weighted sum while minimizing mutual IC.
        """
        n = self._size
        if n == 0:
            self.weights = np.array([])
            return
        if n == 1:
            self.weights = np.array([1.0])
            return

        # Extract single ICs
        ic_ret = np.array([m.get('mean_ic', 0.0) for m in self.single_ics])
        ic_mut = self.mutual_ics.copy()

        # Initialize weights
        weights = np.ones(n) / n
        best_weights = weights.copy()
        best_loss = float('inf')

        for it in range(self.n_iter):
            # Forward: ret_ic_sum = Σ w_i * ic_i, mut_ic_sum = Σ w_i * w_j * ic_ij
            ret_sum = np.dot(weights, ic_ret)
            mut_sum = weights @ ic_mut @ weights
            loss = mut_sum - 2 * ret_sum + 1 + self.alpha_l1 * np.sum(np.abs(weights))

            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()

            # Gradient
            grad_mut = 2 * ic_mut @ weights
            grad_ret = -2 * ic_ret
            grad_l1 = self.alpha_l1 * np.sign(weights)
            gradient = grad_mut + grad_ret + grad_l1

            # Update
            weights = weights - self.lr * gradient
            # Project to non-negative, re-normalize
            weights = np.maximum(weights, 0)
            w_sum = weights.sum()
            if w_sum > EPS:
                weights /= w_sum

            if np.linalg.norm(gradient) < 1e-6:
                break

        self.weights = best_weights

    def _evaluate_ensemble_ic(self, eval_values: Optional[List[np.ndarray]] = None,
                               eval_returns: Optional[np.ndarray] = None) -> float:
        """Compute ensemble IC using current weights."""
        if self._size == 0:
            return 0.0

        if eval_values is None:
            vals = self.values
        else:
            vals = eval_values

        ensemble = np.zeros_like(vals[0])
        for i in range(self._size):
            ensemble += self.weights[i] * vals[i]

        # Average daily IC
        ics = []
        if eval_returns is None:
            # Use factor's own IC values (train set proxy)
            return np.dot(self.weights, [m.get('mean_ic', 0.0) for m in self.single_ics])

        for t in range(ensemble.shape[0]):
            ic = _rank_ic(ensemble[t], eval_returns[t])
            if not np.isnan(ic):
                ics.append(ic)
        if ics:
            return float(np.nanmean(ics))
        return 0.0

    def get_ensemble_value(self) -> Optional[np.ndarray]:
        """Get ensemble factor values."""
        if self._size == 0:
            return None
        result = np.zeros_like(self.values[0])
        for i in range(self._size):
            result += self.weights[i] * self.values[i]
        return result


# ═══════════════════════════════════════════════════════════════════════
#  Section 4: Factor Generation — Random Sampling + Grammar
#  (simplified from RL PPO training)
# ═══════════════════════════════════════════════════════════════════════

def _generate_random_factor(
    builder: ExpressionBuilder,
    seed: int = -1,
    min_tokens: int = 3,
) -> Optional[ExprNode]:
    """
    Generate one random valid factor by sampling tokens with grammar rules.

    This is the simplified version of the RL policy: instead of using a trained
    neural network to select actions, we randomly sample from valid actions.

    A minimum-token budget encourages building complex factors rather than
    trivial single-feature expressions.

    Args:
        builder: ExpressionBuilder to use (will be reset)
        seed: Random seed (-1 for no seeding)
        min_tokens: Minimum tokens to use before allowing SEP

    Returns:
        ExprNode if a valid expression was generated, None otherwise.
    """
    if seed >= 0:
        np.random.seed(seed)

    builder.reset()

    for step in range(MAX_EXPR_LENGTH):
        valid = builder.valid_action_types()

        # Choose action type from valid ones, with complexity bias:
        # - Before min_tokens: prefer building actions (avoid SEP when we can keep building)
        # - After min_tokens or when stuck: consider SEP normally
        available = [t for t, v in valid.items() if v]
        if not available:
            return None

        # Bias: if SEP is valid but we haven't reached min_tokens, reduce its probability
        if valid.get('SEP') and builder.tokens_used < min_tokens and len(available) > 1:
            # Allow SEP but with 1/3 probability compared to other actions
            weights = [0.33 if t == 'SEP' else 1.0 for t in available]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            action_type = np.random.choice(available, p=probs)
        else:
            action_type = np.random.choice(available)

        if action_type == 'SEP':
            tree = builder.get_tree()
            if tree is not None:
                return tree
            continue

        elif action_type == 'FEATURE':
            builder.add_feature(np.random.choice(FEATURES))

        elif action_type == 'CONSTANT':
            builder.add_constant(np.random.choice(CONSTANTS))

        elif action_type == 'OPERATOR':
            valid_ops = builder.get_valid_ops()
            if not valid_ops:
                continue
            builder.add_operator(np.random.choice(valid_ops))

        elif action_type == 'DELTA_TIME':
            builder.add_delta_time(np.random.choice(DELTA_TIMES))

    return None


def generate_factor_batch(
    n_factors: int,
    seed: int = 42,
    max_attempts_per_factor: int = 100,
    verbose: bool = False,
) -> List[ExprNode]:
    """
    Generate a batch of valid random factors.

    Args:
        n_factors: Number of factors to generate
        seed: Random seed
        max_attempts_per_factor: Max random restarts per factor
        verbose: Print progress

    Returns:
        List of generated ExprNode objects
    """
    rng = np.random.RandomState(seed)
    builder = ExpressionBuilder()
    factors = []

    for i in range(n_factors):
        for attempt in range(max_attempts_per_factor):
            # Use a different seed for each attempt
            factor = _generate_random_factor(builder, seed=rng.randint(0, 2**31-1))
            if factor is not None:
                factors.append(factor)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    Generated {len(factors)}/{n_factors} factors...")
                break
        else:
            if verbose:
                print(f"    Warning: failed to generate factor {i+1} after {max_attempts_per_factor} attempts")

    return factors


# ═══════════════════════════════════════════════════════════════════════
#  Section 5: Portfolio Construction
# ═══════════════════════════════════════════════════════════════════════

def build_portfolios_from_ensemble(
    ensemble_values: np.ndarray,
    close: pd.DataFrame,
    top_n: int = 50,
    test_start_date: Optional[str] = None,
    portfolio_method: str = "score_proportional",
) -> pd.DataFrame:
    """
    Build equal-weight long-only portfolios from ensemble factor values.

    Args:
        ensemble_values: (n_dates, n_stocks) numpy array of factor values
        close: (n_dates, n_stocks) DataFrame of close prices
        top_n: Number of stocks to hold
        test_start_date: If given, only build portfolios from this date onward

    Returns:
        pd.DataFrame: (n_dates, n_stocks) portfolio weights (rows sum to 1)
    """
    portfolios = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    stocks = close.columns.tolist()

    if test_start_date is not None:
        start_idx = close.index.get_loc(test_start_date)
    else:
        start_idx = 0

    for t in range(max(start_idx, 1), close.shape[0]):
        day_values = ensemble_values[t]
        day_prices = close.iloc[t].values

        # Rank by factor value (higher = better)
        mask = ~np.isnan(day_values) & ~np.isnan(day_prices) & (day_prices > EPS)
        valid_idx = np.where(mask)[0]

        if len(valid_idx) <= top_n:
            selected = valid_idx
        else:
            # Top-N by factor value
            valid_vals = day_values[valid_idx]
            top_local = np.argsort(valid_vals)[-top_n:]
            selected = valid_idx[top_local]

        # MASE-consistent: score-proportional weights (not equal-weight)
        if len(selected) > 0:
            sel_codes = portfolios.columns[selected]
            sel_scores = pd.Series(day_values[selected], index=sel_codes)
            w = allocate_portfolio_weights(sel_scores, method=portfolio_method)
            portfolios.iloc[t, selected] = w.values

    return portfolios


# ═══════════════════════════════════════════════════════════════════════
#  Section 6: Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AlphaGenConfig:
    """Configuration for AlphaGen baseline."""
    n_generate: int = 500           # Number of random factors to generate
    pool_capacity: int = 20         # AlphaPool capacity
    top_n_stocks: int = 50          # Portfolio size
    holding_period: int = 1         # Rebalance frequency
    mutual_ic_threshold: float = 0.99
    seed: int = 42                  # Random seed
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  Section 7: Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def run_alphagen_baseline(
    config_path: str = "config/config.yaml",
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    universe: Optional[str] = None,
    n_generate: int = 300,
    pool_capacity: int = 20,
    top_n_stocks: int = 50,
    holding_period: Optional[int] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
    # ── RL method selection ──
    rl_method: str = 'reinforce',
    # REINFORCE params (Option B)
    n_episodes: int = 2000,
    reinforce_lr: float = 1e-3,
    reinforce_d_model: int = 64,
    # PPO params (Option C)
    n_timesteps: int = 100_000,
    ppo_d_model: int = 128,
    ppo_n_layers: int = 2,
    # Common RL params
    gamma: float = 1.0,
    ent_coef: float = 0.01,
    device: str = 'cpu',
    forward_period: Optional[int] = None,  # None -> config['evolution']['forward_period'] (10)
    portfolio_method: str = "score_proportional",
) -> Dict:
    """
    Run AlphaGen baseline — RL-based token factor generation.

    Supports three factor generation methods:
      - 'random'    : Random sampling with grammar constraints (no RL, no torch)
      - 'reinforce' : REINFORCE + MLP policy (Option B, default, needs torch)
      - 'ppo'       : MaskablePPO + LSTM policy (Option C, needs torch + sb3-contrib)

    Pipeline:
      1. Load A-share data via main DataLoader
      2. [random] Generate random factor expressions
         [reinforce] Train REINFORCE policy to generate factors
         [ppo] Train MaskablePPO + LSTM policy to generate factors
      3. AlphaPool: deduplicate and optimize ensemble weights
         (populated during RL training for reinforce/ppo)
      4. Build portfolios from ensemble factor on test data
      5. Backtest with unified BacktestEngine

    Args:
        config_path: Path to main project config
        train_start_date: Data start date (YYYY-MM-DD)
        test_end_date: Data end date (YYYY-MM-DD)
        universe: Stock universe
        train_end_date: Training end date
        test_start_date: Test start date
        n_generate: Number of random factors (only used when rl_method='random')
        pool_capacity: AlphaPool capacity
        top_n_stocks: Number of stocks in portfolio
        holding_period: Rebalance frequency
        seed: Random seed
        output_dir: Directory for saving results
        rl_method: Factor generation method ('random', 'reinforce', 'ppo')
        n_episodes: REINFORCE training episodes (Option B)
        reinforce_lr: REINFORCE learning rate
        reinforce_d_model: REINFORCE embedding dimension
        n_timesteps: PPO training timesteps (Option C)
        ppo_d_model: PPO LSTM dimension
        ppo_n_layers: PPO LSTM layers
        gamma: Discount factor (1.0 = no discount, same as original AlphaGen)
        ent_coef: Entropy coefficient for exploration
        device: 'cpu' or 'cuda' for RL training
        forward_period: Forward return period in days for IC evaluation
            (default 10, matching other baselines for fair comparison)

    Returns:
        Dict with performance metrics and factor info
    """
    from dataloader.loader import DataLoader
    from backtest.engine import BacktestEngine

    print("=" * 60)
    print("  AlphaGen Baseline — RL-Based Token Factor Generation")
    print(f"  Method: {rl_method.upper()}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)

    # ── Resolve forward_period / holding_period from config ──────────
    # explicit arg > config.yaml > default, so standalone runs also honor config.
    _ev_cfg = loader.config.get('evolution', {})
    _bt_cfg = loader.config.get('backtest', {}).get('trading', {})
    if not forward_period or forward_period <= 0:
        forward_period = _ev_cfg.get('forward_period', 10)
    if not holding_period or holding_period <= 0:
        holding_period = _bt_cfg.get('holding_period', 1)
    train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')
    bundle = loader.load_data(universe=universe, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    price_data, fundamental_data, industry_data = bundle.full
    train_price, train_fund, train_ind = bundle.train
    test_price, test_fund, test_ind = bundle.test

    close = price_data['close']
    n_dates = len(close.index)
    n_stocks = len(close.columns)
    print(f"  Loaded: {n_dates} dates x {n_stocks} stocks")

    # ── Step 2: Train/test split (centralized in DatasetBundle) ─────────
    print(f"  Train end: {train_end}, Test start: {test_start}")

    # ── Step 3: Prepare numpy data dict ────────────────────────────────
    print("\n[Step 2] Preparing data for expression evaluation...")
    train_data = {
        'close': train_price['close'].values.astype(np.float64),
        'open': train_price['open'].values.astype(np.float64),
        'high': train_price['high'].values.astype(np.float64),
        'low': train_price['low'].values.astype(np.float64),
        'volume': train_price['volume'].values.astype(np.float64),
        'vwap': (train_price['high'].values + train_price['low'].values + train_price['close'].values) / 3.0,
    }
    test_data = {
        'close': test_price['close'].values.astype(np.float64),
        'open': test_price['open'].values.astype(np.float64),
        'high': test_price['high'].values.astype(np.float64),
        'low': test_price['low'].values.astype(np.float64),
        'volume': test_price['volume'].values.astype(np.float64),
        'vwap': (test_price['high'].values + test_price['low'].values + test_price['close'].values) / 3.0,
    }

    # Compute forward returns SEPARATELY on the train and test close slices
    # (NOT on bundle.full). Computing on the full series would let the LAST
    # `forward_period` TRAINING days peek into TEST prices via shift(-period)
    # crossing the train/test boundary — a look-ahead leak in the training
    # target. Computing per-slice keeps each target strictly inside its own
    # window; the last `forward_period` rows naturally become NaN and are
    # dropped during training/eval.
    train_fwd_ret = _compute_forward_returns(
        train_price['close'], periods=[forward_period]
    )[forward_period].values.astype(np.float64)
    test_fwd_ret = _compute_forward_returns(
        test_price['close'], periods=[forward_period]
    )[forward_period].values.astype(np.float64)

    n_train_dates = train_fwd_ret.shape[0]
    print(f"  Train: {n_train_dates} dates, Test: {test_fwd_ret.shape[0]} dates")

    # ── Step 3: Factor generation & AlphaPool ──────────────────────────
    #   random    : generate batch → evaluate → add to pool
    #   reinforce : REINFORCE + MLP policy trains and populates pool
    #   ppo       : MaskablePPO + LSTM policy trains and populates pool
    
    rl_training_stats = {}
    
    if rl_method == 'random':
        # ── Random factor generation (original approach) ──
        print(f"\n[Step 3] Generating {n_generate} random factor expressions...")
        t0 = time.time()
        factors = generate_factor_batch(n_generate, seed=seed, verbose=True)
        elapsed = time.time() - t0
        print(f"  Generated {len(factors)} valid factors in {elapsed:.1f}s")

        if len(factors) == 0:
            print("  ERROR: No valid factors generated!")
            return {'error': 'no_factors_generated'}

        # ── Evaluate factors on training data ──
        print(f"\n[Step 4] Evaluating factors on training data...")
        eval_results = []
        for i, expr in enumerate(factors):
            try:
                raw_value = expr.evaluate(train_data)
                normalized = _normalize_cross_section(raw_value)
                metrics = evaluate_factor(normalized, train_fwd_ret)
                metrics['expr_idx'] = i
                metrics['expr_repr'] = repr(expr)
                eval_results.append((expr, raw_value, normalized, metrics))
            except Exception as e:
                if i < 5:
                    print(f"    Factor {i} eval error: {e}")
                continue

        if not eval_results:
            print("  ERROR: All factors failed evaluation!")
            return {'error': 'all_factors_failed'}

        eval_results.sort(key=lambda x: abs(x[3]['icir']), reverse=True)

        print(f"  Evaluated {len(eval_results)} factors successfully")
        print(f"\n  Top 10 factors by |ICIR|:")
        for rank_i, (expr, raw_val, norm_val, metrics) in enumerate(eval_results[:10]):
            print(f"    {rank_i+1:2d}. {repr(expr):60s}  IC={metrics['mean_ic']:+.4f}  ICIR={metrics['icir']:+.4f}")

        # ── Build AlphaPool ──
        print(f"\n[Step 5] Building AlphaPool (capacity={pool_capacity})...")
        pool = AlphaPool(capacity=pool_capacity)

        n_accepted = 0
        for expr, raw_val, norm_val, metrics in eval_results:
            if isinstance(norm_val, np.ndarray):
                val = norm_val.astype(np.float64)
            else:
                val = norm_val.values.astype(np.float64)

            accepted, ensemble_ic = pool.try_add_factor(expr, val, metrics)
            if accepted:
                n_accepted += 1
                if n_accepted % 5 == 0:
                    print(f"    Pool: {pool.size}/{pool_capacity} accepted, "
                          f"best_ic={pool.best_ic_ret:.4f}")

        print(f"  Final pool: {pool.size} factors, best ensemble IC={pool.best_ic_ret:.4f}")

    elif rl_method in ('reinforce', 'ppo'):
        # ── RL-based factor generation ──
        from rl_alphagen import train_rl_factors
        
        print(f"\n[Step 3] Training RL policy ({rl_method}) to generate factors...")
        print(f"  Pool capacity: {pool_capacity}")
        pool = AlphaPool(capacity=pool_capacity)
        
        pool, rl_training_stats = train_rl_factors(
            method=rl_method,
            train_data=train_data,
            train_fwd_ret=train_fwd_ret,
            pool=pool,
            min_tokens=3,
            n_episodes=n_episodes,
            reinforce_lr=reinforce_lr,
            reinforce_d_model=reinforce_d_model,
            n_timesteps=n_timesteps,
            ppo_d_model=ppo_d_model,
            ppo_n_layers=ppo_n_layers,
            gamma=gamma,
            ent_coef=ent_coef,
            device=device,
            verbose=True,
        )
        
        print(f"\n  Final pool: {pool.size} factors, best ensemble IC={pool.best_ic_ret:.4f}")
        
        # For compatibility with downstream code
        factors = pool.exprs if pool.size > 0 else []
    
    else:
        print(f"  ERROR: Unknown rl_method '{rl_method}'. Use 'random', 'reinforce', or 'ppo'.")
        return {'error': 'invalid_method'}

    if pool.size == 0:
        print("  ERROR: No factors accepted into pool!")
        return {'error': 'pool_empty'}

    # ── Step 7: Ensemble factor on test data ───────────────────────────
    print(f"\n[Step 6] Computing ensemble factor on test data...")
    test_norm_values = []
    for i in range(pool.size):
        raw_val = pool.exprs[i].evaluate(test_data)
        norm_val = _normalize_cross_section(raw_val)
        if isinstance(norm_val, np.ndarray):
            test_norm_values.append(norm_val.astype(np.float64))
        else:
            test_norm_values.append(norm_val.values.astype(np.float64))

    # Build ensemble
    ensemble_test = np.zeros_like(test_norm_values[0])
    for i in range(pool.size):
        ensemble_test += pool.weights[i] * test_norm_values[i]

    # Compute test IC for ensemble
    test_ic = []
    for t in range(ensemble_test.shape[0]):
        ic = _rank_ic(ensemble_test[t], test_fwd_ret[t])
        if not np.isnan(ic):
            test_ic.append(ic)
    test_ic_mean = float(np.nanmean(test_ic)) if test_ic else 0.0
    test_ic_std = float(np.nanstd(test_ic)) if test_ic else 0.0
    test_icir = test_ic_mean / test_ic_std if test_ic_std > EPS else 0.0

    print(f"  Test IC: mean={test_ic_mean:.4f}, std={test_ic_std:.4f}, ICIR={test_icir:.4f}")

    # ── Step 8: Build portfolios ───────────────────────────────────────
    print(f"\n[Step 7] Building top-{top_n_stocks} portfolios...")
    # Reconstruct full close timeline for portfolio building
    portfolios = build_portfolios_from_ensemble(
        ensemble_test, test_price['close'], top_n=top_n_stocks,
        portfolio_method=portfolio_method,
    )
    print(f"  Portfolios: {portfolios.shape}")

    if portfolios.sum().sum() < EPS:
        print("  ERROR: Empty portfolio!")
        return {'error': 'empty_portfolio'}

    # ── Step 9: Backtest ───────────────────────────────────────────────
    print(f"\n[Step 8] Running backtest...")
    prices_aligned = close.loc[portfolios.index].reindex(columns=portfolios.columns)
    engine = BacktestEngine(
        commission=0.001, slippage=0.0, risk_free_rate=0.0,
        holding_period=holding_period if holding_period is not None else 1,
    )
    run_dir = None
    method_name = "alphagen"
    if output_dir:
        _u = universe or loader.data_config.get('universe', {}).get('index', 'hs300')
        _s = train_start_date or loader.data_config.get('train_start_date', 'na')
        _e = test_end_date or loader.data_config.get('test_end_date', 'na')
        _fp = forward_period if forward_period is not None else 10
        _hp = holding_period if holding_period is not None else 1
        param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
        run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
        os.makedirs(run_dir, exist_ok=True)
    _bm = prices_aligned.pct_change().shift(-1).mean(axis=1).dropna()
    _bm.name = 'benchmark_return'
    backtest_metrics = engine.run(portfolios, prices_aligned, benchmark_returns=_bm, save_dir=run_dir)

    # ── Step 10: Assemble results ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AlphaGen Baseline Complete")
    print("=" * 60)
    print(f"  Method:            {rl_method}")
    print(f"  Factors generated:  {len(factors)}")
    print(f"  Pool size:          {pool.size}/{pool_capacity}")
    print(f"  Best ensemble IC:   {pool.best_ic_ret:.4f}")
    print(f"  Test Rank IC:       {test_ic_mean:.4f}")
    print(f"  Test ICIR:          {test_icir:.4f}")
    print(f"  Forward Period:     {forward_period}d")
    print(f"  Total Return:       {backtest_metrics.get('total_return', 0):.4f}")
    print(f"  Annual Return:      {backtest_metrics.get('annual_return', 0):.4f}")
    print(f"  Sharpe Ratio:       {backtest_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  Max Drawdown:       {backtest_metrics.get('max_drawdown', 0):.4f}")

    results = {
        'rl_method': rl_method,
        'annual_return': backtest_metrics.get('annual_return', 0.0),
        'sharpe_ratio': backtest_metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': backtest_metrics.get('max_drawdown', 0.0),
        'information_ratio': backtest_metrics.get('information_ratio', 0.0),
        'calmar_ratio': backtest_metrics.get('calmar_ratio', 0.0),
        'win_rate': backtest_metrics.get('win_rate', 0.0),
        'avg_turnover': backtest_metrics.get('avg_turnover', 0.0),
        'n_factors': len(factors),
        'pool_size': pool.size,
        'pool_capacity': pool_capacity,
        'best_ensemble_ic': float(pool.best_ic_ret),
        'test_mean_rank_ic': test_ic_mean,
        'test_icir': test_icir,
        'train_end': train_end,
        'test_start': test_start,
        'forward_period': forward_period,
        'train_start': train_start,
        'test_end': test_end,
        'holding_period': holding_period,
        'pool_weights': pool.weights.tolist() if pool.size > 0 else [],
        'pool_exprs': [repr(e) for e in pool.exprs],
    }
    
    # Include RL training stats if available
    if rl_training_stats:
        results['rl_training'] = {
            'method': rl_training_stats.get('method', rl_method),
            'final_pool_size': rl_training_stats.get('final_pool_size', 0),
            'best_ic': rl_training_stats.get('best_ic', 0.0),
            'n_factors_added': rl_training_stats.get('n_factors_added', 0),
            'training_time': rl_training_stats.get('training_time', 0.0),
        }
        if 'n_episodes' in rl_training_stats:
            results['rl_training']['n_episodes'] = rl_training_stats['n_episodes']
        if 'n_timesteps' in rl_training_stats:
            results['rl_training']['n_timesteps'] = rl_training_stats['n_timesteps']

    # ── Save results ───────────────────────────────────────────────────
    if output_dir:
        result_path = os.path.join(run_dir, 'results_alphagen.json')
        # Convert numpy types for JSON
        json_results = {}
        for k, v in results.items():
            if isinstance(v, (np.floating,)):
                json_results[k] = float(v)
            elif isinstance(v, (np.integer,)):
                json_results[k] = int(v)
            elif isinstance(v, list):
                json_results[k] = v
            else:
                json_results[k] = v
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {result_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run AlphaGen baseline — RL-based token factor generation'
    )
    parser.add_argument('--config', default='config/config.yaml', help='Path to main config')
    parser.add_argument('--train-start', default=None, help='Data start date')
    parser.add_argument('--test-end', default=None, help='Data end date')
    parser.add_argument('--universe', default=None, help='Stock universe')
    parser.add_argument('--train-end', default=None, help='Train end date')
    parser.add_argument('--test-start', default=None, help='Test start date')
    parser.add_argument('--n-generate', type=int, default=300, help='Factors to generate (random only)')
    parser.add_argument('--pool-capacity', type=int, default=20, help='Pool capacity')
    parser.add_argument('--top-n', type=int, default=50, help='Portfolio size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', default='experiments/alphagen', help='Output directory')
    
    # ── RL method selection ──
    parser.add_argument('--rl-method', default='reinforce',
                        choices=['random', 'reinforce', 'ppo'],
                        help='Factor generation method (default: reinforce)')
    
    # REINFORCE params (Option B)
    parser.add_argument('--n-episodes', type=int, default=2000,
                        help='REINFORCE training episodes (Option B)')
    parser.add_argument('--reinforce-lr', type=float, default=1e-3,
                        help='REINFORCE learning rate')
    parser.add_argument('--reinforce-d-model', type=int, default=64,
                        help='REINFORCE embedding dimension')
    
    # PPO params (Option C)
    parser.add_argument('--n-timesteps', type=int, default=100000,
                        help='PPO training timesteps (Option C)')
    parser.add_argument('--ppo-d-model', type=int, default=128,
                        help='PPO LSTM dimension')
    parser.add_argument('--ppo-n-layers', type=int, default=2,
                        help='PPO LSTM layers')
    
    # Common RL params
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor (1.0 = no discount)')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--device', default='cuda',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device for RL training')
    parser.add_argument('--forward-period', type=int, default=None,
                        help='Forward return period in days for IC evaluation '
                             '(default: config evolution.forward_period, 10)')
    parser.add_argument('--holding-period', type=int, default=None,
                        help='Rebalance frequency in days (1=daily, 5=weekly, 20=monthly). '
                             'Defaults to AlphaGenConfig.holding_period (1).')

    args = parser.parse_args()

    results = run_alphagen_baseline(
        config_path=args.config,
        universe=args.universe,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        n_generate=args.n_generate,
        pool_capacity=args.pool_capacity,
        top_n_stocks=args.top_n,
        holding_period=args.holding_period,
        seed=args.seed,
        output_dir=args.output_dir,
        rl_method=args.rl_method,
        n_episodes=args.n_episodes,
        reinforce_lr=args.reinforce_lr,
        reinforce_d_model=args.reinforce_d_model,
        n_timesteps=args.n_timesteps,
        ppo_d_model=args.ppo_d_model,
        ppo_n_layers=args.ppo_n_layers,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device=args.device,
        forward_period=args.forward_period,
    )

    print("\nFinal results:")
    for k, v in results.items():
        if k not in ('pool_weights', 'pool_exprs', 'rl_training'):
            print(f"  {k}: {v}")
        elif k == 'rl_training' and v:
            print(f"  rl_training:")
            for rk, rv in v.items():
                print(f"    {rk}: {rv}")
