#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas-based Qlib expression evaluator.

Replaces Qlib's D.features() with pandas operations on data from the main
DataLoader. Supports the full Qlib expression grammar used by mcts-llm-alpha.

Supported operators:
  Field refs:  $close, $open, $high, $low, $volume, $vwap
  Ref(x, n):   Lag x by n periods
  Std(x, n):   Rolling std over n periods
  Mean(x, n):  Rolling mean over n periods
  Sum(x, n):   Rolling sum over n periods
  Min(x, n) / Max(x, n):   Rolling min / max
  Med(x, n):   Rolling median
  Mad(x, n):   Mean absolute deviation
  Skew(x, n) / Kurt(x, n):  Rolling skew / kurtosis
  Rank(x):     Cross-sectional rank (0-1) each day
  Corr(x, y, n):  Rolling correlation
  Cov(x, y, n):   Rolling covariance
  Delta(x, n):    x - Ref(x, n)
  Log(x), Abs(x), Sign(x), Sigmoid(x)
  Arithmetic:  +  -  *  /
  Comparison:  >  <  >=  <=  ==  !=  (return 0.0/1.0, lowest precedence)
"""

import re
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd


# ── Expression Tokenizer & Parser ──────────────────────────────────────

class ExprParser:
    """Recursive descent parser for Qlib-like expressions."""

    # Known Qlib function names for validation
    KNOWN_FUNCTIONS = {
        'ref', 'std', 'mean', 'sum', 'min', 'max', 'med', 'mad',
        'skew', 'kurt', 'rank', 'corr', 'cov', 'delta',
        'log', 'abs', 'sign', 'sigmoid',
    }

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Args:
            data: Dict mapping field names to MultiIndex (date, ticker) DataFrames
                  with a single column of values.
                  e.g. {'close': DataFrame, 'open': DataFrame, ...}
        """
        self.data = data
        self.tokens: List[str] = []
        self.pos = 0

    # ── Public API ──────────────────────────────────────────────────

    def evaluate(self, expression: str) -> pd.DataFrame:
        """Parse and evaluate a Qlib expression, returning a factor DataFrame."""
        expression = self._normalize(expression)
        self.tokens = self._tokenize(expression)
        self.pos = 0
        result = self._parse_expression()
        return result

    # ── Normalization ───────────────────────────────────────────────

    def _normalize(self, expr: str) -> str:
        """Normalize whitespace and handle edge cases."""
        expr = expr.strip()
        # Insert spaces around operators and separators (multi-char first in alternation)
        expr = re.sub(r'(>=|<=|==|!=|[+\-*/(),><])', r' \1 ', expr)
        # Collapse multiple spaces
        expr = re.sub(r'\s+', ' ', expr)
        # Fix unary minus: " - 5" after "(" becomes a negative number
        expr = re.sub(r'\(\s*-\s*(\d)', r'( -1 * \1', expr)
        # Remove trailing commas after fields: "$close ," -> "$close"
        expr = re.sub(r'(\$\w+)\s+,', r'\1', expr)
        return expr.strip()

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenize expression into a list of tokens."""
        # Split on whitespace and filter empty
        tokens = [t for t in expr.split() if t]
        return tokens

    # ── Recursive Descent ───────────────────────────────────────────

    def _peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_expression(self) -> pd.Series:
        """expression  :=  comparison ((+|-) comparison)*"""
        left = self._parse_comparison()

        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_comparison()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    def _parse_comparison(self) -> pd.Series:
        """comparison  :=  term ((>|<|>=|<=|==|!=) expression)*"""
        left = self._parse_term()

        cmp_ops = {'>', '<', '>=', '<=', '==', '!='}
        while self._peek() in cmp_ops:
            op = self._consume()
            # Parse right side as a full expression so a > b + c  =  a > (b + c)
            right = self._parse_expression()
            if op == '>':
                left = (left > right).astype(float)
            elif op == '<':
                left = (left < right).astype(float)
            elif op == '>=':
                left = (left >= right).astype(float)
            elif op == '<=':
                left = (left <= right).astype(float)
            elif op == '==':
                left = (left == right).astype(float)
            elif op == '!=':
                left = (left != right).astype(float)
        return left

    def _parse_term(self) -> pd.Series:
        """term  :=  unary ((*|/) unary)*"""
        left = self._parse_unary()

        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_unary()
            if op == '*':
                left = left * right
            else:
                # Avoid division by zero
                right_safe = right.replace(0, np.nan)
                left = left / right_safe
        return left

    def _parse_unary(self) -> pd.Series:
        """unary  :=  '-' factor  |  factor"""
        if self._peek() == '-':
            self._consume()
            # If next is a number, it's a negative literal
            if self._peek() and self._peek().replace('.', '').replace('-', '').isdigit():
                num = float(self._consume())
                return pd.Series(-num, index=self._get_index())
            return -self._parse_factor()
        return self._parse_factor()

    def _parse_factor(self) -> pd.Series:
        """factor  :=  NUMBER | $field | Function(args) | ( expression )"""
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")

        # Number literal
        if re.match(r'^-?\d+(\.\d+)?$', token):
            self._consume()
            val = float(token)
            return pd.Series(val, index=self._get_index())

        # Parenthesized expression
        if token == '(':
            self._consume()
            result = self._parse_expression()
            if self._peek() == ')':
                self._consume()
            return result

        # Field reference: $close, $open, etc.
        if token.startswith('$'):
            self._consume()
            field = token[1:]  # strip '$'
            if field not in self.data:
                raise ValueError(f"Unknown field: ${field}")
            return self.data[field]

        # Function call: FuncName( ... )
        if token.lower() not in self.KNOWN_FUNCTIONS:
            raise ValueError(
                f"Unknown token '{token}' at position {self.pos}. "
                f"This may be an unsubstituted symbolic parameter (w1, w2, ...). "
                f"Known functions: {sorted(self.KNOWN_FUNCTIONS)}"
            )
        return self._parse_function()

    def _parse_function(self) -> pd.Series:
        """Parse a function call: Name(arg1, arg2, ...)"""
        name = self._consume()
        if self._peek() != '(':
            raise ValueError(f"Expected '(' after function {name}, got {self._peek()}")
        self._consume()  # consume '('

        # Parse arguments (comma-separated expressions)
        args = []
        while True:
            if self._peek() == ')':
                self._consume()
                break
            args.append(self._parse_expression())
            if self._peek() == ',':
                self._consume()
            elif self._peek() == ')':
                self._consume()
                break

        return self._eval_function(name, args)

    # ── Function Evaluators ─────────────────────────────────────────

    def _get_index(self) -> pd.MultiIndex:
        """Get the common MultiIndex from all data fields."""
        for v in self.data.values():
            if isinstance(v, pd.Series) and isinstance(v.index, pd.MultiIndex):
                return v.index
        raise ValueError("No data available")

    @staticmethod
    def _to_scalar(val: Union[pd.Series, int, float, np.ndarray]) -> Union[int, float]:
        """Extract scalar from Series or return directly if already scalar."""
        if isinstance(val, pd.Series):
            return float(val.iloc[0]) if len(val) > 0 else 0.0
        return float(val)

    def _eval_function(self, name: str, args: List[pd.Series]) -> pd.Series:
        method = getattr(self, f'_fn_{name.lower()}', None)
        if method is None:
            raise ValueError(f"Unknown function: {name}")
        return method(*args)

    def _fn_ref(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        """Ref(x, n): shift x back by n periods per ticker."""
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').shift(n)

    def _fn_std(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).std())

    def _fn_mean(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).mean())

    def _fn_sum(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).sum())

    def _fn_min(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).min())

    def _fn_max(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).max())

    def _fn_med(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).median())

    def _fn_mad(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        def _mad(g):
            r = g.rolling(n, min_periods=max(2, n//2))
            return r.apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
        return x.groupby(level='instrument').transform(_mad)

    def _fn_skew(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).skew())

    def _fn_kurt(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        return x.groupby(level='instrument').transform(lambda g: g.rolling(n, min_periods=max(2, n//2)).kurt())

    def _fn_rank(self, x: pd.Series, *args) -> pd.Series:
        """Rank(x): cross-sectional rank (0-1) within each day.
        In Qlib, Rank(x, n) is actually Rank(x) -- the second arg is unused in some versions.
        We use rank(pct=True) to match Qlib behavior.
        """
        return x.groupby(level='datetime').rank(pct=True)

    def _fn_corr(self, x: pd.Series, y: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        result_parts = []
        if isinstance(x.index, pd.MultiIndex):
            for inst in x.index.get_level_values('instrument').unique():
                xi = x.xs(inst, level='instrument')
                yi = y.xs(inst, level='instrument')
                corr = xi.rolling(n, min_periods=max(2, n//2)).corr(yi)
                corr.name = inst
                result_parts.append(corr)
        if result_parts:
            return pd.concat(result_parts)
        return pd.Series(np.nan, index=x.index)

    def _fn_cov(self, x: pd.Series, y: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        n = int(self._to_scalar(n))
        result_parts = []
        if isinstance(x.index, pd.MultiIndex):
            for inst in x.index.get_level_values('instrument').unique():
                xi = x.xs(inst, level='instrument')
                yi = y.xs(inst, level='instrument')
                cov = xi.rolling(n, min_periods=max(2, n//2)).cov(yi)
                cov.name = inst
                result_parts.append(cov)
        if result_parts:
            return pd.concat(result_parts)
        return pd.Series(np.nan, index=x.index)

    def _fn_delta(self, x: pd.Series, n: Union[pd.Series, int, float]) -> pd.Series:
        """Delta(x, n) = x - Ref(x, n)"""
        n = int(self._to_scalar(n))
        return x - self._fn_ref(x, n)

    def _fn_log(self, x: pd.Series) -> pd.Series:
        x_safe = x.clip(lower=1e-10)
        return np.log(x_safe)

    def _fn_abs(self, x: pd.Series) -> pd.Series:
        return x.abs()

    def _fn_sign(self, x: pd.Series) -> pd.Series:
        return np.sign(x)

    def _fn_sigmoid(self, x: pd.Series) -> pd.Series:
        # Sigmoid: 1 / (1 + exp(-x))
        x_clipped = np.clip(x, -50, 50)  # prevent overflow
        return 1.0 / (1.0 + np.exp(-x_clipped))


# ── Main Data Bridge ──────────────────────────────────────────────────

def convert_to_multindex(
    price_data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.Series]:
    """
    Convert main DataLoader's price_data dict of (dates × stocks) DataFrames
    into MultiIndex (datetime, instrument) Series suitable for the expression parser.

    Note: Uses 'datetime' / 'instrument' level names to match Qlib convention
    used by the metrics module.

    Args:
        price_data: Dict with keys 'open','high','low','close','volume' etc.
                    Each value is a DataFrame with DatetimeIndex rows and stock code columns.

    Returns:
        Dict mapping field names to MultiIndex Series.
    """
    result = {}
    for field, df in price_data.items():
        if isinstance(df, pd.DataFrame):
            stacked = df.stack()
            stacked.index = stacked.index.set_names(['datetime', 'instrument'])
            result[field] = stacked
    return result


def compute_future_returns(
    price_data_midx: Dict[str, pd.Series],
    forward_period: int = 10,
) -> pd.Series:
    """
    Compute future N-day forward returns.

    For forward_period=1: Ref($close, -1) / $close - 1 (1-day return)
    For forward_period=N: close[t+N] / close[t] - 1 (N-day return)

    Args:
        price_data_midx: MultiIndex price data dict
        forward_period: Number of trading days to look ahead (default: 10)

    Returns:
        Series of forward returns, aligned to current date.
    """
    close = price_data_midx['close']
    future_close = close.groupby(level='instrument').shift(-forward_period)
    returns = future_close / close - 1
    returns.name = 'return'
    return returns


def evaluate_formula_pandas(
    formula: str,
    price_data_midx: Dict[str, pd.Series],
    return_series: pd.Series,
    repo_factors: List[pd.DataFrame],
    start_date: str,
    end_date: str,
    split_date: Optional[str] = None,
    ic_method: str = "spearman",
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    """
    Evaluate a Qlib formula using pandas operations on main DataLoader data.

    This is a drop-in replacement for qlib_evaluator.evaluate_formula_qlib().

    Args:
        formula: Qlib expression string
        price_data_midx: MultiIndex data from convert_to_multindex()
        return_series: Future return series from compute_future_returns()
        repo_factors: List of existing factor DataFrames for diversity calc
        start_date, end_date: Evaluation date range
        split_date: IS/OOS split date for overfitting
        ic_method: "spearman" or "pearson"

    Returns:
        (raw_scores_dict, factor_dataframe)
    """
    from .metrics import calc_effectiveness, calc_stability, calc_turnover
    from .metrics import calc_diversity, calc_overfitting

    try:
        # 1. Parse and evaluate the formula
        print(f"[PandasEval] Evaluating: {formula[:80]}...")
        parser = ExprParser(price_data_midx)
        factor_series = parser.evaluate(formula)

        # 2. Clip to date range
        if start_date:
            factor_series = factor_series[factor_series.index.get_level_values('datetime') >= start_date]
        if end_date:
            factor_series = factor_series[factor_series.index.get_level_values('datetime') <= end_date]

        factor_series.name = 'factor'
        factor_df = factor_series.to_frame()

        # 3. Align with returns
        aligned = pd.concat([factor_df, return_series], axis=1).dropna()

        if len(aligned) < 100:
            print(f"[PandasEval] Insufficient data points: {len(aligned)}")
            return None, None

        # 4. Determine split date if not given
        if split_date is None:
            dates = sorted(aligned.index.get_level_values('datetime').unique())
            split_idx = int(len(dates) * 0.7)
            split_date = str(dates[split_idx])

        # 5. Calculate 5 metrics
        effectiveness = calc_effectiveness(aligned, ic_method)
        stability = calc_stability(aligned, ic_method)
        turnover = calc_turnover(factor_df)
        diversity = calc_diversity(factor_df, repo_factors)
        overfitting = calc_overfitting(aligned, split_date, ic_method)

        raw_scores = {
            "IC": effectiveness,
            "IR": stability,
            "Turnover": turnover,
            "Diversity": diversity,
            "Overfitting": overfitting,
        }

        print(f"[PandasEval] IC={raw_scores['IC']:.4f}, ICIR={raw_scores['IR']:.4f}, "
              f"Turnover={raw_scores['Turnover']:.4f}, "
              f"Diversity={raw_scores['Diversity']:.4f}, "
              f"Overfitting={raw_scores['Overfitting']:.4f}")

        return raw_scores, factor_df

    except Exception as e:
        print(f"[PandasEval] Error evaluating '{formula[:60]}...': {e}")
        import traceback
        traceback.print_exc()
        return None, None
