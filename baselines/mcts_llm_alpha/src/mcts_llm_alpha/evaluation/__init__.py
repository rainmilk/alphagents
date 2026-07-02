#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估模块 - 提供多维度的alpha因子评估功能。

该模块包含了所有评估相关的功能，包括：
- 基于Qlib的真实数据评估
- 相对排名系统
- 多维度综合评估
- 各种性能指标计算
"""

from .relative_ranking import RelativeRankingEvaluator
from .cache import FormulaEvaluationCache, with_cache
from .pandas_evaluator import evaluate_formula_pandas

# Optional Qlib evaluator
_qlib_error = None
try:
    from .qlib_evaluator import evaluate_formula_qlib, evaluate_formula_simple
except ImportError:
    _qlib_error = 'qlib not available'
    evaluate_formula_qlib = None
    evaluate_formula_simple = None

# Optional comprehensive evaluator
try:
    from .comprehensive import ComprehensiveEvaluator, create_evaluator
except ImportError:
    ComprehensiveEvaluator = None
    create_evaluator = None

__all__ = [
    'ComprehensiveEvaluator', 'create_evaluator',
    'evaluate_formula_qlib', 'evaluate_formula_simple',
    'evaluate_formula_pandas',
    'RelativeRankingEvaluator',
    'FormulaEvaluationCache', 'with_cache',
]