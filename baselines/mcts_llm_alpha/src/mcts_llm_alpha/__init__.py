"""MCTS-LLM Alpha挖掘主包。"""

__version__ = "0.1.0"

# Core config and data imports (no external deps)
from .config import load_config, Config
from .data import create_data_provider, MarketDataManager

# Lazy imports for optional dependencies
_llm_import_error = None
try:
    from .llm import LLMClient
except ImportError as e:
    _llm_import_error = str(e)
    LLMClient = None

_formula_import_error = None
try:
    from .formula import sanitize_formula, fix_missing_params
except ImportError as e:
    _formula_import_error = str(e)
    sanitize_formula = None
    fix_missing_params = None

_eval_import_error = None
try:
    from .evaluation import evaluate_formula_qlib, evaluate_formula_simple
except ImportError as e:
    _eval_import_error = str(e)
    evaluate_formula_qlib = None
    evaluate_formula_simple = None

# MCTS (only depends on numpy + stdlib, always safe)
from .mcts import MCTSSearch, MCTSNode, FrequentSubtreeMiner

__all__ = [
    "__version__",
    "load_config", "Config",
    "create_data_provider", "MarketDataManager",
    "LLMClient",
    "MCTSSearch", "MCTSNode", "FrequentSubtreeMiner",
    "sanitize_formula", "fix_missing_params",
    "evaluate_formula_qlib", "evaluate_formula_simple",
]