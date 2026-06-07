"""
Methods Package

This package contains the core methods for the AAAI 2027 LLM Multi-Factor
Stock Selection project:

1. debate.py: Multi-agent debate evaluator
2. evolve.py: Self-evolving factor generator
3. memory.py: Factor memory bank
4. fusion.py: Factor fusion and portfolio construction
"""

from .debate import DebateEvaluator, FactorProposal
from .evolve import SelfEvolvingGenerator, FactorBacktester
from .memory import FactorMemoryBank, MarketStateEncoder
from .fusion import FactorFusion, PortfolioConstructor

__all__ = [
    'DebateEvaluator',
    'FactorProposal',
    'SelfEvolvingGenerator',
    'FactorBacktester',
    'FactorMemoryBank',
    'MarketStateEncoder',
    'FactorFusion',
    'PortfolioConstructor',
]
