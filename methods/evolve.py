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
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


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
    
    
class FactorBacktester:
    """
    Lightweight backtester for factor evaluation.
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize backtester.
        
        Args:
            prices: DataFrame of stock prices
        """
        self.prices = prices
        
    def evaluate(self, factor: CandidateFactor) -> Dict:
        """
        Evaluate a factor.
        
        Args:
            factor: Factor to evaluate
            
        Returns:
            Dict of evaluation metrics
        """
        # Mock evaluation (in practice, calculate actual metrics)
        ic = np.random.uniform(0.02, 0.06)
        sharpe = np.random.uniform(0.5, 2.0)
        win_rate = np.random.uniform(0.5, 0.6)
        
        return {
            'ic': ic,
            'sharpe': sharpe,
            'win_rate': win_rate,
        }


class SelfEvolvingGenerator:
    """
    Self-evolving factor generator.
    
    Generates factors through iterative evolution:
    1. Generate seed factors
    2. Backtest and evaluate
    3. Reflect on failures
    4. Generate improvements
    5. Repeat until convergence
    """
    
    def __init__(
        self,
        llm_model: str = "deepseek-chat",
        n_seeds: int = 20,
    ):
        """
        Initialize self-evolving generator.
        
        Args:
            llm_model: LLM model to use
            n_seeds: Number of seed factors
        """
        self.llm_model = llm_model
        self.n_seeds = n_seeds
        
    def generate_seed_factors(self, n_factors: int = 20) -> List[CandidateFactor]:
        """
        Generate seed factors using LLM.
        
        Args:
            n_factors: Number of factors to generate
            
        Returns:
            List of seed factors
        """
        print(f"Generating {n_factors} seed factors...")
        
        seed_factors = []
        for i in range(n_factors):
            factor = CandidateFactor(
                id=f"seed_{i}",
                expression=f"rank(ts_corr(close, volume, {10 + i}))",
                description=f"Time series correlation factor {i}",
                generation=0,
            )
            seed_factors.append(factor)
        
        print(f"Generated {len(seed_factors)} seed factors")
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
            evaluated_factors = []
            for factor in current_factors:
                metrics = backtester.evaluate(factor)
                factor.ic = metrics['ic']
                factor.sharpe = metrics['sharpe']
                factor.win_rate = metrics['win_rate']
                evaluated_factors.append(factor)
                
                if factor.ic > best_ic:
                    best_ic = factor.ic
            
            # Select top factors
            top_factors = sorted(evaluated_factors, key=lambda x: x.ic, reverse=True)[:10]
            
            # Generate improvements
            improved_factors = self._generate_improvements(top_factors)
            
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
            print(f"  Avg IC (top 10): {round_record.avg_ic:.4f}")
            
            # Check convergence
            if self._check_convergence(evolution_history):
                print("\nConvergence reached!")
                break
        
        # Select best factors
        all_factors = []
        for round_record in evolution_history:
            all_factors.extend(round_record.improved_factors)
        
        best_factors = sorted(all_factors, key=lambda x: x.ic, reverse=True)[:10]
        
        return EvolutionResult(
            best_factors=best_factors,
            evolution_history=evolution_history,
            best_ic=best_ic,
            total_rounds=len(evolution_history),
        )
    
    def _generate_improvements(self, top_factors: List[CandidateFactor]) -> List[CandidateFactor]:
        """
        Generate improved factors based on top performers.
        
        Args:
            top_factors: Top-performing factors
            
        Returns:
            List of improved factors
        """
        improved_factors = []
        
        for i, factor in enumerate(top_factors):
            # Generate mutation (in practice, use LLM)
            improved = CandidateFactor(
                id=f"improved_{factor.id}_{i}",
                expression=f"rank({factor.expression} * rank(returns))",
                description=f"Improved version of {factor.description}",
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
    # Demo
    print("=== Self-Evolving Factor Generator Demo ===\n")
    
    # Generate sample data
    np.random.seed(42)
    n_dates = 100
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    prices = pd.DataFrame(
        10 + np.cumsum(np.random.randn(n_dates, n_stocks) * 0.02, axis=0),
        index=dates,
        columns=stock_codes,
    )
    
    # Initialize generator
    generator = SelfEvolvingGenerator()
    
    # Generate seed factors
    seed_factors = generator.generate_seed_factors(10)
    
    # Initialize backtester
    backtester = FactorBacktester(prices)
    
    # Run evolution
    result = generator.evolve(seed_factors, backtester, n_rounds=5)
    
    print(f"\nEvolution Complete!")
    print(f"  Total rounds: {result.total_rounds}")
    print(f"  Best IC: {result.best_ic:.4f}")
    print(f"  Number of best factors: {len(result.best_factors)}")
    
    print("\n=== Demo Complete ===")
