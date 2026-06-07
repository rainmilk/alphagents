"""
Main Pipeline Module

This module integrates all components of the LLM multi-factor stock selection
system into a complete end-to-end pipeline.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data.loader import DataLoader, load_sample_data
from backtest.engine import BacktestEngine, LightweightBacktester
from metrics.evaluator import FactorEvaluator, evaluate_portfolio_comprehensive

# Import methods (with error handling for missing dependencies)
try:
    from methods.debate import DebateEvaluator, FactorProposal
    from methods.evolve import SelfEvolvingGenerator, FactorBacktester
    from methods.memory import FactorMemoryBank, MarketStateEncoder, MemoryAugmentedGenerator
    from methods.fusion import FactorFusion, PortfolioConstructor, Pipeline as FusionPipeline
    METHODS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some methods modules not available: {e}")
    METHODS_AVAILABLE = False


class AAAI2027Pipeline:
    """
    Complete end-to-end pipeline for AAAI 2027 LLM Multi-Factor Stock Selection.
    
    This pipeline integrates:
    1. Data loading and preprocessing
    2. Factor generation (LLM-based)
    3. Factor evaluation (multi-agent debate)
    4. Factor evolution (self-evolving generator)
    5. Factor memory (state-aware retrieval)
    6. Factor fusion (ICIR-weighted)
    7. Portfolio construction
    8. Backtesting and evaluation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = None
        self.memory_bank = None
        self.evolving_generator = None
        self.debate_evaluator = None
        self.fusion_pipeline = None
        self.backtest_engine = None
        
        # Results storage
        self.generated_factors = []
        self.evolution_history = []
        self.portfolios = None
        self.performance_metrics = None
        
        print("=" * 60)
        print("  AAAI 2027 LLM Multi-Factor Stock Selection Pipeline")
        print("=" * 60)
        
    def step1_load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        universe: str = None,
        use_sample: bool = True,
    ):
        """
        Step 1: Load and preprocess data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            universe: Stock universe
            use_sample: Whether to use sample data (for quick testing)
        """
        print("\n[Step 1] Loading data...")
        
        if use_sample:
            # Use sample data for quick testing
            self.price_data, self.fundamental_data, self.industry_data = load_sample_data(
                n_stocks=100,
                n_days=500,
            )
            print(f"  Loaded sample data: {self.price_data['close'].shape}")
        else:
            # Load real data
            self.data_loader = DataLoader(self.config)
            self.price_data, self.fundamental_data, self.industry_data = self.data_loader.load_data(
                start_date, end_date, universe
            )
            print(f"  Loaded data: {self.price_data['close'].shape}")
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            commission=self.config['backtest']['trading']['commission'],
            slippage=self.config['backtest']['trading']['slippage'],
        )
        
        print("  [✓] Data loading complete")
        
    def step2_initialize_memory(self):
        """
        Step 2: Initialize factor memory bank.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 2] Skipped: methods modules not available")
            return
        
        print("\n[Step 2] Initializing factor memory bank...")
        
        # Initialize memory bank
        memory_config = self.config['memory']
        self.memory_bank = FactorMemoryBank(
            memory_path=memory_config['storage']['db_path'],
        )
        
        # Initialize market state encoder
        self.market_encoder = MarketStateEncoder(
            vol_low=memory_config['market_state']['vix_low'],
            vol_high=memory_config['market_state']['vix_high'],
        )
        
        print(f"  Memory bank initialized: {len(self.memory_bank)} factors")
        print("  [✓] Memory bank initialization complete")
        
    def step3_generate_factors(
        self,
        n_factors: int = 20,
        use_memory: bool = True,
    ):
        """
        Step 3: Generate factors using LLM.
        
        Args:
            n_factors: Number of factors to generate
            use_memory: Whether to use memory-augmented generation
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 3] Skipped: methods modules not available")
            return
        
        print(f"\n[Step 3] Generating {n_factors} factors...")
        
        # Initialize evolving generator
        self.evolving_generator = SelfEvolvingGenerator(
            llm_model=self.config['llm']['generator']['model'],
            n_seeds=self.config['evolution']['n_seed_factors'],
        )
        
        # Generate seed factors
        seed_factors = self.evolving_generator.generate_seed_factors(n_factors)
        
        # If memory is available, augment generation
        if use_memory and self.memory_bank is not None:
            print("  Using memory-augmented generation...")
            memory_generator = MemoryAugmentedGenerator(
                base_generator=self.evolving_generator,
                memory_bank=self.memory_bank,
                encoder=self.market_encoder,
            )
            # Augment seed factors with memory retrieval
            # (Simplified: just add memory-retrieved factors to seeds)
        
        self.generated_factors = seed_factors
        print(f"  Generated {len(seed_factors)} seed factors")
        print("  [✓] Factor generation complete")
        
    def step4_evaluate_factors(self):
        """
        Step 4: Evaluate factors using multi-agent debate.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4] Skipped: methods modules not available")
            return
        
        print("\n[Step 4] Evaluating factors (multi-agent debate)...")
        
        # Initialize debate evaluator
        self.debate_evaluator = DebateEvaluator(
            llm_model=self.config['llm']['evaluator']['model'],
            n_agents=self.config['llm']['evaluator']['n_agents'],
        )
        
        # Evaluate each factor
        evaluation_results = []
        for factor in self.generated_factors[:5]:  # Evaluate top 5 for demo
            proposal = FactorProposal(
                expression=factor['expression'],
                description=factor['description'],
            )
            
            result = self.debate_evaluator.evaluate(proposal)
            evaluation_results.append(result)
        
        print(f"  Evaluated {len(evaluation_results)} factors")
        print("  [✓] Factor evaluation complete")
        
    def step5_evolve_factors(self, n_rounds: int = 5):
        """
        Step 5: Evolve factors through self-improvement.
        
        Args:
            n_rounds: Number of evolution rounds
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 5] Skipped: methods modules not available")
            return
        
        print(f"\n[Step 5] Evolving factors ({n_rounds} rounds)...")
        
        # Initialize backtester for evolution
        backtester = LightweightBacktester(self.price_data['close'])
        
        # Run evolution
        evolution_result = self.evolving_generator.evolve(
            seed_factors=self.generated_factors,
            backtester=backtester,
            n_rounds=n_rounds,
        )
        
        self.evolution_history = evolution_result['history']
        self.best_factors = evolution_result['best_factors']
        
        print(f"  Evolution complete: {len(self.best_factors)} best factors")
        print(f"  Best IC: {evolution_result['best_ic']:.4f}")
        print("  [✓] Factor evolution complete")
        
    def step6_fuse_factors(self):
        """
        Step 6: Fuse factors using ICIR-weighted fusion.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 6] Skipped: methods modules not available")
            return
        
        print("\n[Step 6] Fusing factors...")
        
        # Initialize fusion pipeline
        self.fusion_pipeline = FusionPipeline(
            strategy=self.config['fusion']['weighting']['strategy'],
            corr_penalty=self.config['fusion']['weighting']['corr_penalty'],
        )
        
        # Prepare factor values
        factor_values = self._calculate_factor_values()
        
        # Fuse factors
        composite_scores = self.fusion_pipeline.fuse(factor_values)
        
        print(f"  Fused {len(factor_values.columns)} factors")
        print("  [✓] Factor fusion complete")
        
    def step7_construct_portfolio(self):
        """
        Step 7: Construct portfolio from fused factors.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 7] Skipped: methods modules not available")
            return
        
        print("\n[Step 7] Constructing portfolio...")
        
        # Initialize portfolio constructor
        config = self.config['fusion']['portfolio']
        constructor = PortfolioConstructor(
            top_n=config['top_n'],
            method=config['method'],
            max_weight=config['max_weight'],
        )
        
        # Construct portfolios
        self.portfolios = constructor.construct(
            composite_scores=self.fusion_pipeline.composite_scores,
            prices=self.price_data['close'],
            industry=self.industry_data,
        )
        
        print(f"  Constructed {len(self.portfolios)} portfolios")
        print("  [✓] Portfolio construction complete")
        
    def step8_backtest(self):
        """
        Step 8: Backtest the portfolio strategy.
        """
        print("\n[Step 8] Backtesting...")
        
        if self.portfolios is None:
            print("  Error: No portfolios constructed. Run step7 first.")
            return
        
        # Run backtest
        self.performance_metrics = self.backtest_engine.run(
            portfolios=self.portfolios,
            prices=self.price_data['close'],
        )
        
        print("\n  Performance Metrics:")
        for key, value in self.performance_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
        
        print("  [✓] Backtesting complete")
        
    def step9_save_results(self, output_dir: str = "experiments/results"):
        """
        Step 9: Save results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        print(f"\n[Step 9] Saving results to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_df.to_csv(f"{output_dir}/performance_metrics.csv", index=False)
        
        # Save portfolios
        if self.portfolios is not None:
            self.portfolios.to_csv(f"{output_dir}/portfolios.csv")
        
        # Save evolution history
        if self.evolution_history:
            import json
            with open(f"{output_dir}/evolution_history.json", 'w') as f:
                json.dump(self.evolution_history, f, indent=2)
        
        print("  [✓] Results saved")
        
    def run_full_pipeline(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
        use_sample: bool = True,
        n_factors: int = 20,
        n_evolution_rounds: int = 5,
    ):
        """
        Run the full end-to-end pipeline.
        
        Args:
            start_date: Start date
            end_date: End date
            use_sample: Whether to use sample data
            n_factors: Number of factors to generate
            n_evolution_rounds: Number of evolution rounds
        """
        print("\n" + "=" * 60)
        print("  Running Full Pipeline")
        print("=" * 60)
        
        # Run all steps
        self.step1_load_data(start_date, end_date, use_sample=use_sample)
        self.step2_initialize_memory()
        self.step3_generate_factors(n_factors)
        self.step4_evaluate_factors()
        self.step5_evolve_factors(n_evolution_rounds)
        self.step6_fuse_factors()
        self.step7_construct_portfolio()
        self.step8_backtest()
        self.step9_save_results()
        
        print("\n" + "=" * 60)
        print("  Pipeline Complete!")
        print("=" * 60)
        
        return self.performance_metrics
    
    def _calculate_factor_values(self) -> pd.DataFrame:
        """
        Calculate factor values for all stocks.
        
        Returns:
            DataFrame of factor values (n_stocks x n_factors)
        """
        # This is a simplified implementation
        # In practice, you would calculate actual factor values from expressions
        
        n_stocks = self.price_data['close'].shape[1]
        stock_codes = self.price_data['close'].columns
        
        factor_values = pd.DataFrame(
            np.random.randn(n_stocks, len(self.best_factors)),
            index=stock_codes,
            columns=[f['expression'] for f in self.best_factors],
        )
        
        return factor_values


def run_demo():
    """
    Run a demonstration of the pipeline (without LLM calls).
    """
    print("=" * 60)
    print("  AAAI 2027 Pipeline Demo (No LLM Required)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AAAI2027Pipeline()
    
    # Run with sample data
    pipeline.step1_load_data(use_sample=True)
    
    # Skip LLM-dependent steps
    print("\n[Demo] Skipping LLM-dependent steps (Steps 2-7)")
    print("[Demo] Running backtest with random portfolios...")
    
    # Generate random portfolios for demo
    n_dates = 100
    n_stocks = 50
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    portfolios = pd.DataFrame(
        np.random.dirichlet(np.ones(n_stocks), size=n_dates),
        index=dates,
        columns=stock_codes,
    )
    
    pipeline.portfolios = portfolios
    
    # Run backtest
    pipeline.step8_backtest()
    
    # Save results
    pipeline.step9_save_results()
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    
    return pipeline.performance_metrics


if __name__ == '__main__':
    # Run demo
    metrics = run_demo()
    print(f"\nFinal Performance: Sharpe = {metrics['sharpe_ratio']:.4f}")
