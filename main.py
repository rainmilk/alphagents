# -*- coding: utf-8 -*-
"""
Main Pipeline Module

This module integrates all components of the LLM multi-factor stock selection
system into a complete end-to-end pipeline.

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07

Usage:
    python main.py              # Quick demo (no LLM required)
    python main.py --full       # Full end-to-end pipeline
    python main.py --full --start 2023-01-01 --end 2024-12-31
    python main.py --test --factor-path experiments/YYYYMMDD/results/final_factors.json
    python main.py --test --factor-path PATH --start 2023-01-01 --end 2024-12-31
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml
import warnings

from config import config_path

warnings.filterwarnings('ignore')

# Import all modules
from dataloader.loader import DataLoader, load_sample_data, load_real_data
from backtest.engine import BacktestEngine

# Import methods (with error handling for missing dependencies)
try:
    from methods.debate import DebateEvaluator, FactorProposal
    from methods.evolve import SelfEvolvingGenerator, FactorBacktester, CandidateFactor
    from methods.memory import FactorMemoryBank, FactorEmbedder, MarketStateEncoder, MemoryAugmentedGenerator
    from methods.fusion import FactorFusion, PortfolioConstructor, FactorInfo, PortfolioConfig, FactorNormalizer

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
    3. Factor evolution (self-evolving — backtest coarse-filter)
    4. Factor evaluation (multi-agent debate — quality gate)
    5. Chair synthesis (cross-factor ranking + selection reasons)
    6. Memory retrieval (state-aware)
    7. Factor fusion (ICIR-weighted)
    8. Portfolio construction
    9. Backtesting and evaluation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self._config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
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

        # Recent data window for market state encoding (from config, default 60)
        self.recent_days = self.config.get("memory", {}).get("market_state", {}).get("recent_days", 60)
        
        print("=" * 60)
        print("  AAAI 2027 LLM Multi-Factor Stock Selection Pipeline")
        print("=" * 60)
        
    def step1_load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        universe: str = None,
        use_sample: bool = True,
        data_source: str = "auto",
        force_refresh: bool = False,
        split_train_end: str = None,
        split_test_start: str = None,
    ):
        """
        Step 1: Load and preprocess data.

        Data source selection (single decision point):
        - use_sample=True  → fast synthetic data for testing (n_stocks=100, n_days=500)
        - use_sample=False → real data via westock/AkShare/Tushare with cache

        Args:
            start_date: Start date (YYYY-MM-DD), for real data only
            end_date: End date (YYYY-MM-DD), for real data only
            universe: Stock universe (hs300, zz500, all_a), for real data only
            use_sample: True=sample data, False=real data
            data_source: 'westock', 'akshare', 'tushare', or 'auto'
            force_refresh: Skip cache and re-download real data
            split_train_end: Explicit train end date (overrides config)
            split_test_start: Explicit test start date (overrides config)
        """

        print("\n[Step 1] Loading data...")

        if use_sample:
            # Fast synthetic data — no external API needed
            self.price_data, self.fundamental_data, self.industry_data = load_sample_data(
                n_stocks=100,
                n_days=500,
            )
            self._data_source_label = "sample"
            print(f"  [sample] Loaded: {self.price_data['close'].shape}")
        else:
            # Real data via westock → AkShare → Tushare → synthetic fallback
            self.price_data, self.fundamental_data, self.industry_data = load_real_data(
                universe=universe or self.config['data']['universe']['index'],
                start_date=start_date or self.config['data']['universe']['start_date'],
                end_date=end_date or self.config['data']['universe']['end_date'],
                source=data_source,
                force_refresh=force_refresh,
                config=self.config,
            )
            self._data_source_label = "real"
            print(f"  [real] Loaded: {self.price_data['close'].shape}")

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            commission=self.config['backtest']['trading']['commission'],
            slippage=self.config['backtest']['trading']['slippage'],
            holding_period=self.config.get('backtest', {}).get('trading', {}).get('holding_period', 1),
        )

        # --- Train/Test Split (Critical for preventing overfitting) ---
        # Create DataLoader instance and populate its internal data so split_data() works
        self.data_loader = DataLoader(config_path=self.config.get('_config_path', 'config/config.yaml'))
        self.data_loader.price_data = self.price_data
        self.data_loader.fundamental_data = self.fundamental_data
        self.data_loader.industry_data = self.industry_data
        
        # Read split config from explicit params, config.yaml, or defaults
        train_end_date = split_train_end or self.config.get('data', {}).get('train_end_date', None)
        test_start_date = split_test_start or self.config.get('data', {}).get('test_start_date', None)
        
        if train_end_date is None:
            # Default: use 80/20 split (80% train, 20% test)
            # For default date range 2019-01-01 to 2024-12-31, this is ~2023-12-31
            dates = self.price_data['close'].index
            split_idx = int(len(dates) * 0.8)
            train_end_date = dates[split_idx - 1].strftime('%Y-%m-%d')
            test_start_date = dates[split_idx].strftime('%Y-%m-%d')
            print(f"  [split] Using default 80/20 split: train_end={train_end_date}, test_start={test_start_date}")
        elif test_start_date is None:
            # train_end specified but test_start not — set test_start to day after
            test_start_date = train_end_date
            print(f"  [split] Using explicit split: train_end={train_end_date}, test_start={test_start_date}")
        else:
            print(f"  [split] Using explicit split: train_end={train_end_date}, test_start={test_start_date}")
        
        # Context days: prepend N training days to test_data so rolling-window
        # factor expressions (ts_mean, ts_std, etc.) have history on day 1.
        context_days = self.config.get('data', {}).get('context_days', 30)
        self.train_data, self.test_data = self.data_loader.split_data(
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            context_days=context_days,
        )
        self._train_end_date = train_end_date
        self._test_start_date = test_start_date
        self._context_days = context_days
        
        # --- Save train/test data to data directory ---
        self._save_split_data(train_end_date, test_start_date)
        
        print(f"  [✓] Train/Test split complete")
        print(f"       Train: {train_end_date}, Test: {test_start_date}")
        print("  [✓] Data loading complete")
        
    def _save_split_data(self, train_end_date: str, test_start_date: str):
        """
        Persist train/test data as CSV files under data/train/ and data/test/.

        Directory structure:
            data/train/price/close.csv, open.csv, high.csv, low.csv, volume.csv, amount.csv
            data/train/fundamental/pe.csv, pb.csv, ...
            data/train/industry.csv
            data/test/  (same structure)
            data/split_info.json  — metadata
        """
        
        def _save_category(data_dict, root: str, category: str):
            """Save a dict of DataFrames as CSV files under root/category/."""
            if data_dict is None:
                return 0
            cat_dir = os.path.join(root, category)
            os.makedirs(cat_dir, exist_ok=True)
            count = 0
            for key, df in data_dict.items():
                csv_path = os.path.join(cat_dir, f"{key}.csv")
                df.to_csv(csv_path, index=True, encoding='utf-8-sig')
                count += 1
            return count
        
        def _save_series(series, root: str, filename: str):
            """Save a Series-like object as CSV."""
            if series is None:
                return
            csv_path = os.path.join(root, filename)
            pd.Series(series).to_csv(csv_path, index=True, encoding='utf-8-sig')
        
        for split, data in [("train", self.train_data), ("test", self.test_data)]:
            root = config_path("data", split)
            os.makedirs(root, exist_ok=True)
            
            n_price = _save_category(data.get('price_data'), root, "price")
            n_fund = _save_category(data.get('fundamental_data'), root, "fundamental")
            _save_series(data.get('industry_data'), root, "industry.csv")
            
            print(f"  [save] {split}: {n_price} price CSVs + {n_fund} fundamental CSVs → {root}/")
        
        # Save split metadata
        train_dates = self.train_data['price_data']['close'].index
        test_dates = self.test_data['price_data']['close'].index
        
        split_info = {
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "train_dates": f"{train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')}",
            "test_dates": f"{test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}",
            "train_days": int(len(train_dates)),
            "test_days": int(len(test_dates)),
            "train_n_stocks": self.train_data['price_data']['close'].shape[1],
            "test_n_stocks": self.test_data['price_data']['close'].shape[1],
            "saved_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        info_path = config_path("data", "split_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"  [save] Split info → {info_path}")
        
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
        
        # Initialize embedder
        embedder = FactorEmbedder(
            model_name=memory_config['encoder']['model'],
            device=memory_config['encoder']['device'],
        )
        
        self.memory_bank = FactorMemoryBank(
            embedder=embedder,
            index_path=memory_config['storage']['db_path'],
        )
        
        # Initialize market state encoder
        self.market_encoder = MarketStateEncoder(
            vix_low=memory_config['market_state']['vix_low'],
            vix_high=memory_config['market_state']['vix_high'],
        )
        
        print(f"  Memory bank initialized: {len(self.memory_bank)} factors")
        print("  [✓] Memory bank initialization complete")
        
    def step3_generate_factors(
        self,
        use_memory: bool = True,
    ):
        """
        Step 3: Generate factors using LLM.

        The number of factors is read from config: evolution.n_seed_factors
        
        Args:
            use_memory: Whether to use memory-augmented generation
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 3] Skipped: methods modules not available")
            return
        
        n_seeds = self.config['evolution']['n_seeds']
        print(f"\n[Step 3] Generating {n_seeds} seed factors...")
        
        # Initialize evolving generator
        evo_cfg = self.config['evolution']
        self.evolving_generator = SelfEvolvingGenerator(
            llm_model=self.config['llm']['generator']['model'],
            n_seeds=n_seeds,
            n_best_factors=evo_cfg['n_best_factors'],
            n_improve=evo_cfg.get('n_improve', 10),
            n_mutate=evo_cfg.get('n_mutate', 5),
            convergence_delta=evo_cfg.get('convergence_delta', 0.003),
            convergence_window=evo_cfg.get('convergence_window', 2),
            patience=evo_cfg.get('patience', 3),
            min_ic=evo_cfg.get('min_ic', 0.02),
            min_sharpe=evo_cfg.get('min_sharpe', 0.5),
            max_drawdown=evo_cfg.get('max_drawdown', -0.20),
        )
        
        # Generate seed factors
        seed_factors = self.evolving_generator.generate_seed_factors()
        
        # If memory is available, augment generation
        if use_memory and self.memory_bank is not None and len(self.memory_bank) > 0:
            print("  Using memory-augmented generation...")
            memory_generator = MemoryAugmentedGenerator(
                base_generator=self.evolving_generator,
                memory_bank=self.memory_bank,
                encoder=self.market_encoder,
            )
            # Use memory-augmented generator to produce factors with
            # few-shot examples retrieved from the memory bank
            try:
                # Build market index proxy for state encoding (use TRAIN data only)
                close_source = self.train_data['price_data']['close'] if self.train_data else self.price_data['close']
                market_close = close_source.mean(axis=1)
                market_df = pd.DataFrame({'close': market_close})
                augmented_factors = memory_generator.generate(
                    task_description=f"Generate {n_seeds} alpha factors for A-share stock selection",
                    retrieval_query=(
                        "quality factor with stable profitability and ROE, "
                        "value reversal factor using valuation metrics like PE and PB, "
                        "momentum factor capturing price trends with moving averages, "
                        "volatility factor for low-volatility effect, "
                        "liquidity factor for small-cap premium"
                    ),
                    price_df=market_df,
                )
                if augmented_factors:
                    # Convert dict results to CandidateFactor objects
                    # (MemoryAugmentedGenerator returns list[dict], but evolve() expects CandidateFactor)
                    aug_factor_objs = [
                        CandidateFactor(
                            id=f"memory_aug_{i}",
                            expression=f.get("expression", ""),
                            description=f.get("description", f"Memory-augmented factor {i}"),
                        )
                        for i, f in enumerate(augmented_factors)
                        if f.get("expression")
                    ]
                    self.generated_factors = seed_factors + aug_factor_objs
                    print(f"  Memory-augmented generation: {len(aug_factor_objs)} additional factors")
            except Exception as e:
                print(f"  Memory augmentation skipped: {e}")
                self.generated_factors = seed_factors
        else:
            self.generated_factors = seed_factors
        print(f"  Generated {len(seed_factors)} seed factors")
        print("  [✓] Factor generation complete")
        
    def step4_evolve_factors(self, n_rounds: int = 5, forward_period: int = None):
        """
        Step 4: Self-evolve seed factors through iterative improvement.

        Evolve before debate — the backtester coarsely filters factors,
        then debate provides rigorous multi-expert evaluation on the best.
        
        Args:
            n_rounds: Number of evolution rounds
            forward_period: Forward return horizon in trading days (None → config or 20)
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4] Skipped: methods modules not available")
            return
        
        # Resolve forward_period: explicit arg > config.yaml > default 20
        if forward_period is None:
            forward_period = self.config.get('evolution', {}).get('forward_period', 20)
        
        print(f"\n[Step 4] Evolving factors ({n_rounds} rounds, forward={forward_period}d)...")
        
        # Initialize backtester for evolution — USE TRAINING DATA ONLY
        # Critical: factors must NOT see test data during evolution
        backtester = FactorBacktester(
            prices=self.train_data['price_data'],
            fundamentals=self.train_data['fundamental_data'],
            forward_period=forward_period,
        )
        print(f"  [evolution] Using TRAIN data only: {self._train_end_date}")
        evolution_result = self.evolving_generator.evolve(
            seed_factors=self.generated_factors,
            backtester=backtester,
            n_rounds=n_rounds,
        )
        
        self.evolution_history = evolution_result.evolution_history
        self.best_factors = evolution_result.best_factors
        self._forward_period = forward_period   # persist for step6/_calculate_factor_values

        print(f"  Evolution complete: {len(self.best_factors)} best factors")
        best_ic = self.best_factors[0].ic if self.best_factors else 0.0
        print(f"  Best IC: {best_ic:.4f}")
        print("  [✓] Factor evolution complete")

    def step4b_retrieve_from_memory(self):
        """
        Step 4b: Retrieve historical high-quality factors from memory bank.

        Runs AFTER step4_evolve_factors but BEFORE step5_evaluate_factors so that
        retrieved factors join the candidate pool and are debate-evaluated alongside
        newly evolved factors. This ensures retrieved factors also receive a
        debate_score for fusion.

        Uses state-aware retrieval to find factors from memory that performed
        well in similar market conditions, augmenting the evolved factor pool
        with proven historical factors.

        Retrieved factors are tagged with _from_memory=True so step5 will not
        redundantly re-write them back to the memory bank.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4b] Skipped: methods modules not available")
            return

        if self.memory_bank is None or len(self.memory_bank) == 0:
            print("\n[Step 4b] Skipped: memory bank is empty")
            return

        print("\n[Step 4b] Retrieving factors from memory (state-aware)...")

        # Encode current market state using TRAIN data (not test data)
        # This prevents data leak from test period
        close_df = self.train_data['price_data']['close'].iloc[-self.recent_days:]
        # Pass the full multi-stock DataFrame so _build_market_state can compute
        # corr_matrix from actual pairwise return correlations.
        current_state = self._build_market_state(close_df)

        # Retrieve similar factors from memory
        # Use best evolved factors as query (higher quality than raw seeds)
        query_desc = "multi-factor stock selection using technical indicators"
        query_expr = ""
        query_source = self.best_factors if (
                    hasattr(self, 'best_factors') and self.best_factors) else self.generated_factors
        if query_source:
            # Use the first factor as query
            f = query_source[0]
            if isinstance(f, dict):
                query_desc = f.get('description', query_desc)
                query_expr = f.get('expression', '')
            else:
                query_desc = getattr(f, 'description', query_desc)
                query_expr = getattr(f, 'expression', '')

        retrieval_cfg = self.config['memory'].get('retrieval', {})
        retrieved_factors = self.memory_bank.retrieve(
            query_description=query_desc,
            query_expression=query_expr,
            current_market_state=current_state,
            top_k=retrieval_cfg.get('top_k', 5),
            state_weight=retrieval_cfg.get('state_weight', 0.3),
            min_ic=retrieval_cfg.get('min_ic', 0.02),
        )

        if retrieved_factors:
            print(f"  Retrieved {len(retrieved_factors)} factors from memory")

            # Merge retrieved factors into the factor pool
            # Retrieved factors augment the best evolved factors
            # Deduplicate by expression to avoid duplicates (step5b may be re-run)
            existing_exprs = set()
            if hasattr(self, 'best_factors') and self.best_factors:
                for f in self.best_factors:
                    if isinstance(f, dict):
                        expr = f.get('expression', '')
                    else:
                        expr = getattr(f, 'expression', '')
                    if expr:
                        existing_exprs.add(expr)

            added = 0
            if hasattr(self, 'best_factors') and self.best_factors:
                for rf in retrieved_factors:
                    if rf.expression not in existing_exprs:
                        # Mark as memory-sourced so step5 won't re-save it to memory bank
                        rf._from_memory = True
                        self.best_factors.append(rf)
                        existing_exprs.add(rf.expression)
                        added += 1

            print(f"  Added {added} new factors from memory (skipped {len(retrieved_factors) - added} duplicates)")
            print(f"  Factor pool size after merge: {len(self.best_factors)}")
        else:
            print("  No relevant factors found in memory")

        print("  [✓] Memory retrieval complete")
        
    def step5_evaluate_factors(self):
        """
        Step 5: Evaluate ALL candidate factors via multi-agent debate.

        Runs AFTER step4b_retrieve_from_memory — both newly evolved factors
        and memory-retrieved factors are evaluated together. This ensures every
        factor entering step6 fusion has a valid debate_score.

        Debate serves as the final quality gate, applying 5-expert cross-
        validation to the full candidate pool before fusion.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 5] Skipped: methods modules not available")
            return
        
        print("\n[Step 5] Evaluating factors (multi-agent debate)...")
        
        # Initialize debate evaluator (read API config from config.yaml)
        eval_cfg = self.config['llm']['evaluator']
        api_key = eval_cfg.get('api_key', '') or os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = eval_cfg.get('base_url', "https://api.deepseek.com")

        # Chair Agent config (separate model for arbitration & synthesis)
        chair_cfg = self.config['llm'].get('chair', {})
        chair_api_key = chair_cfg.get('api_key', '') or os.environ.get("CHAIR_API_KEY", "") or api_key

        self.debate_evaluator = DebateEvaluator(
            llm_model=eval_cfg['model'],
            n_agents=eval_cfg['n_agents'],
            n_rounds=eval_cfg.get('n_debate_rounds', 3),
            api_key=api_key,
            base_url=base_url,
            # Chair Agent — typically a stronger model (GPT-4o) for arbitration & synthesis
            chair_model=chair_cfg.get('model', ''),
            chair_api_key=chair_api_key,
            chair_base_url=chair_cfg.get('base_url', ''),
            chair_temperature=chair_cfg.get('temperature', 0.2),
            # Parallel: 5 agents evaluate the same factor concurrently (I/O-bound API calls)
            parallel_eval=eval_cfg.get('parallel_eval', True),
        )
        
        # Evaluate the evolved best_factors (not raw seed factors)
        self.debate_results = []  # (factor, DebateResult) pairs for step6 fusion
        for factor in self.best_factors:
            # Handle both CandidateFactor objects and dict representations
            if isinstance(factor, dict):
                expr = factor['expression']
                desc = factor['description']
            else:
                expr = factor.expression
                desc = factor.description

            proposal = FactorProposal(expression=expr, description=desc)

            result = self.debate_evaluator.evaluate(proposal)
            self.debate_results.append((factor, result))

            # Attach debate score to factor for use in step6 fusion
            if isinstance(factor, dict):
                factor['debate_score'] = result.final_score
            else:
                factor.debate_score = result.final_score

        print(f"  Evaluated {len(self.debate_results)} factors")
        print("  [✓] Factor evaluation complete")

        # Save debated best factors to memory bank
        if self.memory_bank is not None and self.debate_results:
            try:
                # Use TRAIN data for market state encoding (not test data)
                if self.train_data:
                    close_df = self.train_data['price_data']['close'].iloc[-self.recent_days:]
                    # Pass multi-stock DataFrame so corr_matrix is computed from
                    # actual pairwise return correlations (CorrelationRegime is real).
                    current_state = self._build_market_state(close_df)
                else:
                    current_state = None
            except Exception:
                current_state = None
            saved = 0
            for factor, result in self.debate_results:
                if isinstance(factor, dict):
                    expr = factor.get('expression', '')
                    desc = factor.get('description', '')
                    is_from_memory = factor.get('_from_memory', False)
                else:
                    expr = getattr(factor, 'expression', '')
                    desc = getattr(factor, 'description', '')
                    is_from_memory = getattr(factor, '_from_memory', False)
                if not expr:
                    continue
                # Skip factors retrieved from memory — they're already stored there
                if is_from_memory:
                    continue
                try:
                    self.memory_bank.add(
                        expression=expr,
                        description=desc,
                        market_state=current_state,
                        ic=getattr(factor, 'ic', 0.0),
                        icir=getattr(factor, 'icir', 0.0),
                        sharpe=getattr(factor, 'sharpe', 0.0),
                        win_rate=getattr(factor, 'win_rate', 0.0),
                        max_drawdown=getattr(factor, 'max_drawdown', 0.0),
                        source='debate',
                    )
                    saved += 1
                except Exception as e:
                    print(f"  [memory] add failed: {e}")
            if saved:
                print(f"  Saved {saved} factors to memory bank")
        
    def step5b_chair_synthesis(self):
        """
        Step 5b: Chair Agent synthesizes ALL debate results into a comprehensive
        cross-factor report with final rankings, selection reasons, and rejection reasons.

        This is the capstone of the multi-agent debate module. The Chair Agent
        (typically a stronger model like GPT-4o) reviews all debate results
        holistically and produces:

        - Ranked list of factors with 入选理由 (selection reasons)
        - Rejected factors with 淘汰原因 (rejection reasons)
        - Cross-cutting themes and overall confidence assessment

        Output: experiments/{yyyymmdd}/chair_synthesis.json
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 5b] Skipped: methods modules not available")
            return

        if not hasattr(self, 'debate_results') or not self.debate_results:
            print("\n[Step 5b] Skipped: no debate results available")
            return

        print("\n[Step 5b] Chair Agent synthesizing all debate results...")

        # Build (FactorProposal, DebateResult) pairs
        proposals_and_results = []
        for factor, result in self.debate_results:
            if isinstance(factor, dict):
                expr = factor.get('expression', '')
                desc = factor.get('description', '')
            else:
                expr = getattr(factor, 'expression', '')
                desc = getattr(factor, 'description', '')

            proposal = FactorProposal(expression=expr, description=desc)
            proposals_and_results.append((proposal, result))

        # Call Chair synthesis (saves chair_synthesis.json + appends to debate_factors_result.json internally)
        synthesis = self.debate_evaluator.synthesize_all_factors(proposals_and_results)

        # Print summary
        print(f"  Chair synthesis complete:")
        print(f"    Selected: {synthesis['selected_count']} factors")
        print(f"    Rejected: {len(synthesis['rejected_factors'])} factors")
        print(f"    Confidence: {synthesis['chair_confidence']}")
        if synthesis.get('key_themes'):
            print(f"    Key themes:")
            for theme in synthesis['key_themes']:
                print(f"      - {theme}")
        print("  [✓] Chair synthesis complete")

    def step6_fuse_factors(self):
        """
        Step 6: Fuse factors using ICIR² shrinkage fusion with sign-aware,
        IPR correlation penalty, and regime-adaptive tilt.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 6] Skipped: methods modules not available")
            return
        
        print("\n[Step 6] Fusing factors...")

        # ── Guard: need train/test split ──
        if not hasattr(self, 'train_data') or not self.train_data:
            print("  [error] train_data not available. Run step1 (load_data) first.")
            return

        # Initialize fusion with strategy, normalizer and hyperparameters from config
        weighting_cfg = self.config['fusion']['weighting']
        norm_cfg = self.config['fusion'].get('normalization', {})
        normalizer = FactorNormalizer(
            method=norm_cfg.get('method', 'zscore'),
            neutralize_industry=norm_cfg.get('neutralize_industry', True),
        )
        fusion = FactorFusion(
            strategy=weighting_cfg.get('strategy', 'icir2_shrinkage'),
            corr_penalty=weighting_cfg.get('corr_penalty', True),
            corr_threshold=weighting_cfg.get('corr_threshold', 0.7),
            normalizer=normalizer,
            regime_tilt_strength=weighting_cfg.get('regime_tilt_strength', 0.2),
            shrinkage_kappa=weighting_cfg.get('shrinkage_kappa', 0.3),
            ipr_alpha=weighting_cfg.get('ipr_alpha', 2.0),
        )

        # ── 1. Compute factor values on TRAIN data (for weight computation) ──
        train_price = self.train_data['price_data']
        train_fund = self.train_data.get('fundamental_data', {})
        print("  Computing factor values on TRAIN data (no leakage)...")
        train_factor_dict = self._calculate_factor_values(
            price_data=train_price,
            fundamental_data=train_fund,
            forward_period=getattr(self, '_forward_period', None),
        )

        # ── 2. Build factor_infos from backtest results (IC/ICIR from train data) ──
        factor_meta_lookup = {}
        if hasattr(self, 'best_factors') and self.best_factors:
            for f in self.best_factors:
                expr = f.expression if hasattr(f, 'expression') else str(f)
                ic_val = getattr(f, 'ic', 0.0)
                icir_val = getattr(f, 'icir', 0.0)
                sharpe_val = getattr(f, 'sharpe', 0.0)
                factor_meta_lookup[expr] = (ic_val, icir_val, sharpe_val)

        debate_score_map = {}
        if hasattr(self, 'debate_results') and self.debate_results:
            for factor, result in self.debate_results:
                if isinstance(factor, dict):
                    expr = factor.get('expression', '')
                else:
                    expr = getattr(factor, 'expression', '')
                if expr:
                    # result is a DebateResult or dict with final_score
                    score = result.final_score if hasattr(result, 'final_score') else result.get('final_score', 0.0)
                    debate_score_map[expr] = score

        # Use train_factor_dict keys (same factor names, train-period only)
        factor_infos = []
        for name, values_df in train_factor_dict.items():
            if not isinstance(values_df.index, pd.DatetimeIndex):
                values_df = values_df.copy()
                values_df.index = pd.to_datetime(values_df.index)

            meta = factor_meta_lookup.get(name)
            if meta:
                ic_val, icir_val, sharpe_val = meta
                if abs(icir_val) > 1e-8:
                    ic_std_val = ic_val / icir_val
                else:
                    ic_std_val = 1.0
            else:
                ic_val = ic_std_val = sharpe_val = 0.0

            dscore = debate_score_map.get(name, 0.0)
            ic_sign = float(np.sign(ic_val)) if abs(ic_val) > 1e-10 else 1.0
            # Effective sample size for IC/ICIR:
            # factor values have T dates, but last forward_period dates lack forward returns
            _fwd = getattr(self, '_forward_period', 20)
            n_periods = max(2, len(values_df) - _fwd)
            factor_infos.append(FactorInfo(
                name=name, expression=name,
                ic=ic_val, icir=icir_val, ic_std=ic_std_val,
                sharpe=sharpe_val, debate_score=dscore,
                ic_sign=ic_sign,
                n_periods=n_periods,
            ))

        # Save for step7 (which reads weights/signs from JSON, but still needs
        # factor names to iterate over test_factor_dict)
        self.factor_infos = factor_infos

        # ── 4b. Extract training industry data (for industry-neutral normalization) ──
        train_industry = None
        if hasattr(self, 'industry_data') and self.industry_data is not None:
            train_close = train_price.get('close')
            if train_close is not None and isinstance(train_close, pd.DataFrame):
                common_stocks = train_close.columns
                train_industry = self.industry_data[
                    self.industry_data.index.isin(common_stocks)
                ]
                if len(train_industry) > 0:
                    print(f"  Industry data: {len(train_industry)} stocks, "
                          f"{train_industry.nunique()} industries")
                else:
                    train_industry = None
                    print("  [warn] Industry data empty after filtering to train stocks")
            else:
                print("  [warn] train_price['close'] not available for industry filtering")
        else:
            print("  [warn] No industry_data available — "
                  "industry-neutral normalization will be skipped")

        # ── 4c. Encode market state from TRAIN data (avoid lookahead) ──
        market_state = None
        try:
            train_close = train_price.get('close')
            if train_close is not None and isinstance(train_close, pd.DataFrame):
                # Use the shared helper so corr_matrix is computed from actual
                # pairwise return correlations (rather than defaulting to MEDIUM).
                market_state = self._build_market_state(train_close)
                print(f"  Market state (from TRAIN): {market_state.to_string() if market_state else 'None'}")
        except Exception as e:
            print(f"  [warn] Could not encode market state: {e}")

        # ── 5. Fuse: weights and scores from train data only ──
        composite_scores, fusion_meta = fusion.fuse(
            factor_infos,
            train_factor_dict,
            industry=train_industry,
            market_state=market_state,
        )

        print(f"  Fused {len(factor_infos)} factors (strategy={fusion.strategy})")
        print(f"  Corr penalty (train-only): {fusion_meta.get('corr_penalty_applied', False)}")
        print(f"  Regime tilt: {fusion_meta.get('regime_tilt_applied', False)}")
        for fi in factor_infos:
            w = fusion_meta["weights"].get(fi.name, 0.0)
            s = fusion_meta.get("signs", {}).get(fi.name, 1.0)
            print(f"    {fi.name:40s}  w={w:.4f}  sign={s:+.0f}  ic={fi.ic:+.4f}  icir={fi.icir:+.4f}")

        # --- Save fusion results to experiments/{yyyymmdd}/fusion/final_factors.json ---
        date_str = datetime.now().strftime("%Y%m%d")
        fusion_dir = os.path.join(date_str, "fusion")
        fusion_dir = config_path("experiments", fusion_dir)
        os.makedirs(fusion_dir, exist_ok=True)

        # Build factor info list — skip factors that failed evaluation (is_valid=False / nan IC)
        import math as _math
        factor_details = []
        for fi in factor_infos:
            if not getattr(fi, 'is_valid', True):
                print(f"[step6] Skipping invalid factor '{fi.name}' from final_factors.json")
                continue
            ic_val = getattr(fi, 'ic', 0.0)
            if isinstance(ic_val, float) and _math.isnan(ic_val):
                print(f"[step6] Skipping factor '{fi.name}' with nan IC from final_factors.json")
                continue
            factor_details.append({
                "name": fi.name,
                "expression": fi.expression,
                "weight": fusion_meta["weights"].get(fi.name, 0.0),
                "debate_score": getattr(fi, "debate_score", None),
                "n_periods": getattr(fi, "n_periods", 50),
            })

        self.fusion_result = {
            "meta": fusion_meta,
            "factors": factor_details
        }

        fusion_path = os.path.join(fusion_dir, "final_factors.json")
        with open(fusion_path, "w", encoding="utf-8") as f:
            json.dump(self.fusion_result, f, ensure_ascii=False, indent=2)

        print(f"  Fusion results saved to {fusion_path}")
        print("  [✓] Factor fusion complete")
        
    def step7_construct_portfolio(self, fusion_result=None, test_data=None):
        """
        Step7: Construct portfolio from fused factors.

        Uses the weights trained in step6 (via fuse() on train_data) and
        applies them to test_data factor values via FactorFusion.predict().

        Args:
            test_data: Optional external test data dict with keys
                      {'price_data': dict, 'fundamental_data': dict, 'industry_data': Series-like}.
                      If None, uses self.test_data (from step1 split).
                      This allows callers to supply a custom test period without
                      re-running the entire pipeline.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 7] Skipped: methods modules not available")
            return
        
        print("\n[Step 7] Constructing portfolio...")

        # --- Resolve test data: explicit arg > self.test_data ---
        if test_data is not None:
            _test = test_data
            print("  Using externally provided test_data")
        elif hasattr(self, 'test_data') and self.test_data:
            _test = self.test_data
            print("  Using self.test_data (from step1 split)")
        else:
            print("  [error] No test_data available. Pass test_data= or run step1 first.")
            return

        # Initialize portfolio constructor
        portfolio_cfg = self.config['fusion']['portfolio']
        constructor = PortfolioConstructor(PortfolioConfig(
            top_n=portfolio_cfg['top_n'],
            method=portfolio_cfg['method'],
            max_weight=portfolio_cfg['max_weight'],
            max_industry_exposure=portfolio_cfg.get('max_industry_exposure', 0.30),
        ))

        # 1. Compute factor values on test data
        print("  Computing test-period factor values...")
        test_price = _test.get('price_data', {})
        test_fund = _test.get('fundamental_data', None)
        test_factor_dict = self._calculate_factor_values(
            price_data=test_price,
            fundamental_data=test_fund,
            forward_period=getattr(self, '_forward_period', None),
        )

        # 2. 从 final_factors.json 加载训练好的权重和符号
        if fusion_result is None:
            fusion_result = self.fusion_result
        meta = fusion_result['meta']
        saved_weights = meta['weights']
        saved_signs = meta.get('signs', {})

        # normalizer 是无状态的（横截面标准化），参数从 config 读取
        norm_cfg = self.config['fusion'].get('normalization', {})
        normalizer = FactorNormalizer(
            method=norm_cfg.get('method', 'zscore'),
            neutralize_industry=norm_cfg.get('neutralize_industry', True),
        )

        if not saved_weights:
            print("  [error] No saved weights. Run step6 first.")
            return

        _industry = _test.get('industry_data', None)

        # 2a. 找所有因子值的公共日期
        common_dates = None
        for f in self.factor_infos:
            fv = test_factor_dict[f.name]
            if common_dates is None:
                common_dates = fv.index
            else:
                common_dates = common_dates.intersection(fv.index)

        if common_dates is None or len(common_dates) == 0:
            print("  [error] No common dates in test factor values")
            return

        # 2b. Sign-Aware 加权求和: Σ w_i * sign_i * normalize(factor_i)
        weighted = None
        for f in self.factor_infos:
            fv = test_factor_dict[f.name].loc[common_dates]
            fv_norm = normalizer.normalize(fv, _industry)
            w = saved_weights.get(f.name, 0.0)
            s = saved_signs.get(f.name, 1.0)
            fv_weighted = fv_norm * w * s
            if weighted is None:
                weighted = fv_weighted
            else:
                weighted = weighted.add(fv_weighted, fill_value=0)

        test_composite_scores = normalizer.normalize(weighted, _industry)

        if test_composite_scores.empty:
            print("  [error] Computed composite scores are empty")
            return

        print(f"  Test-period composite scores: {len(test_composite_scores)} dates")

        # Build portfolios for test period
        close_copy = _test['price_data']['close'].copy()
        raw_portfolios = constructor.build(
            composite_scores=test_composite_scores,
            prices=close_copy,
            industry=_industry,
        )

        # Convert list[Portfolio] to DataFrame (n_dates x n_stocks) for backtester
        weight_dict = {}
        for pf in raw_portfolios:
            weight_dict[pf.date] = pf.weights
        self.portfolios = pd.DataFrame(weight_dict).T
        self.portfolios.index = pd.to_datetime(self.portfolios.index)
        self.portfolios = self.portfolios.fillna(0.0)

        # 修复：将 portfolios 对齐到价格数据的全部股票
        # pf.weights 只含 top-N 股票（每天数量可能不同），
        # 需 pad 到全股票列表，否则 backtest 中 current_weights
        # 与 price_ratio 形状不匹配（列数不一致）
        all_stocks = _test['price_data']['close'].columns
        self.portfolios = self.portfolios.reindex(columns=all_stocks, fill_value=0.0)

        # Align portfolios to actual trading days (same index as test close prices)
        trading_days = _test['price_data']['close'].index
        self.portfolios = self.portfolios.reindex(trading_days.intersection(self.portfolios.index))
        # Fill any missing dates with previous weights (hold positions on skipped dates)
        self.portfolios = self.portfolios.ffill().fillna(0.0)

        # Crop context window: portfolios built on context dates are invalid for backtest.
        # The real test period starts at _test_start_date (or inferred from test_data._meta).
        test_start = self._test_start_date
        if test_start is None and '_meta' in _test:
            test_start = _test['_meta'].get('test_start_date')
        if test_start is not None:
            test_start_ts = pd.Timestamp(test_start)
            n_before = len(self.portfolios)
            self.portfolios = self.portfolios[self.portfolios.index >= test_start_ts]
            n_cropped = n_before - len(self.portfolios)
            if n_cropped > 0:
                print(f"  Cropped {n_cropped} context dates (test data starts {test_start_ts.date()})")

        print(f"  Constructed {len(self.portfolios)} portfolios (test period, trading days only)")
        print("  [✓] Portfolio construction complete")

    def step8_backtest(self, test_data=None):
        """
        Step 8: Backtest the portfolio strategy.
        """
        print("\n[Step 8] Backtesting...")

        # --- Resolve test data: explicit arg > self.test_data ---
        if test_data is not None:
            _test = test_data
        elif hasattr(self, 'test_data') and self.test_data:
            _test = self.test_data
        else:
            print("  [error] No test_data available for backtest.")
            self.performance_metrics = {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            }
            return

        if self.portfolios is None:
            print("  Error: No portfolios constructed. Run step7 first.")
            self.performance_metrics = {
                'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
                'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
                'n_trading_days': 0,
            }
            return
        
        # Run backtest on TEST data (out-of-sample)
        # Critical: this must use test_data, NOT the full price_data
        self.performance_metrics = self.backtest_engine.run(
            portfolios=self.portfolios,
            prices=_test['price_data']['close'],
        )
        
        print(f"\n  [backtest] Out-of-sample results (test period: {getattr(self, '_test_start_date', 'unknown')})")
        for key, value in self.performance_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
        
        print("  [✓] Backtesting complete")
        
    def step9_save_results(self, output_dir: str = None):
        """
        Step 9: Save results to disk.

        Args:
            output_dir: Directory to save results.
                        If None, defaults to `experiments/{YYYYMMDD}/results/`.
        """
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = os.path.join(date_str, "results")
            output_dir = config_path('experiments', output_dir)

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
            from dataclasses import asdict
            serializable = [asdict(round_) for round_ in self.evolution_history]
            with open(f"{output_dir}/evolution_history.json", 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
        
        print("  [✓] Results saved")

        # Persist memory bank to disk
        if self.memory_bank is not None:
            try:
                self.memory_bank.save()
                print("  [✓] Memory bank saved")
            except Exception as e:
                print(f"  [memory] Save failed: {e}")
        
    def run_full_pipeline(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        use_sample: bool = True,
        data_source: str = "auto",
        n_evolution_rounds: int = 5,
        output_dir: str = None,
        split_train_end: str = None,
        split_test_start: str = None,
        forward_period: int = None,
        holding_period: int = None,
        test_data=None,
    ):
        """
        Run the full end-to-end pipeline.

        All hyper-parameters (n_seed_factors, n_evolution_rounds, etc.) are read from config.yaml.
        Use CLI --n-seeds / --n-evolution-rounds to override config values at runtime.

        Args:
            start_date: Start date (real data only)
            end_date: End date (real data only)
            use_sample: True=sample data, False=real data
            data_source: Real data source ('westock', 'akshare', 'tushare', 'auto')
            n_evolution_rounds: Number of evolution rounds (overrides config)
            output_dir: Output directory for step9. None = experiments/{YYYYMMDD}/results/
            split_train_end: Explicit train end date for train/test split
            split_test_start: Explicit test start date for train/test split
            forward_period: Forward return horizon in trading days (None → config or 20)
            holding_period: Backtest holding period in trading days.
                           1 = daily rebalance, 5 = weekly, 20 = monthly.
                           None → use config or default 1.
            test_data: Optional external test data dict to use in step7/step8.
                       If provided, overrides self.test_data from step1 split.
        """
        print("\n" + "=" * 60)
        data_label = "SAMPLE DATA (fast test)" if use_sample else "REAL DATA"
        print(f"  Running Full Pipeline — {data_label}")
        print("=" * 60)

        # Apply CLI overrides to config before step1 (so BacktestEngine picks them up)
        if holding_period is not None:
            self.config.setdefault('backtest', {}).setdefault('trading', {})['holding_period'] = holding_period

        # Run all steps
        self.step1_load_data(
            start_date, end_date,
            use_sample=use_sample,
            data_source=data_source,
            split_train_end=split_train_end,
            split_test_start=split_test_start,
        )
        self.step2_initialize_memory()
        self.step3_generate_factors()
        self.step4_evolve_factors(n_evolution_rounds, forward_period=forward_period)
        self.step4b_retrieve_from_memory()   # retrieve after evolution → augment candidate pool
        self.step5_evaluate_factors()        # debate ALL candidates (incl. memory factors)
        self.step5b_chair_synthesis()     # Chair synthesis (step 5b)
        self.step6_fuse_factors()
        self.step7_construct_portfolio(test_data=test_data)
        self.step8_backtest(test_data=test_data)
        self.step9_save_results(output_dir)
        
        print("\n" + "=" * 60)
        print("  Pipeline Complete!")
        print("=" * 60)
        
        return self.performance_metrics

    def run_test_pipeline(
        self,
        factor_path: str,
        test_data: dict,
        holding_period: int = None,
        context_days: int = None,
    ):
        """
        Load saved factors from JSON and run test-period portfolio construction + backtest.

        This skips step1–step6 (data loading, evolution, debate, fusion) and directly
        loads the trained factor weights/signs from a final_factors.json file, then
        runs step7 (portfolio construction) and step8 (backtest) on the provided
        test_data.

        Args:
            factor_path: Path to final_factors.json (saved by step6_fuse_factors).
            test_data: Dict with keys {'price_data', 'fundamental_data', 'industry_data'}
                       for the test period. May optionally include context days before
                       the real test start date (prepended by split_data). Required —
                       no fallback to self.test_data.
            holding_period: Backtest holding period override (1=daily, 5=weekly, 20=monthly).
                            None → use config or existing engine setting.
            context_days: Number of calendar days prepended before the real test
                          start date. Used for cropping portfolios in step7 so that
                          rolling-window factors have history on day 1. If None,
                          reads from test_data['_meta'] or falls back to config.

        Returns:
            Performance metrics dict from step8.
        """
        import json
        from types import SimpleNamespace

        print("\n" + "=" * 60)
        print("  Running Test Pipeline — loading saved factors")
        print("=" * 60)

        # ── 1. Load fusion result from JSON ──
        print(f"\n[Load] Reading factors from {factor_path}...")
        with open(factor_path, "r", encoding="utf-8") as f:
            fusion_result = json.load(f)

        saved_factors = fusion_result.get("factors", [])
        meta = fusion_result.get("meta", {})
        saved_weights = meta.get("weights", {})
        saved_signs = meta.get("signs", {})

        if not saved_factors:
            print("  [error] No factors found in the JSON file.")
            return None

        print(f"  Loaded {len(saved_factors)} factors, {len(saved_weights)} weights, {len(saved_signs)} signs")

        # ── 2. Reconstruct self.best_factors (for _calculate_factor_values) ──
        # _calculate_factor_values reads .expression from each factor.
        # We use SimpleNamespace to create lightweight objects with .expression.
        self.best_factors = [
            SimpleNamespace(
                name=f["name"],
                expression=f["expression"],
                ic=0.0,
                icir=0.0,
                sharpe=0.0,
                debate_score=f.get("debate_score", 0.0) or 0.0,
            )
            for f in saved_factors
        ]

        # ── 3. Reconstruct self.factor_infos (for step7 iteration) ──
        # step7 iterates self.factor_infos to get factor names and look up
        # weights/signs from the loaded JSON.
        if METHODS_AVAILABLE:
            self.factor_infos = [
                FactorInfo(
                    name=f["name"],
                    expression=f["expression"],
                    debate_score=f.get("debate_score", 0.0) or 0.0,
                    n_periods=f.get("n_periods", 50),
                )
                for f in saved_factors
            ]
        else:
            print("  [error] methods modules not available — cannot create FactorInfo")
            return None

        # ── 4. Set fusion_result (step7 reads weights/signs from this) ──
        self.fusion_result = fusion_result

        # ── 5. Initialize backtest engine if not already done ──
        if self.backtest_engine is None:
            if holding_period is not None:
                self.config.setdefault('backtest', {}).setdefault('trading', {})['holding_period'] = holding_period
            self.backtest_engine = BacktestEngine(
                commission=self.config['backtest']['trading']['commission'],
                slippage=self.config['backtest']['trading']['slippage'],
                holding_period=self.config.get('backtest', {}).get('trading', {}).get('holding_period', 1),
            )
            print("  Initialized backtest engine from config")
        elif holding_period is not None:
            # Override holding period on existing engine
            self.config.setdefault('backtest', {}).setdefault('trading', {})['holding_period'] = holding_period
            self.backtest_engine = BacktestEngine(
                commission=self.config['backtest']['trading']['commission'],
                slippage=self.config['backtest']['trading']['slippage'],
                holding_period=holding_period,
            )
            print(f"  Re-initialized backtest engine with holding_period={holding_period}")

        # ── 5b. Resolve context_days & test_start_date ──
        # context_days tells step7 how many days to crop from the start of
        # computed portfolios. Resolution order:
        #   1. Explicit context_days parameter
        #   2. test_data['_meta']['context_days'] (set by split_data)
        #   3. config['data']['context_days']
        #   4. Fallback to 0 (no cropping — assumes test_data has no context)
        if context_days is not None:
            self._context_days = context_days
        elif test_data.get('_meta', {}).get('context_days') is not None:
            self._context_days = test_data['_meta']['context_days']
        else:
            self._context_days = self.config.get('data', {}).get('context_days', 0)

        # Resolve test_start_date — step7 uses this to crop context dates.
        # Priority: _meta > _test_start_date from a prior step1 > infer from context_days
        meta = test_data.get('_meta', {})
        if meta.get('test_start_date'):
            self._test_start_date = meta['test_start_date']
        elif not hasattr(self, '_test_start_date') or self._test_start_date is None:
            # Infer: skip context_days rows from the price index
            if self._context_days > 0:
                close_idx = test_data['price_data']['close'].index
                inferred_start = close_idx[min(self._context_days, len(close_idx) - 1)]
                self._test_start_date = str(inferred_start.date())
                print(f"  Inferred test_start_date from context_days={self._context_days}: {self._test_start_date}")
            else:
                # No context — use the first date in test_data
                self._test_start_date = str(test_data['price_data']['close'].index[0].date())

        print(f"  context_days={self._context_days}, test_start_date={self._test_start_date}")

        # ── 6. Run step7 (portfolio construction) ──
        self.step7_construct_portfolio(test_data=test_data)

        # ── 7. Run step8 (backtest) ──
        self.step8_backtest(test_data=test_data)

        print("\n" + "=" * 60)
        print("  Test Pipeline Complete!")
        print("=" * 60)

        return self.performance_metrics

    # ──────────────────────────────────────────────────────────────
    # Helper: build MarketState from close DataFrame
    # ──────────────────────────────────────────────────────────────
    def _build_market_state(
        self,
        close_df: pd.DataFrame,
        shibor: Optional[float] = None,
    ) -> Optional['MarketState']:
        """
        Encode a MarketState from a multi-stock (or single-column) close DataFrame.

        Parameters
        ----------
        close_df : pd.DataFrame
            Shape (dates × stocks) **or** single-column DataFrame with 'close'.
        shibor : float, optional
            Overnight SHIBOR rate used to determine LiquidityRegime.
            When None (default), falls back to LiquidityRegime.NORMAL because
            this project does not currently source SHIBOR data.  Pass an actual
            value if you add macro data to the data pipeline.

        Returns
        -------
        MarketState or None on failure.

        Notes
        -----
        corr_matrix is computed here from rolling stock returns so that
        CorrelationRegime reflects the actual cross-sectional correlation
        during the recent window, rather than always defaulting to MEDIUM.
        """
        try:
            # --- Build corr_matrix from recent returns ---
            corr_matrix: Optional[pd.DataFrame] = None
            if close_df is not None and not close_df.empty:
                # Determine whether this is a multi-stock (dates×stocks) DataFrame
                # or a single-column 'close' series
                if "close" not in close_df.columns and close_df.shape[1] > 1:
                    # Multi-stock: compute pairwise correlation of daily returns
                    ret = close_df.pct_change().dropna(how='all')
                    if ret.shape[0] >= 10 and ret.shape[1] >= 2:
                        # Drop all-NaN columns before computing corr
                        ret = ret.dropna(axis=1, how='any')
                        if ret.shape[1] >= 2:
                            corr_matrix = ret.corr()

            market_state = self.market_encoder.encode(
                close_df,
                shibor=shibor,        # None → LiquidityRegime.NORMAL (acceptable default)
                corr_matrix=corr_matrix,  # now computed from actual returns
            )
            return market_state
        except Exception as e:
            print(f"  [warn] Market state encoding failed: {e}")
            return None

    def _calculate_factor_values(
        self,
        price_data: Optional[Dict] = None,
        fundamental_data: Optional[Dict] = None,
        forward_period: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate factor values for all stocks by evaluating each factor's DSL expression.

        Uses the same _FactorExprEvaluator (recursive-descent parser) that FactorBacktester
        uses during evolution, so the computed values are the REAL factor values defined by
        the expression — not keyword-matched proxies.

        Parameters
        ----------
        price_data : dict, optional
            If provided, use this instead of self.price_data.
            Callers should pass train_data['price_data'] for weight computation,
            or self.price_data (full data) for composite score generation.
        fundamental_data : dict, optional
            If provided, use this instead of self.fundamental_data.
            Callers should pass train_data['fundamental_data'] when computing
            factor values on train-period data, to avoid using future fundamental
            values in the weight computation.

        Returns
        -------
        dict[str, DataFrame]
            Each value is a DataFrame of shape (n_dates, n_stocks) indexed by date.
        """
        from methods.evolve import _FactorExprEvaluator

        _price_data = price_data if price_data is not None else self.price_data
        close = _price_data.get('close')

        # ---- Fallback: no valid price data ----
        if close is None or not isinstance(close, pd.DataFrame):
            dates = pd.date_range('2024-01-01', periods=100, freq='B')
            stocks = [f'STOCK_{i:04d}' for i in range(100)]
            return {
                'random_factor': pd.DataFrame(
                    np.random.randn(100, 100), index=dates, columns=stocks,
                )
            }

        # ---- Build unified data map for the expression evaluator ----
        # Mirrors what FactorBacktester.__init__ does so expressions evaluate identically.
        data_map: Dict[str, pd.DataFrame] = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if field in _price_data and isinstance(_price_data[field], pd.DataFrame):
                data_map[field] = _price_data[field]
            else:
                # Pad with NaN so expressions referencing missing fields degrade gracefully
                data_map[field] = pd.DataFrame(
                    np.nan, index=close.index, columns=close.columns
                )
        # Fundamental data (pe, pb, roe, market_cap, …)
        # Use the explicit parameter if provided; otherwise fall back to self.fundamental_data.
        # This allows callers to pass train-period fundamental data to avoid lookahead.
        _fund = fundamental_data if fundamental_data is not None \
            else (getattr(self, 'fundamental_data', {}) or {})
        for key, df in _fund.items():
            if isinstance(df, pd.DataFrame):
                data_map[key] = df
        # Derived fields
        if 'close' in data_map and 'pe' in data_map:
            pe_safe = data_map['pe'].replace(0, np.nan)
            data_map['eps'] = data_map['close'] / pe_safe
        # Forward returns — period must match step4 to keep IC computation consistent.
        # Resolution order: explicit arg → self._forward_period (set in step4) → config → 20
        if forward_period is None:
            forward_period = getattr(self, '_forward_period', None)
        if forward_period is None:
            forward_period = self.config.get('evolution', {}).get('forward_period', 20)
        data_map['forward_returns'] = close.pct_change(forward_period).shift(-forward_period)


        # ---- Factor list ----
        factors = getattr(self, 'best_factors', None)
        if not factors:
            factors = getattr(self, 'generated_factors', [])

        factor_dict: Dict[str, pd.DataFrame] = {}

        if not factors:
            # ---- Default factor suite (no LLM factors available) ----
            _ret = close.pct_change().fillna(0.0)
            _vol = data_map['volume']
            default_factors = [
                ('momentum_20d',      _ret.rolling(20).sum()),
                ('mean_reversion_5d', -_ret.rolling(5).sum()),
                ('volatility_20d',    -_ret.rolling(20).std()),
                ('volume_zscore',     (_vol / _vol.rolling(20).mean() - 1.0).fillna(0.0)),
                ('momentum_5d',       _ret.rolling(5).sum()),
                ('high_low_spread',   (close / close.rolling(10).min() - 1.0).fillna(0.0)),
            ]
            for name, vals in default_factors:
                factor_dict[name] = vals
        else:
            # ---- Evaluate each factor's DSL expression with the real parser ----
            evaluator = _FactorExprEvaluator(data_map)
            seen_exprs: Dict[str, int] = {}   # dedup: expression → count

            for f in factors:
                expr = f['expression'] if isinstance(f, dict) else getattr(f, 'expression', '')
                if not expr or not expr.strip():
                    continue

                # Deduplicate identical expressions by appending a suffix
                count = seen_exprs.get(expr, 0)
                seen_exprs[expr] = count + 1
                factor_key = expr if count == 0 else f'{expr}__dup{count}'

                try:
                    # Reset parser state for each expression (evaluator is stateful)
                    evaluator._tokens = []
                    evaluator._pos = 0
                    vals = evaluator.evaluate(expr)
                    factor_dict[factor_key] = vals
                except Exception as e:
                    print(f"  [warn] Factor expression eval failed ({expr!r}): {e}")
                    # Deterministic random fallback — keeps pipeline alive,
                    # but with a clearly low IC so the fusion stage down-weights it.
                    rng = np.random.RandomState(abs(hash(expr)) % (2 ** 32))
                    factor_dict[factor_key] = pd.DataFrame(
                        rng.randn(len(close), close.shape[1]),
                        index=close.index,
                        columns=close.columns,
                    )

        # For backward compatibility, also store as attribute
        self._factor_dict = factor_dict
        return factor_dict


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
    
    # Generate random portfolios for demo — use actual trading days from price_data
    n_stocks = min(50, pipeline.price_data['close'].shape[1])
    _trading_days = pipeline.price_data['close'].index
    n_dates = min(100, len(_trading_days))
    dates = _trading_days[:n_dates]
    stock_codes = pipeline.price_data['close'].columns[:n_stocks]
    
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
    parser = argparse.ArgumentParser(
        description='AAAI 2027 LLM Multi-Factor Stock Selection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          Quick demo (sample data, no LLM)
  python main.py --full                                   Full pipeline with sample data
  python main.py --full --real                            Full pipeline with real data
  python main.py --full --real --source westock            Real data via westock (WorkBuddy)
  python main.py --full --real --start 2023-01-01 --end 2024-12-31
  python main.py --full --real --force-refresh             Skip cache, re-download
  python main.py --test --factor-path experiments/20260601/results/final_factors.json
  python main.py --test --factor-path PATH --real --start 2023-01-01 --end 2024-12-31
  python main.py --test --factor-path PATH --holding-period 5
        """,
    )

    parser.add_argument(
        '--full', action='store_true',
        help='Run the full end-to-end pipeline (default: quick demo)',
    )
    parser.add_argument(
        '--real', action='store_true', default=False,
        help='Use real A-share data instead of sample data (default: sample)',
    )
    parser.add_argument(
        '--sample', dest='real', action='store_false',
        help='Use sample/synthetic data (default)',
    )
    parser.add_argument(
        '--source', type=str, default='auto',
        choices=['auto', 'westock', 'akshare', 'tushare'],
        help='Real data source: auto (try westock→akshare→tushare), westock, akshare, tushare (default: auto)',
    )
    parser.add_argument(
        '--force-refresh', action='store_true', default=False,
        help='Skip cache and re-download real data',
    )
    parser.add_argument(
        '--start', type=str, default='2022-01-01',
        help='Start date for real data (YYYY-MM-DD, default: 2022-01-01)',
    )
    parser.add_argument(
        '--end', type=str, default='2024-12-31',
        help='End date for real data (YYYY-MM-DD, default: 2024-12-31)',
    )
    parser.add_argument(
        '--universe', type=str, default='hs300',
        choices=['hs300', 'zz500', 'all_a'],
        help='Stock universe for real data (default: hs300)',
    )
    parser.add_argument(
        '--n-seeds', type=int, default=None,
        help='Override llm.generator.n_seeds from config (default: use config value)',
    )
    parser.add_argument(
        '--n-evolution-rounds', type=int, default=5,
        help='Number of evolution rounds (default: 5)',
    )
    parser.add_argument(
        '--forward-period', type=int, default=None,
        help='Forward return horizon in trading days (None → config or 20)',
    )
    parser.add_argument(
        '--holding-period', type=int, default=None,
        help='Backtest holding period in trading days: 1=daily rebalance (default), 5=weekly, 20=monthly',
    )
    parser.add_argument(
        '--n-best-factors', type=int, default=None,
        help='Override evolution.n_best_factors (default: use config value)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: experiments/YYYYMMDD/results/)',
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run test pipeline: load saved factors from JSON and run step7+step8 on test data',
    )
    parser.add_argument(
        '--factor-path', type=str, default=None,
        help='Path to final_factors.json for --test mode '
             '(default: experiments/YYYYMMDD/results/final_factors.json)',
    )
    parser.add_argument(
        '--context-days', type=int, default=None,
        help='Context days prepended before test_start_date for rolling-window factors '
             '(None → auto-detect from data or config)',
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)',
    )
    args = parser.parse_args()

    if args.full:
        # Full end-to-end pipeline
        pipeline = AAAI2027Pipeline(config_path=args.config)
        
        # Apply CLI overrides to config
        if args.n_seeds is not None:
            pipeline.config['evolution']['n_seeds'] = args.n_seeds
        if args.n_evolution_rounds != 5:
            pipeline.config['evolution']['max_rounds'] = args.n_evolution_rounds
        if args.n_best_factors is not None:
            pipeline.config['evolution']['n_best_factors'] = args.n_best_factors
        
        metrics = pipeline.run_full_pipeline(
            start_date=args.start,
            end_date=args.end,
            use_sample=not args.real,
            data_source=args.source,
            n_evolution_rounds=args.n_evolution_rounds,
            output_dir=args.output_dir,
            forward_period=args.forward_period,
            holding_period=args.holding_period,
        )
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
        else:
            print("\nPipeline completed, but no metrics available.")

    elif args.test:
        # ── Test pipeline: load saved factors → step7 + step8 on test data ──
        pipeline = AAAI2027Pipeline(config_path=args.config)

        # Resolve factor_path: CLI arg → auto-detect from experiments dir
        if args.factor_path:
            factor_path = args.factor_path
        else:
            # Auto-detect: look for the most recent experiments/*/results/final_factors.json
            from pathlib import Path as _Path
            experiments_dir = _Path('experiments')
            candidates = sorted(
                experiments_dir.glob('*/results/final_factors.json'),
                key=lambda p: p.parent.parent.name,  # sort by date dir name
                reverse=True,
            )
            if candidates:
                factor_path = str(candidates[0])
                print(f"Auto-detected factor path: {factor_path}")
            else:
                print("Error: --factor-path not specified and no final_factors.json found "
                      "under experiments/. Run --full first to generate one.")
                exit(1)

        if not os.path.exists(factor_path):
            print(f"Error: factor path does not exist: {factor_path}")
            exit(1)

        # Load data (step1) to get test_data for the test period.
        # The split (train/test) is controlled by split_train_end / split_test_start
        # from config, which defaults to the last ~2 years as test.
        pipeline.step1_load_data(
            start_date=args.start,
            end_date=args.end,
            use_sample=not args.real,
            data_source=args.source,
            force_refresh=args.force_refresh,
        )

        if pipeline.test_data is None:
            print("Error: step1_load_data did not produce test_data (no data loaded?).")
            exit(1)

        # Override holding period if specified
        if args.holding_period is not None:
            pipeline.config.setdefault('backtest', {}).setdefault('trading', {})['holding_period'] = args.holding_period

        metrics = pipeline.run_test_pipeline(
            factor_path=factor_path,
            test_data=pipeline.test_data,
            holding_period=args.holding_period,
            context_days=args.context_days,
        )
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
        else:
            print("\nTest pipeline completed, but no metrics available.")
    else:
        # Quick demo (default)
        metrics = run_demo()
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
        else:
            print("\nDemo completed, but no metrics available.")
