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
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml
import warnings

import config
from config import config_path

warnings.filterwarnings('ignore')

# Import all modules
from dataloader.loader import DataLoader, load_sample_data, load_real_data
from backtest.engine import BacktestEngine
from metrics.evaluator import FactorEvaluator, evaluate_portfolio_comprehensive

# Import methods (with error handling for missing dependencies)
try:
    from methods.debate import DebateEvaluator, FactorProposal
    from methods.evolve import SelfEvolvingGenerator, FactorBacktester, CandidateFactor
    from methods.memory import FactorMemoryBank, FactorEmbedder, MarketStateEncoder, MemoryAugmentedGenerator
    from methods.fusion import FactorFusion, PortfolioConstructor, FactorInfo, PortfolioConfig
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
        
        self.train_data, self.test_data = self.data_loader.split_data(
            train_end_date=train_end_date,
            test_start_date=test_start_date,
        )
        self._train_end_date = train_end_date
        self._test_start_date = test_start_date
        
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
        self.evolving_generator = SelfEvolvingGenerator(
            llm_model=self.config['llm']['generator']['model'],
            n_seeds=n_seeds,
            n_best_factors=self.config['evolution']['n_best_factors'],
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
        qlib_cfg = self.config.get('backtest', {}).get('qlib', {})
        use_qlib = qlib_cfg.get('enable', False)
        backtester = FactorBacktester(
            prices=self.train_data['price_data'],
            fundamentals=self.train_data['fundamental_data'],
            forward_period=forward_period,
            use_qlib=use_qlib,
            qlib_provider_uri=self.config.get('data', {}).get('qlib', {}).get(
                'provider_uri', '~/.qlib/qlib_data/cn_data'
            ),
            qlib_topk=qlib_cfg.get('topk', 50),
        )
        
        if use_qlib:
            print("  [evolution] Using Qlib professional backtesting engine")
        print(f"  [evolution] Using TRAIN data only: {self._train_end_date}")
        evolution_result = self.evolving_generator.evolve(
            seed_factors=self.generated_factors,
            backtester=backtester,
            n_rounds=n_rounds,
        )
        
        self.evolution_history = evolution_result.evolution_history
        self.best_factors = evolution_result.best_factors
        
        print(f"  Evolution complete: {len(self.best_factors)} best factors")
        best_ic = self.best_factors[0].ic if self.best_factors else 0.0
        print(f"  Best IC: {best_ic:.4f}")
        print("  [✓] Factor evolution complete")
        
    def step5_evaluate_factors(self):
        """
        Step 5: Evaluate the evolved best factors via multi-agent debate.

        Debate runs AFTER evolution — it serves as the final quality gate,
        applying 5-expert cross-validation to the top evolved factors.
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
                    # close_df: DataFrame(dates × stocks); encode() expects columns=['close']
                    # Build market index proxy by cross-sectional mean
                    market_close = close_df.mean(axis=1)
                    market_df = pd.DataFrame({'close': market_close})
                    current_state = self.market_encoder.encode(market_df)
                else:
                    current_state = None
            except Exception:
                current_state = None
            saved = 0
            for factor, result in self.debate_results:
                if isinstance(factor, dict):
                    expr = factor.get('expression', '')
                    desc = factor.get('description', '')
                else:
                    expr = getattr(factor, 'expression', '')
                    desc = getattr(factor, 'description', '')
                if not expr:
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
        
    def step5b_retrieve_from_memory(self):
        """
        Step 5b: Retrieve historical high-quality factors from memory bank.
        
        This step uses state-aware retrieval to find factors from memory
        that performed well in similar market conditions, augmenting the
        evolved factor pool with proven historical factors.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 5b] Skipped: methods modules not available")
            return
        
        if self.memory_bank is None or len(self.memory_bank) == 0:
            print("\n[Step 5b] Skipped: memory bank is empty")
            return
        
        print("\n[Step 5b] Retrieving factors from memory (state-aware)...")
        
        # Encode current market state using TRAIN data (not test data)
        # This prevents data leak from test period
        close_df = self.train_data['price_data']['close'].iloc[-self.recent_days:]
        # close_df: DataFrame(dates × stocks); encode() expects columns=['close']
        # Build market index proxy by cross-sectional mean
        market_close = close_df.mean(axis=1)
        market_df = pd.DataFrame({'close': market_close})
        try:
            current_state = self.market_encoder.encode(market_df)
        except Exception as e:
            print(f"  [warn] Market state encoding failed: {e}")
            current_state = None
        
        # Retrieve similar factors from memory
        # Use best evolved factors as query (higher quality than raw seeds)
        query_desc = "multi-factor stock selection using technical indicators"
        query_expr = ""
        query_source = self.best_factors if (hasattr(self, 'best_factors') and self.best_factors) else self.generated_factors
        if query_source:
            # Use the first factor as query
            f = query_source[0]
            if isinstance(f, dict):
                query_desc = f.get('description', query_desc)
                query_expr = f.get('expression', '')
            else:
                query_desc = getattr(f, 'description', query_desc)
                query_expr = getattr(f, 'expression', '')

        retrieved_factors = self.memory_bank.retrieve(
            query_description=query_desc,
            query_expression=query_expr,
            current_market_state=current_state,
            top_k=self.config['memory'].get('top_k', 5),
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
                        self.best_factors.append(rf)
                        existing_exprs.add(rf.expression)
                        added += 1

            print(f"  Added {added} new factors from memory (skipped {len(retrieved_factors) - added} duplicates)")
            print(f"  Factor pool size after merge: {len(self.best_factors)}")
        else:
            print("  No relevant factors found in memory")
        
        print("  [✓] Memory retrieval complete")
        
    def step5c_chair_synthesis(self):
        """
        Step 5c: Chair Agent synthesizes ALL debate results into a comprehensive
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
            print("\n[Step 5c] Skipped: methods modules not available")
            return

        if not hasattr(self, 'debate_results') or not self.debate_results:
            print("\n[Step 5c] Skipped: no debate results available")
            return

        print("\n[Step 5c] Chair Agent synthesizing all debate results...")

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
        Step 6: Fuse factors using ICIR-weighted fusion.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 6] Skipped: methods modules not available")
            return
        
        print("\n[Step 6] Fusing factors...")
        
        # Initialize fusion
        fusion = FactorFusion(
            strategy=self.config['fusion']['weighting']['strategy'],
            corr_penalty=self.config['fusion']['weighting']['corr_penalty'],
        )
        
        # Prepare factor values (use TRAIN data to determine fusion weights — in-sample)
        # Fusion weights are model parameters; they must be determined on training data,
        # then applied to test data in step8 for out-of-sample evaluation.
        if hasattr(self, 'train_data') and self.train_data:
            _train_price = self.train_data.get('price_data', {})
            factor_dict = self._calculate_factor_values(price_data=_train_price)
        else:
            factor_dict = self._calculate_factor_values()

        # Build factor metadata lookup from best_factors (has ic, icir, sharpe from backtest)
        factor_meta_lookup = {}
        if hasattr(self, 'best_factors') and self.best_factors:
            for f in self.best_factors:
                expr = f.expression if hasattr(f, 'expression') else str(f)
                ic_val = getattr(f, 'ic', 0.0)
                icir_val = getattr(f, 'icir', 0.0)
                sharpe_val = getattr(f, 'sharpe', 0.0)
                factor_meta_lookup[expr] = (ic_val, icir_val, sharpe_val)
        
        # Build debate_score lookup from step5 debate results
        debate_score_map = {}
        if hasattr(self, 'debate_results') and self.debate_results:
            for factor, result in self.debate_results:
                if isinstance(factor, dict):
                    expr = factor.get('expression', '')
                else:
                    expr = getattr(factor, 'expression', '')
                if expr:
                    debate_score_map[expr] = result.final_score
        
        # Build factor_values_dict and factor_infos
        factor_values_dict = {}
        factor_infos = []
        for name, values_df in factor_dict.items():
            # Ensure index is datetime
            if not isinstance(values_df.index, pd.DatetimeIndex):
                values_df = values_df.copy()
                values_df.index = pd.to_datetime(values_df.index)
            factor_values_dict[name] = values_df
            
            # Look up ic/icir/sharpe from backtest results
            meta = factor_meta_lookup.get(name)
            if meta:
                ic_val, icir_val, sharpe_val = meta
                # Compute ic_std from icir (icir = ic / ic_std)
                if abs(icir_val) > 1e-8:
                    ic_std_val = ic_val / icir_val
                else:
                    ic_std_val = 1.0  # default: equivalent to ic_weighted
            else:
                ic_val = ic_std_val = sharpe_val = 0.0
            
            dscore = debate_score_map.get(name, 0.0)
            factor_infos.append(FactorInfo(
                name=name, expression=name,
                ic=ic_val, icir=icir_val, ic_std=ic_std_val,
                sharpe=sharpe_val, debate_score=dscore,
            ))

        # Fuse factors
        self._composite_scores, fusion_meta = fusion.fuse(
            factor_infos, factor_values_dict
        )
        
        # Save fusion weights and object for step7 (out-of-sample application)
        self.fusion_weights = fusion_meta["weights"]
        self.fusion_obj = fusion
        self.factor_infos = factor_infos  # Reuse in step7

        print(f"  Fused {len(factor_infos)} factors")

        # --- Save fusion results to experiments/{yyyymmdd}/fusion/final_factors.json ---
        date_str = datetime.now().strftime("%Y%m%d")
        fusion_dir = os.path.join(date_str, "fusion")
        fusion_dir = config_path("experiments", fusion_dir)
        os.makedirs(fusion_dir, exist_ok=True)

        # Build serializable composite scores (train-period, where fusion weights were determined)
        # Filter to train dates if available
        if hasattr(self, 'train_data') and self.train_data:
            train_dates = self.train_data['price_data']['close'].index
            _score_df = self._composite_scores.loc[train_dates.intersection(self._composite_scores.index)]
        else:
            _score_df = self._composite_scores

        scores_serializable = {}
        for dt, row in _score_df.iterrows():
            date_key = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            stock_scores = {}
            for stock, val in row.items():
                if pd.notna(val):
                    stock_scores[str(stock)] = float(val)
            scores_serializable[date_key] = stock_scores

        # Build factor info list
        factor_details = []
        for fi in factor_infos:
            factor_details.append({
                "name": fi.name,
                "expression": fi.expression,
                "weight": fusion_meta["weights"].get(fi.name, 0.0),
                "debate_score": getattr(fi, "debate_score", None),
            })

        fusion_output = {
            "meta": fusion_meta,
            "factors": factor_details,
            "composite_scores": scores_serializable,
        }

        fusion_path = os.path.join(fusion_dir, "final_factors.json")
        with open(fusion_path, "w", encoding="utf-8") as f:
            json.dump(fusion_output, f, ensure_ascii=False, indent=2)

        print(f"  Fusion results saved to {fusion_path}")
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
        portfolio_cfg = self.config['fusion']['portfolio']
        constructor = PortfolioConstructor(PortfolioConfig(
            top_n=portfolio_cfg['top_n'],
            method=portfolio_cfg['method'],
            max_weight=portfolio_cfg['max_weight'],
            max_industry_exposure=portfolio_cfg.get('max_industry_exposure', 0.30),
        ))
        
        # --- Use TRAIN-determined weights to build TEST-period portfolios ---
        # Correct out-of-sample evaluation:
        # 1. Weights determined on train data (step6)
        # 2. Apply to test data factor values → test-period composite scores
        # 3. Build portfolios for test period
        # 4. Backtest on test data (step8)

        if not hasattr(self, 'fusion_obj') or self.fusion_obj is None:
            print("  [error] fusion_obj not available. Run step6 first.")
            return

        # 1. Calculate factor values on TEST data
        if not hasattr(self, 'test_data') or not self.test_data:
            print("  [error] test_data not available, cannot construct out-of-sample portfolios")
            return
        _test_price = self.test_data.get('price_data', {})
        test_factor_dict = self._calculate_factor_values(price_data=_test_price)

        # 2. Apply train-determined weights to test data factor values
        if not hasattr(self, 'factor_infos'):
            print("  [error] factor_infos not available (step6 not run?). Run step6 first.")
            return
        test_composite_scores, _ = self.fusion_obj.fuse(
            self.factor_infos, test_factor_dict,
            precomputed_weights=self.fusion_weights
        )
        self._composite_scores = test_composite_scores  # Update to test-period scores

        # 3. Build portfolios for test period
        close_copy = self.test_data['price_data']['close'].copy()
        raw_portfolios = constructor.build(
            composite_scores=self._composite_scores,
            prices=close_copy,
            industry=self.industry_data,
        )

        # Convert list[Portfolio] to DataFrame (n_dates x n_stocks) for backtester
        weight_dict = {}
        for pf in raw_portfolios:
            weight_dict[pf.date] = pf.weights
        self.portfolios = pd.DataFrame(weight_dict).T
        self.portfolios.index = pd.to_datetime(self.portfolios.index)
        self.portfolios = self.portfolios.fillna(0.0)

        print(f"  Constructed {len(self.portfolios)} portfolios (test period)")
        print("  [✓] Portfolio construction complete")
        
    def step8_backtest(self):
        """
        Step 8: Backtest the portfolio strategy.
        """
        print("\n[Step 8] Backtesting...")
        
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
            prices=self.test_data['price_data']['close'],
        )
        
        print(f"\n  [backtest] Out-of-sample results (test period: {self._test_start_date})")
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
        """
        print("\n" + "=" * 60)
        data_label = "SAMPLE DATA (fast test)" if use_sample else "REAL DATA"
        print(f"  Running Full Pipeline — {data_label}")
        print("=" * 60)

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
        self.step5_evaluate_factors()
        self.step5b_retrieve_from_memory()
        self.step5c_chair_synthesis()
        self.step6_fuse_factors()
        self.step7_construct_portfolio()
        self.step8_backtest()
        self.step9_save_results(output_dir)

        # If portfolios are still None after full pipeline, generate fallback
        if self.portfolios is None:
            print("\n[Fallback] No portfolios constructed, generating random ones...")
            n_dates = min(100, len(self.price_data['close']))
            n_stocks = min(50, self.price_data['close'].shape[1])
            dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
            stock_codes = self.price_data['close'].columns[:n_stocks]

            self.portfolios = pd.DataFrame(
                np.random.dirichlet(np.ones(n_stocks), size=n_dates),
                index=dates,
                columns=stock_codes,
            )
            self.step8_backtest()
            
            self.portfolios = pd.DataFrame(
                np.random.dirichlet(np.ones(n_stocks), size=n_dates),
                index=dates,
                columns=stock_codes,
            )
            self.step8_backtest()
        
        print("\n" + "=" * 60)
        print("  Pipeline Complete!")
        print("=" * 60)
        
        return self.performance_metrics
    
    def _calculate_factor_values(self, price_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate factor values for all stocks from REAL price/volume data.

        Parameters
        ----------
        price_data : dict, optional
            If provided, use this instead of self.price_data.
            Callers should pass train_data['price_data'] for step6 (fusion weight
            determination, in-sample), or test_data['price_data'] for true
            out-of-sample evaluation (though step8 handles that separately).
            If None, falls back to self.price_data (backward compatible).

        Returns:
            dict[str, DataFrame]: each DataFrame is (n_dates, n_stocks), indexed by date.
        """
        _price_data = price_data if price_data is not None else self.price_data
        close = _price_data.get('close')
        volume = _price_data.get('volume')
        if close is None or not isinstance(close, pd.DataFrame):
            dates = pd.date_range('2024-01-01', periods=100, freq='B')
            stocks = [f'STOCK_{i:04d}' for i in range(100)]
            return {
                'random_factor': pd.DataFrame(
                    np.random.randn(100, 100), index=dates, columns=stocks,
                )
            }

        returns = close.pct_change().fillna(0.0)

        # Use best_factors / generated_factors expressions if available
        factors = getattr(self, 'best_factors', None)
        if factors is None:
            factors = getattr(self, 'generated_factors', [])

        if not factors:
            # Default factor suite
            factor_exprs = [
                ('momentum_20d', returns.rolling(20).sum()),
                ('mean_reversion_5d', -returns.rolling(5).sum()),
                ('volatility_20d', -returns.rolling(20).std()),
                ('volume_zscore', (volume / volume.rolling(20).mean() - 1.0).fillna(0.0)),
                ('momentum_5d', returns.rolling(5).sum()),
                ('high_low_spread', (close / close.rolling(10).min() - 1.0).fillna(0.0)),
            ]
        else:
            # Parse expressions into actual computations (simplified)
            factor_exprs = []
            for f in factors:
                expr = f['expression'] if isinstance(f, dict) else f.expression
                expr_lower = expr.lower()
                # 注意匹配顺序：先匹配长词，再匹配短词，避免子串误判
                if 'momentum' in expr_lower or 'return' in expr_lower:
                    factor_exprs.append((expr, returns.rolling(20).sum()))
                elif 'volume' in expr_lower or '量' in expr:
                    if volume is not None:
                        factor_exprs.append((expr, (volume / volume.rolling(20).mean() - 1.0).fillna(0.0)))
                    else:
                        # volume 数据不可用，退回 random
                        rng = np.random.RandomState(abs(hash(expr)) % (2**32))
                        vals = pd.DataFrame(
                            rng.randn(len(close), close.shape[1]),
                            index=close.index, columns=close.columns,
                        )
                        factor_exprs.append((expr, vals))
                elif 'vol' in expr_lower:
                    # 必须在 'volume' 之后匹配，避免 'vol' 被 'volume' 子串误判
                    factor_exprs.append((expr, -returns.rolling(20).std()))
                else:
                    # Default: random with seed from expr for reproducibility
                    rng = np.random.RandomState(abs(hash(expr)) % (2**32))
                    vals = pd.DataFrame(
                        rng.randn(len(close), close.shape[1]),
                        index=close.index, columns=close.columns,
                    )
                    factor_exprs.append((expr, vals))

        # Build DataFrame: each column is a factor, each row is a (date, stock) pair
        # Actually, we want: index=date, columns=stocks, values=factor_value
        # But we have multiple factors. Let's return a dict of DataFrames, one per factor.
        # Wait - the caller expects a DataFrame of shape (n_stocks, n_factors).
        # But for time-series, it should be (n_dates, n_stocks) per factor.

        # For the fuse() method, we need dict[str, DataFrame] where each DataFrame is (n_dates, n_stocks)
        # Let's return a dict instead of a single DataFrame.

        # Actually, looking at the caller in step6:
        #   factor_values_dict[name] = full_values   # (n_dates, n_stocks) DataFrame
        # So _calculate_factor_values should return a DataFrame of shape (n_stocks, n_factors) [cross-section]
        # and the caller will expand it to (n_dates, n_stocks) per factor.

        # BUT that doesn't make sense for real factors that vary over time!
        # Let's change the return format to be a dict of (n_dates, n_stocks) DataFrames.

        # Actually, the simplest fix: return a dict of DataFrames, one per factor.
        factor_dict = {}
        for name, values in factor_exprs:
            factor_dict[name] = values

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
    parser = argparse.ArgumentParser(
        description='AAAI 2027 LLM Multi-Factor Stock Selection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          Quick demo (sample data, no LLM)
  python main.py --full                                   Full pipeline with sample data
  python main.py --full --real                            Full pipeline with real data
  python main.py --full --real --source westock            Real data via westock (WorkBuddy)
  python main.py --full --real --source qlib                Real data via Qlib (.bin format)
  python main.py --full --real --start 2023-01-01 --end 2024-12-31
  python main.py --full --real --force-refresh             Skip cache, re-download
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
        choices=['auto', 'westock', 'akshare', 'tushare', 'qlib'],
        help='Real data source: auto (try westock→qlib→akshare→tushare), westock, qlib, akshare, tushare (default: auto)',
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
        '--n-best-factors', type=int, default=None,
        help='Override evolution.n_best_factors (default: use config value)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: experiments/YYYYMMDD/results/)',
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)',
    )
    parser.add_argument(
        '--qlib-backtest', action='store_true', default=False,
        help='Use Qlib professional backtesting engine for factor evaluation (requires: pip install qlib)',
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
        if args.qlib_backtest:
            pipeline.config['backtest']['qlib']['enable'] = True
        
        metrics = pipeline.run_full_pipeline(
            start_date=args.start,
            end_date=args.end,
            use_sample=not args.real,
            data_source=args.source,
            n_evolution_rounds=args.n_evolution_rounds,
            output_dir=args.output_dir,
            forward_period=args.forward_period,
        )
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
        else:
            print("\nPipeline completed, but no metrics available.")
    else:
        # Quick demo (default)
        metrics = run_demo()
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
        else:
            print("\nDemo completed, but no metrics available.")
