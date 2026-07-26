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
    python main.py --full --train-start 2023-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-12-31
    python main.py --test --factor-path experiments/YYYYMMDD/results/final_factors.json
    python main.py --test --factor-path PATH --train-start 2023-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-12-31
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
import config as global_config  # module holding experiment-dir tags (universe/start/end)

warnings.filterwarnings('ignore')

# Import all modules
from dataloader.loader import DataLoader, load_sample_data, retrieve_dataset
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
        universe: str = None,
        use_sample: bool = True,
        train_start_date: str = None,
        train_end_date: str = None,
        test_start_date: str = None,
        test_end_date: str = None,
        forward_period: int = None,
        holding_period: int = None,
    ):
        """
        Step 1: Load and preprocess data.

        Data source selection (single decision point):
        - use_sample=True  → fast synthetic data for testing (n_stocks=100, n_days=500)
        - use_sample=False → real data RETRIEVED from the local dataset store
                             (datasets/{universe}_{start}_{end}.pkl). Pre-fetch it once
                             with `python load_datasets.py ...`; load_data() never hits
                             the network.

        Args:
            universe: Stock universe (hs300, zz500, all_a), for real data only.
                      None → read from config['data']['universe'].
            use_sample: True=sample data, False=real data
            force_refresh: Skip cache and re-download real data
            train_start_date: Explicit train start date (overrides config)
            train_end_date: Explicit train end date (overrides config)
            test_start_date: Explicit test start date (overrides config)
            test_end_date: Explicit test end date (overrides config)
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
            # Real data: retrieve the pre-fetched archive from the local store.
            # The actual download + preprocessing happens once in
            # `python load_datasets.py ...` (DataLoader.fetch_and_store_dataset).
            # If the archive is missing, retrieve_dataset raises a FileNotFoundError
            # that tells you exactly which `load_datasets.py` command to run.
            _univ = self.config['data']['universe']
            _data = self.config['data']

            # Train/test windows (explicit params override config). Passed through
            # to retrieve_dataset so the returned bundle honors the SAME 4
            # boundaries as split_data() below — mirrors the baselines' load_data().
            # (bundle.train/.test are re-carved by split_data() for the context
            # window, but keeping the slices consistent avoids surprises.)
            _train_start = train_start_date or _data.get('train_start_date')
            _train_end = train_end_date or _data.get('train_end_date')
            _test_start = test_start_date or _data.get('test_start_date')
            _test_end = test_end_date or _data.get('test_end_date')
            bundle = retrieve_dataset(
                universe=universe or _univ.get('index'),
                train_start=_train_start,
                train_end=_train_end,
                test_start=_test_start,
                test_end=_test_end,
            )
            self.price_data, self.fundamental_data, self.industry_data = bundle.full
            self.train_data = bundle.train
            self.test_data = bundle.test
            global_config.train_start_date = train_start_date or _data.get('train_start_date') or 'na'
            global_config.train_end_date = train_end_date or _data.get('train_end_date') or 'na'
            global_config.test_start_date = test_start_date or _data.get('test_start_date') or 'na'
            global_config.test_end_date = test_end_date or _data.get('test_end_date') or 'na'
            self._data_source_label = "real"
            print(f"  [real] Loaded full: {self.price_data['close'].shape}  "
                  f"train: {bundle.train[0]['close'].shape}  "
                  f"test: {bundle.test[0]['close'].shape}")

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

        # Set global experiment-directory tags so config_path("experiments", ...)
        # emits the real universe/date range instead of "NA_NA_NA".
        # Mirrors the side effect of DataLoader.load_data() (dataloader/loader.py
        # L363-365), which is bypassed here because data is assigned directly.
        _univ_cfg = self.config['data']['universe']
        global_config.universe = universe or _univ_cfg.get('index', 'hs300')

        # Forward/holding tags so experiment dirs match the baseline format
        # experiments/{universe}_{start}_{end}_forward-{fp}_holding-{hp}/...
        # CLI override (param) takes priority; otherwise fall back to config.yaml.
        global_config.forward_period = (
            forward_period
            if forward_period is not None
            else self.config.get('evolution', {}).get('forward_period', 10)
        )
        global_config.holding_period = (
            holding_period
            if holding_period is not None
            else self.config.get('backtest', {}).get('trading', {}).get('holding_period', 1)
        )

        # Read split config from explicit params, config.yaml, or defaults
        train_start_date = train_start_date or self.config.get('data', {}).get('train_start_date', None)
        train_end_date = train_end_date or self.config.get('data', {}).get('train_end_date', None)
        test_start_date = test_start_date or self.config.get('data', {}).get('test_start_date', None)
        test_end_date = test_end_date or self.config.get('data', {}).get('test_end_date', None)
        
        # if train_end_date is None:
        #     # Default: use 80/20 split (80% train, 20% test)
        #     # For default date range 2019-01-01 to 2024-12-31, this is ~2023-12-31
        #     dates = self.price_data['close'].index
        #     split_idx = int(len(dates) * 0.8)
        #     train_end_date = dates[split_idx - 1].strftime('%Y-%m-%d')
        #     test_start_date = dates[split_idx].strftime('%Y-%m-%d')
        #     print(f"  [split] Using default 80/20 split: train_end={train_end_date}, test_start={test_start_date}")
        # elif test_start_date is None:
        #     # train_end specified but test_start not — set test_start to day after
        #     test_start_date = train_end_date
        #     print(f"  [split] Using explicit split: train_end={train_end_date}, test_start={test_start_date}")
        # else:
        #     print(f"  [split] Using explicit split: train_end={train_end_date}, test_start={test_start_date}")
        
        # Context days: prepend N training days to test_data so rolling-window
        # factor expressions (ts_mean, ts_std, etc.) have history on day 1.
        context_days = self.config.get('data', {}).get('context_days', 30)
        self.train_data, self.test_data = self.data_loader.split_data(
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            context_days=context_days,
        )
        self._train_start_date = train_start_date
        self._train_end_date = train_end_date
        self._test_start_date = test_start_date
        self._test_end_date = test_end_date
        self._context_days = context_days
        
        # --- Save train/test data to data directory ---
        self._save_split_data(
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
        
        print(f"  [✓] Train/Test split complete")
        print(f"       Train: {train_start_date or '(archive start)'} → {train_end_date}")
        print(f"       Test:  {test_start_date} → {test_end_date or '(archive end)'}")
        print("  [✓] Data loading complete")
        
    def _save_split_data(self, train_start_date: str = None, train_end_date: str = None,
                         test_start_date: str = None, test_end_date: str = None):
        """
        Persist train/test data as CSV files under
        data/{universe}_{train_start}_{test_end}/train/ and .../test/.

        Directory structure (window-keyed, mirrors the experiments/ convention):
            data/{universe}_{train_start}_{test_end}/train/price/close.csv, open.csv, ...
            data/{universe}_{train_start}_{test_end}/train/fundamental/pe.csv, pb.csv, ...
            data/{universe}_{train_start}_{test_end}/train/industry.csv
            data/{universe}_{train_start}_{test_end}/test/  (same structure)
            data/{universe}_{train_start}_{test_end}/split_info.json  — metadata
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
        
        # Window-keyed data tag so the exported train/test CSVs land under
        # data/{universe}_{train_start}_{test_end}/ — matching the experiments/
        # directory convention. Do NOT reuse config_path("data", ...): that
        # branch is intentionally date-blind for raw/cache artifacts, so the
        # split export must build its own window-tagged path.
        _data_tag = f"{global_config.universe}_{self._train_start_date or 'NA'}_{self._test_end_date or 'NA'}"

        for split, data in [("train", self.train_data), ("test", self.test_data)]:
            root = os.path.join("data", _data_tag, split)
            os.makedirs(root, exist_ok=True)
            
            n_price = _save_category(data.get('price_data'), root, "price")
            n_fund = _save_category(data.get('fundamental_data'), root, "fundamental")
            _save_series(data.get('industry_data'), root, "industry.csv")
            
            print(f"  [save] {split}: {n_price} price CSVs + {n_fund} fundamental CSVs → {root}/")
        
        # Save split metadata
        train_dates = self.train_data['price_data']['close'].index
        test_dates = self.test_data['price_data']['close'].index
        
        split_info = {
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "train_dates": f"{train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')}",
            "test_dates": f"{test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}",
            "train_days": int(len(train_dates)),
            "test_days": int(len(test_dates)),
            "train_n_stocks": self.train_data['price_data']['close'].shape[1],
            "test_n_stocks": self.test_data['price_data']['close'].shape[1],
            "saved_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        info_path = os.path.join("data", _data_tag, "split_info.json")
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
        forward_period: int = None,
    ):
        """
        Step 3: Generate factors using LLM.

        The number of factors is read from config: evolution.n_seed_factors

        Args:
            forward_period: Forward return horizon in trading days
                            (None → config['evolution']['forward_period'] or 10).
                            Threaded from args.forward_period via run_full_pipeline
                            so the CLI --forward-period flag is honored in Step 3
                            (previously Step 3 read config directly and ignored it).

        Memory augmentation is enabled implicitly when ``n_seeds_memory_augment > 0``
        (read from config) AND the memory bank has at least one entry. Set the
        count to 0 to skip memory augmentation entirely — no separate boolean flag
        needed.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 3] Skipped: methods modules not available")
            return
        
        evo_cfg = self.config['evolution']
        n_seeds_hypothesis = evo_cfg.get('n_seeds_hypothesis', 0)
        n_seeds_memory_augment = evo_cfg.get('n_seeds_memory_augment', 0)
        n_shots = evo_cfg.get('n_shots', 3)

        # Resolve forward_period: explicit arg > config.yaml > default 10.
        # Mirrors step4_evolve_factors / step1_load_data so Step 3 uses the same
        # horizon that may have been overridden via args.forward_period, instead
        # of silently reading config and ignoring the CLI flag. Persist to
        # self._forward_period for downstream steps (step6 reads it).
        if forward_period is None:
            forward_period = self.config.get('evolution', {}).get('forward_period', 10)
        fwd = forward_period
        self._forward_period = forward_period

        # Step 3 generates SEED factors only (hypothesis-driven + optional
        # memory-augmented). Alpha101 / FAMA-101 factor generation lives in
        # step4c_retrieve_alpha101 (post-debate; does NOT evolve/debate).
        print(f"\n[Step 3] Generating seed factors - "
              f"hypothesis: {n_seeds_hypothesis}, "
              f"memory-augment: {n_seeds_memory_augment}")

        # Initialize the evolving generator (seeds for Step-4 evolution).
        self.evolving_generator = SelfEvolvingGenerator(
            llm_model=self.config['llm']['generator']['model'],
            n_seeds_hypothesis=n_seeds_hypothesis,
            n_seeds_memory_augment=n_seeds_memory_augment,
            n_best_factors=evo_cfg['n_best_factors'],
            n_improve=evo_cfg.get('n_improve', 10),
            n_mutate=evo_cfg.get('n_mutate', 5),
            convergence_delta=evo_cfg.get('convergence_delta', 0.003),
            convergence_window=evo_cfg.get('convergence_window', 2),
            patience=evo_cfg.get('patience', 3),
            min_ic=evo_cfg.get('min_ic', 0.02),
            min_sharpe=evo_cfg.get('min_sharpe', 0.0),
            max_drawdown=evo_cfg.get('max_drawdown', -1.0),
            min_val_ic=evo_cfg.get('min_val_ic', 0.0),
            originality_gate=evo_cfg.get('originality_gate', True),
            dedup_similarity=evo_cfg.get('dedup_similarity', 0.90),
            improve_temperature=evo_cfg.get('improve_temperature', 0.3),
            elitism_carry=evo_cfg.get('elitism_carry', 2),
        )
        
        # Generate seed factors
        seed_factors = self.evolving_generator.generate_seed_factors()
        
        # If memory-augment seeds are requested AND the memory bank has entries,
        # augment via few-shot retrieval. Enabled purely by n_seeds_memory_augment > 0.
        if (n_seeds_memory_augment > 0
                and self.memory_bank is not None and len(self.memory_bank) > 0):
            print(f"  Using memory-augmented generation (target {n_seeds_memory_augment}, few-shot n_shots={n_shots})...")
            memory_generator = MemoryAugmentedGenerator(
                base_generator=self.evolving_generator,
                memory_bank=self.memory_bank,
                encoder=self.market_encoder,
                n_shots=n_shots,
            )
            # Use memory-augmented generator to produce factors with
            # few-shot examples retrieved from the memory bank
            try:
                # Build market index proxy for state encoding (use TRAIN data only)
                close_source = self.train_data['price_data']['close'] if self.train_data else self.price_data['close']
                market_close = close_source.mean(axis=1)
                market_df = pd.DataFrame({'close': market_close})
                augmented_factors = memory_generator.generate(
                    task_description=f"Generate {n_seeds_memory_augment} alpha factors for A-share stock selection",
                    retrieval_query=(
                        "quality factor with stable profitability and ROE, "
                        "value reversal factor using valuation metrics like PE and PB, "
                        "momentum factor capturing price trends with moving averages, "
                        "volatility factor for low-volatility effect, "
                        "liquidity factor for small-cap premium"
                    ),
                    price_df=market_df,
                    n_factors=n_seeds_memory_augment,
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
                else:
                    self.generated_factors = seed_factors
            except Exception as e:
                print(f"  Memory augmentation skipped: {e}")
                self.generated_factors = seed_factors
        else:
            if n_seeds_memory_augment == 0:
                print("  Memory augmentation skipped (n_seeds_memory_augment=0).")
            elif self.memory_bank is None or len(self.memory_bank) == 0:
                print("  Memory augmentation skipped (memory bank empty).")
            self.generated_factors = seed_factors
        print(f"  Total generated seeds: {len(self.generated_factors)} "
              f"({len(seed_factors)} base + "
              f"{len(self.generated_factors) - len(seed_factors)} memory-augmented)")

        print(f"  Total factors: {len(self.generated_factors)} in pool")
        print("  [✓] Factor generation complete")
        
    def _split_train_val(self, train_data: dict, val_ratio: float):
        """Carve a hold-out validation window from the END of the train period.

        Returns ``(train_sub, val_sub)`` dicts mirroring ``train_data``'s
        structure, or ``None`` if a clean split isn't possible (no price data,
        degenerate window, etc.).

        The validation window is the most-recent ``val_ratio`` fraction of the
        train dates (closest to the test distribution); the earlier portion
        drives evolution. The true test set in ``self.test_data`` is NEVER
        touched here — this is purely for in-sample model *selection*.
        """
        price = train_data.get('price_data', {})
        close = price.get('close')
        if close is None or len(close) < 20:
            return None
        n = len(close)
        split_idx = int(n * (1.0 - val_ratio))
        if split_idx <= 0 or split_idx >= n:
            return None
        val_start = close.index[split_idx]

        def _slice(d):
            if isinstance(d, pd.DataFrame):
                return d.iloc[:split_idx], d.iloc[split_idx:]
            return d, d

        train_sub, val_sub = {}, {}
        for key in ('price_data', 'fundamental_data'):
            src = train_data.get(key, {}) or {}
            ts, vs = {}, {}
            for name, df in src.items():
                a, b = _slice(df)
                ts[name] = a
                vs[name] = b
            train_sub[key] = ts
            val_sub[key] = vs
        # Cross-sectional / metadata pass-through (not time series)
        for key in ('industry_data', '_meta'):
            if key in train_data:
                train_sub[key] = train_data[key]
                val_sub[key] = train_data[key]
        val_sub['_meta'] = dict(train_data.get('_meta', {}))
        val_sub['_meta']['val_start'] = str(pd.Timestamp(val_start).date())
        return train_sub, val_sub

    def step4_evolve_factors(self, n_rounds: int = 5, forward_period: int = None):
        """
        Step 4: Self-evolve seed factors through iterative improvement.

        Evolve before debate — the backtester coarsely filters factors,
        then debate provides rigorous multi-expert evaluation on the best.
        
        Args:
            n_rounds: Number of evolution rounds
            forward_period: Forward return horizon in trading days (None → config or 10)
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4] Skipped: methods modules not available")
            return
        
        # Resolve forward_period: explicit arg > config.yaml > default 10
        if forward_period is None:
            forward_period = self.config.get('evolution', {}).get('forward_period', 10)
        
        # Parallel worker count for the evolution-loop backtests (0 = auto-scale
        # to the machine: min(32, cpu_count()+4)). Plumbed into evolve().
        eval_max_workers = int(self.config.get('evolution', {}).get('eval_max_workers', 0) or 0)
        
        print(f"\n[Step 4] Evolving factors ({n_rounds} rounds, forward={forward_period}d)...")

        # --- Train / validation split (anti-overfitting lever) ---
        # Evolution is DRIVEN by train IC but factors are finally RANKED and
        # EARLY-STOPPED on validation IC. The validation window is carved from
        # the most-recent tail of the train period; the true test set is never
        # touched here. val_ratio=0 → train-only selection (original behaviour).
        val_ratio = float(self.config.get('evolution', {}).get('val_ratio', 0.0) or 0.0)
        val_backtester = None
        train_price = self.train_data['price_data']
        train_fund = self.train_data.get('fundamental_data', {})

        if val_ratio > 0 and self.train_data:
            split = self._split_train_val(self.train_data, val_ratio)
            if split is not None:
                train_sub, val_sub = split
                train_price = train_sub['price_data']
                train_fund = train_sub.get('fundamental_data', {})
                try:
                    val_backtester = FactorBacktester(
                        prices=val_sub['price_data'],
                        fundamentals=val_sub.get('fundamental_data', {}),
                        forward_period=forward_period,
                    )
                    vstart = val_sub['_meta'].get('val_start', '?')
                    print(f"  [evolution] Validation holdout ENABLED (val_ratio={val_ratio}):")
                    print(f"    train window: {self._train_start_date} → {vstart}")
                    print(f"    val   window: {vstart} → {self._train_end_date}")
                except Exception as e:
                    print(f"  [evolution] Warning: failed to build val backtester ({e}); "
                          f"falling back to train-only selection.")
                    val_backtester = None
            else:
                print(f"  [evolution] Warning: train period too small to carve "
                      f"val_ratio={val_ratio}; using train-only selection.")

        # Initialize backtester for evolution — USE TRAINING DATA ONLY
        # Critical: factors must NOT see test data during evolution
        backtester = FactorBacktester(
            prices=train_price,
            fundamentals=train_fund,
            forward_period=forward_period,
        )
        # Keep a handle on the EXACT backtester used for evolution (TRAIN window,
        # val_split-aware) so Alpha101 scoring uses
        # data — apples-to-apples with the evolved factors' ICs.
        self._train_backtester = backtester
        # Record whether a validation holdout was actually used, so step5's
        # memory-save can correctly flag factors with a real val_ic.
        self._val_enabled = val_backtester is not None
        print(f"  [evolution] Using TRAIN data only: {self._train_end_date}")
        evolution_result = self.evolving_generator.evolve(
            seed_factors=self.generated_factors,
            backtester=backtester,
            n_rounds=n_rounds,
            val_backtester=val_backtester,
            max_workers=eval_max_workers,
        )

        self.evolution_history = evolution_result.evolution_history
        self.best_factors = evolution_result.best_factors
        self._forward_period = forward_period   # persist for step6/_calculate_factor_values

        print(f"  Evolution complete: {len(self.best_factors)} best factors")
        sel_mode = "validation" if val_backtester else "train"
        print(f"  Selection mode: {sel_mode}-IC")
        for rank, f in enumerate(self.best_factors, 1):
            vic = f.val_ic if (val_backtester and not np.isnan(f.val_ic)) else float('nan')
            print(f"    #{rank}: train_IC={f.ic:.4f}  val_IC={vic:.4f}  {f.expression}")
        print("  [✓] Factor evolution complete")

    def step4b_retrieve_from_memory(self):
        """
        Step 4b: Retrieve historical high-quality factors from memory bank.

        Runs AFTER step4_evolve_factors. Retrieved factors are NOT debated —
        they go directly to self._post_debate_factors and are merged back
        before step6 (fusion), where they participate using |ICIR| as their
        fusion prior.

        Retrieved factors are tagged with _from_memory=True.
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4b] Skipped: methods modules not available")
            return

        if self.memory_bank is None or len(self.memory_bank) == 0:
            print("\n[Step 4b] Skipped: memory bank is empty")
            return

        print("\n[Step 4b] Retrieving factors from memory (state-aware)...")

        # Encode current market state using TRAIN data (not test data)
        close_df = self.train_data['price_data']['close'].iloc[-self.recent_days:]
        current_state = self._build_market_state(close_df)

        query_desc = "multi-factor stock selection using technical indicators"
        query_expr = ""
        query_source = self.best_factors if (
                    hasattr(self, 'best_factors') and self.best_factors) else self.generated_factors
        if query_source:
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

            # Append to post_debate_factors (they skip debate).
            post = getattr(self, '_post_debate_factors', None) or []
            existing_exprs = {getattr(f, 'expression', '') for f in post
                              if getattr(f, 'expression', '')}
            added = 0
            for rf in retrieved_factors:
                if rf.expression not in existing_exprs:
                    rf._from_memory = True
                    post.append(rf)
                    existing_exprs.add(rf.expression)
                    added += 1
            self._post_debate_factors = post

            print(f"  Added {added} new factors from memory → post_debate "
                  f"(skipped {len(retrieved_factors) - added} duplicates)")
            print(f"  Post-debate pool size: {len(self._post_debate_factors)}")
        else:
            print("  No relevant factors found in memory")

        print("  [✓] Memory retrieval complete")

    def step4c_retrieve_alpha101(self, forward_period: int = None):
        """
        Step 4c: Retrieve / generate Alpha101 (and FAMA-101 bridge) factors.

        Produced factors are placed in self._post_debate_factors, so they
        SKIP evolution (Step 4a) and debate (Step 5): they are merged back
        into the fusion pool after Step 5b. This step runs AFTER evolution
        has finished, so these factors cannot be evolved anyway.

        Args:
            forward_period: forward return horizon in trading days
                            (None -> config['evolution']['forward_period'] or 10).
        """
        if not METHODS_AVAILABLE:
            print("\n[Step 4c] Skipped: methods modules not available")
            return

        evo_cfg = self.config['evolution']
        # Resolve forward_period: explicit arg > config.yaml > default 10
        # (mirrors step3 / step4 / step1).
        if forward_period is None:
            forward_period = evo_cfg.get('forward_period', 10)
        fwd = forward_period
        self._forward_period = forward_period

        # Initialize (or preserve) the post-debate pool.
        self._post_debate_factors = getattr(self, '_post_debate_factors', []) or []

        alpha101_retrieval_enabled = bool(evo_cfg.get('retrieve_alpha101', True))
        if not alpha101_retrieval_enabled:
            print("  [step4c] Alpha101 retrieval disabled "
                  "(evolution.retrieve_alpha101=false).")
            return

        gen = self.evolving_generator
        llm_available = getattr(gen, 'use_llm', False) and bool(gen.client)

        # --- Build TRAIN backtester (needed for scoring + FAMA bridge) ---
        try:
            train_price = self.train_data['price_data']
            train_fund = self.train_data.get('fundamental_data', {})
            import os as _os
            bt = FactorBacktester(
                prices=train_price,
                fundamentals=train_fund,
                forward_period=fwd,
            )
        except Exception as _e:
            print(f"  [step4c] Cannot build backtester ({_e}); "
                  f"skipping Alpha101 loading.")
            return

        # --- FAMA-101 bridge (if enabled) ---
        use_fama101_bridge = bool(evo_cfg.get('use_fama101_bridge', False))
        fama_factors = []
        if use_fama101_bridge:
            # Restrict compute to the train+test span (with warm-up) so we
            # never waste cycles on pre-train / post-test rows that are never
            # IC-scored or backtested. The context_days warm-up before
            # train_start mirrors the test context window split_data() already
            # prepends, keeping train-boundary factor values correct (Alpha101
            # longest rolling window is ~20d < context_days). Alpha101 factors
            # are causal, so this introduces no test leakage and in-window
            # values stay bit-identical to a full-series compute.
            full_idx = self.price_data['close'].index
            ts_idx = pd.to_datetime(full_idx)
            ctx = int(getattr(self, '_context_days', 30))
            if self._train_start_date:
                _pos = int(ts_idx.searchsorted(pd.Timestamp(self._train_start_date)))
                _warm_start = ts_idx[max(0, _pos - ctx)]
            else:
                _warm_start = ts_idx[0]
            _end = (pd.Timestamp(self._test_end_date)
                    if self._test_end_date else ts_idx[-1])
            _px = {k: v.loc[slice(_warm_start, _end)]
                   for k, v in self.price_data.items()}
            _n_dt, _n_tkr = _px['close'].shape[0], _px['close'].shape[1]
            print(f"  [step4c] FAMA-101 bridge: computing Alpha101 exposures "
                  f"(forward_period={fwd}) on train+test span with {ctx}d "
                  f"warm-up ({_n_dt} dates x {_n_tkr} tickers; "
                  f"full archive = {len(full_idx)} dates)...")
            try:
                from methods.fama_alpha101_bridge import (
                    compute_fama101_exposures,
                    slice_exposures_to_dates,
                )
                from concurrent.futures import ThreadPoolExecutor

                exposures = compute_fama101_exposures(_px, forward_period=fwd)
                train_dates = self.train_data['price_data']['close'].index
                train_slices = slice_exposures_to_dates(exposures, train_dates)

                fama_factors = []
                for col in exposures.columns:
                    cf = CandidateFactor(
                        id=col, expression=col,
                        description=f"FAMA Alpha101 {col}",
                        generation=0, family='Alpha101')
                    cf._from_fama101 = True
                    cf._from_alpha101_lib = True
                    cf._from_alpha101 = True
                    fama_factors.append(cf)

                def _score_fama(cf):
                    fv = train_slices.get(cf.expression)
                    if fv is None:
                        return
                    try:
                        bt.evaluate(cf, factor_values=fv,
                                    cache_key=cf.expression)
                    except Exception:
                        pass
                _nw = 1  # serial (avoids Windows C-level crash)
                with ThreadPoolExecutor(max_workers=_nw) as _ex:
                    list(_ex.map(_score_fama, fama_factors))

                self._fama101_exposures = exposures
                self._post_debate_factors.extend(fama_factors)
                print(f"  [step4c] FAMA-101 bridge: {len(fama_factors)} "
                      f"factors -> post_debate")
            except Exception as _e:
                import traceback
                traceback.print_exc()
                print(f"  [step4c] FAMA-101 bridge failed: {_e}")

        # --- LLM mining of alpha101-inspired factors ---
        # Runs AFTER the bridge (so fama_factors are available as the scored
        # library for inspiration chains). ALL mined factors go to
        # _post_debate_factors (Step 4c does not participate in debate).
        if llm_available:
            try:
                mining_iters = int(evo_cfg.get('alpha101_llm_mining_iters', 3))
                mining_per_iter = int(evo_cfg.get('alpha101_llm_mining_per_iter', 1))
                mining_max_chain = int(evo_cfg.get('alpha101_llm_mining_max_chain', 5))
                alpha101_ratio = float(evo_cfg.get('alpha101_ratio', 1.0))
                alpha101_top_k = int(evo_cfg.get('alpha101_top_k', 0) or 0)

                _fama_lib = fama_factors if use_fama101_bridge else None
                print(f"  [step4c] LLM mining Alpha101-inspired factors "
                      f"({mining_iters} iters x {mining_per_iter}/call, "
                      f"max_chain={mining_max_chain}, "
                      f"alpha101_ratio={alpha101_ratio}, "
                      f"library={'FAMA-101' if use_fama101_bridge else 'Alpha101'})...")
                mined = gen.llm_mine_alpha101_inspired(
                    backtester=bt,
                    max_workers=max(1, min(32, (_os.cpu_count() or 4) + 4)),
                    n_iters=mining_iters,
                    n_per_iter=mining_per_iter,
                    max_chain_len=mining_max_chain,
                    temperature=getattr(gen, 'improve_temperature', 0.3),
                    alpha101_ratio=alpha101_ratio,
                    alpha101_top_k=alpha101_top_k,
                    fama101_library=_fama_lib,
                )
                if mined:
                    lib_factors = [f for f in mined
                                   if getattr(f, '_from_alpha101_lib', False)]
                    llm_factors = [f for f in mined
                                   if not getattr(f, '_from_alpha101_lib', False)]
                    if lib_factors:
                        self._post_debate_factors.extend(lib_factors)
                        print(f"  [step4c] {len(lib_factors)} library factors "
                              f"-> post_debate")
                    if llm_factors:
                        self._post_debate_factors.extend(llm_factors)
                        print(f"  [step4c] {len(llm_factors)} LLM-mined factors "
                              f"-> post_debate (no evolve/debate)")
            except Exception as _e:
                print(f"  [step4c] LLM mining failed: {_e}")
        else:
            print(f"  [step4c] LLM not available - Alpha101 mining skipped; "
                  f"{len(self._post_debate_factors)} library factors loaded.")

        print(f"  [step4c] Post-debate pool size: "
              f"{len(self._post_debate_factors)}")

    def step4d_select_top_factors(self):
        from collections import Counter
        """
        Step 4d: Filter and rank the accumulated factor pool, keep only the
        top ``n_best4debate`` factors for Step 5 (debate).

        After Step 4 (evolve) + Step 4b (memory retrieval), the factor
        pool may contain 30–50+ factors from heterogeneous sources.  Step 5's multi-agent debate is expensive (5 LLM
        API calls per factor), and Step 6's ``fusion.top_k`` cap would discard
        most of them anyway.  This step pre-filters the pool so Step 5 only
        debates the most promising candidates.

        Ranking metric: ``|train IC|`` (absolute IC, direction-agnostic — the
        backtester direction-aligns quantile metrics by sign(IC), so negative-IC
        factors are just as useful as positive).  All three sources (evolved,
        memory, LLM-mined) carry a train IC, making this a fair universal metric.

        Config:
            evolution.n_best4debate  — top-N to retain (0 = disabled, pass all
                                       through; backward compatible).
        """
        evo_cfg = self.config.get('evolution', {})
        n_best = int(evo_cfg.get('n_best4debate', 0) or 0)

        if n_best <= 0:
            print(f"\n[Step 4d] Skipped: n_best4debate={n_best} "
                  f"(0 = no pre-step5 filter)")
            return

        if not hasattr(self, 'best_factors') or not self.best_factors:
            print("\n[Step 4d] Skipped: no factors in pool")
            return

        pool_size = len(self.best_factors)

        # --- Filter: drop NaN-IC and explicitly-invalid factors ---
        valid = []
        dropped = 0
        for f in self.best_factors:
            ic = getattr(f, 'ic', None) if not isinstance(f, dict) else f.get('ic')
            if ic is None or (isinstance(ic, float) and np.isnan(ic)):
                dropped += 1
                continue
            is_valid = getattr(f, 'is_valid', True) if not isinstance(f, dict) \
                else f.get('is_valid', True)
            if is_valid is False:
                dropped += 1
                continue
            valid.append(f)

        if dropped:
            print(f"\n[Step 4d] Filtered {dropped} invalid/NaN-IC factors "
                  f"({pool_size} → {len(valid)})")

        if not valid:
            print("  [Step 4d] No valid factors after filtering; keeping "
                  "original pool unchanged.")
            return

        # --- Split: debate vs post-debate ---
        # Factors from memory retrieval (step4b) and Alpha101 / FAMA-101
        # library factors skip the expensive multi-agent debate (step5).
        # Only truly evolved factors go to debate. Post-debate factors are
        # held in self._post_debate_factors and merged back before step6
        # (fusion), where they participate via |ICIR| instead of debate_score.
        def _is_post_debate(f):
            if isinstance(f, dict):
                src = f.get('_source', '')
                return src in ('memory', 'alpha101', 'alpha101-llm', 'fama101')
            return (getattr(f, '_from_memory', False)
                    or getattr(f, '_from_alpha101', False)
                    or getattr(f, '_from_fama101', False))

        post_debate = [f for f in valid if _is_post_debate(f)]
        debate = [f for f in valid if not _is_post_debate(f)]

        # Stash for post-step5 merge — APPEND to any pre-existing
        # post_debate factors (from step3/step4b), don't overwrite.
        _existing_pd = getattr(self, '_post_debate_factors', []) or []
        self._post_debate_factors = _existing_pd + post_debate

        def _abs_ic(f):
            if isinstance(f, dict):
                return abs(f.get('ic', 0.0))
            return abs(getattr(f, 'ic', 0.0) or 0.0)

        def _source_tag(f):
            if isinstance(f, dict):
                return f.get('_source', 'evolved')
            if getattr(f, '_from_alpha101_llm', False):
                return 'alpha101-llm'
            if getattr(f, '_from_alpha101', False):
                return 'alpha101'
            if getattr(f, '_from_memory', False):
                return 'memory'
            if getattr(f, '_from_fama101', False):
                return 'fama101-bridge'
            return 'evolved'

        if n_best <= 0:
            # No cap — all debate factors pass through unchanged.
            self.best_factors = debate
            _src_tags = Counter(_source_tag(f) for f in post_debate) if post_debate else {}
            _src_str = ', '.join(f"{s}={n}" for s, n in sorted(_src_tags.items()))
            print(f"\n[Step 4d] n_best4debate=0 (no cap): {len(debate)} factors "
                  f"→ debate; {len(self._post_debate_factors)} post-debate held for post-step5 "
                  f"merge ({_src_str or 'none'}).")
            return

        if len(debate) <= n_best:
            self.best_factors = debate
            _src_tags = Counter(_source_tag(f) for f in post_debate) if post_debate else {}
            _src_str = ', '.join(f"{s}={n}" for s, n in sorted(_src_tags.items()))
            print(f"\n[Step 4d] {len(debate)} debate factors ≤ "
                  f"n_best4debate={n_best} (no truncation); "
                  f"{len(self._post_debate_factors)} post-debate held "
                  f"({_src_str or 'none'}).")
            return

        # --- Rank the debate factors by |IC| descending, keep top-N ---
        debate.sort(key=_abs_ic, reverse=True)
        selected = debate[:n_best]

        # --- Print summary table ---
        _src_tags = Counter(_source_tag(f) for f in post_debate) if post_debate else {}
        _src_str = ', '.join(f"{s}={n}" for s, n in sorted(_src_tags.items()))
        print(f"\n[Step 4d] Selecting top {n_best} of {len(debate)} debate "
              f"factors by |train IC| ({len(post_debate)} post-debate held: "
              f"{_src_str or 'none'}):")
        print(f"  {'Rank':<5} {'Source':<14} {'IC':>8}  Expression")
        print(f"  {'─'*5} {'─'*14} {'─'*8}  {'─'*50}")
        for rank, f in enumerate(selected, 1):
            ic = f.get('ic', 0.0) if isinstance(f, dict) else getattr(f, 'ic', 0.0)
            expr = (f.get('expression', '') if isinstance(f, dict)
                    else getattr(f, 'expression', ''))
            print(f"  {rank:<5} {_source_tag(f):<14} {ic:>+8.4f}  {expr[:60]}")

        # Show what was cut (top of the dropped debate tail)
        tail = debate[n_best:]
        if tail:
            best_cut_ic = _abs_ic(tail[0])
            worst_keep_ic = _abs_ic(selected[-1])
            print(f"\n  Cut-off: |IC| {worst_keep_ic:.4f} → {best_cut_ic:.4f} "
                  f"({len(tail)} debate factors dropped)")
            from collections import Counter
            kept_src = Counter(_source_tag(f) for f in selected)
            cut_src = Counter(_source_tag(f) for f in tail)
            parts = [f"{s}={kept_src.get(s,0)}" for s in sorted(kept_src)]
            print(f"  Survived by source: {', '.join(parts)}")
            parts = [f"{s}={cut_src.get(s,0)}" for s in sorted(cut_src)]
            print(f"  Dropped  by source: {', '.join(parts)}")

        self.best_factors = selected
        print(f"  [✓] Factor pool: {pool_size} → {len(self.best_factors)} debate "
              f"+ {len(self._post_debate_factors)} post-debate held (ready for Step 5)")

    def step5_evaluate_factors(self):
        """
        Step 5: Evaluate EVOLVED candidate factors via multi-agent debate.

        Memory-retrieved (step4b) and Alpha101 / FAMA-101 library factors
        factors are HELD in ``self._post_debate_factors`` and skip the expensive
        debate. They are merged back into the pool AFTER this step, before
        step6 (fusion), where they participate using |ICIR| as their prior.
        Only truly evolved factors go through the 5-expert debate here.

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
            # Chair synthesis prompt bounding / timeout (fixes [Step 5b] APITimeoutError)
            request_timeout=chair_cfg.get('request_timeout', 120.0),
            synthesis_top_n=chair_cfg.get('synthesis_top_n', 50),
            chair_max_tokens=chair_cfg.get('max_tokens', 4096),
            chair_max_per_family=self.config['fusion'].get('chair_max_per_family', 3),
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
                ic = factor.get('ic')
                sharpe = factor.get('sharpe')
                family = factor.get('family') or None
            else:
                expr = factor.expression
                desc = factor.description
                # Thread the REAL backtest evidence + family label into the
                # debate. Without these the experts score factors on narrative
                # alone and the Chair cannot enforce family diversity — see
                # methods/debate.py prompt changes.
                ic = getattr(factor, 'ic', None)
                sharpe = getattr(factor, 'sharpe', None)
                family = getattr(factor, 'family', '') or None

            proposal = FactorProposal(expression=expr, description=desc,
                                      ic=ic, sharpe=sharpe, family=family)

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
            skipped_low_ic = 0
            save_min_ic = self.config.get('memory', {}).get('save_min_ic', 0.0)
            for factor, result in self.debate_results:
                if isinstance(factor, dict):
                    expr = factor.get('expression', '')
                    desc = factor.get('description', '')
                    is_from_memory = factor.get('_from_memory', False)
                    is_from_alpha101 = factor.get('_from_alpha101', False)
                    factor_ic = factor.get('ic')
                else:
                    expr = getattr(factor, 'expression', '')
                    desc = getattr(factor, 'description', '')
                    is_from_memory = getattr(factor, '_from_memory', False)
                    is_from_alpha101 = getattr(factor, '_from_alpha101', False)
                    factor_ic = getattr(factor, 'ic', None)
                if not expr:
                    continue
                # Skip factors retrieved from memory — they're already stored there
                if is_from_memory:
                    continue
                # Skip Alpha101 library factors — they are publicly-known formulas
                # re-scored each run (during factor generation in Step 3),
                # bank would just accumulate library noise.
                if is_from_alpha101:
                    continue
                # Quality gate: don't persist factors below the IC floor, otherwise
                # the bank accumulates noise that Step 4b retrieval feeds back into
                # future pools. Missing IC (None) is kept, not dropped.
                if save_min_ic and factor_ic is not None and factor_ic < save_min_ic:
                    skipped_low_ic += 1
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
                        val_ic=getattr(factor, 'val_ic', 0.0),
                        has_val=bool(getattr(self, '_val_enabled', False)),
                        source='debate',
                    )
                    saved += 1
                except Exception as e:
                    print(f"  [memory] add failed: {e}")
            if saved or skipped_low_ic:
                _msg = f"  Saved {saved} factors to memory bank"
                if skipped_low_ic:
                    _msg += f" (skipped {skipped_low_ic} below save_min_ic={save_min_ic})"
                print(_msg)
        
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
                ic = factor.get('ic')
                sharpe = factor.get('sharpe')
                family = factor.get('family') or None
            else:
                expr = getattr(factor, 'expression', '')
                desc = getattr(factor, 'description', '')
                ic = getattr(factor, 'ic', None)
                sharpe = getattr(factor, 'sharpe', None)
                family = getattr(factor, 'family', '') or None

            proposal = FactorProposal(expression=expr, description=desc,
                                      ic=ic, sharpe=sharpe, family=family)
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

        # ── Persist Chair selection so Step 6 can actually use it ──
        # Without this, the Chair's selected/rejected counts were a no-op report
        # and Step 6 fused ALL best_factors (the "Selected 5 / Fused 13" bug).
        # We store the *expressions* the Chair ranked (its chosen subset) plus
        # the rejected set and per-factor final_score. Whitespace-normalized so
        # they match self.best_factors expressions exactly.
        _ranked = synthesis.get('factors_ranked', []) or []
        _rejected = synthesis.get('rejected_factors', []) or []
        self.chair_synthesis = synthesis
        self.chair_selected_expressions = {
            str(item.get('expression', '')).strip()
            for item in _ranked if str(item.get('expression', '')).strip()
        }
        self.chair_rejected_expressions = {
            str(item.get('expression', '')).strip()
            for item in _rejected if str(item.get('expression', '')).strip()
        }
        self.chair_score_map = {
            str(item.get('expression', '')).strip(): float(item.get('final_score') or 0.0)
            for item in _ranked if str(item.get('expression', '')).strip()
        }
        # Rejected factors also carry final_score — needed so Step 6 can
        # DOWNWEIGHT (soft-reject) them instead of hard-dropping. Without this
        # map, soft-rejected factors would fall back to debate_score 0.0.
        self.chair_rejected_score_map = {
            str(item.get('expression', '')).strip(): float(item.get('final_score') or 0.0)
            for item in _rejected if str(item.get('expression', '')).strip()
        }
        print(f"  [step5b→step6] Chair selection persisted: "
              f"{len(self.chair_selected_expressions)} ranked, "
              f"{len(self.chair_rejected_expressions)} rejected "
              f"(use_chair_selection={self.config['fusion'].get('use_chair_selection', False)})")

        # ── Hard-delete Chair-rejected factors from the pool ──
        # When use_chair_selection is on, the Chair's rejections physically
        # remove factors from best_factors HERE (Step 5b) so they never reach
        # Step 6 fusion. The matching uses the base expression (stripping the
        # '...__dupN' de-dup suffix) — identical to Step 6. Selected factors
        # always win over rejections; factors the Chair neither ranked nor
        # rejected are kept. (At this point best_factors holds only the
        # debaited set — the post-debate memory/Alpha101/FAMA factors are
        # merged back AFTER Step 5, so they are never touched by Chair
        # rejection here.)
        _use_chair = bool(self.config['fusion'].get('use_chair_selection', False))
        if _use_chair:
            _sel = self.chair_selected_expressions
            _rej = self.chair_rejected_expressions
            if _rej:
                _before = len(self.best_factors)
                _kept, _removed = [], []
                for _f in self.best_factors:
                    _expr = (getattr(_f, 'expression', '')
                             if not isinstance(_f, dict)
                             else _f.get('expression', ''))
                    _base = str(_expr).split('__dup')[0].strip()
                    if _base in _sel:
                        _kept.append(_f)          # selected wins over rejected
                    elif _base in _rej:
                        _removed.append(_base)
                    else:
                        _kept.append(_f)          # neither ranked nor rejected → keep
                self.best_factors = _kept
                if _removed:
                    print(f"  [step5b→pool] Chair hard-drop: removed "
                          f"{len(_removed)} rejected factor(s) from best_factors "
                          f"({_before} → {len(self.best_factors)}).")

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
                val_icir_val = getattr(f, 'val_icir', 0.0)
                factor_meta_lookup[expr] = (ic_val, icir_val, sharpe_val, val_icir_val)

        # ── Debate prior score (debate_score) for fusion ──
        # Use the per-factor Step-5 multi-agent debate final_score as the fusion
        # prior for factor weighting.
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

        # ── Final top_k cap on the fusion pool (applied LAST) ──
        # Keep only the top-K factors (by the fusion prior) that actually enter
        # fusion. The prior is the multi-agent debate_score, the same score used
        # to build factor_infos below, so truncating train_factor_dict here keeps
        # it perfectly aligned with factor_infos and fusion.fuse().
        #   0  → no cap (fuse every surviving factor)
        #   >0 → keep the K highest-prior factors (the "final top-K" pool)
        _fusion_top_k = int(self.config['fusion'].get('top_k', 0) or 0)
        if _fusion_top_k > 0 and len(train_factor_dict) > _fusion_top_k:
            def _fusion_prior(_name):
                _s = debate_score_map.get(_name, 0.0)
                if _s == 0.0:
                    # fall back to base expression (dup-suffix / whitespace variant)
                    _s = debate_score_map.get(_name.split('__dup')[0].strip(), 0.0)
                if _s == 0.0:
                # Fallback when no debate prior exists (e.g. --skip-eval
                # skipped Step 5): rank by |ICIR| so the top_k cap still
                    # selects the strongest factors instead of an arbitrary
                    # dict-order prefix. meta = (ic, icir, sharpe, val_icir).
                    _meta = factor_meta_lookup.get(_name)
                    if _meta is None:
                        _meta = factor_meta_lookup.get(_name.split('__dup')[0].strip())
                    if _meta:
                        _s = abs(_meta[1])  # |icir|
                return _s
            _ranked = sorted(train_factor_dict.keys(),
                             key=_fusion_prior, reverse=True)
            _keep_topk = set(_ranked[:_fusion_top_k])
            train_factor_dict = {_k: _v for _k, _v in train_factor_dict.items()
                                 if _k in _keep_topk}
            print(f"  Fusion top_k cap: kept top {_fusion_top_k} of "
                  f"{len(_ranked)} factors by fusion prior "
                  f"(prior {_fusion_prior(_ranked[0]):.2f}.."
                  f"{_fusion_prior(_ranked[_fusion_top_k - 1]):.2f}).")
        elif _fusion_top_k > 0:
            print(f"  Fusion top_k={_fusion_top_k} but only "
                  f"{len(train_factor_dict)} factor(s) in the pool; "
                  f"no truncation needed.")

        # Use train_factor_dict keys (same factor names, train-period only)
        factor_infos = []
        for name, values_df in train_factor_dict.items():
            if not isinstance(values_df.index, pd.DatetimeIndex):
                values_df = values_df.copy()
                values_df.index = pd.to_datetime(values_df.index)

            meta = factor_meta_lookup.get(name)
            if meta is None:
                # dup-suffixed variant (expr__dupN): fall back to the base
                # expression so it keeps its IC/ICIR meta.
                meta = factor_meta_lookup.get(name.split('__dup')[0].strip())
            if meta:
                ic_val, icir_val, sharpe_val, val_icir_val = meta
                if abs(icir_val) > 1e-8:
                    ic_std_val = ic_val / icir_val
                else:
                    ic_std_val = 1.0
            else:
                ic_val = icir_val = ic_std_val = sharpe_val = val_icir_val = 0.0

            dscore = debate_score_map.get(name, 0.0)
            if dscore == 0.0:
                # fall back to base expression (dup-suffix or whitespace variant)
                dscore = debate_score_map.get(name.split('__dup')[0].strip(), 0.0)
            ic_sign = float(np.sign(ic_val)) if abs(ic_val) > 1e-10 else 1.0
            # Effective sample size for IC/ICIR:
            # factor values have T dates, but last forward_period dates lack forward returns
            _fwd = getattr(self, '_forward_period', 10)
            n_periods = max(2, len(values_df) - _fwd)
            factor_infos.append(FactorInfo(
                name=name, expression=name,
                ic=ic_val, icir=icir_val, ic_std=ic_std_val,
                sharpe=sharpe_val, debate_score=dscore,
                ic_sign=ic_sign,
                n_periods=n_periods, val_icir=val_icir_val,
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

        # ── 4c. Encode market state for regime tilt ──
        # Source window: when val_ratio>0 we re-carve the SAME validation tail
        # that step4 used for factor SELECTION, and encode the market state from
        # THAT boundary window (closest to the upcoming test distribution). This
        # makes the regime tilt reflect the "current" regime at the train/test
        # boundary instead of being smoothed across the whole training history.
        # When val_ratio=0 (default) we fall back to the full train span — i.e.
        # exactly the previous behaviour. Test data is NEVER touched (no lookahead);
        # factor VALUES and IC/ICIR weights below still come from the full train.
        market_state = None
        try:
            _ms_source = "TRAIN"
            _ms_close = train_price.get('close')
            _val_ratio = float(self.config.get('evolution', {}).get('val_ratio', 0.0) or 0.0)
            if _val_ratio > 0 and self.train_data:
                _split = self._split_train_val(self.train_data, _val_ratio)
                if _split is not None:
                    _val_close = _split[1].get('price_data', {}).get('close')
                    if isinstance(_val_close, pd.DataFrame) and len(_val_close) > 0:
                        _ms_close = _val_close
                        _ms_source = "VAL-tail (boundary)"
            if _ms_close is not None and isinstance(_ms_close, pd.DataFrame):
                # Shared helper → corr_matrix from actual pairwise return corr,
                # now computed over the chosen boundary window, not full history.
                market_state = self._build_market_state(_ms_close)
                print(f"  Market state (from {_ms_source}): "
                      f"{market_state.to_string() if market_state else 'None'}")
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
        # Persist for test-period Mean IC / ICIR computation in run_test_pipeline.
        self.test_composite_scores = test_composite_scores

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

    def step8_backtest(self, test_data=None, output_dir=None):
        """
        Step 8: Backtest the portfolio strategy.

        Args:
            test_data: Optional test data dict (defaults to self.test_data).
            output_dir: Directory to persist the portfolio-level daily-return
                series (``mase_daily_returns.csv``) and equity curve
                (``mase_portfolio_values.csv``). If None, falls back to the same
                default used by step9_save_results
                (``experiments/{YYYYMMDD}/results/``) so the series lands next to
                the other MASE result files.

                This mirrors what the 9 baselines do via
                ``engine.run(..., save_dir=run_dir)``: each of them persists a
                *portfolio* daily-return series. MASE previously saved only
                ``portfolios.csv`` (per-stock *weights*, not returns) and
                ``performance_metrics.csv``, leaving its portfolio daily return on
                the table. The engine's ``self.returns`` — the very series used to
                compute Sharpe / vol / IR — is the portfolio daily return and is
                what we now persist for comparability.
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
        # Build an equal-weight universe benchmark (market proxy) from the test
        # close prices so Information Ratio reflects *excess* vs the market
        # instead of degenerating to the Sharpe ratio (benchmark=0).
        # bm[t] = equal-weight mean of stock returns over [t, t+1], aligned to
        # the same "from" date labels the engine uses for strategy returns.
        _test_close = _test['price_data']['close']
        _bm = _test_close.pct_change().shift(-1).mean(axis=1).dropna()
        _bm.name = 'benchmark_return'

        # Persist the *portfolio-level* daily-return series (and equity curve)
        # so MASE is directly comparable to the 9 baselines, which each save a
        # portfolio ``daily_returns.csv`` via engine.run(save_dir=run_dir).
        # MASE itself only saved per-stock ``portfolios.csv`` before; the
        # engine's ``self.returns`` (used for Sharpe / vol / IR) is the portfolio
        # daily return and is exactly what we want on disk.
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            _save_dir = config_path('experiments', os.path.join(date_str, "results"))
        else:
            _save_dir = output_dir
        self.performance_metrics = self.backtest_engine.run(
            portfolios=self.portfolios,
            prices=_test['price_data']['close'],
            benchmark_returns=_bm,
            save_dir=_save_dir,
            method_prefix=None,
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

    def _compute_composite_test_ic(self, test_data=None):
        """
        Compute the MASE composite (fused) factor's test-period Mean Rank-IC
        and ICIR against N-day forward returns, and attach them to
        ``self.performance_metrics`` as ``mean_rank_ic`` / ``icir``.

        This mirrors the per-baseline ``mean_rank_ic`` / ``icir`` that the
        baseline runners report (e.g. run_xgboost_simple.py, run_alphagrail.py),
        so every method in the paper's experiment table shares the same IC/ICIR
        column names.

        Must be called AFTER step7 (which sets ``self.test_composite_scores``)
        and ideally after step8 (which populates ``self.performance_metrics``);
        the IC/ICIR keys are merged into whatever dict is already present.

        Args:
            test_data: Test-period data dict. If None, falls back to
                ``self.test_data`` (same resolution order as step7/step8).

        Returns:
            bool: True if IC/ICIR were computed and stored, False otherwise.
        """
        if self.performance_metrics is None:
            self.performance_metrics = {}
        _test = test_data if test_data is not None else getattr(self, 'test_data', None)
        if not _test:
            print("  [warn] No test data available for composite IC/ICIR.")
            return False
        scores = getattr(self, 'test_composite_scores', None)
        if scores is None or (hasattr(scores, 'empty') and scores.empty):
            print("  [warn] No composite test scores available for IC/ICIR.")
            return False
        try:
            from backtest.metrics import factor_ic_metrics
            _close = _test['price_data']['close']
            _fp = getattr(self, '_forward_period', None) \
                or self.config.get('evolution', {}).get('forward_period', 10)
            _fwd = _close.pct_change(_fp).shift(-_fp)
            _common = scores.index.intersection(_fwd.index)
            if len(_common) == 0:
                print("  [warn] No overlapping dates between composite scores "
                      "and forward returns.")
                return False
            _ic = factor_ic_metrics(scores.loc[_common], _fwd.loc[_common])
            self.performance_metrics['mean_rank_ic'] = float(_ic['mean_ic'])
            self.performance_metrics['icir'] = float(_ic['icir'])
            print(f"  Mean Rank-IC (test): {_ic['mean_ic']:.4f}, "
                  f"ICIR (test): {_ic['icir']:.4f}")
            return True
        except Exception as e:
            print(f"  [warn] Could not compute composite test IC/ICIR: {e}")
            return False

    def run_full_pipeline(
        self,
        use_sample: bool = True,
        n_evolution_rounds: int = 5,
        output_dir: str = None,
        universe: str = None,
        train_start_date: str = None,
        train_end_date: str = None,
        test_start_date: str = None,
        test_end_date: str = None,
        forward_period: int = None,
        holding_period: int = None,
        test_data=None,
        skip_eval: bool = False,
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
            train_start_date: Explicit train start date for train/test split
            train_end_date: Explicit train end date for train/test split
            test_start_date: Explicit test start date for train/test split
            test_end_date: Explicit test end date for train/test split
            forward_period: Forward return horizon in trading days (None → config or 10)
            holding_period: Backtest holding period in trading days.
                           1 = daily rebalance, 5 = weekly, 20 = monthly.
                           None → use config or default 1.
            test_data: Optional external test data dict to use in step7/step8.
                       If provided, overrides self.test_data from step1 split.
            skip_eval: If True, skip Step 5 (multi-agent debate) and Step 5b
                       (Chair synthesis) and go straight to Step 6 factor fusion.
                       The fused factor is then built directly from the evolved
                       best_factors (capped by fusion.top_k, ranked by |ICIR| when
                       no debate prior is available).
        """
        print("\n" + "=" * 60)
        data_label = "SAMPLE DATA (fast test)" if use_sample else "REAL DATA"
        print(f"  Running Full Pipeline — {data_label}")
        print("=" * 60)

        # Apply CLI overrides to config before step1 (so BacktestEngine picks them up)
        if holding_period is not None:
            self.config.setdefault('backtest', {}).setdefault('trading', {})['holding_period'] = holding_period

        # Resolve the 4 date boundaries with config fallbacks.
        # (Mirrors the baselines' `loader.data_config` pattern; main.py's config
        #  lives in self.config['data'] instead of a standalone loader object.)
        train_start = train_start_date or self.config.get('data', {}).get('train_start_date', '2023-01-01')
        train_end   = train_end_date   or self.config.get('data', {}).get('train_end_date', '2023-12-31')
        test_start  = test_start_date  or self.config.get('data', {}).get('test_start_date', '2024-01-01')
        test_end    = test_end_date    or self.config.get('data', {}).get('test_end_date', '2025-06-30')

        # Run all steps
        self.step1_load_data(
            use_sample=use_sample,
            universe=universe,
            train_start_date=train_start,
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=test_end,
            forward_period=forward_period,
            holding_period=holding_period,
        )
        self.step2_initialize_memory()
        self.step3_generate_factors(forward_period=forward_period)
        self.step4_evolve_factors(n_evolution_rounds, forward_period=forward_period)
        self.step4b_retrieve_from_memory()   # retrieve after evolution → augment candidate pool
        self.step4c_retrieve_alpha101(forward_period=forward_period)  # Alpha101/FAMA-101 → post_debate
        self.step4d_select_top_factors()     # filter + rank → keep top n_best4debate for Step 5
        if skip_eval:
            print("\n[skip] Step 5 (debate) and Step 5b (Chair synthesis) skipped "
                  "--fusing top_k best_factors directly in Step 6.")
            if bool(self.config.get('fusion', {}).get('use_chair_selection', False)):
                print("  [warn] fusion.use_chair_selection is TRUE but Step 5b was "
                      "skipped, so Chair selection is effectively disabled (no ranked "
                      "set to filter by). Factors enter fusion unfiltered by the Chair.")
        else:
            self.step5_evaluate_factors()        # debate evolved candidates only
            self.step5b_chair_synthesis()     # Chair synthesis (step 5b)
        # --- Merge post-debate factors back into the pool ---
        # Memory-retrieved (step4b) and Alpha101 / FAMA-101 library
        # factors skip the expensive multi-agent debate (step5). They are merged
        # back here so they participate in step6's ICIR²-shrinkage fusion using
        # |ICIR| as their prior (no debate_score).
        if hasattr(self, '_post_debate_factors') and self._post_debate_factors:
            _pd = self._post_debate_factors
            _existing = {getattr(f, 'expression', '') for f in self.best_factors}
            _added = 0
            for f in _pd:
                if getattr(f, 'expression', '') not in _existing:
                    self.best_factors.append(f)
                    _existing.add(getattr(f, 'expression', ''))
                    _added += 1
            print(f"\n  Post-debate merge: added {_added}/{len(_pd)} factors "
                  f"(memory + Alpha101 + FAMA-101) → pool size "
                  f"{len(self.best_factors)}")
            del self._post_debate_factors
        self.step6_fuse_factors()
        self.step7_construct_portfolio(test_data=test_data)
        self.step8_backtest(test_data=test_data, output_dir=output_dir)
        # Compute & persist the composite factor's test-period IC/ICIR so they
        # appear in the saved performance_metrics.csv (full pipeline path).
        self._compute_composite_test_ic(test_data)
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

        # ── 8. Compute test-period Mean Rank-IC / ICIR for the composite factor ──
        # Uses the fused test-period factor scores (self.test_composite_scores)
        # vs the N-day forward returns, all on the out-of-sample period.
        # Stored as mean_rank_ic / icir into self.performance_metrics.
        self._compute_composite_test_ic(test_data)

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
        # Daily returns — 'return' and 'returns' are aliases
        if 'close' in data_map:
            _ret = data_map['close'].pct_change(1)
            data_map['return'] = _ret
            data_map['returns'] = _ret
        # VWAP = amount / volume
        if 'amount' in data_map and 'volume' in data_map:
            vol_safe = data_map['volume'].replace(0, np.nan)
            data_map['vwap'] = data_map['amount'] / vol_safe
        # NOTE: forward_returns is intentionally NOT added to data_map.
        # It holds the N-day FUTURE return; exposing it as an expression field would let a
        # factor reference future prices (look-ahead bias). It is still used as the IC /
        # portfolio target in run_test_pipeline via a separate computation.
        # (`forward_period` param is retained for signature compatibility with callers.)


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

                # FAMA-101 precomputed panel: bypass the expression evaluator
                # entirely and use the panel computed by FAMA's own code (stored
                # on self._fama101_exposures). This is what keeps MASE's Alpha101
                # factors bit-for-bit identical to FAMA's. Slice by the dates of
                # the price_data passed in (TRAIN at step6, full/TEST at step7).
                if (getattr(f, '_from_fama101', False)
                        and getattr(self, '_fama101_exposures', None) is not None
                        and expr in self._fama101_exposures.columns):
                    _col = self._fama101_exposures[expr]
                    _dates = close.index
                    _mask = _col.index.get_level_values('date').isin(_dates)
                    _vals = _col.loc[_mask].unstack('ticker')
                    _vals = _vals.reindex(index=_dates, columns=close.columns)
                    factor_dict[factor_key] = _vals
                    continue

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
  python main.py --full --real --train-start 2023-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-12-31
  python main.py --full --real --force-refresh             Skip cache, re-download
  python main.py --full --skip-eval                       Skip Step 5/5b, fuse top_k directly
  python main.py --test --factor-path experiments/20260601/results/final_factors.json
  python main.py --test --factor-path PATH --real --train-start 2023-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2024-12-31
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
        '--train-start', type=str, default=None,
        help='Train start date (YYYY-MM-DD, default: 2022-01-01)',
    )
    parser.add_argument(
        '--test-end', type=str, default=None,
        help='Test end date (YYYY-MM-DD, default: 2024-12-31)',
    )
    parser.add_argument(
        '--train-end', type=str, default=None,
        help='Train end date (YYYY-MM-DD, default: 2023-12-31)',
    )
    parser.add_argument(
        '--test-start', type=str, default=None,
        help='Test start date (YYYY-MM-DD, default: 2024-01-01)',
    )
    parser.add_argument(
        '--universe', type=str, default=None,
        choices=['hs300', 'zz500', 'all_a'],
        help='Stock universe for real data (default: hs300)',
    )
    parser.add_argument(
        '--n-seeds', type=int, default=None,
        help='[legacy] No longer used (raw Alpha101 factors are not merged into '
             'the pool). Kept for backward CLI compat.',
    )
    parser.add_argument(
        '--n-seeds-hypothesis', type=int, default=None,
        help='Number of hypothesis-driven seed factors (default: config value)',
    )
    parser.add_argument(
        '--alpha101-top-k', dest='alpha101_top_k', type=int, default=None,
        help='Step 4c: number of Stage-1 Alpha101 library factors (top by |IC|) '
             'returned alongside the LLM-mined Stage-2 factors for joint '
             'evaluation (default: config evolution.alpha101_top_k; 0 = '
             'LLM-mined only).',
    )
    parser.add_argument(
        '--alpha101-max-workers', dest='alpha101_max_workers', type=int, default=None,
        help='Step 3 parallel worker count for scoring the Alpha101 library '
             'inside llm_mine_alpha101_inspired '
             '(default: config value; 0 = auto = min(32, cpu_count()+4)).',
    )
    parser.add_argument(
        '--n-seeds-memory-augment', type=int, default=None,
        help='Number of memory-augmented seed factors (default: config value)',
    )
    parser.add_argument(
        '--retrieve-alpha101', dest='retrieve_alpha101', action='store_true',
        default=None,
        help='Enable Step 4c Alpha101 library loading + LLM mining during factor '
             'generation. Scores the Alpha101 library on TRAIN data, builds '
             'inspiration chains, and uses the LLM to evolve novel expressions. '
             'Pass --no-retrieve-alpha101 to skip.',
    )
    parser.add_argument(
        '--no-retrieve-alpha101', dest='retrieve_alpha101', action='store_false',
        help='Disable Alpha101 loading during factor generation (Step 4c).',
    )
    parser.add_argument(
        '--n-shots', type=int, default=None,
        help='Few-shot examples injected into the LLM prompt by MemoryAugmentedGenerator (default: config value)',
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
        '--n-best4debate', dest='n_best4debate', type=int, default=None,
        help='Step 4d pre-step5 filter: keep top-N factors by |train IC| from '
             'the combined pool (evolved + memory + alpha101-llm). '
             '0 = disabled (pass all to Step 5). '
             'Set to match fusion.top_k to avoid wasting debate API calls.',
    )
    parser.add_argument(
        '--fusion-top-k', type=int, default=None,
        help='Cap the number of factors fused (final step6 top_k). '
             '0 or None → no cap (use config fusion.top_k). '
             'Only takes effect in --full mode (test mode loads pre-fused factors).',
    )
    parser.add_argument(
        '--skip-eval', dest='skip_eval', action='store_true', default=False,
        help='Skip Step 5 (multi-agent debate) and Step 5b (Chair synthesis) and '
             'go straight to Step 6 factor fusion. Useful when you want the fused '
             'factor built directly from the evolved best_factors (capped by '
             'fusion.top_k) without the LLM quality-gate round. When skipped, the '
             'step6 top_k cap ranks factors by |ICIR| instead of debate_score.',
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
             '(default: experiments/YYYYMMDD/fusion/final_factors.json)',
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
        evo_overrides = pipeline.config['evolution']
        if args.n_seeds_hypothesis is not None:
            evo_overrides['n_seeds_hypothesis'] = args.n_seeds_hypothesis
        if args.alpha101_top_k is not None:
            evo_overrides['alpha101_top_k'] = args.alpha101_top_k
        if args.alpha101_max_workers is not None:
            evo_overrides['alpha101_max_workers'] = args.alpha101_max_workers
        if args.n_seeds_memory_augment is not None:
            evo_overrides['n_seeds_memory_augment'] = args.n_seeds_memory_augment
        if args.retrieve_alpha101 is not None:
            evo_overrides['retrieve_alpha101'] = args.retrieve_alpha101
        if args.n_shots is not None:
            evo_overrides['n_shots'] = args.n_shots
        if args.n_seeds is not None:
            # [deprecated] No-op: alpha101_top_k is consumed in Step 3.
            pass
        if args.n_evolution_rounds != 5:
            pipeline.config['evolution']['max_rounds'] = args.n_evolution_rounds
        if args.n_best_factors is not None:
            pipeline.config['evolution']['n_best_factors'] = args.n_best_factors
        if args.n_best4debate is not None:
            pipeline.config['evolution']['n_best4debate'] = args.n_best4debate
        if args.fusion_top_k is not None:
            pipeline.config.setdefault('fusion', {})['top_k'] = args.fusion_top_k

        metrics = pipeline.run_full_pipeline(
            train_start_date=args.train_start,
            train_end_date=args.train_end,
            test_start_date=args.test_start,
            test_end_date=args.test_end,
            use_sample=not args.real,
            universe=args.universe,
            n_evolution_rounds=args.n_evolution_rounds,
            output_dir=args.output_dir,
            forward_period=args.forward_period,
            holding_period=args.holding_period,
            skip_eval=args.skip_eval,
        )
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Mean Rank-IC (test): {metrics.get('mean_rank_ic', 0):.4f}")
            print(f"  ICIR (test):    {metrics.get('icir', 0):.4f}")
            print(f"  Turnover:       {metrics.get('avg_turnover', 0):.4f}")
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
                experiments_dir.glob('*/*/fusion/final_factors.json'),
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
        # The split (train/test) is controlled by train_end_date / test_start_date
        # from config, which defaults to the last ~2 years as test.
        pipeline.step1_load_data(
            use_sample=not args.real,
            forward_period=args.forward_period,
            holding_period=args.holding_period,
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
            print(f"  Mean Rank-IC (test): {metrics.get('mean_rank_ic', 0):.4f}")
            print(f"  ICIR (test):    {metrics.get('icir', 0):.4f}")
            print(f"  Turnover:       {metrics.get('avg_turnover', 0):.4f}")
        else:
            print("\nTest pipeline completed, but no metrics available.")
    else:
        # Quick demo (default)
        metrics = run_demo()
        if metrics:
            print(f"\nFinal Performance: Sharpe = {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Mean Rank-IC (test): {metrics.get('mean_rank_ic', 0):.4f}")
            print(f"  ICIR (test):    {metrics.get('icir', 0):.4f}")
            print(f"  Turnover:       {metrics.get('avg_turnover', 0):.4f}")
        else:
            print("\nDemo completed, but no metrics available.")
