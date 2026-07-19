#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCTS-LLM-Alpha baseline runner using the main project's DataLoader.

Replaces Qlib data access with pandas-based evaluation using data from
the main DataLoader. Falls back to LLM-less simulation when no API key is available.

Usage:
    python baselines/run_mcts_llm_alpha.py
    python baselines/run_mcts_llm_alpha.py --iterations 20 --output-dir experiments/mcts_test
"""

import sys
import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "mcts-llm-alpha" / "src"))

from dataloader.loader import DataLoader
from baselines.mcts_llm_alpha.src.mcts_llm_alpha.evaluation.pandas_evaluator import (
    convert_to_multindex,
    compute_future_returns,
    evaluate_formula_pandas,
)
from baselines.mcts_llm_alpha.src.mcts_llm_alpha.evaluation.relative_ranking import RelativeRankingEvaluator
from baselines.mcts_llm_alpha.src.mcts_llm_alpha.config import load_config as load_mcts_config

logger = logging.getLogger(__name__)


# ── Core runner ───────────────────────────────────────────────────────

def run_mcts_llm_alpha_baseline(
    config_path: str = "config/config.yaml",
    output_dir: str = "experiments/mcts_llm_alpha",
    iterations: int = 20,
    use_llm: bool = True,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    forward_period: int = 10,
    holding_period: int = None,  # None -> use mcts_config.holding_period (default 1)
) -> Dict:
    """
    Run MCTS-LLM-Alpha baseline using the main project's DataLoader.

    Args:
        config_path: Path to main project config YAML
        output_dir: Directory to save results
        iterations: Maximum MCTS iterations
        use_llm: Whether to use LLM for formula generation (requires OPENAI_API_KEY)
        start_date: Data start date
        end_date: Data end date
        train_end_date: Last training (IS) date (YYYY-MM-DD). MCTS search & LLM
            only see data up to this date. Falls back to data_config.train_end_date
            then '2023-12-31'.
        test_start_date: First test (OOS) date (YYYY-MM-DD). Evaluation/backtest
            only use data from this date onward. Falls back to
            data_config.test_start_date then '2024-01-01'.

    Returns:
        Dict with metrics: annual_return, sharpe_ratio, max_drawdown, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Date-isolated run directory (one subdir per execution, for multiple runs)
    _date_str = pd.Timestamp.now().strftime("%Y%m%d")
    run_dir = os.path.join(output_dir, _date_str)
    os.makedirs(run_dir, exist_ok=True)

    # ── 1. Load data from main DataLoader ──────────────────────────
    print("=" * 60)
    print("[1/6] Loading data from main DataLoader...")
    print("=" * 60)

    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_series = loader.load_data(
        start_date=start_date,
        end_date=end_date,
    )

    # Convert to MultiIndex format for the pandas evaluator
    price_midx = convert_to_multindex(price_data)
    return_series = compute_future_returns(price_midx, forward_period=forward_period)

    n_dates = len(price_midx['close'].index.get_level_values('datetime').unique())
    n_stocks = len(price_midx['close'].index.get_level_values('instrument').unique())
    print(f"  Loaded: {n_dates} dates × {n_stocks} stocks")
    print(f"  Forward period: {forward_period}d")

    # ── Determine IS/OOS split using explicit dates (matches other baselines) ──
    # Resolve from explicit args first, then fall back to data_config — same
    # precedence as run_alpha_xgboost / run_lstm_baseline / run_alphagrail.
    #   train_end_date : last date the MCTS/LLM search is allowed to see (IS)
    #   test_start_date: first date used for out-of-sample (OOS) evaluation
    # This replaces the old fixed 70/30 percentage split so all baselines
    # share a consistent, comparable train/test boundary driven by calendar dates.
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    # split_date is the OOS boundary used by downstream portfolio/IC filtering.
    # It is the first OOS date, so OOS filtering uses >= split_date (inclusive).
    split_date = test_start

    all_dates = sorted(price_midx['close'].index.get_level_values('datetime').unique())
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  IS/OOS split: train_end={train_end}, test_start={test_start}")

    # ── 1b. Create train-only data for MCTS search & LLM ──────────
    # MCTS search and LLM formula generation/refinement must only see
    # training (IS) data. Using full data (including OOS) during search
    # causes data leakage — the search optimizes on future information.
    train_dates = [d for d in all_dates if pd.Timestamp(str(d)) <= train_end_ts]
    print(f"  Train period: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} dates)")

    train_price_midx = {}
    for key, series in price_midx.items():
        if isinstance(series, pd.Series) and isinstance(series.index, pd.MultiIndex):
            train_mask = series.index.get_level_values('datetime').isin(train_dates)
            train_price_midx[key] = series[train_mask]
        else:
            train_price_midx[key] = series

    if isinstance(return_series, pd.Series) and isinstance(return_series.index, pd.MultiIndex):
        train_mask = return_series.index.get_level_values('datetime').isin(train_dates)
        train_return_series = return_series[train_mask]
    else:
        train_return_series = return_series

    print(f"  Train data: {len(train_dates)} dates (MCTS/LLM will only see this)")

    # ── 2. Create pandas-based evaluator ───────────────────────────
    print("\n[2/6] Creating pandas-based evaluator...")

    # Load mcts config for thresholds
    mcts_config = load_mcts_config()

    # Create evaluate function compatible with what MCTS expects
    def formula_evaluator(formula, repo_factors, node=None):
        """
        Evaluate a Qlib formula using pandas data.
        Returns (scores_dict, factor_dataframe, raw_scores_dict)
        """
        # Substitute symbolic params (w1, w2, ...) if node has formula_info
        concrete_formula = formula
        if node and hasattr(node, 'formula_info') and node.formula_info:
            params_keywords = ['w1', 'w2', 'w3', 'w4', 't1', 't2', 't3']
            if any(p in formula for p in params_keywords):
                selected = node.formula_info.get('selected_params', {})
                if selected:
                    concrete_formula = formula
                    for pn, pv in sorted(selected.items(), key=lambda x: len(x[0]), reverse=True):
                        concrete_formula = re.sub(r'\b' + pn + r'\b', str(pv), concrete_formula)
                    print(f"[Eval] Param substitution: {formula[:60]}... → {concrete_formula[:60]}...")

        raw_scores, factor_df = evaluate_formula_pandas(
            concrete_formula, train_price_midx, train_return_series, repo_factors,
            start_date=start_date, end_date=train_end,
            split_date=None,  # Auto-split within train data for Overfitting metric
            ic_method=mcts_config.evaluation.ic_method,
        )
        if raw_scores is None:
            return None, None, None

        # Use relative ranking to convert raw scores to 0-10 range
        ranker = RelativeRankingEvaluator(
            effectiveness_threshold=mcts_config.evaluation.effectiveness_threshold,
            min_repository_size=3,
        )
        # Add some baseline scores so ranking works
        for _ in range(min(5, len(repo_factors) + 1)):
            if len(ranker.effective_repository) < ranker.min_repository_size:
                ranker.effective_repository.append({
                    'IC': 0.01, 'IR': 0.3, 'Turnover': 0.4, 'Diversity': 0.8,
                })

        scores = ranker.evaluate_formula_with_relative_ranking(raw_scores, repo_factors)
        return scores, factor_df, raw_scores

    # Create comprehensive evaluator wrapping the pandas evaluation
    class PandasComprehensiveEvaluator:
        """Adapter that uses pandas evaluator but implements the same interface."""

        def __init__(self, conf):
            self.config = conf
            self.ranking_evaluator = RelativeRankingEvaluator(
                effectiveness_threshold=conf.evaluation.effectiveness_threshold,
                min_repository_size=3,
            )
            # Seed with baseline metrics so relative ranking works early
            for _ in range(5):
                self.ranking_evaluator.effective_repository.append({
                    'IC': 0.01, 'IR': 0.3, 'Turnover': 0.4, 'Diversity': 0.8,
                })

        def evaluate_formula(self, formula, repo_factors, node=None):
            return formula_evaluator(formula, repo_factors, node)

    evaluator = PandasComprehensiveEvaluator(mcts_config)
    print("  Evaluator ready (pandas-based, no Qlib needed)")

    # ── 3. Initialize MCTS components ──────────────────────────────
    print("\n[3/6] Initializing MCTS components...")

    # Check for LLM availability
    # Read API key and base_url from config first, fall back to env variable
    api_key = ""
    base_url = None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        api_key = main_config.get('llm', {}).get('generator', {}).get('api_key', '')
        base_url = main_config.get('llm', {}).get('generator', {}).get('base_url')
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    has_llm = use_llm and api_key and api_key != "your_openai_api_key_here"

    if has_llm:
        print(f"  LLM enabled: model={mcts_config.llm.model}")
        from baselines.mcts_llm_alpha.src.mcts_llm_alpha.llm import LLMClient
        from baselines.mcts_llm_alpha.src.mcts_llm_alpha.llm.wrapper import create_formula_generator, create_formula_refiner

        llm_client = LLMClient(api_key=api_key, model=mcts_config.llm.model, base_url=base_url)
        formula_generator = create_formula_generator(llm_client, evaluator)
        formula_refiner = create_formula_refiner(llm_client, evaluator)
    else:
        print("  LLM disabled (no API key) — using formula simulation mode")
        from baselines.mcts_llm_alpha.src.mcts_llm_alpha.mcts import MCTSSearch
        formula_generator, formula_refiner = _create_simulated_formula_funcs()

    # ── 4. Create MCTS search ─────────────────────────────────────
    print("\n[4/6] Creating MCTS search instance...")

    from baselines.mcts_llm_alpha.src.mcts_llm_alpha.mcts import MCTSSearch

    mcts = MCTSSearch(
        formula_generator=formula_generator,
        formula_refiner=formula_refiner,
        formula_evaluator=formula_evaluator,
        max_iterations=iterations,
        budget_increment=mcts_config.mcts.budget_increment,
        exploration_constant=mcts_config.mcts.exploration_constant,
        max_depth=mcts_config.mcts.max_depth,
        max_nodes=mcts_config.mcts.max_nodes,
        checkpoint_freq=mcts_config.mcts.checkpoint_freq,
        dimension_temperature=mcts_config.mcts.dimension_temperature,
        effectiveness_threshold=mcts_config.mcts.effectiveness_threshold,
        diversity_threshold=mcts_config.evaluation.diversity_threshold,
        overall_threshold=mcts_config.evaluation.overall_threshold,
        output_dir=run_dir,
    )

    if has_llm and 'llm_client' in locals():
        mcts.llm_client = llm_client

    print(f"  Max iterations: {iterations}")

    # ── 5. Run MCTS search ────────────────────────────────────────
    print(f"\n[5/6] Running MCTS search ({iterations} iterations)...")
    print("=" * 60)

    try:
        best_formula, alpha_repository = mcts.run()

        print("\n" + "=" * 60)
        print("Search complete!")
        print(f"  Best formula: {best_formula}")
        print(f"  Best score: {mcts.best_score:.3f}")
        print(f"  Alpha repository size: {len(alpha_repository)}")
    except Exception as e:
        print(f"  MCTS search error: {e}")
        import traceback
        traceback.print_exc()
        best_formula = "Rank(($close - Ref($close, 5)) / Std($close, 20))"
        alpha_repository = []

    # ── 6. Compute portfolio metrics (unified BacktestEngine) ────
    print(f"\n[6/6] Computing portfolio metrics from best factor...")

    # Get prices DataFrame for BacktestEngine
    prices_df = price_data.get('close') if isinstance(price_data, dict) else None
    if prices_df is None:
        # Try to build from price_midx
        close_series = price_midx.get('close')
        if close_series is not None:
            prices_df = close_series.unstack('instrument')

    # Retrieve selected_params from the best MCTS node (resolves symbolic w1, w2, etc.)
    best_selected_params = getattr(mcts, 'best_selected_params', None) or {}
    if best_selected_params:
        print(f"  Retrieved selected_params from MCTS: {best_selected_params}")
    else:
        print("  No selected_params found in MCTS — will use default param substitution")

    # Split data into IS (train) and OOS (test) — only use test data for final evaluation
    # to avoid data leakage
    test_dates = [d for d in all_dates if pd.Timestamp(str(d)) >= test_start_ts]
    if len(test_dates) < 10:
        print(f"  WARNING: Only {len(test_dates)} OOS dates, using full data")
        test_price_midx = price_midx
        test_return_series = return_series
        test_prices_df = prices_df
    else:
        print(f"  Filtering to OOS period: {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} dates)")
        # Filter price_midx (dict of MultiIndex Series) to OOS dates
        test_price_midx = {}
        for key, series in price_midx.items():
            if isinstance(series, pd.Series) and isinstance(series.index, pd.MultiIndex):
                # MultiIndex: (datetime, instrument)
                oos_mask = series.index.get_level_values('datetime').isin(test_dates)
                test_price_midx[key] = series[oos_mask]
            else:
                test_price_midx[key] = series

        # Filter return_series to OOS dates
        if isinstance(return_series, pd.Series) and isinstance(return_series.index, pd.MultiIndex):
            oos_mask = return_series.index.get_level_values('datetime').isin(test_dates)
            test_return_series = return_series[oos_mask]
        else:
            test_return_series = return_series

        # Filter prices_df to OOS dates
        if prices_df is not None:
            test_prices_df = prices_df.loc[prices_df.index.isin(test_dates)]
        else:
            test_prices_df = None

    if test_prices_df is None:
        print("  WARNING: No price data available for backtest, using zero metrics")
        metrics = _get_default_metrics()
    else:
        # Pass FULL price_midx (not test_price_midx) to compute_portfolio_metrics
        # so ExprParser has complete history for rolling-window operations
        # (e.g. Std($close, 20) needs 20 days of history before OOS start).
        # This is NOT data leakage: factor values at time t only use data up to t.
        # The function internally filters to OOS for portfolio construction (L493)
        # and IC computation (L546).
        metrics = compute_portfolio_metrics(
            best_formula, price_midx, test_return_series,
            test_prices_df, start_date, end_date, split_date, alpha_repository,
            selected_params=best_selected_params,
            save_dir=run_dir,
            holding_period=holding_period if holding_period is not None else mcts_config.holding_period,
        )

    # ── Save results ──────────────────────────────────────────────
    results = {
        'baseline': 'mcts_llm_alpha',
        'iterations': iterations,
        'has_llm': has_llm,
        'best_formula': best_formula,
        'best_selected_params': best_selected_params if best_selected_params else None,
        'best_score': mcts.best_score if hasattr(mcts, 'best_score') else 0,
        'n_alpha_in_repo': len(alpha_repository),
        'forward_period': forward_period,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(run_dir, "mcts_llm_alpha_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    # Save alpha repository
    if alpha_repository:
        repo_path = os.path.join(run_dir, "mcts_alpha_repo.csv")
        repo_df = pd.DataFrame([{
            'formula': a.get('formula', ''),
            'effectiveness': a.get('scores', {}).get('Effectiveness', 0),
            'stability': a.get('scores', {}).get('Stability', 0),
            'turnover': a.get('scores', {}).get('Turnover', 0),
            'diversity': a.get('scores', {}).get('Diversity', 0),
            'overfitting': a.get('scores', {}).get('Overfitting', 0),
        } for a in alpha_repository])
        repo_df.to_csv(repo_path, index=False)
        print(f"  Alpha repo saved: {repo_path}")

    print(f"\n  Results saved: {results_path}")
    print(f"\n  Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    return results


# ── Simulated formula generation (no LLM) ─────────────────────────────

def _create_simulated_formula_funcs():
    """
    Create formula generator/refiner that produces synthetic formulas
    without LLM calls. Useful for testing or when API key is unavailable.
    """
    import random

    FIELDS = ['$close', '$open', '$high', '$low', '$volume']
    # Windowed ops take 2 args: Func(field, n)
    OPS_WINDOW = ['Std', 'Mean', 'Sum', 'Min', 'Max', 'Med', 'Skew', 'Kurt']
    # Pure unary ops take 1 arg: Func(field)
    OPS_PURE_UNARY = ['Log', 'Abs', 'Sigmoid']

    call_count = [0]  # mutable counter

    def _random_formula():
        """Generate a random Qlib expression."""
        field1 = random.choice(FIELDS)
        use_pure_unary = random.random() < 0.15

        if use_pure_unary:
            op1 = random.choice(OPS_PURE_UNARY)
            inner = f"{op1}({field1})"
        else:
            op1 = random.choice(OPS_WINDOW)
            w1 = random.choice([5, 10, 20, 30, 60])
            inner = f"{op1}({field1}, {w1})"

        if random.random() < 0.5:
            field2 = random.choice(FIELDS)
            w2 = random.choice([5, 10, 20])
            inner = f"({inner} - Ref({field2}, {w2}))"

        # Rank: accepts 1 or 2 args (second is ignored by parser)
        w3 = random.choice([5, 10, 20])
        formula = f"Rank({inner}, {w3})"
        return formula

    def generate_formula():
        call_count[0] += 1
        formula = _random_formula()
        portrait = f"Synthetic alpha #{call_count[0]}: {formula}"
        return formula, portrait

    def refine_formula(node, dimension, avoid_patterns=None,
                       repo_examples=None, node_context=None):
        call_count[0] += 1
        formula = _random_formula()
        portrait = f"Refined on {dimension}: {formula}"
        desc = f"Improved {dimension.lower()} by adjusting parameters"
        info = {}
        return formula, portrait, desc, info

    print("  Using simulated formula generation (no LLM)")
    return generate_formula, refine_formula


# ── Portfolio metrics computation ─────────────────────────────────────

def compute_portfolio_metrics(
    best_formula: str,
    price_midx: Dict[str, pd.Series],
    return_series: pd.Series,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    split_date: str,
    alpha_repository: list,
    top_n_stocks: int = 30,
    selected_params: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    holding_period: int = 1,
) -> Dict[str, float]:
    """
    Compute portfolio-level metrics using the unified BacktestEngine.

    Evaluates the best factor formula, constructs daily portfolios (top-N stocks
    by factor score, equal-weighted), and runs the unified BacktestEngine for
    consistent metrics across all baselines.

    Args:
        price_midx: FULL price data (all dates including IS period). ExprParser
                    needs complete history for rolling-window operations (e.g.
                    Std, Mean, Ref). OOS filtering is done internally.
        return_series: OOS-only return series (or full — filtered internally).
        prices: OOS-only prices DataFrame (or full — reindexed internally).
        selected_params: Concrete parameter values from the MCTS best node
                         (e.g. {"w1": 20, "w2": 30, "w3": 10}). Used to
                         resolve symbolic params in best_formula.
    """
    try:
        from baselines.mcts_llm_alpha.src.mcts_llm_alpha.evaluation.pandas_evaluator import ExprParser
        import re

        # Resolve symbolic parameters (w1, w2, t1, t2, etc.)
        # Priority: (1) selected_params from MCTS node, (2) comprehensive defaults
        concrete_formula = best_formula

        # All possible symbolic params used by the LLM prompt template
        all_param_keywords = ["w1", "w2", "w3", "w4", "t1", "t2", "t3"]
        has_symbolic = any(re.search(r"\b" + kw + r"\b", concrete_formula) for kw in all_param_keywords)

        if has_symbolic:
            # Start with selected_params from MCTS (if available), then fill gaps with defaults
            default_params = {
                "w1": 20, "w2": 30, "w3": 10, "w4": 5,
                "t1": 10, "t2": 20, "t3": 5,
            }
            resolved_params = {**default_params, **(selected_params or {})}

            # Sort by name length descending to avoid partial replacements
            for pn, pv in sorted(resolved_params.items(), key=lambda x: len(x[0]), reverse=True):
                # Use word boundaries to avoid corrupting other tokens
                concrete_formula = re.sub(r"\b" + re.escape(pn) + r"\b", str(pv), concrete_formula)

            print(f"[PortfolioMetrics] Param substitution:")
            print(f"  Before: {best_formula}")
            print(f"  After:  {concrete_formula}")
            if selected_params:
                print(f"  MCTS params used: {selected_params}")
            else:
                print(f"  (used default params - no MCTS selected_params available)")

        parser = ExprParser(price_midx)
        factor_series = parser.evaluate(concrete_formula)

        # factor_series is MultiIndex (datetime, instrument) Series
        # Convert to wide DataFrame (date x stock) for portfolio construction
        factor_df = factor_series.unstack(fill_value=np.nan)

        # Only use OOS period for backtest
        oos_dates = factor_df.index[factor_df.index >= split_date]
        if len(oos_dates) < 10:
            oos_dates = factor_df.index
        factor_oos = factor_df.loc[oos_dates]

        # Build portfolios: select top-N stocks by factor score each day
        portfolio_rows = []
        portfolio_dates = []

        for date in factor_oos.index:
            scores = factor_oos.loc[date].dropna()
            if scores.empty:
                continue
            top = scores.nlargest(top_n_stocks)
            if len(top) == 0:
                continue
            w = pd.Series(1.0 / len(top), index=top.index)
            portfolio_rows.append(w)
            portfolio_dates.append(date)

        if not portfolio_rows:
            return _get_default_metrics()

        # Build portfolios DataFrame
        all_stocks = pd.Index(set().union(*(w.index for w in portfolio_rows)))
        portfolios = pd.DataFrame(
            index=pd.DatetimeIndex(portfolio_dates),
            columns=all_stocks,
            dtype=float,
        )
        for i, w in enumerate(portfolio_rows):
            portfolios.loc[portfolio_dates[i], w.index] = w.values
        portfolios = portfolios.fillna(0.0)
        portfolios = portfolios.div(portfolios.sum(axis=1), axis=0).fillna(0.0)

        # Align prices to portfolio dates
        prices_aligned = prices.reindex(portfolios.index)
        prices_aligned = prices_aligned.reindex(columns=portfolios.columns)

        # Run unified backtest
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(
            commission=0.0003,
            slippage=0.001,
            risk_free_rate=0.0,
            holding_period=holding_period,
        )
        metrics = engine.run(portfolios, prices_aligned, save_dir=save_dir)

        # Add IC info
        mean_ic_val = 0.0
        daily_ic = []
        try:
            oos_df = pd.concat([factor_series.rename("factor"), return_series], axis=1).dropna()
            oos_df = oos_df[oos_df.index.get_level_values("datetime") >= split_date]
            for _, group in oos_df.groupby(level="datetime"):
                if len(group) >= 5:
                    ic = group["factor"].corr(group.iloc[:, -1], method="spearman")
                    if not np.isnan(ic):
                        daily_ic.append(ic)
            mean_ic_val = float(np.mean(daily_ic)) if daily_ic else 0.0
        except Exception:
            pass

        metrics["mean_ic"] = round(mean_ic_val, 4)
        # ICIR = mean / std of the daily (test-period) IC series.
        if len(daily_ic) > 1 and np.std(daily_ic, ddof=1) > 0:
            metrics["icir"] = round(float(mean_ic_val / np.std(daily_ic, ddof=1)), 4)
        else:
            metrics["icir"] = 0.0
        metrics["n_alphas"] = len(alpha_repository)
        metrics["best_formula"] = best_formula

        # ── FINAL RESULTS (test period) ──
        print("\n" + "=" * 60)
        print("FINAL RESULTS (Test Period)")
        print("=" * 60)
        print(f"  Annual Return:    {metrics.get('annual_return', 0):.4f}")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.4f}")
        print(f"  Information Ratio:{metrics.get('information_ratio', 0):.4f}")
        print(f"  Mean IC (test):   {metrics.get('mean_ic', 0):.4f}")
        print(f"  ICIR (test):      {metrics.get('icir', 0):.4f}")
        print(f"  Turnover:         {metrics.get('avg_turnover', 0):.4f}")

        return metrics

    except Exception as e:
        print(f"  Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return _get_default_metrics()

def _get_default_metrics() -> Dict:
    return {
        'annual_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'information_ratio': 0.0,
        'mean_ic': 0.0,
        'icir': 0.0,
        'avg_turnover': 0.0,
        'n_alphas': 0,
    }


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='MCTS-LLM-Alpha Baseline Runner')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Maximum MCTS iterations (default: 20)')
    parser.add_argument('--output-dir', type=str, default='experiments/mcts_llm_alpha',
                        help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Main project config path')
    parser.add_argument('--start-date', type=str, default='2020-01-01')
    parser.add_argument('--end-date', type=str, default='2023-12-31')
    parser.add_argument('--train-end', type=str, default=None,
                        help='Last training (IS) date YYYY-MM-DD (default: config train_end_date)')
    parser.add_argument('--test-start', type=str, default=None,
                        help='First test (OOS) date YYYY-MM-DD (default: config test_start_date)')
    parser.add_argument('--no-llm', action='store_true', default=False,
                        help='Disable LLM (use simulated formulas)')
    parser.add_argument('--forward-period', type=int, default=10,
                        help='Forward return period in days (default: 10, should match MASE)')
    parser.add_argument('--holding-period', type=int, default=None,
                        help='Portfolio holding period in days for backtest (default: config value, 1 = daily rebalance)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_mcts_llm_alpha_baseline(
        config_path=args.config,
        output_dir=args.output_dir,
        iterations=args.iterations,
        use_llm=not args.no_llm,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        forward_period=args.forward_period,
        holding_period=args.holding_period,
    )
    print("\nDone!")
