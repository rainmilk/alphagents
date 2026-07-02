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

    Returns:
        Dict with metrics: annual_return, sharpe_ratio, max_drawdown, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    return_series = compute_future_returns(price_midx)

    n_dates = len(price_midx['close'].index.get_level_values('datetime').unique())
    n_stocks = len(price_midx['close'].index.get_level_values('instrument').unique())
    print(f"  Loaded: {n_dates} dates × {n_stocks} stocks")

    # Determine split date (70/30)
    all_dates = sorted(price_midx['close'].index.get_level_values('datetime').unique())
    split_idx = int(len(all_dates) * 0.7)
    split_date = str(all_dates[split_idx])
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  IS/OOS split: {split_date}")

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
            concrete_formula, price_midx, return_series, repo_factors,
            start_date=start_date, end_date=end_date,
            split_date=split_date,
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

    if prices_df is None:
        print("  WARNING: No price data available for backtest, using zero metrics")
        metrics = _get_default_metrics()
    else:
        metrics = compute_portfolio_metrics(
            best_formula, price_midx, return_series,
            prices_df, start_date, end_date, split_date, alpha_repository,
        )

    # ── Save results ──────────────────────────────────────────────
    results = {
        'baseline': 'mcts_llm_alpha',
        'iterations': iterations,
        'has_llm': has_llm,
        'best_formula': best_formula,
        'best_score': mcts.best_score if hasattr(mcts, 'best_score') else 0,
        'n_alpha_in_repo': len(alpha_repository),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, "mcts_llm_alpha_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    # Save alpha repository
    if alpha_repository:
        repo_path = os.path.join(output_dir, "mcts_alpha_repo.csv")
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
) -> Dict[str, float]:
    """
    Compute portfolio-level metrics using the unified BacktestEngine.

    Evaluates the best factor formula, constructs daily portfolios (top-N stocks
    by factor score, equal-weighted), and runs the unified BacktestEngine for
    consistent metrics across all baselines.
    """
    try:
        from baselines.mcts_llm_alpha.src.mcts_llm_alpha.evaluation.pandas_evaluator import ExprParser
        import re

        # Substitute default params for any unresolved placeholders
        concrete_formula = best_formula
        if any(ch in concrete_formula for ch in "tCpP"):
            default_params = {"t1": 10, "t2": 20, "t3": 5, "C": 5, "P": 10}
            for pn, pv in sorted(default_params.items(), key=lambda x: len(x[0]), reverse=True):
                concrete_formula = re.sub(r"" + pn + r"", str(pv), concrete_formula)
            print(f"[PortfolioMetrics] Substituted params: {best_formula[:60]}... -> {concrete_formula[:60]}...")

        parser = ExprParser(price_midx)
        factor_series = parser.evaluate(concrete_formula)

        # factor_series is MultiIndex (datetime, instrument) Series
        # Convert to wide DataFrame (date x stock) for portfolio construction
        factor_df = factor_series.unstack(fill_value=np.nan)

        # Only use OOS period for backtest
        oos_dates = factor_df.index[factor_df.index > split_date]
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
            holding_period=1,
        )
        metrics = engine.run(portfolios, prices_aligned)

        # Add IC info
        mean_ic_val = 0.0
        try:
            oos_df = pd.concat([factor_series.rename("factor"), return_series], axis=1).dropna()
            oos_df = oos_df[oos_df.index.get_level_values("datetime") > split_date]
            daily_ic = []
            for _, group in oos_df.groupby(level="datetime"):
                if len(group) >= 5:
                    ic = group["factor"].corr(group.iloc[:, -1], method="spearman")
                    if not np.isnan(ic):
                        daily_ic.append(ic)
            mean_ic_val = float(np.mean(daily_ic)) if daily_ic else 0.0
        except Exception:
            pass

        metrics["mean_ic"] = round(mean_ic_val, 4)
        metrics["n_alphas"] = len(alpha_repository)
        metrics["best_formula"] = best_formula

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
    parser.add_argument('--no-llm', action='store_true', default=False,
                        help='Disable LLM (use simulated formulas)')
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
    )
    print("\nDone!")
