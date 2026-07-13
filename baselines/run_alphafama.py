# -*- coding: utf-8 -*-
"""
AlphaFAMA Baseline Runner — Integrated with Main Dataloader + LLM Alpha-Mining

This runner:
1. Loads A-share data via the main project's DataLoader
2. Converts to AlphaFAMA's expected format via data_bridge
3. Runs AlphaFAMA's 101-factor generation + IC computation
4. Clusters factors by IC profiles (K-Means)
5. LLM alpha-mining: iteratively generates new factors by fusing top performers
6. Merges LLM-generated + original factors, evaluates via IC-weighted portfolio
7. Returns performance metrics compatible with the main project

The LLM alpha-mining follows the original FAMA (FActor Mining Agent) framework:
  - For each cluster of similar factors, maintain an "experience chain" of top formulas
  - Ask LLM to generate a NEW factor that fuses/extends the chain's best formulas
  - Evaluate the new factor by Rank IC
  - If it performs well, add it to the chain (keep top-15 by |IC|)
  - Iterate for N rounds

Usage:
    python baselines/run_alphafama.py
    python baselines/run_alphafama.py --start 2020-01-01 --end 2024-12-31 --universe hs300
    python baselines/run_alphafama.py --no-llm          # disable LLM mining
    python baselines/run_alphafama.py --llm-iters 20    # set LLM iterations

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
"""

import sys
import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

# ── Path setup: add project root to sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.loader import DataLoader
from baselines.AlphaFAMA.src.data_bridge import (
    convert_price_data_to_alphafama,
    split_alphafama_data,
)
# NOTE: AlphaFAMA modules use relative imports; we import them via the
# full package path to avoid polluting sys.path with AlphaFAMA's src/
# (which would shadow the main project's config.py).
from baselines.AlphaFAMA.src.alpha_functions import AlphaFactory
from baselines.AlphaFAMA.src.factor_matrix import compute_ic_matrix
from baselines.AlphaFAMA.src.constants.formula_map import FORMULA_MAP
from baselines.AlphaFAMA.src.utils.generate_rankics import (
    factor_series_fn,
    compute_rankic,
)
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LLM configuration and function definitions
# ═══════════════════════════════════════════════════════════════════════

# Alpha101 function library description (from original FAMA config)
FUNCTION_DEFINITION = """\
# Functions and Operators from Alpha101:
# returns(x): period-over-period return
# ts_sum(x,n): rolling sum over n periods
# sma(x,n): moving average of x
# stddev(x,n): rolling standard deviation
# correlation(x,y,n): rolling correlation
# covariance(x,y,n): rolling covariance
# ts_rank(x,n): rolling rank (percentile)
# delta(x,n): difference over n lags
# delay(x,n): lag by n periods
# rank(x): cross-sectional rank into [0,1]
# scale(x): scale so sum|x|=1
# ts_argmax(x,n): argmax over n
# ts_argmin(x,n): argmin over n
# decay_linear(x,n): linear decay MA over n
# sign(x): sign of x (+1, 0, -1)
# product(x,n): rolling product over n
# ts_min(x,n): rolling minimum over n
# ts_max(x,n): rolling maximum over n
# abs(x): absolute value
# log(x): natural logarithm
# Available data columns: open, high, low, close, volume, vwap, returns, vol
# adv{n}: n-day moving average of volume (e.g. adv20 = sma(volume, 20))
# Arithmetic operators: +, -, *, /, ** (power)
# Comparison: <, >, <=, >=, ==, !=
# Ternary: (condition ? value_if_true : value_if_false)
"""

# Prompt template (from original deepseek_agent.py)
PROMPT_TEMPLATE = """\
Instruction
You are an alpha generator. You should follow the following rules:
1. The inputs are the alpha factors that are currently performing well, and you are
   required to output a new alpha factor that is generated from the fusion of
   these factors, and your factor must be different from the input factor.
2. Do not repeat example answer.
3. You should return new different factors in a json array.
4. The specific function is defined as follows:
{function_definition}
5. Follow the path in "improve_path". -> Indicates that the following factors have
   better performance than the previous factors. You should refer it to build new
   alpha.

Input Example
alphas: {css}
generate_factor_num: 1
improve_path: {chain}

Output Example
["rank(correlation(open, volume, 10) / rank(open))"]
"""

SYSTEM_PROMPT = (
    "You are an alpha-mining agent implementing the FAMA (FActor Mining Agent) framework. "
    "Generate a new, interpretable financial alpha factor expression as JSON. "
    "The expression should use the Alpha101 function library and standard arithmetic operators. "
    "Only use the functions and data columns listed in the function definition."
)


def _read_llm_config(config_path: str) -> Dict:
    """
    Read LLM configuration from the main project's config.yaml.

    Reads from the 'llm.generator' section, which contains:
        api_key, base_url, model, temperature
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get('llm', {}).get('generator', {})
    return {
        'api_key': llm_cfg.get('api_key', ''),
        'base_url': llm_cfg.get('base_url', ''),
        'model': llm_cfg.get('model', ''),
        'temperature': llm_cfg.get('temperature', 0.7),
    }


def _llm_generate_factor(
    api_key: str,
    base_url: str,
    model: str,
    chain_formulas: List[str],
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Call LLM to generate a new alpha factor expression.

    Args:
        api_key: OpenAI-compatible API key
        base_url: API base URL
        model: Model name
        chain_formulas: List of top-performing formula expressions in the cluster
        temperature: LLM temperature

    Returns:
        A factor expression string, or None if generation failed
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = PROMPT_TEMPLATE.format(
        function_definition=FUNCTION_DEFINITION,
        css=json.dumps(chain_formulas, ensure_ascii=False),
        chain=" -> ".join(chain_formulas),
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip Markdown code fences if present
        fence_match = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence_match:
            raw = fence_match.group(1).strip()

        # Parse JSON array
        gen = json.loads(raw)
        if isinstance(gen, list) and len(gen) > 0:
            return gen[0]
        elif isinstance(gen, str):
            return gen
        else:
            return None

    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON output, skipping")
        return None
    except Exception as e:
        logger.warning(f"LLM API call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Clustering (from original clustering.py)
# ═══════════════════════════════════════════════════════════════════════

def _cluster_factors(
    ic_df: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.Series:
    """
    Cluster factors by their IC time-series profiles using K-Means.

    Args:
        ic_df: IC matrix (dates x factors)
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Series mapping factor name -> cluster label
    """
    from sklearn.cluster import KMeans

    factor_cols = [c for c in ic_df.columns if c.startswith("alpha")]
    if len(factor_cols) == 0:
        return pd.Series(dtype=int)

    X = ic_df[factor_cols].fillna(0).T

    actual_k = min(n_clusters, len(X))
    if actual_k < 1:
        return pd.Series(dtype=int)

    model = KMeans(n_clusters=actual_k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    return pd.Series(labels, index=X.index, name="cluster")


# ═══════════════════════════════════════════════════════════════════════
# LLM Alpha-Mining loop (adapted from deepseek_agent.py)
# ═══════════════════════════════════════════════════════════════════════

def _build_cluster_chains(
    clusters: pd.Series,
    formula_map: Dict[str, str],
    mean_ic: pd.Series,
    max_chain_len: int = 5,
) -> Dict[str, List[str]]:
    """
    Build initial experience chains for each cluster.

    For each cluster, select the top formulas by |IC| as the seed chain.

    Args:
        clusters: Series mapping factor name -> cluster label
        formula_map: Dict mapping alphaXXX -> formula expression string
        mean_ic: Mean IC per factor (Series, index=factor name)
        max_chain_len: Max number of formulas per chain

    Returns:
        Dict mapping "Cluster_{label}" -> list of formula expressions
    """
    chains = {}
    for cluster_label, group_factors in clusters.groupby(clusters):
        # Sort by |IC| descending
        sorted_factors = group_factors.index[
            group_factors.index.map(lambda f: abs(mean_ic.get(f, 0)))
            .argsort()[::-1]
        ]
        chain = []
        for fname in sorted_factors[:max_chain_len]:
            formula = formula_map.get(fname, fname)
            chain.append(formula)
        if chain:
            chains[f"Cluster_{cluster_label}"] = chain
    return chains


def _run_llm_mining(
    train_df: pd.DataFrame,
    clusters: pd.Series,
    formula_map: Dict[str, str],
    mean_ic: pd.Series,
    llm_config: Dict,
    n_iters: int = 10,
    max_chain_len: int = 15,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Run LLM alpha-mining iterations.

    For each iteration:
      1. For each cluster, send the current top formulas to LLM
      2. LLM generates a new factor expression
      3. Evaluate the new factor by Rank IC
      4. If it performs well, add to the chain

    Args:
        train_df: Training data in AlphaFAMA format (MultiIndex date, ticker)
        clusters: Factor -> cluster label mapping
        formula_map: alphaXXX -> formula expression string
        mean_ic: Mean IC per factor
        llm_config: Dict with api_key, base_url, model, temperature
        n_iters: Number of mining iterations
        max_chain_len: Max chain length per cluster

    Returns:
        Tuple of (list of LLM-generated factor expressions, dict of factor -> Rank IC)
    """
    api_key = llm_config.get('api_key', '')
    base_url = llm_config.get('base_url', '')
    model = llm_config.get('model', '')
    temperature = llm_config.get('temperature', 0.7)

    # Build initial chains
    chains = _build_cluster_chains(clusters, formula_map, mean_ic, max_chain_len=5)
    if not chains:
        return [], {}

    # Initialize rankic dict with original factors' IC
    rankic = {}
    for fname, ic_val in mean_ic.items():
        formula = formula_map.get(fname, fname)
        rankic[formula] = float(ic_val)

    generated_factors = []
    returns_col = train_df['returns']

    print(f"\n  LLM Mining: {len(chains)} clusters, {n_iters} iterations")
    print(f"  Model: {model}")

    for iteration in range(1, n_iters + 1):
        new_factors_this_iter = 0
        for cid, chain in chains.items():
            if not chain:
                continue

            # LLM generates a new factor
            new_factor = _llm_generate_factor(
                api_key=api_key,
                base_url=base_url,
                model=model,
                chain_formulas=chain[:5],  # Send top-5 as context
                temperature=temperature,
            )

            if new_factor is None:
                continue

            # Evaluate the new factor
            try:
                # Compute factor series per ticker
                factor_values_list = []
                for ticker, grp in train_df.groupby('ticker'):
                    series = factor_series_fn(grp, new_factor)
                    series.name = new_factor
                    factor_values_list.append(
                        pd.Series(series.values, index=grp.index)
                    )
                factor_series = pd.concat(factor_values_list)

                # Compute Rank IC
                ric = compute_rankic(factor_series, returns_col)
            except Exception as e:
                logger.debug(f"  [Iter {iteration} {cid}] eval error: {e}")
                continue

            if np.isnan(ric):
                continue

            # Add to results
            rankic[new_factor] = ric
            generated_factors.append(new_factor)
            new_factors_this_iter += 1

            # Update chain: add new factor, keep top by |IC|
            updated = chain + [new_factor]
            kept = sorted(updated, key=lambda f: abs(rankic.get(f, 0)), reverse=True)[:max_chain_len]
            chains[cid] = kept

            print(f"  [Iter {iteration} {cid}] IC={ric:.4f} | {new_factor[:80]}")

        if new_factors_this_iter == 0:
            print(f"  [Iter {iteration}] No new factors generated (all failed or skipped)")
        else:
            print(f"  [Iter {iteration}] Generated {new_factors_this_iter} new factors")

    # Deduplicate
    generated_factors = list(dict.fromkeys(generated_factors))
    print(f"\n  LLM Mining complete: {len(generated_factors)} unique factors generated")

    return generated_factors, rankic


def _compute_llm_factor_exposures(
    df: pd.DataFrame,
    factor_expressions: List[str],
) -> pd.DataFrame:
    """
    Compute factor exposures for LLM-generated expressions.

    Evaluates each expression per-ticker using factor_series_fn.

    Args:
        df: AlphaFAMA-format DataFrame (MultiIndex date, ticker)
        factor_expressions: List of factor expression strings

    Returns:
        DataFrame with MultiIndex (date, ticker) and one column per factor
    """
    ex_list = []
    for ticker, grp in df.groupby('ticker'):
        ticker_factors = {}
        for expr in factor_expressions:
            try:
                series = factor_series_fn(grp, expr)
                ticker_factors[expr] = series.values
            except Exception:
                ticker_factors[expr] = np.nan
        ex_list.append(
            pd.DataFrame(ticker_factors, index=grp.index)
        )

    exposures = pd.concat(ex_list)
    return exposures


# ═══════════════════════════════════════════════════════════════════════
# Main baseline runner
# ═══════════════════════════════════════════════════════════════════════

def run_alphafama_baseline(
    config_path: str = "config/config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    context_days: int = 30,
    output_dir: Optional[str] = None,
    use_llm: bool = True,
    llm_iters: int = 10,
    forward_period: Optional[int] = None,
) -> Dict:
    """
    Run AlphaFAMA baseline using the main project's DataLoader.

    When use_llm=True, also runs the FAMA LLM alpha-mining pipeline:
      1. Cluster Alpha101 factors by IC profiles
      2. For each cluster, LLM generates new fused factor expressions
      3. Evaluate LLM factors by Rank IC, maintain experience chains
      4. Merge LLM-generated + original factors for portfolio construction

    Args:
        config_path: Path to the main project config file.
        start_date: Data start date (YYYY-MM-DD).
        end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        context_days: Context window for factor calculation.
        output_dir: Directory for saving results.
        use_llm: Whether to run LLM alpha-mining (default True).
        llm_iters: Number of LLM mining iterations (default 10).
        forward_period: Forward return horizon (trading days) for IC evaluation.
            Defaults to config['evolution']['forward_period'] (10). Must match the
            other baselines so AlphaFAMA's Rank-IC is comparable.

    Returns:
        Dict of performance metrics with keys:
            annual_return, sharpe_ratio, max_drawdown, information_ratio,
            mean_rank_ic, icir, top_ic_factors, n_factors,
            used_llm, llm_model, n_llm_factors
    """
    print("=" * 60)
    print("  AlphaFAMA Baseline — A-Share (via Main DataLoader)")
    if use_llm:
        print("  + FAMA LLM Alpha-Mining Pipeline")
    print("=" * 60)

    # ── Step 1: Load data via main DataLoader ──────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )

    print(f"  Loaded: {len(price_data['close'].index)} trading days x "
          f"{len(price_data['close'].columns)} stocks")

    # ── Resolve forward_period (align with other baselines) ───────────
    if forward_period is None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _cfg = yaml.safe_load(f)
            forward_period = int(
                _cfg.get('evolution', {}).get('forward_period', 10)
            )
        except Exception:
            forward_period = 10
    print(f"  Forward period (IC horizon): {forward_period}d")

    # ── Step 2: Convert to AlphaFAMA format ────────────────────────────
    print("\n[Step 2] Converting data to AlphaFAMA format...")
    af_df = convert_price_data_to_alphafama(
        price_data,
        forward_period=forward_period,
    )
    print(f"  Converted: {len(af_df)} rows, MultiIndex (date, ticker)")

    # ── Step 3: Train/Test split ───────────────────────────────────────
    print("\n[Step 3] Splitting into train/test...")
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')

    train_df, test_df = split_alphafama_data(
        af_df,
        train_end_date=train_end,
        test_start_date=test_start,
        context_days=context_days,
    )

    # ── Step 4: Generate Alpha101 factors ──────────────────────────────
    print("\n[Step 4] Generating Alpha101 factors...")
    train_exposures, train_returns = _compute_factors(train_df)
    test_exposures, test_returns = _compute_factors(test_df)

    n_factors = len(train_exposures.columns)
    print(f"  Generated {n_factors} factors")

    # ── Step 5: Compute Rank-IC on training data ───────────────────────
    print("\n[Step 5] Computing Rank-IC on training data...")
    train_ic = compute_ic_matrix(train_exposures, train_returns)
    mean_train_ic = train_ic.mean()
    avg_ic = mean_train_ic.mean()
    ic_std = mean_train_ic.std()
    icir = avg_ic / ic_std if ic_std > 0 else 0.0

    print(f"  Mean Rank-IC (train): {avg_ic:.4f}, ICIR: {icir:.4f}")

    # Also compute test_ic for evaluation in Step 7
    test_ic = compute_ic_matrix(test_exposures, test_returns)
    avg_test_ic_early = test_ic.mean().mean()
    print(f"  Mean Rank-IC (test, Alpha101): {avg_test_ic_early:.4f}")

    # ── Step 5b: LLM Alpha-Mining ──────────────────────────────────────
    used_llm = False
    llm_model = None
    n_llm_factors = 0
    llm_factor_expressions = []
    llm_rankic = {}

    if use_llm:
        print("\n[Step 5b] LLM Alpha-Mining (FAMA framework)...")
        llm_config = _read_llm_config(config_path)

        if not llm_config.get('api_key') or not llm_config.get('model'):
            print("  WARNING: LLM config not found or incomplete, skipping LLM mining")
            print("  (Falling back to Alpha101-only factors)")
        else:
            llm_model = llm_config['model']
            print(f"  LLM model: {llm_model}")

            # Cluster factors by IC profiles
            n_clusters = min(8, n_factors)
            clusters = _cluster_factors(train_ic, n_clusters=n_clusters)
            print(f"  Clustered {len(clusters)} factors into {n_clusters} clusters")

            # Run LLM mining
            try:
                llm_factor_expressions, llm_rankic = _run_llm_mining(
                    train_df=train_df,
                    clusters=clusters,
                    formula_map=FORMULA_MAP,
                    mean_ic=mean_train_ic,
                    llm_config=llm_config,
                    n_iters=llm_iters,
                    max_chain_len=15,
                )
                n_llm_factors = len(llm_factor_expressions)
                used_llm = n_llm_factors > 0

                if n_llm_factors > 0:
                    print(f"\n  LLM generated {n_llm_factors} new factors")
                    # Show top LLM factors by |IC|
                    llm_sorted = sorted(
                        llm_rankic.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:5]
                    for expr, ic_val in llm_sorted:
                        print(f"    IC={ic_val:.4f} | {expr[:80]}")
                else:
                    print("  LLM mining produced no usable factors")

            except Exception as e:
                print(f"  LLM mining failed: {e}")
                import traceback
                traceback.print_exc()

    # ── Step 6: Merge LLM factors with original factors ────────────────
    if used_llm and n_llm_factors > 0:
        print(f"\n[Step 6] Computing LLM factor exposures on train + test data...")
        try:
            # ── Test side ──
            llm_test_exposures = _compute_llm_factor_exposures(
                test_df, llm_factor_expressions
            )
            llm_test_ic = compute_ic_matrix(llm_test_exposures, test_returns)

            # Merge test exposures: original Alpha101 + LLM-generated
            test_exposures_merged = pd.concat([test_exposures, llm_test_exposures], axis=1)
            # Remove duplicate columns (in case LLM generated an expression identical to an Alpha101)
            test_exposures_merged = test_exposures_merged.loc[:, ~test_exposures_merged.columns.duplicated()]

            # Also merge test_ic for portfolio simulation
            for col in llm_test_ic.columns:
                if col not in test_ic.columns:
                    test_ic[col] = llm_test_ic[col]

            # ── Train side ──
            # Compute LLM factor exposures and IC on training data so that
            # train_ic has columns for LLM factors. Without this, the later
            # train_ic[top_factor_names] lookup (in _simulate_portfolio_from_ic)
            # raises KeyError when top_factor_names includes LLM expressions.
            llm_train_exposures = _compute_llm_factor_exposures(
                train_df, llm_factor_expressions
            )
            llm_train_ic = compute_ic_matrix(llm_train_exposures, train_returns)

            for col in llm_train_ic.columns:
                if col not in train_ic.columns:
                    train_ic[col] = llm_train_ic[col]

            # Merge IC: combine original mean_train_ic with LLM rankic
            # For LLM factors, use their rankic as the IC estimate
            for expr in llm_factor_expressions:
                if expr not in mean_train_ic.index:
                    mean_train_ic[expr] = llm_rankic.get(expr, 0.0)

            n_total = len(test_exposures_merged.columns)
            print(f"  Merged: {n_factors} Alpha101 + {n_llm_factors} LLM = {n_total} total factors")
            print(f"  train_ic columns: {len(train_ic.columns)}, test_ic columns: {len(test_ic.columns)}")
            test_exposures = test_exposures_merged

        except Exception as e:
            print(f"  WARNING: Failed to compute LLM factor exposures: {e}")
            print("  Falling back to Alpha101-only factors")

    # ── Step 7: Evaluate on test data ──────────────────────────────────
    print("\n[Step 7] Evaluating on test data...")
    mean_test_ic = test_ic.mean()
    avg_test_ic = mean_test_ic.mean()

    # Filter to logical test period (exclude context window)
    test_start_ts = pd.Timestamp(test_start)
    logical_test_ic = test_ic[test_ic.index >= test_start_ts]
    if len(logical_test_ic) > 0:
        logical_avg_ic = logical_test_ic.mean().mean()
    else:
        logical_avg_ic = avg_test_ic

    print(f"  Mean Rank-IC (test): {avg_test_ic:.4f}")
    if len(logical_test_ic) > 0:
        print(f"  Mean Rank-IC (test, no context): {logical_avg_ic:.4f}")

    # ── Step 8: Top IC factors ─────────────────────────────────────────
    top_n = min(10, len(mean_train_ic))
    top_factors = mean_train_ic.abs().nlargest(top_n)
    top_factors_dict = {k: float(v) for k, v in top_factors.items()}

    print(f"\n  Top-{top_n} factors by |IC| (train):")
    for f, ic_val in top_factors_dict.items():
        label = f if len(f) <= 60 else f[:57] + "..."
        print(f"    {label}: {ic_val:.4f}")

    # ── Step 9: Simulate portfolio performance (unified BacktestEngine) ─
    print("\n[Step 9] Simulating portfolio performance (unified BacktestEngine)...")

    # Build prices DataFrame for BacktestEngine (close price, date x stock)
    prices = price_data.get('close')
    if prices is None:
        raise ValueError("Missing 'close' price data for backtest")

    simulated_metrics = _simulate_portfolio_from_ic(
        train_ic=train_ic,
        test_ic=test_ic,
        test_exposures=test_exposures,
        test_returns=test_returns,
        top_n_factors=top_factors,
        test_start_date=test_start,
        prices=prices,
        holding_period=1,  # Daily rebalance for fairest comparison
    )

    # ── Step 10: Compile results ───────────────────────────────────────
    total_factors = n_factors + n_llm_factors
    results = {
        'method': 'AlphaFAMA' + ('+LLM' if used_llm else ''),
        'n_factors': total_factors,
        'n_alpha101_factors': n_factors,
        'n_llm_factors': n_llm_factors,
        'used_llm': used_llm,
        'llm_model': llm_model,
        'llm_iters': llm_iters if used_llm else 0,
        'mean_rank_ic_train': float(avg_ic),
        'icir': float(icir),
        'mean_rank_ic_test': float(logical_avg_ic),
        'top_ic_factors': top_factors_dict,
        'annual_return': simulated_metrics.get('annual_return', 0.0),
        'sharpe_ratio': simulated_metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': simulated_metrics.get('max_drawdown', 0.0),
        'information_ratio': simulated_metrics.get('information_ratio', 0.0),
        'calmar_ratio': simulated_metrics.get('calmar_ratio', 0.0),
        'win_rate': simulated_metrics.get('win_rate', 0.0),
        'avg_turnover': simulated_metrics.get('avg_turnover', 0.0),
        'train_end': train_end,
        'test_start': test_start,
    }

    # ── Step 11: Save results ──────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, 'alphafama_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  AlphaFAMA Baseline Complete")
    print("=" * 60)

    return results


def _compute_factors(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Alpha101 factor exposures for each ticker.

    Args:
        df: AlphaFAMA-format DataFrame with MultiIndex (date, ticker).

    Returns:
        Tuple of (exposures_df, returns_df) with MultiIndex (date, ticker).
    """
    ex_list, ret_list = [], []
    for ticker, grp in df.groupby("ticker"):
        alphas = AlphaFactory.all_alphas(grp)
        ex_list.append(
            pd.DataFrame(alphas, index=grp.index).assign(ticker=ticker)
        )
        # IC TARGET must be the forward-period return so AlphaFAMA's Rank-IC is
        # comparable to the other baselines. We keep the daily `returns` column
        # intact for the Alpha101 feature inputs and use `forward_return` here.
        # Rename to 'returns' so compute_ic_matrix (which reads ['returns']) works.
        ret_list.append(
            grp[["forward_return"]]
            .rename(columns={"forward_return": "returns"})
            .assign(ticker=ticker)
        )

    exposures = pd.concat(ex_list)
    returns = pd.concat(ret_list)
    return exposures, returns


def _simulate_portfolio_from_ic(
    train_ic: pd.DataFrame,
    test_ic: pd.DataFrame,
    test_exposures: pd.DataFrame,
    test_returns: pd.DataFrame,
    top_n_factors: pd.Series,
    test_start_date: str,
    prices: pd.DataFrame,
    holding_period: int = 1,
) -> Dict:
    """
    Simulate an IC-weighted portfolio using the unified BacktestEngine.

    Strategy:
    1. Select top-N factors by training |IC|
    2. At each rebalance date, compute weighted score as sum(|IC| * normalized_exposure)
    3. Go long top-50 stocks by score, equal-weight
    4. Use BacktestEngine for consistent metrics

    Args:
        train_ic: IC matrix on training data (dates x factors)
        test_ic: IC matrix on test data (dates x factors)
        test_exposures: Factor exposures on test data, MultiIndex (date, ticker) x factors
        test_returns: Returns on test data, MultiIndex (date, ticker) x ['returns']
        top_n_factors: Top factors by |IC| (Series, index=factor name, value=IC)
        test_start_date: Test start date (to filter out context window)
        prices: Close price DataFrame (date x stock) for BacktestEngine
        holding_period: Rebalance frequency (1=daily, 5=weekly, 20=monthly)

    Returns:
        Dict with metrics from BacktestEngine.
    """
    from backtest.engine import BacktestEngine

    # Get top factor names — filter to those actually present in train_ic
    # (LLM-generated factors may appear in top_n_factors but could be missing
    # from train_ic if the merge step was skipped or partially failed)
    top_factor_names = list(top_n_factors.index[:min(10, len(top_n_factors))])
    available_in_train_ic = [f for f in top_factor_names if f in train_ic.columns]
    if len(available_in_train_ic) < len(top_factor_names):
        missing = set(top_factor_names) - set(available_in_train_ic)
        logger.warning(
            f"_simulate_portfolio_from_ic: {len(missing)} top factors missing "
            f"from train_ic, using {len(available_in_train_ic)}/{len(top_factor_names)}"
        )
    top_factor_names = available_in_train_ic
    if not top_factor_names:
        logger.warning("_simulate_portfolio_from_ic: no factors available in train_ic")
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }

    factor_weights = train_ic[top_factor_names].abs().mean()
    factor_weights = factor_weights / factor_weights.sum()

    # Filter test exposures to logical test period (no context)
    test_start_ts = pd.Timestamp(test_start_date)
    dates_in_range = test_exposures.index.get_level_values('date')
    test_exposures_filtered = test_exposures[dates_in_range >= test_start_ts]

    if test_exposures_filtered.empty:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }

    unique_dates = test_exposures_filtered.index.get_level_values('date').unique().sort_values()

    # Build portfolios DataFrame: each row = one date, values = position weights
    portfolio_rows = []
    portfolio_dates = []

    # Rebalance at holding_period intervals
    rebalance_indices = list(range(0, len(unique_dates), holding_period))

    for idx_pos, i in enumerate(rebalance_indices):
        rebal_date = unique_dates[i]

        try:
            exp = test_exposures_filtered.xs(rebal_date, level='date')
        except KeyError:
            continue

        # Compute composite score = weighted sum of normalized factor exposures
        score = pd.Series(0.0, index=exp.index)
        for f in top_factor_names:
            if f in exp.columns:
                f_vals = exp[f].dropna()
                if len(f_vals) > 1:
                    f_norm = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-10)
                    score.loc[f_norm.index] += factor_weights.get(f, 0.0) * f_norm

        # Select top-50 stocks and equal-weight
        top_stocks = score.nlargest(min(50, len(score)))
        if len(top_stocks) == 0:
            # No stocks selected: emit a zero-weight row (BacktestEngine handles this)
            continue

        w = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        portfolio_rows.append(w)
        portfolio_dates.append(rebal_date)

    if not portfolio_rows:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'win_rate': 0, 'information_ratio': 0, 'avg_turnover': 0,
            'n_trading_days': 0,
        }

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
    engine = BacktestEngine(
        commission=0.0003,
        slippage=0.001,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    metrics = engine.run(portfolios, prices_aligned)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AlphaFAMA baseline with main DataLoader')
    parser.add_argument('--config', default='config/config.yaml', help='Path to main config')
    parser.add_argument('--start', default=None, help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='Data end date (YYYY-MM-DD)')
    parser.add_argument('--universe', default=None, help='Stock universe (hs300, zz500, all_a)')
    parser.add_argument('--train-end', default=None, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', default=None, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--context-days', type=int, default=30, help='Context window days')
    parser.add_argument('--output-dir', default='experiments/alphafama', help='Output directory')
    parser.add_argument('--use-llm', action='store_true', default=True,
                        help='Enable LLM alpha-mining (default: enabled)')
    parser.add_argument('--no-llm', action='store_false', dest='use_llm',
                        help='Disable LLM alpha-mining')
    parser.add_argument('--llm-iters', type=int, default=10,
                        help='Number of LLM mining iterations (default: 10)')
    parser.add_argument('--forward-period', type=int, default=None,
                        help='Forward return horizon (trading days) for IC evaluation. '
                             'Defaults to config evolution.forward_period (10).')

    args = parser.parse_args()

    results = run_alphafama_baseline(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        universe=args.universe,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        context_days=args.context_days,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        llm_iters=args.llm_iters,
        forward_period=args.forward_period,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Method:           {results['method']}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic_train']:.4f}")
    print(f"  ICIR:             {results['icir']:.4f}")
    print(f"  Factors:          {results['n_factors']} (Alpha101: {results.get('n_alpha101_factors', 0)}, LLM: {results.get('n_llm_factors', 0)})")
    print(f"  Used LLM:         {results['used_llm']}")
