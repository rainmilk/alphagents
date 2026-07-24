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
    python baselines/run_alphafama.py --train-start 2020-01-01 --test-end 2024-12-31 --universe hs300
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
from concurrent.futures import ProcessPoolExecutor

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
3. You should return the new factor as a JSON object with a single key "factor".
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
{{"factor": "rank(correlation(open, volume, 10) / rank(open))"}}
"""

SYSTEM_PROMPT = (
    "You are an alpha-mining agent implementing the FAMA (FActor Mining Agent) framework. "
    "Generate a new, interpretable financial alpha factor expression as JSON. "
    "Respond with a JSON object of the form {\"factor\": \"<expression>\"}. "
    "Do NOT wrap it in an array. "
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


def _safe_message_text(message) -> str:
    """Extract usable text from an OpenAI chat-completion message.

    Reasoning models (DeepSeek-R1 / QwQ / o1, …) frequently emit
    chain-of-thought in a separate ``reasoning_content`` field while leaving
    ``content`` empty or ``None``. Reading only ``content`` would silently
    yield '' and poison downstream parsing. Mirror of
    run_alphaagent._extract_message_text.

    Strategy: prefer ``content``; fall back to ``reasoning_content`` (direct
    attribute or under ``model_extra`` on pydantic-based SDKs). Returns ''
    when nothing usable.
    """
    content = getattr(message, "content", None)
    if content:
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        rc = extra.get("reasoning_content")
        if rc:
            return rc
    return ""


def _robust_json_load(raw: str):
    """Parse a JSON array/object out of arbitrary LLM output.

    Models routinely wrap the payload in Markdown fences, chain-of-thought
    (``<think>...</think>``), or explanatory prose. We therefore:
      1. drop any ``<think>`` CoT block,
      2. try a direct ``json.loads``,
      3. otherwise locate the *first* opening bracket and match it to its
         *balanced* closing bracket (so stray brackets in the prose can't
         swallow the real payload — the previous greedy regex did exactly that).
    Returns the parsed Python object, or ``None`` if nothing parseable.
    """
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    for open_c, close_c in (("[", "]"), ("{", "}")):
        # Try EVERY opening bracket as a candidate start. Reasoning/prose often
        # contains brackets (e.g. "[alpha1, alpha2] as reference"); the first one
        # may not be the payload, so we keep scanning until one parses.
        start = cleaned.find(open_c)
        while start != -1:
            depth = 0
            matched = -1
            for i in range(start, len(cleaned)):
                if cleaned[i] == open_c:
                    depth += 1
                elif cleaned[i] == close_c:
                    depth -= 1
                    if depth == 0:
                        matched = i
                        break
            if matched != -1:
                cand = cleaned[start:matched + 1]
                try:
                    return json.loads(cand)
                except (json.JSONDecodeError, ValueError):
                    pass
            start = cleaned.find(open_c, start + 1)
    return None


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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # We now ask the model for a JSON *object* {"factor": "..."} (not a
    # top-level array). That lets us use the OpenAI "json_object" response mode
    # on endpoints that support it, which is the single most reliable way to
    # avoid non-JSON output. Endpoints that reject response_format are handled
    # by the self-heal below (fall back to prompt-only + _robust_json_load,
    # which still copes with fences / CoT / prose-wrapped payloads).
    response = None
    thinking = True  # attempt enable_thinking=False; self-heal if unsupported
    json_mode = True  # attempt response_format=json_object; self-heal if unsupported
    idx = 0
    while idx < 3:
        call_kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        if thinking:
            # Self-hosted reasoning endpoints (e.g. bailian/deepseek-v4-flash)
            # may write CoT into content; disable thinking to get clean output.
            call_kwargs["extra_body"] = {"enable_thinking": False}
        if json_mode:
            call_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**call_kwargs)
        except Exception as e:
            # self-heal: drop the unsupported knob, one at a time, and retry
            if json_mode:
                json_mode = False
                logger.warning(
                    "  Provider rejected response_format (json_object); retrying "
                    "without it. (%s)", e,
                )
                continue
            if thinking:
                thinking = False
                logger.warning(
                    "  Provider rejected extra_body (enable_thinking); retrying "
                    "without thinking control. (%s)", e,
                )
                continue
            logger.warning(f"LLM API call failed: {e}")
            return None

        raw = _safe_message_text(response.choices[0].message).strip()

        # Robust JSON extraction: handles Markdown fences, CoT (<think>...</think>),
        # and prose wrapped around the JSON (e.g. "Here is your factor: [...]").
        # Uses bracket-balanced substring extraction instead of a greedy regex so
        # that explanatory text containing stray brackets can't break parsing.
        gen = _robust_json_load(raw)

        if isinstance(gen, dict):
            # Preferred JSON-object mode: {"factor": "..."} (or common aliases).
            factor = (
                gen.get("factor") or gen.get("expression")
                or gen.get("alpha") or gen.get("formula")
            )
            if isinstance(factor, str) and factor.strip():
                return factor.strip()
            # Fall back to the first string value under any other key.
            for v in gen.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            logger.warning(
                "  LLM returned a JSON object without a usable factor string; "
                "retrying. raw=%r", raw,
            )
        elif isinstance(gen, list) and len(gen) > 0:
            # Legacy/array fallback (in case json_object mode was unavailable).
            for item in gen:
                if isinstance(item, str) and item.strip():
                    return item.strip()
            logger.warning(
                "  LLM returned a JSON array but no usable string factor; "
                "retrying. raw=%r", raw,
            )
        elif isinstance(gen, str) and gen.strip():
            return gen.strip()
        else:
            logger.warning(
                "  LLM factor generation returned non-JSON output (attempt %d); "
                "retrying with strict JSON-only instruction. raw=%r",
                idx + 1, raw,
            )
            messages = messages + [
                {"role": "user", "content":
                 "Your previous reply was not valid JSON. Respond with ONLY a JSON "
                 "object of the form {\"factor\": \"<expression>\"} (no explanations, "
                 "no markdown, no code fences). Nothing else."},
            ]
            idx += 1

    if response is not None:
        logger.warning(
            "LLM returned non-JSON output, skipping. raw=%r",
            _safe_message_text(response.choices[0].message),
        )
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


def _seed_clusters_from_formula_map(
    formula_map: Dict[str, str],
    n_clusters: int = 8,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build seed clusters + placeholder mean-IC from the base Alpha101 formula
    library.

    Used when Alpha101 factor *exposures* are disabled (``use_alpha101=False``):
    instead of clustering the (empty) Alpha101 IC matrix, we seed the LLM
    mining chains directly from the 101 handcrafted *formulas* in
    ``FORMULA_MAP``. This gives the LLM real expressions to evolve without
    ever computing the 101 factor exposures (the single most expensive step in
    the original pipeline).

    Args:
        formula_map: Dict mapping alphaXXX -> formula expression string.
        n_clusters: Number of seed clusters to spread the formulas across
            (round-robin, deterministic).

    Returns:
        Tuple of (clusters, mean_ic):
          - clusters: Series mapping alphaXXX name -> cluster label.
          - mean_ic:  Series mapping alphaXXX name -> 0.0 placeholder. The
            seeds' |IC| only affects initial chain ordering (irrelevant when
            uniform); LLM-generated factors get their real Rank-IC later.
    """
    names = list(formula_map.keys())
    if not names:
        return pd.Series(dtype=int), pd.Series(dtype=float)

    k = max(1, min(n_clusters, len(names)))
    labels = [i % k for i in range(len(names))]
    clusters = pd.Series(labels, index=names, name="cluster")
    mean_ic = pd.Series(0.0, index=names, name="ic")
    return clusters, mean_ic


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
        Tuple of (list of LLM-generated factor expressions, dict of factor ->
        Rank IC, dict of factor -> per-ticker train exposure Series). The third
        element lets Step 6 reuse the train exposures instead of recomputing.
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
    # Cache of per-ticker train exposures for each successfully mined factor.
    # Step 6 reuses these (they are computed on the same train_df) instead of
    # re-evaluating every LLM expression a second time on the training data —
    # that second pass was the bulk of Step 6's cost.
    train_exposure_cache = {}
    returns_col = train_df['returns']

    print(f"\n  LLM Mining: {len(chains)} clusters, {n_iters} iterations")
    print(f"  Model: {model}")

    for iteration in range(1, n_iters + 1):
        new_factors_this_iter = 0
        for cid, chain in chains.items():
            if not chain:
                continue

            # LLM generates a new factor
            # Per-iteration guard: a single malformed/garbage LLM payload
            # (e.g. a JSON object keyed by something other than "factor") must
            # NOT abort the whole mining run — only this iteration is skipped.
            # This is what kept Alpha101-OFF runs from collapsing to zero
            # factors (and thus a zeroed backtest) on one bad generation.
            try:
                new_factor = _llm_generate_factor(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    chain_formulas=chain[:5],  # Send top-5 as context
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(
                    "  [Iter %d %s] LLM generation raised %s; skipping this "
                    "iteration.", iteration, cid, e,
                )
                continue

            if new_factor is None:
                continue

            # Evaluate the new factor
            try:
                # Compute factor series per ticker
                factor_values_list = []
                for ticker, grp in train_df.groupby('ticker'):
                    series = factor_series_fn(grp, new_factor)
                    # NB: name the per-ticker piece with the expression. The
                    # cached copy (train_exposure_cache[new_factor]) is later
                    # pd.concat-ed along axis=1 in Step 6; without a name here
                    # the columns collapse to positional [0,1,2] and stop
                    # matching the expression-string factor names used in
                    # train_ic / top_factor_names → "factors missing from
                    # train_ic". compute_rankic tolerates a 1-col named
                    # DataFrame (it indexes .iloc[:,0]), so this is safe.
                    factor_values_list.append(
                        pd.Series(series.values, index=grp.index, name=new_factor)
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
            train_exposure_cache[new_factor] = factor_series
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

    return generated_factors, rankic, train_exposure_cache


def _eval_ticker_factors(ticker_group, factor_expressions):
    """Evaluate every expression for a single ticker (serial, GIL-bound safe).

    Mirrors one iteration of the original serial loop.
    """
    ticker, grp = ticker_group
    ticker_factors = {}
    for expr in factor_expressions:
        try:
            series = factor_series_fn(grp, expr)
            ticker_factors[expr] = series.values
        except Exception:
            ticker_factors[expr] = np.nan
    return pd.DataFrame(ticker_factors, index=grp.index)


def _compute_llm_factor_exposures(
    df: pd.DataFrame,
    factor_expressions: List[str],
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Compute factor exposures for LLM-generated expressions.

    Evaluates each expression per-ticker using factor_series_fn. This is a
    **serial** implementation — deliberately so. Two parallelism attempts were
    benchmarked and both *lost* to serial for this workload:

      * ProcessPoolExecutor: ~6 s of Windows ``spawn`` re-import cost (it drags
        in the heavy AlphaFactory chain) + pickling every ticker slice to the
        workers, which dominates the small LLM-factor compute.
      * ThreadPoolExecutor: the per-call cost is GIL-bound Python inside
        ``factor_series_fn`` (expression compile + namespace build), so threads
        serialize *and* add overhead.

    The real speedups come from (a) the compile cache in ``factor_series_fn``
    and (b) Step 6 reusing the train exposures already computed during Step 5
    mining (see ``_run_llm_mining``'s returned cache) instead of re-evaluating
    them. ``n_jobs`` is accepted for API symmetry with ``_compute_factors`` but
    is currently unused.

    Args:
        df: AlphaFAMA-format DataFrame (MultiIndex date, ticker)
        factor_expressions: List of factor expression strings
        n_jobs: accepted for API symmetry; currently unused (see note above).

    Returns:
        DataFrame with MultiIndex (date, ticker) and one column per factor
    """
    if not factor_expressions:
        return pd.DataFrame(index=df.index)

    ex_list = [
        _eval_ticker_factors(g, factor_expressions)
        for g in df.groupby("ticker")
    ]
    return pd.concat(ex_list) if ex_list else pd.DataFrame(index=df.index)


# ═══════════════════════════════════════════════════════════════════════
# Main baseline runner
# ═══════════════════════════════════════════════════════════════════════

def run_alphafama_baseline(
    config_path: str = "config/config.yaml",
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    universe: Optional[str] = None,
    context_days: int = 30,
    output_dir: Optional[str] = None,
    use_llm: bool = True,
    use_alpha101: bool = False,
    llm_iters: int = 5,
    forward_period: Optional[int] = None,
    holding_period: Optional[int] = None,
    n_jobs: Optional[int] = None,
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
        train_start_date: Data start date (YYYY-MM-DD).
        test_end_date: Data end date (YYYY-MM-DD).
        universe: Stock universe (hs300, zz500, all_a).
        train_end_date: Last training date (YYYY-MM-DD).
        test_start_date: First test date (YYYY-MM-DD).
        context_days: Context window for factor calculation.
        output_dir: Directory for saving results.
        use_llm: Whether to run LLM alpha-mining (default True).
        use_alpha101: Whether to pre-compute the 101 handcrafted Alpha101
            factor exposures as the starting factor pool (default False).
            When False, Step 4 skips the (expensive) all-Alpha101 computation
            and the factor pool is LLM-generated only — seeded from the base
            Alpha101 *formula library* (FORMULA_MAP) instead of from the
            pre-computed factor ICs. Set True to reproduce the original
            AlphaFAMA behaviour (Alpha101 factors + LLM fusion).
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
    print(f"  Alpha101 factor library: {'ENABLED' if use_alpha101 else 'DISABLED (LLM-only seeds)'}")
    print("=" * 60)

    # ── Step 1: Load data via main DataLoader ──────────────────────────
    print("\n[Step 1] Loading data via main DataLoader...")
    loader = DataLoader(config_path=config_path)
    train_start = train_start_date or loader.data_config.get('train_start_date', '2023-01-01')
    train_end = train_end_date or loader.data_config.get('train_end_date', '2023-12-31')
    test_start = test_start_date or loader.data_config.get('test_start_date', '2024-01-01')
    test_end = test_end_date or loader.data_config.get('test_end_date', '2025-06-30')

    bundle = loader.load_data(universe=universe, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    train_price, train_fund, train_ind = bundle.train
    test_price, test_fund, test_ind = bundle.test
    price_data = bundle.full[0]  # FULL span (kept for backtest prices + logging)

    # ── Guard: clamp requested windows to the cached data range ─────────
    # If config's default train_end/test_end (or CLI args) fall outside the
    # locally cached archive, the slice comes back EMPTY and the run later
    # degenerates or crashes ("报一堆错误" with no explicit args). Detect that
    # and fall back to an 80/20 split of the ACTUAL available data so the
    # baseline always produces results instead of failing on out-of-range
    # config dates.
    _avail = loader.price_data['close'].index
    if len(train_price['close']) < 5 or len(test_price['close']) < 5:
        _af, _al = _avail[0], _avail[-1]
        print(
            f"  WARNING: requested window [{train_start}..{train_end} / "
            f"{test_start}..{test_end}] has insufficient data in the local "
            f"cache (available {_af.date()}..{_al.date()}). Falling back to an "
            f"80/20 train/test split of the available data."
        )
        _n = len(_avail)
        _split = max(1, int(_n * 0.8))
        train_start = str(_avail[0].date())
        train_end = str(_avail[_split - 1].date())
        test_start = str(_avail[_split].date())
        test_end = str(_avail[-1].date())
        bundle = loader.load_data(
            universe=universe, train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
        )
        train_price, train_fund, train_ind = bundle.train
        test_price, test_fund, test_ind = bundle.test
        price_data = bundle.full[0]

    print(f"  Loaded: {len(price_data['close'].index)} trading days x "
          f"{len(price_data['close'].columns)} stocks")

    # ── Resolve forward_period & holding_period from config ───────────
    if forward_period is None or holding_period is None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _cfg = yaml.safe_load(f)
            if forward_period is None:
                forward_period = int(
                    _cfg.get('evolution', {}).get('forward_period', 10)
                )
            if holding_period is None:
                holding_period = int(
                    _cfg.get('backtest', {}).get('trading', {}).get('holding_period', 1)
                )
        except Exception:
            if forward_period is None:
                forward_period = 10
            if holding_period is None:
                holding_period = 1
    print(f"  Forward period (IC horizon): {forward_period}d")
    print(f"  Holding period (rebalance):  {holding_period}d")

    # ── Step 2: Convert train/test slices to AlphaFAMA format ──────────
    # The train/test split is now produced centrally by loader.load_data
    # (bundle.train / bundle.test); no manual date-masking here.
    print("\n[Step 2] Converting data to AlphaFAMA format...")
    train_df = convert_price_data_to_alphafama(
        train_price,
        forward_period=forward_period,
    )
    test_df = convert_price_data_to_alphafama(
        test_price,
        forward_period=forward_period,
    )
    print(f"  Converted: train={len(train_df)} rows, test={len(test_df)} rows, "
          f"MultiIndex (date, ticker)")

    # ── Step 3: Compute forward-return IC target (both branches) ────────
    # The Rank-IC target is the forward-period return, derived from the
    # `forward_return` column produced in Step 2. It depends only on price —
    # NOT on the Alpha101 library — so we compute it ONCE here, unconditionally,
    # for both the Alpha101-on and Alpha101-off paths. This gives a single
    # source of truth for the IC target (previously it was computed two
    # different ways: inline in `_compute_factors` on the on-path, and via a
    # separate `_compute_returns` on the off-path — a drift hazard, and the
    # on-path version also leaked a spurious `ticker` column into the frame).
    print("\n[Step 3] Computing forward-return IC target (train + test)...")
    train_returns = _compute_returns(train_df)
    test_returns = _compute_returns(test_df)
    print(f"  IC target rows: train={len(train_returns)}, test={len(test_returns)}")

    # ── Step 4: Generate Alpha101 factors ──────────────────────────────
    if use_alpha101:
        print("\n[Step 4] Generating Alpha101 factors...")
        train_exposures = _compute_factors(train_df, n_jobs=n_jobs)
        test_exposures = _compute_factors(test_df, n_jobs=n_jobs)

        n_factors = len(train_exposures.columns)
        print(f"  Generated {n_factors} factors")
    else:
        print("\n[Step 4] Alpha101 factors DISABLED — skipping all-Alpha101 "
              "computation (LLM will seed from the base formula library).")
        # Empty exposure frames keep the downstream merge / IC logic uniform.
        # train_returns / test_returns are already computed in Step 3 (they
        # depend only on price, so LLM-generated factors still have a valid
        # IC target).
        train_exposures = pd.DataFrame(index=train_df.index)
        test_exposures = pd.DataFrame(index=test_df.index)
        n_factors = 0

    # ── Step 5: Compute Rank-IC on training data ───────────────────────
    print("\n[Step 5] Computing Rank-IC on training data...")
    if n_factors > 0:
        train_ic = compute_ic_matrix(train_exposures, train_returns, n_jobs=n_jobs)
        test_ic = compute_ic_matrix(test_exposures, test_returns, n_jobs=n_jobs)
    else:
        # No Alpha101 factors → no IC matrix to compute. Use empty frames
        # whose index is a *single-level date axis* (matching
        # compute_ic_matrix's contract), NOT the (date, ticker) MultiIndex of
        # train_df/test_df. Otherwise:
        #   - Step 7's `test_ic.index >= test_start_ts` filter raises TypeError
        #     (you can't compare a MultiIndex to a scalar Timestamp); and
        #   - Step 6's `test_ic[col] = llm_test_ic[col]` mis-aligns (different
        #     index types → all-NaN merge).
        # This keeps the OFF path byte-for-byte consistent with the ON path.
        _train_dates = train_df.index.get_level_values("date").unique()
        _test_dates = test_df.index.get_level_values("date").unique()
        train_ic = pd.DataFrame(
            index=pd.Index(_train_dates, name="date")
        )
        test_ic = pd.DataFrame(
            index=pd.Index(_test_dates, name="date")
        )

    mean_train_ic = train_ic.mean()
    avg_ic = mean_train_ic.mean() if n_factors > 0 else 0.0
    ic_std = mean_train_ic.std() if n_factors > 0 else 0.0
    icir = avg_ic / ic_std if ic_std > 0 else 0.0

    print(f"  Mean Rank-IC (train): {avg_ic:.4f}, ICIR: {icir:.4f}")

    # Also compute test_ic for evaluation in Step 7
    avg_test_ic_early = test_ic.mean().mean() if n_factors > 0 else 0.0
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

            # Build seed clusters for the LLM chains.
            # - Alpha101 ON:  cluster the pre-computed Alpha101 IC matrix and
            #   use those factors (and their ICs) as seeds.
            # - Alpha101 OFF: there is no IC matrix, so seed directly from the
            #   base 101 *formula library* (FORMULA_MAP). This still gives the
            #   LLM real expressions to evolve without ever computing the 101
            #   factor exposures — and is what makes the Alpha101-OFF mode
            #   produce factors at all.
            if use_alpha101 and n_factors > 0:
                n_clusters = min(8, n_factors)
                clusters = _cluster_factors(train_ic, n_clusters=n_clusters)
                mining_mean_ic = mean_train_ic
                print(f"  Clustered {len(clusters)} factors into {n_clusters} clusters")
            else:
                _seed_n = min(8, len(FORMULA_MAP))
                clusters, mining_mean_ic = _seed_clusters_from_formula_map(
                    FORMULA_MAP, n_clusters=_seed_n
                )
                print(f"  Alpha101 OFF → seeded {len(clusters)} LLM chains from "
                      f"{len(FORMULA_MAP)} base formulas")

            # Run LLM mining
            try:
                llm_factor_expressions, llm_rankic, llm_train_exposure_cache = _run_llm_mining(
                    train_df=train_df,
                    clusters=clusters,
                    formula_map=FORMULA_MAP,
                    mean_ic=mining_mean_ic,
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
                test_df, llm_factor_expressions, n_jobs=n_jobs
            )
            llm_test_ic = compute_ic_matrix(llm_test_exposures, test_returns, n_jobs=n_jobs)

            # Merge test exposures: original Alpha101 + LLM-generated
            test_exposures_merged = pd.concat([test_exposures, llm_test_exposures], axis=1)
            # Remove duplicate columns (in case LLM generated an expression identical to an Alpha101)
            test_exposures_merged = test_exposures_merged.loc[:, ~test_exposures_merged.columns.duplicated()]

            # Also merge test_ic for portfolio simulation
            for col in llm_test_ic.columns:
                if col not in test_ic.columns:
                    test_ic[col] = llm_test_ic[col]

            # ── Train side ──
            # Reuse the per-ticker train exposures already computed during LLM
            # mining (_run_llm_mining returns them keyed by expression). The
            # mining ran on this same train_df, so re-evaluating here would be
            # a redundant second pass over every LLM expression — previously
            # the dominant cost of Step 6. We only fall back to recomputation
            # for any expression missing from the cache (defensive; should not
            # happen since every generated factor was evaluated in mining).
            _cached = [e for e in llm_factor_expressions
                       if e in llm_train_exposure_cache]
            _missing = [e for e in llm_factor_expressions
                        if e not in llm_train_exposure_cache]
            _train_parts = [llm_train_exposure_cache[e] for e in _cached]
            if _missing:
                print(f"  [Step 6] {len(_missing)} LLM factors not in mining "
                      f"cache; recomputing on train.")
                _train_parts.append(
                    _compute_llm_factor_exposures(train_df, _missing, n_jobs=n_jobs)
                )
            llm_train_exposures = (
                pd.concat(_train_parts, axis=1) if _train_parts
                else pd.DataFrame(index=train_df.index)
            )
            llm_train_ic = compute_ic_matrix(llm_train_exposures, train_returns, n_jobs=n_jobs)

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

    # ── Guard: no factors at all ───────────────────────────────────────
    # Happens only when Alpha101 is disabled AND LLM produced nothing (no
    # api_key / mining failed / zero usable factors). The downstream sim
    # returns zeroed metrics, but flag it loudly so the empty run is obvious.
    if test_exposures.shape[1] == 0:
        print("\n  WARNING: zero factors produced "
              "(Alpha101 disabled and no LLM factors). "
              "Backtest will report zeroed metrics.")

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

    # ── Test ICIR (out-of-sample) ──
    # Mirrors the train ICIR formula: mean(per-factor mean IC over test dates)
    # / std(per-factor mean IC over test dates). Computed over the logical test
    # period (excluding the context window) for a fair OOS estimate.
    _ic_for_icir = logical_test_ic if len(logical_test_ic) > 0 else test_ic
    per_factor_test_ic = _ic_for_icir.mean()   # Series: per-factor mean IC across test dates
    test_icir = per_factor_test_ic.mean() / per_factor_test_ic.std() if per_factor_test_ic.std() > 0 else 0.0
    print(f"  ICIR (test): {test_icir:.4f}")

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

    run_dir = None
    method_name = "alphafama"
    if output_dir:
        _u = universe or loader.data_config.get('universe', {}).get('index', 'hs300')
        _s = train_start_date or loader.data_config.get('train_start_date', 'na')
        _e = test_end_date or loader.data_config.get('test_end_date', 'na')
        _fp = forward_period if forward_period is not None else 10
        _hp = holding_period if holding_period is not None else 1
        param_dir = f"{_u}_{_s}_{_e}_forward-{_fp}_holding-{_hp}"
        run_dir = os.path.join(os.path.dirname(output_dir), param_dir, method_name)
        os.makedirs(run_dir, exist_ok=True)

    simulated_metrics = _simulate_portfolio_from_ic(
        train_ic=train_ic,
        test_ic=test_ic,
        test_exposures=test_exposures,
        test_returns=test_returns,
        top_n_factors=top_factors,
        test_start_date=test_start,
        prices=prices,
        holding_period=holding_period,
        save_dir=run_dir,
    )

    # ── Step 10: Compile results ───────────────────────────────────────
    total_factors = n_factors + n_llm_factors
    results = {
        'method': 'AlphaFAMA' + ('+LLM' if used_llm else ''),
        'n_factors': total_factors,
        'n_alpha101_factors': n_factors,
        'n_llm_factors': n_llm_factors,
        'use_alpha101': use_alpha101,
        'used_llm': used_llm,
        'llm_model': llm_model,
        'llm_iters': llm_iters if used_llm else 0,
        'mean_rank_ic_train': float(avg_ic),
        'icir': float(test_icir),            # TEST (out-of-sample) ICIR — reported metric
        'icir_train': float(icir),          # preserved in-sample ICIR (for overfit diagnosis)
        'icir_test': float(test_icir),      # explicit OOS ICIR (parallel to AlphaAgent schema)
        'mean_rank_ic_test': float(logical_avg_ic),
        'top_ic_factors': top_factors_dict,
        'annual_return': simulated_metrics.get('annual_return', 0.0),
        'total_return': simulated_metrics.get('total_return', 0.0),
        'sharpe_ratio': simulated_metrics.get('sharpe_ratio', 0.0),
        'max_drawdown': simulated_metrics.get('max_drawdown', 0.0),
        'information_ratio': simulated_metrics.get('information_ratio', 0.0),
        'calmar_ratio': simulated_metrics.get('calmar_ratio', 0.0),
        'win_rate': simulated_metrics.get('win_rate', 0.0),
        'avg_turnover': simulated_metrics.get('avg_turnover', 0.0),
        'train_end': train_end,
        'test_start': test_start,
        'train_start': train_start,
        'test_end': test_end,
        'forward_period': forward_period,
        'holding_period': holding_period,
    }
    # Mirrors run_alphaagent.final_result.json: keep the chosen factors (here, the
    # top-|IC| factors with their ICs) alongside the metrics so the run is
    # reproducible from disk, not just from terminal scrollback.
    results['factors'] = top_factors_dict

    # ── Step 11: Save results ──────────────────────────────────────────
    if output_dir:
        # Unified artifact name across baselines (was alphafama_results.json).
        result_path = os.path.join(run_dir, 'final_result.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {result_path}")

    print("\n" + "=" * 60)
    print("  AlphaFAMA Baseline Complete")
    print("=" * 60)

    return results


# Worker lives in a separate module so Windows 'spawn' children import only
# that lightweight module (not this __main__ script) — avoids the classic
# re-execution / freeze_support recursion. See baselines/alphafama_parallel.py.
from baselines.alphafama_parallel import _compute_factors_chunk


def _compute_factors(
    df: pd.DataFrame,
    n_jobs: int = None,
) -> pd.DataFrame:
    """
    Compute Alpha101 factor *exposures* for each ticker.

    Each ticker's factor computation is fully independent, so we distribute
    tickers across worker processes for a near-linear speedup while keeping the
    numerical results *bit-identical* to the serial loop (the per-ticker
    computation is unchanged; only the scheduling differs). Falls back to the
    serial loop when parallelism is disabled or unavailable.

    The forward-return IC *target* (``returns``) is NOT produced here — it is
    computed once in Step 3 via ``_compute_returns`` so both the Alpha101-on
    and Alpha101-off paths share a single source of truth.

    Args:
        df: AlphaFAMA-format DataFrame with MultiIndex (date, ticker).
        n_jobs: number of worker processes. ``None`` → auto (``cpu_count()-1``,
            min 1). Set to 1 to force serial. Override via env
            ``ALPHAFAMA_N_JOBS``.

    Returns:
        exposures_df with MultiIndex (date, ticker).
    """
    groups = list(df.groupby("ticker"))

    # Resolve worker count (env override → auto → serial).
    if n_jobs is None:
        n_jobs = int(os.environ.get(
            "ALPHAFAMA_N_JOBS", max(1, (os.cpu_count() or 1) - 1)
        ))

    # Serial path: explicitly disabled, single ticker, or nothing to split.
    if n_jobs <= 1 or len(groups) <= 1:
        ex_list = _compute_factors_chunk(groups)
    else:
        n_workers = max(1, min(n_jobs, len(groups)))
        # Contiguous chunks preserve ticker order across the final concat, so
        # the output is identical to the serial loop's row order.
        chunk_size = max(1, (len(groups) + n_workers - 1) // n_workers)
        chunks = [groups[i:i + chunk_size]
                  for i in range(0, len(groups), chunk_size)]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_compute_factors_chunk, chunks))
            ex_list = []
            for cel in results:
                ex_list.extend(cel)
        except Exception as e:  # pragma: no cover - pool fallback safety
            logger.warning(
                f"_compute_factors: parallel path failed ({e}); "
                f"falling back to serial."
            )
            ex_list = _compute_factors_chunk(groups)

    exposures = pd.concat(ex_list)
    return exposures


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the forward-return IC target as a (date, ticker) frame.

    This is the SINGLE source of truth for the Rank-IC target, computed once
    in Step 3 and shared by both the Alpha101-on and Alpha101-off paths. It
    selects the ``forward_return`` column produced by Step 2's conversion and
    renames it to ``returns`` (the name ``compute_ic_matrix`` reads).
    """
    ret_list = []
    for _ticker, grp in df.groupby("ticker"):
        ret_list.append(
            grp[["forward_return"]]
            .rename(columns={"forward_return": "returns"})
        )
    return pd.concat(ret_list)


def _simulate_portfolio_from_ic(
    train_ic: pd.DataFrame,
    test_ic: pd.DataFrame,
    test_exposures: pd.DataFrame,
    test_returns: pd.DataFrame,
    top_n_factors: pd.Series,
    test_start_date: str,
    prices: pd.DataFrame,
    holding_period: int = 1,
    save_dir: Optional[str] = None,
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

    # Sign-aware weights: a factor whose mean IC is negative predicts that
    # high factor values → LOW future returns, so it must contribute to the
    # composite *score* with a flipped sign. Using abs() here (old code) silently
    # inverted every negative-IC factor, randomly flipping long/short and pushing
    # the realized annual return to extreme +/- tails.
    mean_ic = train_ic[top_factor_names].mean()            # signed mean IC per factor
    factor_weights = np.sign(mean_ic) * mean_ic.abs()      # signed magnitude
    factor_weights = factor_weights / factor_weights.abs().sum()

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

    # Build a portfolio for EVERY test date (daily resolution). The actual
    # rebalance cadence is governed solely by `holding_period` in the
    # BacktestEngine (it re-samples every `holding_period`-th portfolio row),
    # exactly like all other baselines. This keeps `holding_period` as the
    # single frequency knob and removes the old double-skip (FAMA pre-sampling
    # at `holding_period` AND the engine re-sampling at `holding_period` again).
    rebalance_indices = list(range(len(unique_dates)))

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
                    # Winsorize to [1%, 99%] before z-scoring: Alpha101 factors are
                    # extremely heavy-tailed, and a single outlier can inflate the
                    # cross-sectional std, letting one stock dominate the ranking and
                    # spiking turnover. Clipping first makes the score robust to tails.
                    lo, hi = f_vals.quantile(0.01), f_vals.quantile(0.99)
                    f_w = f_vals.clip(lower=lo, upper=hi)
                    f_norm = (f_w - f_w.mean()) / (f_w.std() + 1e-10)
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
        commission=0.001,
        slippage=0.0,
        risk_free_rate=0.0,
        holding_period=holding_period,
    )
    metrics = engine.run(portfolios, prices_aligned, save_dir=save_dir)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AlphaFAMA baseline with main DataLoader')
    parser.add_argument('--config', default='config/config.yaml', help='Path to main config')
    parser.add_argument('--train-start', default=None, help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', default=None, help='Data end date (YYYY-MM-DD)')
    parser.add_argument('--universe', default=None, help='Stock universe (hs300, zz500, all_a)')
    parser.add_argument('--train-end', default=None, help='Train end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', default=None, help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--context-days', type=int, default=30, help='Context window days')
    parser.add_argument('--output-dir', default='experiments/alphafama', help='Output directory')
    parser.add_argument('--use-llm', action='store_true', default=True,
                        help='Enable LLM alpha-mining (default: enabled)')
    parser.add_argument('--no-llm', action='store_false', dest='use_llm',
                        help='Disable LLM alpha-mining')
    parser.add_argument('--use-alpha101', action='store_true', default=True,
                        help='Enable the 101 handcrafted Alpha101 factor '
                             'exposures as the starting pool (default: OFF — '
                             'LLM seeds from the base formula library only)')
    parser.add_argument('--no-alpha101', action='store_false', dest='use_alpha101',
                        help='Disable Alpha101 factors (default)')
    parser.add_argument('--llm-iters', type=int, default=5,
                        help='Number of LLM mining iterations (default: 5)')
    parser.add_argument('--forward-period', type=int, default=None,
                        help='Forward return horizon (trading days) for IC evaluation. '
                             'Defaults to config evolution.forward_period (10).')
    parser.add_argument('--holding-period', type=int, default=None,
                        help='Rebalance frequency in days (1=daily, 5=weekly, 20=monthly). '
                             'Defaults to config backtest.trading.holding_period (1).')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Worker processes for Alpha101 factor computation '
                             '(step 4). None=auto (cpu_count-1). 1=serial. '
                             'Env override: ALPHAFAMA_N_JOBS.')

    args = parser.parse_args()

    results = run_alphafama_baseline(
        config_path=args.config,
        universe=args.universe,
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        context_days=args.context_days,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        use_alpha101=args.use_alpha101,
        llm_iters=args.llm_iters,
        forward_period=args.forward_period,
        holding_period=args.holding_period,
        n_jobs=args.n_jobs,
    )

    print("\n" + "=" * 60)
    print("  Final Results (BacktestEngine)")
    print("=" * 60)
    print(f"  Method:           {results['method']}")
    print(f"  Total Return:     {results['total_return']:.4f}")
    print(f"  Annual Return:    {results['annual_return']:.4f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.4f}")
    print(f"  Information Ratio:{results['information_ratio']:.4f}")
    print(f"  Win Rate:         {results['win_rate']:.4f}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.4f}")
    print(f"  Mean Rank-IC:     {results['mean_rank_ic_train']:.4f}")
    print(f"  ICIR (test):      {results['icir']:.4f}")
    print(f"  Factors:          {results['n_factors']} (Alpha101: {results.get('n_alpha101_factors', 0)}, LLM: {results.get('n_llm_factors', 0)})")
    print(f"  Used LLM:         {results['used_llm']}")
