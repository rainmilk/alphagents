# src/deepseek_agent.py

import json
from pathlib import Path
import os
import re

import pandas as pd
from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import openai

from .config import settings
from .data_loader import load_factor_df
from .utils.generate_rankics import factor_series_fn, compute_rankic

# Wire your OpenAI key
openai.api_key = settings.deepseek.api_key
openai.api_base = settings.deepseek.api_base  # <— point at DeepSeek’s endpoint

# Lazy LangChain OpenAI client with self-healing "thinking" control.
# Reasoning endpoints (DeepSeek-R1 style) may emit CoT in
# additional_kwargs['reasoning_content'] and leave .content empty; passing
# enable_thinking=False suppresses that. If the endpoint rejects the kwarg we
# rebuild the client without it on first failure (see one_iteration).
_llm = None
_llm_thinking = True


def _get_llm():
    global _llm, _llm_thinking
    if _llm is None:
        _llm = ChatOpenAI(
            model_name=settings.deepseek.model,
            temperature=settings.deepseek.temperature,
            openai_api_key=settings.deepseek.api_key,
            openai_api_base=settings.deepseek.api_base,
            max_tokens=4096,
            model_kwargs={"enable_thinking": False} if _llm_thinking else {},
        )
    return _llm

# Prompt template for factor generation
prompt_template = """
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


def get_latest_iteration(prefix: str, out_dir: Path) -> int:
    """Scan out_dir for files named prefix_iter{n}.json and return max n or 0."""
    matches = [p.stem for p in out_dir.iterdir() if p.stem.startswith(prefix)]
    iters = [int(name.split("_iter")[-1]) for name in matches if "_iter" in name]
    return max(iters) if iters else 0


def _safe_content(resp) -> str:
    """Extract text from a LangChain AIMessage, falling back to reasoning_content
    for thinking models (stashed in additional_kwargs['reasoning_content'])."""
    content = getattr(resp, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    ak = getattr(resp, "additional_kwargs", None) or {}
    rc = ak.get("reasoning_content") or ak.get("reasoningContent")
    if isinstance(rc, str) and rc.strip():
        return rc
    return ""


def _parse_alpha_json(raw: str):
    """Best-effort parse of a factor spec (JSON array or object) from LLM output.
    Returns the parsed list/dict, or None if nothing recoverable."""
    if not raw:
        return None
    fence = re.match(r"```[a-zA-Z]*\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    candidate = fence.group(1).strip() if fence else raw
    try:
        obj = json.loads(candidate)
        if isinstance(obj, (list, dict)):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", candidate, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, (list, dict)):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def one_iteration(
    df: pd.DataFrame,
    chains: dict,
    rankic: dict,
    out_dir: Path,
    iteration: int
):
    new_chains = {}
    new_rankic = rankic.copy()

    for cid, chain in chains.items():
        prompt = prompt_template.format(
            function_definition=settings.pipeline.function_definition,
            css=json.dumps(chain, ensure_ascii=False),
            chain=" -> ".join(chain)
        )

        messages = [
            SystemMessage(content=(
                "You are an alpha-mining agent implementing the FActor Mining Agent (FAMA) framework. "
                "Generate a new, interpretable financial alpha factor expression as JSON."
            )),
            HumanMessage(content=prompt),
        ]

        # Retry loop (up to 2 attempts) on LLM-call failure or non-JSON output.
        # Mirrors the robustness added to run_alphaagent / run_alphafama: a
        # thinking endpoint that emits CoT is handled by enable_thinking=False
        # (self-healed on rejection) plus reasoning_content fallback in _safe_content.
        factor = None
        for attempt in range(2):
            try:
                resp = _get_llm().invoke(messages)
            except Exception as e:
                global _llm, _llm_thinking
                if _llm_thinking:
                    _llm_thinking = False
                    _llm = None  # force rebuild without thinking control
                    print(f"[Cluster {cid}] ⚠️ enable_thinking rejected; retrying without it. ({e})")
                    continue
                print(f"[Cluster {cid}] ❌ LLM call failed: {e}")
                new_chains[cid] = chain
                break
            raw = _safe_content(resp)
            print(f"[Cluster {cid}] LLM output (raw):\n{raw}")
            parsed = _parse_alpha_json(raw)
            if isinstance(parsed, (list, dict)):
                if isinstance(parsed, list):
                    factor = parsed[0] if parsed else None
                else:
                    # off-contract object: pull the first string value as the expr
                    strs = [v for v in parsed.values() if isinstance(v, str)]
                    factor = strs[0] if strs else None
                if isinstance(factor, str) and factor.strip():
                    break
            print(f"[Cluster {cid}] ❌ JSON parse error (attempt {attempt + 1}), retrying. raw={raw!r}")
            messages = messages + [
                HumanMessage(content=(
                    "Your previous reply was not valid JSON. Respond with ONLY a JSON "
                    "array containing one factor string (no explanations, no markdown, "
                    "no code fences). Nothing else."
                )),
            ]

        if not isinstance(factor, str) or not factor.strip():
            new_chains[cid] = chain
            continue

        try:
            series = factor_series_fn(df, factor)
            ric = compute_rankic(series, df["return_"])
        except Exception as e:
            print(f"[Cluster {cid}] ❌ eval error: {e}")
            new_chains[cid] = chain
            continue

        new_rankic[factor] = ric
        updated = chain + [factor]
        # keep top-15 by |RankIC|
        kept = sorted(updated, key=lambda f: abs(new_rankic.get(f, 0)))[-15:]
        new_chains[cid] = kept
        print(f"[Cluster {cid}] ✅ {factor} → RIC={ric:.4f}")

    # Save iteration JSONs
    chains_file = out_dir / f"experience_chains_iter{iteration}.json"
    rankic_file = out_dir / f"time_series_rankic_iter{iteration}.json"
    chains_file.write_text(json.dumps(new_chains, indent=2, ensure_ascii=False), encoding='utf-8')
    rankic_file.write_text(json.dumps(new_rankic, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ Saved iteration {iteration}: {chains_file}, {rankic_file}")


def run(num_iters: int = 50):
    """Driver loop to run multiple DeepSeek alpha-mining iterations."""
    df = load_factor_df()
    out_dir = Path(settings.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(num_iters):
        last = get_latest_iteration("experience_chains", out_dir)
        if last == 0:
            chain_file = out_dir / "clustered_formulas.json"
            rankic_file = out_dir / "formula_rankic.json"
        else:
            chain_file = out_dir / f"experience_chains_iter{last}.json"
            rankic_file = out_dir / f"time_series_rankic_iter{last}.json"

        chains = json.loads(chain_file.read_text(encoding='utf-8'))
        rankic = json.loads(rankic_file.read_text(encoding='utf-8'))

        one_iteration(df, chains, rankic, out_dir, last + 1)
