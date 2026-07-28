# -*- coding: utf-8 -*-
"""
Multi-Agent Debate Evaluator Module

This module implements a multi-agent debate system for factor evaluation.
Five expert agents (Momentum, Value, Quality, Volatility, Growth)
engage in structured debate to evaluate factors.

Real LLM calls via DeepSeek (OpenAI-compatible API).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
import os
import time
import json
from datetime import datetime

from config import config_path

warnings.filterwarnings('ignore')

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Role-specific system prompts
# IMPORTANT design notes (applied to every role below):
#   * The factor's backtest IC / Sharpe are REAL evidence (when provided). They
#     MUST anchor the 0-10 score — narrative alone is not enough.
#   * Each expert evaluates strictly from ITS OWN domain lens. Be the honest
#     advocate: disagree with peers when theory supports it; never rubber-stamp.
#   * If a factor is largely OUTSIDE your domain, say so in 'concerns' and cap
#     your score at <=6 (you cannot credibly endorse an out-of-domain factor).
ROLE_PROMPTS = {
    "Momentum Expert": (
        "You are a Momentum Expert specializing in quantitative stock selection. "
        "You evaluate factors based on: trend persistence, volume confirmation, "
        "reactivity to price signals, and robustness across market regimes. "
        "A good momentum factor has high Rank-IC, low turnover, and works in "
        "trending markets. "
        "SCORING ANCHORS (use the provided IC/Sharpe when available): "
        "IC>=0.03 -> 8-10; IC 0.015-0.03 -> 6-8; IC 0.005-0.015 -> 4-6; "
        "IC<0.005 or negative Sharpe -> <=5 regardless of narrative. "
        "If no IC/Sharpe is provided, state that your score is narrative-only "
        "and discount it. Be the honest advocate for momentum — disagree with "
        "peers when their argument ignores price/volume dynamics. "
        "If this factor is largely OUTSIDE your domain, state that in 'concerns' "
        "and cap your score at <=6."
    ),
    "Value Expert": (
        "You are a Value Expert specializing in fundamental stock selection. "
        "You evaluate factors based on: fundamental signal quality, mean-reversion properties, "
        "avoidance of value traps, and long-term predictive power. "
        "A good value factor captures mispricing, has economic intuition, "
        "and is robust across business cycles. "
        "SCORING ANCHORS (use the provided IC/Sharpe when available): "
        "IC>=0.03 -> 8-10; IC 0.015-0.03 -> 6-8; IC 0.005-0.015 -> 4-6; "
        "IC<0.005 or negative Sharpe -> <=5 regardless of narrative. "
        "If no IC/Sharpe is provided, state that your score is narrative-only "
        "and discount it. Flag value traps (cheap for a reason) explicitly in "
        "'concerns'. If this factor is largely OUTSIDE your domain, state that "
        "in 'concerns' and cap your score at <=6."
    ),
    "Quality Expert": (
        "You are a Quality Expert specializing in corporate fundamental analysis. "
        "You evaluate factors based on: earnings quality, profitability sustainability, "
        "balance sheet strength, and management efficiency. "
        "A good quality factor identifies companies with durable competitive advantages, "
        "low bankruptcy risk, and consistent cash flow generation. "
        "SCORING ANCHORS (use the provided IC/Sharpe when available): "
        "IC>=0.03 -> 8-10; IC 0.015-0.03 -> 6-8; IC 0.005-0.015 -> 4-6; "
        "IC<0.005 or negative Sharpe -> <=5 regardless of narrative. "
        "If no IC/Sharpe is provided, state that your score is narrative-only "
        "and discount it. Note: a pure valuation field (e.g. pb, pe) is VALUE, "
        "not Quality — reserve high scores for genuine quality signals "
        "(profitability, margins). If this factor is largely OUTSIDE your domain, "
        "state that in 'concerns' and cap your score at <=6."
    ),
    "Volatility Expert": (
        "You are a Volatility Expert specializing in risk-adjusted factor evaluation. "
        "You evaluate factors based on: downside protection, drawdown control, "
        "tail-risk behavior, and risk-adjusted return potential. "
        "A good low-vol factor has stable IC, low max-drawdown contribution, "
        "and positive performance in stress periods. "
        "SCORING ANCHORS (use the provided IC/Sharpe when available): "
        "Sharpe>=1.0 with IC>=0.02 -> 8-10; Sharpe 0.5-1.0 -> 6-8; "
        "Sharpe<0.3 or high drawdown -> <=5 regardless of narrative. "
        "If no IC/Sharpe is provided, state that your score is narrative-only "
        "and discount it. Judge factors on RISK-ADJUSTED behaviour, not raw "
        "return. If this factor is largely OUTSIDE your domain, state that in "
        "'concerns' and cap your score at <=6."
    ),
    "Growth Expert": (
        "You are a Growth Expert specializing in forward-looking signal evaluation. "
        "You evaluate factors based on: earnings growth sustainability, analyst revision trends, "
        "R&D efficiency, and scalability of business model. "
        "A good growth factor identifies companies with accelerating fundamentals, "
        "high ROCIC, and expanding market share. "
        "SCORING ANCHORS (use the provided IC/Sharpe when available): "
        "IC>=0.03 -> 8-10; IC 0.015-0.03 -> 6-8; IC 0.005-0.015 -> 4-6; "
        "IC<0.005 or negative Sharpe -> <=5 regardless of narrative. "
        "If no IC/Sharpe is provided, state that your score is narrative-only "
        "and discount it. Distinguish genuine growth (EPS / earnings "
        "acceleration, e.g. ts_pct_change(eps, N)) from mere valuation — a static pb/pe is NOT growth. "
        "If this factor is largely OUTSIDE your domain, state that in 'concerns' "
        "and cap your score at <=6."
    ),
}

# Canonical factor families — kept consistent with methods/evolve.py's
# _ALL_FAMILIES so a factor's family label means the same thing in BOTH the
# evolution stage (diversity-aware selection) and the debate stage (diversity-
# aware Chair selection). Drift here would silently let Value/Quality factors
# sneak past the diversity gate.
_DEBATE_FAMILIES = [
    "Momentum", "Mean-reversion", "Value/Quality",
    "Volatility", "Liquidity", "Growth", "Other",
]


def _infer_family(expr: str) -> str:
    """Infer a factor's PRIMARY family from its expression.

    EXACT mirror of methods/evolve.py:_infer_family (same keyword priority) so a
    factor carries the same family label in BOTH the evolution stage (diversity-
    aware selection) and the debate stage (diversity-aware Chair selection).
    Drift here would let Value/Quality factors slip past the diversity gate.
    """
    e = (expr or "").lower()
    if any(k in e for k in ("volume", "amt", "turnover", "illiquid")):
        return "Liquidity"
    if any(k in e for k in ("ts_std", "ts_max", "ts_min", "ts_skew", "ts_kurt")):
        return "Volatility"
    if any(k in e for k in ("ts_corr", "ts_cov", "ts_pct_change",
                            "ts_delta", "ts_rank", "returns", "momentum")):
        return "Momentum"
    if any(k in e for k in ("ts_zscore", "reversal", "mean_reversion")):
        return "Mean-reversion"
    if any(k in e for k in ("growth", "eps")):
        return "Growth"
    if any(k in e for k in ("pe", "pb", "ps", "roe", "roa",
                            "market_cap", "value", "margin", "debt", "quality")):
        return "Value/Quality"
    return "Other"


# Structured output schema for independent evaluation
EVAL_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 10},
            "reasoning": {"type": "string"},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "strengths": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "reasoning", "concerns", "strengths"],
    },
}

# Structured output schema for consensus
CONSENSUS_SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "final_score": {"type": "number", "minimum": 0, "maximum": 10},
            "key_insights": {"type": "array", "items": {"type": "string"}},
            "recommendation": {"type": "string", "enum": ["APPROVE", "REJECT", "CONDITIONAL"]},
            "consensus_summary": {"type": "string"},
        },
        "required": ["final_score", "key_insights", "recommendation", "consensus_summary"],
    },
}
class AgentRole(Enum):
    """Expert agent roles."""
    MOMENTUM = "Momentum Expert"
    VALUE = "Value Expert"
    QUALITY = "Quality Expert"
    VOLATILITY = "Volatility Expert"
    GROWTH = "Growth Expert"


@dataclass
class FactorProposal:
    """Factor proposal for debate."""
    expression: str
    description: str
    rationale: Optional[str] = None
    ic: Optional[float] = None  # optional IC from backtest
    sharpe: Optional[float] = None  # optional Sharpe from backtest
    family: Optional[str] = None  # optional factor family label (e.g. 'Momentum',
                                   # 'Value/Quality'). When omitted, _infer_family()
                                   # derives it from the expression so the Chair can
                                   # still enforce family-diverse selection.


@dataclass
class AgentOpinion:
    """Opinion from an expert agent."""
    agent_role: AgentRole
    factor_proposal: FactorProposal
    score: float  # 0-10
    reasoning: str
    concerns: List[str]
    strengths: List[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """Record of a debate round."""
    round_id: int
    opinions: List[AgentOpinion]
    consensus_score: float
    disagreements: List[str]
    summary: str = ""


@dataclass
class DebateResult:
    """Final result of debate."""
    factor_proposal: FactorProposal
    final_score: float
    agent_scores: Dict[str, float]
    key_insights: List[str]
    recommendation: str
    consensus_summary: str = ""
    all_rounds: List[DebateRound] = field(default_factory=list)


class DebateEvaluator:
    """
    Multi-agent debate evaluator for factors.

    Uses real LLM calls (DeepSeek via OpenAI-compatible API) to simulate
    structured debate among 5 expert agents.
    """

    def __init__(
        self,
        llm_model: str = "deepseek-chat",
        n_agents: int = 5,
        n_rounds: int = 3,
        random_seed: int = 42,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        # --- Chair Agent (arbitration / synthesis) ---
        chair_model: str = "",
        chair_api_key: str = "",
        chair_base_url: str = "",
        chair_temperature: float = 0.2,
        # --- Execution control ---
        parallel_eval: bool = True,
        request_timeout: float = 120.0,
        max_retries: int = 3,
        # --- Chair synthesis prompt bounding (prevents timeouts on huge pools) ---
        synthesis_top_n: int = 50,
        chair_max_tokens: int = 8192,
        chair_max_per_family: int = 3,
    ):
        """
        Initialize debate evaluator.

        Args:
            llm_model: LLM model for the 5 expert agents (cheap, e.g. DeepSeek)
            n_agents: Number of agents (max 5, one per role)
            n_rounds: Number of debate rounds
            random_seed: Random seed (used only as fallback)
            api_key: API key for expert agents. Falls back to DEEPSEEK_API_KEY env var.
            base_url: API base URL for expert agents
            chair_model: LLM model for the Chair Agent (arbitration / cross-factor synthesis).
                         If empty, falls back to llm_model (same as experts).
            chair_api_key: API key for Chair Agent. Falls back to api_key if empty.
            chair_base_url: API base URL for Chair Agent. Falls back to base_url if empty.
            chair_temperature: Temperature for Chair Agent calls (default 0.2, lower = more deterministic)
            parallel_eval: If True, 5 agents evaluate the same factor in parallel (ThreadPoolExecutor).
                           Set False for serial mode (easier stack traces when debugging).
        """
        self.llm_model = llm_model
        self.n_agents = min(n_agents, 5)
        self.n_rounds = n_rounds
        self.rng = np.random.RandomState(random_seed)
        self.use_llm = OPENAI_AVAILABLE

        # Chair Agent model settings (separate from expert agents)
        self.chair_model = chair_model or llm_model
        self.chair_api_key = chair_api_key or api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.chair_base_url = chair_base_url or base_url
        self.chair_temperature = chair_temperature
        self.parallel_eval = parallel_eval
        self.request_timeout = request_timeout
        self.max_retries = max(1, int(max_retries))
        self.synthesis_top_n = max(1, int(synthesis_top_n))
        self.chair_max_tokens = max(1, int(chair_max_tokens))
        self.chair_max_per_family = max(2, int(chair_max_per_family))

        # Initialize OpenAI client for EXPERT agents
        self.client = None
        if OPENAI_AVAILABLE:
            key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            if key:
                try:
                    self.client = OpenAI(api_key=key, base_url=base_url)
                    # Test connection with a minimal call
                    test_resp = self.client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
                    # Validate response shape
                    if not hasattr(test_resp, "choices"):
                        raise RuntimeError(
                            f"API test: unexpected response type {type(test_resp).__name__}. "
                            f"Response: {str(test_resp)[:200]}"
                        )
                    self.use_llm = True
                    print(f"  [debate] Expert LLM client initialized: {llm_model} @ {base_url}")
                    print(f"  [debate] Connection test passed.")
                except Exception as e:
                    print(f"  [debate] Warning: Expert API connection test failed: {e}")
                    print(f"  [debate] Falling back to mock mode.")
                    self.client = None
                    self.use_llm = False
            else:
                print(f"  [debate] Warning: No API key for experts. Set api_key or DEEPSEEK_API_KEY.")
                print(f"  [debate] Falling back to mock mode.")
                self.use_llm = False
        else:
            print(f"  [debate] openai package not installed. Running in mock mode.")
            self.use_llm = False

        # Initialize OpenAI client for CHAIR Agent (separate, can be stronger model)
        self.chair_client = None
        self.use_chair_llm = False
        if OPENAI_AVAILABLE:
            chair_key = self.chair_api_key or os.environ.get("CHAIR_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
            if chair_key:
                try:
                    self.chair_client = OpenAI(api_key=chair_key, base_url=self.chair_base_url)
                    # Test connection
                    test_resp = self.chair_client.chat.completions.create(
                        model=self.chair_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
                    if not hasattr(test_resp, "choices"):
                        raise RuntimeError(
                            f"Chair API test: unexpected response type {type(test_resp).__name__}."
                        )
                    self.use_chair_llm = True
                    print(f"  [debate] Chair LLM client initialized: {self.chair_model} @ {self.chair_base_url}")
                except Exception as e:
                    print(f"  [debate] Warning: Chair API connection test failed: {e}")
                    # Fall back to expert client for chair if chair-specific client fails
                    if self.client:
                        print(f"  [debate] Chair falling back to expert model: {self.llm_model}")
                    self.chair_client = None
                    self.use_chair_llm = False
            elif self.client:
                # No separate chair credentials; reuse expert client for chair
                print(f"  [debate] Chair Agent reusing expert model: {self.llm_model}")
            else:
                print(f"  [debate] No Chair client available.")

        # Initialize agents
        self.agents = self._initialize_agents()

        # Accumulated results for saving (appended by evaluate())
        self.results = []  # List of (FactorProposal, DebateResult)

    def _initialize_agents(self) -> List[AgentRole]:
        """Initialize expert agents."""
        all_roles = [
            AgentRole.MOMENTUM,
            AgentRole.VALUE,
            AgentRole.QUALITY,
            AgentRole.VOLATILITY,
            AgentRole.GROWTH,
        ]
        return all_roles[:self.n_agents]

    def evaluate(self, factor_proposal: FactorProposal) -> DebateResult:
        """
        Evaluate a factor through multi-agent debate.

        Args:
            factor_proposal: Factor to evaluate

        Returns:
            DebateResult with final score, agent scores, and recommendation
        """
        expr_short = factor_proposal.expression[:60]
        print(f"\n  [debate] Evaluating: {expr_short}...")
        if factor_proposal.ic is not None:
            print(f"    IC={factor_proposal.ic:.4f}, Sharpe={factor_proposal.sharpe:.4f}")

        # Phase 1: Independent evaluation
        independent_opinions = self._independent_evaluation(factor_proposal)

        # Phase 2: Structured debate
        debate_rounds = self._structured_debate(factor_proposal, independent_opinions)

        # Phase 3: Consensus
        result = self._reach_consensus(factor_proposal, debate_rounds)

        rec = result.recommendation
        print(f"  [debate] Done. Score={result.final_score:.2f}, Recommendation={rec}")

        # Accumulate for saving
        self.results.append((factor_proposal, result))

        return result

    def save_results(self, output_dir: str = None):
        """
        Save accumulated debate results to experiments/{yyyymmdd}/debate/debate_factors_result.json.

        Called after all evaluate() calls are done. The output format is a
        per-factor summary list:
          [
            {
              "factor_expression": "...",
              "factor_description": "...",
              "debate_score": 7.5,
              "recommendation": "select",
              "key_insights": [...],
              "consensus_summary": "..."
            },
            ...
          ]

        Args:
            output_dir: Override output directory. Defaults to experiments/{yyyymmdd}/debate/
        """
        import os, json
        from datetime import datetime
        from dataclasses import asdict

        if not self.results:
            return

        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = os.path.join(date_str, "debate")
            output_dir = config_path("experiments", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "debate_factors_result.json")

        records = []
        for factor_proposal, result in self.results:
            expr = factor_proposal.expression
            desc = factor_proposal.description

            # Convert DebateResult dataclass to dict
            try:
                result_dict = asdict(result)
            except Exception:
                result_dict = {
                    'final_score': getattr(result, 'final_score', 0.0),
                    'recommendation': getattr(result, 'recommendation', 'reject'),
                    'key_insights': getattr(result, 'key_insights', []),
                    'consensus_summary': getattr(result, 'consensus_summary', ''),
                }

            records.append({
                "factor_expression": expr,
                "factor_description": desc,
                "debate_score": result_dict.get('final_score', 0.0),
                "recommendation": result_dict.get('recommendation', 'reject'),
                "key_insights": result_dict.get('key_insights', []),
                "consensus_summary": result_dict.get('consensus_summary', ''),
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  [debate] Debate results saved to {output_path} ({len(records)} factors)")

    def _call_llm(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.3, expect_json: bool = True) -> str:
        """Call EXPERT LLM and return response content."""
        if not self.client:
            raise RuntimeError("Expert LLM client not initialized. Check API key.")
        return self._call_llm_inner(
            self.client, self.llm_model, system_prompt, user_prompt,
            temperature=temperature, expect_json=expect_json,
        )

    def _call_chair_llm(self, system_prompt: str, user_prompt: str,
                         temperature: float = None, expect_json: bool = True,
                         max_tokens: int = None) -> str:
        """Call CHAIR LLM (separate model, typically stronger) and return response content.

        Falls back to expert client if chair client is unavailable.
        max_tokens defaults to self.chair_max_tokens when not explicitly given.
        """
        client = self.chair_client or self.client
        model = self.chair_model if self.chair_client else self.llm_model
        temp = temperature if temperature is not None else self.chair_temperature
        if not client:
            raise RuntimeError("Chair LLM client not initialized. Check API key.")
        return self._call_llm_inner(
            client, model, system_prompt, user_prompt,
            temperature=temp, expect_json=expect_json,
            max_tokens=max_tokens if max_tokens is not None else self.chair_max_tokens,
        )

    def _call_llm_inner(self, client, model: str, system_prompt: str,
                         user_prompt: str, temperature: float = 0.3,
                         expect_json: bool = True, max_tokens: int = None) -> str:
        """Internal LLM call helper — shared by expert and chair clients."""
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        # Cap generation length — without this, the Chair's huge cross-factor
        # JSON can balloon and push the call past request_timeout.
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    timeout=self.request_timeout, **kwargs
                )

                # Debug: inspect resp type to catch non-standard responses
                if not hasattr(resp, "choices"):
                    resp_type = type(resp).__name__
                    resp_preview = str(resp)[:200]
                    raise RuntimeError(
                        f"Unexpected LLM response type '{resp_type}'. "
                        f"Response preview: {resp_preview}"
                    )

                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty content (None).")
                return content

            except Exception as e:
                last_err = e
                import traceback
                tb = traceback.format_exc()
                print(f"  [debate] LLM call attempt {attempt + 1}/{self.max_retries} "
                      f"failed ({type(e).__name__}): {e}")
                # Exponential backoff before retry (capped at 8s) to ride out
                # transient 429/timeout bursts from the API.
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))

        # All retries exhausted — surface the last captured error.
        import traceback
        tb = traceback.format_exc() if last_err else ""
        print(f"  [debate] LLM call failed after {self.max_retries} attempts.")
        print(f"  [debate] Traceback (last frame): {tb.splitlines()[-1] if tb else 'N/A'}")
        if last_err:
            raise last_err
        raise RuntimeError("LLM call failed with no captured error.")

    def _independent_evaluation(self, factor_proposal: FactorProposal) -> List[AgentOpinion]:
        """
        Phase 1: Each agent evaluates the factor independently.

        When self.parallel_eval=True, the 5 agents run in parallel via ThreadPoolExecutor
        (all API calls are I/O-bound). Set parallel_eval=False for serial mode —
        easier stack traces and ordered print output when debugging.
        """
        ic_str = f", IC={factor_proposal.ic:.4f}" if factor_proposal.ic is not None else ""
        sharpe_str = f", Sharpe={factor_proposal.sharpe:.4f}" if factor_proposal.sharpe is not None else ""

        if self.parallel_eval and len(self.agents) > 1:
            # --- Parallel: ThreadPoolExecutor for I/O-bound API calls ---
            import concurrent.futures

            role_order = list(self.agents)  # snapshot to preserve order
            future_to_role = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                for agent in role_order:
                    future = executor.submit(
                        self._evaluate_single_agent,
                        agent, factor_proposal,
                    )
                    future_to_role[future] = agent

                # Collect results in original order
                opinions_by_role = {}
                for future in concurrent.futures.as_completed(future_to_role):
                    agent = future_to_role[future]
                    try:
                        opinion = future.result()
                        opinions_by_role[agent] = opinion
                    except Exception as e:
                        print(f"    [debate] Parallel eval failed for {agent.value}: {e}")
                        # Fallback opinion on executor error
                        opinions_by_role[agent] = AgentOpinion(
                            agent_role=agent,
                            factor_proposal=factor_proposal,
                            score=self.rng.uniform(5.0, 9.0),
                            reasoning=f"As {agent.value}, I think this factor is promising (parallel fallback).",
                            concerns=["Parallel evaluation failed", "Needs more backtesting"],
                            strengths=["N/A due to eval failure"],
                        )

            # Return in original role order
            return [opinions_by_role[agent] for agent in role_order]

        # --- Serial: simple for-loop (debug-friendly) ---
        opinions = []
        for agent in self.agents:
            opinion = self._evaluate_single_agent(agent, factor_proposal)
            opinions.append(opinion)

        return opinions

    def _evaluate_single_agent(
        self,
        agent: "AgentRole",
        factor_proposal: "FactorProposal",
    ) -> "AgentOpinion":
        """Evaluate a single factor from one agent's perspective (used by both
        serial and parallel pathways in _independent_evaluation).

        The user prompt is built PER AGENT so each expert is explicitly told its
        role (belt-and-suspenders with the system prompt) and receives the same
        grounding rules. Previously a single shared template was reused for all
        five agents, so the role name could not be injected per agent.
        """
        role_name = agent.value
        system_prompt = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["Momentum Expert"])

        ic_str = f", IC={factor_proposal.ic:.4f}" if factor_proposal.ic is not None else ""
        sharpe_str = f", Sharpe={factor_proposal.sharpe:.4f}" if factor_proposal.sharpe is not None else ""

        user_prompt = (
            f"You are the {role_name}. Evaluate the following quantitative stock "
            f"selection factor strictly from YOUR domain perspective:\n\n"
            f"Expression: {factor_proposal.expression}\n"
            f"Description: {factor_proposal.description}\n"
            f"{ic_str}{sharpe_str}\n\n"
            f"CRITICAL SCORING RULES:\n"
            f"- The IC/Sharpe above are REAL backtest results — they MUST anchor your 0-10 score. "
            f"Follow your role's scoring anchors. Do NOT score >7 for a factor with IC<0.005 "
            f"or negative Sharpe, no matter how plausible the story.\n"
            f"- If IC/Sharpe are absent, say so in 'reasoning' and treat your score as narrative-only (discount it).\n"
            f"- If this factor is largely OUTSIDE your domain, state that in 'concerns' and cap your score at <=6.\n"
            f"- 'strengths'/'concerns' must be specific to YOUR domain (not generic praise).\n\n"
            f"Provide your evaluation as a JSON object with keys: "
            f"'score' (float 0-10), 'reasoning' (string), "
            f"'concerns' (list of strings), 'strengths' (list of strings)."
        )

        if self.use_llm and self.client:
            try:
                raw = self._call_llm(system_prompt, user_prompt, temperature=0.3)
                parsed = json.loads(raw)
                score = float(parsed["score"])
                reasoning = str(parsed.get("reasoning", ""))
                concerns = [str(c) for c in parsed.get("concerns", [])]
                strengths = [str(s) for s in parsed.get("strengths", [])]
            except Exception as e:
                print(f"    [debate] LLM call failed for {role_name}: {e}. Using fallback.")
                score = self.rng.uniform(5.0, 9.0)
                reasoning = f"As {role_name}, I think this factor is promising (fallback eval)."
                concerns = ["LLM call failed", "Needs more backtesting"]
                strengths = ["N/A due to eval failure"]
        else:
            # Mock mode
            score = self.rng.uniform(5.0, 9.0)
            reasoning = f"As {role_name}, I think this factor is promising (mock eval)."
            concerns = ["Needs more backtesting", "Consider market regime"]
            strengths = ["Good baseline signal"]

        return AgentOpinion(
            agent_role=agent,
            factor_proposal=factor_proposal,
            score=score,
            reasoning=reasoning,
            concerns=concerns,
            strengths=strengths,
        )

    def _structured_debate(self, factor_proposal: FactorProposal,
                            initial_opinions: List[AgentOpinion]) -> List[DebateRound]:
        """
        Phase 2: Structured debate — agents see each other's opinions and refine.
        """
        debate_rounds = []
        previous_opinions = initial_opinions

        for round_id in range(self.n_rounds):
            opinions = []
            disagreements = []

            # Build context from previous round
            context = self._build_debate_context(previous_opinions, round_id)

            # --- Parallelize per-agent evaluation within the round. ---
            # All agents in a round read the SAME previous_opinions (the context
            # built above), so they are mutually independent and can run
            # concurrently. This is the dominant cost of the debate: previously
            # n_rounds * n_agents SERIAL calls. Parallelizing cuts it to
            # n_rounds batches, each bounded by the slowest of n_agents calls.
            if self.parallel_eval and len(self.agents) > 1:
                import concurrent.futures

                role_order = list(self.agents)  # snapshot to preserve order
                future_to_role = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                    for agent in role_order:
                        future = executor.submit(
                            self._evaluate_agent_in_round,
                            agent, factor_proposal, context, round_id,
                        )
                        future_to_role[future] = agent

                    opinions_by_role = {}
                    for future in concurrent.futures.as_completed(future_to_role):
                        agent = future_to_role[future]
                        try:
                            opinions_by_role[agent] = future.result()
                        except Exception as e:
                            print(f"    [debate] Round {round_id + 1} parallel eval failed "
                                  f"for {agent.value}: {e}")
                            # Deterministic fallback on executor error
                            opinions_by_role[agent] = self._fallback_round_opinion(
                                agent, factor_proposal, round_id
                            )

                # Return in original role order
                opinions = [opinions_by_role[agent] for agent in role_order]
            else:
                # Serial path (debug-friendly / single agent)
                opinions = []
                for agent in self.agents:
                    opinions.append(
                        self._evaluate_agent_in_round(agent, factor_proposal, context, round_id)
                    )

            # Calculate consensus and disagreements
            scores = [op.score for op in opinions]
            consensus_score = float(np.mean(scores))
            score_std = float(np.std(scores))
            if score_std > 2.0:
                disagreements.append(f"High score variance: std={score_std:.2f}")

                # --- Arbitration Intervention ---
                # When experts strongly disagree (std > 2.0), escalate to Chair Agent
                # for explicit adjudication. The Chair's ruling is injected into the
                # next round's context.
                if self.use_llm and (self.chair_client or self.client):
                    try:
                        chair_ruling = self._arbitration_intervention(
                            factor_proposal, opinions, round_id, score_std
                        )
                        if chair_ruling:
                            # Add chair's arbitration as an additional "opinion"
                            # so other agents see it in the next round
                            chair_opinion = AgentOpinion(
                                agent_role=AgentRole.MOMENTUM,  # placeholder role
                                factor_proposal=factor_proposal,
                                score=chair_ruling.get("arbitration_score", consensus_score),
                                reasoning=f"[Chair Arbitration] {chair_ruling.get('ruling', '')}",
                                concerns=chair_ruling.get("concerns_addressed", []),
                                strengths=chair_ruling.get("strengths_confirmed", []),
                            )
                            # Replace the chair_opinion's role display
                            chair_opinion.agent_role = AgentRole.MOMENTUM
                            opinions.append(chair_opinion)
                            disagreements.append(
                                f"Chair intervened: {chair_ruling.get('ruling', '')[:100]}"
                            )
                            print(f"    [debate] Chair arbitration: std={score_std:.2f}, "
                                  f"ruling_score={chair_ruling.get('arbitration_score', 'N/A')}")
                    except Exception as e:
                        print(f"    [debate] Arbitration failed: {e}")

            debate_round = DebateRound(
                round_id=round_id,
                opinions=opinions,
                consensus_score=consensus_score,
                disagreements=disagreements,
                summary=f"Round {round_id + 1}: consensus={consensus_score:.2f}, std={score_std:.2f}",
            )
            debate_rounds.append(debate_round)
            previous_opinions = opinions

        return debate_rounds

    def _evaluate_agent_in_round(
        self,
        agent: "AgentRole",
        factor_proposal: "FactorProposal",
        context: str,
        round_id: int,
    ) -> "AgentOpinion":
        """Evaluate one agent's refinement for a single debate round.

        Extracted from the serial loop in ``_structured_debate`` so it can run
        concurrently via ``ThreadPoolExecutor``. It is a pure function of
        (agent, context, round_id) with no shared mutable state, so it is
        thread-safe when dispatched across the thread pool.
        """
        role_name = agent.value
        system_prompt = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["Momentum Expert"])

        user_prompt = (
            f"You are the {role_name} participating in round {round_id + 1}/{self.n_rounds} "
            f"of a factor evaluation debate.\n\n"
            f"Factor to evaluate:\n"
            f"Expression: {factor_proposal.expression}\n"
            f"Description: {factor_proposal.description}\n\n"
            f"Other experts' opinions from the previous round:\n{context}\n\n"
            f"Refine YOUR evaluation of this factor from YOUR domain perspective:\n"
            f"- Keep grounding your score in the REAL backtest IC/Sharpe (shown earlier); "
            f"do not let peer pressure override hard evidence.\n"
            f"- Revise your score ONLY if a peer's argument is financially sound AND contradicts "
            f"your own evidence. Never change just to conform. If you hold your position, explain "
            f"why in 'reasoning'.\n"
            f"- Update 'strengths'/'concerns' to reflect what the debate surfaced.\n"
            f"Respond in JSON with keys: 'score' (float 0-10), 'reasoning' (string), "
            f"'concerns' (list of strings), 'strengths' (list of strings)."
        )

        if self.use_llm and self.client:
            try:
                raw = self._call_llm(system_prompt, user_prompt, temperature=0.3)
                parsed = json.loads(raw)
                score = float(parsed["score"])
                reasoning = str(parsed.get("reasoning", ""))
                concerns = [str(c) for c in parsed.get("concerns", [])]
                strengths = [str(s) for s in parsed.get("strengths", [])]
            except Exception as e:
                print(f"    [debate] Round {round_id + 1} LLM failed for {role_name}: {e}. Using fallback.")
                return self._fallback_round_opinion(agent, factor_proposal, round_id)
        else:
            # Mock mode
            score = self.rng.uniform(5.0, 9.0)
            reasoning = f"After debate round {round_id + 1}, I refine my opinion (mock)."
            concerns = ["Addressed in debate"]
            strengths = ["Refined through debate"]

        return AgentOpinion(
            agent_role=agent,
            factor_proposal=factor_proposal,
            score=score,
            reasoning=reasoning,
            concerns=concerns,
            strengths=strengths,
        )

    def _fallback_round_opinion(
        self,
        agent: "AgentRole",
        factor_proposal: "FactorProposal",
        round_id: int,
    ) -> "AgentOpinion":
        """Deterministic fallback opinion when a round-eval LLM call fails."""
        return AgentOpinion(
            agent_role=agent,
            factor_proposal=factor_proposal,
            score=self.rng.uniform(5.0, 9.0),
            reasoning=f"After debate round {round_id + 1}, I refine my opinion (fallback).",
            concerns=["Addressed in debate (fallback)"],
            strengths=["Refined through debate"],
        )

    def _build_debate_context(self, opinions: List[AgentOpinion], round_id: int) -> str:
        """Build context string from previous round opinions."""
        lines = []
        for op in opinions:
            lines.append(
                f"  {op.agent_role.value}: score={op.score:.1f}, "
                f"reasoning={op.reasoning[:80]}..., "
                f"concerns={op.concerns[:2]}"
            )
        return "\n".join(lines)

    def _arbitration_intervention(
        self,
        factor_proposal: FactorProposal,
        opinions: List[AgentOpinion],
        round_id: int,
        score_std: float,
    ) -> Optional[Dict]:
        """
        Chair Agent intervenes when experts strongly disagree (std > 2.0).

        The Chair:
        1. Identifies the dissenting experts and their arguments
        2. Weighs both sides explicitly
        3. Issues a binding ruling with an adjusted score

        Returns:
            Dict with keys: arbitration_score, ruling, concerns_addressed, strengths_confirmed
            or None if intervention fails.
        """
        # Identify the extremes (highest and lowest scores)
        sorted_ops = sorted(opinions, key=lambda x: x.score)
        low_op = sorted_ops[0]
        high_op = sorted_ops[-1]

        # Build a detailed brief for the Chair
        lines = [
            f"=== ARBITRATION REQUIRED ===",
            f"Factor: {factor_proposal.expression}",
            f"Description: {factor_proposal.description}",
        ]
        if factor_proposal.ic is not None:
            lines.append(f"IC: {factor_proposal.ic:.4f}")
        if factor_proposal.sharpe is not None:
            lines.append(f"Sharpe: {factor_proposal.sharpe:.4f}")
        lines.append(f"\nThe experts are DEADLOCKED (std={score_std:.2f} > 2.0 threshold).")
        lines.append(f"\nDissenting opinions:")
        for op in sorted_ops:
            lines.append(
                f"  - {op.agent_role.value}: score={op.score:.1f}\n"
                f"    Strengths: {op.strengths}\n"
                f"    Concerns: {op.concerns}\n"
                f"    Reasoning: {op.reasoning[:150]}"
            )
        lines.append(
            f"\nAs Chair, you MUST break this deadlock. "
            f"Identify WHICH expert has the stronger argument, and WHY. "
            f"Provide your ruling as JSON with keys: "
            f"'arbitration_score' (float 0-10, your final judgment), "
            f"'ruling' (string, your decisive reasoning), "
            f"'concerns_addressed' (list of strings, which concerns are valid), "
            f"'strengths_confirmed' (list of strings, which strengths you confirm)."
        )

        chair_prompt = "\n".join(lines)

        try:
            raw = self._call_chair_llm(
                system_prompt=(
                    f"You are the Chair Agent — the final arbiter in a multi-expert factor "
                    f"evaluation debate. Five domain experts are deadlocked (std={score_std:.2f}). "
                    f"Your ruling is BINDING. Break the tie with decisive reasoning. "
                    f"Prefer the expert whose argument is most grounded in financial theory "
                    f"AND consistent with the backtest data."
                ),
                user_prompt=chair_prompt,
                temperature=0.2,
            )
            return json.loads(raw)
        except Exception as e:
            print(f"    [debate] Chair arbitration call failed: {e}")
            return None

    def _reach_consensus(self, factor_proposal: FactorProposal,
                         debate_rounds: List[DebateRound]) -> DebateResult:
        """
        Phase 3: Reach final consensus from all debate rounds.
        """
        # Collect all agent scores from final round
        final_round = debate_rounds[-1]
        agent_scores = {}
        for opinion in final_round.opinions:
            agent_scores[opinion.agent_role.value] = opinion.score

        # Use weighted average (later rounds count more)
        all_scores = []
        all_weights = []
        for i, rnd in enumerate(debate_rounds):
            weight = (i + 1) / len(debate_rounds)  # later rounds weighted more
            for op in rnd.opinions:
                all_scores.append(op.score)
                all_weights.append(weight)

        final_score = float(np.average(all_scores, weights=all_weights))

        # Ask Chair Agent (stronger model) for final consensus
        chair_available = bool(self.chair_client) or bool(self.client)
        if self.use_llm and chair_available:
            try:
                summary_prompt = self._build_consensus_prompt(factor_proposal, debate_rounds)
                raw = self._call_chair_llm(
                    system_prompt=(
                        "You are the Chief Investment Officer (Chair Agent) synthesizing a "
                        "multi-agent factor evaluation debate. "
                        "Five experts (Momentum, Value, Quality, Volatility, Growth) have debated "
                        "this factor. Your job is to weigh their ARGUMENTS (not just their numbers), "
                        "resolve disagreements, and produce the FINAL authoritative judgment. "
                        "Your 'final_score' (0-10) MUST stay consistent with the debate score "
                        "distribution shown in the prompt (mean and std) — do not deviate wildly "
                        "without explicit justification. Be decisive — if experts are split, break "
                        "the tie with clear reasoning grounded in the backtest evidence."
                    ),
                    user_prompt=summary_prompt,
                    temperature=0.2,
                )
                parsed = json.loads(raw)
                final_score = float(parsed.get("final_score", final_score))
                key_insights = [str(x) for x in parsed.get("key_insights", [])]
                recommendation = str(parsed.get("recommendation", "REJECT"))
                consensus_summary = str(parsed.get("consensus_summary", ""))
            except Exception as e:
                print(f"  [debate] Chair consensus call failed: {e}. Using fallback.")
                key_insights = [
                    f"Final score: {final_score:.2f}",
                    f"Consensus std: {np.std([op.score for op in final_round.opinions]):.2f}",
                ]
                recommendation = "APPROVE" if final_score >= 7.0 else "REJECT"
                consensus_summary = f"Weighted average of {len(all_scores)} opinions across {len(debate_rounds)} rounds."
        else:
            # Mock consensus
            key_insights = [
                "Factor shows strong momentum characteristics",
                "Low correlation with existing factors",
                "Performs well in bull markets",
            ]
            recommendation = "APPROVE" if final_score >= 7.0 else "REJECT"
            consensus_summary = f"Mock consensus: weighted score {final_score:.2f}"

        return DebateResult(
            factor_proposal=factor_proposal,
            final_score=final_score,
            agent_scores=agent_scores,
            key_insights=key_insights,
            recommendation=recommendation,
            consensus_summary=consensus_summary,
            all_rounds=debate_rounds,
        )

    def _build_consensus_prompt(self, factor_proposal: FactorProposal,
                                debate_rounds: List[DebateRound]) -> str:
        """Build prompt for final consensus LLM call."""
        all_scores = [op.score for rnd in debate_rounds for op in rnd.opinions]
        mean_s = float(np.mean(all_scores)) if all_scores else 0.0
        std_s = float(np.std(all_scores)) if all_scores else 0.0

        lines = [
            f"Factor: {factor_proposal.expression}",
            f"Description: {factor_proposal.description}",
        ]
        if factor_proposal.ic is not None:
            lines.append(f"IC: {factor_proposal.ic:.4f}")
        if factor_proposal.sharpe is not None:
            lines.append(f"Sharpe: {factor_proposal.sharpe:.4f}")
        lines.append(f"\nDebate score distribution: mean={mean_s:.2f}, std={std_s:.2f}")
        lines.append("Debate rounds (scores + each agent's key argument):")
        for rnd in debate_rounds:
            for op in rnd.opinions:
                top_concern = (op.concerns[0] if op.concerns else "")
                top_strength = (op.strengths[0] if op.strengths else "")
                lines.append(
                    f"  R{rnd.round_id + 1} {op.agent_role.value}={op.score:.1f} | "
                    f"strength: {self._truncate(top_strength, 120)} | "
                    f"concern: {self._truncate(top_concern, 120)}"
                )
        lines.append(
            "\nSynthesize the ARGUMENTS above (not just the numbers). "
            "Provide final consensus as JSON with keys: "
            "'final_score' (float 0-10, consistent with the distribution above), "
            "'key_insights' (list of strings — the decisive points that settled the debate), "
            "'recommendation' (APPROVE/REJECT/CONDITIONAL), "
            "'consensus_summary' (string — concise wrap-up of the verdict)."
        )
        return "\n".join(lines)

    def synthesize_all_factors(
        self,
        debate_results: List[Tuple[FactorProposal, "DebateResult"]],
    ) -> Dict:
        """
        Chair Agent synthesizes ALL debate results into a comprehensive cross-factor report.

        This is the FINAL step of multi-agent debate. The Chair Agent:
        1. Reviews all factor evaluations holistically
        2. Ranks factors by combined quality (debate score + backtest metrics)
        3. Outputs selection/rejection reasons for EACH factor
        4. Identifies dominant themes and portfolio construction implications

        Args:
            debate_results: List of (FactorProposal, DebateResult) pairs

        Returns:
            Dict with keys: overall_assessment, factors_ranked, selected_count,
            rejected_factors, key_themes, chair_confidence, timestamp
        """
        if not debate_results:
            return {
                "overall_assessment": "No factors to synthesize.",
                "factors_ranked": [],
                "selected_count": 0,
                "rejected_factors": [],
                "key_themes": [],
                "chair_confidence": "LOW",
                "timestamp": datetime.now().isoformat(),
                "_source": "empty",
            }

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(debate_results)

        # Try Chair Agent synthesis first
        if self.use_llm and (self.chair_client or self.client):
            try:
                raw = self._call_chair_llm(
                    system_prompt=(
                        "You are the Chief Investment Officer (Chair Agent) making FINAL decisions "
                        "on factor selection. You have reviewed multi-agent debate results for ALL "
                        "candidate factors. Your job:\n"
                        "1. Rank factors from best to worst\n"
                        "2. For each SELECTED factor, explain WHY it was chosen (具体的入选理由) — "
                        "reference its backtest IC/Sharpe AND the debate evidence\n"
                        "3. For each REJECTED factor, explain WHY it was eliminated (具体的淘汰原因)\n"
                        "4. Identify cross-cutting themes (e.g., 'momentum dominates', 'value factors weak')\n"
                        "5. Give an overall confidence assessment\n\n"
                        "CRITICAL — FAMILY DIVERSITY (do NOT skip):\n"
                        "Each factor is labeled with its Family (Momentum / Mean-reversion / "
                        "Value-Quality / Volatility / Liquidity / Growth). A good portfolio of "
                        "factors spans MULTIPLE families. Do NOT select 10 factors that are all "
                        "variants of the same idea (e.g., all -rank(pb)-style Value-Quality clones) "
                        f"just because they score similarly. Cap selection to at most "
                        f"{self.chair_max_per_family} factors per "
                        "family UNLESS one family is dramatically superior (then justify it). "
                        "Report portfolio concentration risk explicitly in 'key_themes'.\n\n"
                        "CRITICAL: Be specific. Vague answers like 'performs well' are insufficient. "
                        "Reference specific agent arguments, backtest metrics, and financial logic.\n\n"
                        "Respond in JSON with keys exactly as specified.\n\n"
                        "NOTE: Only the top factors are given in full detail — produce detailed "
                        "selection/rejection entries for THOSE. The compact lower-ranked list is "
                        "provided for context only; do not emit a separate entry per compact line, "
                        "but factor their themes into 'key_themes'."
                    ),
                    user_prompt=prompt,
                    temperature=0.2,
                )
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    # Response was likely truncated by max_tokens — try to
                    # recover a usable prefix before giving up to rule-based.
                    parsed = self._salvage_json(raw)

                # Validate and coerce required fields
                factors_ranked = self._validate_factors_ranked(
                    parsed.get("factors_ranked", [])
                )
                rejected_factors = self._validate_rejected_factors(
                    parsed.get("rejected_factors", [])
                )
                result = {
                    "overall_assessment": str(parsed.get("overall_assessment", "")),
                    "factors_ranked": factors_ranked,
                    # Use the REAL length of the validated ranked list, not the
                    # LLM's self-reported selected_count — the chair LLM has been
                    # observed to miscount (e.g. reporting 5 selected + 9 rejected
                    # = 14 for an input of 13). Aligns with _rule_based_synthesis.
                    "selected_count": len(factors_ranked),
                    "rejected_factors": rejected_factors,
                    "key_themes": [str(t) for t in parsed.get("key_themes", [])],
                    "chair_confidence": str(parsed.get("chair_confidence", "MEDIUM")),
                    "timestamp": datetime.now().isoformat(),
                    "_source": "chair_llm",
                }
                # Consistency check: the union of selected+rejected must cover
                # exactly the input count, with no overlap. The chair LLM is not
                # reliable here, so surface any mismatch instead of hiding it.
                _total_input = len(debate_results)
                _sel_exprs = {str(i.get('expression', '')).strip() for i in factors_ranked}
                _rej_exprs = {str(i.get('expression', '')).strip() for i in rejected_factors}
                _overlap = _sel_exprs & _rej_exprs
                _listed = len(_sel_exprs | _rej_exprs)
                if _listed != _total_input or _overlap:
                    print(f"  [warn] Chair count inconsistency: input={_total_input}, "
                          f"listed={_listed} (selected={len(_sel_exprs)}, "
                          f"rejected={len(_rej_exprs)}, overlap={len(_overlap)}). "
                          f"Chair may have double-counted or missed factors.")
                print(f"  [debate] Chair synthesis complete: {result['selected_count']} selected, "
                      f"{len(result['rejected_factors'])} rejected, "
                      f"confidence={result['chair_confidence']}")
                self._save_synthesis_report(result)
                self.save_results()                       # write per-factor list first
                self._append_synthesis_to_debate_file(result)  # then append chair records
                return result

            except Exception as e:
                print(f"  [debate] Chair synthesis LLM call failed: {e}. Using rule-based fallback.")
        else:
            print(f"  [debate] Chair LLM unavailable. Using rule-based synthesis.")

        # Fallback: rule-based synthesis
        result = self._rule_based_synthesis(debate_results)
        self._save_synthesis_report(result)
        self.save_results()                       # write per-factor list first
        self._append_synthesis_to_debate_file(result)  # then append chair records
        return result

    def _save_synthesis_report(self, synthesis: Dict):
        """
        Save synthesis report to experiments/{yyyymmdd}/chair_synthesis.json.

        Called automatically at the end of synthesize_all_factors().
        """
        import os, json
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join(date_str, "debate")
        output_dir = config_path("experiments", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "chair_synthesis.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(synthesis, f, ensure_ascii=False, indent=2)

        print(f"  [debate] Synthesis report saved: {output_path}")

    def _append_synthesis_to_debate_file(self, synthesis: Dict):
        """
        Append Chair Agent synthesis results to experiments/{yyyymmdd}/debate/debate_factors_result.json.

        Adds per-factor records (selected + rejected) plus an overall metadata record,
        so the debate output JSON is self-contained (all per-factor opinions + Chair ruling).
        """
        import os, json
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        debate_dir = os.path.join(date_str, "debate")
        debate_dir = config_path("experiments", debate_dir)
        os.makedirs(debate_dir, exist_ok=True)
        debate_path = os.path.join(debate_dir, "debate_factors_result.json")

        # Read existing records (append mode)
        existing = []
        if os.path.exists(debate_path) and os.path.getsize(debate_path) > 0:
            try:
                with open(debate_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, Exception):
                existing = []

        # Build chair synthesis records — one entry per ranked factor
        for item in synthesis.get("factors_ranked", []):
            existing.append({
                "factor_expression": item.get("expression", ""),
                "round": "chair_synthesis",
                "round_type": "chair_synthesis",
                "agent_role": "Chair",
                "rank": item.get("rank"),
                "final_score": item.get("final_score"),
                "selection_reason": item.get("selection_reason", ""),
                "strengths": item.get("strengths", []),
                "risks": item.get("risks", []),
            })

        # Rejected factors
        for item in synthesis.get("rejected_factors", []):
            existing.append({
                "factor_expression": item.get("expression", ""),
                "round": "chair_synthesis",
                "round_type": "chair_synthesis",
                "agent_role": "Chair",
                "rank": None,
                "final_score": item.get("final_score"),
                "rejection_reason": item.get("rejection_reason", ""),
            })

        # Overall synthesis metadata
        existing.append({
            "factor_expression": "__chair_synthesis_meta__",
            "round": "chair_synthesis",
            "round_type": "chair_synthesis",
            "agent_role": "Chair",
            "overall_assessment": synthesis.get("overall_assessment", ""),
            "selected_count": synthesis.get("selected_count", 0),
            "key_themes": synthesis.get("key_themes", []),
            "chair_confidence": synthesis.get("chair_confidence", ""),
            "timestamp": synthesis.get("timestamp", ""),
            "source": synthesis.get("_source", ""),
        })

        with open(debate_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"  [debate] Chair synthesis appended to {debate_path}")

    @staticmethod
    def _truncate(text, limit: int) -> str:
        """Truncate a string to `limit` chars, appending '...' if cut."""
        if text is None:
            return ""
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @staticmethod
    def _salvage_json(raw: str):
        """Best-effort repair of a JSON string truncated by a max_tokens cap.

        The Chair response can be cut mid-value when generation hits the token
        limit. This closes an open string, strips a trailing comma, and closes
        open containers in correct LIFO order so a usable prefix still parses.
        Raises (json.JSONDecodeError or similar) if the truncation is too deep
        to repair — callers should then fall back to rule-based synthesis.
        """
        s = raw.rstrip()
        stack = []          # stack of still-open '[' / '{'
        instr = False
        esc = False
        for ch in s:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                instr = not instr
                continue
            if instr:
                continue
            if ch in '[{':
                stack.append(ch)
            elif ch in ']}':
                if stack:
                    stack.pop()
        if instr:
            s += '"'                       # close unterminated string
        s = s.rstrip()
        if s.endswith(','):
            s = s[:-1]                     # drop dangling comma before close
        close_map = {'[': ']', '{': '}'}
        while stack:
            s += close_map[stack.pop()]    # close remaining containers (LIFO)
        return json.loads(s)

    def _build_synthesis_prompt(
        self,
        debate_results: List[Tuple[FactorProposal, "DebateResult"]],
    ) -> str:
        """Build the comprehensive synthesis prompt for the Chair Agent.

        Bounded two ways to stay within the Chair request's latency/timeout budget
        (a single unbounded prompt over 100+ factors on the proxy endpoint routinely
        exceeds request_timeout and silently falls back to rule-based synthesis):

        1. Per-factor verbose fields (consensus summary, key insights) are truncated.
        2. Only the top-`synthesis_top_n` factors (by final_score) get the full
           detail block; the remaining lower-ranked factors are listed compactly so
           the Chair is still aware of them without a full per-factor entry.
        """
        # Sort by debate score descending for clarity
        sorted_results = sorted(
            debate_results,
            key=lambda x: x[1].final_score,
            reverse=True,
        )

        top = sorted_results[: self.synthesis_top_n]
        rest = sorted_results[self.synthesis_top_n :]

        # Family distribution across the WHOLE pool — gives the Chair an at-a-glance
        # view of concentration so it can honor the family-diversity directive.
        fam_counts: Dict[str, int] = {}
        for proposal, _ in sorted_results:
            fam = proposal.family or _infer_family(proposal.expression)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        fam_dist = ", ".join(f"{k}={v}" for k, v in sorted(
            fam_counts.items(), key=lambda kv: -kv[1]))

        lines = [
            "=== CROSS-FACTOR SYNTHESIS ===",
            f"Total factors evaluated: {len(debate_results)}",
            f"Family distribution (whole pool): {fam_dist}",
            f"Detailed review below for the top {len(top)} factors by debate score; "
            f"{len(rest)} lower-ranked factors are summarized compactly at the end.",
            "",
            "Below are the complete debate results for each factor:",
            "",
        ]

        for i, (proposal, result) in enumerate(top):
            fam = proposal.family or _infer_family(proposal.expression)
            lines.append(f"--- Factor #{i + 1} ---")
            lines.append(f"Expression: {proposal.expression}")
            lines.append(f"Description: {proposal.description}")
            lines.append(f"Family: {fam}")
            if proposal.ic is not None:
                lines.append(f"Backtest IC: {proposal.ic:.4f}")
            if proposal.sharpe is not None:
                lines.append(f"Backtest Sharpe: {proposal.sharpe:.4f}")
            lines.append(f"Debate Final Score: {result.final_score:.2f}")
            lines.append(f"Recommendation: {result.recommendation}")
            lines.append(f"Agent Scores: {json.dumps(result.agent_scores)}")
            lines.append(f"Consensus Summary: {self._truncate(result.consensus_summary, 240)}")
            lines.append(f"Key Insights: {self._truncate(json.dumps(result.key_insights), 480)}")
            if result.all_rounds:
                # Show score evolution across rounds
                round_scores = []
                for rnd in result.all_rounds:
                    avg = float(np.mean([op.score for op in rnd.opinions]))
                    round_scores.append(f"R{rnd.round_id + 1}={avg:.2f}")
                lines.append(f"Score Evolution: {' → '.join(round_scores)}")
            lines.append("")

        if rest:
            lines.append("--- Lower-ranked candidates (compact) ---")
            for (proposal, result) in rest:
                fam = proposal.family or _infer_family(proposal.expression)
                lines.append(
                    f"- {proposal.expression} | family={fam} | "
                    f"score={result.final_score:.2f} | {result.recommendation}"
                )
            lines.append("")

        lines.append("---")
        lines.append(
            "Provide your cross-factor synthesis as JSON with keys:\n"
            "  'overall_assessment' (string): Your holistic judgment of the factor pool\n"
            "  'factors_ranked' (array): Top factors ranked from best to worst, each with:\n"
            "    - 'rank' (int), 'expression' (string), 'final_score' (float),\n"
            "    - 'selection_reason' (string, 入选理由 — WHY this factor is selected)\n"
            "    - 'decision' (string, optional): 'selected' (default) or 'conditional' "
            "(approved WITH conditions / monitoring — still KEPT in the pool)\n"
            "    NOTE: keep selection_reason concise (1-2 sentences). Do NOT add 'strengths'/'risks' arrays.\n"
            "  'selected_count' (int): How many factors you recommend selecting\n"
            "  'rejected_factors' (array): Factors NOT selected, each with:\n"
            "    - 'expression' (string), 'final_score' (float),\n"
            "    - 'rejection_reason' (string, 淘汰原因 — WHY this factor was rejected, concise)\n"
            "    - 'decision' (string, optional): 'rejected' (default)\n"
            "  IMPORTANT: Only HARD rejections go in 'rejected_factors'. Factors approved WITH "
            "conditions ('conditional') must be listed in 'factors_ranked' with "
            "decision:'conditional', NOT in 'rejected_factors'.\n"
            "  'key_themes' (array of strings): Cross-cutting observations\n"
            "  'chair_confidence' (string): HIGH / MEDIUM / LOW\n"
            "FAMILY DIVERSITY: each factor above is labeled with its Family. Prefer a "
            "diverse selection that spans MULTIPLE families; DO NOT pick many near-duplicate "
            "factors from the same family. Report any portfolio concentration risk in 'key_themes'.\n"
            "Keep the ENTIRE response compact — verbose output risks being truncated by the token limit."
        )
        return "\n".join(lines)

    def _rule_based_synthesis(
        self,
        debate_results: List[Tuple[FactorProposal, "DebateResult"]],
    ) -> Dict:
        """
        Rule-based fallback synthesis when Chair LLM is unavailable.

        Uses weighted ranking: debate_score * 0.5 + ic * 0.3 + sharpe_normalized * 0.2
        """
        ranked = []
        rejected = []

        for proposal, result in debate_results:
            score = result.final_score
            ic = proposal.ic or 0.0
            sharpe_norm = min(max((proposal.sharpe or 0.0) / 3.0, 0.0), 1.0)
            composite = score * 0.5 + abs(ic) * 10 * 0.3 + sharpe_norm * 2 * 0.2

            entry = {
                "expression": proposal.expression,
                "final_score": score,
                "composite_score": round(composite, 2),
                "strengths": result.key_insights,
                "risks": [],
            }

            if result.recommendation == "APPROVE" or score >= 7.0:
                entry["selection_reason"] = (
                    f"辩论综合评分 {score:.1f}/10，{result.recommendation}。"
                    f"IC={ic:.3f}，{result.consensus_summary}"
                )
                entry["decision"] = "selected"
                ranked.append(entry)
            else:
                entry["rejection_reason"] = (
                    f"辩论综合评分 {score:.1f}/10 < 7.0 阈值，{result.recommendation}。"
                    f"{result.consensus_summary}"
                )
                entry["decision"] = "rejected"
                rejected.append(entry)

        # Sort ranked by composite score descending
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, entry in enumerate(ranked):
            entry["rank"] = i + 1

        # --- Family-balanced capping (mirror evolve._family_balanced_top) ---
        # Without the Chair LLM, we cannot rely on free-text diversity reasoning,
        # so we enforce it structurally: keep at most MAX_PER_FAMILY factors per
        # family, spilling the rest into `rejected` with an explicit reason. This
        # stops a Value/Quality-heavy population from collapsing the final
        # portfolio into a single family when the LLM is unavailable.
        _MAX_PER_FAMILY = 3
        fam_count: Dict[str, int] = {}
        capped_extra = []
        selected = []
        for entry in ranked:  # already composite-desc
            fam = _infer_family(entry["expression"])
            entry["family"] = fam
            if fam_count.get(fam, 0) < _MAX_PER_FAMILY:
                fam_count[fam] = fam_count.get(fam, 0) + 1
                selected.append(entry)
            else:
                capped_extra.append(entry)
        for entry in capped_extra:
            rejected.append({
                "expression": entry["expression"],
                "final_score": entry["final_score"],
                "rejection_reason": (
                    f"家族集中剔除: {entry['family']} 已入选 {_MAX_PER_FAMILY} 个（上限），"
                    f"composite={entry['composite_score']} 但因组合多样性要求移入淘汰。"
                ),
                "decision": "rejected",
            })
        ranked = selected
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, entry in enumerate(ranked):
            entry["rank"] = i + 1

        # Identify themes from agent scores
        themes = []
        all_recommendations = [r.recommendation for _, r in debate_results]
        approve_count = sum(1 for r in all_recommendations if r == "APPROVE")
        themes.append(f"{approve_count}/{len(debate_results)} factors approved by debate")

        avg_score = float(np.mean([r.final_score for _, r in debate_results]))
        themes.append(f"Average debate score: {avg_score:.2f}/10")

        # Family concentration report (post-cap) so the report surfaces diversity
        if ranked:
            fam_dist = {}
            for e in ranked:
                fam_dist[e["family"]] = fam_dist.get(e["family"], 0) + 1
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(fam_dist.items(),
                                                              key=lambda kv: -kv[1]))
            themes.append(f"Selected family distribution: {dist_str}")
            if len(fam_dist) == 1:
                themes.append(
                    "CONCENTRATION RISK: all selected factors belong to a single family — "
                    "population lacks diversity, consider widening the search."
                )

        return {
            "overall_assessment": (
                f"Rule-based synthesis of {len(debate_results)} factors. "
                f"{approve_count} approved, {len(debate_results) - approve_count} rejected. "
                f"Average debate score: {avg_score:.2f}/10."
            ),
            "factors_ranked": ranked,
            "selected_count": len(ranked),
            "rejected_factors": rejected,
            "key_themes": themes,
            "chair_confidence": "LOW",
            "timestamp": datetime.now().isoformat(),
            "_source": "rule_based",
        }

    @staticmethod
    def _validate_factors_ranked(raw: List) -> List[Dict]:
        """Validate and coerce factors_ranked list."""
        cleaned = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cleaned.append({
                "rank": int(item.get("rank", len(cleaned) + 1)),
                "expression": str(item.get("expression", "")),
                "final_score": float(item.get("final_score", 0.0)),
                "selection_reason": str(item.get("selection_reason", "")),
                "decision": str(item.get("decision", "selected")).strip().lower() or "selected",
                "strengths": [str(s) for s in item.get("strengths", [])],
                "risks": [str(r) for r in item.get("risks", [])],
            })
        return cleaned

    @staticmethod
    def _validate_rejected_factors(raw: List) -> List[Dict]:
        """Validate and coerce rejected_factors list."""
        cleaned = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cleaned.append({
                "expression": str(item.get("expression", "")),
                "final_score": float(item.get("final_score", 0.0)),
                "rejection_reason": str(item.get("rejection_reason", "")),
                "decision": str(item.get("decision", "rejected")).strip().lower() or "rejected",
            })
        return cleaned


if __name__ == '__main__':
    # Demo (requires DEEPSEEK_API_KEY env var)
    print("=== Multi-Agent Debate Evaluator Demo ===\n")

    proposal = FactorProposal(
        expression="rank(ts_corr(close, volume, 20))",
        description="Ranking time series correlation of close and volume over 20 days",
        ic=0.03,
        sharpe=1.2,
    )

    evaluator = DebateEvaluator(
        llm_model="deepseek-chat",
        n_agents=5,
        n_rounds=2,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )

    result = evaluator.evaluate(proposal)

    print(f"\nFinal Score: {result.final_score:.2f}")
    print(f"Recommendation: {result.recommendation}")
    print(f"\nAgent Scores:")
    for agent, score in result.agent_scores.items():
        print(f"  {agent}: {score:.2f}")

    print(f"\nKey Insights:")
    for insight in result.key_insights:
        print(f"  - {insight}")

    print("\n=== Demo Complete ===")
