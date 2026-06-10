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
import json
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Role-specific system prompts
ROLE_PROMPTS = {
    "Momentum Expert": (
        "You are a Momentum Expert specializing in quantitative stock selection. "
        "You evaluate factors based on: trend persistence, volume confirmation, "
        "reactivity to price signals, and robustness across market regimes. "
        "A good momentum factor should have high IC, low turnover, and work well "
        "in trending markets. Score factors 0-10 and provide concise reasoning."
    ),
    "Value Expert": (
        "You are a Value Expert specializing in fundamental stock selection. "
        "You evaluate factors based on: fundamental signal quality, mean-reversion properties, "
        "avoidance of value traps, and long-term predictive power. "
        "A good value factor should capture mispricing, have economic intuition, "
        "and be robust across business cycles. Score factors 0-10."
    ),
    "Quality Expert": (
        "You are a Quality Expert specializing in corporate fundamental analysis. "
        "You evaluate factors based on: earnings quality, profitability sustainability, "
        "balance sheet strength, and management efficiency. "
        "A good quality factor should identify companies with durable competitive advantages, "
        "low bankruptcy risk, and consistent cash flow generation. Score factors 0-10."
    ),
    "Volatility Expert": (
        "You are a Volatility Expert specializing in risk-adjusted factor evaluation. "
        "You evaluate factors based on: downside protection, drawdown control, "
        "tail-risk behavior, and risk-adjusted return potential. "
        "A good low-vol factor should have stable IC, low max drawdown contribution, "
        "and positive performance in stress periods. Score factors 0-10."
    ),
    "Growth Expert": (
        "You are a Growth Expert specializing in forward-looking signal evaluation. "
        "You evaluate factors based on: earnings growth sustainability, analyst revision trends, "
        "R&D efficiency, and scalability of business model. "
        "A good growth factor should identify companies with accelerating fundamentals, "
        "high ROCIC, and expanding market share. Score factors 0-10."
    ),
}

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


@dataclass
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
    ):
        """
        Initialize debate evaluator.

        Args:
            llm_model: LLM model name (deepseek-chat, deepseek-reasoner, etc.)
            n_agents: Number of agents (max 5, one per role)
            n_rounds: Number of debate rounds
            random_seed: Random seed (used only as fallback)
            api_key: DeepSeek API key. If empty, reads from DEEPSEEK_API_KEY env var.
            base_url: API base URL
        """
        self.llm_model = llm_model
        self.n_agents = min(n_agents, 5)
        self.n_rounds = n_rounds
        self.rng = np.random.RandomState(random_seed)
        self.use_llm = OPENAI_AVAILABLE

        # Initialize OpenAI client
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
                    print(f"  [debate] LLM client initialized: {llm_model} @ {base_url}")
                    print(f"  [debate] Connection test passed.")
                except Exception as e:
                    print(f"  [debate] Warning: API connection test failed: {e}")
                    print(f"  [debate] Falling back to mock mode.")
                    self.client = None
                    self.use_llm = False
            else:
                print(f"  [debate] Warning: No API key found. Set api_key in config or DEEPSEEK_API_KEY env var.")
                print(f"  [debate] Falling back to mock mode.")
                self.use_llm = False
        else:
            print(f"  [debate] openai package not installed. Running in mock mode.")
            self.use_llm = False

        # Initialize agents
        self.agents = self._initialize_agents()

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

        # Save all expert opinions (independent + debate rounds) to CSV
        self._save_debate_results(factor_proposal, independent_opinions, debate_rounds, result)

        rec = result.recommendation
        print(f"  [debate] Done. Score={result.final_score:.2f}, Recommendation={rec}")
        return result

    def _call_llm(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.3, expect_json: bool = True) -> str:
        """Call LLM and return response content."""
        if not self.client:
            raise RuntimeError("LLM client not initialized. Check API key.")

        kwargs = dict(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(**kwargs)

            # Debug: inspect resp type to catch non-standard responses
            if not hasattr(resp, "choices"):
                # resp is not a ChatCompletion object — likely a raw string/error
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
            # Print the actual exception type and message for debugging
            import traceback
            tb = traceback.format_exc()
            print(f"  [debate] LLM call failed ({type(e).__name__}): {e}")
            print(f"  [debate] Traceback (last frame): {tb.splitlines()[-1] if tb else 'N/A'}")
            raise

    def _independent_evaluation(self, factor_proposal: FactorProposal) -> List[AgentOpinion]:
        """
        Phase 1: Each agent evaluates the factor independently.
        """
        opinions = []
        ic_str = f", IC={factor_proposal.ic:.4f}" if factor_proposal.ic is not None else ""
        sharpe_str = f", Sharpe={factor_proposal.sharpe:.4f}" if factor_proposal.sharpe is not None else ""

        user_prompt_template = (
            f"Please evaluate the following quantitative stock selection factor:\n\n"
            f"Expression: {factor_proposal.expression}\n"
            f"Description: {factor_proposal.description}\n"
            f"{ic_str}{sharpe_str}\n\n"
            f"Provide your evaluation as a JSON object with keys: "
            f"'score' (float 0-10), 'reasoning' (string), "
            f"'concerns' (list of strings), 'strengths' (list of strings)."
        )

        for agent in self.agents:
            role_name = agent.value
            system_prompt = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["Momentum Expert"])

            if self.use_llm and self.client:
                try:
                    raw = self._call_llm(system_prompt, user_prompt_template, temperature=0.3)
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

            opinion = AgentOpinion(
                agent_role=agent,
                factor_proposal=factor_proposal,
                score=score,
                reasoning=reasoning,
                concerns=concerns,
                strengths=strengths,
            )
            opinions.append(opinion)

        return opinions

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

            for i, agent in enumerate(self.agents):
                role_name = agent.value
                system_prompt = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["Momentum Expert"])

                user_prompt = (
                    f"You are participating in round {round_id + 1}/{self.n_rounds} of a factor evaluation debate.\n\n"
                    f"Factor to evaluate:\n"
                    f"Expression: {factor_proposal.expression}\n"
                    f"Description: {factor_proposal.description}\n\n"
                    f"Other experts' opinions from previous round:\n{context}\n\n"
                    f"Based on the above discussion, refine YOUR evaluation of this factor. "
                    f"Respond in JSON with keys: 'score' (float 0-10), 'reasoning' (string), "
                    f"'concerns' (list), 'strengths' (list), 'changed_mind' (bool, whether you revised your score)."
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
                        score = self.rng.uniform(5.0, 9.0)
                        reasoning = f"After debate round {round_id + 1}, I refine my opinion (fallback)."
                        concerns = ["Addressed in debate (fallback)"]
                        strengths = ["Refined through debate"]
                else:
                    score = self.rng.uniform(5.0, 9.0)
                    reasoning = f"After debate round {round_id + 1}, I refine my opinion (mock)."
                    concerns = ["Addressed in debate"]
                    strengths = ["Refined through debate"]

                opinion = AgentOpinion(
                    agent_role=agent,
                    factor_proposal=factor_proposal,
                    score=score,
                    reasoning=reasoning,
                    concerns=concerns,
                    strengths=strengths,
                )
                opinions.append(opinion)

            # Calculate consensus and disagreements
            scores = [op.score for op in opinions]
            consensus_score = float(np.mean(scores))
            score_std = float(np.std(scores))
            if score_std > 2.0:
                disagreements.append(f"High score variance: std={score_std:.2f}")

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

        # Ask LLM for final consensus (if available)
        if self.use_llm and self.client:
            try:
                summary_prompt = self._build_consensus_prompt(factor_proposal, debate_rounds)
                raw = self._call_llm(
                    system_prompt="You are the chief investment officer synthesizing a multi-agent factor evaluation.",
                    user_prompt=summary_prompt,
                    temperature=0.2,
                )
                parsed = json.loads(raw)
                final_score = float(parsed.get("final_score", final_score))
                key_insights = [str(x) for x in parsed.get("key_insights", [])]
                recommendation = str(parsed.get("recommendation", "REJECT"))
                consensus_summary = str(parsed.get("consensus_summary", ""))
            except Exception as e:
                print(f"  [debate] Consensus LLM call failed: {e}. Using fallback.")
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
        lines = [
            f"Factor: {factor_proposal.expression}",
            f"Description: {factor_proposal.description}",
        ]
        if factor_proposal.ic is not None:
            lines.append(f"IC: {factor_proposal.ic:.4f}")
        if factor_proposal.sharpe is not None:
            lines.append(f"Sharpe: {factor_proposal.sharpe:.4f}")
        lines.append("\nDebate rounds summary:")
        for rnd in debate_rounds:
            score_str = ", ".join([f"{op.agent_role.value}={op.score:.1f}" for op in rnd.opinions])
            lines.append(f"  Round {rnd.round_id + 1}: {score_str}")
        lines.append(
            "\nProvide final consensus as JSON with keys: "
            "'final_score' (float 0-10), 'key_insights' (list of strings), "
            "'recommendation' (APPROVE/REJECT/CONDITIONAL), "
            "'consensus_summary' (string)."
        )
        return "\n".join(lines)

    def _save_debate_results(
        self,
        factor_proposal: FactorProposal,
        independent_opinions: List[AgentOpinion],
        debate_rounds: List[DebateRound],
        result: DebateResult,
    ) -> None:
        """
        Save all expert opinions (independent evaluation + debate rounds) to JSON.

        Output path: experiments/{yyyymmdd}/debate/debate_factors_result.json
        Each entry = one expert's opinion for one factor in one round.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join("experiments", date_str, "debate")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "debate_factors_result.json")

        # Read existing records if file already exists (append mode)
        existing = []
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]  # tolerate a single object
            except (json.JSONDecodeError, Exception):
                existing = []

        records = []

        # --- Independent evaluation (Phase 1) ---
        for op in independent_opinions:
            records.append({
                "factor_expression": factor_proposal.expression,
                "factor_description": factor_proposal.description,
                "factor_ic": factor_proposal.ic,
                "factor_sharpe": factor_proposal.sharpe,
                "round": -1,
                "round_type": "independent",
                "agent_role": op.agent_role.value,
                "score": op.score,
                "reasoning": op.reasoning,
                "concerns": op.concerns,
                "strengths": op.strengths,
                "round_consensus_std": None,
                "final_score": result.final_score,
                "recommendation": result.recommendation,
                "key_insights": result.key_insights,
                "consensus_summary": result.consensus_summary,
            })

        # --- Debate rounds (Phase 2) ---
        for rnd in debate_rounds:
            scores_in_round = [op.score for op in rnd.opinions]
            round_std = round(float(np.std(scores_in_round)), 4) if len(scores_in_round) >= 2 else None
            for op in rnd.opinions:
                records.append({
                    "factor_expression": factor_proposal.expression,
                    "factor_description": factor_proposal.description,
                    "factor_ic": factor_proposal.ic,
                    "factor_sharpe": factor_proposal.sharpe,
                    "round": rnd.round_id,
                    "round_type": "debate",
                    "agent_role": op.agent_role.value,
                    "score": op.score,
                    "reasoning": op.reasoning,
                    "concerns": op.concerns,
                    "strengths": op.strengths,
                    "round_consensus_std": round_std,
                    "final_score": result.final_score,
                    "recommendation": result.recommendation,
                    "key_insights": result.key_insights,
                    "consensus_summary": result.consensus_summary,
                })

        # Append to existing and write
        all_records = existing + records
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        print(f"  [debate] Saved {len(records)} opinion records (total {len(all_records)}) to {output_path}")


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
