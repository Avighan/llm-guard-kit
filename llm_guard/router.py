"""
SmartRouter — Cost-optimal LLM routing using risk-based model selection.
=========================================================================

Routes each query to the cheapest model predicted to succeed, based on
the LLM Guard's KNN risk score. Falls back to more capable models only
when the risk score indicates likely failure.

Default cascade (Anthropic):
  Low risk   → claude-haiku-4-5-20251001  ($0.80 / 1M input)
  Medium risk → claude-sonnet-4-6          ($3.00 / 1M input)
  High risk  → claude-opus-4-6            ($15.00 / 1M input)

Cost savings depend on your traffic distribution. If 80% of queries are
low-risk, you pay Haiku rates on 80% of traffic — roughly 5–8x cheaper
than always using Sonnet.

Usage
-----
    from llm_guard import LLMGuard
    from llm_guard.router import SmartRouter

    guard = LLMGuard(api_key="sk-ant-...")
    guard.fit(correct_questions)

    router = SmartRouter(guard)
    result = router.route("What is 15% of 240?")

    print(result.answer)          # "36"
    print(result.model_used)      # "claude-haiku-4-5-20251001"
    print(result.confidence)      # "high"
    print(router.get_cost_stats())# {"total_cost_usd": 0.0003, "savings_vs_opus": 0.0042, ...}

Custom cascade
--------------
    router = SmartRouter(
        guard,
        cascade=[
            {"model": "claude-haiku-4-5-20251001",  "input_cost": 0.80,  "output_cost": 4.00},
            {"model": "claude-sonnet-4-6",           "input_cost": 3.00,  "output_cost": 15.00},
            {"model": "claude-opus-4-6",             "input_cost": 15.00, "output_cost": 75.00},
        ]
    )
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qppg.guard import QPPGLLMGuard as LLMGuard


# ── Default model cascade (Anthropic, March 2026 pricing) ─────────────────────

DEFAULT_CASCADE = [
    {
        "model":        "claude-haiku-4-5-20251001",
        "input_cost":   0.80,    # USD per 1M input tokens
        "output_cost":  4.00,    # USD per 1M output tokens
        "tier":         "low",   # used when risk <= low_threshold
    },
    {
        "model":        "claude-sonnet-4-6",
        "input_cost":   3.00,
        "output_cost":  15.00,
        "tier":         "medium",  # used when risk in (low, high]
    },
    {
        "model":        "claude-opus-4-6",
        "input_cost":   15.00,
        "output_cost":  75.00,
        "tier":         "high",    # used when risk > high_threshold
    },
]


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class RouterResult:
    """Result of a SmartRouter.route() call."""
    answer: str
    model_used: str
    confidence: str          # "high" | "medium" | "low"
    risk_score: float
    was_retried: bool = False
    tool_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: str = ""


# ── SmartRouter ────────────────────────────────────────────────────────────────

class SmartRouter:
    """
    Routes queries to the cheapest model predicted to succeed.

    Uses LLMGuard's KNN risk score to decide:
      Low risk   → cheapest model (default: Haiku)
      Medium risk → middle model (default: Sonnet)
      High risk  → most capable model (default: Opus)

    Parameters
    ----------
    guard : LLMGuard
        A fitted (or unfitted) guard instance. If unfitted, all queries
        route to the highest tier.
    cascade : list of dicts, optional
        Model cascade in ascending cost order. Each dict must have:
        model, input_cost ($/1M), output_cost ($/1M), tier (low/medium/high).
    """

    def __init__(
        self,
        guard: LLMGuard,
        cascade: Optional[List[Dict]] = None,
    ):
        self.guard = guard
        self.cascade = cascade or DEFAULT_CASCADE

        # Validate cascade has at least 2 tiers
        if len(self.cascade) < 2:
            raise ValueError("cascade must have at least 2 models (low + high tier)")

        # Map tier → model config
        self._tier_map = {c["tier"]: c for c in self.cascade}

        # Usage tracking
        self._calls_by_model: Dict[str, int] = {c["model"]: 0 for c in self.cascade}
        self._cost_by_model:  Dict[str, float] = {c["model"]: 0.0 for c in self.cascade}
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ── Core routing ───────────────────────────────────────────────────────────

    def route(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
    ) -> RouterResult:
        """
        Route a query to the appropriate model and return the result.

        The routing decision is made before the LLM is called:
          1. Embed question (uses guard's sentence-transformer)
          2. Compute KNN risk score
          3. Select model tier based on risk vs auto-calibrated thresholds
          4. Call the selected model
          5. Handle resource failures (max_tokens) with retry on same model

        Parameters
        ----------
        question : str
        system_prompt : str, optional
        max_tokens : int

        Returns
        -------
        RouterResult
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. "
                "Think step by step and give your final answer on the last line."
            )

        # ── 1. Score risk ──────────────────────────────────────────────────────
        risk = self.guard._compute_risk_score(question)
        low_t  = self.guard._risk_low_threshold  or 0.3
        high_t = self.guard._risk_high_threshold or 0.7

        # ── 2. Select model tier ───────────────────────────────────────────────
        if risk <= low_t:
            tier       = "low"
            confidence = "high"
        elif risk <= high_t:
            tier       = "medium"
            confidence = "medium"
        else:
            tier       = "high"
            confidence = "low"

        model_cfg = self._tier_map.get(tier, self.cascade[-1])
        model     = model_cfg["model"]

        # ── 3. Apply medium-risk tool if available ─────────────────────────────
        active_system = system_prompt
        tool_used = None
        if tier == "medium":
            emb  = self.guard._embed([question])
            tool = self.guard._match_tool(emb)
            if tool:
                active_system = (
                    f"{system_prompt}\n\n"
                    f"IMPORTANT: {tool.get('system_addition', '')}\n"
                    "Double-check your work before giving the final answer."
                )
                tool_used = tool.get("tool_name")
                self.guard._tool_usage[tool_used] = (
                    self.guard._tool_usage.get(tool_used, 0) + 1
                )
        elif tier == "high":
            active_system = (
                system_prompt
                + "\n\nIf you are not certain about any part of this, "
                "say so explicitly rather than guessing."
            )

        # ── 4. Call LLM with selected model ────────────────────────────────────
        import anthropic
        client = self.guard._get_client()

        def _call(sys, usr, mtok, temp=0.0):
            resp = client.messages.create(
                model=model,
                max_tokens=mtok,
                temperature=temp,
                system=sys,
                messages=[{"role": "user", "content": usr}],
            )
            self.guard.total_calls += 1
            self.guard.total_input_tokens  += resp.usage.input_tokens
            self.guard.total_output_tokens += resp.usage.output_tokens
            self._total_input_tokens  += resp.usage.input_tokens
            self._total_output_tokens += resp.usage.output_tokens
            cost = (
                resp.usage.input_tokens  * model_cfg["input_cost"]  / 1e6
                + resp.usage.output_tokens * model_cfg["output_cost"] / 1e6
            )
            self._cost_by_model[model] = self._cost_by_model.get(model, 0.0) + cost
            self._calls_by_model[model] = self._calls_by_model.get(model, 0) + 1
            return resp.content[0].text, resp.stop_reason, resp.usage

        text, stop_reason, usage = _call(active_system, question, max_tokens)

        # ── 5. Handle resource failure ─────────────────────────────────────────
        retried = False
        if stop_reason == "max_tokens":
            text, _, usage = _call(system_prompt, question, max_tokens * 2)
            retried = True
            tool_used = None

        cost = (
            usage.input_tokens  * model_cfg["input_cost"]  / 1e6
            + usage.output_tokens * model_cfg["output_cost"] / 1e6
        )

        return RouterResult(
            answer=text,
            model_used=model,
            confidence=confidence,
            risk_score=risk,
            was_retried=retried,
            tool_used=tool_used,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(cost, 6),
            raw_response=text,
        )

    # ── Batch routing ──────────────────────────────────────────────────────────

    def route_batch(
        self,
        questions: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
    ) -> List[RouterResult]:
        """
        Route a batch of queries, returning results in order.

        Queries are processed sequentially. Use this when you want to
        bulk-score and route without writing a loop yourself.
        """
        return [self.route(q, system_prompt, max_tokens) for q in questions]

    # ── Pre-score without calling LLM ─────────────────────────────────────────

    def score_only(self, question: str) -> Dict:
        """
        Return the routing decision without calling the LLM.

        Useful for dashboards, logging, or deciding whether to proceed.

        Returns dict with: risk_score, selected_model, confidence, tier.
        """
        risk  = self.guard._compute_risk_score(question)
        low_t  = self.guard._risk_low_threshold  or 0.3
        high_t = self.guard._risk_high_threshold or 0.7

        if risk <= low_t:
            tier, confidence = "low", "high"
        elif risk <= high_t:
            tier, confidence = "medium", "medium"
        else:
            tier, confidence = "high", "low"

        model_cfg = self._tier_map.get(tier, self.cascade[-1])
        return {
            "risk_score":     risk,
            "tier":           tier,
            "confidence":     confidence,
            "selected_model": model_cfg["model"],
            "estimated_cost_per_500_tokens": round(
                model_cfg["input_cost"] * 500 / 1e6
                + model_cfg["output_cost"] * 150 / 1e6, 7
            ),
        }

    # ── Cost stats ─────────────────────────────────────────────────────────────

    def get_cost_stats(self) -> Dict:
        """
        Return cost breakdown and estimated savings vs always using the top model.

        savings_vs_top_model is how much you saved compared to routing everything
        to the most capable (most expensive) model.
        """
        total_cost = sum(self._cost_by_model.values())
        total_calls = sum(self._calls_by_model.values())

        # Hypothetical cost if everything went to top model
        top = self.cascade[-1]
        hypothetical_top = (
            self._total_input_tokens  * top["input_cost"]  / 1e6
            + self._total_output_tokens * top["output_cost"] / 1e6
        )

        # Hypothetical cost if everything went to bottom model
        bottom = self.cascade[0]
        hypothetical_bottom = (
            self._total_input_tokens  * bottom["input_cost"]  / 1e6
            + self._total_output_tokens * bottom["output_cost"] / 1e6
        )

        return {
            "total_calls":          total_calls,
            "total_cost_usd":       round(total_cost, 6),
            "savings_vs_top_model": round(max(0.0, hypothetical_top - total_cost), 6),
            "overhead_vs_cheapest": round(max(0.0, total_cost - hypothetical_bottom), 6),
            "calls_by_model":       dict(self._calls_by_model),
            "cost_by_model":        {k: round(v, 6) for k, v in self._cost_by_model.items()},
            "model_distribution":   {
                k: round(v / total_calls, 3) if total_calls else 0.0
                for k, v in self._calls_by_model.items()
            },
        }
