"""
GuardClient — One-line Python SDK for llm-guard-kit.

Supports two modes with the same API surface:
  • "saas"  (default) — HTTP calls to the LLMGuard API
  • "local"            — Calls AgentGuard directly, zero network (on-premises)

Quick start (SaaS)
------------------
    from llm_guard import GuardClient

    guard = GuardClient(api_key="sk_...")          # or set LLMGUARD_API_KEY env var
    result = guard.score(question, steps, final_answer)
    if result.needs_alert:
        alert(result.failure_mode)

Quick start (on-premises / local library)
-----------------------------------------
    guard = GuardClient(mode="local")             # no network, no API key needed
    result = guard.score(question, steps, final_answer)

Auto-instrumentation decorator
-------------------------------
    @guard.watch                                  # wraps any agent function
    def run_agent(question: str) -> dict:
        ...
        return {"question": q, "steps": steps, "final_answer": ans}

    run_agent("What is the GDP of France?")       # score is computed automatically
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    risk_score:            float
    confidence_tier:       str          # "HIGH" | "MEDIUM" | "LOW"
    needs_alert:           bool
    failure_mode:          Optional[str]
    judge_label:           Optional[str]
    step_count:            int
    behavioral_score:      float
    behavioral_components: Dict[str, float] = field(default_factory=dict)
    latency_ms:            float = 0.0
    domain:                str = "default"

    @property
    def is_high_risk(self) -> bool:
        return self.risk_score >= 0.70

    def __repr__(self) -> str:
        alert = " [ALERT]" if self.needs_alert else ""
        return (
            f"ScoreResult(risk={self.risk_score:.2%}, tier={self.confidence_tier}, "
            f"steps={self.step_count}{alert})"
        )


@dataclass
class MonitorResult:
    risk_score:        float
    risk:              str
    confidence:        str
    predicted_outcome: str
    step_index:        int
    failure_mode:      Optional[str]

    @property
    def should_abort(self) -> bool:
        return self.risk == "high"


# ── GuardClient ───────────────────────────────────────────────────────────────

class GuardClient:
    """
    One-line SDK for real-time agent chain scoring.

    Parameters
    ----------
    api_key : str, optional
        Bearer token (sk_...). Falls back to LLMGUARD_API_KEY env var.
        Not required in mode="local".
    base_url : str, optional
        API base URL. Falls back to LLMGUARD_BASE_URL env var.
        Default: http://localhost:8000 (change to your cloud URL in prod).
    mode : "saas" | "local"
        "saas"  — HTTP calls to the API (default)
        "local" — Calls AgentGuard directly, no network needed
    timeout : float
        HTTP request timeout in seconds (default 30).
    on_alert : callable, optional
        Called with (ScoreResult) whenever needs_alert is True.
        Useful for fire-and-forget notifications without polling.
    """

    _DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
        mode:     str = "saas",
        timeout:  float = 30.0,
        on_alert: Optional[Callable[[ScoreResult], None]] = None,
    ):
        if mode not in ("saas", "local"):
            raise ValueError(f"mode must be 'saas' or 'local', got {mode!r}")

        self.mode     = mode
        self.timeout  = timeout
        self.on_alert = on_alert

        if mode == "saas":
            self.api_key  = api_key or os.environ.get("LLMGUARD_API_KEY", "")
            self.base_url = (base_url or os.environ.get("LLMGUARD_BASE_URL", self._DEFAULT_BASE_URL)).rstrip("/")
            self._agent_guard = None  # lazy
        else:
            # Local mode: import AgentGuard (requires sentence-transformers)
            from llm_guard.agent_guard import AgentGuard  # noqa: PLC0415
            self._agent_guard = AgentGuard()
            self.api_key  = ""
            self.base_url = ""

    # ── Internal HTTP helper ──────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("httpx is required for SaaS mode: pip install httpx") from exc

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = httpx.post(
            f"{self.base_url}{path}",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Step normalisation helper ─────────────────────────────────────────────

    @staticmethod
    def _normalise_steps(steps: List[Any]) -> List[Dict[str, str]]:
        """Accept dicts or objects with thought/action/observation attributes."""
        out = []
        for s in steps:
            if isinstance(s, dict):
                out.append({
                    "thought":     s.get("thought", ""),
                    "action_type": s.get("action_type") or s.get("action", "Action"),
                    "action_arg":  s.get("action_arg")  or s.get("action", ""),
                    "observation": s.get("observation", ""),
                })
            else:
                # assume object with attributes
                out.append({
                    "thought":     getattr(s, "thought", ""),
                    "action_type": getattr(s, "action_type", None) or getattr(s, "action", "Action"),
                    "action_arg":  getattr(s, "action_arg",  None) or getattr(s, "action", ""),
                    "observation": getattr(s, "observation", ""),
                })
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        question:     str,
        steps:        List[Any],
        final_answer: str,
        domain:       str = "default",
        finished:     bool = True,
    ) -> ScoreResult:
        """Score a completed ReAct chain.

        Parameters
        ----------
        question     : The original user question.
        steps        : List of agent steps. Each step can be a dict with keys
                       thought/action_type/action_arg/observation, or any object
                       with those attributes (LangChain AgentAction, etc.).
        final_answer : The agent's final answer string.
        domain       : Tag for grouping/guardrails (e.g. "customer_service").
        finished     : Whether the chain completed (False = in-progress estimate).

        Returns
        -------
        ScoreResult with risk_score, confidence_tier, needs_alert, failure_mode.
        """
        norm_steps = self._normalise_steps(steps)

        if self.mode == "local":
            t0 = time.time()
            r  = self._agent_guard.score_chain(
                question=question, steps=norm_steps, final_answer=final_answer, finished=finished
            )
            return ScoreResult(
                risk_score            = r.risk_score,
                confidence_tier       = r.confidence_tier,
                needs_alert           = r.needs_alert,
                failure_mode          = r.failure_mode,
                judge_label           = r.judge_label,
                step_count            = r.step_count,
                behavioral_score      = r.behavioral_score,
                behavioral_components = r.behavioral_components,
                latency_ms            = (time.time() - t0) * 1000,
                domain                = domain,
            )

        # SaaS mode
        data = self._post("/v1/score_chain", {
            "question":     question,
            "steps":        norm_steps,
            "final_answer": final_answer,
            "domain":       domain,
            "finished":     finished,
        })
        result = ScoreResult(
            risk_score            = data["risk_score"],
            confidence_tier       = data["confidence_tier"],
            needs_alert           = data["needs_alert"],
            failure_mode          = data.get("failure_mode"),
            judge_label           = data.get("judge_label"),
            step_count            = data["step_count"],
            behavioral_score      = data["behavioral_score"],
            behavioral_components = data.get("behavioral_components", {}),
            latency_ms            = data.get("latency_ms", 0.0),
            domain                = domain,
        )
        if result.needs_alert and self.on_alert:
            self.on_alert(result)
        return result

    def score_batch(
        self,
        chains: List[Dict[str, Any]],
        domain: str = "default",
    ) -> List[ScoreResult]:
        """Score up to 100 chains in one call (SaaS) or sequentially (local).

        Each entry in `chains` must be a dict with keys:
          question, steps, final_answer, and optionally domain/finished.
        """
        if self.mode == "local":
            return [
                self.score(
                    question=c["question"],
                    steps=c["steps"],
                    final_answer=c["final_answer"],
                    domain=c.get("domain", domain),
                    finished=c.get("finished", True),
                )
                for c in chains
            ]

        payload_chains = []
        for c in chains:
            payload_chains.append({
                "question":     c["question"],
                "steps":        self._normalise_steps(c["steps"]),
                "final_answer": c["final_answer"],
                "domain":       c.get("domain", domain),
                "finished":     c.get("finished", True),
            })

        data = self._post("/v1/score_chain/batch", {"chains": payload_chains, "domain": domain})
        results = []
        for item in data["results"]:
            r = ScoreResult(
                risk_score            = item["risk_score"],
                confidence_tier       = item["confidence_tier"],
                needs_alert           = item["needs_alert"],
                failure_mode          = item.get("failure_mode"),
                judge_label           = item.get("judge_label"),
                step_count            = item["step_count"],
                behavioral_score      = item["behavioral_score"],
                behavioral_components = item.get("behavioral_components", {}),
                latency_ms            = item.get("latency_ms", 0.0),
                domain                = domain,
            )
            if r.needs_alert and self.on_alert:
                self.on_alert(r)
            results.append(r)
        return results

    def monitor(
        self,
        question:       str,
        steps_so_far:   List[Any],
        current_action: str,
    ) -> MonitorResult:
        """Score a single step mid-chain for early abort decisions.

        Call inside your agent loop after each tool call. If
        result.should_abort is True, consider stopping and escalating.
        """
        norm = self._normalise_steps(steps_so_far)

        if self.mode == "local":
            r = self._agent_guard.monitor_step(
                question=question,
                steps_so_far=norm,
                current_action=current_action,
            )
            return MonitorResult(
                risk_score=r.risk_score, risk=r.risk, confidence=r.confidence,
                predicted_outcome=r.predicted_outcome, step_index=r.step_index,
                failure_mode=r.failure_mode,
            )

        data = self._post("/v1/monitor_step", {
            "question":       question,
            "steps_so_far":   norm,
            "current_action": current_action,
        })
        return MonitorResult(
            risk_score        = data["risk_score"],
            risk              = data["risk"],
            confidence        = data["confidence"],
            predicted_outcome = data["predicted_outcome"],
            step_index        = data["step_index"],
            failure_mode      = data.get("failure_mode"),
        )

    # ── @watch decorator ──────────────────────────────────────────────────────

    def watch(self, func: Callable) -> Callable:
        """Decorator that auto-scores any agent function.

        The decorated function must return a dict with keys:
          question, steps, final_answer
        (and optionally: domain)

        Example
        -------
            @guard.watch
            def run_agent(question: str) -> dict:
                steps, answer = my_react_loop(question)
                return {"question": question, "steps": steps, "final_answer": answer}

            # Score is computed automatically; raises on alert if on_alert raises
            result = run_agent("What year was Paris founded?")
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if not isinstance(output, dict):
                return output  # not an agent output — pass through

            q      = output.get("question", "")
            steps  = output.get("steps", [])
            ans    = output.get("final_answer", output.get("answer", ""))
            domain = output.get("domain", "default")

            score_result = self.score(question=q, steps=steps, final_answer=ans, domain=domain)
            output["_guard"] = score_result
            return output

        return wrapper

    # ── Context manager (optional) ────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
