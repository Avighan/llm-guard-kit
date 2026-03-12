"""
Langfuse Integration for llm-guard-kit
=======================================
Attaches llm-guard risk scores to Langfuse traces as scores.

Usage
-----
    from llm_guard.integrations.langfuse_integration import LangfuseGuard

    guard = LangfuseGuard(
        langfuse_public_key="pk-lf-...",
        langfuse_secret_key="sk-lf-...",
        langfuse_host="https://cloud.langfuse.com",  # optional
    )

    # Score a chain and attach the risk score to a Langfuse trace
    result = guard.score_and_trace(
        trace_id="trace-abc123",
        question="What is the capital of France?",
        steps=[...],
        final_answer="Paris",
    )
    # Langfuse now shows: llm_guard_risk=0.23, tier=HIGH, needs_alert=False

    # Context-manager style — auto-traces every score_chain() call
    with guard.create_traced_guard("trace-abc123") as tg:
        result = tg.score_chain(question, steps, final_answer)

    # LangChain callback (requires langchain-core)
    from llm_guard.integrations.langfuse_integration import LangfuseGuardCallback
    callback = LangfuseGuardCallback(langfuse_guard=guard)
    agent.invoke({"input": question}, config={"callbacks": [callback]})

Requirements
------------
    pip install langfuse               # Langfuse SDK
    pip install langchain-core         # optional, for LangfuseGuardCallback
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Langfuse import — deferred so the module is importable without langfuse
# ---------------------------------------------------------------------------

_LANGFUSE_AVAILABLE = False
_langfuse_mod = None


def _ensure_langfuse():
    global _LANGFUSE_AVAILABLE, _langfuse_mod
    if _LANGFUSE_AVAILABLE:
        return
    try:
        import langfuse as _lf
        _langfuse_mod = _lf
        _LANGFUSE_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "langfuse is required for LangfuseGuard. "
            "Install it with: pip install langfuse"
        )


# ---------------------------------------------------------------------------
# LangfuseGuard
# ---------------------------------------------------------------------------

class LangfuseGuard:
    """
    Wraps AgentGuard and posts risk scores to Langfuse as trace scores.

    Parameters
    ----------
    langfuse_public_key : str
        Langfuse project public key (pk-lf-...).
    langfuse_secret_key : str
        Langfuse project secret key (sk-lf-...).
    langfuse_host : str, optional
        Langfuse server URL. Defaults to https://cloud.langfuse.com.
    guard : AgentGuard, optional
        Pre-configured AgentGuard instance. Created with defaults if omitted.
    score_name : str
        Name of the score written to Langfuse. Default "llm_guard_risk".
    flush_on_score : bool
        If True, call langfuse.flush() after every score to ensure delivery.
        Useful in short-lived processes. Default False.
    """

    def __init__(
        self,
        langfuse_public_key: str,
        langfuse_secret_key: str,
        langfuse_host: Optional[str] = None,
        guard=None,
        score_name: str = "llm_guard_risk",
        flush_on_score: bool = False,
    ):
        self._pk = langfuse_public_key
        self._sk = langfuse_secret_key
        self._host = langfuse_host
        self._score_name = score_name
        self._flush_on_score = flush_on_score
        self._lf_client = None  # created lazily

        if guard is not None:
            self._guard = guard
        else:
            from llm_guard import AgentGuard  # local import to avoid circular deps
            self._guard = AgentGuard()

    # ------------------------------------------------------------------
    # Langfuse client — created on first use so the constructor never
    # raises ImportError (allows guarded import at module level).
    # ------------------------------------------------------------------

    def _client(self):
        """Return (and lazily create) the Langfuse SDK client."""
        if self._lf_client is None:
            _ensure_langfuse()
            kwargs: Dict[str, Any] = {
                "public_key": self._pk,
                "secret_key": self._sk,
            }
            if self._host:
                kwargs["host"] = self._host
            self._lf_client = _langfuse_mod.Langfuse(**kwargs)  # type: ignore[union-attr]
        return self._lf_client

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def score_and_trace(
        self,
        trace_id: str,
        question: str,
        steps: List[Dict],
        final_answer: str,
        observation_id: Optional[str] = None,
    ):
        """
        Score a completed agent chain and attach the result to a Langfuse trace.

        Parameters
        ----------
        trace_id : str
            ID of the Langfuse trace to attach the score to.
        question : str
            The question / task given to the agent.
        steps : list of dict
            Agent reasoning steps (thought/action/observation dicts).
        final_answer : str
            The agent's final output.
        observation_id : str, optional
            If provided, attaches the score to a specific observation span
            inside the trace rather than the trace root.

        Returns
        -------
        ChainTrustResult
            The scoring result from AgentGuard.
        """
        result = self._guard.score_chain(
            question=question,
            steps=steps,
            final_answer=final_answer,
        )

        self._post_score(trace_id, result, observation_id=observation_id)
        return result

    def _post_score(self, trace_id: str, result, observation_id: Optional[str] = None):
        """Post a ChainTrustResult as a Langfuse score."""
        client = self._client()
        tier = result.confidence_tier  # HIGH / MEDIUM / LOW
        needs_alert = result.needs_alert
        failure_mode = result.failure_mode or "none"
        comment = (
            f"tier={tier} needs_alert={needs_alert} "
            f"behavioral={result.behavioral_score:.3f} "
            f"failure_mode={failure_mode}"
        )

        score_kwargs: Dict[str, Any] = {
            "trace_id": trace_id,
            "name": self._score_name,
            "value": round(result.risk_score, 4),
            "data_type": "NUMERIC",
            "comment": comment,
        }
        if observation_id:
            score_kwargs["observation_id"] = observation_id

        try:
            client.score(**score_kwargs)
            if self._flush_on_score:
                client.flush()
        except Exception as exc:
            logger.warning("LangfuseGuard: failed to post score — %s", exc)

    # ------------------------------------------------------------------
    # Context-manager helper
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def create_traced_guard(self, trace_id: str, observation_id: Optional[str] = None):
        """
        Context manager that wraps score_chain() and auto-traces to Langfuse.

        Usage::

            with guard.create_traced_guard("trace-abc123") as tg:
                result = tg.score_chain(question, steps, final_answer)
            # Score is automatically posted to trace-abc123 on exit.
        """
        proxy = _TracedGuardProxy(
            guard=self._guard,
            langfuse_guard=self,
            trace_id=trace_id,
            observation_id=observation_id,
        )
        try:
            yield proxy
        finally:
            # Flush if requested and a result was produced
            if self._flush_on_score and self._lf_client is not None:
                try:
                    self._lf_client.flush()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Passthrough convenience methods
    # ------------------------------------------------------------------

    def score_chain(self, question: str, steps: List[Dict], final_answer: str):
        """Score without tracing (plain AgentGuard passthrough)."""
        return self._guard.score_chain(
            question=question, steps=steps, final_answer=final_answer
        )

    def flush(self):
        """Flush all pending Langfuse events. Call before process exit."""
        if self._lf_client is not None:
            try:
                self._lf_client.flush()
            except Exception as exc:
                logger.warning("LangfuseGuard.flush() failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal proxy used by create_traced_guard()
# ---------------------------------------------------------------------------

class _TracedGuardProxy:
    """Thin proxy returned inside the create_traced_guard() context manager."""

    def __init__(self, guard, langfuse_guard: LangfuseGuard, trace_id: str, observation_id: Optional[str]):
        self._guard = guard
        self._lf_guard = langfuse_guard
        self._trace_id = trace_id
        self._observation_id = observation_id
        self.last_result = None

    def score_chain(self, question: str, steps: List[Dict], final_answer: str):
        result = self._guard.score_chain(
            question=question, steps=steps, final_answer=final_answer
        )
        self.last_result = result
        self._lf_guard._post_score(
            self._trace_id, result, observation_id=self._observation_id
        )
        return result

    def __getattr__(self, name: str):
        """Delegate any other AgentGuard method calls transparently."""
        return getattr(self._guard, name)


# ---------------------------------------------------------------------------
# LangfuseGuardCallback — LangChain callback (optional)
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _BaseCallback
    from langchain_core.agents import AgentAction as _AgentAction
    from langchain_core.agents import AgentFinish as _AgentFinish
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _BaseCallback  # type: ignore
        from langchain.schema import AgentAction as _AgentAction, AgentFinish as _AgentFinish  # type: ignore
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False
        _BaseCallback = object  # type: ignore
        _AgentAction = _AgentFinish = None  # type: ignore


class LangfuseGuardCallback(_BaseCallback):  # type: ignore[valid-type]
    """
    LangChain callback that scores every completed agent chain and posts
    risk scores to Langfuse automatically.

    Parameters
    ----------
    langfuse_guard : LangfuseGuard
        A configured LangfuseGuard instance.
    trace_id : str, optional
        Langfuse trace ID. If omitted, uses the LangChain run_id as trace_id.
    on_alert : callable, optional
        Called with ChainTrustResult when needs_alert=True.

    Raises
    ------
    ImportError
        If langchain-core is not installed.
    """

    def __init__(
        self,
        langfuse_guard: LangfuseGuard,
        trace_id: Optional[str] = None,
        on_alert=None,
    ):
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for LangfuseGuardCallback. "
                "Install it with: pip install langchain-core"
            )
        super().__init__()
        self._lf_guard = langfuse_guard
        self._trace_id = trace_id
        self._on_alert = on_alert
        self._steps: List[Dict] = []
        self._question: str = ""
        self._pending_thought: str = ""
        self._pending_action_type: str = ""
        self._pending_action_arg: str = ""
        self.last_result = None

    # -- LangChain callback hooks ------------------------------------------

    def on_chain_start(self, serialized, inputs, **kwargs):
        if not self._question:
            q = inputs.get("input") or inputs.get("question") or inputs.get("query", "")
            self._question = str(q)[:500]

    def on_agent_action(self, action, run_id=None, **kwargs):
        if not _LANGCHAIN_AVAILABLE:
            return
        import re as _re
        log = getattr(action, "log", "") or ""
        thought = ""
        for line in log.splitlines():
            stripped = line.strip()
            if stripped.startswith("Thought:"):
                thought = stripped[8:].strip()
                break
        if not thought:
            parts = _re.split(r"\nAction:", log, maxsplit=1)
            thought = parts[0].strip()
        self._pending_thought = thought
        self._pending_action_type = getattr(action, "tool", "Search")
        inp = getattr(action, "tool_input", "")
        if isinstance(inp, dict):
            self._pending_action_arg = str(inp.get("query", inp.get("input", str(inp))))
        else:
            self._pending_action_arg = str(inp)

    def on_tool_end(self, output, run_id=None, **kwargs):
        self._steps.append({
            "thought":     self._pending_thought,
            "action_type": self._pending_action_type or "Search",
            "action_arg":  self._pending_action_arg,
            "observation": str(output)[:500],
        })
        self._pending_thought = self._pending_action_type = self._pending_action_arg = ""

    def on_tool_error(self, error, run_id=None, **kwargs):
        self._steps.append({
            "thought":     self._pending_thought,
            "action_type": self._pending_action_type or "Search",
            "action_arg":  self._pending_action_arg,
            "observation": f"[TOOL ERROR] {error}",
        })
        self._pending_thought = self._pending_action_type = self._pending_action_arg = ""

    def on_agent_finish(self, finish, run_id=None, **kwargs):
        if not _LANGCHAIN_AVAILABLE:
            return
        output = getattr(finish, "return_values", {})
        final_answer = output.get("output", "") if isinstance(output, dict) else str(output)
        self._steps.append({
            "thought": "", "action_type": "Finish",
            "action_arg": str(final_answer)[:300], "observation": "",
        })
        # Determine trace ID: explicit > LangChain run_id > fallback
        effective_trace_id = self._trace_id or (str(run_id) if run_id else "unknown")
        result = self._lf_guard.score_and_trace(
            trace_id=effective_trace_id,
            question=self._question or "(unknown)",
            steps=self._steps,
            final_answer=str(final_answer),
        )
        self.last_result = result
        if result.needs_alert and self._on_alert:
            self._on_alert(result)
        self._reset()

    def on_chain_end(self, outputs, run_id=None, **kwargs):
        if self._steps and self.last_result is None:
            final = ""
            if isinstance(outputs, dict):
                final = outputs.get("output", outputs.get("text", str(outputs)))
            effective_trace_id = self._trace_id or (str(run_id) if run_id else "unknown")
            result = self._lf_guard.score_and_trace(
                trace_id=effective_trace_id,
                question=self._question or "(unknown)",
                steps=self._steps,
                final_answer=str(final),
            )
            self.last_result = result
            if result.needs_alert and self._on_alert:
                self._on_alert(result)
            self._reset()

    def _reset(self):
        self._steps = []
        self._question = ""
        self._pending_thought = self._pending_action_type = self._pending_action_arg = ""
        self.last_result = None


# If LangChain not available, replace class with a helpful error stub
if not _LANGCHAIN_AVAILABLE:
    class LangfuseGuardCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "langchain-core is required for LangfuseGuardCallback. "
                "Install it with: pip install langchain-core"
            )
