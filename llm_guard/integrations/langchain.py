"""
LangChain integration for llm-guard-kit.

Plug in with 2 lines — AgentGuard scores every completed agent chain
automatically, calls your on_alert handler when risk >= threshold.

Quick start
-----------
    from llm_guard.integrations.langchain import AgentGuardCallback
    from llm_guard import AgentGuard

    guard    = AgentGuard()
    callback = AgentGuardCallback(guard, on_alert=lambda r: print("ALERT", r))

    # Pass to any LangChain agent:
    agent.invoke({"input": question}, config={"callbacks": [callback]})

    # Or attach at agent construction:
    agent = create_react_agent(llm, tools, prompt, callbacks=[callback])

    # Access last score:
    print(callback.last_result.confidence_tier)   # HIGH / MEDIUM / LOW
    print(callback.last_result.risk_score)        # 0.0–1.0
    print(callback.last_result.needs_alert)       # True when risk >= 0.70

Compatible with LangChain >= 0.1 (langchain-core).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import AgentAction, AgentFinish
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False
        # Stub so the module is importable without langchain installed
        class BaseCallbackHandler:  # type: ignore
            pass
        AgentAction = AgentFinish = None  # type: ignore


class AgentGuardCallback(BaseCallbackHandler if _LANGCHAIN_AVAILABLE else object):
    """
    LangChain callback that scores every completed agent chain with AgentGuard.

    Captures all agent steps (thought → action → observation) automatically.
    When the chain finishes, scores it and calls on_alert if risk >= threshold.

    Parameters
    ----------
    guard : AgentGuard
        A configured AgentGuard instance (behavioral, judge, or local verifier).
    on_alert : callable, optional
        Called with ChainTrustResult when needs_alert=True.
        Signature: on_alert(result: ChainTrustResult) -> None
    on_score : callable, optional
        Called with ChainTrustResult after every chain, regardless of alert.
        Signature: on_score(result: ChainTrustResult) -> None
    tag : str, optional
        Label attached to result for routing (e.g. agent name or domain).
    """

    def __init__(
        self,
        guard,
        on_alert: Optional[Callable] = None,
        on_score: Optional[Callable] = None,
        tag: str = "",
    ):
        if _LANGCHAIN_AVAILABLE:
            super().__init__()
        self._guard    = guard
        self._on_alert = on_alert
        self._on_score = on_score
        self._tag      = tag
        self._reset()

    def _reset(self):
        self._steps: List[Dict] = []
        self._question: str = ""
        self._pending_thought: str = ""
        self._pending_action_type: str = ""
        self._pending_action_arg: str = ""
        self.last_result = None

    # ── Chain entry (captures the question) ──────────────────────────────────

    def on_chain_start(self, serialized, inputs, **kwargs):
        if not self._question:
            q = inputs.get("input") or inputs.get("question") or inputs.get("query", "")
            self._question = str(q)[:500]

    def on_agent_action(self, action, run_id=None, **kwargs):
        """Capture each tool call (thought + action)."""
        if not _LANGCHAIN_AVAILABLE:
            return
        # Parse thought from the action log
        log = getattr(action, "log", "") or ""
        thought = ""
        for line in log.splitlines():
            stripped = line.strip()
            if stripped.startswith("Thought:"):
                thought = stripped[8:].strip()
                break
        if not thought:
            # Fall back: everything before the first Action: line
            parts = re.split(r"\nAction:", log, maxsplit=1)
            thought = parts[0].strip()

        self._pending_thought = thought
        self._pending_action_type = getattr(action, "tool", "Search")
        inp = getattr(action, "tool_input", "")
        if isinstance(inp, dict):
            self._pending_action_arg = str(inp.get("query", inp.get("input", str(inp))))
        else:
            self._pending_action_arg = str(inp)

    def on_tool_end(self, output, run_id=None, **kwargs):
        """Complete the step with the tool observation."""
        self._steps.append({
            "thought":     self._pending_thought,
            "action_type": self._pending_action_type or "Search",
            "action_arg":  self._pending_action_arg,
            "observation": str(output)[:500],
        })
        self._pending_thought = ""
        self._pending_action_type = ""
        self._pending_action_arg = ""

    def on_tool_error(self, error, run_id=None, **kwargs):
        """Record tool errors as observations so the chain is still scoreable."""
        self._steps.append({
            "thought":     self._pending_thought,
            "action_type": self._pending_action_type or "Search",
            "action_arg":  self._pending_action_arg,
            "observation": f"[TOOL ERROR] {error}",
        })
        self._pending_thought = ""
        self._pending_action_type = ""
        self._pending_action_arg = ""

    def on_agent_finish(self, finish, run_id=None, **kwargs):
        """Score the completed chain."""
        if not _LANGCHAIN_AVAILABLE:
            return
        output = getattr(finish, "return_values", {})
        final_answer = output.get("output", "") if isinstance(output, dict) else str(output)

        # Add the Finish step
        self._steps.append({
            "thought":     "",
            "action_type": "Finish",
            "action_arg":  str(final_answer)[:300],
            "observation": "",
        })

        result = self._guard.score_chain(
            question=self._question or "(unknown)",
            steps=self._steps,
            final_answer=str(final_answer),
        )
        # Attach tag for routing
        if self._tag:
            result.__dict__["tag"] = self._tag

        self.last_result = result

        if self._on_score:
            self._on_score(result)
        if result.needs_alert and self._on_alert:
            self._on_alert(result)

        self._reset()

    def on_chain_end(self, outputs, run_id=None, **kwargs):
        """Fallback: score if agent_finish wasn't triggered (e.g. plain chains)."""
        # Only fire if we have steps but haven't scored yet (no agent_finish)
        if self._steps and self.last_result is None:
            final = ""
            if isinstance(outputs, dict):
                final = outputs.get("output", outputs.get("text", str(outputs)))
            result = self._guard.score_chain(
                question=self._question or "(unknown)",
                steps=self._steps,
                final_answer=str(final),
            )
            self.last_result = result
            if self._on_score:
                self._on_score(result)
            if result.needs_alert and self._on_alert:
                self._on_alert(result)
            self._reset()


if not _LANGCHAIN_AVAILABLE:
    # Override with a helpful error class so import works but usage raises clearly
    class AgentGuardCallback:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "langchain-core is required for AgentGuardCallback. "
                "Install it with: pip install langchain-core"
            )
