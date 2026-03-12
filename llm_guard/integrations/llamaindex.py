"""
LlamaIndex integration for llm-guard-kit.

Plug in with 2 lines — AgentGuard scores every completed agent run.

Quick start (LlamaIndex >= 0.10)
---------------------------------
    from llm_guard.integrations.llamaindex import AgentGuardEventHandler
    from llm_guard import AgentGuard

    guard   = AgentGuard()
    handler = AgentGuardEventHandler(guard, on_alert=lambda r: print("ALERT", r))

    # Attach at query engine or agent level:
    from llama_index.core import Settings
    Settings.callback_manager.add_event_handler(handler)

    # Or inject into a specific agent:
    from llama_index.core.callbacks import CallbackManager
    agent = ReActAgent.from_tools(tools, callback_manager=CallbackManager([handler]))

    # Access last result:
    print(handler.last_result.confidence_tier)

Compatible with llama-index-core >= 0.10.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    from llama_index.core.callbacks import CBEventType, EventPayload
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler as LlamaBaseHandler
    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    LlamaBaseHandler = object  # type: ignore
    CBEventType = EventPayload = None  # type: ignore


class AgentGuardEventHandler(LlamaBaseHandler if _LLAMAINDEX_AVAILABLE else object):
    """
    LlamaIndex callback handler that scores completed ReAct agent runs.

    Parameters
    ----------
    guard : AgentGuard
    on_alert : callable, optional
        Called with ChainTrustResult when needs_alert=True.
    on_score : callable, optional
        Called after every completed run.
    """

    event_starts_to_ignore: List = []
    event_ends_to_ignore: List = []

    def __init__(
        self,
        guard,
        on_alert: Optional[Callable] = None,
        on_score: Optional[Callable] = None,
    ):
        self._guard    = guard
        self._on_alert = on_alert
        self._on_score = on_score
        self._steps: List[Dict] = []
        self._question: str = ""
        self._pending: Dict = {}
        self.last_result = None

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._steps = []
        self._question = ""
        self._pending = {}
        self.last_result = None

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_start(
        self,
        event_type,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs,
    ) -> str:
        if not _LLAMAINDEX_AVAILABLE or payload is None:
            return event_id

        # Capture the initial query
        if event_type == CBEventType.QUERY:
            self._question = str(payload.get(EventPayload.QUERY_STR, ""))[:500]

        # Capture tool/function call start
        if event_type in (CBEventType.FUNCTION_CALL, CBEventType.TOOL):
            self._pending = {
                "action_type": str(payload.get(EventPayload.TOOL, {}).get("name", "Tool")),
                "action_arg":  str(payload.get(EventPayload.FUNCTION_CALL, ""))[:300],
                "thought":     "",
            }

        return event_id

    def on_event_end(
        self,
        event_type,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs,
    ) -> None:
        if not _LLAMAINDEX_AVAILABLE or payload is None:
            return

        # Complete a tool step
        if event_type in (CBEventType.FUNCTION_CALL, CBEventType.TOOL) and self._pending:
            observation = str(payload.get(EventPayload.FUNCTION_OUTPUT, ""))[:500]
            step = dict(self._pending)
            step["observation"] = observation
            self._steps.append(step)
            self._pending = {}

        # Score on agent completion
        if event_type == CBEventType.AGENT_STEP:
            response = payload.get(EventPayload.RESPONSE, None)
            if response is not None:
                final_answer = str(getattr(response, "response", response))
                self._score_and_notify(final_answer)

        # Also handle query end as a fallback for non-agent queries
        if event_type == CBEventType.QUERY and self._steps:
            response = payload.get(EventPayload.RESPONSE, None)
            if response is not None:
                final_answer = str(getattr(response, "response", response))
                self._score_and_notify(final_answer)

    def _score_and_notify(self, final_answer: str) -> None:
        if not self._steps:
            return
        result = self._guard.score_chain(
            question=self._question or "(unknown)",
            steps=self._steps,
            final_answer=final_answer,
        )
        self.last_result = result
        self._steps = []
        self._question = ""

        if self._on_score:
            self._on_score(result)
        if result.needs_alert and self._on_alert:
            self._on_alert(result)


if not _LLAMAINDEX_AVAILABLE:
    class AgentGuardEventHandler:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "llama-index-core is required for AgentGuardEventHandler. "
                "Install it with: pip install llama-index-core"
            )
