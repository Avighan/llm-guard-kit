"""
CrewAI integration for llm-guard-kit.

Scores completed CrewAI task executions with AgentGuard.

Quick start
-----------
    from llm_guard.integrations.crewai import AgentGuardCrewCallback
    from llm_guard import AgentGuard

    guard    = AgentGuard()
    callback = AgentGuardCrewCallback(guard, on_alert=lambda r: print("ALERT", r))

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        step_callback=callback.on_step,
        task_callback=callback.on_task_end,
    )
    result = crew.kickoff()
    print(callback.last_result.confidence_tier)

Compatible with crewai >= 0.30.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class AgentGuardCrewCallback:
    """
    CrewAI callback that scores each completed task with AgentGuard.

    Use on_step as the step_callback and on_task_end as the task_callback.

    Parameters
    ----------
    guard : AgentGuard
    on_alert : callable, optional
        Called with ChainTrustResult when needs_alert=True.
    on_score : callable, optional
        Called with ChainTrustResult after every task.
    """

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
        self.last_result = None

    def on_step(self, step_output: Any) -> None:
        """
        Pass as step_callback=callback.on_step to Crew.

        CrewAI step_callback receives an AgentStep or dict with
        thought/action/observation.
        """
        # Handle crewai AgentStep or dict
        if hasattr(step_output, "thought"):
            step = {
                "thought":     str(step_output.thought or "")[:300],
                "action_type": str(getattr(step_output, "tool", "Action")),
                "action_arg":  str(getattr(step_output, "tool_input", ""))[:300],
                "observation": str(getattr(step_output, "result", ""))[:500],
            }
        elif isinstance(step_output, dict):
            step = {
                "thought":     str(step_output.get("thought", ""))[:300],
                "action_type": str(step_output.get("tool", step_output.get("action", "Action"))),
                "action_arg":  str(step_output.get("tool_input", step_output.get("action_input", "")))[:300],
                "observation": str(step_output.get("result", step_output.get("observation", "")))[:500],
            }
        else:
            step = {
                "thought": "", "action_type": "Action",
                "action_arg": str(step_output)[:300], "observation": "",
            }
        self._steps.append(step)

    def on_task_end(self, task_output: Any) -> None:
        """
        Pass as task_callback=callback.on_task_end to Crew.

        Scores the accumulated steps from this task.
        """
        # Extract the task description as the question if available
        if hasattr(task_output, "description"):
            self._question = str(task_output.description)[:500]
        elif hasattr(task_output, "task") and hasattr(task_output.task, "description"):
            self._question = str(task_output.task.description)[:500]

        final_answer = ""
        if hasattr(task_output, "raw"):
            final_answer = str(task_output.raw)
        elif hasattr(task_output, "result"):
            final_answer = str(task_output.result)
        else:
            final_answer = str(task_output)

        if not self._steps:
            # No steps captured — add a minimal synthetic step
            self._steps = [{
                "thought": "", "action_type": "Finish",
                "action_arg": final_answer[:300], "observation": "",
            }]

        result = self._guard.score_chain(
            question=self._question or "(crewai task)",
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

    def reset(self) -> None:
        """Clear accumulated steps between tasks if needed."""
        self._steps = []
        self._question = ""
