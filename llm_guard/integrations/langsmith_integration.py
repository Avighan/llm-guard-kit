"""
LangSmith Integration for llm-guard-kit
=========================================
Register llm-guard as a LangSmith evaluator for run_on_dataset.

Usage
-----
    from llm_guard.integrations.langsmith_integration import LangSmithGuardEvaluator

    evaluator = LangSmithGuardEvaluator()

    # Pass directly to LangSmith's run_on_dataset
    from langsmith import Client
    client = Client()
    results = client.run_on_dataset(
        dataset_name="my-agent-dataset",
        llm_or_chain_factory=my_chain,
        evaluators=[evaluator],
    )

    # Or evaluate a pandas DataFrame of runs offline
    import pandas as pd
    runs_df = pd.DataFrame([...])  # columns: question, steps, final_answer
    summary = evaluator.evaluate_dataset(runs_df)

Requirements
------------
    pip install langsmith        # LangSmith SDK
    pip install pandas           # optional, for evaluate_dataset()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy LangSmith import
# ---------------------------------------------------------------------------

_LANGSMITH_AVAILABLE = False
_EvaluationResult = None


def _ensure_langsmith():
    global _LANGSMITH_AVAILABLE, _EvaluationResult
    if _LANGSMITH_AVAILABLE:
        return
    try:
        from langsmith.schemas import EvaluationResult as _ER
        _EvaluationResult = _ER
        _LANGSMITH_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "langsmith is required for LangSmithGuardEvaluator. "
            "Install it with: pip install langsmith"
        )


# ---------------------------------------------------------------------------
# LangSmithGuardEvaluator
# ---------------------------------------------------------------------------

class LangSmithGuardEvaluator:
    """
    LangSmith evaluator that scores agent chains with AgentGuard.

    Implements the LangSmith evaluator protocol — pass an instance directly
    to ``Client.run_on_dataset(evaluators=[...])``.

    Parameters
    ----------
    guard : AgentGuard, optional
        Pre-configured AgentGuard instance. Defaults are used if omitted.
    threshold : float
        Risk threshold above which the evaluation is considered a failure.
        Default 0.65. Matches the llm-guard MEDIUM/LOW boundary.
    key : str
        LangSmith evaluation key name. Default "llm_guard_risk".

    Examples
    --------
    ::

        evaluator = LangSmithGuardEvaluator(threshold=0.65)

        # LangSmith run_on_dataset
        results = client.run_on_dataset(
            dataset_name="qa-agent-v2",
            llm_or_chain_factory=my_chain,
            evaluators=[evaluator],
        )

        # Direct call on a Run object
        from langsmith.schemas import Run, Example
        score = evaluator(run=my_run, example=my_example)
    """

    # LangSmith checks for this attribute to identify custom evaluators
    evaluation_name: str = "llm_guard_risk"

    def __init__(
        self,
        guard=None,
        threshold: float = 0.65,
        key: str = "llm_guard_risk",
    ):
        self._threshold = threshold
        self._key = key
        self.evaluation_name = key

        if guard is not None:
            self._guard = guard
        else:
            from llm_guard import AgentGuard
            self._guard = AgentGuard()

    # ------------------------------------------------------------------
    # LangSmith evaluator protocol
    # ------------------------------------------------------------------

    def __call__(self, run, example=None):
        """
        Score one LangSmith Run.

        Extracts question, steps, and final_answer from run.inputs / run.outputs,
        calls AgentGuard.score_chain(), and returns an EvaluationResult.

        Parameters
        ----------
        run : langsmith.schemas.Run
            The LangSmith run object produced by your chain.
        example : langsmith.schemas.Example, optional
            The dataset example. Used as fallback source for the question.

        Returns
        -------
        EvaluationResult
            ``score`` is the raw risk float; ``value`` is "pass" or "fail".
        """
        _ensure_langsmith()

        question, steps, final_answer = self._extract_chain(run, example)

        try:
            result = self._guard.score_chain(
                question=question,
                steps=steps,
                final_answer=final_answer,
            )
            risk = round(result.risk_score, 4)
            tier = result.confidence_tier          # HIGH / MEDIUM / LOW
            failure_mode = result.failure_mode or "none"
            needs_alert = result.needs_alert
            comment = (
                f"tier={tier} behavioral={result.behavioral_score:.3f} "
                f"failure_mode={failure_mode} needs_alert={needs_alert}"
            )
            # LangSmith convention: score=1 is good, score=0 is bad
            normalized_score = 1.0 - risk
            passed = risk < self._threshold
        except Exception as exc:
            logger.warning("LangSmithGuardEvaluator: scoring failed — %s", exc)
            risk = 0.5
            normalized_score = 0.5
            tier = "UNKNOWN"
            comment = f"Scoring error: {exc}"
            passed = True  # default to pass on error to avoid false alarms

        return _EvaluationResult(  # type: ignore[misc]
            key=self._key,
            score=normalized_score,
            value="pass" if passed else "fail",
            comment=comment,
        )

    # ------------------------------------------------------------------
    # Batch DataFrame evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(self, runs_df) -> Dict[str, Any]:
        """
        Evaluate a pandas DataFrame of agent runs offline.

        Expected columns (flexible — see below for all accepted variants):
          - question / input / query
          - steps / chain_steps  (list of step dicts, or JSON string)
          - final_answer / output / answer

        Parameters
        ----------
        runs_df : pandas.DataFrame
            DataFrame with one row per agent run.

        Returns
        -------
        dict
            Summary statistics:
            ``n``, ``mean_risk``, ``alert_rate``, ``tier_distribution``,
            ``scores`` (list of per-row dicts).
        """
        try:
            import pandas as _pd
        except ImportError:
            raise ImportError(
                "pandas is required for evaluate_dataset(). "
                "Install it with: pip install pandas"
            )

        import json as _json

        scores = []
        alert_count = 0
        tier_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for _, row in runs_df.iterrows():
            question = _coalesce(row, ["question", "input", "query"], "(unknown)")
            final_answer = _coalesce(row, ["final_answer", "output", "answer"], "")

            raw_steps = _coalesce(row, ["steps", "chain_steps"], [])
            if isinstance(raw_steps, str):
                try:
                    raw_steps = _json.loads(raw_steps)
                except Exception:
                    raw_steps = []
            if not isinstance(raw_steps, list):
                raw_steps = []

            try:
                result = self._guard.score_chain(
                    question=str(question),
                    steps=raw_steps,
                    final_answer=str(final_answer),
                )
                risk = round(result.risk_score, 4)
                tier = result.confidence_tier
                alert = result.needs_alert
            except Exception as exc:
                logger.warning("evaluate_dataset row error: %s", exc)
                risk = 0.5
                tier = "UNKNOWN"
                alert = False

            if alert:
                alert_count += 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            scores.append({
                "question":     str(question)[:120],
                "risk_score":   risk,
                "tier":         tier,
                "needs_alert":  alert,
            })

        n = len(scores)
        mean_risk = round(sum(s["risk_score"] for s in scores) / n, 4) if n else 0.0
        alert_rate = round(alert_count / n, 4) if n else 0.0

        return {
            "n":                n,
            "mean_risk":        mean_risk,
            "alert_rate":       alert_rate,
            "alert_count":      alert_count,
            "tier_distribution": tier_counts,
            "scores":           scores,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_chain(self, run, example=None):
        """
        Extract (question, steps, final_answer) from a LangSmith Run object.

        Handles both dict-style and attribute-style run objects, and the
        common LangChain ReAct output formats.
        """
        import json as _json

        # -- question --
        inputs = _safe_get(run, "inputs", {}) or {}
        question = (
            _coalesce(inputs, ["input", "question", "query"], "")
            or (str(inputs) if inputs else "")
        )
        # Fall back to example inputs if available
        if not question and example is not None:
            ex_inputs = _safe_get(example, "inputs", {}) or {}
            question = _coalesce(ex_inputs, ["input", "question", "query"], "(unknown)")
        question = str(question)[:500]

        # -- final_answer --
        outputs = _safe_get(run, "outputs", {}) or {}
        final_answer = _coalesce(outputs, ["output", "final_answer", "answer", "text"], "")
        if not final_answer and isinstance(outputs, dict):
            final_answer = str(list(outputs.values())[0]) if outputs else ""
        final_answer = str(final_answer)

        # -- steps --
        # LangSmith runs don't store intermediate steps directly; they appear
        # in child runs or outputs["intermediate_steps"].
        steps: List[Dict] = []

        # Check outputs for intermediate_steps (standard LangChain format)
        raw_intermediate = outputs.get("intermediate_steps", []) if isinstance(outputs, dict) else []
        if isinstance(raw_intermediate, str):
            try:
                raw_intermediate = _json.loads(raw_intermediate)
            except Exception:
                raw_intermediate = []

        for item in (raw_intermediate or []):
            # LangChain format: [(AgentAction, observation_str), ...]
            if isinstance(item, (list, tuple)) and len(item) == 2:
                action, observation = item
                thought = ""
                action_type = "Search"
                action_arg = ""
                if hasattr(action, "log"):
                    log = action.log or ""
                    for line in log.splitlines():
                        if line.strip().startswith("Thought:"):
                            thought = line.strip()[8:].strip()
                            break
                    action_type = getattr(action, "tool", "Search")
                    inp = getattr(action, "tool_input", "")
                    action_arg = str(inp.get("query", inp) if isinstance(inp, dict) else inp)
                elif isinstance(action, dict):
                    thought = action.get("thought", "")
                    action_type = action.get("action_type", action.get("tool", "Search"))
                    action_arg = action.get("action_arg", action.get("tool_input", ""))
                steps.append({
                    "thought":     thought,
                    "action_type": action_type,
                    "action_arg":  str(action_arg),
                    "observation": str(observation)[:500],
                })
            elif isinstance(item, dict):
                steps.append(item)

        # Also check if outputs["steps"] is directly present
        if not steps:
            direct_steps = outputs.get("steps", []) if isinstance(outputs, dict) else []
            if isinstance(direct_steps, list):
                steps = direct_steps

        # Synthesize a minimal Finish step so the chain is scoreable
        if final_answer:
            steps.append({
                "thought": "", "action_type": "Finish",
                "action_arg": final_answer[:300], "observation": "",
            })

        return question, steps, final_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(obj, attr: str, default):
    """Get an attribute from an object or dict key, returning default on miss."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _coalesce(mapping, keys: List[str], default: Any) -> Any:
    """Return the first non-empty value found under any of the given keys."""
    if isinstance(mapping, dict):
        for k in keys:
            v = mapping.get(k)
            if v is not None and v != "":
                return v
    else:
        # pandas Series or similar
        for k in keys:
            try:
                v = mapping[k]
                if v is not None and str(v) not in ("", "nan", "None"):
                    return v
            except (KeyError, TypeError, IndexError):
                pass
    return default
