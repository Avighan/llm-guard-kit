"""
process_monitor.py — Domain-agnostic process reliability monitor.
==================================================================

ProcessReliabilityMonitor is the generic entry point for the reliability
monitoring framework. It accepts any StepExtractor (define one to plug in a
new domain) and applies the standard scoring → calibration → alerting pipeline.

For LLM ReAct agents (the default domain), use ProcessReliabilityMonitor.for_domain("llm_react")
which is equivalent to wrapping AgentGuard with LLMReActExtractor.

Quick start — new domain
------------------------
    from llm_guard.process_monitor import ProcessReliabilityMonitor
    from llm_guard.step_extractor import StepExtractor

    class PipelineExtractor(StepExtractor):
        @property
        def feature_names(self):
            return ["null_rate", "row_count_delta", "schema_errors"]
        def extract(self, step):
            return {
                "null_rate":        step.get("null_rate", 0.0),
                "row_count_delta":  step.get("row_count_delta", 0.0),
                "schema_errors":    min(step.get("schema_errors", 0) / 10.0, 1.0),
            }

    monitor = ProcessReliabilityMonitor(extractor=PipelineExtractor())
    result  = monitor.score(steps=pipeline_stages, output=pipeline_result)
    if result.needs_alert:
        print(f"Pipeline risk: {result.risk_score:.2f} — {result.failure_mode}")

Quick start — LLM ReAct (built-in domain)
------------------------------------------
    from llm_guard.process_monitor import ProcessReliabilityMonitor
    monitor = ProcessReliabilityMonitor.for_domain("llm_react")
    result  = monitor.score(steps=react_steps, output=final_answer,
                            context=question)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from llm_guard.step_extractor import StepExtractor, LLMReActExtractor


# ── MonitorResult ─────────────────────────────────────────────────────────────

@dataclass
class MonitorResult:
    """
    Result of ProcessReliabilityMonitor.score().

    Mirrors ChainTrustResult but is domain-agnostic.

    Fields
    ------
    risk_score      : float [0, 1] — higher = more likely a process failure
    confidence_tier : str   — "HIGH" | "MEDIUM" | "LOW"
    needs_alert     : bool  — True when risk_score >= alert_threshold
    failure_mode    : str or None — detected failure pattern, or None
    repair_hint     : str or None — suggested corrective action
    extractor_features : dict  — raw feature values from the StepExtractor
    latency_ms      : float  — scoring latency in milliseconds
    """
    risk_score:          float
    confidence_tier:     str
    needs_alert:         bool
    failure_mode:        Optional[str]      = None
    repair_hint:         Optional[str]      = None
    extractor_features:  Dict[str, float]   = field(default_factory=dict)
    latency_ms:          float              = 0.0


# ── Domain registry ───────────────────────────────────────────────────────────

_DOMAIN_REGISTRY: Dict[str, Type[StepExtractor]] = {
    "llm_react": LLMReActExtractor,
}


def register_domain(name: str, extractor_cls: Type[StepExtractor]) -> None:
    """Register a custom extractor under a domain name for use with for_domain()."""
    _DOMAIN_REGISTRY[name] = extractor_cls


# ── ProcessReliabilityMonitor ─────────────────────────────────────────────────

class ProcessReliabilityMonitor:
    """
    Domain-agnostic process reliability monitor.

    Accepts any StepExtractor, applies the behavioral → judge → calibrate → alert
    pipeline, and returns a MonitorResult.

    Parameters
    ----------
    extractor : StepExtractor
        Feature extractor for this domain. Use LLMReActExtractor() for ReAct agents.
    judge_fn : callable, optional
        External oracle / judge. Signature: (steps: list, output: str) → float [0,1].
        When provided, combined with behavioral score via judge_weight.
    judge_weight : float
        Weight for judge_fn score in ensemble (default 0.5 when judge_fn provided).
    alert_threshold : float
        Risk score above which needs_alert=True. Default 0.70.
    """

    def __init__(
        self,
        extractor:       StepExtractor,
        judge_fn:        Optional[Callable] = None,
        judge_weight:    float              = 0.5,
        alert_threshold: float              = 0.70,
    ):
        self.extractor        = extractor
        self._judge_fn        = judge_fn
        self._judge_weight    = judge_weight
        self._alert_threshold = alert_threshold

    @classmethod
    def for_domain(cls, domain: str, **kwargs) -> "ProcessReliabilityMonitor":
        """
        Create a monitor for a registered domain.

        Built-in domains: "llm_react"
        Custom domains: register with register_domain() first.
        """
        if domain not in _DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown domain {domain!r}. "
                f"Available: {list(_DOMAIN_REGISTRY)}. "
                f"Register new domains with llm_guard.process_monitor.register_domain()."
            )
        return cls(extractor=_DOMAIN_REGISTRY[domain](), **kwargs)

    def score(
        self,
        steps:   List[Dict],
        output:  str = "",
        context: str = "",
    ) -> MonitorResult:
        """
        Score a completed process run.

        Parameters
        ----------
        steps   : list of step dicts — format is domain-specific (defined by extractor)
        output  : final output of the process (final_answer for LLM, pipeline result for ETL, etc.)
        context : optional context / question string (used by answer-side features)

        Returns
        -------
        MonitorResult
        """
        import time
        t0 = time.time()

        # 1. Extract features
        feat_vec  = self.extractor.aggregate(steps, final_answer=output)
        feat_dict = {n: float(v) for n, v in zip(self.extractor.feature_names, feat_vec)}

        # 2. Behavioral score: weighted mean of available features
        behavioral_score = float(np.mean(feat_vec)) if len(feat_vec) > 0 else 0.5
        behavioral_score = float(np.clip(behavioral_score, 0.0, 1.0))

        # 3. Judge blend (optional)
        if self._judge_fn is not None:
            try:
                judge_score = float(self._judge_fn(steps, output))
                risk_score  = (
                    (1.0 - self._judge_weight) * behavioral_score
                    + self._judge_weight * judge_score
                )
            except Exception:
                risk_score = behavioral_score
        else:
            risk_score = behavioral_score

        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # 4. Tiers
        if risk_score < 0.50:
            tier = "HIGH"
        elif risk_score < 0.70:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        needs_alert = risk_score >= self._alert_threshold

        # 5. Failure mode (basic heuristics over extracted features)
        failure_mode = self._detect_failure_mode(feat_dict, steps, output)
        repair_hint  = _REPAIR_HINTS.get(failure_mode)

        latency_ms = (time.time() - t0) * 1000.0

        return MonitorResult(
            risk_score         = round(risk_score, 4),
            confidence_tier    = tier,
            needs_alert        = needs_alert,
            failure_mode       = failure_mode,
            repair_hint        = repair_hint,
            extractor_features = feat_dict,
            latency_ms         = round(latency_ms, 2),
        )

    def _detect_failure_mode(
        self,
        feat_dict: Dict[str, float],
        steps:     List[Dict],
        output:    str,
    ) -> Optional[str]:
        """Heuristic failure mode detection from extracted features."""
        # empty output
        if not output or len(output.strip()) == 0:
            return "empty_output"
        # very short chain, low uncertainty, low backtracking → confident_wrong
        n_steps_norm = feat_dict.get("sc2_step_count", 1.0)
        uncertainty  = feat_dict.get("sc4_uncertainty_density", 1.0)
        backtrack    = feat_dict.get("sc5_backtrack_rate", 1.0)
        if n_steps_norm <= 0.2 and uncertainty < 0.05 and backtrack < 0.05:
            return "confident_wrong"
        # bad retrieval: low retrieval_conf + high empty_obs
        retrieval_conf = feat_dict.get("retrieval_conf", 1.0)
        empty_obs      = feat_dict.get("empty_obs", 0.0)
        if retrieval_conf < 0.05 and empty_obs > 0.5:
            return "retrieval_fail"
        # repeated queries: high backtrack rate
        if backtrack > 0.5:
            return "repeated_query"
        # long chain: many steps
        if n_steps_norm > 0.4:
            return "long_chain"
        return None


# ── Repair hint catalogue ─────────────────────────────────────────────────────

_REPAIR_HINTS: Dict[str, str] = {
    "empty_output":    "No output produced. Start with foundational background facts.",
    "confident_wrong": "Verify: search for a source that confirms the specific claim.",
    "retrieval_fail":  "Searches returned no results. Use broader / alternative terminology.",
    "repeated_query":  "Force query deduplication — no repeat of prior searches.",
    "long_chain":      "Too many steps. Focus on the single most important fact.",
}
