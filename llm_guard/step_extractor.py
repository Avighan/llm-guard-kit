"""
step_extractor.py — StepExtractor ABC for domain-agnostic step feature extraction.
====================================================================================

Plug in a new process domain by implementing StepExtractor and passing the instance
to ProcessReliabilityMonitor. No core library code needs to change.

Quick start (new domain)
------------------------
    from llm_guard.step_extractor import StepExtractor

    class CodeAgentExtractor(StepExtractor):
        @property
        def feature_names(self):
            return ["test_pass_rate", "edit_distance_norm", "lint_errors_norm", "is_repeat_edit"]

        def extract(self, step):
            return {
                "test_pass_rate":     step.get("tests_passed", 0) / max(step.get("tests_total", 1), 1),
                "edit_distance_norm": min(step.get("edit_distance", 0) / 500.0, 1.0),
                "lint_errors_norm":   min(step.get("lint_errors", 0) / 10.0, 1.0),
                "is_repeat_edit":     float(step.get("repeated_file", False)),
            }

    monitor = ProcessReliabilityMonitor(extractor=CodeAgentExtractor())
    result  = monitor.score(steps=my_steps, output=final_output)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


_HEDGE_RE = re.compile(
    r"\b(not sure|uncertain|unclear|might|may|possibly|perhaps|i think|i believe|"
    r"probably|likely|seems|appear|could be|doubt|unsure|i'm not|don't know)\b", re.I
)

_SW = {
    "the","a","an","is","are","was","were","be","been","have","has","do","does","will",
    "would","could","should","may","might","can","this","that","it","of","in","on","at",
    "to","for","with","by","from","and","or","but","not","if","than","which","who","what",
}


def _toks(text: str):
    w = re.findall(r"[a-zA-Z]+", text.lower())
    return {x for x in w if x not in _SW and len(x) > 1}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _toks(a), _toks(b)
    return len(sa & sb) / max(len(sa | sb), 1)


class StepExtractor(ABC):
    """
    Abstract base class for per-step feature extraction.

    Implement this to plug a new process domain into ProcessReliabilityMonitor
    without modifying any core library code.

    Required methods
    ----------------
    feature_names : property → List[str]
        Ordered list of feature names that extract() returns.
        Must be stable across calls (no dynamic keys).

    extract(step: dict) → Dict[str, float]
        Extract scalar [0, 1] features from a single step dict.
        Keys must match feature_names exactly.

    Default method (inherited)
    --------------------------
    aggregate(steps, final_answer) → np.ndarray
        Applies mean per feature across all steps.
        Returns shape (len(feature_names),).
        Override for domain-specific aggregation.
    """

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Ordered list of feature names returned by extract()."""
        ...

    @abstractmethod
    def extract(self, step: Dict) -> Dict[str, float]:
        """Extract scalar features from one step. All values should be in [0, 1]."""
        ...

    def aggregate(self, steps: List[Dict], final_answer: str = "") -> np.ndarray:
        """
        Aggregate per-step features to a single chain-level vector.
        Default: mean per feature. Returns shape (len(feature_names),).
        Override for domain-specific aggregation logic.
        """
        if not steps:
            return np.zeros(len(self.feature_names), dtype=np.float32)
        rows = [self.extract(s) for s in steps]
        names = self.feature_names
        matrix = np.array([[r.get(n, 0.0) for n in names] for r in rows], dtype=np.float32)
        return matrix.mean(axis=0)


class LLMReActExtractor(StepExtractor):
    """
    StepExtractor for LLM agents using the ReAct (Reason + Act) framework.

    Extracts structural + answer-side features suitable for cross-domain scoring.
    Wraps the SC1-SC12 behavioral signals used by llm-guard-kit's LocalVerifier.

    Step dict schema (ReAct format)
    --------------------------------
    thought     : str  — agent's reasoning text
    action_type : str  — "Search" | "Finish" | etc.
    action_arg  : str  — search query or final answer text
    observation : str  — search result or "" for Finish steps

    Features returned by extract()
    --------------------------------
    Structural (domain-invariant — use for cross-domain):
        sc2_step_count          placeholder; replaced by chain-level count in aggregate()
        sc4_uncertainty_density hedging word rate (0=confident, 1=very uncertain)
        sc5_backtrack_rate      placeholder; replaced by repeated-query rate in aggregate()
        retrieval_conf          Jaccard(action_arg, observation) — retrieval quality
        semantic_gap            reasoning change from previous step (0.5 at step 0)

    Behavioral (within-domain):
        thought_len_norm        thought verbosity (norm by 50 words)
        empty_obs               1 if observation too short (<5 words), else 0

    Answer-side (domain-invariant):
        ans_entity_match        placeholder; filled in aggregate() with final_answer
        obs_entity_coverage     placeholder; filled in aggregate() with question entities
    """

    @property
    def feature_names(self) -> List[str]:
        return [
            "sc2_step_count",
            "sc4_uncertainty_density",
            "sc5_backtrack_rate",
            "retrieval_conf",
            "semantic_gap",
            "thought_len_norm",
            "empty_obs",
            "ans_entity_match",
            "obs_entity_coverage",
        ]

    def extract(self, step: Dict) -> Dict[str, float]:
        """Extract per-step features. Chain-level features (sc2, sc5, ans_*) filled in aggregate()."""
        thought    = step.get("thought", "")
        obs        = step.get("observation", "")
        action_arg = step.get("action_arg", step.get("action", ""))
        prev_thought = step.get("_prev_thought")  # injected by aggregate()

        # Retrieval confidence: did the search return what was searched for?
        retrieval_conf = _jaccard(action_arg, obs) if (action_arg and obs) else 0.0

        # Semantic gap: reasoning change from previous step
        semantic_gap = (1.0 - _jaccard(thought, prev_thought)) if prev_thought is not None else 0.5

        # Uncertainty density: hedging words per 10 words
        hedge_count   = len(_HEDGE_RE.findall(thought))
        thought_words = max(len(thought.split()), 1)
        uncertainty_density = min(hedge_count / (thought_words / 10.0 + 1e-6), 1.0)

        # Thought length normalised by 50 words
        thought_len_norm = min(len(thought.split()) / 50.0, 1.0)

        # Empty observation
        empty_obs = float(len(obs.split()) < 5)

        return {
            "sc2_step_count":          0.0,  # filled in aggregate()
            "sc4_uncertainty_density": float(uncertainty_density),
            "sc5_backtrack_rate":      0.0,  # filled in aggregate()
            "retrieval_conf":          float(retrieval_conf),
            "semantic_gap":            float(semantic_gap),
            "thought_len_norm":        float(thought_len_norm),
            "empty_obs":               float(empty_obs),
            "ans_entity_match":        0.0,  # filled in aggregate()
            "obs_entity_coverage":     0.0,  # filled in aggregate()
        }

    def aggregate(self, steps: List[Dict], final_answer: str = "") -> np.ndarray:
        """Full chain aggregation including cross-step and answer-side features."""
        if not steps:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        # Inject _prev_thought for semantic_gap computation
        enriched = []
        for i, s in enumerate(steps):
            e = dict(s)
            e["_prev_thought"] = steps[i - 1].get("thought", "") if i > 0 else None
            enriched.append(e)

        rows = [self.extract(s) for s in enriched]
        names = self.feature_names
        idx   = {n: i for i, n in enumerate(names)}
        matrix = np.array([[r.get(n, 0.0) for n in names] for r in rows], dtype=np.float32)
        agg = matrix.mean(axis=0).copy()

        # sc2: step count normalised by 10
        agg[idx["sc2_step_count"]] = min(len(steps) / 10.0, 1.0)

        # sc5: backtrack rate (repeated action_args)
        queries = [s.get("action_arg", "") for s in steps]
        agg[idx["sc5_backtrack_rate"]] = 1.0 - len(set(queries)) / max(len(queries), 1)

        # Answer-side features (need all observations + final_answer)
        def _cap_toks(text: str):
            return {w for w in re.findall(r"[A-Z][a-zA-Z]+", text) if len(w) > 1}

        obs_all  = " ".join(s.get("observation", "") for s in steps)
        fa_caps  = _cap_toks(final_answer)
        obs_caps = _cap_toks(obs_all)

        agg[idx["ans_entity_match"]] = (
            len(fa_caps & obs_caps) / max(len(fa_caps), 1) if fa_caps else 0.5
        )
        # obs_entity_coverage: set to 0.5 (question not available here)
        agg[idx["obs_entity_coverage"]] = 0.5

        return agg.astype(np.float32)
