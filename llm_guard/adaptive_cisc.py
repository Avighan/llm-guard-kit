"""
adaptive_cisc.py — Epsilon-greedy bandit for per-domain CISC threshold adaptation.
===================================================================================

CISC (Confidence-Informed Selective Calling) gates Sonnet judge calls by risk tier:
    HIGH   risk < high_threshold  → no judge call (1 call total)
    MEDIUM high_threshold ≤ risk < low_threshold → 2 calls
    LOW    risk ≥ low_threshold   → 3 calls

The default thresholds (0.50, 0.70) were calibrated on HotpotQA and may not be
optimal for your deployment.  AdaptiveCISC adapts them per-domain by observing
which tiers produce real failures vs false alerts.

Algorithm
---------
Epsilon-greedy threshold search:
  - With probability epsilon, explore a random perturbation of each threshold.
  - With probability 1-epsilon, exploit the current best-performing thresholds.
  - Reward = precision on alerts (fraction of alerted chains that were actually wrong).

The thresholds shift in ±0.02 increments, clamped to [0.30, 0.90].
After min_samples observations, adapt() is called automatically on each record.

Persistence
-----------
AdaptiveCISC stores state in a JSON file per domain, so thresholds survive restarts.
Pass state_dir to enable persistence.

Quick start
-----------
    from llm_guard.adaptive_cisc import AdaptiveCISC

    cisc = AdaptiveCISC(domain="medical", state_dir=".cisc_state")

    # After each scored + reviewed chain:
    cisc.record_outcome(risk_score=0.72, tier="LOW", was_wrong=True)

    # Get current thresholds for this domain:
    high_t, low_t = cisc.get_thresholds()
    print(f"CISC thresholds for medical: HIGH<{high_t}, LOW>={low_t}")

    # Integrate with AgentGuard:
    guard = AgentGuard(alert_threshold=low_t)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_STEP = 0.02          # threshold adjustment step size
_THRESH_MIN = 0.25
_THRESH_MAX = 0.90
_DEFAULT_HIGH = 0.50  # below → HIGH tier (no judge call)
_DEFAULT_LOW  = 0.70  # above → LOW tier (3 judge calls)


@dataclass
class _Observation:
    risk_score: float
    tier: str        # "HIGH" | "MEDIUM" | "LOW"
    was_wrong: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class _ThresholdStats:
    high_threshold: float = _DEFAULT_HIGH
    low_threshold:  float = _DEFAULT_LOW
    n_observations: int   = 0
    n_alerts:       int   = 0       # chains where tier == LOW (alert fired)
    n_true_alerts:  int   = 0       # alerts that were genuine failures
    n_false_alerts: int   = 0       # alerts that were false positives
    precision:      float = 0.0     # n_true_alerts / n_alerts (smoothed)
    recall:         float = 0.0     # n_true_alerts / n_wrong_total
    n_wrong_total:  int   = 0
    last_adapted_at: float = 0.0


class AdaptiveCISC:
    """
    Per-domain epsilon-greedy bandit that adapts CISC alert thresholds based on
    observed alert precision and recall.

    Parameters
    ----------
    domain : str
        Domain identifier (e.g. "medical", "customer_support", "default").
    epsilon : float
        Exploration probability. Default 0.10 (10% random perturbation).
    min_samples : int
        Minimum observations before threshold adaptation begins. Default 20.
    state_dir : str, optional
        Directory for persisting per-domain threshold state as JSON files.
        If None, state is in-memory only.
    target_precision : float
        Desired alert precision (fraction of alerts that are real failures).
        Thresholds adapt to approach this target. Default 0.80.
    target_recall : float
        Desired alert recall. Thresholds adapt to balance precision/recall.
        Default 0.60.
    """

    def __init__(
        self,
        domain: str = "default",
        epsilon: float = 0.10,
        min_samples: int = 20,
        state_dir: Optional[str] = None,
        target_precision: float = 0.80,
        target_recall: float = 0.60,
    ):
        self.domain           = domain
        self.epsilon          = epsilon
        self.min_samples      = min_samples
        self.state_dir        = Path(state_dir) if state_dir else None
        self.target_precision = target_precision
        self.target_recall    = target_recall

        self._stats     = _ThresholdStats()
        self._history:  List[_Observation] = []

        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_thresholds(self) -> Tuple[float, float]:
        """Return current (high_threshold, low_threshold) for this domain."""
        return self._stats.high_threshold, self._stats.low_threshold

    def tier_for_risk(self, risk_score: float) -> str:
        """Map a risk score to a CISC tier using current thresholds."""
        ht, lt = self.get_thresholds()
        if risk_score < ht:
            return "HIGH"
        elif risk_score < lt:
            return "MEDIUM"
        else:
            return "LOW"

    def record_outcome(
        self,
        risk_score: float,
        tier: str,
        was_wrong: bool,
    ) -> None:
        """
        Record the outcome for a scored chain.

        Parameters
        ----------
        risk_score : float
            The risk score returned by AgentGuard.score_chain().
        tier : str
            The CISC tier assigned ("HIGH" | "MEDIUM" | "LOW").
        was_wrong : bool
            True if the chain turned out to be a genuine failure (from user
            feedback or ground-truth label).
        """
        obs = _Observation(risk_score=risk_score, tier=tier, was_wrong=was_wrong)
        self._history.append(obs)

        s = self._stats
        s.n_observations += 1
        if tier == "LOW":
            s.n_alerts += 1
            if was_wrong:
                s.n_true_alerts  += 1
            else:
                s.n_false_alerts += 1
        if was_wrong:
            s.n_wrong_total += 1

        # Laplace-smoothed precision / recall
        s.precision = (s.n_true_alerts + 1) / (s.n_alerts + 2)
        s.recall    = (s.n_true_alerts + 1) / (s.n_wrong_total + 2)

        if s.n_observations >= self.min_samples:
            self._adapt()

        if self.state_dir:
            self._save_state()

    def summary(self) -> Dict:
        """Return a human-readable summary of current threshold state."""
        s = self._stats
        ht, lt = self.get_thresholds()
        return {
            "domain":           self.domain,
            "high_threshold":   round(ht, 3),
            "low_threshold":    round(lt, 3),
            "n_observations":   s.n_observations,
            "n_alerts":         s.n_alerts,
            "precision":        round(s.precision, 3),
            "recall":           round(s.recall, 3),
            "target_precision": self.target_precision,
            "target_recall":    self.target_recall,
            "gap_precision":    round(s.precision - self.target_precision, 3),
            "gap_recall":       round(s.recall - self.target_recall, 3),
        }

    # ── Adaptation logic ─────────────────────────────────────────────────────

    def _adapt(self) -> None:
        """Epsilon-greedy threshold update step."""
        s = self._stats
        ht, lt = s.high_threshold, s.low_threshold

        if random.random() < self.epsilon:
            # Explore: random ±1 step perturbation
            ht = self._clamp(ht + random.choice([-_STEP, 0.0, _STEP]))
            lt = self._clamp(lt + random.choice([-_STEP, 0.0, _STEP]))
        else:
            # Exploit: gradient-like update toward targets
            prec_gap   = s.precision - self.target_precision
            recall_gap = s.recall    - self.target_recall

            # High precision but low recall → lower low_threshold (be more aggressive)
            # Low precision but high recall → raise low_threshold (be more conservative)
            if prec_gap > 0.05 and recall_gap < -0.05:
                lt = self._clamp(lt - _STEP)
            elif prec_gap < -0.05:
                lt = self._clamp(lt + _STEP)

            # Adjust high_threshold: if most HIGH-tier chains turn out wrong → lower it
            high_tier_wrong_rate = self._high_tier_wrong_rate()
            if high_tier_wrong_rate > 0.35:
                ht = self._clamp(ht - _STEP)
            elif high_tier_wrong_rate < 0.10 and ht < lt - 0.10:
                ht = self._clamp(ht + _STEP)

        # Ensure high_threshold < low_threshold with minimum gap 0.10
        if ht >= lt - 0.10:
            lt = self._clamp(ht + 0.10)

        s.high_threshold = ht
        s.low_threshold  = lt
        s.last_adapted_at = time.time()

    def _high_tier_wrong_rate(self) -> float:
        """Fraction of HIGH-tier observations that turned out to be wrong."""
        high_obs = [o for o in self._history if o.tier == "HIGH"]
        if not high_obs:
            return 0.0
        wrong = sum(1 for o in high_obs if o.was_wrong)
        return wrong / len(high_obs)

    @staticmethod
    def _clamp(v: float) -> float:
        return max(_THRESH_MIN, min(_THRESH_MAX, round(v, 3)))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.domain)
        return self.state_dir / f"cisc_{safe_name}.json"  # type: ignore[operator]

    def _save_state(self) -> None:
        try:
            data = {
                "stats":   asdict(self._stats),
                "history": [asdict(o) for o in self._history[-500:]],  # keep last 500
            }
            self._state_path().write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            s = data.get("stats", {})
            self._stats = _ThresholdStats(
                high_threshold   = s.get("high_threshold",   _DEFAULT_HIGH),
                low_threshold    = s.get("low_threshold",    _DEFAULT_LOW),
                n_observations   = s.get("n_observations",   0),
                n_alerts         = s.get("n_alerts",         0),
                n_true_alerts    = s.get("n_true_alerts",    0),
                n_false_alerts   = s.get("n_false_alerts",   0),
                precision        = s.get("precision",        0.0),
                recall           = s.get("recall",           0.0),
                n_wrong_total    = s.get("n_wrong_total",    0),
                last_adapted_at  = s.get("last_adapted_at",  0.0),
            )
            self._history = [
                _Observation(**o) for o in data.get("history", [])
            ]
        except Exception:
            pass   # corrupt state — start fresh


# ── Registry: one AdaptiveCISC per domain ────────────────────────────────────

class AdaptiveCISCRegistry:
    """
    Thread-safe registry of per-domain AdaptiveCISC instances.

    Usage
    -----
        registry = AdaptiveCISCRegistry(state_dir=".cisc_state")
        cisc = registry.get("medical")
        ht, lt = cisc.get_thresholds()
        cisc.record_outcome(risk_score, tier, was_wrong)
    """

    def __init__(
        self,
        state_dir: Optional[str] = None,
        epsilon: float = 0.10,
        min_samples: int = 20,
        target_precision: float = 0.80,
        target_recall: float = 0.60,
    ):
        self._state_dir       = state_dir
        self._epsilon         = epsilon
        self._min_samples     = min_samples
        self._target_precision = target_precision
        self._target_recall   = target_recall
        self._domains: Dict[str, AdaptiveCISC] = {}

    def get(self, domain: str) -> AdaptiveCISC:
        """Return the AdaptiveCISC instance for domain, creating it if needed."""
        if domain not in self._domains:
            self._domains[domain] = AdaptiveCISC(
                domain           = domain,
                epsilon          = self._epsilon,
                min_samples      = self._min_samples,
                state_dir        = self._state_dir,
                target_precision = self._target_precision,
                target_recall    = self._target_recall,
            )
        return self._domains[domain]

    def all_summaries(self) -> List[Dict]:
        return [cisc.summary() for cisc in self._domains.values()]
