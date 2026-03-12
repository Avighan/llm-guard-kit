"""
drift_detector.py — Production drift detector for AgentGuard deployments.
==========================================================================

Detects when the distribution of behavioral_score shifts significantly from the
calibration baseline (e.g., domain change, agent architecture change, new failure
mode not seen during calibration).

Two detectors:
  DriftDetector    — online sequential detector (CUSUM + rolling PSI)
  DriftMonitor     — wraps AgentGuard: auto-records each score, fires callbacks

When drift is detected, the monitor:
  1. Emits a DriftEvent with severity WARN or ALARM
  2. Optionally resets AdaptiveCISC thresholds to defaults
  3. Marks the LocalVerifier as stale (on_verifier_stale callback)
  4. Calls the on_drift callback for user-defined handling

Validated in exp118:
  - CUSUM detects looping-domain shift in 1 sample after onset
  - KS test correctly gives p=0.818 for same-distribution split (no false positive)
  - PSI(baseline, looping) = 43.6 >> 0.20 threshold

Quick start
-----------
    from llm_guard.drift_detector import DriftMonitor
    from llm_guard.agent_guard import AgentGuard

    guard   = AgentGuard(api_key=...)
    monitor = DriftMonitor(guard, domain="default")

    # After calibration period (≥ 20 chains):
    monitor.set_baseline()

    # On each new chain:
    result  = guard.score_chain(question, steps, answer)
    event   = monitor.record(result.behavioral_score, domain="default")
    if event and event.severity == "ALARM":
        print(f"Domain drift detected! Reset calibration. PSI={event.psi:.2f}")
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


# ── Validated thresholds (exp118) ─────────────────────────────────────────────

_PSI_WARN    = 0.10   # moderate drift: log warning
_PSI_ALARM   = 0.20   # significant drift: reset calibration
_CUSUM_K     = 0.5    # allowance: fraction of baseline std
_CUSUM_H     = 5.0    # decision threshold: multiples of baseline std
_WINDOW_SIZE = 50     # rolling window for PSI comparison
_MIN_WINDOW  = 20     # minimum samples before CUSUM/PSI activate


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DriftEvent:
    """Fired when a drift detector triggers."""
    domain:        str
    severity:      str          # "WARN" | "ALARM"
    detector:      str          # "CUSUM" | "PSI" | "KS"
    psi:           float        # current PSI value (−1 if not computed yet)
    cusum_value:   float        # current CUSUM statistic
    n_samples:     int          # samples seen since baseline
    timestamp:     float = field(default_factory=time.time)
    message:       str   = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaselineStats:
    mean:  float
    std:   float
    n:     int
    scores: List[float] = field(default_factory=list)


# ── Core DriftDetector ────────────────────────────────────────────────────────

class DriftDetector:
    """
    Online drift detector combining CUSUM (sequential) and PSI (batch rolling).

    Parameters
    ----------
    domain : str
        Domain identifier for logging.
    psi_warn : float
        PSI threshold for WARN event. Default 0.10.
    psi_alarm : float
        PSI threshold for ALARM event. Default 0.20.
    cusum_k : float
        CUSUM allowance as fraction of baseline std. Default 0.5.
    cusum_h : float
        CUSUM decision threshold as multiple of baseline std. Default 5.0.
    window_size : int
        Rolling window length for PSI comparison. Default 50.
    min_window : int
        Minimum samples before detectors activate. Default 20.
    """

    def __init__(
        self,
        domain:      str   = "default",
        psi_warn:    float = _PSI_WARN,
        psi_alarm:   float = _PSI_ALARM,
        cusum_k:     float = _CUSUM_K,
        cusum_h:     float = _CUSUM_H,
        window_size: int   = _WINDOW_SIZE,
        min_window:  int   = _MIN_WINDOW,
    ):
        self.domain      = domain
        self.psi_warn    = psi_warn
        self.psi_alarm   = psi_alarm
        self.cusum_k     = cusum_k
        self.cusum_h     = cusum_h
        self.window_size = window_size
        self.min_window  = min_window

        self._baseline:   Optional[BaselineStats]    = None
        self._buffer:     Deque[float]               = deque(maxlen=window_size)
        self._cusum:      float                      = 0.0
        self._n_seen:     int                        = 0
        self._last_event: Optional[DriftEvent]       = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_baseline(self, scores: List[float]) -> None:
        """Set the calibration baseline from a list of scores."""
        arr = np.array(scores, dtype=float)
        self._baseline = BaselineStats(
            mean   = float(np.mean(arr)),
            std    = float(max(np.std(arr), 1e-4)),
            n      = len(arr),
            scores = list(scores),
        )
        self._cusum  = 0.0
        self._n_seen = 0
        self._buffer.clear()

    def update(self, score: float) -> Optional[DriftEvent]:
        """
        Record one new score. Returns a DriftEvent if drift is detected, else None.
        Call this after every score_chain().
        """
        if self._baseline is None:
            self._buffer.append(score)
            return None

        self._buffer.append(score)
        self._n_seen += 1

        if self._n_seen < self.min_window:
            return None

        mu  = self._baseline.mean
        sig = self._baseline.std
        K   = self.cusum_k * sig
        H   = self.cusum_h * sig

        # CUSUM update
        self._cusum = max(0.0, self._cusum + (score - mu) - K)

        # Compute PSI from rolling window vs baseline
        psi = self._compute_psi(list(self._buffer), self._baseline.scores)

        # Fire event if alarm triggered
        if self._cusum > H:
            event = DriftEvent(
                domain      = self.domain,
                severity    = "ALARM",
                detector    = "CUSUM",
                psi         = round(psi, 4),
                cusum_value = round(self._cusum, 4),
                n_samples   = self._n_seen,
                message     = (
                    f"CUSUM={self._cusum:.3f} > H={H:.3f}. "
                    f"PSI={psi:.3f}. Behavioral score distribution has shifted. "
                    f"Consider resetting calibration."
                ),
            )
            self._last_event = event
            self._cusum = 0.0  # reset after alarm (Shewhart restart)
            return event

        if psi >= self.psi_alarm:
            event = DriftEvent(
                domain      = self.domain,
                severity    = "ALARM",
                detector    = "PSI",
                psi         = round(psi, 4),
                cusum_value = round(self._cusum, 4),
                n_samples   = self._n_seen,
                message     = (
                    f"PSI={psi:.3f} ≥ {self.psi_alarm}. "
                    f"Significant domain drift. Reset calibration recommended."
                ),
            )
            self._last_event = event
            return event

        if psi >= self.psi_warn:
            event = DriftEvent(
                domain      = self.domain,
                severity    = "WARN",
                detector    = "PSI",
                psi         = round(psi, 4),
                cusum_value = round(self._cusum, 4),
                n_samples   = self._n_seen,
                message     = (
                    f"PSI={psi:.3f} ≥ {self.psi_warn}. "
                    f"Moderate drift detected. Monitor closely."
                ),
            )
            self._last_event = event
            return event

        return None

    def reset(self) -> None:
        """Reset the CUSUM statistic and rolling buffer (keep baseline)."""
        self._cusum  = 0.0
        self._n_seen = 0
        self._buffer.clear()

    def reset_baseline(self) -> None:
        """Full reset: clear baseline, CUSUM, and buffer."""
        self._baseline   = None
        self._cusum      = 0.0
        self._n_seen     = 0
        self._buffer.clear()
        self._last_event = None

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    @property
    def baseline_mean(self) -> Optional[float]:
        return self._baseline.mean if self._baseline else None

    @property
    def last_event(self) -> Optional[DriftEvent]:
        return self._last_event

    def summary(self) -> dict:
        return {
            "domain":         self.domain,
            "has_baseline":   self.has_baseline,
            "baseline_mean":  round(self._baseline.mean, 4) if self._baseline else None,
            "baseline_std":   round(self._baseline.std, 4)  if self._baseline else None,
            "n_seen":         self._n_seen,
            "cusum":          round(self._cusum, 4),
            "buffer_size":    len(self._buffer),
            "last_severity":  self._last_event.severity if self._last_event else None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_psi(target: List[float], reference: List[float], n_bins: int = 10) -> float:
        eps  = 1e-4
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ref_hist, _ = np.histogram(reference, bins=bins)
        tgt_hist, _ = np.histogram(target,    bins=bins)
        ref_pct = (ref_hist + eps) / (len(reference) + eps * n_bins)
        tgt_pct = (tgt_hist + eps) / (len(target)    + eps * n_bins)
        psi = float(np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct)))
        return max(0.0, round(psi, 6))


# ── DriftMonitor: wraps AgentGuard with per-domain drift detection ─────────────

class DriftMonitor:
    """
    Wraps a DriftDetector per domain and integrates with AdaptiveCISC.

    When drift is detected (severity=ALARM):
      - Calls on_drift(event)
      - Calls on_verifier_stale() if LocalVerifier should be retrained
      - Optionally resets AdaptiveCISC thresholds for the domain

    Parameters
    ----------
    on_drift : callable, optional
        Called with a DriftEvent when drift is detected.
    on_verifier_stale : callable, optional
        Called with domain name when the LocalVerifier should be retrained
        (new calibration data needed due to domain shift).
    auto_reset_cisc : bool
        If True, automatically resets AdaptiveCISC to default thresholds on ALARM.
        Default True.
    state_dir : str, optional
        Directory for persisting per-domain detector state as JSON.
    """

    def __init__(
        self,
        on_drift:          Optional[Callable[[DriftEvent], None]] = None,
        on_verifier_stale: Optional[Callable[[str], None]]        = None,
        auto_reset_cisc:   bool = True,
        state_dir:         Optional[str] = None,
        **detector_kwargs,
    ):
        self.on_drift          = on_drift
        self.on_verifier_stale = on_verifier_stale
        self.auto_reset_cisc   = auto_reset_cisc
        self.state_dir         = Path(state_dir) if state_dir else None
        self._detector_kwargs  = detector_kwargs
        self._detectors:  dict[str, DriftDetector] = {}
        self._cisc_registry    = None  # set via attach_cisc_registry()

        self._domain_thresholds: Dict[str, float] = {}
        self._domain_warmup: Dict[str, List[float]] = {}   # scores since last ALARM per domain
        self._global_threshold: float = 0.70

        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._load_all_states()

    def attach_cisc_registry(self, registry) -> None:
        """Attach an AdaptiveCISCRegistry so drift resets CISC thresholds."""
        self._cisc_registry = registry

    def get_detector(self, domain: str) -> DriftDetector:
        if domain not in self._detectors:
            self._detectors[domain] = DriftDetector(
                domain=domain, **self._detector_kwargs
            )
        return self._detectors[domain]

    def set_baseline(self, scores: List[float], domain: str = "default") -> None:
        """Set the calibration baseline for a domain."""
        det = self.get_detector(domain)
        det.fit_baseline(scores)
        self._save_state(domain)

    def set_domain_threshold(self, domain: str, threshold: float) -> None:
        """Manually set the alert threshold for a domain."""
        self._domain_thresholds[domain] = float(threshold)

    def alert_threshold_for(self, domain: str) -> float:
        """Return calibrated alert threshold for domain, or global default (0.70)."""
        return self._domain_thresholds.get(domain, self._global_threshold)

    def _maybe_update_domain_threshold(self, domain: str, score: float, event) -> None:
        """After ALARM: accumulate scores, estimate threshold when n >= 20."""
        if event is not None and event.severity == "ALARM":
            # Start warmup for this domain
            self._domain_warmup[domain] = [score]
        elif domain in self._domain_warmup:
            # Accumulating post-alarm scores
            self._domain_warmup[domain].append(score)
            if len(self._domain_warmup[domain]) >= 20:
                scores = np.array(self._domain_warmup.pop(domain))
                # p75 of post-alarm scores as domain threshold estimate
                self._domain_thresholds[domain] = float(np.percentile(scores, 75))

    def record(self, score: float, domain: str = "default") -> Optional[DriftEvent]:
        """
        Record one behavioral_score observation for a domain.
        Returns a DriftEvent if drift is detected, else None.

        Typical usage::

            result = guard.score_chain(question, steps, answer)
            event  = monitor.record(result.behavioral_score, domain=req.domain)
            if event:
                logger.warning(event.message)
        """
        det   = self.get_detector(domain)

        # Auto-build baseline from first min_window samples if not yet set
        if not det.has_baseline:
            det._buffer.append(score)
            if len(det._buffer) >= det.min_window:
                det.fit_baseline(list(det._buffer))
            return None

        event = det.update(score)

        if event and event.severity == "ALARM":
            # Reset CISC thresholds if attached
            if self.auto_reset_cisc and self._cisc_registry is not None:
                cisc = self._cisc_registry.get(domain)
                from llm_guard.adaptive_cisc import _DEFAULT_HIGH, _DEFAULT_LOW
                cisc._stats.high_threshold = _DEFAULT_HIGH
                cisc._stats.low_threshold  = _DEFAULT_LOW
                cisc._stats.n_observations = 0
                cisc._history.clear()

            # Notify that LocalVerifier needs retraining
            if self.on_verifier_stale:
                try:
                    self.on_verifier_stale(domain)
                except Exception:
                    pass

            if self.on_drift:
                try:
                    self.on_drift(event)
                except Exception:
                    pass

            # Reset detector so it starts fresh after alarm
            det.reset_baseline()

        elif event and event.severity == "WARN":
            if self.on_drift:
                try:
                    self.on_drift(event)
                except Exception:
                    pass

        if self.state_dir:
            self._save_state(domain)

        self._maybe_update_domain_threshold(domain, score, event)

        return event

    def all_summaries(self) -> List[dict]:
        return [det.summary() for det in self._detectors.values()]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _state_path(self, domain: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in domain)
        return self.state_dir / f"drift_{safe}.json"  # type: ignore[operator]

    def _save_state(self, domain: str) -> None:
        if not self.state_dir:
            return
        det = self._detectors.get(domain)
        if det is None:
            return
        try:
            data = {
                "domain":       domain,
                "baseline":     asdict(det._baseline) if det._baseline else None,
                "cusum":        det._cusum,
                "n_seen":       det._n_seen,
                "buffer":       list(det._buffer),
            }
            self._state_path(domain).write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_all_states(self) -> None:
        if not self.state_dir:
            return
        for p in self.state_dir.glob("drift_*.json"):
            try:
                data    = json.loads(p.read_text())
                domain  = data["domain"]
                det     = self.get_detector(domain)
                if data.get("baseline"):
                    b = data["baseline"]
                    det._baseline = BaselineStats(**b)
                det._cusum  = data.get("cusum", 0.0)
                det._n_seen = data.get("n_seen", 0)
                for s in data.get("buffer", []):
                    det._buffer.append(s)
            except Exception:
                pass
