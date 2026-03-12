"""Tests for per-domain alert threshold calibration in DriftMonitor."""
import sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_guard.drift_detector import DriftMonitor


def test_domain_threshold_initially_global():
    m = DriftMonitor()
    assert m.alert_threshold_for("hotpot") == 0.70


def test_set_domain_threshold_manual():
    m = DriftMonitor()
    m.set_domain_threshold("trivia", 0.55)
    assert abs(m.alert_threshold_for("trivia") - 0.55) < 1e-6


def test_domain_threshold_fallback_to_global():
    m = DriftMonitor()
    m.set_domain_threshold("trivia", 0.55)
    # Different domain still gets global
    assert m.alert_threshold_for("unknown_domain") == 0.70


def test_alert_threshold_for_returns_float():
    m = DriftMonitor()
    result = m.alert_threshold_for("any_domain")
    assert isinstance(result, float)
    assert 0.0 < result <= 1.0


def test_auto_calibration_after_alarm():
    """After 20 post-ALARM scores accumulate, threshold is estimated as p75."""
    m = DriftMonitor()
    domain = "new_domain"
    # Simulate an ALARM event by directly calling _maybe_update_domain_threshold
    # with a mock event that has severity="ALARM"
    class _FakeEvent:
        severity = "ALARM"
    m._maybe_update_domain_threshold(domain, 0.8, _FakeEvent())
    assert domain in m._domain_warmup  # warmup started
    # Feed 19 more scores (total 20 including the alarm-triggering one)
    for v in np.linspace(0.5, 0.9, 19):
        m._maybe_update_domain_threshold(domain, float(v), None)
    # After 20 samples, threshold should be estimated and warmup cleared
    assert domain not in m._domain_warmup, "warmup should be cleared after 20 samples"
    estimated = m.alert_threshold_for(domain)
    assert estimated != 0.70, "threshold should no longer be the global default"
    assert 0.0 < estimated <= 1.0
