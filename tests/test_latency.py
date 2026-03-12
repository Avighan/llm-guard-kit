"""
Latency regression tests — behavioral p50 must be < 50ms.

These tests are fast (200 iterations each) and run in CI without any API key.
They guard against accidental performance regressions in the behavioral path.

Run:
    pytest tests/test_latency.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTION = "What year was the Eiffel Tower completed?"
STEPS = [
    {"thought": "I need to find when the Eiffel Tower was completed.",
     "action_type": "Search", "action_arg": "Eiffel Tower completion year",
     "observation": "The Eiffel Tower was completed in 1889."},
    {"thought": "The tower was completed in 1889.",
     "action_type": "Finish", "action_arg": "1889", "observation": ""},
]
FINAL = "1889"
N_REPS = 100  # enough for stable percentile; fast enough for CI (< 5s)


@pytest.fixture(scope="module")
def guard():
    from llm_guard.agent_guard import AgentGuard
    g = AgentGuard()  # no API key — behavioral only
    # Warm up
    for _ in range(5):
        g.score_chain(QUESTION, STEPS, FINAL)
    return g


def _p50(times_ms: list[float]) -> float:
    return float(np.percentile(times_ms, 50))


class TestBehavioralLatency:
    def test_score_chain_p50_under_50ms(self, guard):
        """Behavioral score_chain() p50 must be < 50ms (SLA target: < 15ms)."""
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            guard.score_chain(QUESTION, STEPS, FINAL)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = _p50(times)
        assert p50 < 50, (
            f"Behavioral p50={p50:.1f}ms exceeds 50ms SLA. "
            f"Target is <15ms. Check for regression."
        )

    def test_score_chain_p99_under_200ms(self, guard):
        """Behavioral p99 must be < 200ms — guards against occasional spike."""
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            guard.score_chain(QUESTION, STEPS, FINAL)
            times.append((time.perf_counter() - t0) * 1000)

        p99 = float(np.percentile(times, 99))
        assert p99 < 200, f"Behavioral p99={p99:.1f}ms exceeds 200ms limit."

    def test_score_chain_returns_valid_result(self, guard):
        """Smoke test: result has risk_score in [0, 1]."""
        result = guard.score_chain(QUESTION, STEPS, FINAL)
        assert hasattr(result, "risk_score")
        assert 0.0 <= result.risk_score <= 1.0
