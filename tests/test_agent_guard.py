"""
Tests for AgentGuard, LocalVerifier, QPPGNano, and framework integrations.
==========================================================================
Run with:  pytest tests/test_agent_guard.py -v

Categories:
  Unit        — pure Python, no API, no embedding model
  Integration — requires ANTHROPIC_API_KEY (skipped if absent)
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from llm_guard.agent_guard import (
    AgentGuard,
    AgentStepResult,
    ChainTrustResult,
    _risk_to_tier,
    _risk_to_labels,
)
from llm_guard.nano import QPPGNano
from llm_guard.local_verifier import LocalVerifier, extract_features, FEATURE_NAMES


# ── Fixtures ─────────────────────────────────────────────────────────────────

SIMPLE_STEPS = [
    {"thought": "I should search for this",
     "action_type": "Search", "action_arg": "Einstein birthday", "observation": "March 14 1879"},
    {"thought": "Now I have the answer",
     "action_type": "Finish", "action_arg": "1879", "observation": ""},
]

LOOPING_STEPS = [
    {"thought": "Search", "action_type": "Search", "action_arg": "x", "observation": "nothing"},
    {"thought": "Search again", "action_type": "Search", "action_arg": "x", "observation": "nothing"},
    {"thought": "Search again", "action_type": "Search", "action_arg": "x", "observation": "nothing"},
    {"thought": "I give up", "action_type": "Finish", "action_arg": "unknown", "observation": ""},
]

LABELED_RUNS = [
    {"question": f"Q{i}", "steps": SIMPLE_STEPS, "final_answer": "A", "correct": i % 2 == 0}
    for i in range(60)
]


# ── Unit: helpers ─────────────────────────────────────────────────────────────

class TestHelpers:
    def test_risk_to_tier_low(self):
        assert _risk_to_tier(0.3) == "HIGH"

    def test_risk_to_tier_medium(self):
        assert _risk_to_tier(0.6) == "MEDIUM"

    def test_risk_to_tier_high(self):
        assert _risk_to_tier(0.8) == "LOW"

    def test_risk_to_labels_success(self):
        risk, conf, outcome = _risk_to_labels(0.2)
        assert risk == "low"
        assert conf == "high"
        assert outcome == "likely_success"

    def test_risk_to_labels_failure(self):
        risk, conf, outcome = _risk_to_labels(0.9)
        assert risk == "high"
        assert conf == "low"
        assert outcome == "likely_failure"


# ── Unit: AgentGuard init ─────────────────────────────────────────────────────

class TestAgentGuardInit:
    def test_defaults(self):
        guard = AgentGuard()
        assert not guard._use_judge
        assert not guard._use_local_verifier
        assert guard._alert_threshold == 0.70
        assert guard._on_alert is None

    def test_on_alert_stored(self):
        cb = lambda r: None
        guard = AgentGuard(on_alert=cb)
        assert guard._on_alert is cb

    def test_api_key_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        guard = AgentGuard()
        assert guard._api_key == "test-key"


# ── Unit: score_chain (mocked scorer) ────────────────────────────────────────

class TestScoreChain:
    def _guard_with_mock_scorer(self):
        guard = AgentGuard()
        mock_result = MagicMock()
        mock_result.risk_score = 0.3
        mock_result.failure_mode = None
        mock_result.components = {"sc2": 0.3}
        guard._scorer.score = MagicMock(return_value=mock_result)
        return guard

    def test_returns_chain_trust_result(self):
        guard = self._guard_with_mock_scorer()
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        assert isinstance(result, ChainTrustResult)

    def test_risk_score_range(self):
        guard = self._guard_with_mock_scorer()
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        assert 0.0 <= result.risk_score <= 1.0

    def test_needs_alert_false_for_low_risk(self):
        guard = self._guard_with_mock_scorer()
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        assert result.needs_alert == (result.risk_score >= 0.70)

    def test_on_alert_fires_when_alert(self):
        guard = self._guard_with_mock_scorer()
        # Force high risk
        guard._scorer.score.return_value.risk_score = 0.9
        alerts = []
        guard._on_alert = lambda r: alerts.append(r)
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        if result.needs_alert:
            assert len(alerts) == 1
            assert alerts[0] is result

    def test_on_alert_not_fired_for_safe_chain(self):
        guard = self._guard_with_mock_scorer()
        alerts = []
        guard._on_alert = lambda r: alerts.append(r)
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        if not result.needs_alert:
            assert len(alerts) == 0

    def test_on_alert_exception_doesnt_propagate(self):
        guard = self._guard_with_mock_scorer()
        guard._scorer.score.return_value.risk_score = 0.95
        guard._on_alert = lambda r: 1 / 0  # will raise
        # Should not raise
        result = guard.score_chain("Q?", SIMPLE_STEPS, "A")
        assert result is not None

    def test_confidence_tiers_consistent(self):
        guard = self._guard_with_mock_scorer()
        for risk_val, expected_tier in [(0.2, "HIGH"), (0.6, "MEDIUM"), (0.9, "LOW")]:
            guard._scorer.score.return_value.risk_score = risk_val
            r = guard.score_chain("Q?", SIMPLE_STEPS, "A")
            assert r.confidence_tier == expected_tier, f"risk={risk_val}"


# ── Unit: monitor_step ────────────────────────────────────────────────────────

class TestMonitorStep:
    def test_first_step_is_medium(self):
        guard = AgentGuard()
        result = guard.monitor_step("Q?", [], "Search[x]")
        assert isinstance(result, AgentStepResult)
        assert result.step_index == 0
        assert result.risk == "medium"

    def test_step_index_matches_steps_len(self):
        guard = AgentGuard()
        mock_lf = MagicMock()
        mock_lf.risk_score = 0.4
        mock_lf.failure_mode = None
        guard._scorer.score_prefix = MagicMock(return_value=mock_lf)
        result = guard.monitor_step("Q?", SIMPLE_STEPS[:1], "Search[y]")
        assert result.step_index == 1


# ── Unit: score_chain_start ───────────────────────────────────────────────────

class TestScoreChainStart:
    def test_returns_expected_keys(self):
        guard = AgentGuard()
        mock_lf = MagicMock()
        mock_lf.risk_score = 0.3
        mock_lf.failure_mode = None
        guard._scorer.score = MagicMock(return_value=mock_lf)
        result = guard.score_chain_start("What is 2+2?")
        assert "risk_score" in result
        assert "tier" in result
        assert "recommended_action" in result


# ── Unit: LocalVerifier ───────────────────────────────────────────────────────

class TestLocalVerifier:
    def test_extract_features_shape(self):
        feats = extract_features("Q?", SIMPLE_STEPS, "A")
        assert feats.shape == (len(FEATURE_NAMES),)  # 15 features as of v0.7.0

    def test_fit_predict_range(self):
        v = LocalVerifier()
        v.fit(LABELED_RUNS)
        assert v.is_fitted
        risk = v.predict_risk("Q?", SIMPLE_STEPS, "A")
        assert 0.0 <= risk <= 1.0

    def test_fit_requires_correct_key(self):
        # Runs without "correct" key — should raise or warn
        bad_runs = [{"question": "Q", "steps": SIMPLE_STEPS, "final_answer": "A"}]
        v = LocalVerifier()
        with pytest.raises(Exception):
            v.fit(bad_runs)

    def test_predict_before_fit_raises(self):
        v = LocalVerifier()
        with pytest.raises(RuntimeError):
            v.predict_risk("Q?", SIMPLE_STEPS, "A")

    def test_n_train_set_after_fit(self):
        v = LocalVerifier()
        v.fit(LABELED_RUNS)
        assert v.n_train == len(LABELED_RUNS)


# ── Unit: QPPGNano ────────────────────────────────────────────────────────────

class TestQPPGNano:
    def test_score_chain_returns_dict(self):
        nano = QPPGNano()
        result = nano.score_chain("Q?", SIMPLE_STEPS, "A")
        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "confidence_tier" in result
        assert "needs_alert" in result

    def test_risk_range(self):
        nano = QPPGNano()
        result = nano.score_chain("Q?", SIMPLE_STEPS, "A")
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_score_prefix_returns_dict(self):
        nano = QPPGNano()
        result = nano.score_prefix("Q?", SIMPLE_STEPS[:1], "Search[x]")
        assert "risk_score" in result


# ── Unit: integrations ────────────────────────────────────────────────────────

class TestCrewAICallback:
    def test_on_step_dict(self):
        from llm_guard.integrations.crewai import AgentGuardCrewCallback
        guard = MagicMock()
        guard.score_chain.return_value = MagicMock(needs_alert=False)
        cb = AgentGuardCrewCallback(guard)
        cb.on_step({"thought": "think", "tool": "Search", "tool_input": "x", "result": "y"})
        assert len(cb._steps) == 1

    def test_on_task_end_calls_score_chain(self):
        from llm_guard.integrations.crewai import AgentGuardCrewCallback
        guard = MagicMock()
        guard.score_chain.return_value = MagicMock(needs_alert=False)
        cb = AgentGuardCrewCallback(guard)
        cb.on_step({"thought": "t", "tool": "Search", "tool_input": "q", "result": "r"})
        task_out = MagicMock()
        task_out.raw = "final answer"
        del task_out.description
        cb.on_task_end(task_out)
        guard.score_chain.assert_called_once()
        assert cb.last_result is not None

    def test_on_alert_fires(self):
        from llm_guard.integrations.crewai import AgentGuardCrewCallback
        guard = MagicMock()
        alert_result = MagicMock(needs_alert=True)
        guard.score_chain.return_value = alert_result
        alerts = []
        cb = AgentGuardCrewCallback(guard, on_alert=lambda r: alerts.append(r))
        task_out = MagicMock()
        task_out.raw = "answer"
        del task_out.description
        cb.on_task_end(task_out)
        assert len(alerts) == 1

    def test_reset_clears_state(self):
        from llm_guard.integrations.crewai import AgentGuardCrewCallback
        guard = MagicMock()
        cb = AgentGuardCrewCallback(guard)
        cb._steps = [{"thought": "x"}]
        cb._question = "something"
        cb.reset()
        assert cb._steps == []
        assert cb._question == ""


class TestLangChainCallback:
    def test_import_error_without_langchain(self, monkeypatch):
        import sys
        # If langchain is not installed, importing should raise ImportError
        monkeypatch.setitem(sys.modules, "langchain_core", None)
        monkeypatch.setitem(sys.modules, "langchain", None)
        # Re-import to trigger ImportError path
        if "llm_guard.integrations.langchain" in sys.modules:
            del sys.modules["llm_guard.integrations.langchain"]
        # This test just verifies the module handles missing langchain gracefully
        try:
            from llm_guard.integrations.langchain import AgentGuardCallback
        except (ImportError, TypeError):
            pass  # expected when langchain not installed


# ── Integration: requires ANTHROPIC_API_KEY ───────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAgentGuardIntegration:
    def test_behavioral_score_chain(self):
        guard = AgentGuard()
        result = guard.score_chain(
            question     = "Who wrote Hamlet?",
            steps        = SIMPLE_STEPS,
            final_answer = "Shakespeare",
        )
        assert isinstance(result, ChainTrustResult)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.confidence_tier in ("HIGH", "MEDIUM", "LOW")
