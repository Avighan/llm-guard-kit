"""
Tests for llm_guard / qppg.guard
=================================
Run with:  pytest tests/test_guard.py -v

Tests are split into three categories:
  Unit        — pure Python, no API, no embedding model loaded
  Embedding   — loads sentence-transformers (no API, ~1s startup)
  Integration — requires ANTHROPIC_API_KEY (skipped if absent)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from dataclasses import fields

# ── import guard ──────────────────────────────────────────────────────────────

from llm_guard import LLMGuard, GuardResult


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_embeddings(n: int, d: int = 384, seed: int = 42) -> np.ndarray:
    """Random unit-normalised embeddings."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, d)).astype(np.float32)
    return (e / np.linalg.norm(e, axis=1, keepdims=True))


def _guard_no_api() -> LLMGuard:
    """Guard instance that will not make any real API calls."""
    return LLMGuard(api_key="fake-key-test")


def _fitted_guard(n_correct: int = 30) -> LLMGuard:
    """Guard with KNN already fitted on synthetic embeddings (no API, no ST)."""
    guard = _guard_no_api()
    emb = _make_embeddings(n_correct)
    guard._fit_knn(emb)
    return guard


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — no API, no embedding model
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuardResult:
    def test_fields_exist(self):
        field_names = {f.name for f in fields(GuardResult)}
        assert {"answer", "risk_score", "confidence", "tool_used",
                "cluster_id", "was_retried", "raw_response"} <= field_names

    def test_blindness_score_alias(self):
        r = GuardResult(answer="x", risk_score=0.42, confidence="high")
        assert r.blindness_score == 0.42

    def test_defaults(self):
        r = GuardResult(answer="a", risk_score=0.1, confidence="high")
        assert r.tool_used is None
        assert r.cluster_id is None
        assert r.was_retried is False
        assert r.raw_response == ""


class TestKNNCore:
    def test_fit_knn_sets_fitted(self):
        guard = _guard_no_api()
        assert not guard._fitted
        guard._fit_knn(_make_embeddings(20))
        assert guard._fitted

    def test_thresholds_auto_calibrated(self):
        guard = _guard_no_api()
        guard._fit_knn(_make_embeddings(50))
        assert guard._risk_low_threshold is not None
        assert guard._risk_high_threshold is not None
        assert guard._risk_low_threshold < guard._risk_high_threshold

    def test_risk_score_unfitted_returns_05(self):
        guard = _guard_no_api()
        # No embedding model, but _fitted is False so we hit the early return
        assert guard._compute_risk_score.__doc__ is not None  # method exists
        # Patch _embed so we don't load ST
        guard._embed = lambda texts: _make_embeddings(len(texts))
        assert not guard._fitted
        score = guard._compute_risk_score("anything")
        assert score == 0.5

    def test_in_distribution_lower_than_out_of_distribution(self):
        """
        KNN distance for a query drawn from the training distribution should be
        lower than for a query drawn from an entirely different distribution.
        """
        rng = np.random.default_rng(0)
        # Training set: cluster near (1,0,...) in first dimension
        train = np.zeros((40, 32), dtype=np.float32)
        train[:, 0] = 1.0 + rng.normal(0, 0.05, 40)
        train = train / np.linalg.norm(train, axis=1, keepdims=True)

        guard = _guard_no_api()
        guard._fit_knn(train)

        # In-distribution query
        in_dist = np.zeros((1, 32), dtype=np.float32)
        in_dist[0, 0] = 1.0
        in_dist = in_dist / np.linalg.norm(in_dist)

        # Out-of-distribution query (opposite side of hypersphere)
        out_dist = np.zeros((1, 32), dtype=np.float32)
        out_dist[0, 0] = -1.0
        out_dist = out_dist / np.linalg.norm(out_dist)

        guard._embed = lambda texts: in_dist
        score_in = guard._compute_risk_score("in-distribution question")

        guard._embed = lambda texts: out_dist
        score_out = guard._compute_risk_score("out-of-distribution question")

        assert score_in < score_out, (
            f"Expected in-distribution score ({score_in:.4f}) < "
            f"out-of-distribution score ({score_out:.4f})"
        )

    def test_threshold_ordering(self):
        """Low threshold should always be <= high threshold."""
        for seed in range(10):
            guard = _guard_no_api()
            guard._fit_knn(_make_embeddings(30, seed=seed))
            assert guard._risk_low_threshold <= guard._risk_high_threshold


class TestQueryRouting:
    """
    Verify routing logic using mocked _call_llm and _compute_risk_score.
    No API or embedding model needed.
    """

    def _make_mock_guard(self, risk: float):
        guard = _fitted_guard()
        guard._compute_risk_score = MagicMock(return_value=risk)
        guard._call_llm = MagicMock(return_value=("mock answer", "end_turn"))
        return guard

    def test_low_risk_returns_high_confidence(self):
        guard = self._make_mock_guard(risk=0.0)  # well below any threshold
        result = guard.query("easy question")
        assert result.confidence == "high"
        assert result.answer == "mock answer"
        assert not result.was_retried

    def test_high_risk_returns_low_confidence(self):
        guard = self._make_mock_guard(risk=999.0)  # well above any threshold
        result = guard.query("hard question")
        assert result.confidence == "low"

    def test_resource_failure_triggers_retry_low_risk(self):
        """stop_reason == max_tokens at low risk → retry with 2x tokens."""
        guard = _fitted_guard()
        guard._compute_risk_score = MagicMock(return_value=0.0)
        guard._call_llm = MagicMock(side_effect=[
            ("truncated answer", "max_tokens"),  # first call
            ("full answer", "end_turn"),          # retry
        ])
        result = guard.query("easy question", max_tokens=100)
        assert result.was_retried
        # Second call must have doubled max_tokens
        # Signature: _call_llm(system_prompt, user_prompt, max_tokens, temperature)
        second_call_args = guard._call_llm.call_args_list[1][0]
        assert second_call_args[2] == 200  # index 2 = max_tokens (100 * 2)

    def test_resource_failure_triggers_retry_high_risk(self):
        """stop_reason == max_tokens at high risk → retry (no tool applied)."""
        guard = _fitted_guard()
        guard._compute_risk_score = MagicMock(return_value=999.0)
        guard._call_llm = MagicMock(side_effect=[
            ("truncated", "max_tokens"),
            ("full answer", "end_turn"),
        ])
        result = guard.query("hard question", max_tokens=200)
        assert result.was_retried

    def test_medium_risk_applies_tool_when_available(self):
        """Medium-risk query with a matching cluster tool should set tool_used."""
        guard = _fitted_guard()
        # Use midpoint between low and high thresholds — safely in medium zone
        # (gap may be very small with random high-d embeddings; midpoint is robust)
        mid_risk = (guard._risk_low_threshold + guard._risk_high_threshold) / 2
        guard._compute_risk_score = MagicMock(return_value=mid_risk)
        guard._call_llm = MagicMock(return_value=("answer with tool", "end_turn"))

        # Inject a fake tool and cluster center
        guard._tools = {
            "0": {
                "tool_name": "error_fix_0",
                "system_addition": "Always show your work.",
                "cluster_idx": 0,
                "cluster_size": 5,
            }
        }
        guard._cluster_centers = np.zeros((1, 384), dtype=np.float32)
        guard._embed = MagicMock(return_value=np.zeros((1, 384), dtype=np.float32))

        result = guard.query("medium difficulty question")
        assert result.tool_used == "error_fix_0"
        assert result.cluster_id == 0

    def test_get_stats(self):
        guard = _fitted_guard()
        guard._call_llm = MagicMock(return_value=("x", "end_turn"))
        guard.total_calls = 5
        guard.total_input_tokens = 1000
        guard.total_output_tokens = 500
        stats = guard.get_stats()
        assert stats["total_calls"] == 5
        assert stats["fitted"] is True
        assert "cost_usd" in stats
        assert "risk_thresholds" in stats


class TestFitPaths:
    def test_fit_requires_enough_examples(self):
        guard = _guard_no_api()
        guard._embed = MagicMock(return_value=_make_embeddings(3))
        # Only 3 examples but n_neighbors=5 → k is clamped to 2 (len-1)
        # Should still fit without error (k = min(n_neighbors, len-1))
        guard.fit(["q1", "q2", "q3"])
        assert guard._fitted

    def test_fit_returns_self_for_chaining(self):
        guard = _guard_no_api()
        guard._embed = MagicMock(return_value=_make_embeddings(10))
        result = guard.fit(["q"] * 10)
        assert result is guard

    def test_fit_from_execution_filters_incorrect(self):
        guard = _guard_no_api()
        # 8 questions; 6 correct (indices 0,2,3,4,5,6), 2 wrong (indices 1,7)
        # n_neighbors=5 requires at least 6 correct examples
        answers = ["correct", "wrong", "correct", "correct",
                   "correct", "correct", "correct", "wrong"]
        guard._embed = MagicMock(return_value=_make_embeddings(6))

        idx = [0]
        def mock_llm(sys, usr, max_tokens=500, temperature=0.0):
            ans = answers[idx[0] % len(answers)]
            idx[0] += 1
            return ans, "end_turn"

        guard._call_llm = mock_llm

        def verifier(q, r):
            return r == "correct"

        questions = [f"q{i}" for i in range(8)]
        guard.fit_from_execution(questions, verifier_fn=verifier)
        assert guard._fitted

    def test_fit_from_execution_raises_if_too_few_correct(self):
        guard = _guard_no_api()
        guard._call_llm = MagicMock(return_value=("wrong", "end_turn"))

        with pytest.raises(ValueError, match="correctly-answered"):
            guard.fit_from_execution(
                ["q1", "q2", "q3"],
                verifier_fn=lambda q, r: False,  # always wrong
            )

    def test_fit_from_consistency_raises_if_too_few_agree(self):
        guard = _guard_no_api()
        responses = ["a", "b", "c", "d", "e"]  # all different → no agreement
        idx = [0]
        def mock_llm(sys, usr, max_tokens=500, temperature=0.0):
            r = responses[idx[0] % len(responses)]
            idx[0] += 1
            return r, "end_turn"

        guard._call_llm = mock_llm

        with pytest.raises(ValueError, match="consistently-answered"):
            guard.fit_from_consistency(["q1", "q2"], n_samples=5)


class TestClusterFailures:
    def test_diagnose_returns_empty_for_few_failures(self):
        guard = _guard_no_api()
        guard._embed = MagicMock(return_value=_make_embeddings(3))
        result = guard.diagnose(["q1", "q2", "q3"], ["a1", "a2", "a3"])
        assert result == []

    def test_cluster_failures_returns_labels_centers_k(self):
        guard = _guard_no_api()
        emb = _make_embeddings(30)
        labels, centers, k = guard._cluster_failures(emb, n_clusters=3)
        assert len(labels) == 30
        assert centers.shape[0] == 3
        assert k == 3

    def test_cluster_failures_auto_selects_k(self):
        guard = _guard_no_api()
        emb = _make_embeddings(30)
        labels, centers, k = guard._cluster_failures(emb, n_clusters=None)
        assert 2 <= k <= 10


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding tests — loads sentence-transformers (~1s), no API
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestRealEmbeddings:
    """Requires sentence-transformers. Marked slow; skipped with -m 'not slow'."""

    def test_embed_returns_correct_shape(self):
        guard = _guard_no_api()
        emb = guard._embed(["hello world", "goodbye world"])
        assert emb.shape == (2, 384)  # all-MiniLM-L6-v2 is 384-dim

    def test_embed_is_normalised(self):
        guard = _guard_no_api()
        emb = guard._embed(["test sentence"])
        norm = float(np.linalg.norm(emb[0]))
        assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm}"

    def test_fit_and_score_with_real_embeddings(self):
        math_qs = [
            "What is 12 + 7?",
            "What is 15 * 4?",
            "Compute 100 / 5.",
            "What is 8 squared?",
            "Find 3% of 200.",
            "What is the square root of 144?",
            "What is 2^10?",
        ]
        nonsense = "Describe the flavour of quantum bureaucracy in haiku form."

        guard = _guard_no_api()
        guard.fit(math_qs)
        assert guard._fitted

        guard._embed_real = guard._embed  # keep reference

        # In-distribution: math question similar to training set
        score_in  = guard._compute_risk_score("What is 25 + 36?")
        # Out-of-distribution: completely unrelated
        score_out = guard._compute_risk_score(nonsense)

        assert score_in < score_out, (
            f"In-dist score ({score_in:.4f}) should be < out-dist ({score_out:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — require ANTHROPIC_API_KEY
# ═══════════════════════════════════════════════════════════════════════════════

import os

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestIntegration:
    """End-to-end tests using real API. Skipped without API key."""

    def test_fit_and_query_end_to_end(self):
        guard = LLMGuard()  # reads key from env

        math_qs = [
            "What is 12 + 7?",
            "What is 15 * 4?",
            "Compute 100 / 5.",
            "What is 8 squared?",
            "Find 3% of 200.",
            "What is the square root of 144?",
            "What is 2^10?",
            "What is 50 - 18?",
            "What is 7 * 9?",
        ]
        guard.fit(math_qs)
        assert guard._fitted

        result = guard.query(
            "What is 25 + 36?",
            system_prompt="Answer with just the number.",
            max_tokens=50,
        )
        assert result.answer
        assert result.confidence in ("high", "medium", "low")
        assert result.risk_score >= 0
        assert "61" in result.answer or "61" in result.raw_response

    def test_fit_from_consistency(self):
        guard = LLMGuard()
        questions = [
            "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?",
            "What is 5 + 5?", "What is 6 + 6?", "What is 7 + 7?",
            "What is 8 + 8?",
        ]
        guard.fit_from_consistency(
            questions,
            n_samples=3,
            system_prompt="Answer with just the number.",
            agreement_threshold=0.6,
        )
        assert guard._fitted

    def test_get_stats_after_query(self):
        guard = LLMGuard()
        math_qs = [f"What is {i} + {i+1}?" for i in range(8)]
        guard.fit(math_qs)
        guard.query("What is 10 + 11?", max_tokens=20)
        stats = guard.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_input_tokens"] > 0
        assert stats["cost_usd"] >= 0
