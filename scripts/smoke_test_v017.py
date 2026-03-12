"""
llm-guard-kit v0.17.0 — Pre-release smoke test
================================================
Run: python scripts/smoke_test_v017.py

Tests all major use cases end-to-end without needing an API key.
Set ANTHROPIC_API_KEY env var to enable judge-based tests (optional).

Expected output: all sections print PASS. No exceptions.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HAVE_API = bool(ANTHROPIC_KEY)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results = []

def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        results.append((label, True))
    except Exception as e:
        print(f"  {FAIL}  {label}: {e}")
        results.append((label, False))

def skip(label, reason):
    print(f"  {SKIP}  {label} ({reason})")
    results.append((label, None))


# ── Sample agent chain data ─────────────────────────────────────────────────
# A short clean chain — should score LOW risk
CORRECT_CHAIN = {
    "question": "What year was the Eiffel Tower built?",
    "steps": [
        {"thought": "I need to search for the construction date.",
         "action_type": "Search", "action_arg": "Eiffel Tower construction year",
         "observation": "The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair. It is located in Paris, France."},
        {"thought": "The answer is 1889.",
         "action_type": "Finish", "action_arg": "1889", "observation": ""},
    ],
    "final_answer": "1889",
    "correct": True,
}

# A looping chain with empty observations — should score HIGH risk
WRONG_CHAIN = {
    "question": "Who wrote Hamlet?",
    "steps": [
        {"thought": "Search for Hamlet author.",
         "action_type": "Search", "action_arg": "Hamlet author",
         "observation": ""},  # empty
        {"thought": "Nothing, try again.",
         "action_type": "Search", "action_arg": "Hamlet author",
         "observation": ""},  # empty
        {"thought": "Search again with different terms.",
         "action_type": "Search", "action_arg": "Hamlet author",
         "observation": ""},  # empty
        {"thought": "Search again.",
         "action_type": "Search", "action_arg": "Hamlet play playwright",
         "observation": ""},  # empty
        {"thought": "Still no result. I'll guess.",
         "action_type": "Finish", "action_arg": "Charles Dickens", "observation": ""},
    ],
    "final_answer": "Charles Dickens",
    "correct": False,
}

MULTI_HOP_CHAIN = {
    "question": "In which city was the author of 'Pride and Prejudice' born?",
    "steps": [
        {"thought": "Find the author of Pride and Prejudice.",
         "action_type": "Search", "action_arg": "author of Pride and Prejudice",
         "observation": "Pride and Prejudice was written by Jane Austen, published in 1813."},
        {"thought": "Find where Jane Austen was born.",
         "action_type": "Search", "action_arg": "Jane Austen birthplace",
         "observation": "Jane Austen was born on 16 December 1775 in Steventon, Hampshire, England."},
        {"thought": "The answer is Steventon.",
         "action_type": "Finish", "action_arg": "Steventon, Hampshire", "observation": ""},
    ],
    "final_answer": "Steventon, Hampshire",
    "correct": True,
}


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 1: MiniJudge — $0 local risk scoring")
print("="*60)

def test_minijudge_import():
    from llm_guard import MiniJudge
    j = MiniJudge()
    assert hasattr(j, "score")

def test_minijudge_scores_wrong_higher():
    from llm_guard import MiniJudge
    j = MiniJudge()
    r_correct = j.score(CORRECT_CHAIN["question"],
                        CORRECT_CHAIN["steps"],
                        CORRECT_CHAIN["final_answer"])
    r_wrong = j.score(WRONG_CHAIN["question"],
                      WRONG_CHAIN["steps"],
                      WRONG_CHAIN["final_answer"])
    assert 0.0 <= r_correct <= 1.0, f"correct score out of range: {r_correct}"
    assert 0.0 <= r_wrong <= 1.0, f"wrong score out of range: {r_wrong}"
    # Wrong chain has 5 steps, 4 empty observations, repeated queries — must score higher
    assert r_wrong > r_correct, \
        f"Expected wrong({r_wrong:.3f}) > correct({r_correct:.3f})"

def test_minijudge_score_chain_alias():
    from llm_guard import MiniJudge
    j = MiniJudge()
    r = j.score_chain(WRONG_CHAIN["question"],
                      WRONG_CHAIN["steps"],
                      WRONG_CHAIN["final_answer"])
    assert 0.0 <= r <= 1.0

check("MiniJudge imports and has .score()", test_minijudge_import)
check("MiniJudge: wrong chain scores higher than correct", test_minijudge_scores_wrong_higher)
check("MiniJudge: score_chain() alias works", test_minijudge_score_chain_alias)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 2: AgentGuard — behavioral scoring (no API key)")
print("="*60)

def test_agentguard_no_key():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    result = guard.score_chain(
        question=WRONG_CHAIN["question"],
        steps=WRONG_CHAIN["steps"],
        final_answer=WRONG_CHAIN["final_answer"],
    )
    assert hasattr(result, "risk_score")
    assert hasattr(result, "confidence_tier")
    assert hasattr(result, "failure_mode")
    assert 0.0 <= result.risk_score <= 1.0
    assert result.confidence_tier in ("HIGH", "MEDIUM", "LOW")

def test_agentguard_tiers():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    r_good = guard.score_chain(CORRECT_CHAIN["question"],
                               CORRECT_CHAIN["steps"],
                               CORRECT_CHAIN["final_answer"])
    r_bad  = guard.score_chain(WRONG_CHAIN["question"],
                               WRONG_CHAIN["steps"],
                               WRONG_CHAIN["final_answer"])
    assert r_bad.risk_score >= r_good.risk_score, \
        f"bad={r_bad.risk_score:.3f} good={r_good.risk_score:.3f}"

def test_agentguard_failure_mode():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    result = guard.score_chain(
        question=WRONG_CHAIN["question"],
        steps=WRONG_CHAIN["steps"],
        final_answer=WRONG_CHAIN["final_answer"],
    )
    # Repeated empty searches must trigger a failure mode
    assert result.failure_mode is not None, "Expected a failure mode for looping chain"
    print(f"\n      failure_mode={result.failure_mode}", end="")

check("AgentGuard: score_chain() without API key", test_agentguard_no_key)
check("AgentGuard: wrong chain risk > correct chain risk", test_agentguard_tiers)
check("AgentGuard: failure mode detected on looping chain", test_agentguard_failure_mode)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 3: A2A Trust Object — generate, sign, verify")
print("="*60)

def test_a2a_create():
    from llm_guard import AgentGuard, A2ATrustObject
    guard = AgentGuard()
    # generate_trust_object() scores the chain internally
    trust = guard.generate_trust_object(
        question=CORRECT_CHAIN["question"],
        steps=CORRECT_CHAIN["steps"],
        final_answer=CORRECT_CHAIN["final_answer"],
    )
    assert isinstance(trust, A2ATrustObject)
    assert trust.risk_score is not None
    assert trust.confidence_tier in ("HIGH", "MEDIUM", "LOW")

def test_a2a_sign_verify():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    trust = guard.generate_trust_object(
        question=CORRECT_CHAIN["question"],
        steps=CORRECT_CHAIN["steps"],
        final_answer=CORRECT_CHAIN["final_answer"],
    )
    trust.sign("my-secret-key-123")
    assert trust.verify("my-secret-key-123") is True
    assert trust.verify("wrong-key") is False

def test_a2a_tamper_detection():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    trust = guard.generate_trust_object(
        question=CORRECT_CHAIN["question"],
        steps=CORRECT_CHAIN["steps"],
        final_answer=CORRECT_CHAIN["final_answer"],
    )
    trust.sign("secret")
    # The HMAC covers the 'answer' field (canonical name), not 'final_answer'
    trust.answer = "tampered answer"  # tamper the signed field
    assert trust.verify("secret") is False, "Tamper of signed 'answer' field must fail verify()"

check("A2ATrustObject: generate from chain", test_a2a_create)
check("A2ATrustObject: sign + verify round-trip", test_a2a_sign_verify)
check("A2ATrustObject: tamper detected after sign", test_a2a_tamper_detection)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 4: MiniJudge + probe_ensemble_blend")
print("="*60)

def test_probe_blend():
    from llm_guard import MiniJudge, probe_ensemble_blend
    j = MiniJudge()
    mini_risk = j.score(WRONG_CHAIN["question"],
                        WRONG_CHAIN["steps"],
                        WRONG_CHAIN["final_answer"])
    ptrue_risk = 0.72  # simulated P(True) score
    blended = probe_ensemble_blend(mini_risk, ptrue_risk, alpha=0.25)
    expected = 0.25 * mini_risk + 0.75 * ptrue_risk
    assert abs(blended - expected) < 1e-6, f"blend={blended} expected={expected}"
    assert 0.0 <= blended <= 1.0

check("probe_ensemble_blend(): correct weighted average", test_probe_blend)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 5: QueryRewriter — low-confidence retry variants")
print("="*60)

def test_query_rewriter():
    from llm_guard import AgentGuard, QueryRewriter
    guard = AgentGuard()
    # First get a trust object (needed by rewriter)
    trust = guard.generate_trust_object(
        question=WRONG_CHAIN["question"],
        steps=WRONG_CHAIN["steps"],
        final_answer=WRONG_CHAIN["final_answer"],
    )
    rewriter = QueryRewriter()
    result = rewriter.rewrite(WRONG_CHAIN["question"], trust=trust, n_variants=3)
    assert hasattr(result, "variants") or isinstance(result, list)
    variants = result.variants if hasattr(result, "variants") else result
    assert len(variants) >= 1
    print(f"\n      variants={variants[:1]}", end="")

check("QueryRewriter: returns query variants for LOW tier", test_query_rewriter)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 6: AdapterRegistry — failure-mode routing")
print("="*60)

def test_adapter_retrieval_fail():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    adapter = guard.activate_adapter("retrieval_fail")
    # AdapterResult wraps config
    assert adapter is not None
    hint = getattr(adapter, "system_hint", None) or getattr(adapter.config, "system_hint", None)
    strategy = getattr(adapter, "search_strategy", None) or getattr(adapter.config, "search_strategy", None)
    assert hint is not None, "system_hint missing"
    assert strategy is not None, "search_strategy missing"
    print(f"\n      strategy={strategy}", end="")

def test_adapter_empty_answer():
    from llm_guard import AgentGuard
    guard = AgentGuard()
    adapter = guard.activate_adapter("empty_answer")
    assert adapter is not None

check("AdapterRegistry: retrieval_fail returns system_hint + search_strategy", test_adapter_retrieval_fail)
check("AdapterRegistry: empty_answer adapter returns result", test_adapter_empty_answer)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 7: WhiteBoxProbe — hidden state probe (fallback mode)")
print("="*60)

def test_whitebox_probe_fallback():
    from llm_guard import WhiteBoxProbe, ProbeResult
    probe = WhiteBoxProbe()  # no model name → fallback/synthetic mode
    result = probe.score_step(
        question=CORRECT_CHAIN["question"],
        steps=CORRECT_CHAIN["steps"],
        step_k=1,  # score at step 1
    )
    assert isinstance(result, ProbeResult), f"got {type(result)}"
    assert 0.0 <= result.hidden_risk <= 1.0

check("WhiteBoxProbe: fallback mode returns ProbeResult", test_whitebox_probe_fallback)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 8: QuickCalibrator — domain-adaptive calibration")
print("="*60)

def test_quick_calibrator_unfitted():
    from llm_guard.quick_calibration import QuickCalibrator
    qc = QuickCalibrator(min_chains=20)
    assert not qc.is_fitted
    needed = qc.min_chains_needed()
    assert needed == 20

def test_quick_calibrator_score():
    from llm_guard.quick_calibration import QuickCalibrator
    from sklearn.isotonic import IsotonicRegression
    qc = QuickCalibrator(min_chains=2)
    qc._calibrator = IsotonicRegression(out_of_bounds="clip")
    qc._calibrator.fit([0.1, 0.5, 0.9], [0, 1, 1])
    qc._is_fitted = True   # correct private attribute name
    assert qc.is_fitted

check("QuickCalibrator: is_fitted=False before fit, min_chains_needed=20", test_quick_calibrator_unfitted)
check("QuickCalibrator: is_fitted=True after manual inject", test_quick_calibrator_score)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 9: Platform integrations — import smoke test")
print("="*60)

def test_langfuse_import():
    from llm_guard.integrations.langfuse_integration import LangfuseGuard
    assert LangfuseGuard is not None

def test_langsmith_import():
    from llm_guard.integrations.langsmith_integration import LangSmithGuardEvaluator
    assert LangSmithGuardEvaluator is not None

def test_prometheus_import():
    from llm_guard.integrations.prometheus_integration import PrometheusMetricsExporter
    assert PrometheusMetricsExporter is not None

def test_datadog_import():
    from llm_guard.integrations.datadog_integration import DatadogGuard
    assert DatadogGuard is not None

def test_prometheus_grafana_json():
    from llm_guard.integrations.prometheus_integration import make_grafana_dashboard_json
    import json
    dash = make_grafana_dashboard_json()
    d = json.loads(dash)
    panels = d["dashboard"]["panels"]
    assert len(panels) >= 4, f"expected ≥4 panels, got {len(panels)}"

check("Langfuse: LangfuseGuard importable", test_langfuse_import)
check("LangSmith: LangSmithGuardEvaluator importable", test_langsmith_import)
check("Prometheus: PrometheusMetricsExporter importable", test_prometheus_import)
check("Datadog: DatadogGuard importable", test_datadog_import)
check("Prometheus: make_grafana_dashboard_json() returns valid JSON with panels", test_prometheus_grafana_json)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 10: Telemetry opt-in — no-op without token")
print("="*60)

def test_telemetry_noop():
    from llm_guard import AgentGuard
    # contribute_labels=True but no token → should NOT raise, just silently skip
    guard = AgentGuard(contribute_labels=True)
    result = guard.score_chain(
        question=CORRECT_CHAIN["question"],
        steps=CORRECT_CHAIN["steps"],
        final_answer=CORRECT_CHAIN["final_answer"],
    )
    assert result is not None  # no crash

check("Telemetry: contribute_labels=True without token is silent no-op", test_telemetry_noop)


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("USE CASE 11: With API key — judge + P(True) [optional]")
print("="*60)

if HAVE_API:
    def test_with_judge():
        from llm_guard import AgentGuard
        guard = AgentGuard(api_key=ANTHROPIC_KEY, use_judge=True)
        result = guard.score_chain(
            question=MULTI_HOP_CHAIN["question"],
            steps=MULTI_HOP_CHAIN["steps"],
            final_answer=MULTI_HOP_CHAIN["final_answer"],
        )
        assert result.judge_label in ("GOOD", "BORDERLINE", "POOR")
        print(f"\n      judge={result.judge_label} risk={result.risk_score:.3f}", end="")

    def test_ptrue():
        from llm_guard import AgentGuard
        guard = AgentGuard(api_key=ANTHROPIC_KEY)
        result = guard.score_with_ptrue(
            question=CORRECT_CHAIN["question"],
            steps=CORRECT_CHAIN["steps"],
            final_answer=CORRECT_CHAIN["final_answer"],
        )
        assert hasattr(result, "ptrue_risk")
        assert 0.0 <= result.ptrue_risk <= 1.0
        print(f"\n      ptrue_risk={result.ptrue_risk:.3f}", end="")

    def test_mini_plus_ptrue_blend():
        from llm_guard import AgentGuard, MiniJudge, probe_ensemble_blend
        guard = AgentGuard(api_key=ANTHROPIC_KEY)
        result = guard.score_with_ptrue(
            question=WRONG_CHAIN["question"],
            steps=WRONG_CHAIN["steps"],
            final_answer=WRONG_CHAIN["final_answer"],
        )
        mini_risk = MiniJudge().score(WRONG_CHAIN["question"],
                                      WRONG_CHAIN["steps"],
                                      WRONG_CHAIN["final_answer"])
        blended = probe_ensemble_blend(mini_risk, result.ptrue_risk, alpha=0.25)
        assert 0.0 <= blended <= 1.0
        print(f"\n      mini={mini_risk:.3f} ptrue={result.ptrue_risk:.3f} blend={blended:.3f}", end="")

    check("AgentGuard + judge: judge_label returned", test_with_judge)
    check("AgentGuard: score_with_ptrue() returns ptrue_risk", test_ptrue)
    check("MiniJudge + P(True) blend: end-to-end", test_mini_plus_ptrue_blend)
else:
    skip("AgentGuard + Haiku judge", "set ANTHROPIC_API_KEY to run")
    skip("score_with_ptrue()", "set ANTHROPIC_API_KEY to run")
    skip("MiniJudge + P(True) blend", "set ANTHROPIC_API_KEY to run")


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

passed  = sum(1 for _, v in results if v is True)
failed  = sum(1 for _, v in results if v is False)
skipped = sum(1 for _, v in results if v is None)

print(f"\n  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped} (set ANTHROPIC_API_KEY to run optional tests)")
print()

if failed == 0:
    print(f"  \033[32m✓ All tests passed — safe to release\033[0m\n")
else:
    print(f"  \033[31m✗ {failed} test(s) failed — fix before releasing\033[0m\n")
    for label, ok in results:
        if ok is False:
            print(f"    FAILED: {label}")
    sys.exit(1)
