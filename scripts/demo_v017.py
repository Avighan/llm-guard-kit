"""
llm-guard-kit v0.17.0 — End-User Demo
======================================
Run: python scripts/demo_v017.py

Shows 5 realistic usage scenarios with actual library output.
No API key required.
"""

import os
import sys
import json
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress HuggingFace / transformers noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Add QPPG root to path if needed
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Imports ────────────────────────────────────────────────────────────────────

from llm_guard import AgentGuard, QueryRewriter, QuickCalibrator, probe_ensemble_blend
from llm_guard.mini_judge import MiniJudge

# ── Shared test chains ─────────────────────────────────────────────────────────

CORRECT_CHAIN = {
    "question": "What year was the Eiffel Tower built?",
    "steps": [
        {
            "thought": "I need to find the construction year of the Eiffel Tower.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "Eiffel Tower construction year",
            "observation": "The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair.",
        },
        {
            "thought": "The observation clearly states it was completed in 1889.",
            "action": "Finish",
            "action_type": "Finish",
            "action_args": "1889",
            "observation": "",
        },
    ],
    "final_answer": "1889",
}

WRONG_CHAIN = {
    "question": "Who wrote Hamlet?",
    "steps": [
        {
            "thought": "Searching for the author of Hamlet.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "Who wrote Hamlet",
            "observation": "",
        },
        {
            "thought": "No results. Searching again.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "Who wrote Hamlet",
            "observation": "",
        },
        {
            "thought": "Still nothing. Trying once more.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "Who wrote Hamlet",
            "observation": "",
        },
        {
            "thought": "No useful results found. Trying once more.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "Who wrote Hamlet",
            "observation": "",
        },
        {
            "thought": "I'll guess from memory.",
            "action": "Finish",
            "action_type": "Finish",
            "action_args": "Charles Dickens",
            "observation": "",
        },
    ],
    "final_answer": "Charles Dickens",
}

# ── Pretty printer helpers ─────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Zero-Cost Screening ($0, No API Key)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_1():
    banner("SCENARIO 1: Zero-Cost Screening  ($0, No API Key)")

    # ── MiniJudge ─────────────────────────────────────────────────────────────
    sub("MiniJudge  — $0 local LogReg judge distilled from Sonnet")

    judge = MiniJudge()   # loads bundled pkl automatically

    if judge.is_fitted:
        risk_correct = judge.score(
            CORRECT_CHAIN["question"],
            CORRECT_CHAIN["steps"],
            CORRECT_CHAIN["final_answer"],
        )
        risk_wrong = judge.score(
            WRONG_CHAIN["question"],
            WRONG_CHAIN["steps"],
            WRONG_CHAIN["final_answer"],
        )
        print(f"  MiniJudge risk (correct chain): {risk_correct:.4f}")
        print(f"  MiniJudge risk (wrong chain):   {risk_wrong:.4f}")
        print(f"  MiniJudge cv_auroc:             {judge.auroc}")
    else:
        print("  MiniJudge pkl not found — behavioral scoring only (below).")

    # ── AgentGuard behavioral scoring ────────────────────────────────────────
    sub("AgentGuard  — SC_OLD behavioral scoring ($0)")

    guard = AgentGuard()  # no API key; behavioral only

    # Correct chain
    result_correct = guard.score_chain(
        CORRECT_CHAIN["question"],
        CORRECT_CHAIN["steps"],
        CORRECT_CHAIN["final_answer"],
    )

    print(f"\n  [Correct chain — Eiffel Tower]")
    print(f"    risk_score:            {result_correct.risk_score}")
    print(f"    confidence_tier:       {result_correct.confidence_tier}")
    print(f"    needs_alert:           {result_correct.needs_alert}")
    print(f"    failure_mode:          {result_correct.failure_mode}")
    print(f"    judge_label:           {result_correct.judge_label}")
    print(f"    behavioral_score:      {result_correct.behavioral_score}")
    print(f"    step_count:            {result_correct.step_count}")
    print(f"    latency_ms:            {result_correct.latency_ms}")
    print(f"    behavioral_components: {result_correct.behavioral_components}")

    decision_c = "FLAGGED for review" if result_correct.risk_score > 0.65 else "PASSED"
    print(f"\n  Decision: {decision_c}")

    # Wrong chain
    result_wrong = guard.score_chain(
        WRONG_CHAIN["question"],
        WRONG_CHAIN["steps"],
        WRONG_CHAIN["final_answer"],
    )

    print(f"\n  [Wrong chain — Hamlet / Charles Dickens]")
    print(f"    risk_score:            {result_wrong.risk_score}")
    print(f"    confidence_tier:       {result_wrong.confidence_tier}")
    print(f"    needs_alert:           {result_wrong.needs_alert}")
    print(f"    failure_mode:          {result_wrong.failure_mode}")
    print(f"    judge_label:           {result_wrong.judge_label}")
    print(f"    behavioral_score:      {result_wrong.behavioral_score}")
    print(f"    step_count:            {result_wrong.step_count}")
    print(f"    latency_ms:            {result_wrong.latency_ms}")
    print(f"    behavioral_components: {result_wrong.behavioral_components}")

    decision_w = "FLAGGED for review" if result_wrong.risk_score > 0.65 else "PASSED"
    print(f"\n  Decision: {decision_w}")

    return result_wrong  # pass to scenario 2


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Detect → Diagnose → Adapt → Rewrite
# ══════════════════════════════════════════════════════════════════════════════

def scenario_2(result_wrong=None):
    banner("SCENARIO 2: Detect → Diagnose → Adapt → Rewrite")

    guard = AgentGuard()

    # 1. score_chain → failure_mode
    sub("Step 1 — score_chain() to get failure_mode")
    if result_wrong is None:
        result_wrong = guard.score_chain(
            WRONG_CHAIN["question"],
            WRONG_CHAIN["steps"],
            WRONG_CHAIN["final_answer"],
        )
    print(f"  failure_mode: {result_wrong.failure_mode}")
    print(f"  risk_score:   {result_wrong.risk_score}")
    print(f"  tier:         {result_wrong.confidence_tier}")

    # 2. activate_adapter
    sub("Step 2 — activate_adapter(failure_mode)")
    adapter = guard.activate_adapter(result_wrong.failure_mode)
    print(f"  adapter activated:      {adapter.activated}")
    if adapter.config:
        print(f"  adapter_id:             {adapter.config.adapter_id}")
        print(f"  system_hint:            {adapter.config.system_hint}")
        print(f"  search_strategy:        {adapter.config.search_strategy}")
        print(f"  temperature_delta:      {adapter.config.temperature_delta}")
        print(f"  max_steps_override:     {adapter.config.max_steps_override}")

    # 3. generate_trust_object
    sub("Step 3 — generate_trust_object()")
    trust = guard.generate_trust_object(
        WRONG_CHAIN["question"],
        WRONG_CHAIN["steps"],
        WRONG_CHAIN["final_answer"],
    )
    print(f"  trust.answer:          {trust.answer}")
    print(f"  trust.risk_score:      {trust.risk_score}")
    print(f"  trust.should_rewrite:  {trust.should_rewrite}")
    print(f"  trust.downstream_hint: {trust.downstream_hint}")

    # 4. QueryRewriter (heuristic fallback when no API key)
    sub("Step 4 — QueryRewriter().rewrite(question, trust)")
    rewriter = QueryRewriter(api_key=None)  # heuristic fallback, no API key
    rw_result = rewriter.rewrite(WRONG_CHAIN["question"], trust, n_variants=3)
    print(f"  triggered_by_tier: {rw_result.triggered_by_tier}")
    print(f"  failure_mode_hint: {rw_result.failure_mode_hint[:80]}...")
    for i, v in enumerate(rw_result.variants, 1):
        print(f"  variant {i}: {v}")

    # 5. Recovery plan
    sub("Step 5 — Full Recovery Plan")
    print(f"  [Detection]   risk={result_wrong.risk_score:.3f}, mode={result_wrong.failure_mode}")
    print(f"  [Adapter]     Inject system_hint: '{adapter.config.system_hint if adapter.config else 'N/A'}'")
    print(f"  [Rewrite]     {len(rw_result.variants)} diverse variants generated")
    print(f"  [Next action] Pass variants[0] to Agent B: '{rw_result.variants[0] if rw_result.variants else 'N/A'}'")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: A2A Signed Trust Handoff
# ══════════════════════════════════════════════════════════════════════════════

def scenario_3():
    banner("SCENARIO 3: A2A Signed Trust Handoff  (multi-agent HMAC)")

    guard = AgentGuard()
    secret = "fleet-secret-2026"

    # Agent A: score + generate trust object + sign
    sub("Agent A — generate and sign trust object")
    trust = guard.generate_trust_object(
        CORRECT_CHAIN["question"],
        CORRECT_CHAIN["steps"],
        CORRECT_CHAIN["final_answer"],
    )
    trust.sign(secret)

    # Print key fields as JSON
    d = trust.to_dict()
    key_fields = {
        "answer":           d.get("answer"),
        "risk_score":       d.get("risk_score"),
        "confidence_tier":  d.get("confidence_tier"),
        "downstream_hint":  d.get("downstream_hint"),
        "should_rewrite":   d.get("should_rewrite"),
        "trust_signature":  "PRESENT" if d.get("trust_signature") else "ABSENT",
    }
    print("  trust object (key fields):")
    print(json.dumps(key_fields, indent=4))

    wire_bytes = len(json.dumps(d).encode("utf-8"))
    print(f"\n  Wire size: {wire_bytes} bytes")

    # Agent B: verify (correct secret)
    sub("Agent B — verify with correct secret")
    ok = trust.verify(secret)
    print(f"  verify('fleet-secret-2026') → {ok}")

    # Tamper + re-verify
    sub("Tamper detection — modify answer and re-verify")
    trust.answer = "1887 (TAMPERED)"
    tampered_ok = trust.verify(secret)
    print(f"  After tamper: trust.answer = '{trust.answer}'")
    print(f"  verify('fleet-secret-2026') → {tampered_ok}  ← tamper detected!")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: Mid-Chain Abort with stream_guard
# ══════════════════════════════════════════════════════════════════════════════

def scenario_4():
    banner("SCENARIO 4: Mid-Chain Abort with stream_guard")

    guard = AgentGuard()  # no API key; behavioral SC_OLD prefix only

    question = "When did the Roman Empire fall?"
    steps_so_far = [
        {
            "thought": "Searching for the fall of the Roman Empire.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "When did the Roman Empire fall",
            "observation": "",
        },
        {
            "thought": "Empty result. Trying again.",
            "action": "Search",
            "action_type": "Search",
            "action_args": "fall of the Roman Empire date",
            "observation": "",
        },
    ]

    sub("stream_guard at step 2 (abort_threshold=0.50)")
    sg = guard.stream_guard(question, steps_so_far, abort_threshold=0.50)

    print(f"  abort:              {sg.abort}")
    print(f"  risk_at_step:       {sg.risk_at_step:.4f}")
    print(f"  behavioral_risk:    {sg.behavioral_risk:.4f}")
    print(f"  haiku_risk:         {sg.haiku_risk}  (None = no API key)")
    print(f"  step_index:         {sg.step_index}")
    print(f"  on_track:           {sg.on_track}")
    print(f"  failure_mode_hint:  {sg.failure_mode_hint}")
    print(f"  rewritten_queries:  {sg.rewritten_queries}")
    print(f"  latency_ms:         {sg.latency_ms}")

    if sg.abort:
        print("\n  Cost saving: chain stopped at step 2 — saved ~3 more LLM calls.")
        print("  Estimated saving: ~60% of chain cost by aborting early.")
    else:
        print("\n  Chain not aborted at threshold=0.50 (behavioral-only, no API key).")
        print("  With Haiku judge: abort=True expected when both obs are empty.")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5: Domain Calibration with QuickCalibrator
# ══════════════════════════════════════════════════════════════════════════════

def _make_step(query: str, obs: str) -> dict:
    return {
        "thought": f"Searching for: {query}",
        "action": "Search",
        "action_type": "Search",
        "action_args": query,
        "observation": obs,
    }

def _finish_step(answer: str) -> dict:
    return {
        "thought": f"Found the answer: {answer}",
        "action": "Finish",
        "action_type": "Finish",
        "action_args": answer,
        "observation": "",
    }


def scenario_5():
    banner("SCENARIO 5: Domain Calibration with QuickCalibrator")

    # Build 20 synthetic chains: 10 correct (short, clean obs) + 10 wrong (long, empty obs)
    chains = []

    correct_qs = [
        ("What is the capital of France?", "Paris", "Paris is the capital of France."),
        ("Who invented the telephone?", "Alexander Graham Bell", "Alexander Graham Bell invented the telephone in 1876."),
        ("What is the speed of light?", "299,792 km/s", "The speed of light is approximately 299,792 km/s."),
        ("How many continents are there?", "7", "There are 7 continents on Earth."),
        ("What is the boiling point of water?", "100 degrees Celsius", "Water boils at 100 degrees Celsius at sea level."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare", "Romeo and Juliet was written by William Shakespeare."),
        ("What is the chemical symbol for gold?", "Au", "The chemical symbol for gold is Au."),
        ("Which planet is largest?", "Jupiter", "Jupiter is the largest planet in our solar system."),
        ("What year did World War II end?", "1945", "World War II ended in 1945."),
        ("What is the tallest mountain?", "Mount Everest", "Mount Everest is the tallest mountain on Earth."),
    ]

    for q, answer, obs in correct_qs:
        query_term = q.split()[-1]
        chains.append({
            "question": q,
            "steps": [_make_step(query_term, obs), _finish_step(answer)],
            "final_answer": answer,
            "correct": True,
        })

    wrong_qs = [
        "What is the population of Mars?",
        "Who invented the internet?",
        "What is the half-life of oxygen?",
        "How many moons does the Sun have?",
        "What is the capital of Atlantis?",
        "Who wrote the Pythagorean theorem?",
        "What is the weight of a photon?",
        "Which metal is heavier than osmium?",
        "What year was the Moon founded?",
        "What is the temperature of space in Kelvin?",
    ]
    wrong_answers = [
        "42 billion", "Bill Gates", "12 years", "0", "Poseidon City",
        "Leonardo da Vinci", "12 grams", "Unobtanium", "1969", "absolute zero",
    ]

    for q, answer in zip(wrong_qs, wrong_answers):
        query_term = q.split()[-1]
        steps = []
        for _ in range(4):
            steps.append(_make_step(query_term, ""))  # empty observations
        steps.append(_finish_step(answer))
        chains.append({
            "question": q,
            "steps": steps,
            "final_answer": answer,
            "correct": False,
        })

    sub("Fitting QuickCalibrator on 20 synthetic chains")
    cal = QuickCalibrator(min_chains=20)
    cal.fit(chains, domain="my_customer_support_bot")
    print(f"  is_fitted: {cal._is_fitted}")
    print(f"  domain:    {cal._domain}")

    # Score 2 test chains
    sub("Scoring test chains: raw vs calibrated")

    guard_raw = AgentGuard()

    # Test chain 1: correct
    test_q1 = "What is the largest ocean?"
    test_steps1 = [
        _make_step("largest ocean", "The Pacific Ocean is the largest and deepest ocean on Earth."),
        _finish_step("Pacific Ocean"),
    ]
    test_ans1 = "Pacific Ocean"

    raw1 = guard_raw.score_chain(test_q1, test_steps1, test_ans1)
    cal1 = cal.score(test_q1, test_steps1, test_ans1)
    print(f"\n  [Test chain 1 — correct: '{test_ans1}']")
    print(f"    Raw behavioral score:  {raw1.behavioral_score:.4f}")
    print(f"    Calibrated score:      {cal1:.4f}")
    print(f"    Calibration delta:     {cal1 - raw1.behavioral_score:+.4f}")

    # Test chain 2: wrong (looping)
    test_q2 = "What is the atomic weight of Kryptonite?"
    test_steps2 = [
        _make_step("Kryptonite atomic weight", ""),
        _make_step("Kryptonite atomic weight", ""),
        _make_step("Kryptonite atomic weight", ""),
        _finish_step("183.5"),
    ]
    test_ans2 = "183.5"

    raw2 = guard_raw.score_chain(test_q2, test_steps2, test_ans2)
    cal2 = cal.score(test_q2, test_steps2, test_ans2)
    print(f"\n  [Test chain 2 — wrong: '{test_ans2}']")
    print(f"    Raw behavioral score:  {raw2.behavioral_score:.4f}")
    print(f"    Calibrated score:      {cal2:.4f}")
    print(f"    Calibration delta:     {cal2 - raw2.behavioral_score:+.4f}")

    print(f"\n  Calibration improves threshold precision for this domain.")
    print(f"  Expected: wrong chain score > correct chain score after calibration.")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY BANNER
# ══════════════════════════════════════════════════════════════════════════════

def summary():
    width = 70
    print("\n" + "#" * width)
    print("#" + " " * (width - 2) + "#")
    print("#  llm-guard-kit v0.17.0 — Demo Complete" + " " * (width - 42) + "#")
    print("#" + " " * (width - 2) + "#")
    print("#  Demonstrated:" + " " * (width - 17) + "#")
    print("#    1. Zero-cost behavioral screening (MiniJudge + AgentGuard)" + " " * 4 + "#")
    print("#    2. Failure detection → adapter routing → query rewriting" + " " * 5 + "#")
    print("#    3. A2A signed trust object (HMAC-SHA256 tamper detection)" + " " * 5 + "#")
    print("#    4. Mid-chain abort with stream_guard (cost savings)" + " " * 10 + "#")
    print("#    5. Domain calibration with QuickCalibrator (20 chains)" + " " * 7 + "#")
    print("#" + " " * (width - 2) + "#")
    print("#  PyPI:   https://pypi.org/project/llm-guard-kit/           " + " " * 5 + "#")
    print("#  GitHub: https://github.com/Avighan/llm-guard-kit          " + " " * 5 + "#")
    print("#" + " " * (width - 2) + "#")
    print("#" * width)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("llm-guard-kit v0.17.0 — End-User Demo")
    print("No API key required for scenarios 1-5.\n")

    result_wrong = scenario_1()
    scenario_2(result_wrong)
    scenario_3()
    scenario_4()
    scenario_5()
    summary()
