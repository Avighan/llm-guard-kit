# LLM Guard Kit — Production Integration Guide

**Version**: 0.6.0
**Last updated**: March 2026
**Validated on**: HotpotQA (200 chains), NQ (100 chains cross-domain)

---

## 1. What Was Changed in v0.6.0

### Problem with earlier versions
`AgentGuard` and `LLMGuard` claimed **AUROC 1.000** and **0.966–0.993** respectively. These were measured **in-sample** on the same data used for KNN calibration — a form of data leakage. On held-out data, the real performance was much lower.

### What v0.6.0 delivers

| Component | Before (v0.5) | After (v0.6) |
|---|---|---|
| `AgentGuard` core scorer | KNN step-embedding (AUROC 1.000 in-sample) | SC_OLD behavioral + optional Sonnet judge |
| Within-domain AUROC | ~0.65 (realistic KNN) | **~0.81** (SC_OLD, exp88 5-fold CV) |
| Cross-domain AUROC | ~0.55 (KNN breaks) | **~0.74** (SC_OLD + Sonnet judge, exp91) |
| Alert precision | Unknown | **0.908** at FPR ≤ 10% (exp92 conformal) |
| A2A handoff | Not implemented | `A2ATrustObject` (exp105) |
| Query diversification | Not implemented | `QueryRewriter` (exp105 architecture) |
| New exports | — | `ChainTrustResult`, `A2ATrustObject`, `QueryRewriter` |

---

## 2. Validated AUROC Numbers (Reference)

All figures are from held-out evaluation unless noted otherwise.

| Experiment | Method | Within-HP | Cross-NQ | Cost |
|---|---|---|---|---|
| exp88 (5-fold CV) | SC_OLD behavioral | **0.812 ± 0.062** | — | $0 |
| exp91 | SC_OLD behavioral | — | **0.672** | $0 |
| exp89 | Sonnet judge alone | 0.7735 | 0.7411 | $0.007/chain |
| exp89/91 | J5 = SC_OLD×1 + judge×3 | **0.7767** | 0.737 | $0.007/chain |
| exp92 | J5 conformal at α=0.10 | P=0.908, R=0.595 | — | — |
| exp107 | Mid-chain Haiku at step 2 | AUROC=0.683 | — | $0.001/step |
| exp105 | A2A trust object | ρ=+0.108 | lift=+0.105 | $0 |

**Key facts:**
- SC_OLD alone is competitive (~0.81) and costs $0 — use it first
- Sonnet judge adds +0.069 cross-domain (0.672→0.741) — worth the cost in production
- Alert threshold 0.70 gives **Precision=0.908** (91% of what you flag is actually wrong)
- 40% of failures are still missed (Recall=0.595) — this is the remaining gap

---

## 3. Quick Start

### Single-agent chain scoring

```python
from llm_guard import AgentGuard

# Mode 1: Behavioral only ($0, ~0.81 AUROC)
guard = AgentGuard()

# Mode 2: Behavioral + Sonnet judge (~$0.007/chain, ~0.74 cross-domain)
guard = AgentGuard(api_key="sk-ant-...", use_judge=True)

steps = [
    {
        "thought": "I need to find when the Eiffel Tower was built",
        "action_type": "Search",
        "action_arg": "Eiffel Tower construction year",
        "observation": "The Eiffel Tower was built from 1887 to 1889...",
    },
    {
        "thought": "It was completed in 1889",
        "action_type": "Finish",
        "action_arg": "1889",
        "observation": "",
    },
]

result = guard.score_chain(
    question="When was the Eiffel Tower built?",
    steps=steps,
    final_answer="1889",
)

print(result.confidence_tier)    # HIGH / MEDIUM / LOW
print(result.risk_score)         # 0.0–1.0
print(result.needs_alert)        # True when risk >= 0.70
print(result.failure_mode)       # "retrieval_fail" | "long_chain" | None
print(result.judge_label)        # "GOOD" | "BORDERLINE" | "POOR" | None
```

### Mid-chain monitoring (intervene before it's too late)

```python
# Call inside your agent loop BEFORE executing each step
step_result = guard.monitor_step(
    question="When was the Eiffel Tower built?",
    steps_so_far=[step1],                        # steps already done
    current_action="Search[Eiffel Tower date]",  # action about to execute
)

if step_result.risk == "high":
    # Agent is heading toward failure at step 2
    # Options: abort, retry, escalate, or flag for human review
    pass
```

Validated: Haiku mid-chain judge at step 2 achieves AUROC=0.683 (Δ+0.156 vs SC_OLD at same step). The step-2 signal is the strongest early-failure indicator.

---

## 4. Agent-to-Agent (A2A) Trust Handoff

### Architecture

```
Agent A (ReAct chain)
    │
    ▼
guard.generate_trust_object(question, steps, final_answer)
    │
    ▼
A2ATrustObject {
    answer: "1889",
    risk_score: 0.23,
    confidence_tier: "HIGH",
    failure_mode: None,
    judge_label: "GOOD",
    downstream_hint: "proceed",
    should_rewrite: False,
}
    │  (serialise to JSON for transport)
    ▼
Agent B receives trust object → conditions its strategy
```

### Producing the trust object (Agent A side)

```python
from llm_guard import AgentGuard, A2ATrustObject

guard = AgentGuard(api_key="sk-ant-...", use_judge=True)

trust = guard.generate_trust_object(
    question="When was the Eiffel Tower built?",
    steps=steps,
    final_answer="1889",
)

# Serialise for JSON/queue/API transport
payload = trust.to_dict()

# Wire format:
# {
#   "answer": "1889",
#   "risk_score": 0.23,
#   "confidence_tier": "HIGH",
#   "failure_mode": null,
#   "judge_label": "GOOD",
#   "downstream_hint": "proceed",
#   "should_rewrite": false,
#   "behavioral_components": {"sc1": 0.0, "sc2": 0.32, ...},
#   "temporal_validity": null
# }
```

### Consuming the trust object (Agent B side)

```python
from llm_guard import A2ATrustObject, QueryRewriter

# Deserialise
trust = A2ATrustObject.from_dict(payload)

# Check if Agent B should rewrite the query
if trust.should_rewrite:                    # True when tier == "LOW"
    rewriter = QueryRewriter(api_key="sk-ant-...")
    variants = rewriter.rewrite(original_question, trust)

    # Use diversified queries
    agent_b_primary_query   = variants.paraphrase   # same intent, different words
    agent_b_fallback_query  = variants.decomposed   # simpler sub-question
    agent_b_alternative     = variants.alternative  # completely different angle
else:
    # Proceed with original question
    agent_b_primary_query = original_question

# Downstream hint tells Agent B exactly what to do:
# "proceed"              → HIGH confidence, run normally
# "proceed_with_caution" → MEDIUM, validate key claims
# "rewrite_query"        → LOW, use QueryRewriter output
# "rewrite_and_verify"   → LOW + retrieval failure, rewrite + verify sources
# "escalate_to_human"    → POOR judge label, send to human review
print(trust.downstream_hint)
```

### Confidence tier thresholds

| Tier | Risk score | Meaning | Action |
|---|---|---|---|
| HIGH | < 0.50 | Agent is likely correct | Proceed normally |
| MEDIUM | 0.50–0.70 | Uncertain | Proceed with monitoring |
| LOW | ≥ 0.70 | Likely wrong | Rewrite query or escalate |

Empirical baseline failure rates (exp105):
- P(chain is wrong) overall: **58%** (HotpotQA dataset)
- P(B fails | A fails): **67.9%** — errors are correlated
- P(B fails | A correct): **57.4%** — baseline
- Lift from trust object: **+10.5 pp** (weak but real signal)

---

## 5. Query Rewriter for A2A Diversification

The fundamental cause of correlated agent errors: both agents share the same
underlying model and use similar search strategies → same knowledge gaps.

The fix: when Agent A has `confidence_tier = "LOW"`, Agent B should search
with **different queries** before running its own chain.

```python
from llm_guard import QueryRewriter

rewriter = QueryRewriter(
    api_key="sk-ant-...",
    model="claude-haiku-4-5-20251001",  # default, cheap (~$0.0005/call)
    risk_threshold=0.70,
)

# Check if rewrite is needed
if rewriter.should_rewrite(trust):
    result = rewriter.rewrite(question, trust)

    print(result.paraphrase)   # "What year did construction on the Eiffel Tower finish?"
    print(result.decomposed)   # "Who commissioned the Eiffel Tower and when?"
    print(result.alternative)  # "What major structures were built in Paris in the 1880s?"

# Or use the convenience wrapper (returns [] when no rewrite needed)
variants = rewriter.rewrite_if_needed(question, trust)
```

**Failure-mode-aware hints**: the rewriter tailors its prompt based on
`trust.failure_mode`:
- `retrieval_fail` → "try different terminology or a different source angle"
- `repeated_query` → "avoid any terms from the original query"
- `long_chain` → "simplify to the single most important fact"
- `no_evidence` → "start with foundational background facts"

---

## 6. Calibration (Optional — Improves Density Signals)

SC_OLD behavioral signals work out-of-the-box with no calibration.
Calibration improves the GMM and obs-pool density signals (+~0.04 AUROC).

```python
# After running your agent on some queries, feed correct runs for calibration
guard.fit_from_agent_runs(
    runs=[
        {
            "question": "...",
            "steps": [...],
            "final_answer": "...",
            "chain_correct": True,   # optional filter key
        },
        ...
    ],
    correct_only=True,   # only use correct runs for density calibration
)
# Requires ≥ 20 runs to activate density signals
# Below that threshold, behavioral SC_OLD is already active and effective
```

---

## 7. Binary Alerting — What to Expect

From exp92 conformal calibration (J5 ensemble, HotpotQA):

| Alert threshold | Precision | Recall | FPR | Coverage |
|---|---|---|---|---|
| 0.70 (default) | **0.908** | 0.595 | ≤ 10% | 38% |
| 0.75 | 0.903 | 0.241 | ≤ 5% | 15.5% |

- **Coverage**: fraction of chains that trigger an alert
- **Precision**: fraction of alerts that are actual failures
- **Recall**: fraction of all failures that are caught (40% are missed)

**Practical implication**: with a 58% baseline failure rate, the guard lifts
alert precision from 58% (flag everything) to **91%** at 10% FPR. The remaining
40% of failures that slip through are the hard cases where even the judge is uncertain.

For higher recall, use the two-stage OR system (exp107):
- Haiku mid-chain at step 2 OR J5 final → **Recall=0.416**, Precision=0.740

---

## 8. Cost Reference

| Mode | Cost per chain | AUROC |
|---|---|---|
| Behavioral only (`use_judge=False`) | **$0** | ~0.81 within |
| + Sonnet judge (`use_judge=True`) | ~$0.007 | ~0.74 cross |
| + Mid-chain Haiku step 2 | ~$0.001/step | AUROC 0.683 at step 2 |
| + Query rewriter (Haiku) | ~$0.0005/rewrite | +diversification |

---

## 9. Open Problems (Known Gaps)

These are **not solved** by this release:

| Problem | Status | Path forward |
|---|---|---|
| Recall = 59.5% (40% of failures missed) | Open | Lower alert threshold (α=0.20) or use OR two-stage |
| Cross-domain conformal breaks | Open | Per-domain recalibration with 25 labeled chains |
| A2A correlation ρ=+0.108 (weak lift) | Partially addressed | QueryRewriter for structural diversification |
| Real search vs fake-search | Unvalidated | Test on real retrieval (Serper, Tavily) |
| 1-step chains (SC_OLD = 0.50) | Mitigated | Sonnet judge works on 1-step chains (AUROC 0.774) |

---

## 10. File Reference

| File | Purpose |
|---|---|
| `llm_guard/agent_guard.py` | `AgentGuard` — SC_OLD + Sonnet judge, main class |
| `llm_guard/trust_object.py` | `A2ATrustObject` — A2A trust envelope dataclass |
| `llm_guard/query_rewriter.py` | `QueryRewriter` — query diversification |
| `llm_guard/__init__.py` | Public API exports |
| `qppg_service/label_free_scorer.py` | `LabelFreeScorer` — SC_OLD behavioral signals |
| `qppg_service/server.py` | FastAPI HTTP server |
| `experiments/exp88_*.py` | SC_OLD 5-fold CV validation |
| `experiments/exp89_sonnet_judge.py` | J5 ensemble validation |
| `experiments/exp91_*.py` | Cross-domain NQ validation |
| `experiments/exp92_*.py` | Conformal calibration (Precision=0.908) |
| `experiments/exp105_a2a_handoff_confidence.py` | A2A trust object empirical validation |
| `experiments/exp107_midchain_recall.py` | Mid-chain intervention validation |
