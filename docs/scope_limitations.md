# Scope and Limitations of llm-guard-kit

*Last updated: 2026-03-14. Based on validated experiments exp92, exp113, exp142, exp156, exp_publishability, exp_ptrue_significance.*

---

## What Is Validated

### Behavioural ensemble (SC_OLD) — within-domain

| Domain | AUROC | n | CI (95%) | Status |
|--------|-------|---|----------|--------|
| HotpotQA (within-domain) | 0.817 | 200 | [0.74, 0.89] | ✅ Validated |
| TriviaQA (cross-domain) | 0.659 | 1000 | [0.614, 0.705] | ✅ Validated |
| 2WikiMultiHop (cross-domain) | 0.703 | 200 | [0.628, 0.775] | ✅ Validated |
| MuSiQue (cross-domain) | 0.613 | 200 | [0.509, 0.710] | ⚠ CI too wide |
| Natural Questions (cross-domain) | 0.524 | 200 | [0.447, 0.609] | ❌ Near-random |

**Takeaway:** Behavioural features reliably detect multi-hop reasoning failures (HotpotQA,
2Wiki, TriviaQA). They do not generalise to open-domain factoid QA (NQ) or
harder compositional reasoning (MuSiQue CI crosses 0.5).

### Alert precision — J5 (SC_OLD + Sonnet judge) at FPR ≤ 10%

| Metric | Value | 95% CI | n |
|--------|-------|--------|---|
| Alert precision | 0.896 | [0.826, 0.960] | n_alert=77, n=200 |
| Alert recall | 0.595 | — | — |
| FPR | 0.095 | — | — |

**Takeaway:** When the system flags a chain as HIGH-RISK (FPR ≤ 10%), it is wrong
≥ 83% of the time (lower CI bound). High-precision alerting is validated.

### P(True) Haiku judge — cross-domain (exp142 + exp_ptrue_significance)

| Domain | P(True) AUROC | 95% CI | n |
| ------ | ------------- | ------ | - |
| HotpotQA-test | 0.759 | [0.638, 0.871] | 60 |
| TriviaQA-v100 | 0.682 | [0.555, 0.801] | 60 |
| 2WikiMultiHop | 0.758 | [0.639, 0.864] | 60 |

P(True) + behavioral ensemble (2WikiMultiHop): **0.817 [0.706, 0.912]**

**Takeaway:** P(True) (Haiku rating 1–5) is statistically significant on all 3 validated
domains. All lower CI bounds > 0.5 with n=60. The 2WikiMultiHop ensemble reaches AUROC ≥ 0.80.
Zero API calls required if chains are cached.

### Stream guard (mid-chain abort at step 2)

| Metric | Value | n | Source |
| ------ | ----- | - | ------ |
| AUROC (Haiku-blended, 59.5% cache) | 0.666 | 200 | exp113 |
| AUROC (cache-only estimate) | 0.682 | 200 | exp_publishability |
| Cache-only 95% CI | [0.607, 0.755] | 200 | exp_publishability |
| Precision at n_flagged=21 | 0.714 | 200 | exp113 |

**Takeaway:** Stream guard at step 2 provides useful early-warning signal
(AUROC ≈ 0.67), but is weaker than the full-chain signal (AUROC 0.817).
Use for latency-sensitive applications only; prefer full-chain scoring when time allows.

---

## Naive Baseline Comparison

A trivial baseline (score = number_of_search_steps / 7.0) gives the following AUROCs:

| Domain | n | Naive AUROC | SC_OLD AUROC | SC_OLD advantage |
|--------|---|-------------|--------------|-----------------|
| HotpotQA | 200 | 0.554 | 0.817 | **+0.263** |
| TriviaQA | 1000 | 0.720 | 0.659 | −0.061 |
| 2WikiMultiHop | 200 | 0.551 | 0.703 | **+0.151** |
| MuSiQue | 200 | 0.625 | 0.613 | −0.012 |
| NQ | 200 | 0.547 | 0.524 | −0.023 |

**Observation:** SC_OLD substantially outperforms the naive baseline on multi-hop
reasoning tasks (HotpotQA, 2Wiki). On simpler tasks (TriviaQA, NQ), agents either
finish quickly when correct or loop when wrong — making raw step-count equally or
more predictive. SC_OLD's behavioural features add the most value on
complex multi-hop tasks where failure modes are richer.

---

## Known Limitations

### L1 — P(True) on exp134–137 (n=37) vs exp142 (n=60×3, validated)

- **exp134–137:** n=37 test chains — confidence intervals were wide (±0.05–0.10 AUROC).
  Those gains (isotonic calibration AUROC 0.871, ptrue_weight 0.876) are directionally
  real but cannot be claimed as statistically significant at n=37.
- **exp142 validation (n=60 per domain, exp_ptrue_significance.py, $0):**

| Domain | P(True) AUROC | 95% CI | Significant? |
| ------ | ------------- | ------ | ------------ |
| HotpotQA-test | 0.759 | [0.638, 0.871] | ✅ YES |
| TriviaQA-v100 | 0.682 | [0.555, 0.801] | ✅ YES |
| 2WikiMultiHop | 0.758 | [0.639, 0.864] | ✅ YES |

P(True) + behavioral ensemble (2Wiki): **0.817 [0.706, 0.912]** — above 0.80 threshold.

- **Status:** P(True) is now statistically validated across 3 domains.
  Isotonic calibration gains remain preliminary (exp134–137 n=37).

### L2 — FARL taxonomy significance

- **Current:** 5 novel failure chains from FARL Phase 2 (300 iterations).
  AUROC delta +0.042; CIs overlap ([0.725, 0.814] vs [0.682, 0.775]).
- **Effect:** Taxonomy contribution is not statistically significant yet.
- **Fix:** Need ≥50 novel chains for non-overlapping 95% CIs. Requires ~1,000+
  additional FARL iterations (~$7 API budget).

### L3 — NQ is out of scope

- **Current:** AUROC 0.524 (CI [0.447, 0.609]) — crosses 0.5, not better than random.
- **Reason:** Natural Questions are single-hop factoid lookups. The agent either
  finds the answer or gives up; behavioural patterns (step diversity, observation
  alignment) are not discriminative for this failure mode.
- **Scope statement:** SC_OLD does **not** support open-domain factoid QA agents.

### L4 — MuSiQue is partial coverage

- **Current:** AUROC 0.613 but CI [0.509, 0.710] — lower bound crosses 0.5.
- **Reason:** MuSiQue requires 4-hop compositional reasoning. At n=200 with 72.5%
  failure rate, the CI is too wide to claim validity.
- **Scope statement:** MuSiQue coverage is preliminary; treat as "borderline"
  until n≥500 is tested.

### L5 — No GPU / white-box probe

- **Current:** WhiteBoxProbe exists in `llm_guard/white_box_probe.py` but only
  validated on synthetic data (AUROC 0.983 synthetic, SNR=0.45).
- **Effect:** Real deployment requires a GPU (Llama-3-8B, 4-bit, T4 sufficient).
  No real-model AUROC is available yet.
- **Fix:** `notebooks/white_box_minijudge.ipynb` (Colab T4 free tier).

---

## Out-of-Scope Domains

| Domain | Expected AUROC | Reason | Status |
|--------|---------------|--------|--------|
| Code generation / HumanEval | Unknown | Different action space (write/run/debug vs Search) | Not tested |
| Math / GSM8K | Unknown | Arithmetic chains lack retrieval steps | Not tested |
| SQL / structured output | Unknown | Tested in exp151 but not validated on failure detection | Preliminary |
| Customer service dialogs | Unknown | exp152 preliminary | Preliminary |
| NQ open-domain factoid | ≈ 0.52 | Validated near-random | Out of scope |
| Multi-agent pipelines (>2 hops) | Unknown | Trust propagation only, no QA labels | Not tested |

---

## What the System Does Well

1. **Multi-hop QA failure detection** — SC_OLD AUROC 0.82 on HotpotQA,
   0.70 on 2WikiMultiHop, 0.66 on TriviaQA (all validated with CIs).
2. **High-precision alerting** — J5 precision 0.90 (95% CI: 0.83–0.96) at FPR ≤ 10%.
3. **Zero-cost behavioural scoring** — SC_OLD requires no API calls, no labels.
4. **Mid-chain early warning** — stream_guard AUROC ≈ 0.67 at step 2.
5. **Signed trust propagation** — A2A trust object: 0.034ms sign, 100% tamper detection.
6. **Failure-mode-aware adapters** — empty_answer precision 1.000 (10/10);
   retrieval_fail precision 0.694 (R=0.371).

## What the System Does Not Do

1. Explain *why* a chain failed (only detection, not diagnosis).
2. Guarantee recall — at FPR ≤ 10%, recall is 0.595 (only 60% of failures are caught).
3. Work on code/math/SQL/factoid domains (unvalidated or known-random).
4. Replace human review for high-stakes decisions (designed as a filtering layer).
5. Operate without some domain calibration for alert thresholds (conformal calibration
   requires ≥50 correct chains from the target domain).

---

## Recommended Deployment Configuration

```python
# For multi-hop ReAct agents on web-search QA:
guard = AgentGuard(api_key=ANTHROPIC_KEY)

# Conformal calibration (collect 50+ correct chains from your domain first)
guard.conformal_alert_threshold(cal_scores, cal_labels, alpha=0.10)

# Tier-based alerting:
result = guard.score_chain(question, steps, final_answer)
if result.tier == "ALERT":
    # Precision ≥ 0.83 (lower CI bound) — route to human
    ...
elif result.tier == "REVIEW":
    # Soft flag — log and monitor
    ...

# For latency-sensitive applications, use stream_guard at step 2:
sg_result = guard.stream_guard(question, steps_so_far[:2], abort_threshold=0.65)
if sg_result.should_abort:
    # Early abort — AUROC ≈ 0.67, use for cost savings not precision
    ...
```

---

## Future Work

1. **White-box probe on Llama-3-8B** (Colab T4): expected AUROC ≥ 0.75.
   See `notebooks/white_box_minijudge.ipynb`.
2. **LoRA victim fine-tuning** (Colab A100): expected 15-30% lower failure rate
   on FARL taxonomy questions. See `notebooks/lora_victim_finetuning.ipynb`.
3. **Code/math agent chains** (2-week project): test SC_OLD feature transfer.
   See `experiments/exp_code_math_chains.py`.
4. **FARL significance** (~$7 API): 1000+ iterations → ≥50 novel chains →
   statistically significant MiniJudge improvement.
5. **P(True) at n≥200** (~$3 API): widen CI bounds on calibration gains.
