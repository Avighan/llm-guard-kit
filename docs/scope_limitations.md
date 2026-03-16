# Scope and Limitations of llm-guard-kit

*Last updated: 2026-03-16 (v0.25.0). Based on validated experiments exp92, exp113, exp142, exp156, exp_publishability, exp_ptrue_significance, exp_ptrue_expanded, exp_musique_expand, exp_musique_ptrue, exp_musique_production_eval, exp_crossdomain_retrain, exp_musique_rag, exp_musique_farl, exp_code_math_farl, exp_nonreact_auroc, exp_minijudge_v2, exp_trivia_ci, exp_mondrian_cp, exp_step_transformer, exp_step_transformer_multidomain, exp160_qara_v2, exp_zero_shot_calibrator, exp_nli_grounding, exp_math_verifier, exp_bell_test.*

---

## What Is Validated

### Behavioural ensemble (SC_OLD) — within-domain

| Domain | AUROC | n | CI (95%) | Status |
|--------|-------|---|----------|--------|
| HotpotQA (within-domain) | 0.817 | 200 | [0.74, 0.89] | ✅ Validated |
| TriviaQA (cross-domain) | 0.659 | 1000 | [0.615, 0.703] | ✅ Validated (stratified bootstrap) |
| 2WikiMultiHop (cross-domain) | 0.521 | 200 | [0.440, 0.600] | ❌ CI crosses 0.5 — NOT validated |
| MuSiQue SC_OLD (cross-domain) | 0.583 | 200 | [0.485, 0.675] | ❌ CI crosses 0.5 — NOT validated |
| MuSiQue P(True) (cross-domain) | 0.684 | 500 | [0.638, 0.728] | ✅ Validated (P(True) only) |
| Natural Questions (cross-domain) | 0.524 | 200 | [0.447, 0.605] | ❌ Near-random |

**⚠ Correction (exp_trivia_ci, 2026-03-16):** Previously reported 2WikiMultiHop AUROC 0.703
was from exp156 non-stratified bootstrap with a blended score. Stratified bootstrap on SC_OLD
standalone gives 0.521 CI[0.440, 0.600] — CI crosses 0.5, not statistically validated.
The 0.703 figure should not be cited. Only TriviaQA (0.659) is a valid cross-domain SC_OLD claim.

**Takeaway:** SC_OLD behavioural features are validated only on HotpotQA (within-domain)
and TriviaQA (cross-domain). All other cross-domain results have CIs crossing 0.5.
Use P(True) for 2Wiki, MuSiQue; do not use SC_OLD alone beyond HotpotQA/TriviaQA.

### Alert precision — J5 (SC_OLD + Sonnet judge) at FPR ≤ 10%

| Metric | Value | 95% CI | n |
|--------|-------|--------|---|
| Alert precision | 0.896 | [0.826, 0.960] | n_alert=77, n=200 |
| Alert recall | 0.595 | — | — |
| FPR | 0.095 | — | — |

**Takeaway:** When the system flags a chain as HIGH-RISK (FPR ≤ 10%), it is wrong
≥ 83% of the time (lower CI bound). High-precision alerting is validated.

### P(True) Haiku judge — cross-domain (exp142 + exp_ptrue_significance + exp_musique_ptrue)

| Domain | P(True) AUROC | 95% CI | n | Status |
| ------ | ------------- | ------ | - | ------ |
| HotpotQA-test | 0.759 | [0.638, 0.871] | 60 | ✅ Significant |
| TriviaQA-v100 | 0.682 | [0.555, 0.801] | 60 | ✅ Significant |
| 2WikiMultiHop | 0.758 | [0.639, 0.864] | 60 | ✅ Significant |
| MuSiQue (4-hop) | 0.684 | [0.638, 0.728] | 500 | ✅ Significant (tightest CI) |

P(True) + behavioral ensemble (2WikiMultiHop): **0.817 [0.706, 0.912]**

**Takeaway:** P(True) (Haiku rating 1–5) is statistically significant on all 4 tested domains,
including MuSiQue where SC_OLD fails completely (0.506). MuSiQue CI [0.638, 0.728] is the
tightest across all domains (n=500). P(True) is the recommended signal for hard compositional QA.

### MiniJudge v2 ($0 local judge, SC_OLD + embedding features)

Validated on HP 200 chains, 5-fold CV (exp_minijudge_v2):

| Model        | AUROC  | 95% CI         | Cost         |
|--------------|--------|----------------|--------------|
| MiniJudge v1 | 0.802  | [0.771, 0.833] | $0/chain     |
| MiniJudge v2 | 0.816  | [0.775, 0.857] | $0/chain     |
| Haiku judge  | ~0.620 | —              | ~$0.001/call |
| Sonnet judge | 0.774  | —              | ~$0.005/call |

Top new feature (v2): `f12_ans_question_sim` — cosine(question, final_answer), |coef|=0.539.
Embedding gain: +0.014 AUROC. Behavioral features still dominant (sc9_obs_util, search_count_norm).

### Non-ReAct / Function-calling formats

SC_OLD tested on 4 format variants via simulation on HP 200 chains (exp_nonreact_auroc):

| Format                         | AUROC | Δ vs baseline | Verdict       |
|--------------------------------|-------|---------------|---------------|
| Baseline ReAct                 | 0.708 | —             | —             |
| No-thought (function-calling)  | 0.714 | +0.006        | ✅ TRANSFERS  |
| tool_call action_type          | 0.721 | +0.013        | ✅ TRANSFERS  |
| Minimal (obs only)             | 0.714 | +0.006        | ✅ TRANSFERS  |
| Single-step factoid            | 0.755 | +0.047        | ✅ TRANSFERS  |

**Key finding:** The thought field is not load-bearing. Structural features (step count,
loop rate, observation gaps) persist in OpenAI/Claude tool-use format. SC_OLD is safe to
document as supporting function-calling agents.

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
| 2WikiMultiHop | 200 | 0.551 | 0.521 | −0.030 (SC_OLD BELOW naive!) |
| MuSiQue | 500 | — | 0.506 | — (out of scope) |
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
  Isotonic calibration gains from exp134–137 (n=37) were noise — at n=180
  (exp_isotonic_significance.py) the gain is +0.0015. P(True) raw scores are
  already well-calibrated; no isotonic step is needed in production.

### L2 — FARL taxonomy significance

- **Current (final, after 1,300 iterations total):** 16 novel failure chains.
  AUROC delta +0.0017; CIs fully overlap (Baseline [0.726, 0.812] vs FARL [0.726, 0.815]).
- **Root cause:** Feature space saturated at 2 failure modes (repeated_query, confident_wrong).
  Cycles 2–5 produced 0 additional novel chains. Lowering novelty threshold to 0.15 recovered
  11 chains in cycle 1 but subsequent cycles found nothing new.
- **MuSiQue FARL (exp_musique_farl):** 1 novel chain out of 70 iterations ($0.91 spent).
  Same saturation pattern — 4-hop compositional QA collapses to 1 failure archetype in P(True)
  feature space. Domain saturation is a structural property of single-domain FARL, not a budget issue.
- **Effect:** FARL taxonomy has qualitative value (3 failure modes documented) but
  is not statistically significant. The 16 novel chains are too few and too similar
  to move the MiniJudge AUROC beyond the SC_OLD baseline CI.
- **Path to significance:** Requires qualitatively different failure domains (code/math agents,
  multi-agent pipelines) — not more iterations on the same QA domain. Current FARL
  is domain-saturated for HotpotQA-style and MuSiQue-style ReAct agents.
  See `experiments/exp_code_math_farl.py` for code/math new action space.

### L3 — NQ is out of scope

- **Current:** AUROC 0.524 (CI [0.447, 0.609]) — crosses 0.5, not better than random.
- **Reason:** Natural Questions are single-hop factoid lookups. The agent either
  finds the answer or gives up; behavioural patterns (step diversity, observation
  alignment) are not discriminative for this failure mode.
- **Scope statement:** SC_OLD does **not** support open-domain factoid QA agents.

### L4 — MuSiQue: SC_OLD fails, P(True) + DeepVerifier succeeds

- **SC_OLD (exp_musique_expand, n=500):** AUROC 0.506 [0.501, 0.576] — essentially random.
  SC_OLD behavioural features do not discriminate MuSiQue failures.
- **P(True) alone (exp_musique_ptrue, n=500):** AUROC **0.684 [0.638, 0.728]** — significant.
  Alert precision (conformal, FPR≤10%): **0.935 [0.870, 0.985]** — stronger than HotpotQA's 0.896.
- **RAG agent (gold paragraphs, exp_musique_rag, n=199):** P(True) AUROC **0.711 [0.633, 0.783]**
  — slightly higher than web-search (0.684) because reasoning quality is more discriminable
  when the agent reads gold paragraphs. Error rate drops from 74% → 32.2% with RAG.
  SC_OLD on RAG chains: 0.540 [0.503, 0.623] — marginal improvement but still weak.
- **DeepLocalVerifier (retrained on TV+2Wiki) + P(True) ensemble:** AUROC **0.725 [0.677, 0.772]**
  — +0.041 above P(True) alone. DeepVerifier alone = 0.684 (same as P(True), different signal).
  (exp_crossdomain_retrain: trained on TV-1000 + 2Wiki-200, zero-shot to MuSiQue-500)
- **Scope statement:** For MuSiQue, deploy P(True)+DeepVerifier ensemble (AUROC 0.725);
  skip SC_OLD entirely. DeepVerifier must be retrained on multi-domain data (not HP-only).

### L5 — No GPU / white-box probe

- **Current:** WhiteBoxProbe exists in `llm_guard/white_box_probe.py` but only
  validated on synthetic data (AUROC 0.983 synthetic, SNR=0.45).
- **Effect:** Real deployment requires a GPU (Llama-3-8B, 4-bit, T4 sufficient).
  No real-model AUROC is available yet.
- **Fix:** `notebooks/white_box_minijudge.ipynb` (Colab T4 free tier).
  MuSiQue 500-chain JSON is available locally for upload: `results/exp_musique_expand/musique_chains_200_499.json`.

### L6 — NIM backend: structural overconfidence

- **exp130 finding:** NIM llama-3.1-8b rates 92% of chains as risk=0.0 regardless of correctness.
  NIM llama-3.3-70b: 86% max-confidence. Both miss 87.5% of wrong chains.
- **Root cause:** These models were not fine-tuned for the P(True) quality-rating task.
  They conflate "I can generate an answer" with "the answer is correct."
- **NIM AUROC:** 0.60 (8b), 0.56 (70b) vs Haiku baseline 0.739 — 0.14–0.18 below.
- **Fix path:** Fine-tune a NIM model on (chain, rating) pairs, or use Anthropic as primary.
- **Current status:** NIM is **vendor failover only** — not a primary judge replacement.

---

## Out-of-Scope Domains

| Domain | Expected AUROC | Reason | Status |
|--------|---------------|--------|--------|
| Code web-search agents | 0.833 (P(True)) | exp_code_math_farl n=100; caveat: only 1 failure | ⚠ Marginal (n=1 failure) |
| Math / GSM8K reasoning | 0.500 | exp_code_math_farl n=100; RANDOM — confirmed OOS | ❌ Out of scope |
| SQL / structured output | Unknown | Tested in exp151 but not validated on failure detection | Preliminary |
| Customer service dialogs | Unknown | exp152 preliminary | Preliminary |
| NQ open-domain factoid | ≈ 0.52 | Validated near-random | Out of scope |
| Multi-agent pipelines (>2 hops) | Unknown | Trust propagation only, no QA labels | Not tested |

---

## What the System Does Well

1. **Multi-hop QA failure detection** — SC_OLD AUROC 0.82 on HotpotQA (within-domain),
   0.66 on TriviaQA (cross-domain, CI validated). 2Wiki SC_OLD 0.521 is NOT validated;
   use P(True) 0.758 for 2WikiMultiHop.
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

## New Results (v0.24.0, 2026-03-16)

### StepTransformerVerifier (exp_step_transformer)

2-layer Transformer over step embeddings (MiniLM frozen, ~150K trainable params).
Trained on HP 70% split, evaluated on HP hold-out + TriviaQA cross-domain.

| Model | HP hold-out AUROC | TV cross-domain AUROC |
|-------|-------------------|----------------------|
| SC_OLD behavioral | 0.654 | 0.710 CI[0.657, 0.761] |
| DeepLocalVerifier | 0.715 CI[0.579, 0.841] | 0.764 CI[0.725, 0.804] |
| LSTMRiskAccumulator | 0.723 CI[0.584, 0.849] | 0.744 CI[0.704, 0.784] |
| **StepTransformerVerifier** | **0.737** | **0.658** |

**Finding:** Transformer improves within-domain (0.737 > 0.723) but underperforms LSTM
cross-domain (0.658 < 0.744). The Transformer overfits HP-specific semantic patterns.
Fix: train with multi-domain data (HP + TriviaQA + ToolBench) — multi-domain SupCon
(QARA v2) is the architectural solution.

### Mondrian Conformal Prediction (exp_mondrian_cp)

Domain-specific conformal thresholds at FPR ≤ 10%:

| Domain | AUROC | Mondrian threshold | Mondrian precision | Uniform precision | Δ precision |
|--------|-------|-------------------|-------------------|------------------|-------------|
| TriviaQA | 0.659 | 0.528 | 0.310 | 0.300 | +0.010 |
| 2WikiMultiHop | 0.521 | 0.557 | 0.500 | 0.471 | +0.029 |
| MuSiQue | 0.583 | 0.599 | 0.800 | 0.804 | −0.004 |
| NQ | 0.524 | 0.439 | 0.667 | 0.500 | +0.167 |

**Finding:** Mondrian CP provides modest precision gains (+0.01 to +0.17) but at dramatic
recall cost on low-AUROC domains. 2Wiki recall drops 0.444→0.037; MuSiQue 0.616→0.055.
Interpretation: Mondrian correctly recognises it has low discrimination power on these
domains and raises the alert bar, flagging only the clearest failures. Use Mondrian CP
only if precision is more important than recall in your deployment.

## New Results (v0.25.0, 2026-03-16)

### StepTransformerVerifier — multi-domain training (exp_step_transformer_multidomain)

| Condition | HP | TV | 2Wiki | NQ | Mean cross |
|-----------|----|----|-------|-----|------------|
| C1: HP only | 0.676 | 0.575 | 0.582 | 0.647 | 0.601 |
| **C2: HP+TV (best)** | **0.563** | **0.705** | **0.640** | **0.634** | **0.660** |
| C3: HP+TV+2Wiki | 0.679 | 0.656 | 0.674 | 0.545 | 0.625 |
| C4: all 4 domains | 0.538 | 0.669 | 0.623 | 0.593 | 0.628 |
| LSTM baseline | 0.723 | 0.744 | — | — | — |

**Finding:** Joint training (C2 HP+TV) improves cross-domain mean AUROC from 0.601 → 0.660 (+0.059).
TV AUROC improves from 0.575 → 0.705 (+0.130). But LSTM still superior on TV (0.744 vs 0.705).
Adding more domains (C3, C4) degrades performance due to distribution dilution.
**Verdict: LSTM > Transformer for cross-domain. Transformer best within-domain only.**

### QARA v2 — 4-domain SupCon (exp160_qara_v2)

| Signal | TV AUROC | Status |
|--------|----------|--------|
| MiniLM baseline (kNN) | 0.581 | — |
| 2d-QARA (previous) | 0.536 | ❌ Inverted! (−0.045) |
| **QARA v2 (4-domain SupCon)** | **0.583** | +0.002 over baseline |

**Finding:** Semantic similarity fine-tuning provides near-zero improvement (+0.002).
Root cause: The task requires detecting reasoning *quality*, not semantic *similarity*.
A chain can be semantically close to a correct chain while being wrong. QARA v2 is not recommended.

### ZeroShotCalibrator (exp_zero_shot_calibrator)

Alert precision at FPR ≤ 10% on HP test (n=124 chains):

| Method | Precision | 95% CI | Recall | Human labels |
|--------|-----------|--------|--------|--------------|
| Uncalibrated (thresh=0.65) | 0.000 | [0.000, 0.000] | 0.000 | 0 |
| **ZeroShotCalibrator** | **0.708** | [0.574, 0.835] | 0.453 | **0** |
| QuickCalibrator | 0.850 | [0.667, 1.000] | 0.227 | 20 |

**Finding:** ZeroShotCalibrator (P(True) pseudo-labels, 0 human labels) achieves 0.708 precision.
Gap vs QuickCalibrator: −0.142 precision, but ZSC has higher recall (0.453 vs 0.227).
Pseudo-label quality was "low" (high_conf=0.36, noise≈0.64) — performance should improve
with a cleaner/more balanced calibration set. Useful when no human labels are available.
**Use case:** Zero-label deployment; accept precision tradeoff vs QuickCalibrator.**

### NLI Grounding (exp_nli_grounding) — NEGATIVE RESULT

| Domain | NLI AUROC | 95% CI | SC_OLD AUROC | NLI advantage |
|--------|-----------|--------|--------------|---------------|
| HotpotQA | 0.444 | [0.300, 0.616] | 0.619 | **−0.175 (NLI WORSE)** |
| TriviaQA | 0.576 | [0.357, 0.838] | 0.616 | −0.040 |

**Finding:** NLI entailment (answer→observation) does NOT improve over SC_OLD.
On HotpotQA, NLI is actually BELOW random (0.44). Signal is inverted/noisy.
Root cause: Web-snippet observations are too noisy for NLI to reliably measure entailment.
Wrong answers often cite plausible-looking observations that appear entailed.
**NLIGroundingVerifier should NOT be used in production.**

### MathVerifier (exp_math_verifier) — CONFIRMED

| Metric | Value |
|--------|-------|
| Math chain detection rate | 100% (40/40) |
| TPR (wrong caught) | 100% (20/20) |
| FPR (correct flagged) | 15% (3/20) |
| Integration test | 7/7 checks passed |

**Finding:** MathVerifier (SymPy + Python exec sandbox) correctly identifies math chains
and scores them deterministically. TPR=100% at FPR=15% is strong for the math domain
where SC_OLD gives AUROC=0.500 (random). MathVerifier is the recommended signal for
GSM8K-style arithmetic and SymPy-verifiable computation.
**Use:** `score_chain()` auto-routes via `is_math_chain()`; result in `behavioral_components["math_verified_risk"]`

## New Results (2026-03-16, post-FARL Phase 2)

### P(True) expanded to n=200 (exp_ptrue_expanded) — CONFIRMED

| Metric | Value |
|--------|-------|
| AUROC | 0.767 |
| 95% CI | [0.697, 0.829] |
| CI width | ±0.066 (target ≤ ±0.05; requires ~350 chains) |
| n | 200 (correct=88, wrong=112) |
| Cost | $0.00 (fully cached) |

**Status:** The P(True) AUROC = 0.767 is statistically validated at n=200. Lower CI bound 0.697
is solidly above 0.60. Previously cited as "preliminary (n=37)"; can now be claimed with confidence.
The claim "P(True) AUROC 0.767 [0.697, 0.829], n=200" replaces the earlier preliminary figure.

### Bell test — embedding-space evidence-grounding (exp_bell_test v2) — NEGATIVE RESULT

Tested hypothesis: `partial_sim(Q, A | E)` — Q-A cosine after projecting out evidence E —
should be higher in `confident_wrong` chains (answer from parametric priors) than in correct chains.

| Analysis | Feature | AUROC | 95% CI | Verdict |
| -------- | ------- | ----- | ------ | ------- |
| A: HP only (n=200) | f_bell | 0.448 | [0.361, 0.531] | Null |
| B: FARL_CW (n=79) vs HP correct (n=84) | f_bell | 0.455 | [0.369, 0.542] | Null |
| B: FARL_CW vs HP correct | f13 (sim_A_E) | **0.646** | [0.562, 0.731] | Signal |

**Key finding:** The projection-based Bell test feature has zero discriminative power even
with 79 curated `confident_wrong` chains. The embedding-space proxy cannot distinguish
"answer from priors" vs "answer from evidence" — both produce similar cosine similarity profiles.

**Unexpected signal:** f13 = sim(answer, evidence) has AUROC 0.646 when comparing FARL
`confident_wrong` (mean f13=0.661) vs HP correct (mean f13=0.557). FARL adversarial
`confident_wrong` chains have higher answer-evidence similarity, likely because the hunter
generates questions where evidence sounds plausible but is misinterpreted. This is a
distribution property of adversarial chains, not a production deployment signal.

**Conclusion:** f_bell is not worth adding to MiniJudge. The Bell test concept requires
white-box access (attention weights); output-level embeddings are insufficient.
f13 (already in MiniJudge v2) remains the correct evidence-grounding feature.

---

## Future Work

1. **White-box probe on Llama-3-8B** (Colab T4): expected AUROC ≥ 0.75.
   See `notebooks/white_box_minijudge.ipynb`.
2. **LoRA victim fine-tuning** (Colab A100): expected 15-30% lower failure rate
   on FARL taxonomy questions. See `notebooks/lora_victim_finetuning.ipynb`.
3. **Code/math agent FARL** (~$5, 2 weeks): discover novel failure modes in new action spaces.
   See `experiments/exp_code_math_farl.py` — runs 100 code + 100 math iterations.
4. **FARL significance:** CLOSED — 1,552 iterations run, 357 novel chains; CIs still overlap
   (delta −0.0056). Domain saturation confirmed. FARL taxonomy has qualitative value only.
5. **P(True) at n≥200:** CLOSED — AUROC 0.767 [0.697, 0.829], n=200 (exp_ptrue_expanded).
   CI width ±0.066; signal confirmed, lower CI > 0.60.
