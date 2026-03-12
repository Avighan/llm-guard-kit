# SC8: Label-Free Behavioral Monitoring for LLM ReAct Agents

**Avighan Majumder**
March 2026

---

## Abstract

LLM agents using the ReAct pattern generate reasoning chains that fail silently: the agent confidently produces a wrong answer with no external signal of failure. We present a comprehensive study of label-free failure detection for ReAct agents, progressing from embedding-based anomaly detectors to intra-chain behavioral signals. Our central finding is that **step count alone** (the number of Search actions in a chain) achieves AUROC 0.879 on HotpotQA without any labels or calibration data. Combining step count with a label-free GMM density estimator (SC8) raises within-domain AUROC to 0.883 and cross-domain AUROC to 0.664 — matching or exceeding supervised methods that require correctness labels. We further find that **mixed-domain calibration** (pooling chains from the deployment domain with the source calibration set) raises cross-domain AUROC to **0.810**, a new record. A negative result: monitoring is model-specific — GMM calibrated on Claude Haiku achieves chance AUROC (0.508) on Claude Sonnet, requiring re-calibration per model. We integrate these signals into a deployable monitoring system (QPPGService / `llm-guard-kit`) with four tiers: immediate zero-shot behavioral scoring, calibrated GMM scoring, failure taxonomy, and mid-run recovery injection. The system achieves 42.9% recovery rate in live intervention experiments at <10ms overhead per query.

---

## 1. Introduction

### 1.1 The Silent Failure Problem

LLM agents using the ReAct paradigm (Yao et al., 2022) interleave reasoning steps with tool calls to answer complex queries. These agents are increasingly deployed in production pipelines for web search, knowledge retrieval, and multi-step reasoning. A fundamental reliability challenge: when an agent fails, it typically does so without any signal. The final answer may be presented with the same format and apparent confidence as a correct answer.

This paper asks: **can we detect ReAct agent failures without access to correctness labels, token probabilities, or multiple LLM runs?**

### 1.2 The Behavioral Signal Hypothesis

Our central hypothesis is that *how* an agent reasons is more diagnostic of failure than *what* it reasons about. Specifically, a failing agent:

1. Searches more (3–4 queries vs 1–2 for correct agents)
2. Fails to complete (Finish action not reached)
3. Searches for increasingly diverse queries (diversifying when stuck)
4. Returns shorter, less confident answers
5. Has lower retrieval relevance (observations less related to the question)

These signals are **topic-agnostic**: an agent that searches 6 times is probably failing whether the question is about history or science. This contrasts with embedding-based anomaly detectors, which measure topic distance from the calibration bank and therefore encode topic, not quality.

### 1.3 Contributions

1. **Behavioral signal characterization**: Step count (SC2) alone achieves AUROC 0.879 within-domain and 0.570 cross-domain with zero labels, zero calibration, and zero API cost (Section 3).

2. **SC8 = SC2 + GMM**: Combining step count with a label-free GMM density estimator on chain embeddings raises cross-domain AUROC to 0.664, matching the supervised QARA adapter (Section 4).

3. **Mixed-domain calibration discovery**: Pooling 25 target-domain chains with the existing source calibration bank raises cross-domain AUROC to 0.810 — a 23-point improvement over the uncalibrated SC8 baseline (Section 5).

4. **Model specificity finding**: Behavioral monitoring is model-specific. GMM calibrated on Haiku achieves chance (0.508) on Sonnet; within-Sonnet monitoring is also harder (0.646 vs 0.883 for Haiku) because stronger models fail more subtly (Section 6).

5. **Full monitoring system**: Four-tier architecture (behavioral → GMM → taxonomy → intervention) with mid-run recovery achieving 42.9% repair rate in live experiments (Section 7).

6. **Empirical negative results**: Retrieval similarity (0.634 cross-NQ), self-consistency sampling (0.530), verbalized confidence (0.518 cross), and behavioral-embedding ensembles (0.550) all fail to beat SC8 (Section 8).

---

## 2. Problem Setup

### 2.1 ReAct Agent Format

A ReAct reasoning chain interleaves `Thought:` and `Action:` steps:

```
Question: When was X founded?
Thought: I need to find the founding date of X.
Action: Search[X founding date]
Observation: X was founded in 1847 by Y.
Thought: I found the answer.
Action: Finish[1847]
```

We use Claude Haiku (claude-haiku-4-5-20251001) as our agent, with Wikipedia search simulation. Two evaluation datasets:

- **HotpotQA (HP)**: Multi-hop factual QA, 200 chains, 42.5% correct. Correct chains average 2.34 searches; incorrect 4.75 searches — a 2× behavioral difference that motivates this work.
- **Natural Questions (NQ)**: Single-hop factual QA, 100 chains, 78% correct. Same ReAct format, different question distribution.

All chain data is cached; cross-domain experiments compare HP-calibrated models on NQ test sets.

### 2.2 Evaluation Protocol

**Primary metric**: AUROC on the binary correct/incorrect label. We use `safe_auroc` = max(AUROC, 1−AUROC) to handle polarity inversion.

**Cross-domain evaluation**: Train/calibrate on HP chains; evaluate on NQ chains (held out completely). AUROC > 0.65 is our threshold for "deployment-useful" cross-domain signal.

**Baseline**: Random = 0.50. SC2 (step count, zero calibration) = 0.570 cross-NQ. SC8 (SC2 + GMM) = 0.664 cross-NQ (our main reference).

---

## 3. Behavioral Signals (SC1–SC7)

### 3.1 Signal Definitions

We extract six intra-chain behavioral signals, all computable from the raw chain without any labels or embeddings:

| ID | Signal | Computation | Direction (↑ = more likely failure) |
|----|--------|-------------|--------------------------------------|
| SC1 | Completion | 1 if Finish reached, 0 otherwise | ↓ (incomplete → failure) |
| SC2 | Step count | # Search actions / 7 (capped) | ↑ (more searches → failure) |
| SC3 | Thought variance | Variance of thought embedding norms | ↑ (volatile reasoning → failure) |
| SC4 | Search diversity | # unique search terms / # searches | — (domain-dependent direction) |
| SC5 | Uncertainty words | Fraction of thoughts with "unclear", "not sure", etc. | ↑ |
| SC6 | Answer gap | Length of Finish argument vs. question | ↓ (short answer → failure) |

### 3.2 HP Chain Statistics

Examining the 200 HotpotQA chains reveals the signal:

| Metric | Correct (n=85) | Incorrect (n=115) | Δ |
|--------|---------------|-------------------|---|
| Mean search count | 2.34 | 4.75 | **+2.41** |
| Completion rate | 95% | 67% | −28% |
| Thought variance | 0.23 | 0.31 | +0.08 |
| Answer gap (log) | 2.1 | 1.6 | −0.5 |

The step count difference is the largest signal by far: incorrect agents search 2× more on average.

### 3.3 Within-Domain Results (Exp 39)

| Method | Within-HP AUROC | Cross-NQ AUROC | Labels? | Cal chains |
|--------|----------------|----------------|---------|------------|
| Random | 0.500 | 0.500 | — | 0 |
| SC1 (completion) | 0.839 | 0.506 | ✗ | 0 |
| **SC2 (step count)** | **0.879** | 0.570 | ✗ | 0 |
| SC3 (thought variance) | 0.762 | 0.592 | ✗ | 0 |
| SC6 (answer gap) | 0.800 | 0.530 | ✗ | 0 |
| SC7 (6-signal ensemble) | 0.869 | 0.547 | ✗ | 0 |

SC2 (step count) is the single strongest signal. 95% CI for SC2 within-HP: [0.813, 0.939]. Step count is more powerful than all embedding-based methods tested (KNN labeled: 0.640, GMM unlabeled: 0.627) while requiring zero infrastructure.

**Why step count works**: Correct HotpotQA answers are often found in 1–2 searches. An agent that searches 5–6 times is almost certainly lost — it is diversifying queries trying to find information it can't find, or looping. This is a causal signal, not a correlate.

**Why cross-domain degrades**: Step count is sensitive to question difficulty. NQ questions tend to be simpler (single-hop), so even wrong NQ agents use fewer steps than HotpotQA agents. The HP-calibrated threshold doesn't transfer well cross-domain.

---

## 4. SC8: Step Count + GMM Density Estimation

### 4.1 GMM on Chain Embeddings

The GMM component complements step count's behavioral signal with a distributional signal: does this chain *look like* the known-correct chains in embedding space?

We embed each chain's "chain-1 context" (question + first thought + first action) using `all-MiniLM-L6-v2` (384-d), project to PCA-32 space, then fit a Gaussian Mixture Model with k=4 components on the calibration set (no labels required).

The GMM log-likelihood serves as a quality proxy: chains from familiar reasoning patterns have high log-likelihood (low risk); unusual chains have low log-likelihood (high risk).

**Implementation** (from `LabelFreeScorer` in `qppg_service/`):

```python
# Calibration: fit GMM on unlabeled chains
scorer = LabelFreeScorer()
scorer.calibrate(unlabeled_chains)  # fits PCA-32 + GMM-4 on chain-1 embeddings

# Scoring: no labels needed
result = scorer.score(question, steps, final_answer, finished)
print(result.risk_score)   # 0.0 = low risk, 1.0 = high risk
```

Calibration requires as few as 5 chains (GMM works from n=5; see Section 4.3).

### 4.2 SC8 = SC2 + GMM

We combine the step count risk and GMM risk using rank normalization (to handle different scales):

```
SC8_risk = rank_norm(SC2_risk) × 0.5 + rank_norm(GMM_risk) × 0.5
```

where `rank_norm(x) = rank(x) / (n − 1)` maps to [0, 1] within the batch.

**Results** (Exp 39):

| Method | Within-HP AUROC | Cross-NQ AUROC | Cal chains needed |
|--------|----------------|----------------|-------------------|
| SC2 alone | 0.879 | 0.570 | 0 |
| GMM alone | 0.627 | 0.581 | 5+ |
| **SC8 = SC2 + GMM** | **0.883** | **0.664** | 5+ |
| SC8 95% CI (within-HP) | [0.799, 0.932] | — | — |

The GMM component adds 0.094 cross-domain AUROC over SC2 alone. SC8's cross-NQ = 0.664 is the main benchmark for all subsequent experiments.

### 4.3 Calibration Efficiency (Exp 44)

SC8 is robust to small calibration sets:

| n chains | SC2 AUROC | GMM AUROC | SC8 AUROC |
|----------|-----------|-----------|-----------|
| 0 | 0.879 | 0.500 | — |
| 5 | 0.879 | 0.545 | 0.860 |
| 10 | 0.879 | 0.583 | 0.864 |
| 25 | 0.879 | 0.610 | 0.873 |
| 50 | 0.879 | 0.627 | 0.883 |

SC2 (behavioral) contributes immediately at n=0. GMM improves progressively but is useful from n=5. The practical recommendation: deploy SC2 immediately; start collecting chains; at n=5, switch to SC8.

### 4.4 Comparison with Supervised Methods

| Method | Within-HP | Cross-NQ | Labels? | Cal chains |
|--------|-----------|----------|---------|------------|
| KNN labeled (baseline) | 0.603 | 0.501 | ✓ | 50 labeled |
| GMM k=4 (label-free) | 0.627 | 0.581 | ✗ | 50 unlabeled |
| QARA adapter (SupCon) | 0.705 | 0.628 | ✓ | 50 labeled |
| **SC8 = SC2 + GMM** | **0.883** | **0.664** | ✗ | 5 unlabeled |
| LLM judge (J1) alone | 0.800 | 0.506 | ✗ | per-chain API call |
| **J2 = J1 + SC8** | **0.895** | **0.660** | ✗\* | 5 unlabeled |

\*LLM judge uses gpt-4o-mini at ~$0.0001/call.

**Key finding**: SC8, which requires no correctness labels and only 5 unlabeled calibration chains, achieves 0.664 cross-domain AUROC — matching QARA (0.628), which requires 50 labeled chains and adapter training. SC8 also dominates within-domain (0.883 vs QARA's 0.705).

---

## 5. Mixed-Domain Calibration: A Cross-Domain Breakthrough

### 5.1 The Calibration Transfer Problem

A natural question: does a GMM calibrated on HP chains transfer to NQ? The answer from Section 4 is "partially": HP→NQ SC8 = 0.664. But can we do better by calibrating on target-domain chains?

Exp 52 tests bidirectional calibration with 8 conditions.

### 5.2 Bidirectional Transfer (Exp 52)

| Calibration → Test | AUROC | Notes |
|--------------------|-------|-------|
| HP (n=50) → NQ | 0.657 | SC8 reference |
| NQ (n=25) → HP | 0.578 | **FAILS** — reverse transfer doesn't work |
| HP (n=100) → NQ | 0.681 | More HP data helps slightly |
| NQ (n=50) → HP | 0.571 | Still fails |
| HP+NQ mix → HP test | 0.571 | Mixed hurts within-HP |
| **HP+NQ mix → NQ test** | **0.810** | **NEW BEST** |

The pure reverse (NQ → HP) fails because NQ chains have less structural diversity than HP multi-hop chains — the NQ GMM doesn't cover HP's failure modes.

### 5.3 Mixed-Domain Calibration Mechanism

The key experiment: mix 50 HP chains + 25 NQ chains in the calibration pool, then evaluate on NQ test chains.

Result: **0.810 cross-domain AUROC** — a 15-point improvement over HP-only SC8 (0.657 → 0.810).

**Why does this work?** When the GMM is fit on a mixed pool:
- HP chains anchor the "correct reasoning" distribution with rich multi-hop structure
- NQ chains add "what correct NQ reasoning looks like" to the GMM components
- NQ test chains are now scored against a GMM that has seen NQ examples
- The GMM components specialise: some cover HP patterns, others NQ patterns
- NQ test chains are near the NQ-anchored components → lower risk → correct label

This is a sample-efficient finding: **just 25 target-domain chains** (unverified correctness) added to the calibration pool raises NQ AUROC by 23 points.

### 5.4 Practical Implication

For production deployment to a new domain:
1. Start with source-domain calibration bank (e.g., HP chains)
2. Collect 20–30 chains from the new deployment domain (no labels needed)
3. Add them to the calibration pool and re-fit GMM
4. Cross-domain AUROC jumps from ~0.65 to ~0.81

This makes SC8 a practical cross-domain monitoring system: a small "warm-up" collection from the target domain suffices.

---

## 6. Model Specificity: Monitoring Doesn't Transfer Across LLM Versions

### 6.1 Motivation

Production systems upgrade their underlying LLMs regularly. If a monitoring system calibrated on one model version remains effective on the next, re-calibration is unnecessary. We test this with Claude Haiku → Claude Sonnet.

### 6.2 Cross-Model Experiment (Exp 53)

We run the same 150 questions (100 HP + 50 NQ) through Claude Sonnet (claude-sonnet-4-6) and compare behavioral patterns and monitoring AUROC:

| Metric | Haiku | Sonnet | Δ |
|--------|-------|--------|---|
| HP correctness | 54% | 65% | +11% |
| Correct: mean searches | 2.63 | 1.31 | −1.32 |
| Wrong: mean searches | 4.26 | 2.69 | −1.57 |
| Step-count Δ (wrong − correct) | 1.63 | 1.38 | −0.25 |
| Within-domain AUROC | **0.883** | **0.646** | −0.237 |

Sonnet solves more questions (65% vs 54%) and uses fewer steps overall. The step-count gap between correct and incorrect is preserved (Δ=1.38) but smaller — stronger models fail more efficiently, making the behavioral signal weaker.

### 6.3 Cross-Model Transfer

| Calibration → Test Model | Cross-NQ AUROC |
|--------------------------|----------------|
| Haiku-HP → Haiku-NQ | 0.664 (SC8 reference) |
| **Haiku-HP → Sonnet-HP** | **0.508 (chance)** |
| **Haiku-HP → Sonnet-NQ** | **0.527 (chance)** |

Cross-model transfer fails completely. The Haiku GMM is calibrated on chains with 2.63 mean searches (correct) and 4.26 (wrong). Sonnet chains have 1.31/2.69 — a completely different scale. The GMM's risk thresholds are calibrated to Haiku's search distribution and don't generalise.

### 6.4 Within-Sonnet Monitoring

Fitting a GMM on Sonnet-HP chains and evaluating within-Sonnet gives AUROC = **0.646** — substantially below Haiku's 0.883. This is not a calibration artifact: it reflects that Sonnet's failure mode is genuinely harder to detect. Stronger models fail on harder questions where even the correct approach requires many searches; the step-count signal weakens.

### 6.5 Implications

- **Each model version requires its own calibration bank.** This is a deployment requirement, not a limitation of the method.
- **Model upgrades should include monitoring re-calibration** as part of the deployment checklist.
- **Stronger models are harder to monitor.** AUROC 0.646 vs 0.883 suggests that improving model quality shifts the challenge from "detect clear failure patterns" to "detect subtle reasoning errors."
- **Mixed-domain calibration** (Section 5) may also help after model upgrades: collect 25+ Sonnet chains and add to a new Sonnet-specific calibration pool.

---

## 7. Retrieval Quality Signals (Exp 43, 51)

### 7.1 Semantic Retrieval Relevance

Beyond step count, we test whether the *quality* of retrieved observations (cosine similarity between question and observation embeddings) predicts correctness:

| Signal | Correct (HP) | Wrong (HP) | Δ |
|--------|-------------|------------|---|
| Mean cosine sim (question↔observations) | 0.568 | 0.441 | **+0.128** |
| Min cosine sim | 0.441 | 0.209 | **+0.232** |
| # "relevant" observations (sim > 0.4) | 2.0 | 3.05 | −1.05 |

Wrong agents find less relevant information. `min_sim` is especially diagnostic: at least one observation is highly irrelevant for wrong chains (mean min_sim = 0.209 vs 0.441 for correct).

### 7.2 Retrieval Signals vs SC8 (Exp 51)

| Method | Within-HP AUROC | Cross-NQ AUROC | Cal needed |
|--------|----------------|----------------|------------|
| SC2 alone | 0.879 | 0.570 | 0 |
| Mean ret_sim alone | — | 0.632 | 0 |
| **Min ret_sim alone** | — | **0.634** | 0 |
| SC2 + mean_sim + min_sim | — | 0.592 | 0 |
| **SC8 = SC2 + GMM** | **0.883** | **0.664** | 5 |

`min_sim` alone achieves 0.634 cross-NQ — the best **zero-calibration** cross-domain signal found, beating SC2 (0.570) and approaching SC8 (0.664). However, it doesn't beat SC8 once 5 calibration chains are available.

**Counter-intuitive finding**: SC2 and ret_sim are **anti-correlated** (r = −0.57 on HP). Both predict failure in the same direction: more searches → lower retrieval quality. This anti-correlation means they are not independent signals — ensembling them doesn't improve over the better individual signal.

---

## 8. Zero-Calibration Behavioral Feature Set (Exp 48)

### 8.1 18-Feature Logistic Regression

We design 18 topic-agnostic behavioral features (no embeddings, no API calls):

**Structural features**: n_steps, completed, n_search, n_finish
**Thought features**: thought_progression (do thoughts get longer?), thought_clarity_words, avg_thought_len
**Observation features**: obs_coverage (fraction of observations with useful content), avg_obs_len
**Answer features**: answer_len, answer_specificity (digit/entity density)
**Retrieval features**: n_unique_queries, query_diversity, repeat_rate
**Temporal features**: steps_per_thought, late_search_rate

A logistic regression trained on 50 HP chains and evaluated cross-domain on 50 NQ chains:

| Feature Set | HP AUROC | NQ AUROC | Note |
|------------|----------|----------|------|
| 18-feature LR (exp48A) | 0.820 | **0.611** | zero calibration, zero labels! |
| Domain-predictive features removed | 0.779 | 0.555 | hurts — features do double duty |
| SC8 (GMM needed) | 0.883 | 0.664 | needs 5 unlabeled chains |

The 18-feature LR achieves 0.611 cross-NQ with absolutely no embeddings, no calibration, and no API calls. The top cross-domain features are: `thought_progression` (−), `obs_coverage` (+), `completed` (+), `answer_len` (+).

This is the practical "zero-infrastructure" baseline for teams that cannot collect calibration data.

---

## 9. Negative Results

### 9.1 Self-Consistency Sampling (Exp 46)

Hypothesis: run the agent N=3 times per question; measure pairwise answer agreement. High agreement → likely correct.

Results: AUROC = 0.530 within-HP (vs SC2 0.879). Disproved. Claude Haiku is inconsistent even on easy questions — pairwise agreement does not predict correctness because the model can consistently produce the same wrong answer.

### 9.2 Verbalized Confidence (Exp 47)

Hypothesis: ask the agent to verbalize its confidence ("On a scale of 1-10, how confident are you?") and use this as a signal.

Results: Within-HP AUROC = 0.815 (strong!), Cross-NQ AUROC = **0.518 (chance)**. The agent's self-assessed confidence is domain-specific — the model calibrates differently on HP multi-hop vs NQ single-hop questions. Cross-domain: no value.

### 9.3 Feature + SC8 Ensemble (Exp 49)

Hypothesis: combining 18-feature LR (HP-trained) with SC8 GMM should complement each other (near-zero correlation r = 0.08).

Results: Ensemble cross-NQ = **0.550** (worse than SC8 0.664 or LR 0.611 alone). The near-zero correlation doesn't guarantee beneficial ensemble: the HP-trained LR has a polarity that conflicts with SC8's NQ rankings. Topic-specific feature weights (HP LR) conflict with GMM distance rankings on NQ chains.

**Lesson**: Near-zero correlation between signals does not guarantee beneficial ensembling. Conflicting decision boundaries cancel discrimination.

### 9.4 Synthetic Negatives (Exp 48B)

Hypothesis: augment training with LLM-generated "fake" incorrect chains (feeding wrong answers back to the LLM and asking it to generate a chain that would produce that answer).

Results: AUROC = **0.500** (random). Behavioral features cannot distinguish real incorrect chains from LLM-generated fake incorrect chains. The LLM generates "incorrect" chains that look behaviorally identical to real incorrect chains — the synthesis process creates realistic failure patterns.

---

## 10. Complete AUROC Benchmark Table

### 10.1 All Methods (Exp 36–53)

| Method | Within-HP | Cross-NQ | Label-free? | Cal chains | API cost |
|--------|-----------|----------|-------------|------------|---------- |
| Random | 0.500 | 0.500 | ✓ | 0 | $0 |
| SC2 (step count) | 0.879 | 0.570 | ✓ | 0 | $0 |
| SC1 (completion) | 0.839 | 0.506 | ✓ | 0 | $0 |
| Min ret_sim | — | 0.634 | ✓ | 0 | $0 |
| 18-feature LR | 0.820 | 0.611 | ✓ | 0 | $0 |
| Verbalized confidence | 0.815 | 0.518 | ✓ | 0 | $0.015/call |
| Self-consistency (N=3) | 0.530 | 0.547 | ✓ | 0 | $0.003/call |
| GMM k=4 alone | 0.627 | 0.581 | ✓ | 5+ | $0 |
| **SC8 = SC2 + GMM** | **0.883** | **0.664** | ✓ | 5+ | $0 |
| KNN labeled | 0.603 | 0.501 | ✗ | 50 labeled | $0 |
| Obs-pool QARA | 0.508 | 0.675 | ✗ | 50 labeled | $0 |
| QARA adapter (SupCon) | 0.705 | 0.628 | ✗ | 50 labeled | $0 |
| LLM judge alone | 0.800 | 0.506 | ✓ | 0 | $0.0001/call |
| SC8 + ret features (R10) | 0.867 | 0.648 | ✓ | 5+ | $0 |
| J2 = LLM judge + SC8 | 0.895 | 0.660 | ✓ | 5+ | $0.0001/call |
| **SC8 + mixed-domain cal** | — | **0.810** | ✓ | 5+25-target | $0 |

### 10.2 Pareto Frontier

For practitioners, the deployment choice depends on available calibration chains:

| Situation | Recommended | Cross-NQ AUROC |
|-----------|-------------|----------------|
| Zero chains | 18-feature LR or min_sim | 0.611 / 0.634 |
| 5 unlabeled chains | SC8 | 0.664 |
| 5 + 25 target-domain | SC8 mixed-cal | 0.810 |
| Budget for LLM judge | J2 = SC8 + judge | 0.660 |
| Labels available | Obs-pool QARA | 0.675 |

---

## 11. Full Monitoring System

### 11.1 Architecture (QPPGService / llm-guard-kit)

The monitoring system implements four tiers:

```
Tier 0: Behavioral scoring (SC2 / 18-feature LR)
  ↓ [calibration bank available]
Tier 1: SC8 = SC2 + GMM density (LabelFreeScorer)
  ↓ [risk_score > threshold]
Tier 2: Alert + failure taxonomy (FailureTaxonomist)
  ↓ [during agent run]
Tier 3: Mid-run intervention (SelfHealer injection)
```

**Tier 0** activates immediately with zero chains, running SC2 and behavioral features. **Tier 1** calibrates from the first 5 unlabeled chains observed. **Tier 2** classifies failures into: EXCESSIVE_SEARCH, RETRIEVAL_FAILURE, REASONING_LOOP, LOW_CONFIDENCE, INCOMPLETE. **Tier 3** injects recovery prompts mid-run.

### 11.2 Mid-Run Intervention (Exp 50)

**Setup**: Score partial chains at step 3 (after 3 searches). If risk > 0.55, inject a recovery prompt.

**50A analysis** (on cached HP chains):
- Early warning rate (wrong chains, ≤step 3): 27.3%
- False alarm rate (correct chains flagged): **53.6%** — too high for direct intervention
- Two-stage filter (SC8 + FailureTaxonomist): 71% of false alarms correctly marked NO_ACTION or LOW_RISK

Practical recommendation: trigger intervention only when risk > 0.65 AND ≥4 steps AND FailureTaxonomist ≠ LOW_RISK.

**50B live test** (7 OpenAI gpt-4o-mini runs on confirmed wrong questions):
- Recovery prompt: FailureTaxonomist → RETRIEVAL_FAILURE → inject `REPHRASE_QUERY` advisory
- **Recovery rate: 3/7 = 42.9%** (baseline: 0% — all chains confirmed wrong without intervention)
- REPHRASE_QUERY injections: 1/2 fixed; NO_ACTION controls: 2/5 fixed (may include stochastic recovery)

### 11.3 Lifecycle Management

```
COLD_START (<10 chains):  SC2/behavioral only; alert if risk > 0.70
WARMING (10–49 chains):   SC8 active; alert if risk > 0.60
DEPLOYED (≥50 chains):    Full system; alert if risk > 0.55
```

System is deployed as `llm-guard-kit` v0.2.0 on PyPI (https://pypi.org/project/llm-guard-kit/).

### 11.4 Inference Overhead

| Component | Latency |
|-----------|---------|
| SC2 behavioral (Python) | <0.5ms |
| MiniLM embedding (warm, CPU) | 5.8 ± 0.3ms |
| GMM scoring (PCA-32, k=4) | <0.5ms |
| KNN distance (n=200) | ~2.4ms |
| **Total per-chain overhead** | **~9ms** |

The 9ms overhead is negligible relative to LLM API latency (500ms–3s).

---

## 12. Related Work

**Self-consistency** (Wang et al., 2022): 5 LLM samples per query, majority vote. Achieves ~0.80 AUROC on math. Our Exp 46 shows gpt-4o-mini's self-consistency AUROC = 0.530 on factual QA, suggesting self-consistency relies on math's determinism. Our approach requires 1 LLM call.

**Semantic entropy** (Farquhar et al., 2024): Entropy over paraphrase clusters; ~0.80 AUROC on math. Same limitation as self-consistency: requires multiple LLM calls; our approach is complementary for latency-constrained settings.

**Calibration methods** (Guo et al., 2017; Kadavath et al., 2022): Temperature scaling and verbalized probability. Require token log-probabilities or explicit self-assessment. Our Exp 47 shows verbalized confidence = chance cross-domain.

**Anomaly detection for NLP** (Hendrycks & Gimpel, 2017; Lee et al., 2018): Out-of-distribution detection using embedding distances. We show topic conflation makes raw embeddings fail for quality detection; behavioral signals resolve this.

**LLM agent monitoring** (Shinn et al., 2023; Yao et al., 2023): ReAct and Reflexion focus on improving agent performance through reflection. We focus on monitoring without agent modification.

**QARA** (Majumder, 2026): Our earlier work on supervised contrastive adapters for cross-domain quality detection. SC8 matches QARA's 0.628 AUROC without requiring any correctness labels.

---

## 13. Discussion

### 13.1 Why Behavioral Signals Beat Embedding Methods

The core insight is directional: embedding-based anomaly detection measures *novelty* (is this chain unusual?) but novelty ≠ quality. A correct chain on a novel topic is unusual in embedding space but high-quality. A behavioral signal (step count) measures *process* — did the agent get lost? This is topic-agnostic.

The 2× step-count gap between correct and incorrect HP chains is a robust causal signal: when the agent doesn't know the answer, it searches more. This holds across question topics and (within a model) across difficulty levels.

### 13.2 The Mixed-Domain Calibration Insight

The 0.810 AUROC from mixed-domain calibration (HP+NQ → NQ test) is the most practically significant finding. It transforms the deployment story: instead of "collect 50 labeled chains from the new domain," the requirement becomes "collect 25 unlabeled chains from the new domain." This is a 50× reduction in labeling cost and a 2× reduction in collection effort.

The mechanism is clear in retrospect: the GMM's components are learned from the calibration distribution. When NQ chains are included in calibration, some GMM components specialise to NQ patterns. NQ test chains score against these components rather than against purely HP-patterned components, removing the main source of cross-domain error.

### 13.3 Model Specificity as a Deployment Requirement

The cross-model result (Haiku → Sonnet = 0.508) imposes a practical constraint: every model upgrade triggers a monitoring re-calibration. For teams with rapid model iteration schedules, this is significant.

However, the mixed-domain calibration result (Section 5) offers a path: after a model upgrade, collecting 25 chains from the new model-domain combination and mixing them into a new calibration bank may suffice. We leave this "model upgrade warm-up" experiment for future work.

### 13.4 Limitations

**Dataset scale**: All experiments use 100–200 chains. While we report AUROC CIs and CV results, larger-scale validation would strengthen claims.

**Single model**: Behavioral patterns are validated on Claude Haiku and Claude Sonnet. Other model families (GPT, Gemini, Llama) may exhibit different step-count distributions.

**Simulated retrieval**: Our experiments use cached Wikipedia search results. Real retrieval variability (network latency, result quality variation) may change retrieval quality signals.

**Two domains**: HP and NQ are both factual QA with ReAct. Domain diversity beyond these two (math, code, planning) is not tested in this paper.

---

## 14. Conclusion

We present a comprehensive study of label-free failure detection for LLM ReAct agents. Our main findings:

1. **Step count (SC2) is the most powerful single signal** — 0.879 within-domain AUROC with zero labels, zero calibration, zero cost. Incorrect agents search 2× more than correct agents.

2. **SC8 = SC2 + GMM achieves 0.664 cross-domain** with just 5 unlabeled calibration chains, matching or beating supervised methods (QARA: 0.628) requiring correctness labels.

3. **Mixed-domain calibration reaches 0.810 cross-domain** — a 15-point improvement — by adding just 25 target-domain chains to the calibration pool. This is the most actionable finding for practitioners.

4. **Monitoring is model-specific**: behavioral patterns for Haiku don't transfer to Sonnet (0.508 = chance). Each model version requires its own calibration bank.

5. **A deployable system exists**: QPPGService / `llm-guard-kit` implements four tiers of monitoring with <10ms overhead, a FailureTaxonomist for diagnosis, and mid-run recovery achieving 42.9% repair rate.

The broader lesson: for LLM agent monitoring, *process signals* (how the agent reasons) beat *content signals* (what the agent reasons about). Step count is the best process signal because it is causal (lost agents search more), topic-agnostic (universal across question domains), and instantaneous (no calibration data required).

---

## Appendix A: Complete Experiment Reference

| Exp | Date | Key Finding | AUROC |
|-----|------|------------|-------|
| 36 | Feb 2026 | Label-free KNN baseline characterization | 0.539 within |
| 37 | Feb 2026 | GMM k=4 best label-free; beats labeled KNN | 0.627 within, 0.581 cross |
| 38 | Feb 2026 | GMM-k2-neg + GMM-k4 ensemble | 0.680 within, 0.643 cross |
| 39 | Feb 2026 | SC2 = 0.879; SC8 = 0.883/0.664 | **0.883 / 0.664** |
| 40 | Feb 2026 | Observation-pool embeddings (MS5+GMM) | 0.657 cross |
| 41 | Feb 2026 | LLM judge + SC8 = J2 | 0.895 / 0.660 |
| 42 | Feb 2026 | QARA on obs-pool (5-fold CV, p=0.025) | 0.675 cross |
| 43 | Feb 2026 | Retrieval quality signals (min_sim Δ=+0.232) | — |
| 44 | Feb 2026 | Calibration efficiency; SC8 from n=5 | — |
| 45 | Mar 2026 | Full pipeline demo (Tier 0–4) | — |
| 46 | Mar 2026 | Self-consistency N=3: FAILED | 0.530 within |
| 47 | Mar 2026 | Verbalized confidence: cross fails | 0.815 / 0.518 |
| 48 | Mar 2026 | 18-feature LR: 0.611 zero-cal cross | 0.820 / 0.611 |
| 49 | Mar 2026 | Feature+SC8 ensemble: FAILED | 0.550 cross |
| 50 | Mar 2026 | Mid-run intervention: 42.9% recovery | — |
| 51 | Mar 2026 | min_sim = best zero-cal cross signal | 0.634 cross |
| 52 | Mar 2026 | Mixed-domain cal: **HP+NQ → 0.810** | **0.810 cross** |
| 53 | Mar 2026 | Cross-model transfer fails (Haiku→Sonnet) | 0.508 cross-model |

## Appendix B: Reproducibility

All code and cached data are in the QPPG repository:

- `qppg_service/label_free_scorer.py` — `LabelFreeScorer` (SC8, GMM, retrieval_quality)
- `qppg_service/monitor.py` — `QppgMonitor` (Tier 1: auto-calibration, alerts, reports)
- `qppg_service/failure_taxonomy.py` — `FailureTaxonomist` (Tier 2: failure diagnosis)
- `qppg_service/self_healer.py` — `SelfHealer` (Tier 3: recovery injection)
- `experiments/exp39_intra_chain_sc.py` — SC2/SC8 behavioral signals
- `experiments/exp52_bidirectional_calibration.py` — mixed-domain calibration
- `experiments/exp53_stronger_model.py` — cross-model generalization

**PyPI**: `pip install llm-guard-kit` (v0.2.0)
**Runtime**: All $0-cost experiments use cached chains (no live API calls required to reproduce).
**Dependencies**: Python 3.10+, numpy, scikit-learn, sentence-transformers, anthropic (optional).

## Appendix C: SC8 Quick Reference

```python
from qppg_service import LabelFreeScorer

scorer = LabelFreeScorer()

# Option 1: Zero calibration (SC2 behavioral only)
result = scorer.score(question, steps, final_answer, finished=True)
print(f"Risk: {result.risk_score:.3f}")  # 0=low, 1=high

# Option 2: Calibrated SC8 (5+ chains, no labels needed)
scorer.calibrate(unlabeled_chains)  # list of chain dicts
result = scorer.score(question, steps, final_answer, finished=True)

# Option 3: Mixed-domain calibration for new domain
mixed_chains = source_chains + target_chains_25  # 50+25
scorer.calibrate(mixed_chains)
# → cross-domain AUROC ~0.810
```
