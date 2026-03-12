# QARA: Cross-Domain Quality Detection for LLM ReAct Agents via Supervised Contrastive Adapter Learning

**Avighan Majumder**
March 2026

---

## Abstract

Large Language Model (LLM) agents using the ReAct (Reason + Act) pattern generate reasoning chains that vary widely in quality, yet no lightweight method exists to detect failures before they propagate through agentic pipelines. We show that KNN anomaly scoring on frozen sentence-transformer embeddings achieves near-perfect within-domain failure detection (AUROC 0.96–1.00) but collapses to near-chance cross-domain (AUROC ~0.50–0.55) because raw embeddings encode topic, not reasoning quality. To solve this, we introduce **QARA** (Quality-Aware Reasoning Adapter): a lightweight 2-layer MLP (~98K parameters) trained with Supervised Contrastive (SupCon) loss using cross-domain positive pairs, forcing the adapter to cluster reasoning quality regardless of topic. On HotpotQA (multi-hop factual QA) as the evaluation domain, QARA trained on HotpotQA + Natural Questions achieves cross-domain AUROC 0.628 ± 0.077 vs. baseline 0.480 ± 0.134 (sign test p = 0.031, all 5 folds positive), with 1.7× variance reduction. We further show that domain format compatibility matters more than domain count: HotpotQA + NQ (both ReAct factual QA, +0.15 AUROC) outperforms adding TriviaQA + NQ together (TV+NQ format conflict). We integrate these findings into **QPPGService**, a deployable confidence-scoring service with lifecycle management (COLD\_START → WARMING → DEPLOYED). The full system adds less than 20ms of inference overhead and requires zero LLM fine-tuning.

---

## 1. Introduction

LLM agents using ReAct (Yao et al., 2022) generate step-by-step reasoning chains that interleave thoughts and actions. These agents are increasingly deployed in production systems for tasks ranging from factual question answering to code generation to multi-step tool use. A fundamental reliability challenge arises: the agent may produce a confident-sounding but incorrect chain, and downstream components have no lightweight signal to detect this before acting on the result.

### 1.1 The Failure Detection Problem

Existing approaches to LLM confidence estimation fall into two categories:

1. **Token-probability methods** (Kadavath et al., 2022): Access to token log-probabilities provides a calibration signal, but this is unavailable through most commercial APIs and unreliable on RLHF-tuned models.
2. **Self-consistency methods** (Wang et al., 2022): Run the model multiple times and measure agreement. Effective but requires 3–5× the API cost.
3. **Semantic entropy** (Farquhar et al., 2024): Entropy over paraphrase clusters achieves ~0.80 AUROC on math tasks but again requires multiple LLM calls.

We propose a complementary approach: **anomaly detection on chain embeddings**. Given a bank of known-correct reasoning chains, embed new chains with a sentence-transformer and measure their KNN distance to the correct bank. High distance → the chain is in unfamiliar territory → likely a failure.

This approach has key advantages: it adds only one forward pass through a 22M-parameter embedding model (<15ms), requires no access to token probabilities, and does not depend on running the LLM multiple times.

### 1.2 The Cross-Domain Problem

Within a single domain, KNN anomaly scoring works remarkably well: AUROC 0.993 on HumanEval (code), 0.992 on TriviaQA (factual), 0.966 on GSM8K (math). However, a critical limitation emerges in production: the calibration bank is often built from one task type (e.g., HotpotQA multi-hop questions), while the agent subsequently encounters queries from different distributions (e.g., Natural Questions, TriviaQA).

We show that cross-domain performance collapses to near-chance (AUROC 0.47–0.55) because raw sentence-transformer embeddings encode **topic similarity** rather than **reasoning quality**. A correct HotpotQA chain about Roman history and a correct NQ chain about Roman history are close in embedding space — not because they share quality properties, but because they share vocabulary.

### 1.3 Contributions

This paper makes the following contributions:

1. **Empirical characterization** of the within-domain / cross-domain AUROC gap for LLM ReAct agents (Section 3).
2. **QARA**: a lightweight supervised contrastive adapter that achieves statistically significant cross-domain improvement with ~98K trainable parameters and ~1s training time (Section 4).
3. **Domain compatibility analysis**: showing that format-compatible training domains (ReAct factual QA) provide better SupCon signal than format-diverse domains; adding incompatible domains hurts performance (Section 5).
4. **QPPGService**: an end-to-end deployable service integrating QARA with lifecycle management, HTTP API, and production-ready confidence scoring (Section 6).

---

## 2. Background

### 2.1 ReAct Chain Format

A ReAct reasoning chain interleaves natural language thoughts with structured actions. For factual QA:

```
Thought: I need to find when X was founded.
Action: Search[X founding date]
Observation: X was founded in 1847.
Thought: Now I can answer.
Action: Finish[1847]
```

We extract the **chain-1 context** — the first thought and action — as our quality signal:

```
TASK: {question}
STEP 1: Thought: {thought} | Action: {action_type}[{action_arg}]
CURRENT: {action_type}[{action_arg}]
```

This design choice is deliberate: chain-1 captures the agent's initial decomposition of the problem, which is predictive of whether the full chain will succeed. A correct initial search query (e.g., `Search[founding date of X]`) strongly predicts success; an incorrect one (e.g., `Search[X history]` when the question requires a specific date) predicts failure.

### 2.2 Sentence-Transformer Embeddings

We use `all-MiniLM-L6-v2` (Wang et al., 2020) throughout: a 22M-parameter model producing 384-dimensional unit-normalised vectors. This model is chosen for:
- Fast inference (~5.8ms warm on CPU)
- Local deployment (no API dependency)
- Strong performance on semantic similarity tasks

### 2.3 KNN Anomaly Score

Given a calibration bank **C** (embeddings of known-correct chains, shape n × 384) and a test embedding **q**, the anomaly score is:

```
score(q) = mean_distance(q, knn(q, C, k=5))
```

where `knn(q, C, k)` returns the k nearest neighbours of q in C. Higher score = farther from familiar territory = more likely a failure.

AUROC is computed over the test set using `score(q)` to rank chains, with correct chains as positives.

---

## 3. The Within-Domain / Cross-Domain Gap

### 3.1 Experimental Setup

We collect reasoning chains across four domains:

| Domain | Dataset | Agent | N | Correct Rate | Chain Format |
|--------|---------|-------|---|-------------|-------------|
| HotpotQA (HP) | HotpotQA validation | Claude Haiku ReAct | 200 | 42.5% | Multi-hop ReAct |
| TriviaQA (TV) | TriviaQA validation | Claude Haiku ReAct | 100 | 100%\* | Single-hop ReAct |
| Natural Questions (NQ) | nq\_open validation | Claude Haiku ReAct | 100 | 78% | Single-hop ReAct |
| GSM8K | GSM8K test | Claude Haiku ReAct | 100 | 91% | Math ReAct |

\*TriviaQA: we sample only correct chains for training signal.

For HotpotQA, we use a strict 50/50 train/test split: 100 chains in the calibration bank, 100 chains in the test set (35 correct, 65 incorrect). This avoids data leakage from the initial results in Experiment 31 that achieved inflated AUROC from using all 200 HP chains in both training and testing.

### 3.2 Within-Domain Results (Experiments 18–23)

Within a single domain, KNN anomaly scoring achieves near-perfect AUROC:

| Domain | Within-Domain AUROC |
|--------|-------------------|
| HotpotQA | 0.60–1.00\* |
| TriviaQA | 0.992 |
| HumanEval (code) | 0.993 |
| GSM8K (math) | 0.966 |

\*HotpotQA within-domain varies 0.60–1.00 depending on question uniqueness in the calibration bank. When the calibration and test sets contain related questions (same difficulty distribution), AUROC approaches 1.0. When questions are fully disjoint topics, AUROC drops to ~0.60, reflecting the topic-quality conflation problem within a single domain.

### 3.3 Cross-Domain Baseline (Experiment 32)

When the calibration bank (TV correct chains) is used to score HotpotQA test chains:

| Calibration | Test | AUROC |
|-------------|------|-------|
| HP train correct | HP test | 0.640 (within-domain) |
| TV correct | HP test | 0.518 (cross-domain) |

The drop from 0.640 to 0.518 is not a failure of the approach — it is a diagnostic: TV chains are about different topics than HP chains, so TV KNN distances measure topic distance, not quality. A correct HP chain about quantum physics is far from the TV bank (which covers different factual topics), giving it a high "failure" score even though it's correct.

The same inversion is observed for NQ → HP calibration (AUROC = 0.473), where NQ chains actually have *inverted* signal: they're close to HP correct chains in topic space, making the baseline score misleading.

---

## 4. QARA: Quality-Aware Reasoning Adapter

### 4.1 Architecture

QARA is a 2-layer MLP adapter applied on top of frozen MiniLM-L6-v2 embeddings:

```
Input: 384-d MiniLM embedding (frozen)
Layer 1: Linear(384 → 256) + ReLU + LayerNorm
Layer 2: Linear(256 → 64) + L2-normalisation
Output: 64-d quality-aware embedding
```

Total trainable parameters: **~98,304** (~0.45% of MiniLM's 22M).
Training time: **~1 second** on CPU (200 epochs, 200 chains, Adam lr=3e-4).

The L2 normalisation at the output ensures that all adapted embeddings lie on a unit hypersphere, enabling cosine-distance KNN scoring.

### 4.2 Supervised Contrastive Training

We train with Supervised Contrastive (SupCon) loss (Khosla et al., 2020), modified for cross-domain quality learning:

```
L_SupCon = -1/|P(i)| × Σ_{p∈P(i)} log [
    exp(z_i · z_p / τ) / Σ_{a∈A(i)} exp(z_i · z_a / τ)
]
```

where:
- `z_i` is the L2-normalised adapter output for chain i
- `P(i)` = all correct chains in the batch, regardless of domain ("quality-positive pairs")
- `A(i)` = all other chains in the batch (positive + negative)
- `τ = 0.1` (temperature)

**The critical design choice**: positive pairs cross domain boundaries. A correct HP chain and a correct TV chain are treated as positives — they share the quality property "correct reasoning" even though they are about different topics. This forces the adapter to cluster by quality, not by topic.

After training, KNN in the adapted 64-d space separates correct from incorrect chains across domains.

### 4.3 Why SupCon with Cross-Domain Positives Works

The intuition: MiniLM embeds chains based on their surface content. The adapter learns a residual rotation that prioritises quality signals over topic signals. The cross-domain positive pairs provide the supervision signal:

> "This HotpotQA chain about astronomy and this TriviaQA chain about history should be neighbours — they're both correct. This HotpotQA chain about astronomy that failed should be far from both."

After enough such pairs, the adapter builds a quality manifold where correctness clusters together independent of topic.

**Ablation evidence**: Training with only HP chains (single domain) inverts the signal — AUROC drops from 0.518 (baseline) to 0.483. The adapter over-clusters HP correct chains, pushing TV and NQ chains toward HP incorrect regions. Cross-domain positive pairs are not just helpful — they are essential.

---

## 5. Multi-Domain Training Analysis

### 5.1 Experiment 32: 2-Domain QARA (HP + TV)

The initial rigorous evaluation using a proper 50/50 HP train/test split and TV correct chains as extra domain signal:

| Condition | Within-HP AUROC | TV→HP AUROC |
|-----------|----------------|------------|
| Baseline (raw MiniLM) | 0.641 | 0.518 |
| QARA ablation (HP only) | 0.685 | 0.483 |
| QARA full (HP + TV) | 0.611 | **0.617** |

Cross-domain improvement: +0.099 over baseline. The variance reduction is notable: σ(baseline) = 0.140, σ(QARA) = 0.062 across 5 CV folds.

### 5.2 Experiment 33: 4-Domain QARA with GSM8K/HumanEval

Adding math (GSM8K) and code (HumanEval) as training domains:

| Training Domains | HP test AUROC |
|-----------------|--------------|
| Baseline | 0.585 |
| HP + TV (2-domain) | 0.596 |
| HP + TV + GSM8K (3-domain) | 0.552 |
| HP + TV + GSM8K + HumanEval (4-domain) | 0.553 |

**Adding more domains hurts.** The 4-domain model underperforms the 2-domain model (0.553 < 0.596). Root cause: GSM8K and HumanEval use a different chain format (math notation, code blocks) that conflicts with the ReAct factual QA format. The SupCon loss treats GSM8K correct chains as positives with HP correct chains, but their chains don't share quality structure — their surface form is too different for the adapter to learn a unified quality signal.

**Key insight**: Domain compatibility (shared format) matters more than domain count.

### 5.3 Experiment 35: 3-Domain QARA with Natural Questions

Natural Questions (NQ) uses the same ReAct factual format as HotpotQA. We test three configurations:

| Training Condition | Within-HP | TV→HP | NQ→HP |
|-------------------|-----------|-------|-------|
| Baseline MiniLM | 0.603 | 0.523 | 0.473 |
| 2-domain (HP+TV) | 0.611 | 0.599 | 0.567 |
| 3-domain (HP+TV+NQ) | 0.597 | 0.607 | 0.610 |
| **HP+NQ ablation (no TV)** | **0.705** | **0.624** | **0.630** |

**HP+NQ (no TV) is the best model** — better than adding TV to the mix. This is the domain compatibility finding in its strongest form:
- NQ + HP: same ReAct format, similar question style → clean quality signal
- TV + NQ together: conflicting format nuances → SupCon pairs are noisy → adapter learns a compromise that underperforms

The HP+NQ model achieves within-HP AUROC = 0.705 (+0.12 over baseline), showing that a format-compatible auxiliary domain improves even within-domain performance by building a richer quality manifold.

### 5.4 Statistical Significance (5-fold Cross-Validation)

We run 5-fold CV on the HP+NQ model to confirm the NQ→HP improvement is stable:

| Fold | Baseline NQ→HP | QARA NQ→HP | Δ |
|------|---------------|-----------|---|
| 1 | 0.468 | 0.514 | +0.046 |
| 2 | 0.589 | 0.609 | +0.021 |
| 3 | 0.260 | 0.616 | +0.356 |
| 4 | 0.497 | 0.708 | +0.211 |
| 5 | 0.583 | 0.690 | +0.107 |
| **Mean** | **0.480 ± 0.134** | **0.628 ± 0.077** | **+0.148** |

All 5 folds show positive delta:
- **Sign test**: p = (0.5)^5 = **0.031 < 0.05** (statistically significant)
- **Paired t-test**: t = 2.41, df = 4 (one-tailed p ≈ 0.036)
- **Variance reduction**: 1.7× (σ: 0.134 → 0.077)

The QARA adapter not only improves the mean cross-domain AUROC but stabilises it — the deployment system becomes more predictable.

### 5.5 Summary: Domain Selection Principles

| Principle | Evidence |
|-----------|---------|
| Format compatibility > domain count | 2-domain HP+NQ > 3-domain HP+TV+NQ |
| Format diversity hurts | 4-domain (with GSM8K/HumanEval) < 2-domain (0.553 vs 0.596) |
| Cross-domain positives are essential | HP-only QARA inverts signal (0.518→0.483) |
| Within-domain also improves | HP+NQ within-HP 0.705 vs baseline 0.603 (+0.10) |

---

## 6. QPPGService: Deployment Integration

### 6.1 Lifecycle Management

Production deployment requires handling the cold-start problem: there are no correct chains to calibrate on initially. QPPGService implements a three-stage lifecycle:

```
COLD_START  (<10 chains)  →  score() returns None; collect chains
WARMING     (10–49)       →  returns raw KNN scores (low confidence)
DEPLOYED    (≥50 chains)  →  full confidence scoring with QARA adapter
```

The 50-chain threshold is motivated by Experiment 30 (online calibration): below 50 chains, within-domain AUROC is unstable (<0.70); above 50, diversity-maximised selection gives AUROC > 0.90.

### 6.2 Confidence Scoring

At DEPLOYED state, `score(chain_1_context)` returns:

1. **QARA-adapted KNN distance**: embed chain-1 context → apply adapter → KNN distance in 64-d quality space
2. **Confidence value** in [0, 1]: fraction of calibration-admission distances exceeding the test distance
3. **Interpretation**: confidence = 0.8 means "80% of known-correct chains have higher KNN distance; this chain is in familiar territory"

### 6.3 REST API

The QPPGService is wrapped in a FastAPI server with three endpoints:

```
POST /score           — score a single chain-1 context
POST /calibrate       — add a verified correct chain
POST /bulk-calibrate  — seed from a batch of historical logs
GET  /status          — lifecycle state, chain count, AUROC estimate
```

### 6.4 Deployment Performance

| Component | Latency |
|-----------|---------|
| MiniLM embedding (warm) | 5.8 ± 0.3ms |
| QARA adapter forward pass | <0.5ms |
| KNN scoring (n=200) | ~2.4ms |
| **Total overhead** | **~9ms** |

The full system adds less than 10ms to the LLM call latency (which dominates at 500ms–3s).

---

## 7. Discussion

### 7.1 What QARA Learns

The adapter learns a quality manifold in 64-d space. Examining the geometry of this space reveals:
- Correct chains from HP and NQ cluster together despite different topics
- Incorrect chains scatter: some cluster near correct chains on the quality manifold (borderline failures), others are far from all correct chains (confident failures)
- The adapter's primary transformation is a rotation that de-emphasises topic dimensions and amplifies dimensions correlated with reasoning coherence (e.g., appropriate search specificity, correct action type selection)

### 7.2 Why Chain-1 Is Sufficient

We use only the first thought and action from the reasoning chain. This is sufficient because:
1. The first action type (Search vs Finish vs Calculate) reveals whether the agent understands the task type
2. The first search query specificity predicts whether the agent can decompose the question
3. Using only chain-1 enables scoring **before** the full chain runs — enabling pre-emptive intervention

In production, this means: score the chain-1 context after the first LLM call; if confidence is low, trigger a re-prompt or escalation before the agent proceeds through multiple expensive tool calls.

### 7.3 Limitations

**Small calibration dataset**: Our experiments use 50–100 training chains per domain. Performance would improve with more data. The QARA adapter is designed to be retrained periodically as the production calibration bank grows.

**Run-to-run variance**: With small datasets, training variance is ~±0.07 AUROC. The 5-fold CV confirms a consistent positive trend, but point estimates from single runs should be interpreted with this variance in mind.

**Domain format sensitivity**: The TV+NQ conflict suggests that careful domain curation is needed for multi-domain training. A practitioner should verify that training domains share the same chain format before combining them.

**API coverage**: Currently validated on Claude Haiku (claude-haiku-4-5-20251001). The embedding-based approach is model-agnostic; the chain format and correctness labels are model-specific.

### 7.4 Relation to Prior Work

**Semantic entropy** (Farquhar et al., 2024): Achieves ~0.80 AUROC on math with 5 LLM calls per query. QARA achieves 0.63–0.70 AUROC on cross-domain factual QA with 1 embedding forward pass. The approaches are complementary: QARA is better for latency-constrained production; semantic entropy is better when multiple API calls are affordable.

**Calibration methods** (Guo et al., 2017): Temperature scaling post-calibration improves probability calibration but requires access to token probabilities. QARA requires only chain embeddings.

**Domain adaptation** (Ben-David et al., 2010): QARA is a form of cross-domain feature adaptation, learning to project features into a domain-invariant quality space. The SupCon objective with cross-domain positive pairs is equivalent to domain-adversarial training but without a domain classifier.

---

## 8. Conclusion

We presented QARA, a lightweight supervised contrastive adapter that achieves statistically significant cross-domain quality detection for LLM ReAct agents. Key takeaways:

1. **Raw embeddings encode topic, not quality** — the cross-domain gap (AUROC 0.96 → 0.50) is a diagnostic of this conflation, not a failure of the embedding model.
2. **SupCon with cross-domain positive pairs** is the correct inductive bias — it explicitly teaches the adapter to cluster reasoning quality across domain boundaries.
3. **Domain format compatibility is paramount** — adding format-incompatible domains (code, math) hurts performance; format-compatible domains (NQ + HotpotQA, both ReAct factual QA) complement each other.
4. **The improvement is stable** — all 5 CV folds show positive delta (sign test p = 0.031); QARA reduces cross-domain AUROC variance by 1.7×.
5. **Deployment is lightweight** — ~98K adapter parameters, ~1s training, ~9ms total inference overhead.

The broader implication: for quality detection in LLM agentic pipelines, the choice of calibration domain is as important as the method. A well-matched auxiliary domain transforms a near-chance cross-domain predictor into a useful deployment signal.

---

## Appendix A: Experiment Summary

| Exp | Description | Key Result |
|-----|------------|-----------|
| 18 | HotpotQA baseline within-domain | AUROC varies with topic overlap |
| 21–23 | Multi-domain within-domain | AUROC 0.966–1.000 |
| 30 | Online calibration cold-start | 50 chains → AUROC > 0.90 |
| 31 | QARA initial results (leakage caveat) | Cross-domain AUROC 1.0 (inflated) |
| 32 | QARA rigorous 50/50 split + 5-fold CV | Cross-domain +0.099, σ reduced 2.3× |
| 33 | 4-domain QARA (GSM8K + HumanEval) | Hurts: 0.553 < 0.596 (format mismatch) |
| 34 | QPPGService deployment demo | Lifecycle COLD→WARMING→DEPLOYED works |
| 35 | NQ 3-domain QARA + 5-fold CV | HP+NQ: NQ→HP 0.628 ± 0.077 (p=0.031) |

## Appendix B: Reproducibility

All code, data caches, and results are in the QPPG repository:

- Experiment scripts: `experiments/exp{18–35}_*.py`
- HotpotQA cache (~1,619 chains): `results/exp18_agent_step_failure/cache/`
- NQ cache (404 chains): `results/exp35_nq_3domain_qara/nq_cache/`
- TriviaQA chains: `results/exp23_multidomain/cross_domain_data.json`
- QPPGService: `qppg_service/` (service.py, server.py)
- Exp 35 full results: `results/exp35_nq_3domain_qara/results.json`

**Runtime**: All experiments reproducible in <15 minutes on CPU (M-series MacBook).
**API cost for exp35**: $0.00 (all NQ chains cached from initial run; 404 cached entries).
**Dependencies**: Python 3.12, numpy, scipy, scikit-learn, mlx-lm / sentence-transformers, anthropic, torch.

## Appendix C: QARA Architecture Detail

```python
class QARAAdapter(nn.Module):
    def __init__(self, in_dim=384, hidden=256, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=-1, keepdim=True)  # L2 normalise

# SupCon loss (temperature=0.1, cross-domain positive pairs)
def supcon_loss(z, labels_1hot, temperature=0.1):
    # Positive mask: any two chains sharing label=1 (correct), across all domains
    pos_mask = labels_1hot @ labels_1hot.T
    pos_mask.fill_diagonal_(0)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float('-inf'))  # exclude self
    log_sum = torch.logsumexp(sim, dim=1, keepdim=True)
    log_probs = sim - log_sum
    loss = -(pos_mask * log_probs).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return loss.mean()

# Training: 200 epochs, Adam lr=3e-4, cosine LR decay
```

## References

- Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Khosla, P. et al. (2020). Supervised Contrastive Learning. *NeurIPS 2020*.
- Wang, W. et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression. *NeurIPS 2020*.
- Wang, X. et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
- Farquhar, S. et al. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature 2024*.
- Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. *arXiv 2207.05221*.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Ben-David, S. et al. (2010). A Theory of Learning from Different Domains. *Machine Learning 79*.
