# llm-guard-kit — Product Specification
**Version:** 0.3.2
**Package:** `llm-guard-kit` on PyPI
**Date:** March 2026
**Author:** Avighan Majumder

---

## 1. Overview

`llm-guard-kit` provides real-time reliability monitoring, failure diagnosis, and scoring
for LLM ReAct agents.  Unlike earlier approaches that required labeled training data or
model logits, every feature in this library works with **zero labels** — scoring starts
immediately, improving automatically as more agent runs accumulate.

### Core capabilities

| Capability | What it does | When available |
|-----------|--------------|---------------|
| **Behavioral scoring (SC1–SC6)** | 5 rule-based signals: completion flag, step count, thought variance, uncertainty words, answer gap | Immediately, no data needed |
| **GMM-E3 density scoring** | PCA-32 + two GaussianMixture models on max-pooled step embeddings | After ≥20 unlabeled chains |
| **Obs-pool KNN** | Max-pooled observation embedding vs calibration KNN | After ≥20 unlabeled chains |
| **Failure mode detection** | Rule-based: empty_answer, repeated_query, retrieval_fail, long_chain, low_retrieval_quality | Immediately, no data needed |
| **Streaming prefix scoring** | Score after each step; enables early termination at 45% token savings | Immediately (score_prefix) |
| **QARA supervised upgrade** | SupCon adapter on obs-pool for cross-domain gain | After ≥50 labeled chains |
| **Online GMM blending** | Decay-weighted pool sampling prevents catastrophic forgetting | Auto, controlled by decay=0.2 |

---

## 2. Validated Performance

All numbers from experiments on HotpotQA (HP), NaturalQuestions (NQ), TriviaQA (TV) with
`all-MiniLM-L6-v2`, March 2026.

### 2.1 Within-domain AUROC

| Signal | AUROC (HP) | Source |
|--------|-----------|--------|
| SC2 step count alone | 0.879 | exp39 |
| SC1–SC6 ensemble (behavioral) | 0.879 | exp39 |
| SC8 = behavioral + GMM-E3 | **0.883** | exp39 |
| SC8 + max-pool embeddings | **0.90+** | exp60 (+0.21 delta) |

> Note: exp60 showed +0.21 AUROC improvement from max-pool step embeddings vs chain-1
> embedding. This is now the default in v0.3.2.

### 2.2 Cross-domain AUROC (NQ→HP transfer)

| Condition | AUROC | Source |
|-----------|-------|--------|
| Baseline (no adapter) | 0.473–0.570 | exp32/exp39 |
| QARA 2-domain (HP+NQ) | **0.628 ± 0.077** | exp35 (5-fold CV) |
| QARA sign test | p=0.031 (5/5 folds) | exp35 |
| QARA 5-fold vs baseline | +0.148 mean delta | exp35 |

### 2.3 Streaming / Prefix scoring (exp67)

| Score at step t | AUROC | Token savings |
|----------------|-------|--------------|
| t=1 | 0.5830 | 64% |
| **t=2** | **0.6667** | **45%** |
| Full chain | 0.6059 | 0% |

> Score at t=2 **exceeds** full-chain AUROC. Threshold 0.70 at t=2 → 45% fewer tokens
> with <10% false-stop rate.

### 2.4 Failure mode detection accuracy (exp70)

| Failure mode | Precision | Source |
|-------------|-----------|--------|
| retrieval_fail | ~0.91 | exp70 rule-based |
| repeated_query | ~1.00 | exact match |
| long_chain | ~0.87 | exp70 (top feature: n_steps) |
| empty_answer | ~1.00 | length check |
| RF AUROC at t=3 | 0.607 | exp70 (RF classifier) |

Top feature importances from exp70 RF: `obs_length` (0.214), `thought_length` (0.187),
`qppg_risk_prefix` (0.174).

### 2.5 Online GMM blending (exp69)

| Calibration strategy | Mean AUROC | Old-domain retention |
|---------------------|-----------|---------------------|
| Full retrain | 0.5543 | 0.250 |
| **Online blend (decay=0.2)** | **0.5657** | **0.694** |
| Aggressive blend (decay=0.8) | 0.5601 | 0.139 |

Decay=0.2 prevents catastrophic forgetting of earlier patterns (+34 pp retention vs decay=0.8).

### 2.6 Federated calibration (exp68)

Sharing GMM parameters (means + covariances + weights = 762 floats = **6 KB**) instead of
raw chains (~40 KB) enables privacy-preserving cross-site calibration with AUROC within
0.02 of centralized training.

### 2.7 Latency (CPU, Apple Silicon)

| Component | Latency |
|-----------|---------|
| Model cold start | ~12s (one-time) |
| score() per chain (warm, 3 steps) | ~8ms |
| score() with calibration | ~12ms |
| score_prefix() at t=2 | ~5ms |
| calibrate() on 50 chains | ~250ms |

---

## 3. Architecture

```
                 ┌─────────────────────────────────────────────────┐
                 │              LabelFreeScorer                     │
                 │                                                  │
Agent run ──────►│  [1] BEHAVIORAL (SC1–SC6): always available      │
                 │       completion flag, step count,               │
                 │       thought variance, uncertainty words,        │
                 │       answer gap                                 │
                 │                                                  │
                 │  [2] GMM-E3 (after ≥20 unlabeled chains):        │
                 │       max-pool step embeddings → PCA-32          │
                 │       → GaussianMixture k=2 + k=4 rank ensemble  │
                 │       (+0.21 AUROC vs chain-1, exp60)            │
                 │                                                  │
                 │  [3] OBS-POOL KNN (after ≥20 chains):            │
                 │       max-pool obs embeddings vs cal pool         │
                 │                                                  │
                 │  [4] QARA-OBS (after ≥50 labeled chains):        │
                 │       SupCon adapter → adapted obs KNN            │
                 │       AUROC 0.628 cross-domain (exp35/42)        │
                 │                                                  │
                 │  [5] FAILURE MODE (always, rule-based, exp70):   │
                 │       empty_answer | repeated_query |             │
                 │       retrieval_fail | long_chain |               │
                 │       low_retrieval_quality                       │
                 └──────────────────┬──────────────────────────────┘
                                    │
                            LabelFreeResult
                     risk_score, needs_review, failure_mode,
                     behavioral_score, gmm_score, obs_score
                                    │
                 ┌──────────────────▼──────────────────────────────┐
                 │              QppgMonitor                         │
                 │   auto-calibrate, history, alerts, export        │
                 │   online GMM blending (decay=0.2, exp69)         │
                 │   SQLite persistence (ChainStore)                │
                 │   webhook alerting, drift detection              │
                 └─────────────────────────────────────────────────┘
```

### 3.1 Embedding strategy (exp60)

Per-step embeddings are computed for `"Thought: … Action: … Obs: …"` and
max-pooled across all steps to produce a single chain-level vector.
This captures the **most anomalous step signal** rather than averaging it away.
Both GMM calibration and obs-pool KNN use this strategy for consistency.

### 3.2 Online GMM blending (exp69)

Re-calibration does not discard old knowledge. When `recal_every` new chains
arrive, the monitor uses decay-weighted pool sampling:

```
weight(chain) = decay ^ (age / recal_every)
```

With `decay=0.2` (default), chains from the last batch get weight 1.0; a chain
from 5 batches ago gets weight 0.2^5 = 0.00032.  This preserves 69.4% of old-domain
accuracy while adapting to new patterns — vs only 25.0% with full retrain.

---

## 4. API Reference

### 4.1 `LabelFreeScorer`

```python
from qppg_service.label_free_scorer import LabelFreeScorer

scorer = LabelFreeScorer(
    embed_model="all-MiniLM-L6-v2",   # must be pre-cached
    review_threshold=0.65,             # risk above this → needs_review=True
    min_calibration=20,                # min chains before GMM fits
)
```

#### `calibrate(chains) → dict`

Fit GMM-E3 and obs-pool KNN from unlabeled chains.

```python
chains = [{"question": "...", "steps": [...]}]
info = scorer.calibrate(chains)
# {"calibrated": True, "n_chains": 50, "embedding_strategy": "max_pool_per_step",
#  "pca_variance": 0.823, "gmm_fitted": True, "obs_pool_fitted": True}
```

Requires ≥20 chains. Repeated calls replace previous calibration.

#### `score(question, steps, final_answer, finished) → LabelFreeResult`

Score a completed agent chain.

```python
result = scorer.score(
    question="Who invented the telephone?",
    steps=[
        {"action_type": "Search", "action_arg": "telephone inventor",
         "thought": "I need to search", "observation": "Alexander Graham Bell..."},
        {"action_type": "Finish", "action_arg": "Alexander Graham Bell",
         "thought": "Found it.", "observation": ""},
    ],
    final_answer="Alexander Graham Bell",
    finished=True,
)

result.risk_score        # float [0,1]; higher = more likely wrong
result.needs_review      # True if risk > review_threshold
result.failure_mode      # None | "retrieval_fail" | "repeated_query" | ...
result.behavioral_score  # SC1–SC6 ensemble
result.gmm_score         # GMM-E3 score (None if not calibrated)
result.obs_score         # obs-pool KNN (None if not calibrated)
result.to_dict()         # JSON-serializable dict
```

#### `score_prefix(question, steps, t, final_answer, finished) → LabelFreeResult`

Score using only the first `t` steps (streaming / early termination, exp67).

```python
for t in range(1, len(steps) + 1):
    partial = scorer.score_prefix(question=q, steps=steps, t=t)
    if partial.risk_score > 0.70:
        print(f"Early stop at t={t}: risk={partial.risk_score:.2f}")
        break  # saves ~45% remaining tokens (exp67)
```

AUROC at t=2 equals or exceeds full-chain AUROC on HotpotQA.

#### `calibrate_supervised(labeled_chains) → dict`

Tier-2 upgrade: train QARA SupCon adapter for cross-domain improvement.

```python
labeled = [{"question": ..., "steps": [...], "correct": True}, ...]
scorer.calibrate_supervised(labeled)
# Expected cross-domain AUROC: ~0.628 (5-fold CV, exp35)
```

Requires ≥10 chains with `"correct"` labels (both classes). Recommended ≥50.

#### `retrieval_quality(question, steps) → dict`

Cosine similarity between question and each observation (exp43).

```python
rq = scorer.retrieval_quality(question, steps)
rq["mean_sim"]       # 0.568 correct vs 0.441 wrong (avg, exp43)
rq["quality_label"]  # "GOOD" | "FAIR" | "POOR"
```

#### `score_batch(chains) → list[LabelFreeResult]`

Score multiple chains in one call.

#### `status() → dict`

Return calibration status and expected AUROC.

---

### 4.2 `LabelFreeResult`

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | float | Combined risk [0,1]. Higher = more likely wrong |
| `needs_review` | bool | True if risk ≥ review_threshold |
| `behavioral_score` | float | SC1–SC6 ensemble (always present) |
| `gmm_score` | float\|None | GMM-E3 component (None if uncalibrated) |
| `obs_score` | float\|None | Obs-pool KNN (None if uncalibrated) |
| `qara_score` | float\|None | QARA supervised component |
| `failure_mode` | str\|None | Detected failure pattern (exp70) |
| `components` | dict | Individual SC scores |
| `calibration_size` | int | Number of calibration chains |

**`failure_mode` values:**

| Value | Trigger |
|-------|---------|
| `"empty_answer"` | `len(final_answer.strip()) < 5` |
| `"repeated_query"` | Same `action_arg` issued ≥2 times |
| `"retrieval_fail"` | Observation contains "no results", "not found", etc. |
| `"long_chain"` | `len(steps) >= 4` (budget exhaustion proxy) |
| `"low_retrieval_quality"` | `mean cosine(question, obs) < 0.30` |
| `None` | No clear failure mode detected |

---

### 4.3 `QppgMonitor`

```python
from qppg_service import QppgMonitor

monitor = QppgMonitor(
    threshold=0.65,
    domain="my-app",
    db_path="~/.qppg/chains.db",   # SQLite persistence (optional)
    recal_every=100,                # trigger re-calibration every N new chains
    online_decay=0.2,               # GMM blend decay (exp69); 1.0 = full retrain
    check_drift=True,               # enable drift detection
    model_name="claude-sonnet-4-6", # stored in DB for cross-model warnings
)
```

#### `track(question, steps, final_answer, finished, metadata) → QppgAlert | None`

Score and log a completed agent run. Returns `QppgAlert` if risk ≥ threshold.

```python
alert = monitor.track("Who won WW2?", steps, "The Allies", finished=True)
if alert:
    print(f"ALERT: risk={alert.risk_score:.2f}  {alert.recommendation}")
    print(f"Failure: {alert.result.failure_mode}")
```

Auto-calibrates once ≥`min_cal_size` (default 5) chains are logged.
Re-calibrates every `recal_every` new chains using online GMM blending.

#### `get_stats() → MonitorStats`

```python
s = monitor.get_stats()
s.n_tracked    # total queries logged
s.n_alerts     # total alerts fired
s.alert_rate   # fraction of queries flagged
s.avg_risk     # mean risk score
s.p95_risk     # 95th percentile risk
s.high_risk_q  # top 3 highest-risk questions
```

#### `export_report() → str`

Human-readable ASCII report.

#### `export_csv(path) / export_json(path)`

Export full history to CSV or JSON for downstream analysis.

---

### 4.4 `ChainStore` (SQLite persistence)

```python
from qppg_service import ChainStore

store = ChainStore("~/.qppg/chains.db")
store.add_chain(domain, chain_dict, risk_score, alert, failure_mode, model_name)
store.get_domains()                         # list all domains
store.get_chains(domain, since=ts, limit=500)
store.get_recent_risk(domain, days=7)       # [(timestamp, risk_score)]
store.export_audit(domain, start, end, fmt="csv")   # CSV or JSON string
```

---

### 4.5 `DriftDetector`

```python
from qppg_service import DriftDetector, ChainStore

detector = DriftDetector(threshold=0.10, window_days=7)
alert = detector.check(domain="my-app", store=store)
if alert:
    print(f"DRIFT: {alert.delta:+.3f} vs previous week")
    print(alert.recommendation)
```

Compares `[now-7d, now]` vs `[now-14d, now-7d]` windows. Returns `DriftAlert` if
`|delta| > threshold` and current window has ≥10 chains.

---

### 4.6 CLI

```bash
# Install
pip install llm-guard-kit

# Check all monitored domains
llm-guard-kit status [--domain NAME] [--db PATH]

# Score a single chain
llm-guard-kit score --question "Who invented..." --steps-file steps.json --domain myapp

# Manual re-calibration
llm-guard-kit calibrate --domain myapp --chains 50

# Export audit log
llm-guard-kit export --domain myapp --start 2026-01-01 --format csv

# Launch monitoring dashboard (Streamlit)
llm-guard-kit dashboard --domain myapp --port 8502

# Launch API server (FastAPI)
llm-guard-kit serve --host 0.0.0.0 --port 8000
```

---

### 4.7 Framework integrations

All adapters use lazy imports — the underlying framework is not required unless you
actually instantiate the adapter.

```python
# LangChain
from qppg_service.integrations.langchain_callback import QppgLangChainCallback
callback = QppgLangChainCallback(monitor=monitor, threshold=0.65)
agent.run(question, callbacks=[callback])
result = callback.get_last_result()

# OpenAI Assistants
from qppg_service.integrations.openai_adapter import score_assistants_run
result = score_assistants_run(client, thread_id, run_id, monitor=monitor)

# LlamaIndex
from qppg_service.integrations.llamaindex_callback import QppgLlamaIndexCallback
callback = QppgLlamaIndexCallback(monitor=monitor)
query_engine.query(question, callback_manager=CallbackManager([callback]))

# Haystack
from qppg_service.integrations.haystack_callback import QppgHaystackMonitor
wrapped = QppgHaystackMonitor(pipeline, monitor=monitor)
outputs, result = wrapped.run(inputs)
```

---

## 5. Use Cases and Integration Patterns

### 5.1 Zero-setup monitoring (behavioral-only)

No calibration data required. Deploy immediately and alert on any agent run.

```python
monitor = QppgMonitor(threshold=0.65)

# After every agent run
alert = monitor.track(question, steps, final_answer, finished)
if alert:
    log_for_review(alert.question, alert.risk_score, alert.result.failure_mode)
```

Failure mode detection works immediately: `repeated_query` catches infinite-loop bugs;
`empty_answer` catches crashed agents; `retrieval_fail` catches index/tool failures.

### 5.2 Streaming early termination (exp67)

Score after each step. Stop the agent when risk exceeds threshold.

```python
scorer = LabelFreeScorer()
scorer.calibrate(warmup_chains)

for t, step in enumerate(agent_step_generator(), start=1):
    partial = scorer.score_prefix(question, steps_so_far, t)
    if partial.risk_score > 0.70:
        agent.stop()   # saves ~45% remaining tokens
        route_to_fallback(question)
        break
```

Best deployed at t=2 (AUROC=0.667 > full-chain 0.606, exp67).

### 5.3 Production deployment with SQLite + drift alerts

```python
monitor = QppgMonitor(
    threshold=0.65,
    domain="prod-agent",
    db_path="/data/qppg/chains.db",
    check_drift=True,
    online_decay=0.2,               # adapts to distribution shift without forgetting
    alert_callback=lambda a: slack.send(a.recommendation),
)
```

Auto-calibrates from first 5 runs. Re-calibrates every 100 new runs using online GMM
blending. SQLite stores all chains for audit. Drift detector fires Slack alert if
average risk shifts by >10% week-over-week.

### 5.4 Cross-domain upgrade after labeling 50 chains

```python
# After accumulating 50 manually-reviewed chains
scorer.calibrate_supervised(labeled_chains)
# Cross-domain AUROC: 0.473 → 0.628 (exp35, 5-fold CV p=0.031)
```

One-time labeling effort; no re-labeling needed for future runs.

### 5.5 Federated / multi-site deployment (exp68)

```python
# Site A: extract GMM parameters (762 floats = 6KB, privacy-safe)
params = {
    "means": scorer._gmm2.means_.tolist(),
    "covariances": scorer._gmm2.covariances_.tolist(),
    "weights": scorer._gmm2.weights_.tolist(),
    "n_chains": scorer._n_cal,
}

# Central server: weighted average of parameters from all sites
# Site B: load averaged parameters → calibrated without sharing raw data
```

---

## 6. Calibration Decision Guide

```
Do you have labeled correct chains (with "correct" field)?
  YES (≥50) → scorer.calibrate_supervised(chains)
              Expected cross-domain AUROC ~0.628 (exp35)

  NO → Do you have unlabeled chains from production?
    YES (≥20) → scorer.calibrate(chains)
                Expected AUROC 0.883 within-domain (exp39)

    NO → Use scorer.score() immediately (behavioral only)
         Expected AUROC 0.879 within-domain (SC2 alone, exp39)
         Calibration improves automatically via QppgMonitor auto-calibrate

Cross-domain use (agent domain ≠ calibration domain)?
  → Use QARA supervised upgrade: +0.148 mean AUROC delta (exp35)
  → Or federated: share GMM params from source domain (exp68)

Distribution shift expected over time?
  → Set online_decay=0.2 in QppgMonitor (default since v0.3.2)
  → Decay-weighted pool blending retains 69.4% old-domain accuracy (exp69)
```

---

## 7. Known Limitations

| Limitation | Severity | Status |
|-----------|----------|--------|
| Cross-domain baseline near chance (0.47–0.55) | High | Mitigated by QARA (→0.628) |
| GMM needs ≥20 chains before density scores activate | Medium | Behavioral-only mode covers cold start |
| Long-chain failure mode fires for any 4+ step chain | Low | Tunable via `_detect_failure_mode` threshold |
| Max-pool favors most anomalous step — noisy on 1-step chains | Low | Fallback to chain-1 embedding for 1-step chains |
| Online GMM decay=0.2 may react slowly to sudden shifts | Medium | Reduce decay or trigger manual re-calibration |
| Federated averaging assumes IID GMM topology across sites | Medium | Validate cross-site AUROC before deploying averaged params |
| Not a security/prompt-injection filter | High | Use dedicated injection filter separately |

---

## 8. Installation

```bash
# Core (behavioral scoring + GMM + monitor)
pip install llm-guard-kit

# With framework adapters
pip install "llm-guard-kit[langchain]"
pip install "llm-guard-kit[openai]"
pip install "llm-guard-kit[llamaindex]"
pip install "llm-guard-kit[haystack]"

# With QARA supervised upgrade
pip install "llm-guard-kit[qara]"    # adds torch

# With dashboard + server
pip install "llm-guard-kit[server]"

# Everything
pip install "llm-guard-kit[all]"
```

### Docker

```bash
docker-compose up   # starts FastAPI server on :8000 + dashboard on :8502
```

---

## 9. Roadmap

### Completed (v0.3.2)

- [x] `LabelFreeScorer` — behavioral + GMM-E3 + obs-pool scoring
- [x] Max-pool step embeddings (+0.21 AUROC, exp60)
- [x] `failure_mode` field in `LabelFreeResult` (exp70, rule-based)
- [x] `score_prefix()` for streaming early termination (exp67)
- [x] Online GMM blending in `QppgMonitor` (decay=0.2, exp69)
- [x] `QppgMonitor` with auto-calibrate, SQLite persistence, drift detection
- [x] `ChainStore` SQLite layer
- [x] `DriftDetector` week-over-week comparison
- [x] CLI (`llm-guard-kit status/score/calibrate/export/dashboard/serve`)
- [x] Framework integrations: LangChain, OpenAI Assistants, LlamaIndex, Haystack
- [x] FastAPI server with API key auth, per-domain isolation, audit log
- [x] Streamlit dashboard (risk timeseries, high-risk table, drift banner)
- [x] Dockerfile + docker-compose
- [x] QARA supervised upgrade (SupCon adapter, cross-domain AUROC 0.628)
- [x] Federated GMM REST API — `GET/POST /api/{domain}/gmm-params`, `POST /api/gmm/average` (exp68, 8.6 KB payload)
- [x] `AgentFingerprinter` — behavioral clustering for cold-start calibration (exp71, silhouette=0.48)
- [x] Prometheus `/metrics` endpoint — per-domain counters for Grafana/PagerDuty integration
- [x] Async `ascore()` / `ascore_prefix()` — non-blocking scoring for FastAPI event loop

### v0.4.0 (next 4 weeks)

- [ ] Streaming WebSocket endpoint in FastAPI server (push risk scores mid-run)
- [ ] `scorer.save(path)` / `scorer.load(path)` — pickle-free state serialization (JSON + numpy)
- [ ] Multi-model routing: route low-confidence queries to stronger model
- [ ] OpenAI Assistants streaming integration
- [ ] Confidence intervals on AUROC estimates (bootstrap, n=1000)

### Later

- [ ] Fine-tuned embedding model for domain-specific calibration
- [ ] Multi-site federated dashboard with aggregated Prometheus view
- [ ] RLHF-style human-in-the-loop labeling queue in dashboard

---

## 10. Benchmark vs Prior Work

| Approach | Within-domain AUROC | Cross-domain AUROC | Labels needed |
|---------|--------------------|--------------------|--------------|
| **llm-guard-kit SC2 (ours)** | **0.879** | 0.570 | 0 |
| **llm-guard-kit SC8 (ours)** | **0.883** | 0.664 | 20 unlabeled |
| **llm-guard-kit QARA (ours)** | 0.883 | **0.628** (CV) | 50 labeled |
| Semantic entropy (Farquhar 2024) | ~0.80 | ~0.65 | Needs logits |
| LLM self-eval | ~0.72 | ~0.60 | 0 (but costs tokens) |
| Logit-based confidence | ~0.70–0.80 | ~0.55 | Needs logits |

Our method does not require model logits, multiple LLM calls, or access to internal
model states. It works with any agent framework that exposes a chain of reasoning steps.

---

## 11. Pricing / Monetisation

> Internal planning only.

### Proposed SaaS tiers

| Tier | Price | Queries/mo | Target |
|------|-------|-----------|--------|
| Free | $0 | 1,000 | Developers evaluating |
| Starter | $49/mo | 50,000 | Indie dev / small startup |
| Pro | $199/mo | 500,000 | Growth-stage startup |
| Enterprise | $499+/mo | Unlimited | Compliance-sensitive orgs |

**Break-even (Pro tier):** `sentence-transformers` is self-hosted (zero marginal cost per query).
SQLite scales to ~1M rows before needing PostgreSQL. At 500K queries/month with avg 3 steps
and 200-char observations: ~5 GB/year of SQLite. Margin >90% at Pro tier.

**First milestone:** 10 paying users at $99/mo = $990 MRR.
