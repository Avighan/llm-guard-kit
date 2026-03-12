# llm-guard-kit

**Real-time reliability monitoring, A2A trust management, and self-repair for LLM agents.**

[![PyPI](https://img.shields.io/pypi/v/llm-guard-kit.svg)](https://pypi.org/project/llm-guard-kit/)
[![Python](https://img.shields.io/pypi/pyversions/llm-guard-kit.svg)](https://pypi.org/project/llm-guard-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What it does

`llm-guard-kit` wraps any ReAct / tool-calling LLM agent with a reliability stack — **no labels required on day one**:

| Component | What it does |
| --------- | ------------ |
| `AgentGuard` | Score completed chains with SC_OLD behavioral signals + optional Sonnet judge. Emit A2A trust objects. |
| `A2ATrustObject` | Structured confidence envelope for agent-to-agent handoff (answer + risk + tier + failure_mode + hint). |
| `QueryRewriter` | When Agent A has low confidence, generate 3 diverse query reformulations for Agent B. |
| `LabelFreeScorer` | Raw behavioral risk scoring in <15 ms. Zero cold-start. |
| `QppgMonitor` | Drop-in agent monitor. Auto-calibrates, fires alerts, persists to SQLite, exports CSV. |
| `FailureTaxonomist` | Diagnoses *why* a chain failed (retrieval failure, excessive search, hallucination, …). |
| `SelfHealer` | Converts failure diagnosis into prompt injections that repair the agent mid-run. |

**Validated AUROC — held-out evaluation (HotpotQA within-domain, TriviaQA cross-domain):**

| Method | Within-domain | Cross-domain | Cost/chain |
| ------ | ------------- | ------------ | ---------- |
| **MiniJudge** (SC_OLD + LogReg, distilled from Sonnet) | **0.747 ± 0.10** | — | **$0** |
| SC_OLD behavioral ensemble (n=0 labels) | **0.817** | 0.703 (2Wiki) / 0.659 (TV) | $0 |
| SC_OLD + Sonnet judge (J5) | 0.777 | **0.741** | ~$0.007 |
| Conformal alert precision at FPR ≤ 10% | **0.908** | — | — |
| Mid-chain Haiku at step 2 | 0.683 | — | ~$0.001/step |

> v0.17.0: `MiniJudge` achieves 0.747 AUROC (HP 5-fold CV) at zero inference cost — within 2.7 pp of Sonnet judge.
> Cross-domain live validation (exp156/156b): **2WikiMultiHop 0.703** [0.628, 0.775] ✅ · **TriviaQA 0.659** [0.614, 0.705] ✅ · MuSiQue 0.613 (CI wide) · NQ 0.524 (open-domain factoid, near-random).
> All figures from held-out evaluation. See `docs/production_integration.md` for full methodology.

---

## Install

```bash
# Core (no API key needed)
pip install llm-guard-kit

# With specific framework integrations
pip install "llm-guard-kit[langchain]"   # LangChain agents
pip install "llm-guard-kit[openai]"      # OpenAI Assistants
pip install "llm-guard-kit[llamaindex]"  # LlamaIndex
pip install "llm-guard-kit[haystack]"    # Haystack pipelines

# HTTP server + dashboard
pip install "llm-guard-kit[server]"

# Everything
pip install "llm-guard-kit[all]"
```

Requires Python 3.9+.

---

## Table of Contents

1. [MiniJudge — $0 Local Judge (v0.17.0)](#0-minijudge--0-local-judge-v0170)
2. [AgentGuard + A2A Trust (v0.6.0)](#1-agentguard--a2a-trust-v060)
3. [Quick Start — Drop-in Monitor](#1-quick-start)
4. [Framework Integrations](#2-framework-integrations)
5. [Full Pipeline — Detect → Diagnose → Repair](#3-full-pipeline)
6. [Persistence & Auto-Calibration](#4-persistence--auto-calibration)
7. [CLI Reference](#5-cli-reference)
8. [HTTP API Server](#6-http-api-server)
9. [Monitoring Dashboard](#7-monitoring-dashboard)
10. [Docker Deployment](#8-docker-deployment)
11. [SaaS API Key Auth](#9-saas-api-key-auth)
12. [Drift Detection](#10-drift-detection)
13. [Agent Step Format](#agent-step-format)

---

## 0. MiniJudge — $0 Local Judge (v0.17.0)

`MiniJudge` is a logistic regression model trained on 11 behavioral features (SC_OLD), distilled from Sonnet soft labels. **AUROC 0.747 on HotpotQA, zero inference cost.**

```python
from llm_guard import MiniJudge

judge = MiniJudge()  # loads pre-trained weights automatically
risk = judge.score(question, steps, final_answer)  # float in [0, 1]
```

For maximum accuracy, blend MiniJudge with P(True):

```python
from llm_guard import MiniJudge, AgentGuard, probe_ensemble_blend

guard = AgentGuard(api_key="sk-ant-...")
result = guard.score_with_ptrue(question, steps, final_answer)
mini_risk = MiniJudge().score(question, steps, final_answer)
blended = probe_ensemble_blend(mini_risk, result.ptrue_risk, alpha=0.25)  # AUROC ~0.74
```

Opt-in telemetry — feeds your data back into the retrain pipeline:

```python
guard = AgentGuard(
    api_key="sk-ant-...",
    contribute_labels=True,
    telemetry_token="ghp_...",         # GitHub PAT with repo scope
    telemetry_repo="your-org/llm-guard-labels",
)
# Every score_chain() + update_isotonic(feedback) call sends 11 floats + 1 bit
```

---

## 1. AgentGuard + A2A Trust (v0.6.0)

### Chain scoring with validated behavioral signals

```python
from llm_guard import AgentGuard

# Zero cost — behavioral SC_OLD signals only (~0.81 AUROC within-domain)
guard = AgentGuard()

# With Sonnet judge (~$0.007/chain, ~0.74 AUROC cross-domain)
guard = AgentGuard(api_key="sk-ant-...", use_judge=True)

result = guard.score_chain(
    question="When was the Eiffel Tower built?",
    steps=[
        {"thought": "Search for construction date",
         "action_type": "Search", "action_arg": "Eiffel Tower construction year",
         "observation": "The Eiffel Tower was built from 1887 to 1889..."},
        {"thought": "Completed in 1889",
         "action_type": "Finish", "action_arg": "1889", "observation": ""},
    ],
    final_answer="1889",
)

print(result.confidence_tier)   # "HIGH" / "MEDIUM" / "LOW"
print(result.risk_score)        # 0.0–1.0, higher = more likely wrong
print(result.needs_alert)       # True when risk >= 0.70 (Precision=0.908)
print(result.failure_mode)      # "retrieval_fail" | "long_chain" | None
print(result.judge_label)       # "GOOD" | "BORDERLINE" | "POOR" | None
```

### A2A trust handoff

```python
# Agent A produces a trust object
trust = guard.generate_trust_object(question, steps, final_answer)
payload = trust.to_dict()   # JSON-serialisable for queue/API transport

# Agent B receives it and conditions its strategy
from llm_guard import A2ATrustObject, QueryRewriter

trust = A2ATrustObject.from_dict(payload)
print(trust.downstream_hint)  # "proceed" / "proceed_with_caution" /
                               # "rewrite_query" / "escalate_to_human"

# When Agent A had low confidence, diversify Agent B's queries
rewriter = QueryRewriter(api_key="sk-ant-...")
variants = rewriter.rewrite_if_needed(question, trust)
# variants = [paraphrase, decomposed_sub_question, alternative_angle]
# Returns [] when no rewrite needed (HIGH/MEDIUM confidence)
```

### Mid-chain monitoring

```python
# Call inside your agent loop BEFORE each step executes
step = guard.monitor_step(
    question="When was the Eiffel Tower built?",
    steps_so_far=[step1],
    current_action="Search[Eiffel Tower date]",
)
if step.risk == "high":
    pass  # intervene early — AUROC 0.683 at step 2 (Δ+0.156 vs behavioral)
```

---

## 1. Quick Start

### Zero setup — monitor from query 1

```python
from qppg_service import QppgMonitor

monitor = QppgMonitor(threshold=0.65)   # alert above this risk score

# Call after every agent run
alert = monitor.track(
    question    = "Which city is older, Rome or Athens?",
    steps       = agent_steps,           # see step format below
    final_answer= "Athens",
    finished    = True,
)

if alert:
    print(f"HIGH RISK ({alert.risk_score:.2f}): {alert.recommendation}")

# Get a stats report
print(monitor.export_report())
monitor.export_csv("agent_risk_log.csv")
```

Works on query 1. No training. No labels. AUROC 0.879 within-domain out of the box.

### With SQLite persistence (recommended for production)

```python
monitor = QppgMonitor(
    threshold   = 0.65,
    db_path     = "~/.qppg/chains.db",   # auto-creates on first run
    domain      = "prod",                 # namespace for multi-domain setups
    model_name  = "claude-opus-4-6",
    recal_every = 100,                    # re-calibrate GMM every N new chains
    check_drift = True,                   # auto-detect distributional drift
)
```

---

## 2. Framework Integrations

### LangChain

Drop a callback into any LangChain `AgentExecutor` — no code changes to your agent:

```python
pip install "llm-guard-kit[langchain]"
```

```python
from langchain.agents import AgentExecutor, create_react_agent
from qppg_service.integrations.langchain_callback import QppgLangChainCallback
from qppg_service import QppgMonitor

# With persistence
monitor  = QppgMonitor(db_path="~/.qppg/prod.db", domain="prod")
callback = QppgLangChainCallback(monitor=monitor, threshold=0.65)

# Attach to any AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke(
    {"input": "What year was the Eiffel Tower built?"},
    config={"callbacks": [callback]},
)

# Check result
score = callback.get_last_result()
if score and score.needs_review:
    print(f"HIGH RISK: {score.risk_score:.3f}")
    print(f"Behavioral: {score.behavioral_score:.3f}")
```

**What gets captured automatically:**

- `on_agent_action` → thought + tool name + tool input → Search/Finish step
- `on_tool_end` → tool output → observation
- `on_agent_finish` → final answer + full chain scoring

Tool name mapping: `tavily_search`, `duckduckgo_search`, `wikipedia`, `retriev*` → `"Search"`.

---

### OpenAI Assistants API

```python
pip install "llm-guard-kit[openai]"
```

```python
from openai import OpenAI
from qppg_service.integrations.openai_adapter import score_assistants_run
from qppg_service import QppgMonitor

client  = OpenAI()
monitor = QppgMonitor(db_path="~/.qppg/prod.db", domain="assistants")

# Run your assistant normally
thread = client.beta.threads.create()
client.beta.threads.messages.create(thread.id, role="user", content="When was Python created?")
run = client.beta.threads.runs.create_and_poll(thread.id, assistant_id="asst_xxx")

# Score the completed run
result = score_assistants_run(
    client,
    thread_id = thread.id,
    run_id    = run.id,
    monitor   = monitor,                  # optional: persists to SQLite
    question  = "When was Python created?",  # optional: auto-extracted from thread
)
print(f"Risk: {result.risk_score:.3f}  Review: {result.needs_review}")
```

---

### LlamaIndex

```python
pip install "llm-guard-kit[llamaindex]"
```

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from qppg_service.integrations.llamaindex_callback import QppgLlamaIndexCallback
from qppg_service import QppgMonitor

monitor = QppgMonitor(db_path="~/.qppg/prod.db", domain="llamaindex")
qppg_cb = QppgLlamaIndexCallback(monitor=monitor, threshold=0.65)
Settings.callback_manager = CallbackManager([qppg_cb])

# Your index and query engine work unchanged
index        = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response     = query_engine.query("What is RAG?")

result = qppg_cb.get_last_result()
if result and result.needs_review:
    print(f"HIGH RISK: {result.risk_score:.3f}")
```

---

### Haystack

```python
pip install "llm-guard-kit[haystack]"
```

```python
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from qppg_service.integrations.haystack_callback import QppgHaystackMonitor
from qppg_service import QppgMonitor

# Build your pipeline normally
pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))
pipeline.connect("retriever", "generator")

monitor = QppgMonitor(db_path="~/.qppg/prod.db", domain="haystack")
qppg    = QppgHaystackMonitor(pipeline, monitor=monitor)

# Run through the wrapper instead of pipeline.run()
outputs, result = qppg.run({"retriever": {"query": "Who invented Python?"}})

if result and result.needs_review:
    print(f"HIGH RISK: {result.risk_score:.3f}")
```

---

## 3. Full Pipeline

Detect → Diagnose → Repair in one block:

```python
from qppg_service import QppgMonitor, FailureTaxonomist, SelfHealer

monitor = QppgMonitor(threshold=0.65, db_path="~/.qppg/prod.db")
tx      = FailureTaxonomist()
healer  = SelfHealer()

alert = monitor.track(question, steps, final_answer, finished=True)

if alert:
    # Diagnose WHY it failed
    failure = tx.classify(question, steps, final_answer, finished=True)
    print(failure.primary_mode)    # "EXCESSIVE_SEARCH" | "RETRIEVAL_FAILURE" | ...
    print(failure.explanation)     # human-readable explanation
    print(failure.confidence)      # 0–1

    # Get a repair prompt to inject back into the agent
    action = healer.suggest(failure, question, steps, final_answer)
    print(action.action_type)        # "FORCE_FINISH" | "REPHRASE_QUERY" | ...
    print(action.prompt_injection)   # ready to inject as next agent message
    print(action.urgency)            # "HIGH" | "MEDIUM" | "LOW"
```

**Failure modes detected:**

| Mode | When triggered | Suggested repair |
| ---- | -------------- | ---------------- |
| `RETRIEVAL_FAILURE` | mean cosine(obs, question) < 0.35 | `REPHRASE_QUERY` |
| `EXCESSIVE_SEARCH` | > 4 search steps | `CONSOLIDATE` or `FORCE_FINISH` |
| `CONFLICTING_EVIDENCE` | high thought variance + diverse queries | `CONSOLIDATE` |
| `INSUFFICIENT_EVIDENCE` | weak retrieval + ≥ 2 searches | `ADDITIONAL_SEARCH` |
| `ANSWER_UNSUPPORTED` | answer words absent from reasoning | `VERIFY_ANSWER` |
| `PREMATURE_STOP` | ≤ 1 search, no clean finish | `ADDITIONAL_SEARCH` (urgent) |
| `LOW_RISK` | no flags | `NO_ACTION` |

---

## 4. Persistence & Auto-Calibration

### SQLite store (ChainStore)

```python
from qppg_service import ChainStore

store = ChainStore("~/.qppg/chains.db")

# Query your stored data
domains = store.get_domains()                        # ["prod", "staging"]
pool    = store.get_calibration_pool("prod", n=200) # last 200 chains as dicts
stats   = store.get_domain_stats("prod")
# {"n_chains": 523, "n_alerts": 41, "avg_risk": 0.47, "last_auroc": 0.83}

# Export audit log
csv_str  = store.export_audit("prod", fmt="csv")
json_str = store.export_audit("prod", fmt="json")

# Clear a domain (e.g. after model upgrade)
store.clear_domain("prod")
```

### Mixed-domain warm-up

If you have an existing domain with chains, you can bootstrap a new domain from it:

```python
from qppg_service import ChainStore, LabelFreeScorer

store  = ChainStore("~/.qppg/chains.db")
scorer = LabelFreeScorer()

# Copy 25 chains from "staging" to "prod" as cross-domain calibration
# (boosts cross-domain AUROC from ~0.50 to ~0.81)
pool = store.get_calibration_pool("staging", n=25)
scorer.calibrate(pool)
# Now scorer works on "prod" questions with much better accuracy
```

---

## 5. CLI Reference

```bash
pip install "llm-guard-kit[server]"
```

### Status — see all monitored domains

```bash
llm-guard-kit status
llm-guard-kit status --domain prod
```

Output:

```text
DOMAIN               CHAINS   ALERTS AVG RISK    AUROC DRIFT
--------------------------------------------------------------------
prod                    523       41    0.467    0.883 OK
staging                  89        7    0.512      n/a OK
```

### Score — score a single chain from a JSON file

```bash
# Prepare a chain file
cat > chain.json << 'EOF'
{
  "question": "Who invented the telephone?",
  "steps": [
    {"thought": "I should search.", "action_type": "Search",
     "action_arg": "telephone inventor", "observation": "Alexander Graham Bell..."},
    {"thought": "Found it.", "action_type": "Finish",
     "action_arg": "Alexander Graham Bell", "observation": ""}
  ],
  "final_answer": "Alexander Graham Bell",
  "finished": true
}
EOF

llm-guard-kit score --steps-file chain.json --domain prod
```

Output:

```text
Risk score   : 0.312  (OK)
Behavioral   : 0.287
GMM score    : 0.291
Retrieval    : mean_sim=0.612  min_sim=0.489  (GOOD)
Search steps : 1
```

### Calibrate — warm up a domain

```bash
# From a JSON file of chain dicts
llm-guard-kit calibrate --domain prod --chains-file my_chains.json

# Mixed-domain warm-up (copy from another domain)
llm-guard-kit calibrate --domain prod --source-domain staging --chains 25
```

### Recalibrate — after a model upgrade

```bash
# IMPORTANT: calibration is model-specific. Cross-model AUROC ≈ 0.508 (chance).
llm-guard-kit recalibrate --domain prod --new-model claude-opus-4-6
```

### Export — audit log

```bash
llm-guard-kit export --domain prod --format csv > audit.csv
llm-guard-kit export --domain prod --start 2026-01-01 --end 2026-03-01 --format json
```

### Serve — launch the API server

```bash
llm-guard-kit serve --domain prod --port 8000 --host 0.0.0.0
```

### Dashboard — launch with monitoring UI

```bash
llm-guard-kit dashboard --domain prod --port 8000
# Open http://localhost:8000/dashboard
```

---

## 6. HTTP API Server

For multi-language / microservice deployments:

```bash
pip install "llm-guard-kit[server]"
llm-guard-kit serve --port 8000 --host 0.0.0.0
```

Interactive docs: `http://localhost:8000/docs`

### Score a chain

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who invented Python?",
    "steps": [
      {"thought": "Search for it.", "action_type": "Search",
       "action_arg": "Python creator", "observation": "Guido van Rossum..."},
      {"thought": "Done.", "action_type": "Finish",
       "action_arg": "Guido van Rossum", "observation": ""}
    ]
  }'
```

Response:

```json
{
  "confidence": 0.72,
  "needs_review": false,
  "deployment_status": "DEPLOYED",
  "risk_score": 0.31,
  "behavioral_score": 0.29,
  "gmm_score": 0.33
}
```

### Calibrate (add a verified chain)

```bash
curl -X POST http://localhost:8000/calibrate \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "steps": [...], "correct": true}'
```

### Bulk calibrate (seed from existing logs)

```bash
curl -X POST http://localhost:8000/bulk-calibrate \
  -H "Content-Type: application/json" \
  -d '{"chains": [{"question":"...", "steps":[...], "correct":true}, ...]}'
```

### Deployment status

```bash
curl http://localhost:8000/status
```

### Reset (destructive — clears calibration)

```bash
curl -X POST "http://localhost:8000/reset?confirm=YES_RESET"
```

---

## 7. Monitoring Dashboard

Start the server with dashboard enabled:

```bash
llm-guard-kit serve --domain prod --port 8000 --db ~/.qppg/chains.db --dashboard
# or:
python -m qppg_service.server --domain prod --port 8000 --db ~/.qppg/chains.db --dashboard
```

Open `http://localhost:8000/dashboard`

**Dashboard features:**

- **Deployment status banner** — COLD START / WARMING / DEPLOYED + Est. AUROC
- **Drift alert banner** — fires when mean risk shifts > 0.10 over a 7-day window
- **5 KPI cards** — Total queries, Alerts, Avg risk, Calibration pool size, Avg search steps
- **Risk timeline** — 24h / 7d / 30d selector, alert threshold line, color-coded points
- **Failure mode breakdown** — horizontal bar chart of classified failure types
- **Chain log** — paginated table, sortable, filter All/Alerts-only, search, model column
- **Failure Modes section** — detailed view with descriptions
- **Domain switcher** — switch between domains without restarting
- **Export CSV** — download the current view as CSV

---

## 8. Docker Deployment

### Single command

```bash
git clone https://github.com/avighan/qppg
cd qppg
docker compose up -d
```

The server starts on `http://localhost:8000`.

### docker-compose.yml (included)

```yaml
version: "3.9"
services:
  qppg:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - qppg_data:/data
    environment:
      QPPG_DOMAIN: "default"
      QPPG_DB:     "/data/chains.db"
      QPPG_PORT:   "8000"
      QPPG_HOST:   "0.0.0.0"
      # QPPG_ADMIN_KEY: "your-secret-admin-key"  # enable API key creation
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/status"]
      interval: 30s
volumes:
  qppg_data:
```

### Environment variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `QPPG_DOMAIN` | `default` | Default scoring domain |
| `QPPG_DB` | `/data/chains.db` | SQLite database path |
| `QPPG_PORT` | `8000` | HTTP port |
| `QPPG_HOST` | `0.0.0.0` | Bind address |
| `QPPG_ADMIN_KEY` | *(unset)* | Secret to enable `/api/keys` endpoint |

### Production with nginx reverse proxy

```nginx
server {
    listen 443 ssl;
    server_name qppg.yourcompany.com;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
    }
}
```

---

## 9. SaaS API Key Auth

For multi-tenant deployments where different customers score their own agents:

### Step 1: Start server with admin key and SQLite

```bash
python -m qppg_service.server \
  --port 8000 --db /data/chains.db \
  --admin-key "your-secret-admin-key" \
  --dashboard
```

### Step 2: Create a customer API key

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-secret-admin-key" \
  -d '{"customer_id": "acme-corp", "domain_prefix": "prod"}'
```

Response:

```json
{
  "api_key":       "jN2xR...abc",      ← Save this — shown once only
  "key_id":        "a3f9b2d1c0e4",
  "customer_id":   "acme-corp",
  "domain_prefix": "prod"
}
```

### Step 3: Score with Bearer token auth

```bash
curl -X POST http://localhost:8000/api/prod/score \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer jN2xR...abc" \
  -d '{"question": "When was Python created?", "steps": [...]}'
```

Response:

```json
{
  "risk_score":       0.34,
  "needs_review":     false,
  "behavioral_score": 0.31,
  "gmm_score":        0.37,
  "calibration_size": 47
}
```

### Step 4: Export audit log

```bash
curl "http://localhost:8000/api/prod/audit-log?fmt=csv&start=1700000000" \
  -H "Authorization: Bearer jN2xR...abc" \
  > audit.csv
```

**Rate limit:** 1,000 requests/hour per API key (in-memory sliding window).

**Domain isolation:** Each customer's data is stored under `{customer_id}:{domain}` — fully isolated.

---

## 10. Drift Detection

`DriftDetector` compares mean risk score in the current 7-day window vs the previous 7-day window and fires an alert if the shift exceeds a threshold:

```python
from qppg_service import ChainStore, DriftDetector, DriftAlert

store    = ChainStore("~/.qppg/chains.db")
detector = DriftDetector(threshold=0.10, window_days=7, min_samples=10)

alert = detector.check("prod", store)
if alert:
    print(f"Drift detected!")
    print(f"  Direction:    {alert.direction}")         # "UP" or "DOWN"
    print(f"  Delta:        {alert.delta:+.3f}")        # +0.142
    print(f"  Current mean: {alert.current_mean:.3f}")  # 0.612
    print(f"  Prev mean:    {alert.previous_mean:.3f}") # 0.470
    print(f"  Samples:      {alert.n_current}")         # 47
    print(alert.recommendation)
```

**When drift fires:** Run `llm-guard-kit recalibrate` if you recently upgraded your LLM model (cross-model AUROC ≈ 0.508 — calibration does not transfer across models).

---

## Agent Step Format

Every integration expects steps in this format:

```python
steps = [
    {
        "thought":      "I need to find when the Eiffel Tower was built.",
        "action_type":  "Search",           # "Search" | "Finish" | any custom tool name
        "action_arg":   "Eiffel Tower construction date",
        "observation":  "The Eiffel Tower was built between 1887 and 1889..."
    },
    {
        "thought":      "I now have the answer.",
        "action_type":  "Finish",
        "action_arg":   "1889",
        "observation":  ""
    }
]
```

Framework adapters (LangChain, OpenAI, LlamaIndex, Haystack) build this format automatically.

---

## Retrieval Quality Diagnostics

A standalone signal you can use independently of the full monitor:

```python
from qppg_service import LabelFreeScorer

scorer = LabelFreeScorer()
rq = scorer.retrieval_quality(question, steps)
# {
#   "mean_sim":     0.41,        # average cosine(observation, question)
#   "min_sim":      0.22,        # worst retrieval step
#   "quality_label":"POOR",      # "GOOD" | "OK" | "POOR"
#   "per_step":     [0.52, 0.22, ...]
# }
```

Correct agents: `mean_sim = 0.554`; wrong agents: `0.458` (Δ+0.096, p<0.01).

---

---

## 11. MCP Server — Claude Desktop & Cursor Integration (v0.9.0)

`llm-guard-kit` ships a **Model Context Protocol server** that exposes all scoring tools directly inside Claude Desktop and Cursor — no code required.

### Setup

```bash
pip install "llm-guard-kit[mcp]"
```

### Start the server

```bash
# stdio transport — for Claude Desktop and Cursor (default)
llm-guard-mcp

# SSE transport — for web clients
llm-guard-mcp --sse --port 8765
```

### Add to Claude Desktop

Open `~/.claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "llm-guard-kit": {
      "command": "python3",
      "args": ["-m", "llm_guard_mcp.server"],
      "cwd": "/path/to/your/project",
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Restart Claude Desktop. Eight new tools appear in the tool panel.

### Add to Cursor

Add the same block to `~/.cursor/mcp.json`, then invoke with `@llm-guard-kit` in any Cursor chat.

---

### Tool examples (all outputs verified)

#### `score_chain` — score a completed agent chain

```python
# In Claude / Cursor, call the tool directly. In Python:
import asyncio
from llm_guard_mcp.server import _dispatch

steps = [
    {"thought": "Search for Python creation date.",
     "action_type": "Search", "action_arg": "Python creation year",
     "observation": "Python was created by Guido van Rossum, released in 1991."},
    {"thought": "Found the answer.",
     "action_type": "Finish", "action_arg": "1991", "observation": ""},
]

result = asyncio.run(_dispatch("score_chain", {
    "question": "What year was Python created?",
    "steps": steps,
    "final_answer": "1991",
}))
```

Response:

```json
{
  "chain_id": 1,
  "risk_score": 0.2958,
  "tier": "HIGH",
  "needs_review": true,
  "needs_alert": true,
  "beh_score": 0.2958,
  "ptrue_score": 0.0,
  "rl_score": null,
  "interpretation": "HIGH risk (0.296): Agent failed, contradicted itself, or gave up. Do not use without verification.",
  "action": "Block answer. Alert human reviewer. Do not send to end user."
}
```

> `tier` is behavioral-only when no `ANTHROPIC_API_KEY` is set. With a key, `ptrue_score` is populated and AUROC improves from 0.682 → 0.775.

---

#### `stream_check` — mid-chain abort (call after step 2)

```python
# Check after 2 steps — abort early if chain is failing
result = asyncio.run(_dispatch("stream_check", {
    "question": "What year was Python created?",
    "steps_so_far": steps[:1],   # just the first step
}))
```

Response (on-track chain):

```json
{
  "should_abort": false,
  "risk_score": 0.5376,
  "step": 1,
  "chain_id": null,
  "message": "CONTINUE: Chain looks on-track so far."
}
```

Response (failing chain — empty observations, repeated queries):

```json
{
  "should_abort": false,
  "risk_score": 0.4964,
  "step": 2,
  "chain_id": null,
  "message": "CONTINUE: Chain looks on-track so far."
}
```

> `should_abort=true` fires when `risk_score >= 0.65`. Abort the agent immediately and surface a failure message to the user.

---

#### `submit_feedback` — human label → RL signal

```python
# After a user confirms an answer is correct or wrong
result = asyncio.run(_dispatch("submit_feedback", {
    "chain_id": 1,
    "correct": True,
    "note": "Verified via Wikipedia",
}))
```

Response:

```json
{
  "status": "feedback_recorded",
  "chain_id": 1,
  "label": "correct",
  "n_labeled": 1,
  "retrain_ready": false,
  "retrain_reason": "Need 29 more labels before first retrain (have 1)",
  "note": "Verified via Wikipedia"
}
```

> After 30 labels (min 5 per class), `retrain_ready` becomes `true`. Call `trigger_retrain` or let `score_chain` auto-retrain.

---

#### `get_metrics` — summary stats for last N days

```python
result = asyncio.run(_dispatch("get_metrics", {"days": 7}))
```

Response:

```json
{
  "summary": {
    "days": 7.0,
    "total": 5,
    "aborted": 0,
    "tier_LOW": 0,
    "tier_MEDIUM": 0,
    "tier_HIGH": 5,
    "mean_risk": 0.2952,
    "n_labeled": 2,
    "n_wrong_labeled": 0,
    "label_error_rate": 0.0
  },
  "recent_chains": [
    {"id": 5, "question": "What year was Python created?...", "risk": 0.296, "tier": "HIGH", "labeled": true}
  ],
  "rl_status": {
    "model_trained": false,
    "labels_since_last_train": 2
  }
}
```

---

#### `get_auroc` — rolling AUROC from labeled feedback

```python
result = asyncio.run(_dispatch("get_auroc", {"days": 30}))
```

Response (insufficient labels):

```json
{
  "auroc": null,
  "window_days": 30.0,
  "n_labeled": 2,
  "drift_alert": null,
  "auroc_history": [],
  "baseline_ref": {
    "behavioral_only": 0.682,
    "ptrue_ensemble": 0.775,
    "integration_target": 0.795
  },
  "status": "INSUFFICIENT DATA (need ≥10 labels)"
}
```

Once you have ≥10 labels, `auroc` is populated and `drift_alert` fires if AUROC drops ≥5pp from its peak.

---

#### `trigger_retrain` — retrain RL model from labels

```python
result = asyncio.run(_dispatch("trigger_retrain", {}))
```

Response (not enough labels yet):

```json
{
  "status": "skipped",
  "reason": "Need 28 more labels before first retrain (have 2)"
}
```

Response after 30+ labels (balanced classes):

```json
{
  "status": "trained",
  "n_labels": 42,
  "cv_auroc": 0.831,
  "precision": 0.867,
  "recall": 0.714,
  "model": "LogisticRegression",
  "features": ["risk_score", "beh_score", "ptrue_score", "n_steps_norm", "tier_num"]
}
```

---

#### `get_pending_review` — HIGH-risk chains awaiting labels

```python
result = asyncio.run(_dispatch("get_pending_review", {"limit": 5}))
```

Response:

```json
{
  "pending_count": 3,
  "pending": [
    {
      "chain_id": 3,
      "question": "What year was Python created?",
      "answer": "1991",
      "risk_score": 0.295,
      "tier": "HIGH",
      "how_to_label": "Call submit_feedback with chain_id=3, correct=true/false"
    }
  ],
  "total_labeled": 2,
  "retrain_ready": false,
  "retrain_reason": "Need 28 more labels before first retrain (have 2)"
}
```

---

#### `get_rl_status` — full RL training history and drift

```python
result = asyncio.run(_dispatch("get_rl_status", {}))
```

Response:

```json
{
  "model_trained": false,
  "total_labels": 2,
  "labels_wrong": 0,
  "labels_correct": 2,
  "labels_since_last_train": 2,
  "retrain_ready": false,
  "retrain_reason": "Need 28 more labels before first retrain (have 2)",
  "training_history": [],
  "auroc_trend": [],
  "drift_alert": null,
  "rl_loop_config": {
    "retrain_every_n_labels": 20,
    "min_labels_for_first_train": 30,
    "min_per_class": 5,
    "features": ["risk_score", "beh_score", "ptrue_score", "n_steps_norm", "tier_num"]
  }
}
```

---

### RL feedback loop summary

| Step               | What to call          | When                               |
| ------------------ | --------------------- | ---------------------------------- |
| Agent finishes     | `score_chain`         | Every chain                        |
| User checks step 2 | `stream_check`        | Optional — saves API cost          |
| User marks answer  | `submit_feedback`     | Every manual review                |
| After 30 labels    | `trigger_retrain`     | Auto or manual                     |
| Monitor quality    | `get_auroc`           | Daily / weekly                     |
| Review queue       | `get_pending_review`  | When assigning to human reviewers  |

Labels are stored in SQLite at `~/.llm_guard_mcp/metrics.db`. Persists across restarts.

---

## License

MIT — see [LICENSE](LICENSE).

## Research Background

Built on experiments exp18–53 validating against HotpotQA, NaturalQuestions, TriviaQA, and GSM8K.

Paper draft: `docs/research_paper.md`
