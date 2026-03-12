# LLMGuard User Guide

**Version 0.1.3 · llm-guard-kit**

---

## What is LLMGuard?

LLMGuard wraps any LLM call with a three-stage reliability layer:

1. **Predict** — scores every query for failure risk in <15ms *before* the LLM responds
2. **Diagnose** — clusters accumulated failures into a labeled error taxonomy
3. **Repair** — synthesises targeted fix instructions; injects them automatically on future failures

It is model-agnostic at the scoring layer (uses sentence-transformers locally) and currently uses Claude for LLM calls and for generating repair tools.

**Validated AUROC** across three benchmarks:

| Benchmark | Domain | AUROC |
|-----------|--------|-------|
| MATH-500 | Math | 0.966 |
| HumanEval | Code | 0.993 |
| TriviaQA | Factual QA | 0.992 |

---

## Install

```bash
# Core library only
pip install llm-guard-kit

# With QARA adapter training (requires PyTorch ~2GB)
pip install "llm-guard-kit[qara]"

# With API server
pip install "llm-guard-kit[server]"

# With web dashboard
pip install "llm-guard-kit[server]" streamlit

# Everything
pip install "llm-guard-kit[qara,server]" streamlit
```

Requires Python 3.9+ and an Anthropic API key.

---

## Quick Start (5 minutes)

```python
from llm_guard import LLMGuard

guard = LLMGuard(api_key="sk-ant-...")

# Step 1: calibrate on questions your LLM handles correctly
guard.fit(correct_questions=[
    "What is the capital of France?",
    "What is 12 × 15?",
    # ... 50+ examples recommended
])

# Step 2: query with automatic risk scoring
result = guard.query("What is 15% of 240?")
print(result.answer)      # "36"
print(result.confidence)  # "high" | "medium" | "low"
print(result.risk_score)  # 0.09  (lower = more familiar)
```

---

## Use Cases

### 1. Customer Support Bot

**Problem:** Your support bot gives wrong answers to unfamiliar product questions.

**Solution:** Calibrate on your FAQ bank; flag low-confidence answers for human review.

```python
guard = LLMGuard(api_key="...")

# Calibrate on resolved support tickets where answer was correct
guard.fit(correct_questions=resolved_ticket_questions)

# For each incoming ticket
result = guard.query(incoming_ticket)

if result.confidence == "low":
    route_to_human(incoming_ticket)
else:
    send_to_customer(result.answer)
```

**Continuous improvement:** Submit feedback as tickets are resolved. The KNN pool grows and
repair tools are synthesised from recurring error patterns.

---

### 2. Code Generation (automated calibration)

**Problem:** You have no labeled data, but you can run unit tests.

**Solution:** Use `fit_from_execution()` with a test runner as the verifier.

```python
def test_runner(question, response):
    """Returns True if the generated code passes tests."""
    try:
        namespace = {}
        exec(compile(response, "<llm>", "exec"), namespace)
        return namespace.get("test_result", False)
    except Exception:
        return False

guard = LLMGuard(api_key="...")
guard.fit_from_execution(
    questions=coding_challenges,
    verifier_fn=test_runner,
)

result = guard.query("Write a binary search implementation.")
print(result.confidence)
```

---

### 3. Math / STEM Tutoring

**Problem:** Students get wrong step-by-step solutions; you want targeted fixes.

**Solution:** Use `learn_from_errors()` to cluster wrong answers and synthesise repair prompts.

```python
guard.fit(correct_questions=easy_math_questions)

# After collecting failures from students
guard.learn_from_errors(
    failed_questions=wrong_qs,
    model_answers=model_wrong_answers,
    correct_answers=ground_truth,
)

# Future queries near a known error cluster get auto-fixed
result = guard.query("If a train travels at 60 mph for 2.5 hours, how far does it go?")
print(result.tool_used)   # "error_fix_0" — repair prompt injected
```

---

### 4. Medical / High-Stakes QA

**Problem:** You need to know *before* responding whether the answer is likely reliable.

**Solution:** Use risk_score as a gating signal; never respond if confidence is "low".

```python
result = guard.query(
    "What is the maximum safe dose of paracetamol for an adult?",
    system_prompt="You are a medical information assistant. Be precise and cite guidelines."
)

if result.confidence == "low":
    return "I'm not confident about this — please consult a healthcare professional."
elif result.confidence == "medium":
    return result.answer + "\n\n⚠️ Please verify with a healthcare professional."
else:
    return result.answer
```

---

### 5. Multi-Domain Enterprise (QARA)

**Problem:** You calibrated on legal QA but also need to handle HR and finance questions. Cross-domain AUROC drops to near-chance (~0.48) without adaptation.

**Solution:** Train QARA with labeled data from each domain. Cross-domain AUROC rises to ~0.63.

```python
guard.fit(correct_questions=legal_correct_questions)

guard.fit_qara([
    {"name": "legal",   "questions": legal_qs,   "labels": legal_labels},
    {"name": "hr",      "questions": hr_qs,      "labels": hr_labels},
    {"name": "finance", "questions": finance_qs, "labels": finance_labels},
])

guard.save_qara("enterprise_adapter.pkl")

# Future sessions — load without retraining
guard.fit(correct_questions=legal_correct_questions)
guard.load_qara("enterprise_adapter.pkl")
```

**QARA format note:** Works best when domains share a reasoning format (open-ended factual QA). Math + code have different embedding geometry and benefit less.

---

### 6. Agent Step Monitoring

**Problem:** Your ReAct / tool-use agent takes wrong intermediate steps before producing a final answer.

**Solution:** Use `AgentGuard` to monitor each reasoning step.

```python
from llm_guard import AgentGuard

agent_guard = AgentGuard(api_key="...")
agent_guard.fit(known_correct_reasoning_steps)

for step in agent.run_steps(task):
    result = agent_guard.check_step(step)
    if result.risk_score > 0.8:
        agent.halt_and_retry(step)
```

---

### 7. Deployed API Service (REST)

**Problem:** Your LLM is called from multiple services in different languages.

**Solution:** Deploy the FastAPI server; any service calls `/v1/query` and `/v1/feedback`.

```bash
pip install "llm-guard-kit[server]"
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger UI at `http://localhost:8000/docs`.

**Node.js client:**
```js
const res = await fetch("http://your-server/v1/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "Who wrote Hamlet?" })
});
const { query_id, answer, confidence } = await res.json();
```

---

### 8. Web Dashboard (MVP UI)

For non-developer users or monitoring:

```bash
pip install streamlit
streamlit run app/frontend.py
```

Pages:
- **Dashboard** — live metrics, learning loop progress, confidence accuracy
- **Calibrate** — paste questions, upload CSV, or auto-calibrate via self-consistency
- **Query** — interactive query with risk gauge + one-click feedback
- **Error Analysis** — failure clusters, Prompt Healer
- **QARA Adapter** — train, save, load the cross-domain adapter

---

## How It Works

### Risk Scoring (KNN Anomaly Detection)

1. During calibration, embed all known-correct questions with `all-MiniLM-L6-v2` (384-dim)
2. Build a KNN index (sklearn `NearestNeighbors`, Euclidean distance, k=5)
3. At query time: embed the new question → compute mean distance to k nearest correct examples
4. High distance = unfamiliar territory = high failure risk

Risk thresholds are auto-calibrated from the training distribution:
- `risk_low_threshold`  = 75th percentile of within-correct-set distances
- `risk_high_threshold` = 95th percentile

These work across any domain without manual tuning.

### Routing Logic

```
risk ≤ low_threshold    → direct LLM call              → confidence "high"
risk ≤ high_threshold   → apply repair tool if known   → confidence "medium"
risk > high_threshold   → call with uncertainty flag   → confidence "low"
```

Resource failures (`stop_reason == "max_tokens"`) trigger an automatic retry with 2× tokens regardless of risk tier.

### Prompt Healer

`learn_from_errors()` clusters failure embeddings (KMeans, silhouette-optimal k) then asks the LLM to:
1. Describe the error pattern for each cluster
2. Write a 2-sentence repair instruction

Future queries whose embedding is near a known failure cluster have the repair instruction prepended to the system prompt automatically.

### QARA (Quality-Aware Reasoning Adapter)

**Problem:** Raw MiniLM embeddings conflate topic identity with quality. A factual QA question about Napoleon and a factual QA question about DNA look very different in embedding space — but both should be scored the same way.

**Solution:** Train a 384→256→64 MLP adapter using **Supervised Contrastive loss**:
- Positive pairs = any two **correct** chains, regardless of domain
- Negatives = all other pairs

After training, the adapter maps questions into a 64-dim space where:
- Correct questions from any domain cluster together
- Incorrect questions are pushed apart

Inference is pure numpy (~0.5ms); PyTorch is only needed for training.

---

## Continuous Learning Loop

The `GuardManager` (used by the API server and dashboard) runs three mechanisms automatically as feedback arrives via `/v1/feedback` or the UI:

| Mechanism | Trigger | Effect |
|-----------|---------|--------|
| **KNN Expansion** | Any correct feedback | Query added to calibration pool; KNN re-fitted (<100ms) |
| **Prompt Healing** | Every 5 incorrect answers | `learn_from_errors()` synthesises new repair tools |
| **QARA Re-training** | Every 50 labeled examples (≥10 each class) | Cross-domain adapter re-trained on all accumulated data |

Each mechanism can also be triggered manually from the dashboard or via the API.

---

## GuardResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `answer` | str | LLM response |
| `risk_score` | float | Mean KNN distance; higher = more likely to fail |
| `confidence` | str | `"high"` / `"medium"` / `"low"` |
| `tool_used` | str \| None | Repair tool ID if a cluster fix was applied |
| `cluster_id` | int \| None | Error cluster matched (if any) |
| `was_retried` | bool | True if a resource-failure retry fired |

---

## Configuration

```python
guard = LLMGuard(
    api_key          = "sk-ant-...",
    model            = "claude-haiku-4-5-20251001",   # any Claude model
    embedding_model  = "all-MiniLM-L6-v2",            # any sentence-transformers model
    n_neighbors      = 5,                              # k for KNN scoring
)
```

**Server environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required |
| `GUARD_MODEL` | `claude-haiku-4-5-20251001` | Claude model |
| `GUARD_STATE_PATH` | `guard_state.pkl` | Persistence file |

---

## Limitations

- **Calibration quality matters.** `fit()` needs ≥6 correct examples. More is better — 50–200 is the sweet spot. `fit_from_consistency()` works best when baseline accuracy > 70%.
- **Embeddings detect unfamiliar phrasing, not unfamiliar reasoning.** Two syntactically similar questions requiring very different reasoning may receive similar risk scores.
- **Repair tools are heuristic.** `learn_from_errors()` generates prompt additions using the LLM — they improve average accuracy but are not guaranteed to fix every instance of a cluster.
- **QARA requires labeled data.** You need labeled examples from each target domain (≥10 correct and ≥10 incorrect per domain).
- **Anthropic-only.** OpenAI and Ollama support is on the roadmap.
- **Not a security filter.** This tool detects factual and reasoning failures, not prompt injection, jailbreaks, or adversarial inputs.

---

## FAQ

**Q: How many calibration examples do I need?**
A: Minimum 6. Practically, 50–200 gives stable risk thresholds. The guard keeps improving as you send `is_correct=True` feedback — every correct answer expands the pool.

**Q: Do I need labels to calibrate?**
A: No. Use `fit_from_consistency()` (samples each question 5× — those with 80%+ agreement are treated as correct) or `fit_from_execution()` (automated verifier for code/math/SQL).

**Q: When should I use QARA?**
A: When you're calibrated on domain A but serving queries from domain B, and your observed AUROC on those queries is below ~0.6. Install the `[qara]` extra and provide labeled data from both domains.

**Q: Does QARA require PyTorch at query time?**
A: No. Training uses PyTorch but the weights are extracted to numpy arrays. Inference is pure numpy, adding ~0.5ms per query.

**Q: How does the continuous learning loop work?**
A: Every `POST /v1/feedback` call either expands the calibration pool (if correct) or adds to the error log (if incorrect). The error log triggers `learn_from_errors()` every 5 errors, and QARA re-training every 50 labeled examples. All thresholds are configurable in `app/manager.py`.

**Q: Can I use this with OpenAI models?**
A: The scoring engine is model-agnostic (local sentence-transformers). The LLM call layer currently only supports Anthropic. OpenAI support is planned for the next release.

**Q: How do I save and restore the guard state?**
A: The `GuardManager` automatically saves to `guard_state.pkl` after every feedback call. On restart it loads from that file. To save the QARA adapter separately: `guard.save_qara("adapter.pkl")`.

---

## Citation

```bibtex
@software{majumder2025llmguard,
  author  = {Majumder, Avighan},
  title   = {LLMGuard: KNN-based failure prediction for large language models},
  year    = {2025},
  url     = {https://github.com/avighan/qppg},
  note    = {AUROC 0.966--0.993 on math, code, and factual QA}
}
```

For the QARA adapter specifically:

```bibtex
@techreport{majumder2025qara,
  author = {Majumder, Avighan},
  title  = {QARA: Cross-Domain Quality Detection for LLM ReAct Agents
            via Supervised Contrastive Adapter Learning},
  year   = {2025},
  note   = {Cross-domain AUROC 0.628 on HP+NQ (baseline 0.480)}
}
```
