# We Built a Local LLM Judge That Costs $0 to Run — and It Works

**AUROC 0.747 on HotpotQA. Zero API calls at inference time. Ships in one line of pip.**

At Anthropic rates, running a Haiku judge on every LLM agent chain costs roughly $0.001–$0.005 per chain. That sounds trivial until you're running 10,000 chains a day in production — suddenly you're looking at $50/day just for confidence scoring. We wanted to see how far pure behavioral features could go.

The answer: surprisingly far.

## The Problem

LLM agents fail silently. A ReAct agent tasked with multi-hop question answering will confidently output a wrong answer with no signal to the caller. Existing solutions either require a separate judge call (another LLM API round-trip, more latency, more cost) or return no confidence at all.

We've been building `llm-guard-kit` to solve this. The core insight: **wrong chains behave differently than correct ones**, and you can detect this without ever calling an LLM at inference time.

## The MiniJudge

`MiniJudge` is a logistic regression model trained on 11 hand-engineered behavioral features extracted from ReAct agent chains:

- **Loop rate**: How often does the agent repeat the same search query?
- **Empty observation rate**: How often does retrieval return nothing?
- **Answer gap**: What fraction of steps lack a clear answer trajectory?
- **Step norm, obs util, coherence drop**, and 5 more structural signals

These 11 numbers encode what a chain "feels like" — and wrong chains feel different. They loop more. They retrieve less. They backtrack.

We trained MiniJudge using **knowledge distillation**: instead of hard correct/wrong labels, we used soft labels from Claude Sonnet (0.0 = good, 0.5 = borderline, 1.0 = poor). This gives logistic regression a richer gradient signal and lets the model learn borderline cases that hard labels would throw away.

## The Results

On HotpotQA (n=200, 5-fold CV):

| Signal | AUROC | Inference cost |
|--------|-------|---------------|
| Sonnet judge | 0.774 | ~$0.005/chain |
| MiniJudge (SC_OLD + LogReg) | **0.747** | **$0.000/chain** |
| Haiku judge | 0.706 | ~$0.001/chain |
| Raw behavioral ensemble | 0.817* | $0.000/chain |

*SC_OLD within-domain; 0.659 on TriviaQA cross-domain (n=1000, validated)

MiniJudge gets within **2.7 pp of Sonnet** at zero inference cost. That's the distillation paying off — the model has internalized Sonnet's judgment into 11 float multiplications.

## Cross-Domain Reality Check

We ran live chain generation across 4 domains using the Haiku API (total cost: $3.23):

- **2WikiMultiHop** (n=200): SC_OLD AUROC = **0.703** [0.628, 0.775] ✅ valid
- **TriviaQA** (n=1,000): SC_OLD AUROC = 0.659 [0.614, 0.705] ✅ valid
- **MuSiQue** (n=200): AUROC = 0.613 — CI too wide; extremely hard multi-hop (73% error rate)
- **NQ open** (n=200): AUROC = 0.524 — CI crosses 0.5, open-domain factoid near-random

The honest finding: behavioral features generalize well to multi-hop QA (TriviaQA, 2Wiki) but not to open-domain factoid (NQ). Wrong NQ chains don't loop — they just answer confidently from model knowledge without retrieval. That's a different failure mode that SC_OLD can't detect.

## Using It

```python
pip install llm-guard-kit

from llm_guard import MiniJudge

judge = MiniJudge()  # loads pre-trained model
risk = judge.score(question, steps, final_answer)  # → float in [0,1]

if risk > 0.65:
    # flag for review or trigger retry
```

For highest accuracy, blend with P(True):

```python
from llm_guard import AgentGuard, MiniJudge, probe_ensemble_blend

guard = AgentGuard(api_key=key)
result = guard.score_with_ptrue(question, steps, final_answer)
mini_risk = MiniJudge().score(question, steps, final_answer)

blended = probe_ensemble_blend(mini_risk, result.ptrue_risk, alpha=0.25)
```

## What's Next

The 2.7 pp gap between MiniJudge and Sonnet can close with more training data. Every production deployment running with `contribute_labels=True` feeds anonymized behavioral features back into the retrain pipeline — 11 floats and a bit, nothing else, SHA-256 hashed domain identifier.

If you're running LLM agents in production and want confidence scoring without the API overhead, give it a try:

```
pip install llm-guard-kit==0.17.0
```

Source: [github.com/your-org/llm-guard-kit] | Docs: [pypi.org/project/llm-guard-kit]
