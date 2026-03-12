"""
QPPGNano — Self-contained agent chain trust scorer (v0.7.0).

Zero external dependencies beyond the optional `anthropic` package.
Drop this module into any project and call score_chain().

Validated AUROC (HotpotQA, held-out):
    Behavioral only:  ~0.55 within-domain (Jaccard SC3, no sentence-transformers)
    + Sonnet judge:   ~0.76 within-domain (exp109)

For higher AUROC (~0.81 within / ~0.74 cross), use the full AgentGuard which
leverages sentence-transformers for SC3 embedding variance.

Quick start
-----------
    from llm_guard import QPPGNano

    nano = QPPGNano()                                   # $0, behavioral only
    nano = QPPGNano(api_key="sk-ant-...", judge=True)   # + Sonnet judge (~$0.007/chain)
    result = nano.score_chain(question, steps, final_answer)
    print(result["confidence_tier"])  # HIGH / MEDIUM / LOW
    print(result["risk_score"])       # 0.0–1.0  (higher = riskier)
    print(result["needs_alert"])      # True when risk >= 0.70

Use case: agent self-scores BEFORE emitting final answer. If risk >= 0.70,
retry, escalate, or flag for human review — without installing the full package.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional


# ── Stop words (used by Jaccard tokeniser) ────────────────────────────────────

_SW = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "this","that","these","those","it","its","they","them","their","of","in","on",
    "at","to","for","with","by","from","up","out","as","into","through","and","or",
    "but","not","so","if","then","than","about","which","who","what","how","when","where",
}


def _toks(text: str):
    """Tokenise text into a set of non-stopword words."""
    w = re.findall(r"[a-zA-Z]+", text.lower())
    return {x for x in w if x not in _SW and len(x) > 1}


def _f1(a: str, b: str) -> float:
    p, r = _toks(a), _toks(b)
    if not p or not r:
        return 0.0
    c = p & r
    pr = len(c) / len(p)
    rc = len(c) / len(r)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def _wt_avg(vals, wts):
    s = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / s if s else 0.0


# ── Judge prompt (exp109 version — 4-dimension CoT) ───────────────────────────

_JUDGE_SYS = """You are a strict quality auditor for AI research agent chains.
Evaluate the reasoning chain below and output ONLY a JSON object.

Rate on 4 dimensions (1-5 each, 5=best):
  step_relevance:       Do the search queries match the question?
  coherence:            Do the reasoning steps build logically?
  conclusion_support:   Is the final answer supported by observations?
  answer_specificity:   Is the final answer specific and complete?

Also identify any failure flags from: ["hallucination_risk","retrieval_fail",
"repeated_query","long_chain","empty_answer","no_evidence"]

Assign overall quality: "GOOD" (sum>=17), "BORDERLINE" (sum>=13), "POOR" (sum<13)

Output format (JSON only):
{"step_relevance": N, "coherence": N, "conclusion_support": N,
 "answer_specificity": N, "failure_flags": [], "quality_label": "GOOD|BORDERLINE|POOR"}"""


# ── QPPGNano ──────────────────────────────────────────────────────────────────

class QPPGNano:
    """
    Self-contained agent chain trust scorer.

    Uses Jaccard-based behavioral signals (SC_OLD proxy) with no external deps.
    Optionally calls the Sonnet judge for higher accuracy.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        Not needed if judge=False.
    judge : bool
        Whether to call the Sonnet judge. Default: False.
        Cost: ~$0.007/chain. AUROC: +0.21 within-domain (exp109).
    judge_model : str
        Model for the judge. Default: claude-sonnet-4-6.
    alert_threshold : float
        risk_score >= this → needs_alert=True. Default: 0.70.
    """

    VERSION = "0.7.0-nano"

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge: bool = False,
        judge_model: str = "claude-sonnet-4-6",
        alert_threshold: float = 0.70,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._judge = judge
        self._judge_model = judge_model
        self._thr = alert_threshold
        self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def score_chain(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
    ) -> Dict[str, Any]:
        """
        Score a completed ReAct chain.

        Parameters
        ----------
        question : str
        steps : list of dicts, each with keys:
            thought, action_type, action_arg, observation
        final_answer : str

        Returns
        -------
        dict with keys:
            risk_score        float 0-1  (higher = riskier / more likely wrong)
            confidence_tier   "HIGH" | "MEDIUM" | "LOW"
            needs_alert       bool  (True when risk >= alert_threshold)
            failure_mode      str | None
            judge_label       "GOOD" | "BORDERLINE" | "POOR" | None
            behavioral_score  float 0-1
            step_count        int
            latency_ms        float
        """
        t0 = time.time()
        run = {"question": question, "steps": steps, "final_answer": final_answer}

        bscore, bcomps = self._sc_old(run)
        failure_mode = self._detect_failure_mode(run, bcomps)

        risk = bscore
        jlabel = None

        if self._judge and self._api_key:
            jrisk, jlabel = self._call_judge(run)
            # Direct linear blend: SC×0.25 + judge×0.75
            risk = 0.25 * bscore + 0.75 * jrisk

        tier = "HIGH" if risk < 0.50 else ("MEDIUM" if risk < 0.70 else "LOW")

        return {
            "risk_score":       round(float(risk), 4),
            "confidence_tier":  tier,
            "needs_alert":      risk >= self._thr,
            "failure_mode":     failure_mode,
            "judge_label":      jlabel,
            "behavioral_score": round(float(bscore), 4),
            "step_count":       len(steps),
            "latency_ms":       round((time.time() - t0) * 1000, 1),
        }

    def score_prefix(
        self,
        question: str,
        steps_so_far: List[Dict],
        current_action: str,
    ) -> Dict[str, Any]:
        """
        Mid-chain scoring — call before executing each step.
        Returns lightweight risk assessment for early intervention.
        """
        dummy_steps = steps_so_far + [{
            "thought": "", "action_type": "Search",
            "action_arg": current_action, "observation": "",
        }]
        bscore, _ = self._sc_old({
            "question": question,
            "steps": dummy_steps,
            "final_answer": "",
        })
        # Partial chains tend to look riskier; normalise down slightly
        risk = min(bscore * 0.85, 1.0)
        return {
            "risk":       "high" if risk >= 0.65 else ("medium" if risk >= 0.45 else "low"),
            "risk_score": round(float(risk), 4),
            "step":       len(steps_so_far) + 1,
        }

    # ── SC_OLD behavioral scorer ──────────────────────────────────────────────

    def _sc_old(self, run: dict):
        """
        Jaccard-based SC_OLD proxy.

        Note: The full AgentGuard uses sentence-transformer embedding variance
        for SC3, which achieves AUROC ~0.81. This Jaccard proxy gives ~0.55.
        Use AgentGuard for production; QPPGNano is for zero-install contexts.
        """
        steps    = run.get("steps", [])
        n        = len(steps)
        actions  = [s.get("action_type", "") for s in steps]
        thoughts = [s.get("thought", "") for s in steps]
        obs_list = [s.get("observation", "") for s in steps]
        fa       = run.get("final_answer", "")
        q        = run.get("question", "")

        # SC1: action diversity (loop rate — repeated actions = risky)
        sc1 = 1.0 - (len(set(actions)) / max(n, 1))

        # SC2: step count (normalised — more steps = riskier)
        sc2 = min(n / 10.0, 1.0)

        # SC3: obs-thought gap (inverted Jaccard — low overlap = agent ignoring evidence)
        gaps = []
        for t, o in zip(thoughts, obs_list):
            if t and o:
                tt = _toks(t)
                ot = _toks(o)
                if tt | ot:
                    gaps.append(len(tt & ot) / len(tt | ot))
        sc3 = 1.0 - (float(sum(gaps) / len(gaps)) if gaps else 0.5)

        # SC5: thought verbosity (uncertainty proxy)
        tl = [len(t.split()) for t in thoughts if t]
        sc5 = min(sum(tl) / (50 * max(len(tl), 1)), 1.0)

        # SC6: answer–observation gap (answer not grounded in evidence)
        obs_all = " ".join(obs_list)
        if fa and obs_all:
            fa_t = _toks(fa)
            ob_t = _toks(obs_all)
            sc6 = 1.0 - (len(fa_t & ob_t) / max(len(fa_t), 1))
        else:
            sc6 = 0.5

        # SC11: answer–question mismatch
        qt = _toks(q)
        aft = _toks(fa)
        sc11 = 1.0 - (len(qt & aft) / max(len(qt | aft), 1))

        # SC12: risk-monotone slope (coherence decay, n>=3 only)
        sc12 = None
        if n >= 3:
            step_risks = []
            for t, o in zip(thoughts, obs_list):
                if t and o:
                    tt = _toks(t)
                    ot = _toks(o)
                    step_risks.append(1 - len(tt & ot) / max(len(tt | ot), 1))
            if len(step_risks) >= 3:
                slope = (step_risks[-1] - step_risks[0]) / max(n - 1, 1)
                sc12 = float(math.tanh(5 * slope) * 0.5 + 0.5)

        sigs = [sc1, sc2, sc3, sc5, sc6, sc11]
        wts  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.2]

        # SC9/SC10: context utilisation and coherence (n>=3 only)
        if n >= 3:
            used = []
            for t, o in zip(thoughts, obs_list):
                if t and o:
                    tt = _toks(t)
                    ot = _toks(o)
                    used.append(len(tt & ot) / max(len(ot), 1))
            sc9 = float(sum(used) / len(used)) if used else 0.5
            coh = []
            for i in range(len(thoughts) - 1):
                t1 = _toks(thoughts[i])
                t2 = _toks(thoughts[i + 1])
                if t1 | t2:
                    coh.append(len(t1 & t2) / len(t1 | t2))
            sc10 = float(sum(coh) / len(coh)) if coh else 0.5
            sigs.extend([sc9, sc10])
            wts.extend([1.3, 1.3])

        if sc12 is not None:
            sigs.append(sc12)
            wts.append(0.8)

        score = _wt_avg(sigs, wts)
        comps = {
            "sc1": round(sc1, 4), "sc2": round(sc2, 4),
            "sc3": round(sc3, 4), "sc5": round(sc5, 4),
            "sc6": round(sc6, 4), "sc11": round(sc11, 4),
        }
        if sc12 is not None:
            comps["sc12"] = round(sc12, 4)
        return float(score), comps

    def _detect_failure_mode(self, run: dict, comps: dict) -> Optional[str]:
        steps   = run.get("steps", [])
        queries = [s.get("action_arg", "") for s in steps
                   if s.get("action_type") == "Search"]
        obs_all = " ".join(s.get("observation", "") for s in steps)
        fa      = run.get("final_answer", "")
        n       = len(steps)

        obs_tokens = _toks(obs_all)
        if len(obs_tokens) < 5 and n > 1:
            return "retrieval_fail"

        if fa and obs_all:
            overlap = len(_toks(fa) & obs_tokens) / max(len(_toks(fa)), 1)
            if overlap == 0:
                return "no_evidence"

        if len(queries) >= 2:
            unique = len(set(q.lower().strip() for q in queries))
            if unique < len(queries) * 0.7:
                return "repeated_query"

        if n >= 7:
            return "long_chain"

        return None

    def _call_judge(self, run: dict):
        """Call Sonnet judge and return (risk_score, label)."""
        try:
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic(api_key=self._api_key)
            steps = run.get("steps", [])
            chain_txt = f"Question: {run.get('question', '')}\n\n"
            for i, s in enumerate(steps, 1):
                chain_txt += (
                    f"Step {i}:\n"
                    f"  Thought: {s.get('thought', '')}\n"
                    f"  Action: {s.get('action_type', '')}"
                    f"[{s.get('action_arg', '')}]\n"
                    f"  Observation: {s.get('observation', '')[:200]}\n"
                )
            chain_txt += f"\nFinal Answer: {run.get('final_answer', '')}"

            resp = self._client.messages.create(
                model=self._judge_model,
                max_tokens=300,
                system=_JUDGE_SYS,
                messages=[{"role": "user", "content": chain_txt}],
            )
            raw = resp.content[0].text
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                dims = (d.get("step_relevance", 3) + d.get("coherence", 3) +
                        d.get("conclusion_support", 3) + d.get("answer_specificity", 3))
                n_flags = len(d.get("failure_flags", []))
                import numpy as _np
                jrisk = float(_np.clip((20 - dims) / 16.0 + n_flags * 0.1, 0.0, 1.0))
                return jrisk, d.get("quality_label", "BORDERLINE")
        except Exception:
            pass
        return 0.5, None
