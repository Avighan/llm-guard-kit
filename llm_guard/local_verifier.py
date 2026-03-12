"""
LocalVerifier — Sklearn LogReg trained on 15 SC behavioral features.

Replaces the Sonnet judge ($0.007/chain) with a local classifier ($0 at inference)
after training on 200 labeled agent runs.

Validated performance (exp111, HotpotQA, 5-fold CV):
    LogReg AUROC:   0.8035 ± 0.078  — beats Sonnet judge (0.7672)
    J_LOCAL AUROC:  0.7846           — SC×1 + LogReg×3 ensemble
    Required labels: 200 chains with correctness labels

Feature set (15 Jaccard-based + answer-side signals):
    sc1_loop_rate         — fraction of repeated action types
    sc2_step_count        — normalised step count
    sc3_obs_thought_gap   — inverted Jaccard obs-thought overlap (high = risky)
    sc5_thought_len       — thought verbosity (uncertainty proxy)
    sc6_ans_obs_gap       — answer not grounded in observations
    sc11_ans_q_mismatch   — answer doesn't overlap with question
    sc9_context_use       — fraction of observation tokens used in thought
    sc10_coherence        — inter-step thought coherence
    sc12_risk_slope       — risk-monotone slope (coherence decay)
    n_steps_norm          — step count normalised by 10
    n_searches_norm       — search count normalised by 8
    is_multi_step         — binary: chain has ≥ 2 steps

Quick start
-----------
    from llm_guard import LocalVerifier

    verifier = LocalVerifier()
    verifier.fit(runs, labels)           # runs: list of dicts, labels: list[bool]
    risk = verifier.predict_risk(question, steps, final_answer)

    # Or use AgentGuard with use_local_verifier=True:
    guard = AgentGuard(use_local_verifier=True)
    guard.fit_verifier(labeled_runs)     # [{"question":..,"steps":..,"final_answer":..,"correct":bool}]
    result = guard.score_chain(question, steps, final_answer)

    # Save / load the trained model:
    verifier.save("local_verifier.pkl")
    verifier = LocalVerifier.load("local_verifier.pkl")
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ── Stop words (same as QPPGNano) ─────────────────────────────────────────────

_SW = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "this","that","these","those","it","its","they","them","their","of","in","on",
    "at","to","for","with","by","from","up","out","as","into","through","and","or",
    "but","not","so","if","then","than","about","which","who","what","how","when","where",
}


def _toks(text: str):
    w = re.findall(r"[a-zA-Z]+", text.lower())
    return {x for x in w if x not in _SW and len(x) > 1}


def _cap_toks(text: str):
    """Capitalized content tokens (named entity proxies)."""
    return {w for w in re.findall(r"[A-Z][a-zA-Z]+", text) if len(w) > 1}


# ── Feature names (exp111) ────────────────────────────────────────────────────

FEATURE_NAMES = [
    "sc1_loop_rate",
    "sc2_step_count",
    "sc3_obs_thought_gap",
    "sc5_thought_len",
    "sc6_ans_obs_gap",
    "sc11_ans_q_mismatch",
    "sc9_context_use",
    "sc10_coherence",
    "sc12_risk_slope",
    "n_steps_norm",
    "n_searches_norm",
    "is_multi_step",
    "ans_entity_match",
    "ans_len_type_match",
    "obs_entity_coverage",
]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(question: str, steps: List[Dict], final_answer: str) -> np.ndarray:
    """
    Extract 15 behavioral + answer-side features from a ReAct chain.

    Returns a 1-D numpy array of shape (15,).
    All features are scaled to roughly [0, 1] — StandardScaler is applied
    at fit time inside the sklearn Pipeline.
    """
    n        = len(steps)
    actions  = [s.get("action_type", "") for s in steps]
    thoughts = [s.get("thought", "") for s in steps]
    obs_list = [s.get("observation", "") for s in steps]
    fa       = final_answer
    q        = question

    # SC1: action loop rate
    sc1 = 1.0 - (len(set(actions)) / max(n, 1))

    # SC2: step count
    sc2 = min(n / 10.0, 1.0)

    # SC3: obs-thought gap (inverted Jaccard — high = agent ignoring evidence)
    gaps = []
    for t, o in zip(thoughts, obs_list):
        if t and o:
            tt = _toks(t); ot = _toks(o)
            if tt | ot:
                gaps.append(len(tt & ot) / len(tt | ot))
    sc3 = 1.0 - (float(np.mean(gaps)) if gaps else 0.5)

    # SC5: thought verbosity
    tl = [len(t.split()) for t in thoughts if t]
    sc5 = min(sum(tl) / (50 * max(len(tl), 1)), 1.0)

    # SC6: answer–observation gap
    obs_all = " ".join(obs_list)
    if fa and obs_all:
        fa_t = _toks(fa); ob_t = _toks(obs_all)
        sc6 = 1.0 - (len(fa_t & ob_t) / max(len(fa_t), 1))
    else:
        sc6 = 0.5

    # SC11: answer–question mismatch
    qt = _toks(q); aft = _toks(fa)
    sc11 = 1.0 - (len(qt & aft) / max(len(qt | aft), 1))

    # SC9: context utilisation (n>=3)
    if n >= 3:
        used = []
        for t, o in zip(thoughts, obs_list):
            if t and o:
                tt = _toks(t); ot = _toks(o)
                used.append(len(tt & ot) / max(len(ot), 1))
        sc9 = float(np.mean(used)) if used else 0.5
    else:
        sc9 = 0.5

    # SC10: inter-step coherence (n>=3)
    if n >= 3:
        coh = []
        for i in range(len(thoughts) - 1):
            t1 = _toks(thoughts[i]); t2 = _toks(thoughts[i + 1])
            if t1 | t2:
                coh.append(len(t1 & t2) / len(t1 | t2))
        sc10 = float(np.mean(coh)) if coh else 0.5
    else:
        sc10 = 0.5

    # SC12: risk-monotone slope (n>=3)
    if n >= 3:
        step_risks = []
        for t, o in zip(thoughts, obs_list):
            if t and o:
                tt = _toks(t); ot = _toks(o)
                step_risks.append(1 - len(tt & ot) / max(len(tt | ot), 1))
        if len(step_risks) >= 3:
            slope = (step_risks[-1] - step_risks[0]) / max(n - 1, 1)
            sc12 = float(math.tanh(5 * slope) * 0.5 + 0.5)
        else:
            sc12 = 0.5
    else:
        sc12 = 0.5

    n_searches = sum(1 for a in actions if a == "Search")

    # ── 3 domain-invariant answer-side features ───────────────────────────────

    obs_all_text = " ".join(obs_list)
    obs_cap = _cap_toks(obs_all_text)

    # ans_entity_match: fraction of answer's capitalized tokens found in observations
    fa_cap = _cap_toks(fa)
    if fa_cap:
        ans_entity_match = len(fa_cap & obs_cap) / len(fa_cap)
    else:
        ans_entity_match = 0.5

    # ans_len_type_match: factoid question with suspiciously long answer
    _factoid_prefixes = ("who ", "when ", "where ", "what year", "which ", "how many", "how much")
    q_lower = q.lower().strip()
    is_factoid = any(q_lower.startswith(p) for p in _factoid_prefixes)
    ans_word_count = len(fa.split())
    ans_len_type_match = 1.0 if (is_factoid and ans_word_count > 8) else 0.0

    # obs_entity_coverage: fraction of question's capitalized tokens covered by observations
    q_cap = _cap_toks(q)
    if q_cap:
        obs_entity_coverage = len(q_cap & obs_cap) / len(q_cap)
    else:
        obs_entity_coverage = 0.5

    return np.array([
        sc1, sc2, sc3, sc5, sc6, sc11,
        sc9, sc10, sc12,
        min(n / 10.0, 1.0),              # n_steps_norm
        min(n_searches / 8.0, 1.0),      # n_searches_norm
        float(n >= 2),                   # is_multi_step
        ans_entity_match,                # ans_entity_match
        ans_len_type_match,              # ans_len_type_match
        obs_entity_coverage,             # obs_entity_coverage
    ], dtype=float)


# ── LocalVerifier ─────────────────────────────────────────────────────────────

class LocalVerifier:
    """
    LogReg classifier trained on 12 SC behavioral features.

    Replaces the Sonnet judge after training on ≥ 200 labeled agent runs.
    AUROC 0.8035 within-domain (exp111, 5-fold CV, n=200).

    Parameters
    ----------
    C : float
        Regularisation parameter for LogisticRegression. Default: 1.0.
    max_iter : int
        Max iterations for solver. Default: 1000.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self._C = C
        self._max_iter = max_iter
        self._pipeline = None
        self._fitted = False
        self._n_train = 0

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        runs: List[Dict],
        labels: Optional[List[bool]] = None,
    ) -> "LocalVerifier":
        """
        Train the LogReg classifier.

        Parameters
        ----------
        runs : list of dicts
            Each dict must have "question", "steps", "final_answer" keys.
            If labels is None, the dict must also have "correct": bool.
        labels : list of bool, optional
            Correctness labels (True = correct). If omitted, read from
            run["correct"].

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if labels is None:
            labels = [bool(r.get("correct", False)) for r in runs]

        X = np.array([
            extract_features(r["question"], r.get("steps", []), r.get("final_answer", ""))
            for r in runs
        ])
        # Label convention: 1 = WRONG (risk signal), 0 = correct
        y = np.array([0 if c else 1 for c in labels])

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self._C, max_iter=self._max_iter,
                                       random_state=42, class_weight="balanced")),
        ])
        self._pipeline.fit(X, y)
        self._fitted = True
        self._n_train = len(runs)
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_risk(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
    ) -> float:
        """
        Return a risk score in [0, 1]. Higher = more likely wrong.

        Raises RuntimeError if the verifier has not been fitted.
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError(
                "LocalVerifier is not fitted. Call fit() or load() first."
            )
        x = extract_features(question, steps, final_answer).reshape(1, -1)
        # Probability of class 1 (wrong)
        return float(self._pipeline.predict_proba(x)[0, 1])

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the fitted verifier to a pickle file."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted LocalVerifier.")
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str) -> "LocalVerifier":
        """Load a previously saved LocalVerifier from a pickle file."""
        obj = pickle.loads(Path(path).read_bytes())
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a LocalVerifier: {type(obj)}")
        return obj

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_train(self) -> int:
        return self._n_train

    def feature_importances(self) -> Dict[str, float]:
        """
        Return absolute LogReg coefficients as a proxy for feature importance.
        Requires the verifier to be fitted.
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("LocalVerifier is not fitted.")
        coefs = self._pipeline.named_steps["clf"].coef_[0]
        return {name: round(float(abs(c)), 4) for name, c in zip(FEATURE_NAMES, coefs)}

    def __repr__(self) -> str:
        status = f"fitted on {self._n_train} chains" if self._fitted else "not fitted"
        return f"LocalVerifier(C={self._C}, {status})"
