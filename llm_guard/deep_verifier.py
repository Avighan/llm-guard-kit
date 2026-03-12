"""
DeepLocalVerifier — Bootstrap-Ensemble MLP with uncertainty quantification.
LSTMRiskAccumulator — Step-level LSTM for continuous chain risk scoring.

Validated performance:
    DeepLocalVerifier alone AUROC:        0.715   (exp143, 3-domain: HP→TV→NQ)
    DeepLocalVerifier + P(True) AUROC:    0.8548  (exp138, n=37 — small sample, use with caution)
    LSTMRiskAccumulator alone AUROC:      0.545   (exp143 3-domain revalidation; exp138 was n=37 overfit)
    LSTMRiskAccumulator HP 5-fold CV:     0.7915 ± 0.038  (exp138)

Key properties:
    DeepLocalVerifier:
      - sklearn MLPClassifier bootstrap ensemble (30 models × 80% subsample)
      - Input: 7 features (beh, ptrue, se, sc3, sc6, n_steps_norm, empty_frac)
      - Outputs: risk ∈ [0,1] + uncertainty (bootstrap std) ∈ [0,1]
      - Requires: 150+ labeled source-domain chains for training
      - PyPI extra: just scikit-learn (already required)

    LSTMRiskAccumulator:
      - Tiny LSTM (hidden=32, 1 layer) over per-step features
      - Input per step: [retrieval_conf, semantic_gap, thought_len_n, empty_obs, repeat_obs, step_idx_n]
      - No judge calls — purely behavioral, $0 at inference after training
      - Requires: PyTorch (optional dep, `pip install torch`)
      - Requires: 150+ labeled source-domain chains for training

Quick start
-----------
    from llm_guard import DeepLocalVerifier, LSTMRiskAccumulator

    # DeepLocalVerifier (sklearn, no extra deps)
    verifier = DeepLocalVerifier()
    verifier.fit(labeled_runs)  # [{"question":..,"steps":..,"final_answer":..,"correct":bool}]
    risk, uncertainty = verifier.score(question, steps, final_answer, ptrue=0.4)

    # Combine with P(True) for best cross-domain AUROC:
    guard  = AgentGuard(api_key="sk-ant-...")
    ptrue  = guard.score_with_ptrue(question, steps, final_answer).risk_score
    risk, unc = verifier.score(question, steps, final_answer, ptrue=ptrue)
    # Ensemble: 0.5 * risk + 0.5 * ptrue → AUROC ~0.85

    # LSTMRiskAccumulator (requires torch)
    lstm = LSTMRiskAccumulator()
    lstm.fit(labeled_runs)
    risk = lstm.score(question, steps, final_answer)  # 0.545 cross-domain (exp143 3-domain revalidation)

    # Save / load
    verifier.save("deep_verifier.pkl")
    verifier = DeepLocalVerifier.load("deep_verifier.pkl")
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Feature helpers (shared) ──────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "up", "out", "as", "into", "through",
    "and", "or", "but", "not", "so", "if", "then", "than", "about",
    "which", "who", "what", "how", "when", "where",
}

def _toks(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)

def _behavioral_risk(steps: list, final_answer: str) -> float:
    """SC_OLD behavioral risk from step list and final answer."""
    if not steps:
        return 0.5
    obs_list  = [s.get("observation", "") for s in steps]
    th_obs    = np.mean([_jaccard(s.get("thought", ""), s.get("observation", "")) for s in steps])
    obs_fa    = np.mean([_jaccard(obs, final_answer) for obs in obs_list]) if obs_list else 0.0
    empty_obs = sum(1 for o in obs_list if len(o.split()) < 5) / max(len(obs_list), 1)
    rep       = 1.0 - len(set(obs_list)) / max(len(obs_list), 1)
    beh = 0.5 - 0.3 * th_obs - 0.3 * obs_fa + 0.3 * empty_obs + 0.2 * rep
    return float(np.clip(beh, 0.0, 1.0))

def _extract_7features(
    steps: list,
    final_answer: str,
    ptrue: float = 0.5,
    se: float = 0.0,
) -> np.ndarray:
    """
    7-dimensional feature vector:
      [beh, ptrue, se, sc3_gap, sc6_obs_ans_gap, n_steps_norm, empty_frac]
    """
    n        = len(steps)
    fa       = final_answer
    obs_list = [s.get("observation", "") for s in steps]

    beh = _behavioral_risk(steps, fa)

    # sc3: thought-obs Jaccard gap (1 - overlap = more risky)
    gaps = [_jaccard(s.get("thought", ""), s.get("observation", "")) for s in steps
            if s.get("thought") and s.get("observation")]
    sc3 = 1.0 - (float(np.mean(gaps)) if gaps else 0.5)

    # sc6: fraction of answer tokens not found in any observation
    obs_all = " ".join(obs_list)
    if fa and obs_all:
        fa_t  = _toks(fa)
        ob_t  = _toks(obs_all)
        sc6   = 1.0 - len(fa_t & ob_t) / max(len(fa_t), 1)
    else:
        sc6 = 0.5

    n_steps_norm = min(n / 10.0, 1.0)
    empty_frac   = sum(1 for o in obs_list if len(o.split()) < 5) / max(n, 1)

    return np.array([beh, ptrue, se, sc3, sc6, n_steps_norm, empty_frac], dtype=np.float32)

def _extract_step_sequence(
    steps: list,
    final_answer: str,
    max_steps: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Per-step feature matrix (max_steps, 6) + actual length.
    Features per step: [retrieval_conf, semantic_gap, thought_len_n,
                        empty_obs, repeat_obs, step_idx_n]
    # NOTE: Feature layout changed in v0.14.0. Models trained on v0.13.0 features
    # are incompatible. Retrain LSTMRiskAccumulator.fit() after upgrading.
    """
    fa       = final_answer
    obs_seen = set()
    seq      = []
    for i, s in enumerate(steps[:max_steps]):
        obs    = s.get("observation", "")
        thought = s.get("thought", "")

        # retrieval_conf: Jaccard(action_arg, observation) — did search return what was asked?
        action_arg     = s.get("action_arg", s.get("action", ""))
        retrieval_conf = _jaccard(action_arg, obs) if (action_arg and obs) else 0.0

        # semantic_gap: 1 - Jaccard(thought_i, thought_{i-1}) — reasoning change between steps
        prev_thought   = steps[i - 1].get("thought", "") if i > 0 else None
        semantic_gap   = (1.0 - _jaccard(thought, prev_thought)) if prev_thought is not None else 0.5

        th_len_n   = min(len(thought.split()) / 50.0, 1.0)
        empty_obs  = float(len(obs.split()) < 5)
        repeat_obs = float(obs in obs_seen)
        obs_seen.add(obs)
        step_idx_n = i / max(max_steps - 1, 1)
        seq.append([retrieval_conf, semantic_gap, th_len_n, empty_obs, repeat_obs, step_idx_n])

    n_real = len(seq)
    while len(seq) < max_steps:
        seq.append([0.0] * 6)
    return np.array(seq, dtype=np.float32), n_real

def _prep_run(run: dict, ptrue: float = 0.5, se: float = 0.0):
    """Extract features from a labeled run dict."""
    steps = run.get("steps", [])
    fa    = run.get("final_answer", run.get("answer", ""))
    feat  = _extract_7features(steps, fa, ptrue=ptrue, se=se)
    seq, n_real = _extract_step_sequence(steps, fa)
    label = 0 if run.get("correct", True) else 1
    return feat, seq, n_real, label


# ── DeepLocalVerifier ─────────────────────────────────────────────────────────

class DeepLocalVerifier:
    """
    Bootstrap-ensemble MLP for uncertainty-aware chain risk scoring.

    Architecture: 30 MLPClassifiers, each trained on an 80% bootstrap subsample.
    Risk   = mean predicted failure probability over the ensemble.
    Uncertainty = std of predicted probabilities (higher = model is unsure).

    Validated cross-domain AUROC:
        MLP alone:         0.715   (exp143, 3-domain: HP→TV→NQ)
        MLP + P(True) 50%: 0.8548  (exp138, n=37 — small sample, use with caution)

    Uncertainty signal: wrong chains have higher uncertainty (std 0.124 vs 0.088).
    """

    N_BOOT      = 30
    BOOT_FRAC   = 0.80
    HIDDEN      = (32, 16)
    MAX_ITER    = 500
    ALPHA       = 0.01   # L2 regularization
    VAL_FRAC    = 0.15

    REPLAY_MAX  = 2000  # max replay buffer size for partial_fit

    def __init__(self, n_boot: int = N_BOOT, random_state: int = 42) -> None:
        self.n_boot           = n_boot
        self.random_state     = random_state
        self._models: Optional[list] = None
        self._fitted          = False
        self._replay_runs: List[Dict]  = []
        self._replay_ptrue: List[float] = []
        self._replay_se: List[float]    = []

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        labeled_runs: List[Dict],
        ptrue_values: Optional[List[float]] = None,
        se_values: Optional[List[float]] = None,
    ) -> "DeepLocalVerifier":
        """
        Train the bootstrap ensemble on labeled ReAct chains.

        Parameters
        ----------
        labeled_runs : list of dicts
            Each dict must have: "steps", "final_answer" (or "answer"), "correct" (bool).
        ptrue_values : list[float], optional
            P(True) risk score for each run.  Defaults to 0.5 if not provided.
        se_values : list[float], optional
            Semantic entropy score.  Defaults to 0.0 if not provided.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        n = len(labeled_runs)
        pt_list = ptrue_values if ptrue_values is not None else [0.5] * n
        se_list = se_values    if se_values    is not None else [0.0] * n

        X = np.array([
            _extract_7features(
                r.get("steps", []),
                r.get("final_answer", r.get("answer", "")),
                ptrue=float(pt_list[i]),
                se=float(se_list[i]),
            )
            for i, r in enumerate(labeled_runs)
        ])
        y = np.array([0 if r.get("correct", True) else 1 for r in labeled_runs])

        boot_n     = int(n * self.BOOT_FRAC)
        self._models = []
        rng_global = np.random.RandomState(self.random_state)

        for seed in range(self.n_boot):
            rng   = np.random.RandomState(rng_global.randint(0, 2**31))
            idx   = rng.choice(n, size=boot_n, replace=True)
            model = Pipeline([
                ("sc",  StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=self.HIDDEN,
                    max_iter=self.MAX_ITER,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=self.VAL_FRAC,
                    n_iter_no_change=10,
                    alpha=self.ALPHA,
                ))
            ])
            model.fit(X[idx], y[idx])
            self._models.append(model)

        self._fitted = True
        return self

    def partial_fit(
        self,
        new_runs: List[Dict],
        ptrue_values: Optional[List[float]] = None,
        se_values: Optional[List[float]] = None,
    ) -> "DeepLocalVerifier":
        """
        Incrementally update the ensemble with new labeled runs (continual learning).

        Appends new_runs to an internal replay buffer (capped at REPLAY_MAX=2000),
        then re-fits all bootstrap models on the combined buffer.  Requires ≥ 20
        buffered samples; silently skips if below that threshold.

        Parameters
        ----------
        new_runs     : list of labeled run dicts (same format as fit())
        ptrue_values : P(True) scores for new_runs (defaults to 0.5)
        se_values    : semantic entropy scores for new_runs (defaults to 0.0)
        """
        n_new   = len(new_runs)
        pt_new  = ptrue_values if ptrue_values is not None else [0.5] * n_new
        se_new  = se_values    if se_values    is not None else [0.0] * n_new

        self._replay_runs  += new_runs
        self._replay_ptrue += list(pt_new)
        self._replay_se    += list(se_new)

        # Trim to cap
        if len(self._replay_runs) > self.REPLAY_MAX:
            self._replay_runs  = self._replay_runs[-self.REPLAY_MAX:]
            self._replay_ptrue = self._replay_ptrue[-self.REPLAY_MAX:]
            self._replay_se    = self._replay_se[-self.REPLAY_MAX:]

        if len(self._replay_runs) < 20:
            return self  # not enough data yet

        self.fit(self._replay_runs, self._replay_ptrue, self._replay_se)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(
        self,
        question: str,
        steps: list,
        final_answer: str,
        ptrue: float = 0.5,
        se: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Score a chain.

        Returns
        -------
        risk        : float ∈ [0, 1] — mean failure probability across ensemble
        uncertainty : float ∈ [0, 1] — bootstrap std (higher = less certain)
        """
        if not self._fitted or not self._models:
            raise RuntimeError("Call fit() before score().")

        feat = _extract_7features(steps, final_answer, ptrue=ptrue, se=se).reshape(1, -1)
        preds = np.array([m.predict_proba(feat)[0, 1] for m in self._models])
        return float(preds.mean()), float(preds.std())

    def score_run(
        self,
        run: dict,
        ptrue: float = 0.5,
        se: float = 0.0,
    ) -> Tuple[float, float]:
        """Convenience wrapper: score a run dict directly."""
        return self.score(
            run.get("question", ""),
            run.get("steps", []),
            run.get("final_answer", run.get("answer", "")),
            ptrue=ptrue,
            se=se,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the fitted ensemble to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump({"models": self._models, "n_boot": self.n_boot,
                         "random_state": self.random_state}, f)

    @classmethod
    def load(cls, path: str) -> "DeepLocalVerifier":
        """Load a previously saved DeepLocalVerifier."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(n_boot=d["n_boot"], random_state=d["random_state"])
        obj._models = d["models"]
        obj._fitted = True
        return obj


# ── LSTMRiskAccumulator ───────────────────────────────────────────────────────

class LSTMRiskAccumulator:
    """
    Step-level LSTM risk accumulator for continuous chain risk scoring.

    Processes each ReAct step as a feature vector and pools the hidden states
    to produce a single risk score.  Purely behavioral — no judge calls required.

    Architecture:
        LSTM(input=6, hidden=32, layers=1) → mean pooling over valid steps
        → Dropout(0.3) → Linear(32→1) → sigmoid

    Validated performance:
        HP 5-fold CV AUROC:            0.7915 ± 0.038  (exp138)
        Cross-domain AUROC (3-domain): 0.545  (exp143 revalidation — exp138 was n=37 overfit)
        Use DeepLocalVerifier (0.715 cross-domain) for better cross-domain performance.

    Requirements: PyTorch (`pip install torch`).
    """

    HIDDEN_SIZE = 32
    INPUT_SIZE  = 6
    MAX_STEPS   = 8
    BATCH_SIZE  = 32
    EPOCHS      = 80
    LR          = 5e-3
    DROPOUT     = 0.3
    WEIGHT_DECAY = 1e-3

    PARTIAL_LR     = 1e-4   # conservative LR for continual updates
    PARTIAL_EPOCHS = 10     # epochs per partial_fit call
    REPLAY_MAX     = 2000

    def __init__(self, random_state: int = 42) -> None:
        self.random_state  = random_state
        self._model        = None
        self._fitted       = False
        self._replay_runs: List[Dict] = []

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for LSTMRiskAccumulator. "
                "Install with: pip install torch"
            ) from e

        class _LSTMModel(nn.Module):
            def __init__(inner, hidden=self.HIDDEN_SIZE, inp=self.INPUT_SIZE, dropout=self.DROPOUT):
                super().__init__()
                inner.lstm    = nn.LSTM(inp, hidden, num_layers=1, batch_first=True)
                inner.dropout = nn.Dropout(dropout)
                inner.fc      = nn.Linear(hidden, 1)

            def forward(inner, x, lengths):
                out, _ = inner.lstm(x)  # (B, T, H)
                # Mean pooling over valid timesteps only
                mask = torch.zeros(out.shape[0], out.shape[1], 1, device=out.device)
                for i, l in enumerate(lengths):
                    mask[i, :max(int(l), 1)] = 1.0
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                return torch.sigmoid(inner.fc(inner.dropout(pooled))).squeeze(-1)

        return _LSTMModel()

    def fit(self, labeled_runs: List[Dict]) -> "LSTMRiskAccumulator":
        """
        Train the LSTM on labeled ReAct chains.

        Parameters
        ----------
        labeled_runs : list of dicts
            Each dict must have: "steps", "final_answer" (or "answer"), "correct" (bool).

        .. warning::
            LSTMRiskAccumulator is a SINGLE-DOMAIN signal only.
            Cross-domain AUROC = 0.545 (exp143 3-domain revalidation — HP/TV/NQ).
            The LSTM learns domain-specific sequential step patterns (e.g. which
            search query position predicts failure on HotpotQA) that do not transfer.
            For cross-domain use, prefer DeepLocalVerifier (0.715) or the L1
            behavioral features alone (0.77 HP→TriviaQA).
        """
        import warnings
        warnings.warn(
            "LSTMRiskAccumulator cross-domain AUROC = 0.545 (exp143). "
            "Use DeepLocalVerifier or L1 behavioral features for cross-domain scoring.",
            UserWarning,
            stacklevel=2,
        )
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        seqs, n_reals, labels = [], [], []
        for r in labeled_runs:
            steps = r.get("steps", [])
            fa    = r.get("final_answer", r.get("answer", ""))
            seq, n_real = _extract_step_sequence(steps, fa, self.MAX_STEPS)
            seqs.append(seq)
            n_reals.append(n_real)
            labels.append(0 if r.get("correct", True) else 1)

        X_t = torch.FloatTensor(np.array(seqs))
        y_t = torch.FloatTensor(labels)
        nr  = np.array(n_reals)
        n   = len(X_t)

        self._model = self._build_model()
        opt         = optim.Adam(self._model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        loss_fn     = nn.BCELoss()

        for _ in range(self.EPOCHS):
            idx = np.random.permutation(n)
            self._model.train()
            for start in range(0, n, self.BATCH_SIZE):
                b_idx = idx[start:start + self.BATCH_SIZE]
                bx, by, bl = X_t[b_idx], y_t[b_idx], nr[b_idx]
                pred = self._model(bx, bl)
                loss = loss_fn(pred, by)
                opt.zero_grad(); loss.backward(); opt.step()

        self._fitted = True
        return self

    def partial_fit(self, new_runs: List[Dict]) -> "LSTMRiskAccumulator":
        """
        Incrementally fine-tune the LSTM on new labeled runs (continual learning).

        Appends new_runs to a replay buffer (capped at REPLAY_MAX=2000), then runs
        PARTIAL_EPOCHS=10 epochs of Adam at a conservative LR=1e-4 on the combined
        buffer.  Requires PyTorch and a prior call to fit().  Silently skips if
        the model is not yet fitted or if the buffer has fewer than 20 samples.
        """
        if not self._fitted or self._model is None:
            return self  # no base model yet; defer to full fit()

        self._replay_runs += new_runs
        if len(self._replay_runs) > self.REPLAY_MAX:
            self._replay_runs = self._replay_runs[-self.REPLAY_MAX:]

        if len(self._replay_runs) < 20:
            return self

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            seqs, n_reals, labels = [], [], []
            for r in self._replay_runs:
                steps = r.get("steps", [])
                fa    = r.get("final_answer", r.get("answer", ""))
                seq, n_real = _extract_step_sequence(steps, fa, self.MAX_STEPS)
                seqs.append(seq)
                n_reals.append(n_real)
                labels.append(0 if r.get("correct", True) else 1)

            X_t = torch.FloatTensor(np.array(seqs))
            y_t = torch.FloatTensor(labels)
            nr  = np.array(n_reals)
            n   = len(X_t)

            opt     = optim.Adam(self._model.parameters(), lr=self.PARTIAL_LR)
            loss_fn = nn.BCELoss()

            for _ in range(self.PARTIAL_EPOCHS):
                idx = np.random.permutation(n)
                self._model.train()
                for start in range(0, n, self.BATCH_SIZE):
                    b_idx = idx[start:start + self.BATCH_SIZE]
                    bx, by, bl = X_t[b_idx], y_t[b_idx], nr[b_idx]
                    pred = self._model(bx, bl)
                    loss = loss_fn(pred, by)
                    opt.zero_grad(); loss.backward(); opt.step()
        except Exception:
            pass  # never break if torch unavailable or on error

        return self

    def score(
        self,
        question: str,
        steps: list,
        final_answer: str,
    ) -> float:
        """
        Score a chain using the step-level LSTM.

        Returns
        -------
        risk : float ∈ [0, 1] — failure probability
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before score().")
        import torch
        fa  = final_answer
        seq, n_real = _extract_step_sequence(steps, fa, self.MAX_STEPS)
        X   = torch.FloatTensor(seq).unsqueeze(0)  # (1, T, 6)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X, [n_real])
        return float(pred.item())

    def score_run(self, run: dict) -> float:
        """Convenience wrapper: score a run dict directly."""
        return self.score(
            run.get("question", ""),
            run.get("steps", []),
            run.get("final_answer", run.get("answer", "")),
        )

    def save(self, path: str) -> None:
        """Save the fitted LSTM to a file (requires torch.save)."""
        import torch
        torch.save({
            "state_dict": self._model.state_dict(),
            "random_state": self.random_state,
            "config": {
                "hidden_size": self.HIDDEN_SIZE,
                "input_size":  self.INPUT_SIZE,
                "max_steps":   self.MAX_STEPS,
                "dropout":     self.DROPOUT,
            }
        }, path)

    @classmethod
    def load(cls, path: str) -> "LSTMRiskAccumulator":
        """Load a previously saved LSTMRiskAccumulator."""
        import torch
        d   = torch.load(path, map_location="cpu")
        obj = cls(random_state=d["random_state"])
        obj._model = obj._build_model()
        obj._model.load_state_dict(d["state_dict"])
        obj._fitted = True
        return obj
