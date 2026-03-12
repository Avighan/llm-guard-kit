"""
WhiteBoxProbe — Hidden-state uncertainty probing for open-weight LLMs (exp117).

Reads internal model representations (hidden states / attention entropy) at
reasoning step k to detect failure before the chain completes.  Complements
SC_OLD (behavioral, $0) and the Sonnet judge (black-box, $0.007/chain) with a
direct signal from the model's own internal uncertainty.

EHC cross-research validation
------------------------------
  EHC Prediction 3: "Internal representations encode uncertainty — a linear
  probe on hidden states at layer L predicts chain failure with AUROC ≥ 0.75."
  WhiteBoxProbe directly tests this prediction.

Real-model validation results
------------------------------
  exp145 (Qwen2.5-1.5B generates its own chains, Apple MPS):
      HP same-distribution AUROC:  0.625
      TriviaQA cross-domain AUROC: 0.640
      Mean cross-domain:           0.633
      EHC Prediction 3:            NOT confirmed (threshold 0.65 not met)
      Note: correct chains show HIGHER hidden-state variance (inverted direction —
            model activates more discriminatively when generating correct answers).

  exp146 (Qwen2.5-3B, same setup):
      Mean cross-domain AUROC:     0.585
      EHC Prediction 3:            NOT confirmed
      Root cause: n≈28–60 training samples + low chain accuracy (23–27%) limits
      correct/wrong contrast. Production path: Llama-3-8B with n≥200 expected ≥0.75.

  probe_ensemble_blend (exp148, Qwen2.5-1.5B proxy + P(True)):
      alpha=0.25 (25% probe, 75% P(True)):  mean AUROC 0.737  ← +1.6pp over P(True) alone
      Use probe_ensemble_blend(probe_score, ptrue_score, alpha=0.25) for best result.

ECL cross-research
------------------
  Hidden-state entropy is a direct proxy for the model's energy state.
  High entropy at reasoning layers → high cognitive load → maps to ECL's
  "arousal" homeostatic drive.  Entropy collapse → consolidation signal.

TCI cross-research
------------------
  WhiteBoxProbe is the Phase 3 "direct observation" component in TCI:
    Phase 1: Behavioral (SC_OLD, $0)
    Phase 2: Black-box judge (Sonnet, $0.007)
    Phase 3: White-box probe (hidden states, $0 after model load)
    Phase 4: Multi-agent trust (mesh routing, A2A)

Probe architecture
------------------
  1. Extract token hidden states at step k from layer L (default: last 4 layers)
  2. Pool over position dimension (mean-pool over thought tokens)
  3. Compute attention entropy per head: H = -sum(p * log(p))
  4. Concatenate: [mean_hidden (d_model), attn_entropy (n_heads)] → feature vector
  5. Train logistic regression probe (sklearn) on labeled runs
  6. Predict P(wrong) from probe features

Requirements
------------
  Open-weight model: any HuggingFace causal LM
    Tested with: Llama-3-8B, Mistral-7B, Phi-3-mini, Qwen2.5-1.5B, Qwen2.5-3B
    Requires: transformers >= 4.38, torch >= 2.1
  Training data: ≥ 100 labeled agent runs (question, steps, correct: bool)

Usage
-----
    from llm_guard.white_box_probe import WhiteBoxProbe

    # Initialise (loads HF model)
    probe = WhiteBoxProbe(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="cuda",
        probe_layers=[-1, -2, -3, -4],  # last 4 layers
    )

    # Train on labeled runs
    probe.fit(labeled_runs, step_k=2)

    # Score a chain at step k
    score = probe.score_step(question, steps[:2])
    # score.hidden_risk  — P(wrong) from probe [0, 1]
    # score.attn_entropy — mean attention entropy at step k
    # score.layer_risks  — per-layer probe scores

    # Use in AgentGuard (requires AgentGuard.attach_probe())
    guard.attach_probe(probe)
    result = guard.stream_guard(question, steps_so_far)
    # result.probe_risk is now populated
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sklearn is a soft dependency (required for probe training, not for inference)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    """
    Result of WhiteBoxProbe.score_step().

    Fields
    ------
    hidden_risk : float
        P(chain is wrong) from linear probe on hidden states [0, 1].
    attn_entropy : float
        Mean attention entropy across all heads at step k [nats].
        High entropy → model is uncertain about next token.
    entropy_by_layer : list of float
        Per-layer mean attention entropy.
    layer_risks : list of float
        Per-layer logistic probe scores.
    step_k : int
        Step index this was evaluated at.
    latency_ms : float
        Time for forward pass + feature extraction.
    model_name : str
        HF model identifier.
    fallback : bool
        True when model not loaded; probe was run in simulation mode.
    """
    hidden_risk: float
    attn_entropy: float
    entropy_by_layer: List[float] = field(default_factory=list)
    layer_risks: List[float] = field(default_factory=list)
    step_k: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_risk":      round(self.hidden_risk, 4),
            "attn_entropy":     round(self.attn_entropy, 4),
            "entropy_by_layer": [round(x, 4) for x in self.entropy_by_layer],
            "layer_risks":      [round(x, 4) for x in self.layer_risks],
            "step_k":           self.step_k,
            "latency_ms":       round(self.latency_ms, 1),
            "model_name":       self.model_name,
            "fallback":         self.fallback,
        }


# ── Feature extraction utilities ──────────────────────────────────────────────

def _attn_entropy(attn_weights: "torch.Tensor") -> float:
    """
    Compute mean attention entropy across all heads and positions.

    attn_weights: [batch, heads, seq_len, seq_len]
    Returns scalar mean entropy in nats.
    """
    import torch
    # Clamp to avoid log(0)
    p    = attn_weights.clamp(min=1e-9)
    ent  = -(p * p.log()).sum(dim=-1)   # [batch, heads, seq_len]
    return float(ent.mean().item())


def _pool_hidden(hidden: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> np.ndarray:
    """
    Mean-pool hidden states over the sequence dimension.

    hidden: [batch, seq_len, d_model]
    Returns: [d_model] numpy array
    """
    import torch
    if mask is not None:
        # Mask padding tokens
        mask_f = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    else:
        pooled = hidden.mean(dim=1)
    return pooled[0].cpu().float().numpy()


def _build_step_prompt(question: str, steps: List[Dict]) -> str:
    """Format question + steps as a prompt string for the LM."""
    parts = [f"Question: {question}"]
    for i, s in enumerate(steps, 1):
        thought  = s.get("thought", "").strip()[:200]
        act_type = s.get("action_type", "")
        act_arg  = s.get("action_arg", "").strip()[:150]
        obs      = s.get("observation", "").strip()[:200]
        parts.append(f"Step {i}: {thought}\nAction: {act_type}[{act_arg}]\nResult: {obs}")
    return "\n\n".join(parts)


# ── WhiteBoxProbe ─────────────────────────────────────────────────────────────

class WhiteBoxProbe:
    """
    Linear probe on LM hidden states for chain-failure prediction.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B").
        Model must support output_hidden_states=True and output_attentions=True.
    device : str
        Torch device. Default "cpu"; use "cuda" for GPU.
    probe_layers : list of int
        Layer indices to extract (negative = from end). Default: [-1, -2, -3, -4].
    d_probe : int or None
        PCA reduction dimension before logistic probe. None = no PCA.
        Reduces memory and overfitting for large hidden sizes.
    max_seq_len : int
        Max tokens for forward pass. Default 512.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cpu",
        probe_layers: Optional[List[int]] = None,
        d_probe: Optional[int] = 64,
        max_seq_len: int = 512,
    ):
        self.model_name   = model_name
        self.device       = device
        self.probe_layers = probe_layers or [-1, -2, -3, -4]
        self.d_probe      = d_probe
        self.max_seq_len  = max_seq_len

        self._model      = None
        self._tokenizer  = None
        self._probe      = None       # LogisticRegression
        self._scaler     = None       # StandardScaler
        self._pca        = None       # PCA (optional)
        self._is_fitted  = False
        self._score_polarity = 1      # +1 normal, -1 inverted (auto-detected in fit)

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the HuggingFace model and tokenizer.

        Raises ImportError if transformers/torch is not installed.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(
                "WhiteBoxProbe requires: pip install transformers torch\n"
                f"Original error: {e}"
            )

        print(f"  Loading {self.model_name} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype="auto",
            device_map=self.device if self.device != "cpu" else None,
        )
        if self.device == "cpu" or self.device.startswith("cuda"):
            self._model = self._model.to(self.device)
        self._model.eval()
        print(f"  Model loaded: {self.model_name}")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ── Feature extraction ─────────────────────────────────────────────────────

    def extract_features(
        self,
        question: str,
        steps: List[Dict],
        step_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract probe feature vector from hidden states at step k.

        Parameters
        ----------
        question : str
        steps : list of step dicts
        step_k : int or None
            How many steps to include. None = all steps.

        Returns
        -------
        np.ndarray of shape [n_features]
            Concatenation of:
              - mean-pooled hidden states from each probe layer: [len(layers) × d_model]
              - mean attention entropy from each probe layer:    [len(layers)]
        """
        import torch

        k     = step_k if step_k is not None else len(steps)
        text  = _build_step_prompt(question, steps[:k])
        enc   = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            out = self._model(**enc)

        hidden_states = out.hidden_states   # tuple: (n_layers+1,) each [1, seq, d]
        attentions    = out.attentions       # tuple: (n_layers,) each [1, heads, seq, seq]

        n_layers = len(hidden_states) - 1   # exclude embedding layer
        features = []

        for layer_idx in self.probe_layers:
            # Resolve negative indices
            resolved  = layer_idx % (n_layers + 1)
            h_layer   = hidden_states[resolved]   # [1, seq, d_model]
            pooled    = _pool_hidden(h_layer, enc.get("attention_mask"))
            features.append(pooled)

            # Attention entropy (attentions has one fewer entry than hidden_states)
            attn_resolved = resolved % n_layers
            if attn_resolved < len(attentions):
                ent = _attn_entropy(attentions[attn_resolved])
                features.append(np.array([ent]))

        return np.concatenate(features)

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        labeled_runs: List[Dict],
        step_k: int = 2,
        C: float = 1.0,
    ) -> "WhiteBoxProbe":
        """
        Train the linear probe on labeled agent runs.

        Parameters
        ----------
        labeled_runs : list of dict
            Each dict must have: "question", "steps", "correct" (bool).
        step_k : int
            Number of steps to include in the probe input. Default 2 (same as stream_guard).
        C : float
            Logistic regression regularisation inverse. Default 1.0.

        Returns
        -------
        self (for chaining)
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("sklearn required: pip install scikit-learn")
        if not self.model_loaded:
            raise RuntimeError("Call load_model() before fit().")

        print(f"  Extracting features from {len(labeled_runs)} runs (step_k={step_k})...")
        X, y = [], []
        for run in labeled_runs:
            try:
                feat = self.extract_features(run["question"], run["steps"], step_k)
                X.append(feat)
                y.append(int(not run["correct"]))  # 1 = wrong
            except Exception:
                continue

        X = np.array(X)
        y = np.array(y)

        # PCA reduction (optional)
        if self.d_probe is not None and X.shape[1] > self.d_probe:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.d_probe)
            X         = self._pca.fit_transform(X)

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)

        self._probe = LogisticRegression(C=C, max_iter=500, random_state=42)
        self._probe.fit(X, y)
        self._is_fitted = True

        # Auto-detect polarity: if training AUROC < 0.5 the probe learned the
        # inverted direction (observed on Qwen2.5 — correct chains have HIGHER
        # hidden-state variance).  Flip so hidden_risk always means P(wrong).
        if len(np.unique(y)) >= 2:
            from sklearn.metrics import roc_auc_score as _ras
            train_proba = self._probe.predict_proba(X)[:, 1]
            train_auroc = _ras(y, train_proba)
            if train_auroc < 0.5:
                self._score_polarity = -1
                print(f"  NOTE: probe direction inverted (train AUROC={train_auroc:.3f} < 0.5) "
                      f"— flipping polarity so hidden_risk = P(wrong)")
            else:
                self._score_polarity = 1

        print(f"  Probe fitted on {len(y)} samples ({y.sum()} wrong, {(~y.astype(bool)).sum()} correct)")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def score_step(
        self,
        question: str,
        steps: List[Dict],
        step_k: Optional[int] = None,
    ) -> ProbeResult:
        """
        Score a (partial) chain at step k.

        Returns ProbeResult with hidden_risk and attn_entropy.
        Falls back to a neutral score (0.5) when model is not loaded.
        """
        t0 = time.time()
        k  = step_k if step_k is not None else len(steps)

        if not self.model_loaded:
            # Fallback: return neutral score without model
            return ProbeResult(
                hidden_risk=0.5,
                attn_entropy=float("nan"),
                step_k=k,
                latency_ms=round((time.time() - t0) * 1000, 1),
                model_name=self.model_name,
                fallback=True,
            )

        try:
            import torch

            feat = self.extract_features(question, steps, k)

            # Per-layer attention entropy from feat (every other block is entropy)
            # feat structure: [hidden_pool_L1, ent_L1, hidden_pool_L2, ent_L2, ...]
            n_layers   = len(self.probe_layers)
            d_model    = feat.shape[0] // (n_layers + 1)  # approximate
            ent_vals   = []
            for i in range(n_layers):
                start = (d_model + 1) * i + d_model
                if start < feat.shape[0]:
                    ent_vals.append(float(feat[start]))
            attn_ent = float(np.mean(ent_vals)) if ent_vals else float("nan")

            # Probe inference
            if self._is_fitted:
                X = feat.reshape(1, -1)
                if self._pca is not None:
                    X = self._pca.transform(X)
                X = self._scaler.transform(X)
                raw_prob = float(self._probe.predict_proba(X)[0, 1])
                # Apply polarity correction: if inverted direction detected in fit(),
                # _score_polarity == -1 and we flip so hidden_risk = P(wrong)
                prob = raw_prob if self._score_polarity == 1 else (1.0 - raw_prob)
            else:
                # Unfitted: use attention entropy as heuristic
                prob = min(1.0, max(0.0, attn_ent / 3.0)) if not np.isnan(attn_ent) else 0.5

            latency = (time.time() - t0) * 1000

            return ProbeResult(
                hidden_risk=round(prob, 4),
                attn_entropy=round(attn_ent, 4),
                step_k=k,
                latency_ms=round(latency, 1),
                model_name=self.model_name,
                fallback=False,
            )

        except Exception as e:
            return ProbeResult(
                hidden_risk=0.5,
                attn_entropy=float("nan"),
                step_k=k,
                latency_ms=round((time.time() - t0) * 1000, 1),
                model_name=self.model_name,
                fallback=True,
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_probe(self, path: str) -> None:
        """Save trained probe (scaler + logreg + PCA) to disk. Does not save model weights."""
        if not self._is_fitted:
            raise RuntimeError("Probe is not fitted. Call fit() first.")
        state = {
            "probe":       self._probe,
            "scaler":      self._scaler,
            "pca":         self._pca,
            "probe_layers": self.probe_layers,
            "d_probe":     self.d_probe,
            "model_name":  self.model_name,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_probe(cls, path: str, **init_kwargs) -> "WhiteBoxProbe":
        """Load a saved probe. Call load_model() separately to attach the LM."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            model_name=state["model_name"],
            probe_layers=state["probe_layers"],
            d_probe=state["d_probe"],
            **init_kwargs,
        )
        obj._probe     = state["probe"]
        obj._scaler    = state["scaler"]
        obj._pca       = state["pca"]
        obj._is_fitted = True
        return obj

    def __repr__(self) -> str:
        return (
            f"WhiteBoxProbe("
            f"model={self.model_name!r}, "
            f"layers={self.probe_layers}, "
            f"fitted={self._is_fitted}, "
            f"loaded={self.model_loaded}"
            f")"
        )


def probe_ensemble_blend(
    probe_score: float,
    ptrue_score: float,
    alpha: float = 0.25,
) -> float:
    """
    Blend WhiteBoxProbe risk with P(True) risk for best cross-domain AUROC.

    Validated (exp148, Qwen2.5-1.5B proxy on exp141 Haiku-generated chains):
        alpha=0.25 (25% probe, 75% P(True)):  mean AUROC 0.737  <- recommended
        alpha=0.00 (P(True) only):            mean AUROC 0.720
        alpha=1.00 (probe only):              mean AUROC 0.633

    Parameters
    ----------
    probe_score : float
        WhiteBoxProbe.score_step().hidden_risk  in [0, 1]
    ptrue_score : float
        P(True) risk score from AgentGuard.score_with_ptrue()  in [0, 1]
    alpha : float
        Weight for probe_score. Default 0.25 (validated optimal).

    Returns
    -------
    float : blended risk in [0, 1]
    """
    return float(alpha * probe_score + (1.0 - alpha) * ptrue_score)
