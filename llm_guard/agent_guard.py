"""
AgentGuard — Real-time reliability monitoring for multi-step LLM agents.
=========================================================================

Validated performance (HotpotQA within-domain / NQ cross-domain):
  Chain-level AUROC (behavioral SC_OLD only):         ~0.81 within / ~0.68 cross-TV
  Chain-level AUROC (SC_OLD + Sonnet judge):           ~0.78 within / ~0.74 cross-NQ
  Chain-level AUROC (local verifier, n=200):           ~0.80 within (exp111, $0 inference)
  Chain-level AUROC (structural verifier, n=200):      ~0.73 cross-TV (exp119, $0)
  Chain-level AUROC (P(True) Haiku alone):             ~0.74 cross-TV (exp120, ~$0.0003/call)
  Chain-level AUROC (behavioral + P(True) ensemble):   ~0.78 cross-TV (exp120, best cross-domain)
  Chain-level AUROC (ptrue weight=0.9, n=37 test):     ~0.88 (exp136, TriviaQA held-out)
  Chain-level AUROC (isotonic-calibrated ptrue):        ~0.87 (exp134, TriviaQA held-out)
  Conformal alerting (J5 at FPR ≤ 10%):               Precision=0.908, Recall=0.595
  Mid-chain judge at step 2:                           AUROC=0.683 (Δ+0.156 vs SC_OLD)

  Source: exp88 (behavioral CV), exp89 (Sonnet judge), exp92 (conformal),
          exp107 (mid-chain), exp105 (A2A trust object), exp111 (local verifier),
          exp119 (structural-only features), exp120 (P(True) cross-domain),
          exp124 (feature normalization, marginal +2pp)

Note on earlier AUROC 1.000 claim
----------------------------------
  Earlier versions of this module reported step-level AUROC 1.000 measured
  on the same data used for KNN calibration (in-sample evaluation).
  That figure does NOT generalise to unseen agent runs. Realistic estimates
  on held-out data are ~0.81 within-domain and ~0.74 cross-domain.
  See exp88 5-fold CV and exp89 for reliable figures.

Three operating modes
---------------------
  1. Behavioral only (use_judge=False, default)
     Zero API cost. AUROC ~0.81 within-domain. Good for high-throughput.

  2. Behavioral + Sonnet judge (use_judge=True)
     ~$0.007/chain. AUROC ~0.74 cross-domain. Best for cross-domain pipelines
     or when precision of the alert tier is critical.

  3. Behavioral + local verifier (use_local_verifier=True, requires 200 labels)
     Zero inference cost after training. AUROC ~0.80 within-domain (exp111).
     Beats Sonnet judge within-domain; no cross-domain validation yet.
     Requires: guard.fit_verifier(labeled_runs) before scoring.

Scoring signals used
--------------------
  SC1   Completion flag (0=Finish, 1=fallback)
  SC2   Step count (single strongest signal, AUROC ~0.88)
  SC3   Thought variance (embedding-based)
  SC5   Uncertainty word density
  SC6   Answer-thought token gap
  SC8   Backtracking rate (repeated queries)
  SC9   Observation utilisation
  SC10  Thought-observation coherence
  SC11  Answer-question mismatch
  SC12  Risk-monotone slope (coherence decay across steps)
  Judge Sonnet CoT assessment: GOOD / BORDERLINE / POOR (optional)

J5 ensemble (exp89): SC_OLD × 1 + Sonnet judge × 3, normalised to [0, 1].
  Judge risk mapping: GOOD→0.25, BORDERLINE→0.55, POOR→0.85

Confidence tiers (exp92 conformal calibration):
  HIGH   risk_score < 0.50 → proceed normally
  MEDIUM 0.50 ≤ risk < 0.70 → proceed with monitoring
  LOW    risk_score ≥ 0.70 → alert / escalate / rewrite query

A2A trust object
----------------
  generate_trust_object() emits an A2ATrustObject with the full trust envelope.
  Downstream agents import and condition their search strategy on it.
  When confidence_tier == "LOW", pass to QueryRewriter for query diversification.

Quick start
-----------
    from llm_guard import AgentGuard

    guard = AgentGuard()                              # behavioral only, $0
    # guard = AgentGuard(api_key="sk-ant-...", use_judge=True)  # + Sonnet judge

    # 1. Score a completed chain
    result = guard.score_chain(
        question="Who was older, Einstein or Bohr?",
        steps=[
            {"thought": "Search for Einstein birth year",
             "action_type": "Search", "action_arg": "Einstein birth year",
             "observation": "Albert Einstein was born on March 14, 1879..."},
            {"thought": "Einstein (1879) is older than Bohr (1885)",
             "action_type": "Finish", "action_arg": "Einstein", "observation": ""},
        ],
        final_answer="Einstein",
    )
    print(result.confidence_tier)   # HIGH / MEDIUM / LOW
    print(result.risk_score)        # 0.0-1.0, higher = more likely wrong
    print(result.needs_alert)       # True when risk >= 0.70

    # 2. Get A2A trust object (for handoff to Agent B)
    trust = guard.generate_trust_object(question, steps, final_answer)
    payload = trust.to_dict()       # JSON-serialisable for wire transport

    # 3. Mid-chain monitoring (call at each step during agent loop)
    step_result = guard.monitor_step(
        question="Who was older, Einstein or Bohr?",
        steps_so_far=[step1],
        current_action="Search[Niels Bohr birth year]",
    )
    if step_result.risk == "high":
        pass  # intervene: retry, escalate, or abort early

    # 4. Calibrate on your own correct chains (improves GMM + obs-pool signals)
    guard.fit_from_agent_runs(my_correct_runs)

    # 5. Pre-screen before running the agent
    pre = guard.score_chain_start("Who was president when X happened?")
    print(pre["predicted_outcome"])  # "likely_success" | "uncertain" | "likely_failure"

    # 6. Local verifier (replaces Sonnet judge after 200 labeled runs, $0 inference)
    guard = AgentGuard(use_local_verifier=True)
    guard.fit_verifier(labeled_runs)  # [{"question":..,"steps":..,"final_answer":..,"correct":bool}]
    result = guard.score_chain(question, steps, final_answer)
    # Uses J_LOCAL = SC_OLD×1 + LocalVerifier×3 (AUROC ~0.80 within-domain, exp111)

    # 7. Pseudo-label conformal calibration (no human labels needed)
    guard = AgentGuard(use_judge=True)
    guard.calibrate_from_agreement(unlabeled_runs_with_agent_b_answers)
    # Uses 3-agent agreement as pseudo-labels for conformal alerting (exp110)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Ensure qppg_service is importable whether running from repo root or installed
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from qppg_service.label_free_scorer import LabelFreeScorer
from llm_guard.trust_object import A2ATrustObject, MeshResult, StreamGuardResult, TemporalValidity
from llm_guard.local_verifier import LocalVerifier
from llm_guard.adapter_registry import AdapterRegistry, AdapterConfig, AdapterResult
from llm_guard.step_normalizer import normalize_steps


# ── Judge prompt (exp89 4-step CoT, Sonnet) ───────────────────────────────────

_JUDGE_SYSTEM = """\
You are a strict evaluator of AI reasoning chains. Assess whether the agent \
produced a correct, well-supported answer.

Evaluate in JSON only:
{
  "diagnosis": "one sentence: what the agent did and why it may be wrong",
  "label": "GOOD or BORDERLINE or POOR",
  "confidence": 1-5
}

GOOD      = searches are productive, reasoning is sound, answer is well-supported
BORDERLINE = some gaps or uncertainty but answer may be correct
POOR      = clear reasoning failures, answer is unsupported or likely wrong

Output ONLY the JSON object."""

_JUDGE_USER_TMPL = """\
QUESTION: {question}

AGENT CHAIN:
{chain_text}

FINAL ANSWER: {final_answer}"""

# Judge label → risk score mapping (calibrated on exp89 distribution)
_JUDGE_RISK: Dict[str, float] = {"GOOD": 0.25, "BORDERLINE": 0.55, "POOR": 0.85}

# J5 ensemble weights: SC_OLD × 1, Sonnet judge × 3
_J5_SC_WEIGHT    = 1.0
_J5_JUDGE_WEIGHT = 3.0

_HEDGE_RE = re.compile(
    r"\b(not sure|uncertain|unclear|might|may|possibly|perhaps|i think|i believe|"
    r"probably|likely|seems|appear|could be|doubt|unsure|i'm not|don't know)\b", re.I
)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AgentStepResult:
    """Result of AgentGuard.monitor_step() — per-step mid-chain scoring."""
    risk_score: float        # 0-1; higher = more likely in a failing chain
    risk: str                # "low" | "medium" | "high"
    confidence: str          # "high" | "medium" | "low"  (inverse of risk)
    predicted_outcome: str   # "likely_success" | "uncertain" | "likely_failure"
    step_index: int          # 0-based index of the current step
    failure_mode: Optional[str] = None   # detected failure mode or None


@dataclass
class ChainTrustResult:
    """
    Result of AgentGuard.score_chain() — full-chain assessment.

    When needs_alert=True, treat the chain as a likely failure.
    Expected precision at that threshold: ~0.908 (FPR ≤ 10%, exp92).
    """
    risk_score: float              # J5 composite score [0, 1]
    confidence_tier: str           # HIGH / MEDIUM / LOW
    needs_alert: bool              # True if risk >= alert_threshold (default 0.70)
    failure_mode: Optional[str]    # detected failure pattern or None
    judge_label: Optional[str]     # GOOD / BORDERLINE / POOR / None
    step_count: int                # number of Search steps
    behavioral_score: float        # SC_OLD score before judge blending
    behavioral_components: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


# ── PtrueWeightBandit ─────────────────────────────────────────────────────────

class PtrueWeightBandit:
    """
    UCB1 multi-armed bandit for adaptive ptrue_weight tuning.

    Maintains 5 arms (candidate weights: [0.5, 0.65, 0.75, 0.85, 0.9]).
    Selects the arm with the highest UCB1 upper confidence bound each call.
    Arm rewards come from human feedback: 1 - |predicted_correct - actual_correct|.

    Quick start
    -----------
        guard = AgentGuard(api_key="sk-ant-...")
        guard.enable_ptrue_bandit()
        result = guard.score_with_ptrue(question, steps, answer)
        # After feedback arrives:
        guard._ptrue_bandit.update(
            weight  = result.behavioral_components.get("ptrue_weight_used", 0.9),
            reward  = 1.0 if chain_was_correct else 0.0,
        )
    """

    ARMS = [0.50, 0.65, 0.75, 0.85, 0.90]

    def __init__(self) -> None:
        n = len(self.ARMS)
        self._counts  = np.zeros(n)    # n_k: times each arm was pulled
        self._rewards = np.zeros(n)    # sum of rewards per arm
        self._total   = 0              # total pulls

    def select(self) -> float:
        """Return the ptrue_weight for this call (UCB1 selection)."""
        n = len(self.ARMS)
        # Cold start: try each arm once before using UCB
        for i in range(n):
            if self._counts[i] == 0:
                return self.ARMS[i]
        ucb = self._rewards / self._counts + np.sqrt(
            2.0 * math.log(self._total) / self._counts
        )
        return self.ARMS[int(np.argmax(ucb))]

    def update(self, weight: float, reward: float) -> None:
        """Record a reward for the arm corresponding to weight."""
        try:
            idx = self.ARMS.index(weight)
        except ValueError:
            idx = int(np.argmin(np.abs(np.array(self.ARMS) - weight)))
        self._counts[idx]  += 1
        self._rewards[idx] += float(reward)
        self._total        += 1

    def best_weight(self) -> float:
        """Return the arm with the highest mean reward so far."""
        if self._total == 0:
            return 0.90  # default before any data
        means = np.where(self._counts > 0, self._rewards / self._counts, 0.0)
        return self.ARMS[int(np.argmax(means))]

    def stats(self) -> dict:
        """Return per-arm pull counts and mean rewards for inspection."""
        return {
            w: {"n": int(self._counts[i]),
                "mean_reward": round(float(self._rewards[i] / max(self._counts[i], 1)), 3)}
            for i, w in enumerate(self.ARMS)
        }


# ── AgentGuard ────────────────────────────────────────────────────────────────

class AgentGuard:
    """
    Real-time reliability monitor for multi-step LLM agents.

    Uses validated behavioral signals (SC_OLD, exp88) as the zero-cost baseline,
    optionally enhanced with a Sonnet judge (exp89) for cross-domain accuracy.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Required only when use_judge=True.
        Falls back to ANTHROPIC_API_KEY env var.
    use_judge : bool
        If True, call Sonnet after each completed chain (~$0.007/chain).
        Improves cross-domain AUROC from ~0.67 to ~0.74.
        Default False (behavioral only, $0 cost).
    use_local_verifier : bool
        If True, use a trained LocalVerifier instead of the Sonnet judge.
        $0 inference cost. AUROC ~0.80 within-domain (exp111).
        Requires calling fit_verifier() before scoring.
        Cannot be combined with use_judge=True (local verifier takes precedence).
    judge_model : str
        Model for the judge. Default: claude-sonnet-4-6 (exp89 validated).
    alert_threshold : float
        Risk score above which needs_alert=True. Default 0.70.
        At this threshold: Precision=0.908, Recall=0.595 (exp92 conformal).
    review_threshold : float
        Threshold passed to LabelFreeScorer for needs_review flag. Default 0.65.
    on_alert : callable, optional
        Called with ChainTrustResult whenever needs_alert=True.
        Use for webhook dispatch, Slack notifications, or routing to human review.
        Example: on_alert=lambda r: requests.post(webhook_url, json=r.__dict__)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_judge: bool = False,
        use_local_verifier: bool = False,
        judge_model: str = "claude-sonnet-4-6",
        alert_threshold: float = 0.70,
        review_threshold: float = 0.65,
        on_alert: Optional[Callable[["ChainTrustResult"], None]] = None,
        agent_format: str = "auto",
        nim_api_key: Optional[str] = None,
        nim_base_url: str = "https://integrate.api.nvidia.com/v1",
        nim_judge_model: str = "meta/llama-3.3-70b-instruct",
        nim_ptrue_model: str = "meta/llama-3.1-8b-instruct",
        contribute_labels: bool = False,
        telemetry_token: Optional[str] = None,
        telemetry_repo: str = "amajumder/llm-guard-labels",
    ):
        self._api_key             = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._nim_api_key         = nim_api_key or os.environ.get("NIM_API_KEY", "")
        self._nim_base_url        = nim_base_url
        self._nim_judge_model     = nim_judge_model
        self._nim_ptrue_model     = nim_ptrue_model
        self._use_judge           = use_judge
        self._use_local_verifier  = use_local_verifier
        self._judge_model         = judge_model
        self._alert_threshold     = alert_threshold
        self._on_alert            = on_alert
        self._agent_format        = agent_format   # "auto"|"react"|"openai"|"langgraph"|"autogen"|"langchain"
        self._scorer              = LabelFreeScorer(review_threshold=review_threshold)
        self._local_verifier: Optional[LocalVerifier] = None
        self._structural_verifier: Optional[object] = None  # sklearn Pipeline, exp119/124
        self._use_structural_verifier: bool = False
        self._structural_feat_mu: Optional[list] = None   # target-domain normalization stats
        self._structural_feat_std: Optional[list] = None
        self._iso_calibrator: Optional[object] = None    # IsotonicRegression, exp134
        self._ptrue_bandit: Optional["PtrueWeightBandit"] = None   # UCB1 adaptive weight
        self._iso_buffer: list = []        # (score, label) pairs for online isotonic update
        self._iso_buffer_max   = 500       # cap to keep memory bounded
        self._iso_refit_every  = 10        # refit after this many new samples
        self._iso_buffer_last_n = 0        # buffer length at last refit
        self._client              = None
        self._nim_client          = None
        self._fitted              = False
        # Opt-in telemetry (contribute_labels=True)
        self._contribute_labels = contribute_labels
        self._telemetry_token   = telemetry_token
        self._telemetry_repo    = telemetry_repo
        self._telemetry: Optional["TelemetryClient"] = None  # type: ignore[name-defined]
        if contribute_labels and telemetry_token:
            from llm_guard.telemetry import TelemetryClient
            self._telemetry = TelemetryClient(telemetry_token, telemetry_repo)
        # Storage for last scored chain (used by telemetry on feedback)
        self._last_chain: Optional[dict] = None
        self._last_domain: str = ""

    @property
    def _has_anthropic(self) -> bool:
        return bool(self._api_key)

    @property
    def _has_nim(self) -> bool:
        return bool(self._nim_api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _get_nim_client(self):
        if self._nim_client is None:
            from openai import OpenAI
            self._nim_client = OpenAI(
                api_key=self._nim_api_key,
                base_url=self._nim_base_url,
            )
        return self._nim_client

    def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 150,
        temperature: float = 0.0,
        model_override: Optional[str] = None,
        prefer_nim: bool = False,
    ) -> Optional[str]:
        """
        Unified LLM call. Uses NVIDIA NIM if prefer_nim=True and NIM key is set,
        otherwise falls back to Anthropic. Returns the raw text response or None.

        Priority:
          - prefer_nim=True + NIM key present → NIM (OpenAI-compatible)
          - prefer_nim=False + Anthropic key present → Anthropic
          - If preferred backend unavailable → try the other
          - Both unavailable → return None
        """
        # Determine backend
        use_nim = (prefer_nim and self._has_nim) or (not self._has_anthropic and self._has_nim)
        use_anthropic = not use_nim and self._has_anthropic

        if use_nim:
            model = model_override or self._nim_judge_model
            try:
                resp = self._get_nim_client().chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if self._has_anthropic:
                    use_anthropic = True  # fallback
                else:
                    return None

        if use_anthropic:
            model = model_override or self._judge_model
            try:
                resp = self._get_client().messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text.strip()
            except Exception:
                return None

        return None

    # ── Calibration ────────────────────────────────────────────────────────────

    def fit_from_agent_runs(
        self,
        runs: List[Dict],
        correct_only: bool = True,
    ) -> "AgentGuard":
        """
        Calibrate the internal LabelFreeScorer on completed agent runs.

        Feeds chains into the GMM + obs-pool calibration (exp60).
        Requires ≥ 20 chains for density signals to activate.
        Below that threshold, behavioral SC_OLD already provides AUROC ~0.81.

        Parameters
        ----------
        runs : list of dicts
            Each dict must have "question", "steps", "final_answer" keys.
            Optional: "chain_correct": bool for filtering.
        correct_only : bool
            If True, use only runs where chain_correct=True (or key is absent).
        """
        chains = []
        for run in runs:
            if correct_only and not run.get("chain_correct", True):
                continue
            chains.append({
                "question":     run["question"],
                "steps":        run.get("steps", []),
                "final_answer": run.get("final_answer", ""),
            })
        if len(chains) >= 5:
            self._scorer.calibrate(chains)
            self._fitted = True
        return self

    def fit_verifier(
        self,
        runs: List[Dict],
    ) -> "AgentGuard":
        """
        Train the local verifier on labeled agent runs (exp111).

        Trains a LogisticRegression on 12 Jaccard behavioral features.
        After training, use_local_verifier=True routes score_chain() through
        J_LOCAL = SC_OLD×1 + LocalVerifier×3 (AUROC ~0.80 within-domain).

        Requires ≥ 50 labeled chains (recommended: ≥ 200 for stable CV scores).
        Automatically activates use_local_verifier after fitting.

        Parameters
        ----------
        runs : list of dicts
            Each dict must have "question", "steps", "final_answer", "correct" keys.
            "correct" (bool): True if the chain produced the right answer.
        """
        if len(runs) < 50:
            import warnings
            warnings.warn(
                f"fit_verifier called with only {len(runs)} runs. "
                "Recommend ≥ 200 for stable AUROC estimates (exp111).",
                UserWarning,
                stacklevel=2,
            )
        verifier = LocalVerifier()
        verifier.fit(runs)
        self._local_verifier = verifier
        self._use_local_verifier = True
        return self

    def fit_structural_verifier(
        self,
        runs: List[Dict],
        target_unlabeled_runs: Optional[List[Dict]] = None,
    ) -> "AgentGuard":
        """
        Train a structural-only LogReg for cross-domain deployment (exp119/exp124).

        Uses only SC2/SC3/SC5 — structural features that measure HOW the agent
        reasons, not WHAT it reasons about. These generalise across domains
        (HP → TriviaQA cross-domain AUROC 0.729 vs 0.682 for full behavioral score).

        Content features SC11/SC12 cause a 46pp domain gap when trained on HP
        and tested on TriviaQA — they are excluded here.

        Parameters
        ----------
        runs : list of dicts
            Labeled source-domain runs: "question", "steps", "final_answer", "correct".
            Recommend ≥ 200 labeled chains from the source domain.
        target_unlabeled_runs : list of dicts, optional
            UNLABELED target-domain chains (no "correct" key needed).
            When provided, features are normalized to the target domain's
            per-feature statistics before training (exp124: +2pp AUROC, marginal).
            Labels are NOT used — only mean/std of behavioral features.

        Validated performance (HP-train → TriviaQA-test):
            Without target normalization (exp119): AUROC 0.729 cross-domain
            With target normalization    (exp124): AUROC 0.750 cross-domain (marginal +0.021)
            SC_OLD behavioral (no training)      : AUROC 0.682 cross-domain
            All-12-feature LogReg cross-domain   : AUROC 0.541 (overfits content features)

        Note on exp124 normalization: improvement is +0.021 on n_incorrect=8 — borderline.
        Use target normalization when you have ≥ 20 unlabeled target chains available.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError("sklearn required for fit_structural_verifier: pip install scikit-learn") from e

        if len(runs) < 20:
            import warnings
            warnings.warn(
                f"fit_structural_verifier called with only {len(runs)} runs. "
                "Recommend ≥ 200 for stable cross-domain AUROC (exp119).",
                UserWarning, stacklevel=2,
            )

        import numpy as np

        def _score_structural(chain_list):
            feats, labels = [], []
            for run in chain_list:
                try:
                    r  = self.score_chain(run["question"], run["steps"], run["final_answer"])
                    bc = r.behavioral_components
                    feats.append([bc.get("sc2", 0.0), bc.get("sc3", 0.0), bc.get("sc5", 0.0)])
                    if "correct" in run:
                        labels.append(int(not run["correct"]))
                except Exception:
                    pass
            return np.array(feats) if feats else np.zeros((0, 3)), labels

        X_src, y_list = _score_structural(runs)
        if len(y_list) < 10:
            raise ValueError(f"Only {len(y_list)} chains could be scored. Need ≥ 10.")
        y = np.array(y_list)
        if len(set(y)) < 2:
            raise ValueError("All runs have the same label. Need both correct and incorrect chains.")

        # Optional: normalize source features to target domain scale (exp124)
        self._structural_feat_mu  = None
        self._structural_feat_std = None
        if target_unlabeled_runs and len(target_unlabeled_runs) >= 10:
            X_tgt, _ = _score_structural(target_unlabeled_runs)
            if len(X_tgt) >= 10:
                mu  = np.mean(X_tgt, axis=0)
                std = np.where(np.std(X_tgt, axis=0) < 1e-8, 1.0, np.std(X_tgt, axis=0))
                X_src = (X_src - mu) / std
                self._structural_feat_mu  = mu.tolist()
                self._structural_feat_std = std.tolist()

        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        clf.fit(X_src, y)
        self._structural_verifier = clf
        self._use_structural_verifier = True
        return self

    def enable_ptrue_bandit(self) -> "AgentGuard":
        """
        Enable UCB1 adaptive ptrue_weight tuning.

        After calling this, score_with_ptrue() uses the bandit to select the
        best-performing ptrue_weight from [0.5, 0.65, 0.75, 0.85, 0.9].
        The selected weight is stored in behavioral_components["ptrue_weight_used"].

        To close the feedback loop, call after each human label arrives:
            guard._ptrue_bandit.update(weight, reward)
        where reward = 1.0 if the chain was scored correctly, else 0.0.
        """
        self._ptrue_bandit = PtrueWeightBandit()
        return self

    def calibrate_from_agreement(
        self,
        runs: List[Dict],
        agreement_threshold: float = 0.5,
        alpha: float = 0.10,
    ) -> "AgentGuard":
        """
        Pseudo-label conformal calibration using 3-agent agreement (exp110).

        Generates pseudo-labels from multi-agent agreement (no human annotation),
        then computes a conformal alerting threshold at the target FPR (alpha).

        Validated precision gap vs true labels: ~6.1% (marginal).
        Recall is substantially lower than true-label conformal (0.207 vs 0.552).
        Use with caveat — best for cold-start deployment before labels accumulate.

        Parameters
        ----------
        runs : list of dicts
            Each dict must have "question", "steps", "final_answer" keys.
            Also needs "agent_b_answer" and "agent_c_answer" (str) for agreement.
        agreement_threshold : float
            Mean pairwise token-F1 >= this → pseudo-label as correct. Default 0.5.
        alpha : float
            Target FPR for conformal coverage (1 - alpha). Default 0.10 (10% FPR).
        """
        import re as _re

        def _toks_local(text: str):
            w = _re.findall(r"[a-zA-Z]+", text.lower())
            return {x for x in w if len(x) > 1}

        def _f1_local(a: str, b: str) -> float:
            p, r = _toks_local(a), _toks_local(b)
            if not p or not r:
                return 0.0
            c = p & r
            pr = len(c) / len(p)
            rc = len(c) / len(r)
            return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

        pseudo_correct_risks = []
        for run in runs:
            fa  = run.get("final_answer", "")
            ans_b = run.get("agent_b_answer", "")
            ans_c = run.get("agent_c_answer", "")
            if not (fa and ans_b and ans_c):
                continue
            f1_ab = _f1_local(fa, ans_b)
            f1_ac = _f1_local(fa, ans_c)
            f1_bc = _f1_local(ans_b, ans_c)
            mean_f1 = (f1_ab + f1_ac + f1_bc) / 3.0
            if mean_f1 >= agreement_threshold:
                lf_result = self._scorer.score(
                    run["question"], run.get("steps", []), fa
                )
                pseudo_correct_risks.append(float(lf_result.risk_score))

        if len(pseudo_correct_risks) < 10:
            return self  # not enough data; keep existing threshold

        risks_arr = np.array(pseudo_correct_risks)
        n = len(risks_arr)
        adjusted_q = min(1.0, (1 + 1.0 / n) * (1 - alpha))
        new_threshold = float(np.quantile(risks_arr, adjusted_q))
        self._alert_threshold = new_threshold
        return self

    # ── Chain scoring ──────────────────────────────────────────────────────────

    def score_chain(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
        finished: bool = True,
        agent_format: Optional[str] = None,
    ) -> ChainTrustResult:
        """
        Score a completed reasoning chain.

        Runs SC_OLD behavioral signals (always) and optionally the Sonnet judge
        (when use_judge=True). Combines as J5 = SC_OLD×1 + judge×3.

        Parameters
        ----------
        question : str
        steps : list of dicts
            ReAct dicts OR any supported agent format (openai, langgraph, autogen,
            langchain). If agent_format is not "react", steps are normalised
            automatically and a UserWarning is emitted.
        final_answer : str
        finished : bool
            True if the chain reached a Finish action.
        agent_format : str, optional
            Override the instance-level agent_format for this call.
            One of: "auto", "react", "openai", "langgraph", "autogen", "langchain".

        Returns
        -------
        ChainTrustResult
        """
        t0 = time.time()

        # 0. Normalise steps from any format → canonical ReAct dicts
        fmt = agent_format or self._agent_format
        steps = normalize_steps(steps, agent_format=fmt, warn=True)

        # 1. Behavioral SC_OLD (always runs, $0)
        lf_result    = self._scorer.score(question, steps, final_answer, finished)
        sc_old_risk  = float(lf_result.risk_score)
        failure_mode = self._detect_failure_mode(steps, final_answer) or lf_result.failure_mode
        components   = dict(lf_result.components) if hasattr(lf_result, "components") else {}
        step_count   = sum(1 for s in steps if s.get("action_type") == "Search")

        # 2. Optional Sonnet judge (~$0.007/chain) or local verifier ($0)
        judge_label = None
        judge_risk  = None

        if self._use_structural_verifier and self._structural_verifier is not None:
            # Structural-only: SC2/SC3/SC5 LogReg (exp119/exp124, AUROC ~0.729-0.750 cross-domain)
            import numpy as _np
            bc = components
            fv = _np.array([[bc.get("sc2", 0.0), bc.get("sc3", 0.0), bc.get("sc5", 0.0)]])
            # Apply target normalization if fit with target_unlabeled_runs (exp124)
            if self._structural_feat_mu is not None:
                mu  = _np.array(self._structural_feat_mu)
                std = _np.array(self._structural_feat_std)
                fv  = (fv - mu) / std
            judge_risk  = float(self._structural_verifier.predict_proba(fv)[0, 1])
            judge_label = "STRUCTURAL"
        elif self._use_local_verifier and self._local_verifier is not None and self._local_verifier.is_fitted:
            # J_LOCAL: SC_OLD×1 + LocalVerifier×3 (exp111, AUROC ~0.80 within-domain)
            judge_risk  = self._local_verifier.predict_risk(question, steps, final_answer)
            judge_label = "LOCAL"
        elif self._use_judge and self._api_key:
            # J5: SC_OLD×1 + Sonnet judge×3 (exp89, AUROC ~0.78 within / ~0.74 cross)
            judge_label, judge_risk = self._run_judge(question, steps, final_answer)

        # 3. Ensemble: SC_OLD×1 + secondary signal×3 (normalised)
        if judge_risk is not None:
            risk_score = (
                _J5_SC_WEIGHT * sc_old_risk + _J5_JUDGE_WEIGHT * judge_risk
            ) / (_J5_SC_WEIGHT + _J5_JUDGE_WEIGHT)
        else:
            risk_score = sc_old_risk

        confidence_tier = _risk_to_tier(risk_score)
        needs_alert     = risk_score >= self._alert_threshold
        latency         = (time.time() - t0) * 1000

        result = ChainTrustResult(
            risk_score=round(float(risk_score), 4),
            confidence_tier=confidence_tier,
            needs_alert=needs_alert,
            failure_mode=failure_mode,
            judge_label=judge_label,
            step_count=step_count,
            behavioral_score=round(sc_old_risk, 4),
            behavioral_components=components,
            latency_ms=round(latency, 1),
        )

        if needs_alert and self._on_alert is not None:
            try:
                self._on_alert(result)
            except Exception:
                pass  # never let the alert callback break the scoring path

        # Store last chain for telemetry submission on subsequent feedback call
        self._last_chain = {"question": question, "steps": steps, "final_answer": final_answer}
        self._last_domain = getattr(self, '_domain', '')

        return result

    def score_with_ptrue(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
        ptrue_weight: float = 0.9,
        finished: bool = True,
    ) -> "ChainTrustResult":
        """
        Score chain using behavioral + P(True) Haiku ensemble (exp120, exp134-136).

        P(True): prompt Haiku to rate answer correctness 1-5. Domain-agnostic
        because it uses the model's own uncertainty signal rather than learned
        structural features.

        Validated performance (HP-train → real TriviaQA test):
            Behavioral only              : AUROC 0.682 (exp120)
            P(True) Haiku alone          : AUROC 0.739 (exp120)
            Ensemble 50/50               : AUROC 0.775 (exp120)
            Optimal weight (ptrue=0.9)   : AUROC 0.876 (exp136, n=37 test)
            Isotonic-calibrated ptrue    : AUROC 0.871 (exp134, n=37 test)

        Default ptrue_weight changed to 0.9 (exp136 grid-search optimal).
        Call calibrate_isotonic() to enable post-hoc score calibration (exp134).

        Cost: ~$0.0003/call (Haiku).

        Parameters
        ----------
        question, steps, final_answer : chain components
        ptrue_weight : float
            Weight of P(True) in the ensemble. Default 0.9 (exp136 validated).
            Range [0, 1]. 0 = behavioral only, 1 = P(True) only.
        finished : bool
            Whether the chain reached a Finish action.

        Returns
        -------
        ChainTrustResult with risk_score = blend of behavioral + P(True).
        """
        import re as _re

        # Use bandit-selected weight if bandit is enabled; otherwise use arg default
        if self._ptrue_bandit is not None:
            ptrue_weight = self._ptrue_bandit.select()

        base = self.score_chain(question, steps, final_answer, finished)
        if not self._has_anthropic and not self._has_nim:
            return base  # no API key, return behavioral only

        prompt = (
            f"Question: {question}\n\n"
            f"Answer: {final_answer}\n\n"
            f"Rate how likely this answer is correct on a scale of 1 to 5:\n"
            f"1=definitely wrong, 2=probably wrong, 3=uncertain, "
            f"4=probably correct, 5=definitely correct.\n\n"
            f"Reply with only the single digit (1, 2, 3, 4, or 5)."
        )
        try:
            # Use cheaper/faster model for P(True): Haiku (Anthropic) or llama-3.1-8b (NIM)
            ptrue_model = (
                self._nim_ptrue_model if (self._has_nim and not self._has_anthropic)
                else "claude-haiku-4-5-20251001"
            )
            text = self._call_llm(
                system="You are a helpful assistant that rates answer quality.",
                user=prompt,
                max_tokens=5,
                temperature=0.0,
                model_override=ptrue_model,
                prefer_nim=(self._has_nim and not self._has_anthropic),
            )
            if text is None:
                return base
            m = _re.search(r"[1-5]", text)
            if m:
                ptrue_risk = round(1.0 - (int(m.group()) - 1) / 4.0, 4)
                # Apply isotonic calibration if fitted (exp134: +3.8pp AUROC)
                if self._iso_calibrator is not None:
                    try:
                        ptrue_risk = float(self._iso_calibrator.predict(
                            np.array([ptrue_risk])
                        )[0])
                    except Exception:
                        pass
                beh_weight = 1.0 - ptrue_weight
                blended = beh_weight * base.risk_score + ptrue_weight * ptrue_risk
                from dataclasses import replace as _dc_replace
                result = _dc_replace(
                    base,
                    risk_score=round(float(blended), 4),
                    confidence_tier=_risk_to_tier(blended),
                    needs_alert=blended >= self._alert_threshold,
                )
                result.behavioral_components["ptrue_risk"]        = ptrue_risk
                result.behavioral_components["ptrue_weight_used"] = ptrue_weight
                return result
        except Exception:
            pass
        return base  # fallback to behavioral if API call fails

    def calibrate_isotonic(
        self,
        cal_scores: List[float],
        cal_labels: List[int],
    ) -> "AgentGuard":
        """
        Fit IsotonicRegression post-hoc calibration on ptrue risk scores (exp134).

        After fitting, score_with_ptrue() converts discrete 5-level ptrue scores
        (0.0/0.25/0.5/0.75/1.0) to continuous calibrated probabilities.
        Validated improvement: AUROC 0.833 → 0.871 on TriviaQA (n=37 test, exp134).

        Parameters
        ----------
        cal_scores : list of float
            Raw ptrue_risk values from calibration chains.
        cal_labels : list of int
            Ground truth: 1 = incorrect chain, 0 = correct chain.

        Returns
        -------
        self  (for method chaining)

        Example
        -------
            guard.calibrate_isotonic(cal_ptrue_scores, cal_labels)
            result = guard.score_with_ptrue(q, steps, answer)
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(np.array(cal_scores), np.array(cal_labels, dtype=float))
            self._iso_calibrator = ir
        except ImportError:
            pass  # sklearn not installed; calibration skipped silently
        return self

    def update_isotonic(self, score: float, label: int) -> "AgentGuard":
        """
        Online isotonic calibration update — add one (score, label) pair from
        production feedback and re-fit when enough new samples have accumulated.

        Call this from your feedback endpoint after receiving a human correctness
        label for a scored chain.  Isotonic regression re-fits automatically every
        _iso_refit_every=10 new samples, keeping calibration current.

        Parameters
        ----------
        score : float
            Raw risk score returned by score_chain() or score_with_ptrue().
        label : int
            1 = chain was incorrect (should alert), 0 = chain was correct.
        """
        self._iso_buffer.append((float(score), int(label)))
        # Trim to cap
        if len(self._iso_buffer) > self._iso_buffer_max:
            self._iso_buffer = self._iso_buffer[-self._iso_buffer_max:]
        # Re-fit when _iso_refit_every new samples have been added
        n_new = len(self._iso_buffer) - self._iso_buffer_last_n
        if n_new >= self._iso_refit_every and len(self._iso_buffer) >= 10:
            scores_arr = [s for s, _ in self._iso_buffer]
            labels_arr = [l for _, l in self._iso_buffer]
            self.calibrate_isotonic(scores_arr, labels_arr)
            self._iso_buffer_last_n = len(self._iso_buffer)

        # Opt-in telemetry: submit anonymized features + label if enabled
        if self._telemetry is not None:
            try:
                from llm_guard.mini_judge import _extract_features
                if self._last_chain is not None:
                    feats = _extract_features(self._last_chain).tolist()
                    was_correct = (int(label) == 0)
                    telem_label = 0 if was_correct else 1
                    from llm_guard import __version__ as _ver
                    self._telemetry.submit(
                        feats,
                        telem_label,
                        domain=self._last_domain,
                        version=_ver,
                    )
            except Exception:
                pass  # telemetry must never break the feedback path

        return self

    @staticmethod
    def conformal_alert_threshold(
        cal_scores: List[float],
        cal_labels: List[int],
        alpha: float = 0.15,
    ) -> float:
        """
        Compute a conformal alerting threshold with finite-sample precision guarantee
        (Conformal Risk Control, Bates et al. 2021; exp135).

        Returns threshold t such that the empirical miscoverage rate on the
        calibration set is at most alpha.  Set guard.alert_threshold = t to
        deploy with a quantified false-negative bound.

        Validated on TriviaQA (exp135): alpha=0.15 → Precision=0.50, Recall=0.33.

        Parameters
        ----------
        cal_scores : list of float
            Risk scores on calibration chains.
        cal_labels : list of int
            Ground truth: 1 = incorrect (should be alerted), 0 = correct.
        alpha : float
            Target miscoverage rate. Default 0.15 (at most 15% of wrong chains missed).

        Returns
        -------
        float : threshold t. Pass as AgentGuard(alert_threshold=t).

        Example
        -------
            t = AgentGuard.conformal_alert_threshold(cal_scores, cal_labels, alpha=0.15)
            guard = AgentGuard(alert_threshold=t)
        """
        wrong_scores = np.array([s for s, y in zip(cal_scores, cal_labels) if y == 1])
        if len(wrong_scores) == 0:
            return 0.5
        n = len(wrong_scores)
        idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        idx = max(0, min(idx, n - 1))
        return float(np.sort(wrong_scores)[idx])

    @staticmethod
    def kalman_smooth_risks(
        step_risks: List[float],
        Q: float = 0.2,
        R: float = 0.05,
    ) -> float:
        """
        Apply a 1D Kalman filter to smooth per-step risk scores (exp137, novel).

        Treats step-level risk observations as noisy measurements of the chain's
        latent reliability. Kalman smoothing reduces measurement noise and gives
        a more stable final risk estimate than the raw last-step or mean.

        Q=0.2, R=0.05 are the grid-search optimal parameters validated on
        TriviaQA (exp137).  Kalman+ptrue ensemble AUROC: 0.828.

        Parameters
        ----------
        step_risks : list of float
            Per-step risk scores (e.g., from monitor_step() or Jaccard distances).
        Q : float
            Process noise variance. Higher → faster adaptation to risk changes.
        R : float
            Measurement noise variance. Higher → smoother output.

        Returns
        -------
        float : Kalman-filtered final risk estimate in [0, 1].

        Example
        -------
            step_scores = [guard.monitor_step(q, steps[:k]).risk_score for k in range(1, n+1)]
            kalman_risk = AgentGuard.kalman_smooth_risks(step_scores)
        """
        if not step_risks:
            return 0.5
        x = float(step_risks[0])
        P = 1.0
        for z in step_risks[1:]:
            P = P + Q
            K = P / (P + R)
            x = x + K * (float(z) - x)
            P = (1.0 - K) * P
        return float(np.clip(x, 0.0, 1.0))

    # ── A2A trust object ───────────────────────────────────────────────────────

    def generate_trust_object(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
        finished: bool = True,
    ) -> A2ATrustObject:
        """
        Produce an A2ATrustObject for agent-to-agent handoff (exp105).

        The trust object is the standardised confidence envelope that downstream
        agents consume to decide whether to proceed, verify, or rewrite the query.

        Serialise with trust.to_dict() for JSON/queue transport.
        Deserialise with A2ATrustObject.from_dict(payload).

        Returns
        -------
        A2ATrustObject
        """
        result = self.score_chain(question, steps, final_answer, finished)

        downstream_hint = _build_downstream_hint(
            result.confidence_tier, result.judge_label, result.failure_mode
        )

        return A2ATrustObject(
            answer=final_answer,
            risk_score=result.risk_score,
            confidence_tier=result.confidence_tier,
            failure_mode=result.failure_mode,
            step_count=result.step_count,
            judge_label=result.judge_label,
            downstream_hint=downstream_hint,
            should_rewrite=result.confidence_tier == "LOW",
            behavioral_components=result.behavioral_components,
            temporal_validity=None,
        )

    # ── Mid-chain monitoring ───────────────────────────────────────────────────

    def monitor_step(
        self,
        question: str,
        steps_so_far: List[Dict],
        current_action: str,
    ) -> AgentStepResult:
        """
        Score a single agent step in real time (mid-chain intervention, exp107).

        Call BEFORE executing each action in your agent loop. High risk at step 2
        is a strong early-failure signal (AUROC=0.683, Δ+0.156 vs SC_OLD at step 2).

        Parameters
        ----------
        question : str
        steps_so_far : list of dicts
            Steps already completed (empty list = first step).
        current_action : str
            The action the agent is about to take.

        Returns
        -------
        AgentStepResult
        """
        t = len(steps_so_far)

        if t == 0:
            return AgentStepResult(
                risk_score=0.5, risk="medium", confidence="medium",
                predicted_outcome="uncertain", step_index=0,
            )

        lf_result  = self._scorer.score_prefix(question, steps_so_far, t)
        risk_score = float(lf_result.risk_score)
        risk, confidence, predicted = _risk_to_labels(risk_score)

        return AgentStepResult(
            risk_score=round(risk_score, 4),
            risk=risk,
            confidence=confidence,
            predicted_outcome=predicted,
            step_index=t,
            failure_mode=lf_result.failure_mode,
        )

    # ── Mid-chain stream guard (exp113) ───────────────────────────────────────

    def stream_guard(
        self,
        question: str,
        steps_so_far: List[Dict],
        abort_threshold: float = 0.65,
        step_for_judge: int = 2,
        rewrite_on_abort: bool = True,
    ) -> StreamGuardResult:
        """
        Real-time abort decision after step N of the agent loop (exp113).

        Call this after step `step_for_judge` (default: 2) has completed.
        When abort=True, stop the current chain and use result.rewritten_queries
        to restart with a diversified query.

        This enables the two-stage OR strategy (exp107/exp113):
          Stage 1: stream_guard() at step 2  →  early abort + rewrite
          Stage 2: score_chain() at finish   →  J5 final alert
          Combined Recall = 0.416, Precision = 0.740 (vs J5-only: R=0.595, P=0.908)

        Performance of stream_guard() alone at step 2 (exp107 baseline):
          SC_OLD prefix  AUROC = 0.527
          Haiku judge    AUROC = 0.683   ← Haiku dominates; use when api_key set
          Combined       AUROC ≈ 0.683 (Haiku-weighted)

        Cost: ~$0.001/chain for Haiku judge call at step 2.
              $0 if api_key is not set (behavioral SC_OLD prefix only, AUROC ≈ 0.527).

        Parameters
        ----------
        question : str
            The original question posed to the agent.
        steps_so_far : list of dicts
            Steps completed so far.  Should contain at least `step_for_judge` steps.
            If fewer steps are available the method scores what it has.
        abort_threshold : float
            Combined risk above which abort=True.  Default 0.65 (lower than the
            chain-level 0.70 because mid-chain false positives are cheaper to absorb
            than letting a bad chain run to completion).
        step_for_judge : int
            Which step boundary to evaluate at.  Default 2 (strongest signal per
            exp107: AUROC 0.683 at step 2 vs 0.691 at step 1).
        rewrite_on_abort : bool
            When True and abort=True, call QueryRewriter to generate 3 diverse
            query reformulations.  Requires api_key to be set.  Default True.

        Returns
        -------
        StreamGuardResult
        """
        import time as _time
        t0 = _time.time()

        t = min(len(steps_so_far), step_for_judge)

        # ── Behavioral SC_OLD prefix (always runs, $0) ──────────────────────
        lf_result      = self._scorer.score_prefix(question, steps_so_far, t)
        behavioral_risk = float(lf_result.risk_score)
        failure_mode   = lf_result.failure_mode

        # ── Optional Haiku mid-chain judge (~$0.001) ─────────────────────────
        haiku_risk: Optional[float] = None
        on_track: Optional[bool]    = None

        if self._api_key and t >= 1:
            haiku_risk, on_track = self._run_haiku_step_judge(
                question, steps_so_far, t
            )

        # ── Combine: Haiku 70%, SC_OLD 30% when Haiku available (exp107) ────
        # exp113 finding: SC_OLD prefix at step 2 is sign-inverted without
        # calibration (1-AUROC=0.57, direction opposite to full-chain scoring).
        # Fix: invert SC_OLD prefix when used standalone at step 2.
        sc_risk_for_blend = behavioral_risk
        if t <= 2:
            # Invert — at step 2, lower SC_OLD risk = higher failure probability
            sc_risk_for_blend = 1.0 - behavioral_risk

        if haiku_risk is not None:
            combined_risk = 0.30 * sc_risk_for_blend + 0.70 * haiku_risk
        else:
            combined_risk = sc_risk_for_blend

        # on_track=False is a hard early-warning override
        if on_track is False:
            combined_risk = max(combined_risk, 0.68)

        abort = combined_risk >= abort_threshold

        # ── Query rewriting when aborting ────────────────────────────────────
        rewritten: List[str] = []
        if abort and rewrite_on_abort and self._api_key:
            try:
                from llm_guard.query_rewriter import QueryRewriter
                from llm_guard.trust_object import A2ATrustObject, TemporalValidity
                # Build a minimal trust object to drive the rewriter
                _dummy = A2ATrustObject(
                    answer="",
                    risk_score=combined_risk,
                    confidence_tier="LOW",
                    failure_mode=failure_mode,
                    step_count=t,
                    judge_label=None,
                    downstream_hint="rewrite_query",
                    should_rewrite=True,
                )
                rw = QueryRewriter(api_key=self._api_key)
                rewritten = rw.rewrite_if_needed(question, _dummy)
            except Exception:
                pass

        latency = (_time.time() - t0) * 1000

        return StreamGuardResult(
            abort=abort,
            risk_at_step=round(combined_risk, 4),
            step_index=t - 1,          # 0-based
            on_track=on_track,
            failure_mode_hint=failure_mode,
            behavioral_risk=round(behavioral_risk, 4),
            haiku_risk=round(haiku_risk, 4) if haiku_risk is not None else None,
            rewritten_queries=rewritten,
            latency_ms=round(latency, 1),
        )

    def _run_haiku_step_judge(
        self,
        question: str,
        steps: List[Dict],
        k: int,
    ):
        """
        Call Haiku partial-chain judge at step k.  Returns (risk_float, on_track_bool).

        Uses the same prompt and JSON format as exp107 so that exp107's partial
        cache files can be reused by exp113.
        """
        _HAIKU_SYSTEM = (
            "You are an AI reasoning chain evaluator.\n"
            "You are given a PARTIAL reasoning chain (the agent is still mid-task).\n"
            "Assess whether it is on track to produce a correct final answer."
        )
        _HAIKU_USER = (
            "Evaluate this PARTIAL AI reasoning chain (agent still running).\n\n"
            "Question: {question}\n\n"
            "Partial chain ({k} steps so far):\n{chain_text}\n\n"
            "Do NOT penalise for lack of final answer. Assess queries and reasoning so far.\n\n"
            "Respond with ONLY this JSON (no extra text):\n"
            '{{"step_relevance": 1-5, "coherence": 1-5, '
            '"on_track": true_or_false, "risk_assessment": "LOW_or_MEDIUM_or_HIGH"}}'
        )
        _RISK_MAP = {"LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.85}

        chain_lines = []
        for i, s in enumerate(steps[:k], 1):
            thought = s.get("thought", "").strip()[:200]
            a_type  = s.get("action_type", "")
            a_arg   = s.get("action_arg", "").strip()[:150]
            obs     = s.get("observation", "").strip()[:200]
            chain_lines.append(
                f"Step {i}: {thought}\n  Action: {a_type}[{a_arg}]\n  Result: {obs}"
            )
        chain_text = "\n".join(chain_lines) or "(no steps yet)"

        user_msg = _HAIKU_USER.format(
            question=question, k=k, chain_text=chain_text
        )

        try:
            # Use cheaper model for mid-chain: Haiku (Anthropic) or llama-3.1-8b (NIM)
            step_model = (
                self._nim_ptrue_model if (self._has_nim and not self._has_anthropic)
                else "claude-haiku-4-5-20251001"
            )
            raw = self._call_llm(
                system=_HAIKU_SYSTEM,
                user=user_msg,
                max_tokens=80,
                temperature=0.0,
                model_override=step_model,
                prefer_nim=(self._has_nim and not self._has_anthropic),
            )
            if raw is None:
                return None, None
            # Normalise literal placeholder tokens the model sometimes echoes
            raw = re.sub(r':\s*true_or_false\b', ': true', raw)
            raw = re.sub(r':\s*(LOW|MEDIUM|HIGH)_or_\S+', r': "\1"', raw)
            raw = re.sub(r':\s*1-5\b', ': 3', raw)
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None, None
            data     = json.loads(m.group())
            on_track = bool(data.get("on_track", True))
            ra       = str(data.get("risk_assessment", "MEDIUM")).upper().strip()
            risk     = _RISK_MAP.get(ra, 0.55)
            return risk, on_track
        except Exception:
            return None, None

    # ── Conditional mesh routing (exp115) ─────────────────────────────────────

    def route_to_mesh(
        self,
        question: str,
        trust: "A2ATrustObject",
        agent_answers: Optional[Dict[str, str]] = None,
        theta_high: float = 0.60,
        theta_low: float = 0.30,
    ) -> "MeshResult":
        """
        Conditional multi-agent consensus routing (exp115).

        When Agent A has LOW confidence, consult B and C in parallel and use
        pairwise token-F1 agreement to either upgrade the tier or escalate.

        Parameters
        ----------
        question : str
            Original question (used only for context in logging).
        trust : A2ATrustObject
            Trust object emitted by Agent A.  route_to_mesh() is typically
            called when trust.confidence_tier == "LOW".
        agent_answers : dict, optional
            {agent_id: answer_str} for agents B, C (and optionally D).
            When None, falls back to trust.answer only (single-agent mode,
            no upgrade possible).
        theta_high : float
            Mean pairwise F1 >= theta_high  →  upgrade tier to MEDIUM.
        theta_low : float
            Mean pairwise F1 < theta_low    →  escalate_to_human=True.

        Returns
        -------
        MeshResult

        Notes
        -----
        Expected AUROC improvement (exp108/exp115):
          Agreement standalone: 0.7278
          J5 + agreement (J6_new): 0.7876
        Two-stage OR strategy:
          stream_guard() at step 2  OR  score_chain().needs_alert at finish
          THEN route_to_mesh() on LOW tier  →  best recall in QPPG history.
        """
        from llm_guard.trust_object import MeshResult

        t0 = time.time()

        # Build answer pool: Agent A + any supplied agents
        all_answers: Dict[str, str] = {"A": trust.answer}
        if agent_answers:
            all_answers.update(agent_answers)

        ids = list(all_answers.keys())
        answers = [all_answers[k] for k in ids]

        # Compute pairwise token-F1
        pairwise: Dict[str, float] = {}
        f1_values: list = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_key = f"{ids[i]}{ids[j]}"
                f1 = _token_f1(answers[i], answers[j])
                pairwise[pair_key] = f1
                f1_values.append(f1)

        mean_f1 = float(np.mean(f1_values)) if f1_values else 0.0

        # Tier upgrade / escalation logic
        original_tier = trust.confidence_tier
        if mean_f1 >= theta_high:
            upgraded_tier = "MEDIUM" if original_tier == "LOW" else original_tier
        else:
            upgraded_tier = original_tier

        escalate = mean_f1 < theta_low

        # Choose consensus answer: the one with highest mean pairwise F1 to others
        if not escalate and len(answers) > 1:
            best_idx, best_score = 0, -1.0
            for i, ans_i in enumerate(answers):
                others = [answers[j] for j in range(len(answers)) if j != i]
                mean_to_others = float(np.mean([_token_f1(ans_i, o) for o in others]))
                if mean_to_others > best_score:
                    best_score = mean_to_others
                    best_idx = i
            consensus_answer: Optional[str] = answers[best_idx]
        elif len(answers) == 1:
            consensus_answer = answers[0]
        else:
            consensus_answer = None

        latency = (time.time() - t0) * 1000

        return MeshResult(
            original_tier=original_tier,
            upgraded_tier=upgraded_tier,
            agreement_score=round(mean_f1, 4),
            escalate_to_human=escalate,
            consensus_answer=consensus_answer,
            agent_answers=all_answers,
            pairwise_f1=pairwise,
            theta_high=theta_high,
            theta_low=theta_low,
            latency_ms=round(latency, 1),
        )

    # ── Adaptive adapter selection (exp116) ───────────────────────────────────

    @staticmethod
    def should_retry(
        attempt_risks: list,
        max_retries: int = 3,
        low_threshold: float = 0.40,
        high_threshold: float = 0.65,
        min_improvement: float = 0.05,
    ) -> dict:
        """
        Retry budget advisor: decides whether to retry, use the result, or accept
        uncertainty based on the risk score trajectory across attempts.

        Implements Karpathy's 'march of nines' principle: each retry is expensive;
        only retry when there is evidence of improvement or budget remains.

        Parameters
        ----------
        attempt_risks   : list of risk scores from each attempt (most recent last)
        max_retries     : maximum attempts before accepting uncertainty
        low_threshold   : risk below this → result is acceptable (use it)
        high_threshold  : risk above this → strong failure signal
        min_improvement : minimum per-attempt risk reduction to justify retrying

        Returns
        -------
        dict with keys:
          action          — "use_result" | "retry" | "accept_uncertainty"
          attempt         — current attempt number
          budget_remaining — retries left
          reason          — explanation
          suggestion      — what to do next
        """
        n = len(attempt_risks)
        budget_remaining = max(0, max_retries - n)

        # No attempts yet
        if n == 0:
            return {
                "action": "retry",
                "attempt": 0,
                "budget_remaining": max_retries,
                "reason": "No attempts made yet.",
                "suggestion": "Run the agent chain and pass the risk score to attempt_risks.",
            }

        latest = attempt_risks[-1]

        # Latest attempt is LOW risk — accept it
        if latest < low_threshold:
            return {
                "action": "use_result",
                "attempt": n,
                "budget_remaining": budget_remaining,
                "risk_score": round(latest, 4),
                "reason": f"Risk {latest:.3f} is below threshold {low_threshold} — answer is reliable.",
                "suggestion": "Use this answer directly.",
            }

        # Budget exhausted
        if budget_remaining <= 0:
            best = min(attempt_risks)
            return {
                "action": "accept_uncertainty",
                "attempt": n,
                "budget_remaining": 0,
                "risk_score": round(best, 4),
                "reason": f"Max retries ({max_retries}) reached. Best risk was {best:.3f}.",
                "suggestion": (
                    "Return best attempt with uncertainty caveat, or respond 'I don't know'. "
                    "Consider using activate_adapter('retrieval_fail') for recovery hints."
                ),
            }

        # Check improvement trend (need ≥2 attempts)
        if n >= 2:
            trend = attempt_risks[-1] - attempt_risks[-2]   # negative = improving
            avg_trend = (attempt_risks[-1] - attempt_risks[0]) / (n - 1)

            # Getting worse or stalling → stop
            if trend > min_improvement:
                return {
                    "action": "accept_uncertainty",
                    "attempt": n,
                    "budget_remaining": budget_remaining,
                    "risk_score": round(min(attempt_risks), 4),
                    "reason": f"Risk increased +{trend:.3f} last attempt — agent is not converging.",
                    "suggestion": (
                        "Stop retrying. Use the lowest-risk attempt or return 'I don't know'. "
                        "Risk trajectory: " + " → ".join(f"{r:.2f}" for r in attempt_risks)
                    ),
                }

            # Improving meaningfully → encourage retry
            if trend < -min_improvement:
                return {
                    "action": "retry",
                    "attempt": n,
                    "budget_remaining": budget_remaining,
                    "risk_score": round(latest, 4),
                    "reason": f"Risk improved {trend:.3f} last attempt — still converging.",
                    "suggestion": (
                        f"Retry ({budget_remaining} attempts left). "
                        "Risk trajectory: " + " → ".join(f"{r:.2f}" for r in attempt_risks)
                    ),
                }

            # Stalling (small change) — retry only if HIGH risk
            if latest >= high_threshold:
                return {
                    "action": "retry",
                    "attempt": n,
                    "budget_remaining": budget_remaining,
                    "risk_score": round(latest, 4),
                    "reason": f"High risk {latest:.3f} with small trend {trend:+.3f} — one more attempt.",
                    "suggestion": f"Retry once more ({budget_remaining} left). Consider rephrasing the query.",
                }
            else:
                return {
                    "action": "accept_uncertainty",
                    "attempt": n,
                    "budget_remaining": budget_remaining,
                    "risk_score": round(latest, 4),
                    "reason": f"Risk stalled at {latest:.3f} ({trend:+.3f} trend) — further retries unlikely to help.",
                    "suggestion": "Use this answer with a caveat or return partial response.",
                }

        # First attempt, risk is medium-to-high: retry if budget allows
        return {
            "action": "retry" if budget_remaining > 0 else "accept_uncertainty",
            "attempt": n,
            "budget_remaining": budget_remaining,
            "risk_score": round(latest, 4),
            "reason": f"First attempt risk {latest:.3f} is above threshold {low_threshold}.",
            "suggestion": (
                f"Retry ({budget_remaining} left). Try rephrasing the question or using a different search strategy."
                if budget_remaining > 0 else
                "No retries left. Use best available answer with caveat."
            ),
        }

    def activate_adapter(
        self,
        failure_mode: Optional[str],
        registry: Optional[AdapterRegistry] = None,
    ) -> AdapterResult:
        """
        Select the appropriate recovery adapter for a detected failure mode (exp116).

        Call this after score_chain() or generate_trust_object() to get a
        structured adapter configuration for the next agent invocation.
        The adapter provides a system_hint (prompt injection), search_strategy,
        temperature_delta, and (optionally) a max_steps_override.

        Parameters
        ----------
        failure_mode : str or None
            Failure mode from ChainTrustResult.failure_mode or A2ATrustObject.failure_mode.
            Known values: "retrieval_fail", "repeated_query", "long_chain",
            "empty_answer", "low_retrieval_quality", "no_evidence", None.
        registry : AdapterRegistry, optional
            Custom registry.  Uses the built-in default registry when None.

        Returns
        -------
        AdapterResult
            Contains the selected AdapterConfig (system_hint, search_strategy,
            temperature_delta, max_steps_override, adapter_id).

        Example
        -------
            result  = guard.score_chain(question, steps, final_answer)
            adapter = guard.activate_adapter(result.failure_mode)
            if adapter.activated:
                # Inject adapter.config.system_hint into the next agent's system prompt
                # Adjust temperature by adapter.config.temperature_delta
                pass

        Cross-research (EHC / ECL)
        --------------------------
        Adapters are EHC "reusable primitives" — specialised micro-modules.
        Activation mirrors ECL's homeostatic drive mechanism: when a failure mode
        is detected (drive fired), the corresponding adapter is selected.

        Validated in exp116 (failure mode precision + coverage on HP chains).
        Expected: failure_mode detected on ~40-60% of wrong chains;
        adapter activation precision (fraction wrong) ~0.70-0.84.
        """
        reg = registry if registry is not None else AdapterRegistry()
        return reg.get(failure_mode)

    def _detect_failure_mode(
        self,
        steps: List[Dict],
        final_answer: str,
    ) -> Optional[str]:
        """
        Detect a failure mode from an agent chain's steps and final answer.

        Returns a failure mode string (e.g. "confident_wrong", "retrieval_fail")
        or None if no specific failure mode is detected.

        Parameters
        ----------
        steps : list of dict
            Agent steps, each with keys: thought, action_type, action_arg, observation.
        final_answer : str
            The agent's final answer string.
        """
        # confident_wrong: short chain + no hedging + no backtracking
        search_steps = [s for s in steps if s.get("action_type", "").lower() != "finish"]
        if len(search_steps) <= 2:
            all_thoughts = " ".join(s.get("thought", "") for s in steps)
            has_hedge = bool(_HEDGE_RE.search(all_thoughts))
            queries = [s.get("action_arg", "") for s in search_steps]
            has_backtrack = len(queries) > len(set(queries))
            if not has_hedge and not has_backtrack:
                return "confident_wrong"

        return None

    # ── Pre-screening ──────────────────────────────────────────────────────────

    def score_chain_start(self, question: str) -> Dict:
        """
        Pre-screen a question BEFORE running the agent at all.

        Most signal comes from the chain itself — this is primarily useful for
        routing obviously difficult questions to a stronger model upfront.

        Returns dict with: risk_score, tier, confidence, predicted_outcome,
        recommended_action.
        """
        lf_result  = self._scorer.score(question, [], "", False)
        risk_score = float(lf_result.risk_score)
        tier       = _risk_to_tier(risk_score)
        _, confidence, predicted = _risk_to_labels(risk_score)

        recommendation = {
            "LOW":    "use_stronger_model_or_add_verification",
            "MEDIUM": "proceed_with_monitoring",
            "HIGH":   "proceed",
        }[tier]

        return {
            "risk_score":         round(risk_score, 4),
            "tier":               tier,
            "confidence":         confidence,
            "predicted_outcome":  predicted,
            "recommended_action": recommendation,
        }

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict:
        """Return a summary of the guard's current configuration and validated performance."""
        mode = "behavioral"
        if self._use_structural_verifier and self._structural_verifier is not None:
            mode = "structural_logreg (SC2/SC3/SC5, exp119, cross-domain)"
        elif self._use_local_verifier and self._local_verifier is not None:
            mode = f"j_local (verifier trained on {self._local_verifier.n_train} chains)"
        elif self._use_judge:
            mode = f"j5_sonnet ({self._judge_model})"

        return {
            "mode":                        mode,
            "use_judge":                   self._use_judge,
            "use_local_verifier":          self._use_local_verifier,
            "use_structural_verifier":     self._use_structural_verifier,
            "judge_model":                 self._judge_model if self._use_judge else None,
            "alert_threshold":             self._alert_threshold,
            "calibrated":                  self._fitted,
            "verifier_fitted":             self._local_verifier.is_fitted if self._local_verifier else False,
            "structural_verifier_fitted":  self._structural_verifier is not None,
            "expected_auroc": {
                "within_domain_behavioral":                "~0.81 (exp88 5-fold CV)",
                "within_domain_j5_sonnet":                 "~0.78 (exp89 single run)",
                "within_domain_j_local_200":               "~0.80 (exp111 5-fold CV)",
                "cross_domain_behavioral":                 "~0.68 (exp119/120 TriviaQA)",
                "cross_domain_structural_logreg_200":      "~0.73 (exp119 TriviaQA, $0)",
                "cross_domain_ptrue_haiku":                "~0.74 (exp120 TriviaQA, ~$0.0003/call)",
                "cross_domain_behavioral_ptrue_ensemble":  "~0.78 (exp120 TriviaQA, best cross-domain)",
                "cross_domain_j5_sonnet":                  "~0.74 (exp91 NQ)",
            },
            "conformal_alerting_at_threshold_0.70": {
                "fpr_guarantee": "≤ 10%",
                "precision":     0.908,
                "recall":        0.595,
                "source":        "exp92",
            },
        }

    # ── Backward-compatible helper (kept for existing integrations) ────────────

    @staticmethod
    def format_step_context(
        question: str,
        steps_so_far: List[Dict],
        current_action: str,
    ) -> str:
        """Serialise a step into a string (legacy format, kept for compatibility)."""
        parts = [f"TASK: {question}"]
        for i, s in enumerate(steps_so_far, 1):
            thought = s.get("thought", "")[:200]
            if "action_type" in s and "action_arg" in s:
                action = f"{s['action_type']}[{s['action_arg']}]"[:200]
            else:
                action = str(s.get("action", ""))[:200]
            parts.append(f"STEP {i}: Thought: {thought} | Action: {action}")
        parts.append(f"CURRENT: {current_action[:300]}")
        return "\n".join(parts)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_judge(self, question: str, steps: List[Dict], final_answer: str):
        """Call judge (exp89). Uses Anthropic Sonnet or NVIDIA NIM. Returns (label, risk_float) or (None, None)."""
        chain_text = _format_chain_for_judge(steps)
        user_msg   = _JUDGE_USER_TMPL.format(
            question=question,
            chain_text=chain_text,
            final_answer=final_answer,
        )
        # NIM judge model default: meta/llama-3.1-70b-instruct (no system-prompt via Anthropic format)
        nim_model = self._nim_judge_model
        try:
            raw = self._call_llm(
                system=_JUDGE_SYSTEM,
                user=user_msg,
                max_tokens=150,
                temperature=0.0,
                model_override=nim_model if self._has_nim and not self._has_anthropic else None,
                prefer_nim=(self._has_nim and not self._has_anthropic),
            )
            if raw is None:
                return None, None
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None, None
            data  = json.loads(m.group())
            label = str(data.get("label", "BORDERLINE")).upper().strip()
            if label not in _JUDGE_RISK:
                label = "BORDERLINE"
            return label, _JUDGE_RISK[label]
        except Exception:
            return None, None


# ── Module-level helpers ──────────────────────────────────────────────────────

_MESH_STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","on","at","for","with","by","from","up","and","or","but","not",
    "so","if","then","than","about","which","who","what","how","when","where",
}


def _token_f1(a: str, b: str) -> float:
    """Mean token-F1 between two answer strings (stop-word filtered)."""
    def toks(text: str) -> set:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if w not in _MESH_STOP_WORDS and len(w) > 1}

    pa, rb = toks(a), toks(b)
    if not pa or not rb:
        return 0.0
    c = pa & rb
    prec = len(c) / len(pa)
    rec  = len(c) / len(rb)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _risk_to_tier(risk_score: float) -> str:
    if risk_score < 0.50:
        return "HIGH"
    elif risk_score < 0.70:
        return "MEDIUM"
    return "LOW"


def _risk_to_labels(risk_score: float):
    if risk_score < 0.50:
        return "low", "high", "likely_success"
    elif risk_score < 0.70:
        return "medium", "medium", "uncertain"
    return "high", "low", "likely_failure"


def _build_downstream_hint(
    tier: str,
    judge_label: Optional[str],
    failure_mode: Optional[str],
) -> str:
    if tier == "HIGH":
        return "proceed"
    if tier == "MEDIUM":
        return "proceed_with_caution"
    if judge_label == "POOR":
        return "escalate_to_human"
    if failure_mode in ("retrieval_fail", "no_evidence", "repeated_query"):
        return "rewrite_and_verify"
    return "rewrite_query"


def _format_chain_for_judge(steps: List[Dict]) -> str:
    parts = []
    for i, s in enumerate(steps, 1):
        thought = s.get("thought", "").strip()[:300]
        a_type  = s.get("action_type", "")
        a_arg   = s.get("action_arg", "").strip()[:200]
        obs     = s.get("observation", "").strip()[:300]
        parts.append(
            f"Step {i}: {thought}\n"
            f"  Action: {a_type}[{a_arg}]\n"
            f"  Result: {obs}"
        )
    return "\n".join(parts) if parts else "(no steps)"
