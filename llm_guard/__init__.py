"""
llm-guard-kit v0.17.0: Real-time reliability monitoring, failure diagnosis,
cross-domain scoring, A2A trust management, and NVIDIA NIM backend for LLM agents.

Validated performance (held-out evaluation only — no in-sample figures):
  Within-domain AUROC (behavioral, $0):         ~0.76  (exp153, HP 200 real chains, 5-fold CV)
  Within-domain AUROC (+ Sonnet judge):         ~0.78  (exp89)
  Within-domain AUROC (+ local verifier):       ~0.80  (exp111, 200 labels, $0 inference)
  Cross-domain AUROC (behavioral L1, $0):       ~0.77  (exp153, HP→TriviaQA real chains)
  Cross-domain AUROC (L1+L5 semantic, $0):      ~0.75  (exp155, HP→TriviaQA)
  Cross-domain AUROC (P(True) Haiku alone):     ~0.74  (exp120, ~$0.0003/call, zero-shot)
  Cross-domain AUROC (behavioral+P(True) 50/50):~0.78  (exp120, best cross-domain signal)
  Alert precision at FPR ≤ 10%:                 0.908  (exp92 conformal calibration)
  Mid-chain abort at step 2:                     AUROC 0.683  (exp107)
  Behavioral latency p50:                        ~19ms  (exp_latency, no API key)
  Non-HF domains (code/SQL/CS):                 AUROC=1.0 on structured chains (exp150-152)

v0.17.0 new:
  MiniJudge                  — $0 local judge distilled from Sonnet (AUROC 0.747, exp159)
  Cross-domain TV validated  — AUROC 0.660 [CI 0.614–0.705] n=1000 chains (exp156)
  NQ cross-domain            — AUROC 0.524 (open-domain factoid, harder domain)
  QuickCalibrator            — domain-adaptive isotonic calibration (20-chain setup)
  Platform integrations      — Langfuse, LangSmith, Prometheus/Grafana, Datadog

v0.16.1 new:
  Fix FastAPI server dep to >=0.115 (starlette 0.52 compat — middleware 3-tuple fix)
  exp153 domain-invariant selection: L1+selected L2-L4 = 0.778 cross-domain (+0.8pp vs L1 alone)

v0.16.0 new:
  POST /v2/calibrate/fit   — hosted calibration endpoint: fit DeepLocalVerifier on labeled chains
  exp153 multilevel feats  — L1+L2+L3+L4 30 features; L1 baseline dominates cross-domain (0.770)
  exp155 semantic encoder  — L5 (sentence-transformer cosine) + Transformer encoder + energy baseline
  exp150-152 domain valid  — code interpreter, Chinook SQL, customer service domains validated
  exp154 Mistral-7B probe  — pending model download (expected ≥0.65 based on Qwen2.5 exp145/146)
  Latency SLA benchmark    — behavioral p50=19ms, local verifier p50=19ms

v0.15.0 new:
  probe_ensemble_blend()     — blend WhiteBoxProbe + P(True) at alpha=0.25 (+1.6pp AUROC, exp148)
  Corrected LSTM AUROC       — LSTMRiskAccumulator cross-domain 0.545 not 0.8280 (exp143 3-domain)
  Real probe validation      — exp145/146 Qwen2.5 results documented (mean=0.633/0.585)

v0.14.0 new:
  ProcessReliabilityMonitor  — generic domain-agnostic monitor (plug in any StepExtractor)
  StepExtractor ABC          — define extract(step) to add new process domains
  LLMReActExtractor          — ReAct-specific extractor (default for AgentGuard)
  confident_wrong adapter    — detects short confident chains; routes to verify_claim
  Per-domain threshold       — DriftMonitor auto-calibrates alert threshold after ALARM
  Domain-invariant LSTM      — retrieval_conf + semantic_gap replace content-heavy features

v0.10.0 new:
  AgentGuard(nim_api_key="nvapi-...")  — NVIDIA NIM backend (llama-3.1-70b judge, 8b P(True))
  compare_backends MCP tool            — per-backend AUROC A/B tracking over time
  Automatic Anthropic↔NIM fallback     — no single point of failure

v0.9.0 new:
  guard.score_with_ptrue()           — P(True) ensemble (exp120, ~$0.0003/call)
  guard.fit_structural_verifier()    — structural-only LogReg for cross-domain (exp119/124)
  guard.fit_structural_verifier(..., target_unlabeled_runs=...) — +2pp with unlabeled target data

Quick start
-----------
    from llm_guard import AgentGuard, LLMGuard, SmartRouter

    # --- Chain scoring (behavioral, $0) ---
    guard  = AgentGuard()
    result = guard.score_chain(question, steps, final_answer)
    trust  = guard.generate_trust_object(question, steps, final_answer)

    # --- Cross-domain scoring with P(True) via Anthropic (~$0.0003/call) ---
    guard  = AgentGuard(api_key="sk-ant-...")
    result = guard.score_with_ptrue(question, steps, final_answer)

    # --- Cross-domain scoring with P(True) via NVIDIA NIM (llama-3.1-8b) ---
    guard  = AgentGuard(nim_api_key="nvapi-...")
    result = guard.score_with_ptrue(question, steps, final_answer)
    # result.risk_score is behavioral+P(True) ensemble (AUROC ~0.78 cross-domain)

    # --- Cross-domain structural LogReg ($0 inference, 200 labeled source chains) ---
    guard = AgentGuard()
    guard.fit_structural_verifier(labeled_source_runs)
    # Optional: pass unlabeled target chains for feature normalization (+2pp)
    guard.fit_structural_verifier(labeled_source_runs, target_unlabeled_runs=target_runs)
    result = guard.score_chain(question, steps, final_answer)

    # --- Local verifier (within-domain, $0 inference after 200 labels) ---
    guard = AgentGuard(use_local_verifier=True)
    guard.fit_verifier(labeled_runs)
    result = guard.score_chain(question, steps, final_answer)

    # --- Zero-install nano scorer (no package needed) ---
    from llm_guard import QPPGNano
    nano = QPPGNano()
    result = nano.score_chain(question, steps, final_answer)

    # --- Cost-optimal model routing ---
    router = SmartRouter(LLMGuard(api_key="sk-ant-..."))
    result = router.route("What is 15% of 240?")
    print(result.model_used)  # Haiku for easy, Sonnet/Opus for hard
"""

from qppg.guard import QPPGLLMGuard as LLMGuard
from qppg.guard import GuardResult
from llm_guard.client import GuardClient, ScoreResult, MonitorResult as _ClientMonitorResult
from llm_guard.step_extractor import StepExtractor, LLMReActExtractor
from llm_guard.process_monitor import ProcessReliabilityMonitor, MonitorResult
from llm_guard.router import SmartRouter, RouterResult
from llm_guard.agent_guard import AgentGuard, AgentStepResult, ChainTrustResult, PtrueWeightBandit
from llm_guard.trust_object import A2ATrustObject, TrustHop, MeshResult, StreamGuardResult, TemporalValidity
from llm_guard.query_rewriter import QueryRewriter, RewriteResult
from llm_guard.nano import QPPGNano
from llm_guard.local_verifier import LocalVerifier, FEATURE_NAMES, extract_features
from llm_guard.adapter_registry import AdapterRegistry, AdapterConfig, AdapterResult
from llm_guard.white_box_probe import WhiteBoxProbe, ProbeResult, probe_ensemble_blend
from llm_guard.step_normalizer import normalize_steps, validate_step_coverage
from llm_guard.adaptive_cisc import AdaptiveCISC, AdaptiveCISCRegistry
from llm_guard.drift_detector import DriftDetector, DriftMonitor, DriftEvent
from llm_guard.deep_verifier import DeepLocalVerifier, LSTMRiskAccumulator
from llm_guard.quick_calibration import QuickCalibrator
from llm_guard.mini_judge import MiniJudge
from llm_guard.integrations.langchain import AgentGuardCallback
from llm_guard.integrations.llamaindex import AgentGuardEventHandler
from llm_guard.integrations.crewai import AgentGuardCrewCallback

__version__ = "0.20.1"
__all__ = [
    # One-line SDK client (SaaS + local modes)
    "GuardClient",
    "ScoreResult",
    # Core guard (KNN + repair)
    "LLMGuard",
    "GuardResult",
    # Cost-optimal routing
    "SmartRouter",
    "RouterResult",
    # Agent monitoring (SC_OLD + Sonnet judge / local verifier)
    "AgentGuard",
    "AgentStepResult",
    "ChainTrustResult",
    "PtrueWeightBandit",
    # A2A trust object + stream guard (exp113) + mesh routing (exp115) + multi-hop chain (v0.4)
    "A2ATrustObject",
    "TrustHop",
    "StreamGuardResult",
    "MeshResult",
    "TemporalValidity",
    # Query diversification
    "QueryRewriter",
    "RewriteResult",
    # Zero-install nano scorer
    "QPPGNano",
    # Local verifier (LogReg on 12 SC features, $0 inference)
    "LocalVerifier",
    "FEATURE_NAMES",
    "extract_features",
    # Adaptive adapter selection (exp116)
    "AdapterRegistry",
    "AdapterConfig",
    "AdapterResult",
    # White-box hidden-state probe (exp117) + ensemble blend (exp148)
    "WhiteBoxProbe",
    "ProbeResult",
    "probe_ensemble_blend",
    # Framework integrations (2-line drop-in callbacks)
    "AgentGuardCallback",
    "AgentGuardEventHandler",
    "AgentGuardCrewCallback",
    # Multi-format step normalization
    "normalize_steps",
    "validate_step_coverage",
    # Adaptive CISC threshold bandit
    "AdaptiveCISC",
    "AdaptiveCISCRegistry",
    # Drift detection (exp118: CUSUM + PSI, validated)
    "DriftDetector",
    "DriftMonitor",
    "DriftEvent",
    # Deep verifiers (exp138: bootstrap MLP + LSTM risk accumulator)
    "DeepLocalVerifier",
    "LSTMRiskAccumulator",
    # Mini-judge: $0 local judge distilled from Sonnet (exp159, AUROC 0.747)
    "MiniJudge",
    # Domain-adaptive calibration (QuickCalibrator: 20-chain isotonic setup)
    "QuickCalibrator",
    # Generic framework (v0.14.0)
    "StepExtractor",
    "LLMReActExtractor",
    "ProcessReliabilityMonitor",
    "MonitorResult",
]
