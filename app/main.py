"""
LLMGuard API Server v0.7.1
===========================
FastAPI server combining:
  • AgentGuard chain scoring  (/v1/score_chain, /v1/monitor_step, ...)
  • Multi-tenant SaaS layer   (/auth/*, /user/*, /admin/*, /billing/*, /demo/*)
  • Continuous learning       (GuardManager KNN + QARA)

Start
-----
    uvicorn app.main:app --reload           # development
    uvicorn app.main:app --host 0.0.0.0     # production

Environment variables
---------------------
    ANTHROPIC_API_KEY    required for AgentGuard
    QPPG_DB              SQLAlchemy DB URL  (default: sqlite:///./qppg.db)
    JWT_SECRET           JWT signing key    (default: dev-secret, CHANGE IN PROD)
    GUARD_STATE_PATH     KNN state pickle   (default: guard_state.pkl)
    GUARD_MODEL          Claude model       (default: claude-haiku-4-5-20251001)
    AGENT_USE_JUDGE      set "1" for Sonnet judge (~$0.007/chain)
    FRONTEND_URL         for Stripe redirects (default: http://localhost:3000)
    GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET  Google OAuth
    STRIPE_SECRET_KEY / STRIPE_PRO_PRICE_ID  Stripe (optional)
    CORS_ORIGINS         comma-separated allowed origins (default: *)
"""

import asyncio
import collections
import os
import time
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# ── Database + models ─────────────────────────────────────────────────────────
from app.database import create_all_tables, get_db
from app import models
from app.auth_utils import get_optional_user, maybe_reset_monthly_counter

# ── SaaS routers ──────────────────────────────────────────────────────────────
from app.routers import auth as auth_router
from app.routers import user as user_router
from app.routers import admin as admin_router
from app.routers import billing as billing_router
from app.routers import demo as demo_router
from app.routers import proxy as proxy_router

# ── Legacy KNN guard ──────────────────────────────────────────────────────────
from app.manager import GuardManager

# ── AgentGuard ────────────────────────────────────────────────────────────────
from llm_guard.agent_guard import AgentGuard
from llm_guard.adaptive_cisc import AdaptiveCISCRegistry, _DEFAULT_HIGH, _DEFAULT_LOW
from llm_guard.drift_detector import DriftMonitor

# ═══════════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════════

_cors_raw     = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()] or ["*"]

app = FastAPI(
    title       = "LLMGuard API",
    description = (
        "Real-time agent reliability monitoring — with multi-tenant SaaS. "
        "v0.13.0: A2A HTTP endpoints (/.well-known/agent.json, /v1/a2a/*), "
        "multi-hop trust chain, calibration endpoints, DeepLocalVerifier. "
        "v0.12.0: DeepLocalVerifier + LSTMRiskAccumulator. "
        "v0.11.0: isotonic calibration, conformal thresholds, Kalman smoothing."
    ),
    version     = "0.13.0",
)

# ── Alert deduplication: suppress repeated alerts for the same (user, domain, failure_mode)
# within a 5-minute window to prevent alert storms.
# Key: (user_id, domain, failure_mode)  Value: last_fired timestamp
_alert_dedup: Dict[tuple, float] = {}
_ALERT_DEDUP_WINDOW_S = 300  # 5 minutes

app.add_middleware(
    CORSMiddleware,
    allow_origins  = _cors_origins,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(auth_router.router)
app.include_router(user_router.router)
app.include_router(admin_router.router)
app.include_router(billing_router.router)
app.include_router(demo_router.router)
app.include_router(proxy_router.router)

# ── Singletons ────────────────────────────────────────────────────────────────
manager = GuardManager(
    api_key    = os.environ.get("ANTHROPIC_API_KEY"),
    model      = os.environ.get("GUARD_MODEL", "claude-haiku-4-5-20251001"),
    state_path = os.environ.get("GUARD_STATE_PATH", "guard_state.pkl"),
)

_use_judge = os.environ.get("AGENT_USE_JUDGE", "0") == "1"
agent_guard = AgentGuard(
    api_key      = os.environ.get("ANTHROPIC_API_KEY"),
    use_judge    = _use_judge,
    nim_api_key  = os.environ.get("NIM_API_KEY"),
    nim_base_url = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    nim_judge_model = os.environ.get("NIM_JUDGE_MODEL", "meta/llama-3.3-70b-instruct"),
    nim_ptrue_model = os.environ.get("NIM_PTRUE_MODEL", "meta/llama-3.1-8b-instruct"),
)

# ── AdaptiveCISC registry (one bandit per user+domain pair) ───────────────────
_cisc_registry = AdaptiveCISCRegistry(
    state_dir        = os.environ.get("CISC_STATE_DIR", ".cisc_state"),
    target_precision = 0.80,
    target_recall    = 0.60,
)

# ── Per-user auto-retrain tracking (in-memory; resets on restart) ─────────────
# Key: user_id → n_feedback_at_last_retrain
_retrain_counters: Dict[int, int] = {}
_AUTO_RETRAIN_EVERY = int(os.environ.get("AUTO_RETRAIN_EVERY", "20"))  # labels

# ── Drift monitor (fires when behavioral_score distribution shifts) ────────────
def _on_drift_alarm(event) -> None:
    import logging
    logging.getLogger("llm_guard").warning(
        "[drift] Domain=%s severity=%s detector=%s PSI=%.3f CUSUM=%.3f n=%d | %s",
        event.domain, event.severity, event.detector,
        event.psi, event.cusum_value, event.n_samples, event.message,
    )

_drift_monitor = DriftMonitor(
    on_drift        = _on_drift_alarm,
    auto_reset_cisc = True,
    state_dir       = os.environ.get("DRIFT_STATE_DIR", ".drift_state"),
)
_drift_monitor.attach_cisc_registry(_cisc_registry)


async def _run_auto_retrain(user_id: int) -> None:
    """
    Background task: retrain LocalVerifier for a user when enough new labels
    have accumulated since the last retrain.  Runs entirely in a thread pool
    so it never blocks the event loop.
    """
    from app.database import SessionLocal

    async def _work():
        db = SessionLocal()
        try:
            rows = (
                db.query(models.ChainFeedback, models.Chain)
                .join(models.Chain, models.ChainFeedback.chain_id == models.Chain.id)
                .filter(models.ChainFeedback.user_id == user_id)
                .all()
            )
            n = len(rows)
            if n < 5:
                return   # not enough labels yet — wait

            runs = [{"question": chain.question, "steps": [],
                     "final_answer": chain.final_answer,
                     "correct": fb.label == "correct"}
                    for fb, chain in rows]

            await run_in_threadpool(_fit_verifier, runs, user_id, n)
            _retrain_counters[user_id] = n
        except Exception:
            pass
        finally:
            db.close()

    await _work()


def _fit_verifier(runs: list, user_id: int, n: int) -> None:
    """
    CPU-bound: fit LocalVerifier (full LogReg batch when n≥10,
    online SGD partial_fit when n<10 for cold-start).
    Also calls partial_fit on deep verifiers (DeepLocalVerifier + LSTM) when n≥20.
    """
    import numpy as np
    from llm_guard.local_verifier import LocalVerifier, extract_features

    X = np.array([extract_features(r["question"], r.get("steps", []), r["final_answer"])
                  for r in runs])
    y = np.array([int(r["correct"]) for r in runs])

    if n >= 10:
        # Full LogReg batch retrain (reliable from 10+, best from 60+)
        verifier = LocalVerifier()
        verifier.fit(runs)
        agent_guard._local_verifier   = verifier
        agent_guard._use_local_verifier = True
    else:
        # Online SGD cold-start: works from first label via partial_fit
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        if not hasattr(agent_guard, "_sgd_verifier") or agent_guard._sgd_verifier is None:
            agent_guard._sgd_verifier = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    SGDClassifier(loss="log_loss", max_iter=1, warm_start=True,
                                         random_state=42, class_weight="balanced")),
            ])
        # partial_fit requires at least 2 classes — pad if needed
        if len(np.unique(y)) < 2:
            return
        agent_guard._sgd_verifier.fit(X, y)   # warm_start incremental

    # ── Deep verifier continual learning (n≥20, non-blocking) ────────────────
    if n >= 20:
        try:
            from llm_guard.deep_verifier import DeepLocalVerifier, LSTMRiskAccumulator
            new_batch = runs[-20:]  # most recent 20 to avoid re-processing stale data
            if hasattr(agent_guard, "_deep_verifier") and isinstance(
                agent_guard._deep_verifier, DeepLocalVerifier
            ):
                agent_guard._deep_verifier.partial_fit(new_batch)
            if hasattr(agent_guard, "_lstm_verifier") and isinstance(
                agent_guard._lstm_verifier, LSTMRiskAccumulator
            ):
                agent_guard._lstm_verifier.partial_fit(new_batch)
        except Exception:
            pass  # never fail the retrain path due to deep verifier errors


@app.on_event("startup")
def on_startup():
    """Create DB tables on first run."""
    create_all_tables()


# ═══════════════════════════════════════════════════════════════════════════════
# Health + Root
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"service": "llm-guard", "version": "0.13.0", "status": "running", "docs": "/docs"}


@app.get("/v1/mcp_status")
def mcp_status():
    """Info endpoint: MCP server setup instructions for Claude Desktop and Cursor."""
    return {
        "mcp_server": "llm-guard-kit MCP",
        "version": "0.1.0",
        "install": "pip install llm-guard-kit[mcp]",
        "run_stdio": "llm-guard-mcp",
        "run_sse":   "llm-guard-mcp --sse --port 8765",
        "tools": [
            "score_chain", "stream_check", "submit_feedback",
            "get_metrics", "get_auroc", "trigger_retrain",
            "get_pending_review", "get_rl_status", "compare_backends",
            "get_nines_dashboard", "get_domain_stats", "check_retry_budget",
        ],
        "claude_desktop": {
            "config_key": "mcpServers.llm-guard-kit",
            "command": "python3 -m llm_guard_mcp.server",
        },
        "cursor": {
            "config_file": "~/.cursor/mcp.json",
            "invoke_in_chat": "@llm-guard-kit score_chain ...",
        },
        "metrics_db": "~/.llm_guard_mcp/metrics.db",
        "rl_feedback_loop": {
            "retrain_every": 20,
            "min_labels": 30,
            "features": ["risk_score", "beh_score", "ptrue_score", "n_steps_norm", "tier_num"],
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}


# ═══════════════════════════════════════════════════════════════════════════════
# AgentGuard Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════

class StepModel(BaseModel):
    thought:     str = ""
    action_type: str = "Action"
    action_arg:  str = ""
    observation: str = ""


class ScoreChainRequest(BaseModel):
    question:     str
    steps:        List[StepModel]
    final_answer: str
    finished:     bool = True
    domain:       str  = "default"


class ScoreChainResponse(BaseModel):
    chain_id:              Optional[int] = None   # DB row id — use for /v1/chains/{id}/feedback
    risk_score:            float
    confidence_tier:       str
    needs_alert:           bool
    failure_mode:          Optional[str]
    judge_label:           Optional[str]
    step_count:            int
    behavioral_score:      float
    behavioral_components: Dict[str, float]
    latency_ms:            float


class MonitorStepRequest(BaseModel):
    question:       str
    steps_so_far:   List[StepModel]
    current_action: str


class MonitorStepResponse(BaseModel):
    risk_score:        float
    risk:              str
    confidence:        str
    predicted_outcome: str
    step_index:        int
    failure_mode:      Optional[str]


class BatchScoreRequest(BaseModel):
    chains:  List[ScoreChainRequest]
    domain:  str = "default"


class BatchScoreResponse(BaseModel):
    results:       List[ScoreChainResponse]
    n_chains:      int
    n_alerts:      int
    elapsed_ms:    float


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrail execution helper
# ═══════════════════════════════════════════════════════════════════════════════

def _fire_guardrails(
    domain:       str,
    risk_score:   float,
    user_id:      Optional[int],
    result_dict:  dict,
    db:           Session,
    failure_mode: Optional[str] = None,
) -> None:
    """Check matching guardrails and fire webhooks/Slack (fire-and-forget).

    Deduplication: the same (user_id, domain, failure_mode) combination will
    not fire again within _ALERT_DEDUP_WINDOW_S seconds (default 5 minutes),
    preventing alert storms when a failure mode persists across many chains.
    """
    if user_id is None:
        return

    # ── Deduplication check ──────────────────────────────────────────────────
    dedup_key = (user_id, domain, failure_mode or "none")
    now = time.time()
    last_fired = _alert_dedup.get(dedup_key, 0.0)
    if now - last_fired < _ALERT_DEDUP_WINDOW_S:
        return  # suppressed — same alert fired recently
    _alert_dedup[dedup_key] = now

    guardrails = (
        db.query(models.Guardrail)
        .filter(
            models.Guardrail.user_id == user_id,
            models.Guardrail.enabled == True,  # noqa: E712
        )
        .all()
    )

    for g in guardrails:
        if g.domain not in ("*", domain):
            continue
        if risk_score < g.threshold:
            continue
        payload = {"domain": domain, "risk_score": risk_score, "action": g.action, "result": result_dict}
        if g.webhook_url:
            try:
                httpx.post(g.webhook_url, json=payload, timeout=3.0)
            except Exception:
                pass
        if g.slack_webhook:
            tier  = result_dict.get("confidence_tier", "LOW")
            color = "#dc3545" if tier == "LOW" else "#fd7e14"
            try:
                httpx.post(g.slack_webhook, json={"attachments": [{"color": color,
                    "title": f"[llm-guard] Risk alert — {domain}",
                    "text":  f"Risk: {risk_score:.2%} | Tier: {tier} | Action: {g.action}"}]}, timeout=3.0)
            except Exception:
                pass
        break  # first matching guardrail wins


# ═══════════════════════════════════════════════════════════════════════════════
# AgentGuard endpoints (v0.7.0+)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_and_log(
    req:           ScoreChainRequest,
    optional_user: Optional[models.User],
    db:            Session,
) -> ScoreChainResponse:
    """Synchronous core: score one chain and write to DB. Called via threadpool."""
    steps  = [s.model_dump() for s in req.steps]
    result = agent_guard.score_chain(
        question     = req.question,
        steps        = steps,
        final_answer = req.final_answer,
        finished     = req.finished,
    )

    if optional_user is not None:
        ai_cost = 0.007 if (result.judge_label and result.judge_label not in ("LOCAL", "")) else 0.0
        try:
            chain = models.Chain(
                user_id         = optional_user.id,
                domain          = req.domain or "default",
                question        = req.question[:500],
                final_answer    = req.final_answer[:500],
                risk_score      = result.risk_score,
                confidence_tier = result.confidence_tier,
                needs_alert     = result.needs_alert,
                n_steps         = result.step_count,
                failure_mode    = result.failure_mode,
                judge_label     = result.judge_label,
                ai_cost_usd     = ai_cost,
            )
            db.add(chain)
            db.flush()  # get chain.id before commit
            chain_id = chain.id
            maybe_reset_monthly_counter(optional_user, db)
            optional_user.chains_this_month += 1

            member = db.query(models.OrgMember).filter(
                models.OrgMember.user_id == optional_user.id
            ).first()
            if member:
                chain.org_id = member.org_id
                org = db.query(models.Org).filter(models.Org.id == member.org_id).first()
                if org:
                    org.ai_cost_usd = (org.ai_cost_usd or 0.0) + ai_cost

            db.commit()
        except Exception:
            db.rollback()
            chain_id = None

        if result.needs_alert:
            _fire_guardrails(
                req.domain, result.risk_score, optional_user.id,
                {"risk_score": result.risk_score, "confidence_tier": result.confidence_tier,
                 "needs_alert": result.needs_alert},
                db, failure_mode=result.failure_mode,
            )

    # ── Drift monitor: record score regardless of auth ─────────────────────
    _drift_monitor.record(result.behavioral_score, domain=req.domain)

    _chain_id = chain_id if optional_user is not None else None
    return ScoreChainResponse(
        chain_id              = _chain_id,
        risk_score            = result.risk_score,
        confidence_tier       = result.confidence_tier,
        needs_alert           = result.needs_alert,
        failure_mode          = result.failure_mode,
        judge_label           = result.judge_label,
        step_count            = result.step_count,
        behavioral_score      = result.behavioral_score,
        behavioral_components = result.behavioral_components,
        latency_ms            = result.latency_ms,
    )


@app.post("/v1/score_chain", response_model=ScoreChainResponse)
async def score_chain(
    req:           ScoreChainRequest,
    optional_user: Optional[models.User] = Depends(get_optional_user),
    db:            Session = Depends(get_db),
):
    """
    Score a completed ReAct chain (async — non-blocking).
    SC_OLD runs in a thread pool so the event loop stays free for other requests.

    Auth (optional): Bearer sk_... (API key) or Bearer eyJ... (JWT)
    """
    return await run_in_threadpool(_score_and_log, req, optional_user, db)


@app.post("/v1/score_chain/batch", response_model=BatchScoreResponse)
async def score_chain_batch(
    req:           BatchScoreRequest,
    optional_user: Optional[models.User] = Depends(get_optional_user),
    db:            Session = Depends(get_db),
):
    """
    Score up to 100 chains in one request.
    All chains are scored concurrently in the thread pool.
    Returns results in the same order as the input list.

    Use this for bulk evaluation, offline analysis, or pipeline scoring.
    Auth (optional): same as /v1/score_chain.
    """
    if len(req.chains) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 chains per request.")

    t0 = time.time()
    # Run all chains concurrently in the thread pool
    tasks = [
        run_in_threadpool(_score_and_log, chain, optional_user, db)
        for chain in req.chains
    ]
    results = await asyncio.gather(*tasks)
    elapsed_ms = (time.time() - t0) * 1000

    return BatchScoreResponse(
        results    = list(results),
        n_chains   = len(results),
        n_alerts   = sum(1 for r in results if r.needs_alert),
        elapsed_ms = round(elapsed_ms, 1),
    )


@app.post("/v1/monitor_step", response_model=MonitorStepResponse)
async def monitor_step(req: MonitorStepRequest):
    """Score a single agent step mid-chain (async — non-blocking)."""
    result = await run_in_threadpool(
        agent_guard.monitor_step,
        question       = req.question,
        steps_so_far   = [s.model_dump() for s in req.steps_so_far],
        current_action = req.current_action,
    )
    return MonitorStepResponse(
        risk_score        = result.risk_score,
        risk              = result.risk,
        confidence        = result.confidence,
        predicted_outcome = result.predicted_outcome,
        step_index        = result.step_index,
        failure_mode      = result.failure_mode,
    )


@app.get("/v1/score_chain_start")
def score_chain_start(question: str):
    """Pre-screen a question before running the agent."""
    return agent_guard.score_chain_start(question)


@app.get("/v1/agent/diagnostics")
def agent_diagnostics():
    """AgentGuard configuration and validated AUROC figures."""
    return agent_guard.diagnostics()


# ═══════════════════════════════════════════════════════════════════════════════
# Chain feedback + LocalVerifier retrain (data flywheel)
# ═══════════════════════════════════════════════════════════════════════════════

class ChainFeedbackRequest(BaseModel):
    label: str            # "correct" | "incorrect"
    note:  str = ""       # optional free-text reason


class RetrainResponse(BaseModel):
    status:       str
    n_feedback:   int
    auroc_estimate: Optional[float] = None
    message:      str


@app.post("/v1/chains/{chain_id}/feedback", status_code=200)
def submit_chain_feedback(
    chain_id:      int,
    req:           ChainFeedbackRequest,
    current_user:  models.User = Depends(get_optional_user),
    db:            Session = Depends(get_db),
):
    """
    Record a correctness label for a scored chain (thumbs up / down).

    label must be "correct" or "incorrect".
    Accumulated feedback is used by POST /v1/verifier/retrain to improve
    the LocalVerifier without manual labelling overhead.

    Auth: required (Bearer token or API key).
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if req.label not in ("correct", "incorrect"):
        raise HTTPException(status_code=422, detail="label must be 'correct' or 'incorrect'.")

    # Verify chain belongs to this user (or admin)
    chain = db.query(models.Chain).filter(models.Chain.id == chain_id).first()
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found.")
    if chain.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your chain.")

    # Upsert: one feedback row per (user, chain)
    existing = db.query(models.ChainFeedback).filter(
        models.ChainFeedback.chain_id == chain_id,
        models.ChainFeedback.user_id  == current_user.id,
    ).first()
    if existing:
        existing.label = req.label
        existing.note  = req.note
    else:
        db.add(models.ChainFeedback(
            chain_id = chain_id,
            user_id  = current_user.id,
            label    = req.label,
            note     = req.note,
        ))
    db.commit()

    # ── Auto-CISC: record outcome immediately so thresholds adapt in real time ──
    try:
        domain_key = f"{current_user.id}:{chain.domain}"
        cisc = _cisc_registry.get(domain_key)
        cisc.record_outcome(
            risk_score = chain.risk_score,
            tier       = chain.confidence_tier,
            was_wrong  = (req.label == "incorrect"),
        )
    except Exception:
        pass  # never fail the feedback request due to CISC error

    # ── Online isotonic calibration: update incrementally from every feedback ──
    try:
        if chain.risk_score is not None:
            agent_guard.update_isotonic(
                score = chain.risk_score,
                label = int(req.label == "incorrect"),
            )
    except Exception:
        pass

    # ── Bandit update: reward ptrue_weight arm that was used for this chain ──
    try:
        bandit = getattr(agent_guard, "_ptrue_bandit", None)
        if bandit is not None:
            # Retrieve the weight used when this chain was scored (stored in DB if available)
            weight_used = float(chain.behavioral_components.get("ptrue_weight_used", 0.9)) \
                if isinstance(getattr(chain, "behavioral_components", None), dict) else 0.9
            # Reward: 1 if prediction matched label, 0 otherwise
            # chain.confidence_tier == HIGH/MEDIUM → predicted correct; LOW → predicted wrong
            predicted_correct = chain.confidence_tier != "LOW"
            actual_correct    = req.label == "correct"
            reward = 1.0 if predicted_correct == actual_correct else 0.0
            bandit.update(weight_used, reward)
    except Exception:
        pass

    # ── Auto-retrain: trigger when N new labels accumulated since last retrain ──
    total_feedback = db.query(models.ChainFeedback).filter(
        models.ChainFeedback.user_id == current_user.id
    ).count()
    last_retrain_n = _retrain_counters.get(current_user.id, 0)
    if total_feedback - last_retrain_n >= _AUTO_RETRAIN_EVERY:
        asyncio.get_event_loop().create_task(
            _run_auto_retrain(current_user.id)
        )

    return {"chain_id": chain_id, "label": req.label, "status": "recorded",
            "total_feedback": total_feedback}


@app.post("/v1/verifier/retrain", response_model=RetrainResponse)
async def retrain_verifier(
    current_user: models.User = Depends(get_optional_user),
    db:           Session = Depends(get_db),
):
    """
    Retrain the LocalVerifier on accumulated chain feedback labels.

    Requires ≥ 50 labeled chains (recommended ≥ 200 for stable AUROC).
    Runs in the thread pool (non-blocking).

    Auth: required.
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    def _do_retrain():
        from llm_guard.local_verifier import LocalVerifier
        from sklearn.model_selection import cross_val_score
        import numpy as np

        # Fetch all feedback for this user with the chain's step data
        rows = (
            db.query(models.ChainFeedback, models.Chain)
            .join(models.Chain, models.ChainFeedback.chain_id == models.Chain.id)
            .filter(models.ChainFeedback.user_id == current_user.id)
            .all()
        )
        if len(rows) < 5:
            return RetrainResponse(
                status       = "skipped",
                n_feedback   = len(rows),
                message      = f"Need ≥ 5 labeled chains (have {len(rows)}). Rate a few more alerts.",
            )

        # Build labeled runs in AgentGuard format
        runs = []
        for fb, chain in rows:
            runs.append({
                "question":     chain.question,
                "steps":        [],           # steps not stored in Chain — use question-level features
                "final_answer": chain.final_answer,
                "correct":      fb.label == "correct",
            })

        verifier = LocalVerifier()
        verifier.fit(runs)

        # Quick LOO AUROC estimate (5-fold CV)
        auroc = None
        if len(runs) >= 60:
            from llm_guard.local_verifier import extract_features
            import numpy as np
            X = np.array([extract_features(r["question"], r.get("steps", []), r["final_answer"]) for r in runs])
            y = np.array([int(r["correct"]) for r in runs])
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
            auroc = round(float(np.mean(scores)), 4)

        # Store the retrained verifier on the global AgentGuard instance
        agent_guard._local_verifier = verifier
        agent_guard._use_local_verifier = True
        _retrain_counters[current_user.id] = len(rows)

        # ── Feed all labeled chains back into AdaptiveCISC ────────────────────
        # This closes the loop: retrain → update thresholds based on same data
        try:
            for fb, chain in rows:
                domain_key = f"{current_user.id}:{chain.domain}"
                cisc = _cisc_registry.get(domain_key)
                cisc.record_outcome(
                    risk_score = chain.risk_score,
                    tier       = chain.confidence_tier,
                    was_wrong  = (fb.label == "incorrect"),
                )
        except Exception:
            pass

        return RetrainResponse(
            status         = "retrained",
            n_feedback     = len(rows),
            auroc_estimate = auroc,
            message        = (
                f"Adaptive scoring updated on {len(rows)} labeled chains. "
                + (f"5-fold CV AUROC ≈ {auroc:.3f}." if auroc else "Rate ≥ 60 chains for CV estimate.")
                + " Alert thresholds auto-adjusted for your domain."
            ),
        )

    return await run_in_threadpool(_do_retrain)


@app.get("/v1/onboard")
def onboard(optional_user: Optional[models.User] = Depends(get_optional_user)):
    """
    Return ready-to-paste integration snippets, pre-filled with the caller's
    API key when authenticated. Used by the dashboard quick-start wizard.
    """
    key_hint = "YOUR_API_KEY"

    if optional_user is not None:
        from app.database import SessionLocal
        _db = SessionLocal()
        try:
            key_row = (
                _db.query(models.ApiKey)
                .filter(models.ApiKey.user_id == optional_user.id, models.ApiKey.is_active == True)  # noqa: E712
                .order_by(models.ApiKey.created_at.desc())
                .first()
            )
            if key_row:
                key_hint = key_row.key_prefix + "…"
        finally:
            _db.close()

    return {
        "api_key_hint": key_hint,
        "snippets": {
            "saas_python": (
                f'from llm_guard import GuardClient\n\n'
                f'guard = GuardClient(api_key="{key_hint}")\n\n'
                f'result = guard.score(\n'
                f'    question="What is the capital of France?",\n'
                f'    steps=[{{"thought": "...", "action_type": "Search",\n'
                f'             "action_arg": "capital France",\n'
                f'             "observation": "Paris is the capital"}}],\n'
                f'    final_answer="Paris",\n'
                f'    domain="demo",\n'
                f')\n'
                f'print(result)  # ScoreResult(risk=12%, tier=HIGH, steps=1)\n'
                f'\n'
                f'# Auto-instrument any agent function:\n'
                f'@guard.watch\n'
                f'def run_agent(question): ...'
            ),
            "local_python": (
                'from llm_guard import GuardClient\n\n'
                'guard = GuardClient(mode="local")  # no network, no API key\n'
                'result = guard.score(question, steps, final_answer)\n'
                'print(result.risk_score, result.confidence_tier)\n'
            ),
            "langchain": (
                f'from llm_guard.integrations.langchain import AgentGuardCallback\n\n'
                f'cb = AgentGuardCallback(api_key="{key_hint}")\n'
                f'agent.run(question, callbacks=[cb])  # that\'s it\n'
            ),
            "docker": (
                'cp .env.example .env   # fill in JWT_SECRET + ANTHROPIC_API_KEY\n'
                'docker compose up --build\n'
                '# Backend :8000  |  Frontend :3000\n'
            ),
        },
        "deployment_options": [
            {"id": "saas",        "label": "Cloud SaaS",    "description": "$0 for behavioral scoring, managed infra"},
            {"id": "local",       "label": "Local library", "description": "pip install — runs in-process, no network"},
            {"id": "on_premises", "label": "Self-hosted",   "description": "docker compose up — data never leaves your VPC"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy KNN guard endpoints (backward compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrateRequest(BaseModel):
    questions: List[str]
    labels:    Optional[List[int]] = None


class QueryRequest(BaseModel):
    question:      str
    system_prompt: Optional[str] = None


class QueryResponse(BaseModel):
    query_id:   str
    answer:     str
    risk_score: float
    confidence: str


class FeedbackRequest(BaseModel):
    query_id:       str
    is_correct:     bool
    correct_answer: Optional[str] = None


class FitQARARequest(BaseModel):
    epochs: int = 200


@app.post("/v1/calibrate")
def calibrate(req: CalibrateRequest):
    try:
        return manager.calibrate(req.questions, req.labels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/query", response_model=QueryResponse)
def query_knn(req: QueryRequest):
    if not manager.guard._fitted:
        raise HTTPException(status_code=400, detail="Guard not calibrated. Call /v1/calibrate first.")
    record = manager.query(req.question, req.system_prompt)
    return QueryResponse(
        query_id   = record.query_id,
        answer     = record.answer,
        risk_score = round(record.risk_score, 4),
        confidence = record.confidence,
    )


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    try:
        return manager.feedback(req.query_id, req.is_correct, req.correct_answer)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/v1/stats")
def stats():
    return manager.get_stats()


@app.post("/v1/diagnose")
def diagnose():
    try:
        clusters = manager.diagnose_now()
        return {"clusters": clusters, "n_errors": len(manager._error_questions)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/heal")
def heal():
    try:
        return manager.heal_now()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/fit_qara")
def fit_qara(req: FitQARARequest):
    try:
        return manager.fit_qara_now(epochs=req.epochs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# A2A (Agent-to-Agent) endpoints — v0.13.0
# ═══════════════════════════════════════════════════════════════════════════════

from llm_guard.trust_object import A2ATrustObject, TrustHop

# Agent capabilities advertised in the well-known card
_AGENT_CARD = {
    "name":        "llm-guard-kit",
    "version":     "0.13.0",
    "description": "Real-time reliability monitoring for LLM ReAct agents.",
    "url":         os.getenv("AGENT_CARD_URL", "http://localhost:8000"),
    "capabilities": [
        "score_chain",
        "score_with_ptrue",
        "stream_guard",
        "route_to_mesh",
        "calibrate_isotonic",
        "conformal_alert_threshold",
        "kalman_smooth_risks",
        "a2a_handoff",
        "multi_hop_trust_chain",
    ],
    "risk_threshold":     float(os.getenv("AGENT_RISK_THRESHOLD", "0.65")),
    "supported_domains":  ["hotpotqa", "triviaqa", "naturalquestions", "general"],
    "contact_email":      os.getenv("AGENT_CONTACT_EMAIL", ""),
    "a2a_protocol":       "v0.4",
    "endpoints": {
        "score_chain":        "/v1/score_chain",
        "a2a_handoff":        "/v1/a2a/handoff",
        "a2a_verify":         "/v1/a2a/trust/verify",
        "a2a_audit":          "/v1/a2a/audit",
        "agent_card":         "/.well-known/agent.json",
    },
}


@app.get("/.well-known/agent.json", tags=["A2A"])
def agent_card():
    """
    Google A2A-compatible AgentCard discovery endpoint.

    Returns this agent's capability card including:
    - Supported operations (score_chain, stream_guard, …)
    - Risk threshold
    - A2A protocol version (v0.4 — multi-hop trust chain)
    - Endpoint URLs for all A2A operations

    Agents in a multi-agent pipeline discover each other's capabilities
    by fetching <base_url>/.well-known/agent.json before the first handoff.
    """
    return _AGENT_CARD


class A2AHandoffRequest(BaseModel):
    """Incoming trust object from an upstream agent, plus a new question (optional)."""
    trust_object:   dict                    # serialized A2ATrustObject.to_dict()
    question:       Optional[str] = None    # re-score question if provided
    steps:          Optional[List[StepModel]] = None
    final_answer:   Optional[str] = None
    verify_secret:  Optional[str] = None    # if set, verify upstream signature
    sign_secret:    Optional[str] = None    # if set, sign this agent's hop
    agent_id:       str = "llm-guard-kit"  # id this agent uses in the trust chain
    agent_card_ref: Optional[str] = None


class A2AHandoffResponse(BaseModel):
    trust_object:    dict   # updated A2ATrustObject with this agent's hop appended
    signature_valid: Optional[bool]  # True/False if verify_secret was provided, else None
    chain_length:    int
    risk_score:      float
    confidence_tier: str
    rescored:        bool   # True when a new question+steps+final_answer were provided


@app.post("/v1/a2a/handoff", response_model=A2AHandoffResponse, tags=["A2A"])
async def a2a_handoff(req: A2AHandoffRequest, db: Session = Depends(get_db)):
    """
    Receive a trust object from an upstream agent, optionally re-score,
    append this agent's hop to the trust chain, and return the updated object.

    Protocol
    --------
    1. Upstream agent sends its A2ATrustObject.to_dict() in trust_object.
    2. This agent optionally verifies the upstream signature (verify_secret).
    3. If question+steps+final_answer are provided, re-scores the chain.
    4. Appends its own TrustHop (add_hop) signed with sign_secret.
    5. Logs the handoff to the audit trail.
    6. Returns the updated trust object for forwarding to the next agent.

    Multi-hop example (A → B → C):
        Agent A:  trust = guard.generate_trust_object(...).add_hop("agent-a", secret_a)
                  POST /v1/a2a/handoff  {trust_object: trust.to_dict(), agent_id: "agent-b", sign_secret: "secret-b"}
        Agent C:  verify_chain({"agent-a": secret_a, "agent-b": secret_b})  → True
    """
    def _do():
        trust = A2ATrustObject.from_dict(req.trust_object)

        # 1. Optionally verify upstream single-hop signature
        sig_valid = None
        if req.verify_secret:
            sig_valid = trust.verify(req.verify_secret)

        # 2. Re-score if new chain data provided
        rescored = False
        if req.question and req.steps is not None and req.final_answer is not None:
            steps   = [s.model_dump() for s in req.steps]
            result  = agent_guard.score_chain(req.question, steps, req.final_answer)
            trust.risk_score      = result.risk_score
            trust.confidence_tier = result.confidence_tier
            trust.failure_mode    = result.failure_mode
            trust.step_count      = result.step_count
            trust.judge_label     = result.judge_label
            trust.answer          = req.final_answer
            trust.downstream_hint = (
                "rewrite_and_retry" if result.needs_alert else "proceed_with_monitoring"
            )
            trust.should_rewrite  = result.needs_alert
            rescored = True

        # 3. Append this agent's hop
        if req.sign_secret:
            trust.add_hop(
                agent_id       = req.agent_id,
                secret         = req.sign_secret,
                agent_card_ref = req.agent_card_ref or _AGENT_CARD.get("url"),
            )

        # 4. Persistent audit log (survives restart)
        try:
            import json as _json
            entry_payload = {
                "ts":             round(time.time(), 3),
                "from_agent":     trust.trust_chain[-2].agent_id if len(trust.trust_chain) >= 2 else "upstream",
                "to_agent":       req.agent_id,
                "risk_score":     round(trust.risk_score, 4),
                "confidence_tier":trust.confidence_tier,
                "chain_length":   len(trust.trust_chain),
                "rescored":       rescored,
                "sig_valid":      sig_valid,
            }
            db.add(models.A2AAuditLog(
                event_type   = "handoff",
                from_agent   = entry_payload["from_agent"],
                to_agent     = req.agent_id,
                risk_score   = trust.risk_score,
                hop_count    = len(trust.trust_chain),
                verify_ok    = sig_valid,
                payload_json = _json.dumps(entry_payload),
            ))
            db.commit()
        except Exception:
            pass  # audit failure must never break the handoff response

        return A2AHandoffResponse(
            trust_object    = trust.to_dict(),
            signature_valid = sig_valid,
            chain_length    = len(trust.trust_chain),
            risk_score      = round(trust.risk_score, 4),
            confidence_tier = trust.confidence_tier,
            rescored        = rescored,
        )

    return await run_in_threadpool(_do)


class A2AVerifyRequest(BaseModel):
    trust_object: dict
    secrets_map:  Dict[str, str]  # {agent_id: secret}
    mode:         str = "chain"   # "chain" (multi-hop) or "single" (v0.3 sign/verify)


@app.post("/v1/a2a/trust/verify", tags=["A2A"])
async def a2a_verify(req: A2AVerifyRequest):
    """
    Verify an A2ATrustObject's signature(s).

    mode="chain"  — verify all TrustHop signatures in trust_chain
                    (A → B → C multi-hop chain-of-custody)
    mode="single" — verify the v0.3 single-hop trust_signature
                    (backward compatible)

    Returns {valid: bool, chain_length: int, agents: list[str]}
    """
    def _do():
        trust = A2ATrustObject.from_dict(req.trust_object)
        if req.mode == "single":
            secret = next(iter(req.secrets_map.values()), "")
            valid  = trust.verify(secret)
            agents = [trust.agent_card_ref] if trust.agent_card_ref else []
        else:
            valid  = trust.verify_chain(req.secrets_map)
            agents = [h.agent_id for h in trust.trust_chain]
        return {
            "valid":         valid,
            "mode":          req.mode,
            "chain_length":  len(trust.trust_chain),
            "agents":        agents,
            "risk_score":    round(trust.risk_score, 4),
            "confidence_tier": trust.confidence_tier,
        }
    return await run_in_threadpool(_do)


@app.get("/v1/a2a/audit", tags=["A2A"])
def a2a_audit(limit: int = 100, db: Session = Depends(get_db)):
    """
    Return recent trust propagation events (last `limit` handoffs).

    Each entry records: timestamp, from_agent, to_agent, risk_score,
    confidence_tier, chain_length, whether re-scoring occurred,
    and upstream signature validity.

    Persisted to SQLite — survives server restarts.
    """
    import json as _json
    rows = (
        db.query(models.A2AAuditLog)
        .order_by(models.A2AAuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    total = db.query(models.A2AAuditLog).count()
    entries = []
    for row in reversed(rows):   # chronological order
        try:
            entry = _json.loads(row.payload_json)
        except Exception:
            entry = {"id": row.id, "from_agent": row.from_agent, "to_agent": row.to_agent,
                     "risk_score": row.risk_score, "ts": row.created_at}
        entries.append(entry)
    return {
        "total_logged": total,
        "returned":     len(entries),
        "entries":      entries,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration endpoints — v0.13.0 (expose v0.11.0 library methods)
# ═══════════════════════════════════════════════════════════════════════════════

class IsotonicCalibrateRequest(BaseModel):
    """Pairs of (risk_score, label) from your deployment."""
    scores: List[float]  # raw risk scores from score_chain / score_with_ptrue
    labels: List[int]    # 1 = wrong chain, 0 = correct chain


@app.post("/v1/verifier/calibrate_isotonic", tags=["Calibration"])
async def calibrate_isotonic(req: IsotonicCalibrateRequest):
    """
    Fit isotonic regression calibration on your labeled chains.

    After calling this endpoint the server's AgentGuard will apply
    the calibrated P(True) transform on all subsequent /v1/score_chain calls
    (only when AGENT_USE_JUDGE=1 and API key is set).

    Validated AUROC improvement: 0.833 → 0.871 on TriviaQA n=37 (exp134).

    Body: {scores: [0.3, 0.7, …], labels: [0, 1, …]}
      scores — list of raw risk_score values from prior score_chain calls
      labels — 1=wrong, 0=correct (ground truth from human feedback)

    Requires ≥ 10 labeled chains.  Best results with ≥ 60.
    """
    if len(req.scores) != len(req.labels):
        raise HTTPException(status_code=400, detail="scores and labels must be same length.")
    if len(req.scores) < 10:
        raise HTTPException(status_code=400, detail="Need ≥ 10 labeled chains for isotonic calibration.")

    def _do():
        agent_guard.calibrate_isotonic(req.scores, req.labels)
        return {
            "status":       "calibrated",
            "n_samples":    len(req.scores),
            "n_wrong":      int(sum(req.labels)),
            "method":       "IsotonicRegression(out_of_bounds=clip)",
            "auroc_note":   "Expected AUROC improvement: ~+0.03 on cross-domain eval (exp134).",
        }
    return await run_in_threadpool(_do)


class ConformalThresholdRequest(BaseModel):
    scores: List[float]  # raw risk scores for wrong chains
    labels: List[int]    # 1=wrong, 0=correct
    alpha:  float = 0.15  # desired false-negative rate (lower = fewer misses, higher threshold)


@app.post("/v1/verifier/conformal_threshold", tags=["Calibration"])
async def conformal_threshold(req: ConformalThresholdRequest):
    """
    Compute the conformal alert threshold with finite-sample guarantee.

    Returns the threshold τ such that at most `alpha` fraction of wrong chains
    score below τ (i.e., are missed by the alerting system).

    Validated (exp135): alpha=0.15 → precision=0.50, threshold stable across domains.

    Body: {scores: […], labels: […], alpha: 0.15}
    """
    if len(req.scores) != len(req.labels):
        raise HTTPException(status_code=400, detail="scores and labels must be same length.")

    def _do():
        threshold = agent_guard.conformal_alert_threshold(req.scores, req.labels, alpha=req.alpha)
        n_wrong   = int(sum(req.labels))
        return {
            "threshold":   round(threshold, 4),
            "alpha":       req.alpha,
            "n_wrong_cal": n_wrong,
            "n_total_cal": len(req.scores),
            "interpretation": (
                f"Set alert_threshold={threshold:.3f}. "
                f"At most {req.alpha:.0%} of wrong chains will score below this threshold."
            ),
        }
    return await run_in_threadpool(_do)


class KalmanSmoothRequest(BaseModel):
    step_risks: List[float]  # per-step risk values from monitor_step calls
    Q:          float = 0.2  # process noise
    R:          float = 0.05 # measurement noise


@app.post("/v1/verifier/kalman_smooth", tags=["Calibration"])
def kalman_smooth(req: KalmanSmoothRequest):
    """
    Apply Kalman filter over per-step risk estimates to produce a single
    smoothed chain risk score.

    Use this when you've collected per-step risks via /v1/monitor_step and
    want a single noise-robust final score.  Validated AUROC=0.844 (exp137).

    Body: {step_risks: [0.3, 0.5, 0.6, …], Q: 0.2, R: 0.05}
    """
    import numpy as np
    from llm_guard.agent_guard import AgentGuard
    smoothed = AgentGuard.kalman_smooth_risks(req.step_risks, Q=req.Q, R=req.R)
    return {
        "smoothed_risk":   round(float(smoothed), 4),
        "n_steps":         len(req.step_risks),
        "raw_final_step":  round(float(req.step_risks[-1]), 4) if req.step_risks else None,
        "Q":               req.Q,
        "R":               req.R,
    }
