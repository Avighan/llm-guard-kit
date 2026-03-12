"""
/user/* endpoints — chains, analytics, API keys, orgs, guardrails,
workflows, pipelines, usage, onboarding, configs.

All endpoints require Bearer authentication (JWT or sk_ API key).
Multi-tenant isolation: every query filters by current_user.id.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.auth_utils import (
    get_current_user,
    generate_api_key,
    encrypt_config, decrypt_config,
    maybe_reset_monthly_counter,
)

router = APIRouter(prefix="/user", tags=["user"])

FREE_CHAIN_LIMIT = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Chains & Analytics
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/chains/recent")
def recent_chains(
    limit: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chains = (
        db.query(models.Chain)
        .filter(models.Chain.user_id == current_user.id)
        .order_by(models.Chain.timestamp.desc())
        .limit(limit)
        .all()
    )
    # Fetch any feedback the user has submitted for these chains
    chain_ids = [c.id for c in chains]
    feedback_map: dict = {}
    if chain_ids:
        fb_rows = (
            db.query(models.ChainFeedback)
            .filter(
                models.ChainFeedback.chain_id.in_(chain_ids),
                models.ChainFeedback.user_id == current_user.id,
            )
            .all()
        )
        feedback_map = {fb.chain_id: fb.label for fb in fb_rows}

    return [
        {
            "id":              c.id,
            "domain":          c.domain,
            "question":        c.question[:200] if c.question else "",
            "final_answer":    c.final_answer[:100] if c.final_answer else "",
            "risk_score":      c.risk_score,
            "confidence_tier": c.confidence_tier,
            "needs_alert":     c.needs_alert,
            "failure_mode":    c.failure_mode,
            "n_steps":         c.n_steps,
            "timestamp":       c.timestamp,
            "feedback":        feedback_map.get(c.id),  # "correct" | "incorrect" | None
        }
        for c in chains
    ]


@router.get("/domains")
def domains(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Single aggregation query — SQLite-compatible (no FILTER clause)
    from sqlalchemy import case
    rows = (
        db.query(
            models.Chain.domain,
            func.count(models.Chain.id).label("chain_count"),
            func.max(models.Chain.timestamp).label("last_seen"),
            func.avg(models.Chain.risk_score).label("avg_risk"),
            func.sum(
                case((models.Chain.needs_alert == True, 1), else_=0)  # noqa: E712
            ).label("n_alerts"),
        )
        .filter(models.Chain.user_id == current_user.id)
        .group_by(models.Chain.domain)
        .order_by(func.max(models.Chain.timestamp).desc())
        .all()
    )
    return [
        {
            "domain":      r.domain,
            "chain_count": r.chain_count,
            "last_seen":   r.last_seen or 0,
            "avg_risk":    round(float(r.avg_risk), 4) if r.avg_risk is not None else None,
            "n_alerts":    int(r.n_alerts or 0),
        }
        for r in rows
    ]


@router.get("/timeseries")
def timeseries(
    days:   int    = Query(30, ge=1, le=365),
    bucket: str    = Query("day"),
    domain: str    = Query(""),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    since = int(time.time()) - days * 86400

    q = db.query(models.Chain).filter(
        models.Chain.user_id  == current_user.id,
        models.Chain.timestamp >= since,
    )
    if domain:
        q = q.filter(models.Chain.domain == domain)
    chains = q.order_by(models.Chain.timestamp.asc()).all()

    # Bucket sizes in seconds
    bucket_s = {"hour": 3600, "day": 86400, "week": 604800}.get(bucket, 86400)

    buckets: Dict[int, Dict[str, Any]] = {}
    for c in chains:
        key = (c.timestamp // bucket_s) * bucket_s
        if key not in buckets:
            buckets[key] = {"timestamp": key, "risks": [], "n_chains": 0, "n_alerts": 0}
        buckets[key]["risks"].append(c.risk_score)
        buckets[key]["n_chains"] += 1
        if c.needs_alert:
            buckets[key]["n_alerts"] += 1

    result = []
    for ts in sorted(buckets):
        b = buckets[ts]
        risks = b["risks"]
        result.append({
            "timestamp": ts,
            "avg_risk":  round(sum(risks) / len(risks), 4) if risks else 0.0,
            "n_chains":  b["n_chains"],
            "n_alerts":  b["n_alerts"],
        })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# API Keys
# ═══════════════════════════════════════════════════════════════════════════════

class CreateKeyRequest(BaseModel):
    domain_prefix: str = ""


@router.get("/keys")
def list_keys(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    keys = (
        db.query(models.ApiKey)
        .filter(models.ApiKey.user_id == current_user.id, models.ApiKey.is_active == True)  # noqa: E712
        .order_by(models.ApiKey.created_at.desc())
        .all()
    )
    return [
        {
            "key_prefix":    k.key_prefix,
            "domain_prefix": k.domain_prefix,
            "created_at":    k.created_at,
            "request_count": k.request_count,
            "is_active":     k.is_active,
        }
        for k in keys
    ]


@router.post("/keys")
def create_key(
    req: CreateKeyRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    full_key, prefix, key_hash = generate_api_key()
    key = models.ApiKey(
        user_id       = current_user.id,
        key_prefix    = prefix,
        key_hash      = key_hash,
        domain_prefix = req.domain_prefix,
    )
    db.add(key)
    db.commit()
    return {"api_key": full_key, "key_prefix": prefix}  # full_key shown ONCE


@router.delete("/keys/{prefix}")
def revoke_key(
    prefix: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key = db.query(models.ApiKey).filter(
        models.ApiKey.key_prefix == prefix,
        models.ApiKey.user_id    == current_user.id,
    ).first()
    if key:
        key.is_active = False
        db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Organizations
# ═══════════════════════════════════════════════════════════════════════════════

class CreateOrgRequest(BaseModel):
    name: str


class InviteRequest(BaseModel):
    email: str


class AcceptInviteRequest(BaseModel):
    token: str


@router.get("/orgs")
def list_orgs(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    memberships = (
        db.query(models.OrgMember)
        .filter(models.OrgMember.user_id == current_user.id)
        .all()
    )
    result = []
    for m in memberships:
        org = db.query(models.Org).filter(models.Org.id == m.org_id).first()
        if org:
            result.append({
                "id":         org.id,
                "name":       org.name,
                "owner_id":   org.owner_id,
                "role":       m.role,
                "created_at": org.created_at,
            })
    return result


@router.post("/orgs")
def create_org(
    req: CreateOrgRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    org = models.Org(name=req.name, owner_id=current_user.id)
    db.add(org)
    db.flush()
    member = models.OrgMember(org_id=org.id, user_id=current_user.id, role="owner")
    db.add(member)
    db.commit()
    return {"id": org.id, "name": org.name, "owner_id": org.owner_id, "role": "owner", "created_at": org.created_at}


@router.get("/orgs/{org_id}/members")
def list_members(
    org_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Only members of the org may list other members
    membership = db.query(models.OrgMember).filter(
        models.OrgMember.org_id  == org_id,
        models.OrgMember.user_id == current_user.id,
    ).first()
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this org")

    members = db.query(models.OrgMember).filter(models.OrgMember.org_id == org_id).all()
    result = []
    for m in members:
        u = db.query(models.User).filter(models.User.id == m.user_id).first()
        if u:
            result.append({"user_id": u.id, "email": u.email, "name": u.name, "role": m.role, "joined_at": m.joined_at})
    return result


@router.post("/orgs/{org_id}/invite")
def invite_member(
    org_id: int,
    req: InviteRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    membership = db.query(models.OrgMember).filter(
        models.OrgMember.org_id  == org_id,
        models.OrgMember.user_id == current_user.id,
        models.OrgMember.role.in_(["owner", "admin"]),
    ).first()
    if not membership:
        raise HTTPException(status_code=403, detail="Only org owners/admins can invite")

    token = uuid.uuid4().hex
    invite = models.InviteToken(
        org_id     = org_id,
        inviter_id = current_user.id,
        email      = req.email.lower().strip(),
        token      = token,
        expires_at = int(time.time()) + 7 * 86400,  # 7 days
    )
    db.add(invite)
    db.commit()
    return {"token": token, "invited": True}


@router.post("/orgs/accept")
def accept_invite(
    req: AcceptInviteRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    invite = db.query(models.InviteToken).filter(
        models.InviteToken.token    == req.token,
        models.InviteToken.accepted == False,  # noqa: E712
    ).first()
    if invite is None:
        raise HTTPException(status_code=404, detail="Invalid or already-used invite token")
    if invite.expires_at < int(time.time()):
        raise HTTPException(status_code=410, detail="Invite token expired")

    # Don't re-add if already a member
    existing = db.query(models.OrgMember).filter(
        models.OrgMember.org_id  == invite.org_id,
        models.OrgMember.user_id == current_user.id,
    ).first()
    if not existing:
        db.add(models.OrgMember(org_id=invite.org_id, user_id=current_user.id, role="member"))

    invite.accepted = True
    db.commit()
    return {"joined": True, "org_id": invite.org_id}


@router.delete("/orgs/{org_id}/members/{member_id}")
def remove_member(
    org_id:    int,
    member_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    caller = db.query(models.OrgMember).filter(
        models.OrgMember.org_id  == org_id,
        models.OrgMember.user_id == current_user.id,
        models.OrgMember.role.in_(["owner", "admin"]),
    ).first()
    if not caller:
        raise HTTPException(status_code=403, detail="Only org owners/admins can remove members")

    target = db.query(models.OrgMember).filter(
        models.OrgMember.org_id  == org_id,
        models.OrgMember.user_id == member_id,
    ).first()
    if target and target.role != "owner":
        db.delete(target)
        db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrails
# ═══════════════════════════════════════════════════════════════════════════════

class GuardrailRequest(BaseModel):
    domain:        str   = "*"
    threshold:     float = 0.70
    action:        str   = "alert"
    webhook_url:   str   = ""
    slack_webhook: str   = ""
    enabled:       bool  = True
    name:          str   = ""   # ignored, kept for frontend compat


@router.get("/guardrails")
def list_guardrails(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(models.Guardrail)
        .filter(models.Guardrail.user_id == current_user.id)
        .order_by(models.Guardrail.created_at.desc())
        .all()
    )
    return [
        {
            "id":            g.id,
            "domain":        g.domain,
            "threshold":     g.threshold,
            "action":        g.action,
            "webhook_url":   g.webhook_url,
            "slack_webhook": g.slack_webhook,
            "enabled":       g.enabled,
            "created_at":    g.created_at,
        }
        for g in rows
    ]


@router.post("/guardrails")
def upsert_guardrail(
    req: GuardrailRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = db.query(models.Guardrail).filter(
        models.Guardrail.user_id == current_user.id,
        models.Guardrail.domain  == req.domain,
    ).first()

    if existing:
        existing.threshold     = req.threshold
        existing.action        = req.action
        existing.webhook_url   = req.webhook_url
        existing.slack_webhook = req.slack_webhook
        existing.enabled       = req.enabled
        db.commit()
        g = existing
    else:
        g = models.Guardrail(
            user_id       = current_user.id,
            domain        = req.domain,
            threshold     = req.threshold,
            action        = req.action,
            webhook_url   = req.webhook_url,
            slack_webhook = req.slack_webhook,
            enabled       = req.enabled,
        )
        db.add(g)
        db.commit()
        db.refresh(g)

    return {"id": g.id, "domain": g.domain, "threshold": g.threshold, "action": g.action,
            "webhook_url": g.webhook_url, "slack_webhook": g.slack_webhook,
            "enabled": g.enabled, "created_at": g.created_at}


@router.delete("/guardrails/{domain:path}")
def delete_guardrail(
    domain: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    g = db.query(models.Guardrail).filter(
        models.Guardrail.user_id == current_user.id,
        models.Guardrail.domain  == domain,
    ).first()
    if g:
        db.delete(g)
        db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Workflows
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowRequest(BaseModel):
    id:      Optional[int] = None
    name:    str
    nodes:   List[Any] = []
    edges:   List[Any] = []
    enabled: bool = True


class ToggleRequest(BaseModel):
    enabled: bool


@router.get("/workflows/{workflow_id}")
def get_workflow(
    workflow_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    w = db.query(models.Workflow).filter(
        models.Workflow.id == workflow_id,
        models.Workflow.user_id == current_user.id,
    ).first()
    if not w:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"id": w.id, "name": w.name, "nodes": json.loads(w.nodes or "[]"),
            "edges": json.loads(w.edges or "[]"), "enabled": w.enabled,
            "created_at": w.created_at, "updated_at": w.updated_at}


@router.get("/workflows")
def list_workflows(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(models.Workflow).filter(models.Workflow.user_id == current_user.id).order_by(models.Workflow.updated_at.desc()).all()
    return [{"id": w.id, "name": w.name, "nodes": json.loads(w.nodes or "[]"),
             "edges": json.loads(w.edges or "[]"), "enabled": w.enabled,
             "created_at": w.created_at, "updated_at": w.updated_at} for w in rows]


@router.post("/workflows")
def save_workflow(
    req: WorkflowRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = int(time.time())
    if req.id:
        w = db.query(models.Workflow).filter(models.Workflow.id == req.id, models.Workflow.user_id == current_user.id).first()
        if w:
            w.name       = req.name
            w.nodes      = json.dumps(req.nodes)
            w.edges      = json.dumps(req.edges)
            w.enabled    = req.enabled
            w.updated_at = now
            db.commit()
            db.refresh(w)
            return {"id": w.id, "name": w.name, "enabled": w.enabled, "updated_at": w.updated_at}

    w = models.Workflow(user_id=current_user.id, name=req.name,
                        nodes=json.dumps(req.nodes), edges=json.dumps(req.edges),
                        enabled=req.enabled, created_at=now, updated_at=now)
    db.add(w)
    db.commit()
    db.refresh(w)
    return {"id": w.id, "name": w.name, "enabled": w.enabled, "updated_at": w.updated_at}


@router.patch("/workflows/{workflow_id}/toggle")
def toggle_workflow(
    workflow_id: int,
    req: ToggleRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    w = db.query(models.Workflow).filter(models.Workflow.id == workflow_id, models.Workflow.user_id == current_user.id).first()
    if w:
        w.enabled = req.enabled
        db.commit()
    return {"enabled": req.enabled}


@router.delete("/workflows/{workflow_id}")
def delete_workflow(
    workflow_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    w = db.query(models.Workflow).filter(models.Workflow.id == workflow_id, models.Workflow.user_id == current_user.id).first()
    if w:
        db.delete(w)
        db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipelines
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineRequest(BaseModel):
    id:      Optional[int] = None
    name:    str
    nodes:   List[Any] = []
    edges:   List[Any] = []
    enabled: bool = True


@router.get("/pipelines/{pipeline_id}")
def get_pipeline(
    pipeline_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = db.query(models.Pipeline).filter(
        models.Pipeline.id == pipeline_id,
        models.Pipeline.user_id == current_user.id,
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return {"id": p.id, "name": p.name, "nodes": json.loads(p.nodes or "[]"),
            "edges": json.loads(p.edges or "[]"), "enabled": p.enabled,
            "created_at": p.created_at, "updated_at": p.updated_at}


@router.get("/pipelines")
def list_pipelines(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(models.Pipeline).filter(models.Pipeline.user_id == current_user.id).order_by(models.Pipeline.updated_at.desc()).all()
    return [{"id": p.id, "name": p.name, "enabled": p.enabled, "created_at": p.created_at, "updated_at": p.updated_at} for p in rows]


@router.post("/pipelines")
def save_pipeline(
    req: PipelineRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = int(time.time())
    if req.id:
        p = db.query(models.Pipeline).filter(models.Pipeline.id == req.id, models.Pipeline.user_id == current_user.id).first()
        if p:
            p.name = req.name; p.nodes = json.dumps(req.nodes); p.edges = json.dumps(req.edges)
            p.enabled = req.enabled; p.updated_at = now
            db.commit(); db.refresh(p)
            return {"id": p.id, "name": p.name, "enabled": p.enabled, "updated_at": p.updated_at}
    p = models.Pipeline(user_id=current_user.id, name=req.name,
                        nodes=json.dumps(req.nodes), edges=json.dumps(req.edges),
                        enabled=req.enabled, created_at=now, updated_at=now)
    db.add(p); db.commit(); db.refresh(p)
    return {"id": p.id, "name": p.name, "enabled": p.enabled, "updated_at": p.updated_at}


@router.get("/pipelines/{pipeline_id}/runs")
def pipeline_runs(
    pipeline_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = db.query(models.Pipeline).filter(models.Pipeline.id == pipeline_id, models.Pipeline.user_id == current_user.id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    runs = db.query(models.PipelineRun).filter(models.PipelineRun.pipeline_id == pipeline_id).order_by(models.PipelineRun.created_at.desc()).limit(50).all()
    return [{"run_id": r.run_id, "result": r.result, "risk_score": r.risk_score, "status": r.status, "error": r.error, "created_at": r.created_at} for r in runs]


@router.patch("/pipelines/{pipeline_id}/toggle")
def toggle_pipeline(
    pipeline_id: int,
    req: ToggleRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = db.query(models.Pipeline).filter(models.Pipeline.id == pipeline_id, models.Pipeline.user_id == current_user.id).first()
    if p:
        p.enabled = req.enabled; db.commit()
    return {"enabled": req.enabled}


@router.delete("/pipelines/{pipeline_id}")
def delete_pipeline(
    pipeline_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = db.query(models.Pipeline).filter(models.Pipeline.id == pipeline_id, models.Pipeline.user_id == current_user.id).first()
    if p:
        db.delete(p); db.commit()
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Usage, Onboarding, Configs
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/usage")
def usage(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    maybe_reset_monthly_counter(current_user, db)
    limit = None if current_user.plan == "pro" else FREE_CHAIN_LIMIT
    pct   = round(current_user.chains_this_month / FREE_CHAIN_LIMIT * 100, 1) if limit else None
    return {"plan": current_user.plan, "used": current_user.chains_this_month, "limit": limit, "pct": pct}


@router.get("/onboarding")
def get_onboarding(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ob = db.query(models.OnboardingState).filter(models.OnboardingState.user_id == current_user.id).first()
    if not ob:
        ob = models.OnboardingState(user_id=current_user.id)
        db.add(ob); db.commit(); db.refresh(ob)
    return {"done": ob.done, "step": ob.step}


class OnboardingUpdateRequest(BaseModel):
    done: Optional[bool] = None
    step: Optional[int]  = None


@router.post("/onboarding")
def update_onboarding(
    req: OnboardingUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ob = db.query(models.OnboardingState).filter(models.OnboardingState.user_id == current_user.id).first()
    if not ob:
        ob = models.OnboardingState(user_id=current_user.id)
        db.add(ob)
    if req.done is not None:
        ob.done = req.done
    if req.step is not None:
        ob.step = req.step
    ob.updated_at = int(time.time())
    db.commit()
    return {"done": ob.done, "step": ob.step}


@router.get("/configs")
def get_configs(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cfg_row = db.query(models.UserConfig).filter(models.UserConfig.user_id == current_user.id).first()
    if not cfg_row or not cfg_row.config_enc:
        return {}
    data = decrypt_config(cfg_row.config_enc)
    # Mask secret values: show only first 4 chars + ***
    masked = {}
    for k, v in data.items():
        if v and isinstance(v, str) and len(v) > 6:
            masked[k] = v[:4] + "***"
        else:
            masked[k] = v
    return masked


class ConfigRequest(BaseModel):
    openai_key:     Optional[str] = None
    anthropic_key:  Optional[str] = None
    smtp_host:      Optional[str] = None
    smtp_port:      Optional[int] = None
    smtp_user:      Optional[str] = None
    smtp_password:  Optional[str] = None
    smtp_from:      Optional[str] = None


@router.post("/configs")
def save_configs(
    req: ConfigRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cfg_row = db.query(models.UserConfig).filter(models.UserConfig.user_id == current_user.id).first()
    existing = decrypt_config(cfg_row.config_enc) if cfg_row and cfg_row.config_enc else {}

    # Merge: only update fields that are not None in the request
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    existing.update(updates)

    enc = encrypt_config(existing)
    if cfg_row:
        cfg_row.config_enc = enc
        cfg_row.updated_at = int(time.time())
    else:
        db.add(models.UserConfig(user_id=current_user.id, config_enc=enc))
    db.commit()
    return {"status": "ok"}


# ── Pipeline versions (used by pipelines editor) ───────────────────────────────

@router.get("/pipelines/{pipeline_id}/versions")
def list_pipeline_versions(
    pipeline_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pipeline = db.query(models.Pipeline).filter(
        models.Pipeline.id == pipeline_id,
        models.Pipeline.user_id == current_user.id,
    ).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    versions = (
        db.query(models.PipelineVersion)
        .filter(models.PipelineVersion.pipeline_id == pipeline_id)
        .order_by(models.PipelineVersion.version_num.desc())
        .all()
    )
    return [
        {"id": v.id, "version_num": v.version_num, "note": v.note, "created_at": v.created_at}
        for v in versions
    ]


class PipelineVersionRequest(BaseModel):
    note: Optional[str] = None


@router.post("/pipelines/{pipeline_id}/versions")
def create_pipeline_version(
    pipeline_id: int,
    req: PipelineVersionRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pipeline = db.query(models.Pipeline).filter(
        models.Pipeline.id == pipeline_id,
        models.Pipeline.user_id == current_user.id,
    ).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    last = (
        db.query(func.max(models.PipelineVersion.version_num))
        .filter(models.PipelineVersion.pipeline_id == pipeline_id)
        .scalar() or 0
    )
    v = models.PipelineVersion(
        pipeline_id = pipeline_id,
        user_id     = current_user.id,
        version_num = last + 1,
        nodes       = pipeline.nodes,
        edges       = pipeline.edges,
        note        = req.note or "",
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return {"id": v.id, "version_num": v.version_num, "note": v.note, "created_at": v.created_at}


@router.post("/pipelines/{pipeline_id}/versions/{version_id}/restore")
def restore_pipeline_version(
    pipeline_id: int,
    version_id:  int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pipeline = db.query(models.Pipeline).filter(
        models.Pipeline.id == pipeline_id,
        models.Pipeline.user_id == current_user.id,
    ).first()
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    version = db.query(models.PipelineVersion).filter(
        models.PipelineVersion.id == version_id,
        models.PipelineVersion.pipeline_id == pipeline_id,
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    pipeline.nodes      = version.nodes
    pipeline.edges      = version.edges
    pipeline.updated_at = int(time.time())
    db.commit()
    return {"status": "ok", "restored_version": version.version_num}


# ── A/B Compare ───────────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    pipeline_a_id: int
    pipeline_b_id: int
    query: str


@router.get("/compare")
def list_comparisons(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List past A/B comparisons for this user."""
    rows = (
        db.query(models.Comparison)
        .filter(models.Comparison.user_id == current_user.id)
        .order_by(models.Comparison.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id":            c.id,
            "pipeline_a_id": c.pipeline_a_id,
            "pipeline_b_id": c.pipeline_b_id,
            "query":         c.query[:200],
            "risk_score_a":  c.risk_score_a,
            "risk_score_b":  c.risk_score_b,
            "status":        c.status,
            "created_at":    c.created_at,
        }
        for c in rows
    ]


@router.post("/compare")
def run_comparison(
    req: CompareRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a comparison record (pipeline execution is async / future work)."""
    for pid in (req.pipeline_a_id, req.pipeline_b_id):
        p = db.query(models.Pipeline).filter(
            models.Pipeline.id == pid,
            models.Pipeline.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail=f"Pipeline {pid} not found")
    c = models.Comparison(
        user_id       = current_user.id,
        pipeline_a_id = req.pipeline_a_id,
        pipeline_b_id = req.pipeline_b_id,
        query         = req.query,
        status        = "pending",
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return {
        "id":            c.id,
        "pipeline_a_id": c.pipeline_a_id,
        "pipeline_b_id": c.pipeline_b_id,
        "query":         c.query,
        "result_a":      "",
        "result_b":      "",
        "trace_a":       [],
        "trace_b":       [],
        "risk_score_a":  None,
        "risk_score_b":  None,
        "status":        c.status,
        "created_at":    c.created_at,
    }


@router.get("/compare/{comparison_id}")
def get_comparison(
    comparison_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.query(models.Comparison).filter(
        models.Comparison.id == comparison_id,
        models.Comparison.user_id == current_user.id,
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return {
        "id":            c.id,
        "pipeline_a_id": c.pipeline_a_id,
        "pipeline_b_id": c.pipeline_b_id,
        "query":         c.query,
        "result_a":      c.result_a,
        "result_b":      c.result_b,
        "trace_a":       json.loads(c.trace_a or "[]"),
        "trace_b":       json.loads(c.trace_b or "[]"),
        "risk_score_a":  c.risk_score_a,
        "risk_score_b":  c.risk_score_b,
        "status":        c.status,
        "created_at":    c.created_at,
    }


# ── Test sets ─────────────────────────────────────────────────────────────────

class TestSetRequest(BaseModel):
    name: str
    pipeline_id: Optional[int] = None
    cases: Optional[List[Dict[str, Any]]] = None


@router.get("/test-sets")
def list_test_sets(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(models.TestSet)
        .filter(models.TestSet.user_id == current_user.id)
        .order_by(models.TestSet.created_at.desc())
        .all()
    )
    return [
        {
            "id":          t.id,
            "name":        t.name,
            "pipeline_id": t.pipeline_id,
            "n_cases":     len(json.loads(t.cases or "[]")),
            "created_at":  t.created_at,
        }
        for t in rows
    ]


@router.post("/test-sets")
def create_test_set(
    req: TestSetRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = int(time.time())
    t = models.TestSet(
        user_id     = current_user.id,
        pipeline_id = req.pipeline_id,
        name        = req.name,
        cases       = json.dumps(req.cases or []),
        created_at  = now,
        updated_at  = now,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {
        "id":          t.id,
        "name":        t.name,
        "pipeline_id": t.pipeline_id,
        "cases":       json.loads(t.cases),
        "created_at":  t.created_at,
    }


@router.get("/test-sets/{test_set_id}")
def get_test_set(
    test_set_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    t = db.query(models.TestSet).filter(
        models.TestSet.id == test_set_id,
        models.TestSet.user_id == current_user.id,
    ).first()
    if not t:
        raise HTTPException(status_code=404, detail="Test set not found")
    return {
        "id":          t.id,
        "name":        t.name,
        "pipeline_id": t.pipeline_id,
        "cases":       json.loads(t.cases or "[]"),
        "created_at":  t.created_at,
    }


@router.get("/test-sets/{test_set_id}/runs")
def list_test_runs(
    test_set_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    t = db.query(models.TestSet).filter(
        models.TestSet.id == test_set_id,
        models.TestSet.user_id == current_user.id,
    ).first()
    if not t:
        raise HTTPException(status_code=404, detail="Test set not found")
    runs = (
        db.query(models.TestRun)
        .filter(models.TestRun.test_set_id == test_set_id)
        .order_by(models.TestRun.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "id":         r.id,
            "status":     r.status,
            "passed":     r.passed,
            "failed":     r.failed,
            "created_at": r.created_at,
        }
        for r in runs
    ]


@router.post("/test-sets/{test_set_id}/run")
def run_test_set(
    test_set_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    t = db.query(models.TestSet).filter(
        models.TestSet.id == test_set_id,
        models.TestSet.user_id == current_user.id,
    ).first()
    if not t:
        raise HTTPException(status_code=404, detail="Test set not found")
    run = models.TestRun(
        test_set_id = test_set_id,
        user_id     = current_user.id,
        status      = "pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return {"run_id": run.id, "status": run.status, "created_at": run.created_at}


@router.delete("/test-sets/{test_set_id}")
def delete_test_set(
    test_set_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    t = db.query(models.TestSet).filter(
        models.TestSet.id == test_set_id,
        models.TestSet.user_id == current_user.id,
    ).first()
    if t:
        db.delete(t)
        db.commit()
    return {"status": "ok"}


@router.get("/test-runs/{run_id}")
def get_test_run(
    run_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    r = db.query(models.TestRun).filter(
        models.TestRun.id == run_id,
        models.TestRun.user_id == current_user.id,
    ).first()
    if not r:
        raise HTTPException(status_code=404, detail="Test run not found")
    return {
        "id":         r.id,
        "test_set_id": r.test_set_id,
        "status":     r.status,
        "passed":     r.passed,
        "failed":     r.failed,
        "results":    json.loads(r.results or "[]"),
        "error":      r.error,
        "created_at": r.created_at,
    }
