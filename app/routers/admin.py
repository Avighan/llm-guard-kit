"""
/admin/* endpoints — platform management (admin users only).

Covers:
  • Platform-wide stats (users, orgs, chains, AI spend)
  • User management (list, promote/demote admin, deactivate)
  • Org management (list, upgrade plan)
  • AI cost breakdown per user/org
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.auth_utils import get_admin_user

router = APIRouter(prefix="/admin", tags=["admin"])

# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
def platform_stats(
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    now       = int(time.time())
    today_start = now - (now % 86400)

    total_users  = db.query(func.count(models.User.id)).scalar() or 0
    total_orgs   = db.query(func.count(models.Org.id)).scalar() or 0
    total_chains = db.query(func.count(models.Chain.id)).scalar() or 0
    total_cost   = db.query(func.sum(models.Chain.ai_cost_usd)).scalar() or 0.0
    chains_today = db.query(func.count(models.Chain.id)).filter(models.Chain.timestamp >= today_start).scalar() or 0
    alerts_today = db.query(func.count(models.Chain.id)).filter(
        models.Chain.timestamp   >= today_start,
        models.Chain.needs_alert == True,  # noqa: E712
    ).scalar() or 0

    # Daily chain counts (last 14 days)
    daily = []
    for i in range(13, -1, -1):
        day_start = today_start - i * 86400
        day_end   = day_start + 86400
        count = db.query(func.count(models.Chain.id)).filter(
            models.Chain.timestamp >= day_start,
            models.Chain.timestamp <  day_end,
        ).scalar() or 0
        daily.append({"timestamp": day_start, "n_chains": count})

    return {
        "total_users":      total_users,
        "total_orgs":       total_orgs,
        "total_chains":     total_chains,
        "total_ai_cost_usd": round(float(total_cost), 4),
        "chains_today":     chains_today,
        "alerts_today":     alerts_today,
        "daily_chains":     daily,
    }


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users")
def list_users(
    page: int = Query(1, ge=1),
    q:    str = Query(""),
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    PAGE_SIZE = 50
    query = db.query(models.User)
    if q:
        query = query.filter(models.User.email.ilike(f"%{q}%") | models.User.name.ilike(f"%{q}%"))
    total = query.count()
    users = query.order_by(models.User.created_at.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    return {
        "total": total,
        "page":  page,
        "users": [
            {
                "id":                u.id,
                "email":             u.email,
                "name":              u.name,
                "plan":              u.plan,
                "is_admin":          u.is_admin,
                "is_active":         u.is_active,
                "chains_this_month": u.chains_this_month,
                "created_at":        u.created_at,
            }
            for u in users
        ],
    }


@router.post("/users/{user_id}/toggle-admin")
def toggle_admin(
    user_id: int,
    admin: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own admin status")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_admin = not user.is_admin
    db.commit()
    return {"user_id": user_id, "is_admin": user.is_admin}


@router.delete("/users/{user_id}")
def deactivate_user(
    user_id: int,
    admin: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = False
    db.commit()
    return {"status": "ok", "user_id": user_id}


class SetPlanRequest(BaseModel):
    plan: str  # "free" | "pro"


@router.post("/users/{user_id}/set-plan")
def set_user_plan(
    user_id: int,
    req: SetPlanRequest,
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    if req.plan not in ("free", "pro"):
        raise HTTPException(status_code=400, detail="Plan must be 'free' or 'pro'")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.plan = req.plan
    db.commit()
    return {"user_id": user_id, "plan": user.plan}


@router.post("/users/{user_id}/reactivate")
def reactivate_user(
    user_id: int,
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = True
    db.commit()
    return {"status": "ok", "user_id": user_id}


@router.post("/users/{user_id}/reset-usage")
def reset_usage(
    user_id: int,
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.chains_this_month = 0
    user.month_reset_at    = int(time.time())
    db.commit()
    return {"status": "ok", "user_id": user_id}


@router.get("/users/{user_id}/chains")
def user_chains(
    user_id: int,
    limit: int = Query(50, ge=1, le=200),
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    chains = (
        db.query(models.Chain)
        .filter(models.Chain.user_id == user_id)
        .order_by(models.Chain.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "user_id": user_id,
        "email":   user.email,
        "chains": [
            {
                "id":              c.id,
                "domain":          c.domain,
                "question":        c.question[:200] if c.question else "",
                "risk_score":      c.risk_score,
                "confidence_tier": c.confidence_tier,
                "needs_alert":     c.needs_alert,
                "ai_cost_usd":     c.ai_cost_usd,
                "timestamp":       c.timestamp,
            }
            for c in chains
        ],
    }


# ── Orgs ──────────────────────────────────────────────────────────────────────

@router.get("/orgs")
def list_orgs(
    page: int = Query(1, ge=1),
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    PAGE_SIZE = 50
    orgs  = db.query(models.Org).order_by(models.Org.created_at.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE).all()
    total = db.query(func.count(models.Org.id)).scalar() or 0
    result = []
    for org in orgs:
        owner = db.query(models.User).filter(models.User.id == org.owner_id).first()
        member_count = db.query(func.count(models.OrgMember.id)).filter(models.OrgMember.org_id == org.id).scalar() or 0
        chain_count  = db.query(func.count(models.Chain.id)).filter(models.Chain.org_id == org.id).scalar() or 0
        result.append({
            "id":           org.id,
            "name":         org.name,
            "owner_email":  owner.email if owner else "",
            "owner_plan":   owner.plan  if owner else "free",
            "member_count": member_count,
            "chain_count":  chain_count,
            "ai_cost_usd":  round(float(org.ai_cost_usd), 4),
            "created_at":   org.created_at,
        })
    return {"total": total, "page": page, "orgs": result}


@router.post("/orgs/{org_id}/upgrade")
def upgrade_org(
    org_id: int,
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    org = db.query(models.Org).filter(models.Org.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Org not found")
    owner = db.query(models.User).filter(models.User.id == org.owner_id).first()
    if owner:
        owner.plan = "pro"
        db.commit()
    return {"status": "ok", "org_id": org_id, "plan": "pro"}


# ── AI Cost Breakdown ─────────────────────────────────────────────────────────

@router.get("/costs")
def cost_breakdown(
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Per-user AI cost breakdown: chains scored, judge calls, total cost."""
    rows = (
        db.query(
            models.User.id,
            models.User.email,
            models.User.plan,
            func.count(models.Chain.id).label("n_chains"),
            func.sum(models.Chain.ai_cost_usd).label("total_cost"),
        )
        .outerjoin(models.Chain, models.Chain.user_id == models.User.id)
        .group_by(models.User.id)
        .order_by(func.sum(models.Chain.ai_cost_usd).desc())
        .all()
    )
    result = []
    for r in rows:
        judge_calls = db.query(func.count(models.Chain.id)).filter(
            models.Chain.user_id   == r.id,
            models.Chain.judge_label.notin_(["LOCAL", None, ""]),
        ).scalar() or 0
        result.append({
            "user_id":       r.id,
            "email":         r.email,
            "plan":          r.plan,
            "chains":        r.n_chains or 0,
            "judge_calls":   judge_calls,
            "total_cost_usd": round(float(r.total_cost or 0.0), 4),
        })
    return result
