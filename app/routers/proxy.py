"""
/proxy/* endpoints — LLM proxy stats and request log.
"""

import time

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth_utils import get_current_user
from app.database import get_db
from app import models

router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.get("/stats")
def proxy_stats(
    days: int = Query(7, ge=1, le=90),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Aggregated proxy statistics for the given time window."""
    since = int(time.time()) - days * 86400
    q = db.query(models.ProxyRequest).filter(
        models.ProxyRequest.user_id    == current_user.id,
        models.ProxyRequest.created_at >= since,
    )
    n_calls         = q.count()
    total_prompt    = q.with_entities(func.sum(models.ProxyRequest.prompt_tokens)).scalar() or 0
    total_completion= q.with_entities(func.sum(models.ProxyRequest.completion_tokens)).scalar() or 0
    avg_latency     = q.with_entities(func.avg(models.ProxyRequest.latency_ms)).scalar()
    avg_risk        = q.with_entities(func.avg(models.ProxyRequest.risk_score)).scalar()
    n_errors        = q.filter(models.ProxyRequest.status != "ok").count()

    return {
        "n_calls":          n_calls,
        "total_prompt":     int(total_prompt),
        "total_completion": int(total_completion),
        "avg_latency":      round(float(avg_latency), 1) if avg_latency is not None else None,
        "avg_risk":         round(float(avg_risk), 4)    if avg_risk    is not None else None,
        "n_errors":         n_errors,
    }


@router.get("/requests")
def proxy_requests(
    limit: int = Query(50, ge=1, le=500),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Recent proxied requests."""
    rows = (
        db.query(models.ProxyRequest)
        .filter(models.ProxyRequest.user_id == current_user.id)
        .order_by(models.ProxyRequest.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":               r.id,
            "provider":         r.provider,
            "model":            r.model,
            "prompt_tokens":    r.prompt_tokens,
            "completion_tokens":r.completion_tokens,
            "risk_score":       r.risk_score,
            "latency_ms":       r.latency_ms,
            "status":           r.status,
            "created_at":       r.created_at,
        }
        for r in rows
    ]
