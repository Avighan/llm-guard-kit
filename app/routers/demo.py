"""
/demo/* endpoints — unauthenticated scoring for onboarding flow.

No DB writes. Returns AgentGuard behavioral score so new users
can see a live result without creating an account first.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/demo", tags=["demo"])


class DemoStep(BaseModel):
    thought:     str = ""
    action:      str = ""
    observation: str = ""
    # also accept the AgentGuard native format
    action_type: str = ""
    action_arg:  str = ""


class DemoScoreRequest(BaseModel):
    question:     str
    steps:        List[DemoStep]
    final_answer: str


@router.post("/score")
def demo_score(req: DemoScoreRequest):
    """
    Score a demo chain with AgentGuard (behavioral SC_OLD, $0 cost).
    Used by the onboarding wizard to show a live risk score without auth.
    """
    from llm_guard.agent_guard import AgentGuard

    guard = AgentGuard()

    # Normalize steps to AgentGuard format
    steps = []
    for s in req.steps:
        steps.append({
            "thought":     s.thought,
            "action_type": s.action_type or s.action or "Action",
            "action_arg":  s.action_arg  or s.action or "",
            "observation": s.observation,
        })

    result = guard.score_chain(
        question     = req.question,
        steps        = steps,
        final_answer = req.final_answer,
    )

    return {
        "risk_score":       result.risk_score,
        "confidence_tier":  result.confidence_tier,
        "needs_review":     result.needs_alert,
        "needs_alert":      result.needs_alert,
        "behavioral_score": result.behavioral_score,
        "gmm_score":        None,
        "failure_mode":     result.failure_mode,
        "chain_id":         None,
    }
