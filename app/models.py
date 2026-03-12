"""
SQLAlchemy ORM models for llm-guard-kit SaaS.

Multi-tenant isolation:
  • Every user-owned row has user_id (and optionally org_id)
  • Org rows additionally carry ai_cost_usd for AI cost management
  • Admin users (is_admin=True) can query all rows regardless of owner
"""

import time
from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Integer, String, Text, BigInteger
)
from sqlalchemy.orm import relationship

from app.database import Base


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id                  = Column(Integer, primary_key=True, index=True)
    email               = Column(String(254), unique=True, index=True, nullable=False)
    name                = Column(String(200), default="")
    password_hash       = Column(String(256), nullable=True)   # None for OAuth-only users
    google_id           = Column(String(128), unique=True, nullable=True, index=True)
    plan                = Column(String(20), default="free")   # "free" | "pro"
    is_admin            = Column(Boolean, default=False)
    is_active           = Column(Boolean, default=True)
    chains_this_month   = Column(Integer, default=0)
    month_reset_at      = Column(BigInteger, default=lambda: int(time.time()))
    stripe_customer_id  = Column(String(128), nullable=True, unique=True)
    created_at          = Column(BigInteger, default=lambda: int(time.time()))

    orgs       = relationship("OrgMember", back_populates="user", cascade="all, delete-orphan")
    api_keys   = relationship("ApiKey",    back_populates="user", cascade="all, delete-orphan")
    chains     = relationship("Chain",     back_populates="user")
    guardrails = relationship("Guardrail", back_populates="user", cascade="all, delete-orphan")
    workflows  = relationship("Workflow",  back_populates="user", cascade="all, delete-orphan")
    pipelines  = relationship("Pipeline",  back_populates="user", cascade="all, delete-orphan")
    config     = relationship("UserConfig", back_populates="user", uselist=False, cascade="all, delete-orphan")
    onboarding = relationship("OnboardingState", back_populates="user", uselist=False, cascade="all, delete-orphan")


# ── Organizations ─────────────────────────────────────────────────────────────

class Org(Base):
    __tablename__ = "orgs"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(200), nullable=False)
    owner_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ai_cost_usd = Column(Float, default=0.0)   # running total AI spend
    created_at  = Column(BigInteger, default=lambda: int(time.time()))

    members    = relationship("OrgMember", back_populates="org", cascade="all, delete-orphan")
    api_keys   = relationship("ApiKey",    back_populates="org")
    chains     = relationship("Chain",     back_populates="org")
    guardrails = relationship("Guardrail", back_populates="org")
    workflows  = relationship("Workflow",  back_populates="org")
    pipelines  = relationship("Pipeline",  back_populates="org")
    invites    = relationship("InviteToken", back_populates="org", cascade="all, delete-orphan")


class OrgMember(Base):
    __tablename__ = "org_members"

    id        = Column(Integer, primary_key=True)
    org_id    = Column(Integer, ForeignKey("orgs.id"), nullable=False, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role      = Column(String(20), default="member")   # "owner" | "admin" | "member"
    joined_at = Column(BigInteger, default=lambda: int(time.time()))

    org  = relationship("Org",  back_populates="members")
    user = relationship("User", back_populates="orgs")


# ── API Keys ──────────────────────────────────────────────────────────────────

class ApiKey(Base):
    __tablename__ = "api_keys"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    org_id        = Column(Integer, ForeignKey("orgs.id"), nullable=True)
    key_prefix    = Column(String(20),  unique=True, index=True, nullable=False)
    key_hash      = Column(String(256), nullable=False)
    domain_prefix = Column(String(200), default="")
    request_count = Column(Integer, default=0)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="api_keys")
    org  = relationship("Org",  back_populates="api_keys")


# ── Scored Chains ─────────────────────────────────────────────────────────────

class Chain(Base):
    __tablename__ = "chains"

    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    org_id          = Column(Integer, ForeignKey("orgs.id"),  nullable=True, index=True)
    domain          = Column(String(200), default="default", index=True)
    question        = Column(Text, default="")
    final_answer    = Column(Text, default="")
    risk_score      = Column(Float, nullable=False)
    confidence_tier = Column(String(10), default="HIGH")  # HIGH / MEDIUM / LOW
    needs_alert     = Column(Boolean, default=False)
    n_steps         = Column(Integer, default=0)
    failure_mode    = Column(String(100), nullable=True)
    judge_label     = Column(String(20),  nullable=True)
    ai_cost_usd     = Column(Float, default=0.0)
    timestamp       = Column(BigInteger, default=lambda: int(time.time()), index=True)
    created_at      = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="chains")
    org  = relationship("Org",  back_populates="chains")


# ── Guardrails ────────────────────────────────────────────────────────────────

class Guardrail(Base):
    __tablename__ = "guardrails"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    org_id        = Column(Integer, ForeignKey("orgs.id"),  nullable=True)
    domain        = Column(String(200), default="*")   # "*" = all domains
    threshold     = Column(Float, default=0.70)
    action        = Column(String(20), default="alert")  # "alert" | "block" | "fallback"
    webhook_url   = Column(String(500), default="")
    slack_webhook = Column(String(500), default="")
    enabled       = Column(Boolean, default=True)
    created_at    = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="guardrails")
    org  = relationship("Org",  back_populates="guardrails")


# ── Workflows ─────────────────────────────────────────────────────────────────

class Workflow(Base):
    __tablename__ = "workflows"

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    org_id     = Column(Integer, ForeignKey("orgs.id"),  nullable=True)
    name       = Column(String(200), nullable=False)
    nodes      = Column(Text, default="[]")  # JSON
    edges      = Column(Text, default="[]")  # JSON
    enabled    = Column(Boolean, default=True)
    created_at = Column(BigInteger, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="workflows")
    org  = relationship("Org",  back_populates="workflows")


# ── Pipelines ─────────────────────────────────────────────────────────────────

class Pipeline(Base):
    __tablename__ = "pipelines"

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    org_id     = Column(Integer, ForeignKey("orgs.id"),  nullable=True)
    name       = Column(String(200), nullable=False)
    nodes      = Column(Text, default="[]")  # JSON
    edges      = Column(Text, default="[]")  # JSON
    enabled    = Column(Boolean, default=True)
    created_at = Column(BigInteger, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="pipelines")
    org  = relationship("Org",  back_populates="pipelines")
    runs = relationship("PipelineRun", back_populates="pipeline", cascade="all, delete-orphan")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id          = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False, index=True)
    run_id      = Column(String(64), unique=True, index=True)
    result      = Column(Text, default="")
    trace       = Column(Text, default="[]")   # JSON
    risk_score  = Column(Float, nullable=True)
    status      = Column(String(20), default="done")  # "done" | "error"
    error       = Column(Text, nullable=True)
    created_at  = Column(BigInteger, default=lambda: int(time.time()))

    pipeline = relationship("Pipeline", back_populates="runs")


# ── Invite Tokens ─────────────────────────────────────────────────────────────

class InviteToken(Base):
    __tablename__ = "invite_tokens"

    id         = Column(Integer, primary_key=True)
    org_id     = Column(Integer, ForeignKey("orgs.id"), nullable=False, index=True)
    inviter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    email      = Column(String(254), nullable=False)
    token      = Column(String(64), unique=True, index=True, nullable=False)
    accepted   = Column(Boolean, default=False)
    created_at = Column(BigInteger, default=lambda: int(time.time()))
    expires_at = Column(BigInteger, nullable=False)

    org = relationship("Org", back_populates="invites")


# ── User Config (encrypted) ───────────────────────────────────────────────────

class UserConfig(Base):
    __tablename__ = "user_configs"

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    config_enc = Column(Text, default="")   # Fernet-encrypted JSON blob
    created_at = Column(BigInteger, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="config")


# ── Onboarding State ──────────────────────────────────────────────────────────

class OnboardingState(Base):
    __tablename__ = "onboarding_states"

    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    done       = Column(Boolean, default=False)
    step       = Column(Integer, default=0)
    created_at = Column(BigInteger, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, default=lambda: int(time.time()))

    user = relationship("User", back_populates="onboarding")


# ── Pipeline Versions ─────────────────────────────────────────────────────────

class PipelineVersion(Base):
    __tablename__ = "pipeline_versions"

    id          = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"),     nullable=False, index=True)
    version_num = Column(Integer, nullable=False, default=1)
    nodes       = Column(Text, default="[]")  # JSON snapshot
    edges       = Column(Text, default="[]")  # JSON snapshot
    note        = Column(String(500), default="")
    created_at  = Column(BigInteger, default=lambda: int(time.time()))

    pipeline = relationship("Pipeline", backref="versions")


# ── A/B Comparisons ──────────────────────────────────────────────────────────

class Comparison(Base):
    __tablename__ = "comparisons"

    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    pipeline_a_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    pipeline_b_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    query         = Column(Text, nullable=False)
    result_a      = Column(Text, default="")
    result_b      = Column(Text, default="")
    trace_a       = Column(Text, default="[]")  # JSON
    trace_b       = Column(Text, default="[]")  # JSON
    risk_score_a  = Column(Float, nullable=True)
    risk_score_b  = Column(Float, nullable=True)
    status        = Column(String(20), default="pending")  # "pending" | "done" | "error"
    error         = Column(Text, nullable=True)
    created_at    = Column(BigInteger, default=lambda: int(time.time()))


# ── Test Sets ─────────────────────────────────────────────────────────────────

class TestSet(Base):
    __tablename__ = "test_sets"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=True)
    name        = Column(String(200), nullable=False)
    cases       = Column(Text, default="[]")  # JSON: [{question, expected_answer, ...}]
    created_at  = Column(BigInteger, default=lambda: int(time.time()))
    updated_at  = Column(BigInteger, default=lambda: int(time.time()))

    runs = relationship("TestRun", back_populates="test_set", cascade="all, delete-orphan")


class TestRun(Base):
    __tablename__ = "test_runs"

    id          = Column(Integer, primary_key=True)
    test_set_id = Column(Integer, ForeignKey("test_sets.id"), nullable=False, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"),      nullable=False, index=True)
    status      = Column(String(20), default="pending")  # "pending"|"running"|"done"|"error"
    results     = Column(Text, default="[]")   # JSON: per-case results
    passed      = Column(Integer, default=0)
    failed      = Column(Integer, default=0)
    error       = Column(Text, nullable=True)
    created_at  = Column(BigInteger, default=lambda: int(time.time()))
    updated_at  = Column(BigInteger, default=lambda: int(time.time()))

    test_set = relationship("TestSet", back_populates="runs")


# ── Chain Feedback (thumbs up / down, used for verifier retraining) ───────────

class ChainFeedback(Base):
    """
    User-provided correctness label for a scored chain.

    Populated when the user clicks thumbs-up / thumbs-down on an alert in the
    dashboard.  Accumulated feedback is used by POST /v1/verifier/retrain to
    incrementally improve the LocalVerifier without human-labelling overhead.
    """
    __tablename__ = "chain_feedback"

    id         = Column(Integer, primary_key=True)
    chain_id   = Column(Integer, ForeignKey("chains.id"), nullable=False, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"),  nullable=False, index=True)
    label      = Column(String(10), nullable=False)   # "correct" | "incorrect"
    note       = Column(String(500), default="")      # optional free-text
    created_at = Column(BigInteger, default=lambda: int(time.time()))

    chain = relationship("Chain", backref="feedback")
    user  = relationship("User")


# ── Proxy Requests ────────────────────────────────────────────────────────────

class ProxyRequest(Base):
    __tablename__ = "proxy_requests"

    id               = Column(Integer, primary_key=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    api_key_prefix   = Column(String(20), nullable=True)
    provider         = Column(String(20), default="openai")   # "openai" | "anthropic"
    model            = Column(String(100), default="")
    prompt_tokens    = Column(Integer, default=0)
    completion_tokens= Column(Integer, default=0)
    risk_score       = Column(Float, nullable=True)
    latency_ms       = Column(Integer, default=0)
    status           = Column(String(20), default="ok")   # "ok" | "error"
    error            = Column(Text, nullable=True)
    created_at       = Column(BigInteger, default=lambda: int(time.time()), index=True)


# ── A2A Audit Log (persistent, survives restart) ──────────────────────────────

class A2AAuditLog(Base):
    """
    Persistent audit trail for A2A trust propagation events.

    Replaces the prior in-memory `_a2a_audit_log` deque.
    Each handoff via POST /v1/a2a/handoff writes one row.
    GET /v1/a2a/audit queries this table with a limit parameter.
    """
    __tablename__ = "a2a_audit_log"

    id           = Column(Integer, primary_key=True, index=True)
    event_type   = Column(String(50), default="handoff", index=True)  # "handoff" | "verify"
    from_agent   = Column(String(200), default="upstream")
    to_agent     = Column(String(200), default="")
    trust_id     = Column(String(64),  nullable=True, index=True)
    risk_score   = Column(Float,       nullable=True)
    hop_count    = Column(Integer,     default=0)
    verify_ok    = Column(Boolean,     nullable=True)   # for "verify" events
    payload_json = Column(Text,        default="{}")    # full event JSON for inspection
    created_at   = Column(BigInteger,  default=lambda: int(time.time()), index=True)
