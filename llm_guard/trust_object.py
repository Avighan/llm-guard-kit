"""
A2ATrustObject — Structured trust handoff between agents.

Defined and validated in exp105 (A2A Handoff Confidence Protocol).
This is the standard envelope an agent emits at the end of a chain so that
a downstream agent can condition its strategy on upstream confidence.

Key findings (exp105):
  - Inter-agent error correlation ρ = +0.108 (errors are weakly correlated)
  - P(B fails | A fails)  = 0.679 vs P(B fails | A correct) = 0.574, lift +0.105
  - Best downstream strategy: Agent B scores itself (AUROC 0.808) rather than
    trusting the chain product formula (AUROC 0.661)
  - When confidence_tier = LOW, Agent B should rewrite the query before
    proceeding (see QueryRewriter in query_rewriter.py)

Confidence tier thresholds (from exp92 conformal calibration):
  HIGH   risk_score < 0.50  → proceed normally
  MEDIUM 0.50 ≤ risk < 0.70 → proceed with monitoring
  LOW    risk_score ≥ 0.70  → trigger query rewriter or human review

Usage
-----
    # Agent A produces a trust object after its chain completes
    from llm_guard.agent_guard import AgentGuard

    guard = AgentGuard(api_key="sk-ant-...", use_judge=True)
    trust = guard.generate_trust_object(question, steps, final_answer)

    # Serialise for wire transport (JSON, message queue, etc.)
    payload = trust.to_dict()

    # Agent B deserialises and conditions its strategy
    trust = A2ATrustObject.from_dict(payload)
    if trust.should_rewrite:
        from llm_guard.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(api_key="sk-ant-...")
        alt_queries = rewriter.rewrite(question, trust)
        # use alt_queries for Agent B's search strategy
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TemporalValidity:
    """
    Time-sensitivity metadata for a chain (exp106).

    Activated only when the question is classified as time-sensitive.
    Agent B should re-verify if tv_risk > 0.5.
    """
    is_time_sensitive: bool = False
    stale_score: float = 0.0       # 0=fresh, 1=stale observations
    recency_score: float = 0.0     # fraction of steps with recency signals
    tv_risk: float = 0.0           # combined temporal validity risk [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemporalValidity":
        return cls(
            is_time_sensitive=d.get("is_time_sensitive", False),
            stale_score=d.get("stale_score", 0.0),
            recency_score=d.get("recency_score", 0.0),
            tv_risk=d.get("tv_risk", 0.0),
        )


@dataclass
class TrustHop:
    """
    One node in a multi-hop A2A trust chain (v0.4, exp114+).

    Each agent that handles a chain appends a TrustHop via
    A2ATrustObject.add_hop().  The hop is HMAC-signed over:
      (prev_chain_hash, agent_id, risk_score, timestamp)
    so downstream agents can verify the full chain-of-custody.

    Fields
    ------
    agent_id : str
        Identifier for the agent that added this hop (e.g. "search-agent-1").
    risk_score : float
        This agent's risk assessment at the time of handoff.
    timestamp : float
        Unix timestamp (seconds) when this hop was added.
    agent_card_ref : str or None
        URI to the agent's capability card (Google A2A AgentCard.url).
    hop_signature : str or None
        HMAC-SHA256 hex digest over the canonical hop payload.
    """
    agent_id:       str
    risk_score:     float
    timestamp:      float
    agent_card_ref: Optional[str] = None
    hop_signature:  Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":       self.agent_id,
            "risk_score":     round(self.risk_score, 4),
            "timestamp":      round(self.timestamp, 3),
            "agent_card_ref": self.agent_card_ref,
            "hop_signature":  self.hop_signature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrustHop":
        return cls(
            agent_id       = d.get("agent_id", ""),
            risk_score     = float(d.get("risk_score", 0.5)),
            timestamp      = float(d.get("timestamp", 0.0)),
            agent_card_ref = d.get("agent_card_ref"),
            hop_signature  = d.get("hop_signature"),
        )


@dataclass
class A2ATrustObject:
    """
    Structured confidence envelope emitted by an agent at chain completion.

    Carry this alongside the answer when passing output to a downstream
    agent, orchestrator, or human review queue. The downstream consumer
    uses confidence_tier and failure_mode to decide whether to proceed,
    re-verify, or rewrite the query.

    Fields
    ------
    answer : str
        The agent's final answer.
    risk_score : float
        Composite risk score [0, 1]. Higher = more likely wrong.
        From J5 ensemble (SC_OLD behavioral + Sonnet judge) when
        use_judge=True, else SC_OLD behavioral only.
    confidence_tier : str
        "HIGH"   risk < 0.50  — proceed normally
        "MEDIUM" 0.50 ≤ risk < 0.70  — proceed with monitoring
        "LOW"    risk ≥ 0.70  — rewrite query or escalate
    failure_mode : str or None
        Detected failure pattern from LabelFreeScorer:
          "retrieval_fail"        observation contains no-results signal
          "repeated_query"        same search issued twice
          "long_chain"            4+ steps (budget exhaustion)
          "empty_answer"          final answer is absent or trivially short
          "low_retrieval_quality" mean cosine(q, obs) < 0.30
          "no_evidence"           no token overlap between queries and obs
          None                    no failure mode detected
    step_count : int
        Number of Search steps in the chain.
    judge_label : str or None
        Sonnet judge verdict: "GOOD" | "BORDERLINE" | "POOR" | None
        None when use_judge=False (saves ~$0.007/chain).
    downstream_hint : str
        Plain-English recommendation for the downstream agent.
          "proceed"                    HIGH confidence, no action needed
          "proceed_with_caution"       MEDIUM, validate key claims
          "rewrite_query"              LOW, try alternative formulation
          "rewrite_and_verify"         LOW + temporal validity issue
          "escalate_to_human"          POOR judge label, high risk
    should_rewrite : bool
        Shortcut: True when confidence_tier == "LOW".
        Set to True when QueryRewriter should be invoked before Agent B runs.
    behavioral_components : dict
        Raw SC_OLD component scores (sc1..sc12) for debugging / logging.
    temporal_validity : TemporalValidity or None
        Time-sensitivity metadata. None when TV scoring is disabled.
    """

    answer: str
    risk_score: float
    confidence_tier: str                # HIGH / MEDIUM / LOW
    failure_mode: Optional[str]
    step_count: int
    judge_label: Optional[str]          # GOOD / BORDERLINE / POOR / None
    downstream_hint: str
    should_rewrite: bool
    behavioral_components: Dict[str, float] = field(default_factory=dict)
    temporal_validity: Optional[TemporalValidity] = None
    # v0.3 provenance fields (exp114)
    trust_signature: Optional[str] = None   # HMAC-SHA256 of canonical trust fields
    agent_card_ref: Optional[str] = None    # URI referencing Agent A's capability card
    # v0.4 multi-hop chain-of-custody
    trust_chain: List[TrustHop] = field(default_factory=list)
    # v0.5 replay protection — set automatically by sign()
    issued_at: Optional[float] = None       # Unix timestamp when signed
    ttl: int = 300                          # seconds until signature expires (default 5 min)

    # ── Signing / verification (exp114 A2A v0.5) ─────────────────────────────

    _SIGN_FIELDS = (
        "answer", "risk_score", "confidence_tier", "failure_mode",
        "step_count", "judge_label", "downstream_hint", "should_rewrite",
        "issued_at", "ttl",
    )

    def sign(self, secret: str) -> "A2ATrustObject":
        """
        Compute HMAC-SHA256 over canonical trust fields and store in trust_signature.
        Sets issued_at to current time for replay protection.

        Returns self to allow chaining:
            trust = guard.generate_trust_object(...).sign(secret)
        """
        self.issued_at = time.time()
        payload = {k: getattr(self, k) for k in self._SIGN_FIELDS}
        canon   = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.trust_signature = hmac.new(
            secret.encode(), canon.encode(), hashlib.sha256
        ).hexdigest()
        return self

    def verify(self, secret: str, check_expiry: bool = True) -> bool:
        """
        Verify the trust_signature using the shared secret.

        Returns True when signature is valid and (if check_expiry=True)
        the object has not expired (issued_at + ttl > now).
        Returns False when trust_signature is None, tampered, or expired.

        Parameters
        ----------
        secret : str
            Shared secret known to Agent A and Agent B.
        check_expiry : bool
            If True (default), reject objects older than ttl seconds.
            Set False only for offline/testing scenarios.
        """
        if not self.trust_signature:
            return False
        # Replay protection: reject expired objects
        if check_expiry and self.issued_at is not None:
            age = time.time() - self.issued_at
            if age > self.ttl:
                return False
        payload  = {k: getattr(self, k) for k in self._SIGN_FIELDS}
        canon    = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(
            secret.encode(), canon.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.trust_signature, expected)

    # ── Multi-hop chain-of-custody (v0.4) ────────────────────────────────────

    def _chain_hash(self) -> str:
        """SHA-256 of the current trust_chain list (first 16 hex chars)."""
        payload = json.dumps(
            [h.to_dict() for h in self.trust_chain],
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def add_hop(
        self,
        agent_id:       str,
        secret:         str,
        agent_card_ref: Optional[str] = None,
    ) -> "A2ATrustObject":
        """
        Append this agent to the trust chain with an HMAC-signed hop.

        Each hop is signed over:
            {prev_chain_hash, agent_id, risk_score, timestamp}

        This creates a chain-of-custody: Agent C can verify that B received
        from A and that neither tampered with the risk_score.

        Parameters
        ----------
        agent_id : str
            Identifier for the calling agent (e.g. "retrieval-agent-1").
        secret : str
            Shared secret known to this agent and any downstream verifier.
        agent_card_ref : str, optional
            URI to this agent's capability card.

        Returns self for chaining:
            trust = guard.generate_trust_object(...).add_hop("agent-a", secret_a)
            # pass to agent B, which calls:
            trust.add_hop("agent-b", secret_b)
        """
        prev_hash = self._chain_hash()
        ts        = round(time.time(), 3)
        hop_data  = json.dumps({
            "prev_hash":  prev_hash,
            "agent_id":   agent_id,
            "risk_score": round(self.risk_score, 4),
            "timestamp":  ts,
        }, sort_keys=True, separators=(",", ":"))
        hop_sig = hmac.new(
            secret.encode(), hop_data.encode(), hashlib.sha256
        ).hexdigest()
        self.trust_chain.append(TrustHop(
            agent_id       = agent_id,
            risk_score     = self.risk_score,
            timestamp      = ts,
            agent_card_ref = agent_card_ref,
            hop_signature  = hop_sig,
        ))
        return self

    def verify_chain(self, secrets_map: Dict[str, str]) -> bool:
        """
        Verify every hop signature in the trust chain.

        Parameters
        ----------
        secrets_map : dict[agent_id, secret]
            Mapping from agent_id to the shared secret for that agent.
            Every agent_id in the chain must have an entry.

        Returns
        -------
        True  — all hop signatures are valid (full chain-of-custody intact).
        False — any hop has an invalid/missing signature, or a secret is unknown.

        Empty chain (no hops) returns True.

        Example
        -------
            ok = trust.verify_chain({"agent-a": secret_a, "agent-b": secret_b})
        """
        if not self.trust_chain:
            return True
        prev_hops: List[TrustHop] = []
        for hop in self.trust_chain:
            prev_payload = json.dumps(
                [h.to_dict() for h in prev_hops],
                sort_keys=True, separators=(",", ":"),
            )
            prev_hash = hashlib.sha256(prev_payload.encode()).hexdigest()[:16]
            hop_data  = json.dumps({
                "prev_hash":  prev_hash,
                "agent_id":   hop.agent_id,
                "risk_score": round(hop.risk_score, 4),
                "timestamp":  round(hop.timestamp, 3),
            }, sort_keys=True, separators=(",", ":"))
            secret = secrets_map.get(hop.agent_id, "")
            if not secret:
                return False
            expected = hmac.new(
                secret.encode(), hop_data.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(hop.hop_signature or "", expected):
                return False
            prev_hops.append(hop)
        return True

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def is_high_risk(self) -> bool:
        """True when the chain is likely wrong (risk >= 0.70)."""
        return self.risk_score >= 0.70

    @property
    def precision_at_alert(self) -> float:
        """
        Expected precision when this object triggers an alert.
        Based on exp92 conformal calibration:
          α=0.10 (FPR ≤ 10%) → Precision = 0.908
          α=0.05 (FPR ≤  5%) → Precision = 0.903
        Returns 0.0 when confidence_tier != "LOW" (no alert triggered).
        """
        return 0.908 if self.confidence_tier == "LOW" else 0.0

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON transport."""
        d = {
            "answer":                 self.answer,
            "risk_score":             round(self.risk_score, 4),
            "confidence_tier":        self.confidence_tier,
            "failure_mode":           self.failure_mode,
            "step_count":             self.step_count,
            "judge_label":            self.judge_label,
            "downstream_hint":        self.downstream_hint,
            "should_rewrite":         self.should_rewrite,
            "behavioral_components":  {k: round(v, 4) for k, v in self.behavioral_components.items()},
            "temporal_validity":      self.temporal_validity.to_dict() if self.temporal_validity else None,
            "trust_signature":        self.trust_signature,
            "agent_card_ref":         self.agent_card_ref,
            "trust_chain":            [h.to_dict() for h in self.trust_chain],
            "issued_at":              self.issued_at,
            "ttl":                    self.ttl,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "A2ATrustObject":
        """Deserialise from a plain dict (wire format). Backward-compatible with v0.1–v0.4."""
        tv = d.get("temporal_validity")
        return cls(
            answer=d.get("answer", ""),
            risk_score=float(d.get("risk_score", 0.5)),
            confidence_tier=d.get("confidence_tier", "MEDIUM"),
            failure_mode=d.get("failure_mode"),
            step_count=int(d.get("step_count", 0)),
            judge_label=d.get("judge_label"),
            downstream_hint=d.get("downstream_hint", "proceed_with_caution"),
            should_rewrite=bool(d.get("should_rewrite", False)),
            behavioral_components=d.get("behavioral_components", {}),
            temporal_validity=TemporalValidity.from_dict(tv) if tv else None,
            trust_signature=d.get("trust_signature"),
            agent_card_ref=d.get("agent_card_ref"),
            trust_chain=[TrustHop.from_dict(h) for h in d.get("trust_chain", [])],
            issued_at=d.get("issued_at"),
            ttl=int(d.get("ttl", 300)),
        )

    def __repr__(self) -> str:
        return (
            f"A2ATrustObject("
            f"tier={self.confidence_tier}, "
            f"risk={self.risk_score:.3f}, "
            f"judge={self.judge_label}, "
            f"hint={self.downstream_hint!r}"
            f")"
        )


@dataclass
class StreamGuardResult:
    """
    Result of AgentGuard.stream_guard() — mid-chain abort decision (exp113).

    Call stream_guard() after step 2 of your agent loop.  When abort=True,
    stop the current chain immediately and use rewritten_queries to restart
    with a diversified query.  This enables the two-stage OR strategy that
    achieves Recall=0.416, Precision=0.740 — the best recall in the QPPG
    experiment history (exp107 / exp113).

    Fields
    ------
    abort : bool
        True when the chain should be aborted.  Fires when combined
        step-level risk >= abort_threshold (default 0.65).
    risk_at_step : float
        Combined step-level risk score [0, 1].
        Blends behavioral SC_OLD prefix score and Haiku mid-chain judge.
    step_index : int
        0-based index of the last step evaluated (typically 1 for step 2).
    on_track : bool or None
        Haiku's on_track field.  False = clear early-warning signal.
        None when Haiku judge was not called (no API key supplied).
    failure_mode_hint : str or None
        Detected behavioral failure mode or None.
    behavioral_risk : float
        SC_OLD prefix score [0, 1] (always computed, $0).
    haiku_risk : float or None
        Haiku mid-chain judge risk [0, 1].
        None when Haiku was not called.
    rewritten_queries : list of str
        3 diverse query reformulations (paraphrase, decomposed, alternative).
        Populated only when abort=True and a QueryRewriter is available.
        Empty list when abort=False or no API key for QueryRewriter.
    latency_ms : float
        Total wall-clock time for this call in milliseconds.

    Two-stage OR strategy
    ---------------------
    Flag the chain as a failure if EITHER:
      (a) stream_guard() returns abort=True  (early warning at step 2), OR
      (b) final score_chain() returns needs_alert=True  (J5 at completion)

    Expected performance of two-stage OR vs J5-only (exp107/exp113):
      J5-only:          Precision=0.908, Recall=0.595
      Two-stage OR:     Precision=0.740, Recall=0.416   ← +Recall at lower P
      Two-stage boost:  Precision=0.809, Recall=0.191   ← upweight step-2 signal
    """

    abort: bool
    risk_at_step: float
    step_index: int
    on_track: Optional[bool]
    failure_mode_hint: Optional[str]
    behavioral_risk: float
    haiku_risk: Optional[float]
    rewritten_queries: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abort":               self.abort,
            "risk_at_step":        round(self.risk_at_step, 4),
            "step_index":          self.step_index,
            "on_track":            self.on_track,
            "failure_mode_hint":   self.failure_mode_hint,
            "behavioral_risk":     round(self.behavioral_risk, 4),
            "haiku_risk":          round(self.haiku_risk, 4) if self.haiku_risk is not None else None,
            "rewritten_queries":   self.rewritten_queries,
            "latency_ms":          round(self.latency_ms, 1),
        }


@dataclass
class MeshResult:
    """
    Result of AgentGuard.route_to_mesh() — conditional multi-agent consensus (exp115).

    When Agent A has LOW confidence, route_to_mesh() queries agents B and C in
    parallel, computes pairwise token-F1 agreement, and either upgrades the
    confidence tier (high agreement) or escalates to human review (very low
    agreement).

    Two-stage OR strategy with mesh:
      stream_guard() at step 2  →  score_chain() at finish  →  route_to_mesh() on LOW
      Expected AUROC: agreement standalone 0.7278; J5+agreement 0.7876 (exp108)

    Fields
    ------
    original_tier : str
        Agent A's confidence_tier before mesh consultation (should be "LOW").
    upgraded_tier : str
        Tier after mesh agreement: "MEDIUM" if agreement >= theta_high, else unchanged.
    agreement_score : float
        Mean pairwise token-F1 across all agent pairs [0, 1]. High = consensus.
    escalate_to_human : bool
        True when agreement < theta_low (agents fundamentally disagree — no consensus).
    consensus_answer : str or None
        Best-agreement answer from the mesh. The answer with highest mean pairwise F1
        to others. None when escalate_to_human=True.
    agent_answers : dict
        Raw answers from each agent {agent_id: answer_str}.
    pairwise_f1 : dict
        F1 between each pair {\"AB\": float, \"AC\": float, \"BC\": float}.
    theta_high : float
        Agreement threshold for tier upgrade. Default 0.60.
    theta_low : float
        Agreement threshold below which we escalate. Default 0.30.
    latency_ms : float
        Total wall-clock time for this call in milliseconds.
    """

    original_tier: str
    upgraded_tier: str
    agreement_score: float
    escalate_to_human: bool
    consensus_answer: Optional[str]
    agent_answers: Dict[str, str]
    pairwise_f1: Dict[str, float]
    theta_high: float = 0.60
    theta_low: float = 0.30
    latency_ms: float = 0.0

    @property
    def tier_upgraded(self) -> bool:
        """True when mesh agreement improved the confidence tier."""
        return self.upgraded_tier != self.original_tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tier":    self.original_tier,
            "upgraded_tier":    self.upgraded_tier,
            "agreement_score":  round(self.agreement_score, 4),
            "escalate_to_human": self.escalate_to_human,
            "consensus_answer": self.consensus_answer,
            "agent_answers":    self.agent_answers,
            "pairwise_f1":      {k: round(v, 4) for k, v in self.pairwise_f1.items()},
            "theta_high":       self.theta_high,
            "theta_low":        self.theta_low,
            "latency_ms":       round(self.latency_ms, 1),
        }
