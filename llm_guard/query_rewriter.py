"""
QueryRewriter — Diversify queries for Agent B when Agent A had low confidence.

This is the product-design fix for the A2A accuracy problem identified in exp105.

The root cause of correlated agent failures:
  - Agents share the same underlying model → same knowledge gaps
  - Agents use similar search strategies on the same question
  - Result: P(B fails | A fails) = 0.679 vs P(B fails | A correct) = 0.574
    (only +0.105 lift — barely above independent baseline)

The architectural fix (this module):
  When Agent A emits confidence_tier = "LOW" (risk ≥ 0.70), Agent B should NOT
  re-run the same query. Instead, QueryRewriter generates 3 diverse reformulations:

    1. Paraphrase      — same question, different wording, breaks lexical bias
    2. Decomposed      — split into 2–3 simpler sub-questions, attacks from parts
    3. Alternative     — completely different search angle or framing

  Expected gain: P(B fails | A fails) drops from 0.679 toward independent 0.574
  when Agent B uses diversified queries instead of the same approach.

Cost: ~$0.0005 per rewrite call (Haiku, ~300 output tokens).

Failure-mode-aware hints:
  The rewriter reads trust_obj.failure_mode to tailor the prompt:
    "retrieval_fail"  → emphasise finding a different source/angle
    "repeated_query"  → explicitly avoid the original search terms
    "long_chain"      → suggest a direct/concise approach
    "no_evidence"     → suggest checking foundational facts first
    default           → generic diversification

Usage
-----
    from llm_guard.query_rewriter import QueryRewriter
    from llm_guard.agent_guard import AgentGuard

    guard   = AgentGuard(api_key="sk-ant-...", use_judge=True)
    rewriter = QueryRewriter(api_key="sk-ant-...")

    trust = guard.generate_trust_object(question, steps, final_answer)

    if rewriter.should_rewrite(trust):
        variants = rewriter.rewrite(question, trust)
        # variants = [paraphrase, decomposed, alternative]
        # Pass variants[0] to Agent B as primary, variants[1] as fallback

    # Or use the convenience wrapper that does both:
    variants = rewriter.rewrite_if_needed(question, trust)
    # Returns [] when confidence_tier != "LOW" (no-op)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from llm_guard.trust_object import A2ATrustObject


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RewriteResult:
    """Result of QueryRewriter.rewrite()."""
    original_question: str
    variants: List[str]           # [paraphrase, decomposed, alternative]
    failure_mode_hint: str        # what failure mode drove the rewrite
    triggered_by_tier: str        # confidence tier that triggered the rewrite
    latency_ms: float = 0.0
    model_used: str = ""

    @property
    def paraphrase(self) -> str:
        """Direct rephrasing of the same question."""
        return self.variants[0] if len(self.variants) > 0 else self.original_question

    @property
    def decomposed(self) -> str:
        """Decomposed sub-question approach."""
        return self.variants[1] if len(self.variants) > 1 else self.original_question

    @property
    def alternative(self) -> str:
        """Alternative angle / framing."""
        return self.variants[2] if len(self.variants) > 2 else self.original_question


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are a search query strategist. Given a question that an AI agent failed to \
answer correctly, generate 3 diverse reformulations that Agent B should try instead.

The original agent likely failed because: {failure_hint}

Generate exactly 3 variants in JSON:
{{
  "paraphrase": "same intent, completely different wording",
  "decomposed": "break into a simpler first step or sub-question",
  "alternative": "completely different angle or framing of the same question"
}}

Rules:
- Each variant must be a complete, standalone question or search query
- paraphrase must NOT use any of the same key nouns as the original
- decomposed should be the most basic fact needed to answer the question
- alternative should approach from a different direction entirely
- Output ONLY the JSON object, nothing else"""

_FAILURE_HINTS = {
    "retrieval_fail":        "its searches found no relevant results — try different terminology or a different source angle",
    "repeated_query":        "it kept repeating the same search without progress — avoid any terms from the original query",
    "long_chain":            "it used too many steps and got lost — simplify to the single most important fact",
    "empty_answer":          "it could not produce a final answer — start with the most basic verifiable fact",
    "low_retrieval_quality": "its retrieved context was not relevant — try a more specific or more general framing",
    "no_evidence":           "searches returned results with no overlap to the question — try foundational background facts first",
    None:                    "it was uncertain — try a fresh perspective with different vocabulary and framing",
}


# ── QueryRewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """
    Generates diverse query reformulations when Agent A has low confidence.

    Uses Haiku by default (cheap, ~$0.0005/call) since the rewrite step runs
    before Agent B and adds latency. Sonnet can be used for higher quality.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model : str
        Model for rewriting. Default: Haiku (fast, cheap).
    risk_threshold : float
        Rewrite when risk_score >= this value. Default: 0.70 (tier LOW).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        risk_threshold: float = 0.70,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._risk_threshold = risk_threshold
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ── Public API ────────────────────────────────────────────────────────────

    def should_rewrite(self, trust: A2ATrustObject) -> bool:
        """
        Return True when Agent B should use rewritten queries.

        Triggers when:
          - confidence_tier == "LOW"  (risk >= 0.70), OR
          - judge_label == "POOR", OR
          - temporal validity tv_risk > 0.5 (stale answer likely)
        """
        if trust.confidence_tier == "LOW":
            return True
        if trust.judge_label == "POOR":
            return True
        if trust.temporal_validity and trust.temporal_validity.tv_risk > 0.5:
            return True
        return False

    def rewrite(
        self,
        question: str,
        trust: A2ATrustObject,
        n_variants: int = 3,
    ) -> RewriteResult:
        """
        Generate diverse query reformulations for Agent B.

        Parameters
        ----------
        question : str
            The original question Agent A tried to answer.
        trust : A2ATrustObject
            The trust object emitted by Agent A (used for failure hint).
        n_variants : int
            Number of variants to return. Default: 3 (paraphrase, decomposed, alternative).

        Returns
        -------
        RewriteResult
            Contains variants list plus metadata about the rewrite.
        """
        failure_hint = _FAILURE_HINTS.get(trust.failure_mode, _FAILURE_HINTS[None])

        # Add temporal validity context if relevant
        if trust.temporal_validity and trust.temporal_validity.is_time_sensitive:
            failure_hint += " Also note: this question is time-sensitive — explicitly look for recent information."

        system = _SYSTEM.format(failure_hint=failure_hint)
        user_msg = f"Original question: {question}"

        t0 = time.time()
        client = self._get_client()

        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=300,
                temperature=0.7,   # some randomness for diversity
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            latency = (time.time() - t0) * 1000

            variants = self._parse_variants(raw, question)

        except Exception as e:
            # Fallback: return simple heuristic reformulations
            latency = (time.time() - t0) * 1000
            variants = self._heuristic_fallback(question, trust.failure_mode)

        return RewriteResult(
            original_question=question,
            variants=variants[:n_variants],
            failure_mode_hint=failure_hint,
            triggered_by_tier=trust.confidence_tier,
            latency_ms=round(latency, 1),
            model_used=self._model,
        )

    def rewrite_if_needed(
        self,
        question: str,
        trust: A2ATrustObject,
    ) -> List[str]:
        """
        Convenience wrapper: rewrite only when should_rewrite() is True.

        Returns [] when no rewrite is needed (Agent A had HIGH/MEDIUM confidence).
        Returns list of variant strings when rewrite is triggered.

        Usage
        -----
            variants = rewriter.rewrite_if_needed(question, trust)
            if variants:
                agent_b_query = variants[0]   # use paraphrase as primary
            else:
                agent_b_query = question      # use original directly
        """
        if not self.should_rewrite(trust):
            return []
        result = self.rewrite(question, trust)
        return result.variants

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_variants(self, raw: str, fallback_question: str) -> List[str]:
        """Parse JSON response into list of variant strings."""
        # Extract JSON block (handle markdown fences)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return self._heuristic_fallback(fallback_question)

        try:
            data = json.loads(m.group())
            variants = [
                data.get("paraphrase", fallback_question),
                data.get("decomposed", fallback_question),
                data.get("alternative", fallback_question),
            ]
            # Filter empty / identical to original
            out = []
            for v in variants:
                v = str(v).strip()
                if v and v != fallback_question:
                    out.append(v)
            # Pad with fallback if needed
            while len(out) < 3:
                out.append(fallback_question)
            return out
        except (json.JSONDecodeError, KeyError):
            return self._heuristic_fallback(fallback_question)

    @staticmethod
    def _heuristic_fallback(question: str, failure_mode: Optional[str] = None) -> List[str]:
        """
        Zero-API fallback when the Haiku call fails.
        Returns simple but useful reformulations without any API call.
        """
        q = question.strip().rstrip("?")

        # Paraphrase: add "Please explain" framing
        paraphrase = f"What is the answer to: {q}?"

        # Decomposed: strip to key nouns
        # Simple heuristic: take first 6 words as a focused query
        words = q.split()
        core = " ".join(words[:min(6, len(words))])
        decomposed = f"What is {core}?"

        # Alternative: invert the question
        alternative = f"Provide background information about: {q}"

        return [paraphrase, decomposed, alternative]
