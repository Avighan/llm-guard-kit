"""
AdapterRegistry — Failure-mode-aware adapter selection for LLM agents (exp116).

Maps detected failure modes (from LabelFreeScorer) to adapter configurations.
In a full LoRA deployment, each adapter corresponds to a fine-tuned set of
low-rank weight updates that specialises the base model for a specific failure
recovery strategy.

In deployment without open-weight models, the registry still provides:
  1. Prompt-level adapter configs (system prompt injection, search strategy hints)
  2. Routing config for which downstream model or strategy to use
  3. Structured logging of which adapter was activated per chain

Adapter types (exp116)
----------------------
  retrieval_fail    Searches found no results → use broader/alternative terminology
  repeated_query    Agent repeated same search → force query deduplication constraint
  long_chain        4+ steps, getting lost → direct/concise single-fact approach
  empty_answer      No final answer produced → ground to most basic verifiable fact
  default           No specific failure detected → mild confidence-boosting hint

EHC cross-research note
------------------------
  Adapters are EHC "reusable primitives" — specialised micro-modules selected
  by a router (AgentGuard). The registry implements EHC's core abstraction:
    intelligence = hierarchical compression into reusable primitives
  Each adapter is a compressed representation of a failure-recovery strategy.

ECL cross-research note
-----------------------
  Adapter activation mirrors ECL's homeostatic drive mechanism.  When a specific
  failure mode is detected, the corresponding adapter is activated (drive fired),
  analogous to ECL selecting the appropriate consolidation strategy based on
  energy state.

Usage
-----
    from llm_guard.adapter_registry import AdapterRegistry

    registry = AdapterRegistry()
    config   = registry.get(failure_mode="retrieval_fail")
    print(config.system_hint)        # prompt injection text
    print(config.search_strategy)    # "broaden_query" / "decompose" / etc.
    print(config.adapter_id)         # "retrieval_fail_v1"

    # Or use via AgentGuard:
    result = guard.score_chain(question, steps, final_answer)
    config = guard.activate_adapter(result.failure_mode)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ── AdapterConfig ─────────────────────────────────────────────────────────────

@dataclass
class AdapterConfig:
    """
    Configuration for a single failure-recovery adapter.

    In a LoRA deployment, adapter_id references the saved weights directory.
    In a prompt-only deployment, system_hint and search_strategy provide
    equivalent guidance without any fine-tuned weights.

    Fields
    ------
    adapter_id : str
        Unique identifier for this adapter (e.g. "retrieval_fail_v1").
    failure_mode : str
        The failure mode this adapter targets.
    system_hint : str
        System-prompt injection text to steer the agent toward recovery.
        Prepend to the agent's system prompt when this adapter is active.
    search_strategy : str
        Recommended search strategy for this failure mode.
          "broaden_query"   — use more general terms
          "decompose"       — break question into sub-facts
          "deduplicate"     — avoid repeating prior queries
          "direct_fact"     — ask for single most important fact
          "foundational"    — check basic background facts first
          "verify_claim"    — search for a source that confirms the specific claim
          "default"         — no special search constraint
    temperature_delta : float
        Suggested temperature adjustment from base (positive = more creative).
        Useful for retrieval_fail (+0.2) and empty_answer (+0.1).
    max_steps_override : int or None
        Override max search steps if set. long_chain adapter caps at 2.
    priority : int
        Activation priority when multiple failure modes detected (lower = higher).
    lora_weights_path : str or None
        Path to LoRA weights if available. None = prompt-only mode.
    """
    adapter_id: str
    failure_mode: str
    system_hint: str
    search_strategy: str
    temperature_delta: float = 0.0
    max_steps_override: Optional[int] = None
    priority: int = 5
    lora_weights_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── AdapterResult ──────────────────────────────────────────────────────────────

@dataclass
class AdapterResult:
    """
    Result of AdapterRegistry.get() — selected adapter + activation metadata.

    Fields
    ------
    activated : bool
        True when a specific (non-default) adapter was selected.
    config : AdapterConfig
        The selected adapter configuration.
    failure_mode_input : str or None
        The failure_mode string passed to get().
    fallback : bool
        True when the default adapter was used (no specific adapter matched).
    """
    activated: bool
    config: AdapterConfig
    failure_mode_input: Optional[str]
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activated":          self.activated,
            "config":             self.config.to_dict(),
            "failure_mode_input": self.failure_mode_input,
            "fallback":           self.fallback,
        }


# ── Default adapter configs ────────────────────────────────────────────────────

_DEFAULT_ADAPTERS: List[AdapterConfig] = [
    AdapterConfig(
        adapter_id="retrieval_fail_v1",
        failure_mode="retrieval_fail",
        system_hint=(
            "Previous searches returned no relevant results. "
            "Use DIFFERENT terminology — synonyms, broader terms, or related concepts. "
            "Avoid repeating any term used in prior searches."
        ),
        search_strategy="broaden_query",
        temperature_delta=0.2,
        max_steps_override=None,
        priority=1,
    ),
    AdapterConfig(
        adapter_id="repeated_query_v1",
        failure_mode="repeated_query",
        system_hint=(
            "You previously issued the same search query more than once without progress. "
            "You MUST NOT repeat any previous search query. "
            "Before each search, check your prior queries and ensure the new query is distinct."
        ),
        search_strategy="deduplicate",
        temperature_delta=0.0,
        max_steps_override=None,
        priority=2,
    ),
    AdapterConfig(
        adapter_id="long_chain_v1",
        failure_mode="long_chain",
        system_hint=(
            "You are using too many steps. Focus on the single most important fact needed "
            "to answer the question. Answer directly after finding that one fact — "
            "do not continue searching."
        ),
        search_strategy="direct_fact",
        temperature_delta=0.0,
        max_steps_override=2,
        priority=3,
    ),
    AdapterConfig(
        adapter_id="empty_answer_v1",
        failure_mode="empty_answer",
        system_hint=(
            "You failed to produce a final answer. Start by identifying the single most "
            "basic, verifiable fact about this topic. Once you find any concrete fact, "
            "use it to construct a partial answer rather than searching indefinitely."
        ),
        search_strategy="foundational",
        temperature_delta=0.1,
        max_steps_override=None,
        priority=4,
    ),
    AdapterConfig(
        adapter_id="low_retrieval_quality_v1",
        failure_mode="low_retrieval_quality",
        system_hint=(
            "Retrieved context has low relevance to the question. "
            "Try a more specific search using proper nouns and exact names, "
            "or a more general search using only the core concept."
        ),
        search_strategy="decompose",
        temperature_delta=0.1,
        max_steps_override=None,
        priority=2,
    ),
    AdapterConfig(
        adapter_id="no_evidence_v1",
        failure_mode="no_evidence",
        system_hint=(
            "Your searches returned results with no overlap to the question. "
            "Start with foundational background facts before targeting the "
            "specific question. Build up from basics."
        ),
        search_strategy="foundational",
        temperature_delta=0.0,
        max_steps_override=None,
        priority=2,
    ),
    AdapterConfig(
        adapter_id="confident_wrong_v1",
        failure_mode="confident_wrong",
        system_hint=(
            "Your previous answer was made with very few reasoning steps and no expressed uncertainty. "
            "This pattern often indicates overconfidence. "
            "Verify your answer explicitly: search for a source that CONFIRMS the specific claim in your answer. "
            "If you cannot verify it, express uncertainty and try an alternative."
        ),
        search_strategy="verify_claim",
        temperature_delta=0.0,
        max_steps_override=3,
        priority=1,
    ),
    AdapterConfig(
        adapter_id="default_v1",
        failure_mode="default",
        system_hint=(
            "Consider whether you have enough evidence to answer confidently. "
            "If uncertain, try a different search angle before finalising."
        ),
        search_strategy="default",
        temperature_delta=0.0,
        max_steps_override=None,
        priority=10,
    ),
]


# ── AdapterRegistry ────────────────────────────────────────────────────────────

class AdapterRegistry:
    """
    Registry mapping failure modes to adapter configurations.

    The registry ships with 7 built-in adapters (retrieval_fail, repeated_query,
    long_chain, empty_answer, low_retrieval_quality, no_evidence, confident_wrong) plus a default.
    Custom adapters can be registered at runtime via register().

    Parameters
    ----------
    adapters : list of AdapterConfig, optional
        Overrides the default adapter set. Merged with defaults if merge=True.
    merge : bool
        When True (default), custom adapters are merged with built-ins.
        When False, only the supplied adapters are used.
    """

    def __init__(
        self,
        adapters: Optional[List[AdapterConfig]] = None,
        merge: bool = True,
    ):
        self._adapters: Dict[str, AdapterConfig] = {}

        # Load defaults first
        if merge or adapters is None:
            for a in _DEFAULT_ADAPTERS:
                self._adapters[a.failure_mode] = a

        # Override/add custom adapters
        if adapters:
            for a in adapters:
                self._adapters[a.failure_mode] = a

    def register(self, config: AdapterConfig) -> None:
        """Register a custom adapter, overriding existing entry for that failure_mode."""
        self._adapters[config.failure_mode] = config

    def get(self, failure_mode: Optional[str]) -> AdapterResult:
        """
        Select the adapter for a given failure mode.

        Returns the default adapter when failure_mode is None or not registered.

        Parameters
        ----------
        failure_mode : str or None
            Failure mode string from LabelFreeScorer / ChainTrustResult.

        Returns
        -------
        AdapterResult
        """
        if failure_mode and failure_mode in self._adapters:
            config   = self._adapters[failure_mode]
            fallback = False
            activated = True
        else:
            config   = self._adapters.get("default", _DEFAULT_ADAPTERS[-1])
            fallback = True
            activated = failure_mode is not None  # activated=True if failure_mode was set, just no specific adapter

        return AdapterResult(
            activated=activated,
            config=config,
            failure_mode_input=failure_mode,
            fallback=fallback,
        )

    def list_adapters(self) -> List[str]:
        """Return all registered failure mode keys."""
        return [k for k in self._adapters if k != "default"]

    def __repr__(self) -> str:
        return f"AdapterRegistry(adapters={self.list_adapters()})"
