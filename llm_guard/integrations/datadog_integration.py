"""
Datadog Integration for llm-guard-kit
=======================================
Emits llm-guard risk scores as Datadog custom metrics via DogStatsD.

Usage
-----
    from llm_guard.integrations.datadog_integration import DatadogGuard

    guard = DatadogGuard(
        dd_api_key="your-datadog-api-key",   # or set DD_API_KEY env var
        tags=["env:production", "service:my-agent"],
    )

    result = guard.score_chain(
        question="Who founded Tesla?",
        steps=[...],
        final_answer="Elon Musk",
        domain="factual_qa",
    )
    # Emits to Datadog (via DogStatsD on localhost:8125):
    #   llm_guard.risk_score       (gauge)   = 0.42  tags=[..., domain:factual_qa, tier:medium]
    #   llm_guard.behavioral_score (gauge)   = 0.38  tags=[...]
    #   llm_guard.chains           (count)   = 1     tags=[...]
    #   llm_guard.alerts           (count)   = 0|1   tags=[...] (only emitted when needs_alert=True)
    #   llm_guard.tier             (gauge)   = 0|1|2 (0=HIGH, 1=MEDIUM, 2=LOW)

Backend selection
-----------------
    Priority: datadog DogStatsD library → plain UDP statsd → no-op (with warning).

    - datadog library (``pip install datadog``): full DogStatsD including tag support.
    - statsd library (``pip install statsd``): plain statsd without native tag support
      (tags are appended to metric names).
    - No library installed: metrics are silently dropped; a one-time warning is logged.

Requirements (optional)
-----------------------
    pip install datadog     # recommended — DogStatsD with tag support
    # OR
    pip install statsd      # plain statsd fallback
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection (lazy, done once on first use)
# ---------------------------------------------------------------------------

_BACKEND: Optional[str] = None  # "datadog" | "statsd" | "noop"
_statsd_client = None           # the connected client, set by _init_client()
_init_lock = __import__("threading").Lock()


def _init_client(
    dd_api_key: Optional[str],
    statsd_host: str,
    statsd_port: int,
    tags: Optional[List[str]],
    namespace: str,
):
    """
    Attempt to initialise the best available statsd backend.
    Sets module-level _BACKEND and _statsd_client.
    Thread-safe; idempotent after first call.
    """
    global _BACKEND, _statsd_client

    with _init_lock:
        if _BACKEND is not None:
            return  # already initialised

        # 1. Try datadog library (DogStatsD)
        try:
            from datadog import initialize, statsd as dd_statsd
            init_kwargs: Dict[str, Any] = {
                "statsd_host": statsd_host,
                "statsd_port": statsd_port,
            }
            if dd_api_key:
                init_kwargs["api_key"] = dd_api_key
            initialize(**init_kwargs)
            _statsd_client = dd_statsd
            _BACKEND = "datadog"
            logger.debug("DatadogGuard: using datadog DogStatsD backend (%s:%d)", statsd_host, statsd_port)
            return
        except ImportError:
            pass

        # 2. Fall back to plain statsd library
        try:
            import statsd as _statsd_lib
            _statsd_client = _statsd_lib.StatsClient(
                host=statsd_host, port=statsd_port, prefix=namespace
            )
            _BACKEND = "statsd"
            logger.warning(
                "DatadogGuard: datadog library not found; falling back to plain statsd "
                "(no tag support). Install datadog for full Datadog integration: "
                "pip install datadog"
            )
            return
        except ImportError:
            pass

        # 3. No-op backend
        _BACKEND = "noop"
        logger.warning(
            "DatadogGuard: neither datadog nor statsd library found. "
            "Metrics will be silently dropped. "
            "Install with: pip install datadog"
        )


# ---------------------------------------------------------------------------
# Metric emission helpers
# ---------------------------------------------------------------------------

def _emit_gauge(name: str, value: float, tags: List[str]) -> None:
    """Emit a gauge metric to the configured backend."""
    if _BACKEND == "datadog":
        try:
            _statsd_client.gauge(name, value, tags=tags)
        except Exception as exc:
            logger.debug("DatadogGuard emit_gauge failed: %s", exc)

    elif _BACKEND == "statsd":
        # Plain statsd: encode tags in metric name (Graphite-style)
        tag_suffix = "." + ".".join(t.replace(":", "_") for t in tags) if tags else ""
        try:
            _statsd_client.gauge(name + tag_suffix, value)
        except Exception as exc:
            logger.debug("DatadogGuard emit_gauge (statsd) failed: %s", exc)
    # noop: do nothing


def _emit_count(name: str, value: int, tags: List[str]) -> None:
    """Emit a count/increment metric to the configured backend."""
    if _BACKEND == "datadog":
        try:
            _statsd_client.increment(name, value=value, tags=tags)
        except Exception as exc:
            logger.debug("DatadogGuard emit_count failed: %s", exc)

    elif _BACKEND == "statsd":
        tag_suffix = "." + ".".join(t.replace(":", "_") for t in tags) if tags else ""
        try:
            _statsd_client.incr(name + tag_suffix, count=value)
        except Exception as exc:
            logger.debug("DatadogGuard emit_count (statsd) failed: %s", exc)
    # noop: do nothing


# ---------------------------------------------------------------------------
# Tier encoding helper
# ---------------------------------------------------------------------------

_TIER_VALUE: Dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


# ---------------------------------------------------------------------------
# DatadogGuard
# ---------------------------------------------------------------------------

class DatadogGuard:
    """
    Drop-in replacement for AgentGuard that emits Datadog metrics on every
    score_chain() call.

    All other AgentGuard methods (generate_trust_object, monitor_step, etc.)
    are delegated transparently via ``__getattr__``.

    Parameters
    ----------
    dd_api_key : str, optional
        Datadog API key. If omitted, reads from ``DD_API_KEY`` environment variable.
        Required by the datadog library for API calls; not needed for DogStatsD-only use.
    tags : list of str, optional
        Global tags attached to every metric, e.g. ["env:prod", "service:my-agent"].
    statsd_host : str
        DogStatsD agent host. Default "localhost".
    statsd_port : int
        DogStatsD agent port. Default 8125.
    namespace : str
        Metric name prefix. Default "llm_guard".
    guard : AgentGuard, optional
        Pre-configured AgentGuard to wrap. Created with ``**guard_kwargs`` if omitted.
    **guard_kwargs
        Passed to AgentGuard() when no guard is provided.
        Example: ``api_key="sk-ant-..."`` to enable the Sonnet judge.

    Examples
    --------
    ::

        guard = DatadogGuard(tags=["env:production", "service:qa-bot"])
        result = guard.score_chain(
            question="...", steps=[...], final_answer="...",
            domain="factual_qa",
        )
        guard.flush()  # call before process exit

    ::

        # Async-friendly: flush explicitly when done
        guard = DatadogGuard(dd_api_key=os.environ["DD_API_KEY"])
        for q, steps, ans in my_dataset:
            guard.score_chain(q, steps, ans, domain="batch_eval")
        guard.flush()
    """

    def __init__(
        self,
        dd_api_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        statsd_host: str = "localhost",
        statsd_port: int = 8125,
        namespace: str = "llm_guard",
        guard=None,
        **guard_kwargs,
    ):
        self._dd_api_key = dd_api_key or os.environ.get("DD_API_KEY") or ""
        self._global_tags: List[str] = list(tags or [])
        self._statsd_host = statsd_host
        self._statsd_port = statsd_port
        self._namespace = namespace

        # Initialise the statsd backend (deferred — no-op if library absent)
        _init_client(
            dd_api_key=self._dd_api_key,
            statsd_host=statsd_host,
            statsd_port=statsd_port,
            tags=self._global_tags,
            namespace=namespace,
        )

        if guard is not None:
            self._guard = guard
        else:
            from llm_guard import AgentGuard
            self._guard = AgentGuard(**guard_kwargs)

    # ------------------------------------------------------------------
    # score_chain() override — scores + emits metrics
    # ------------------------------------------------------------------

    def score_chain(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
        domain: str = "default",
    ):
        """
        Score a completed agent chain and emit Datadog metrics.

        Parameters
        ----------
        question : str
            The question / task given to the agent.
        steps : list of dict
            Agent reasoning steps.
        final_answer : str
            The agent's final output.
        domain : str
            Domain tag attached to every metric. Default "default".

        Returns
        -------
        ChainTrustResult
            Scoring result from the underlying AgentGuard.
        """
        result = self._guard.score_chain(
            question=question,
            steps=steps,
            final_answer=final_answer,
        )

        self._emit(result, domain=domain)
        return result

    # ------------------------------------------------------------------
    # Internal metric emission
    # ------------------------------------------------------------------

    def _emit(self, result, domain: str) -> None:
        """Emit all metrics for one ChainTrustResult."""
        tags = self._build_tags(result, domain)

        # Gauges
        _emit_gauge(f"{self._namespace}.risk_score",       result.risk_score,       tags)
        _emit_gauge(f"{self._namespace}.behavioral_score", result.behavioral_score, tags)
        _emit_gauge(
            f"{self._namespace}.tier",
            float(_TIER_VALUE.get(result.confidence_tier, 1)),
            tags,
        )

        # Counters
        _emit_count(f"{self._namespace}.chains", 1, tags)
        if result.needs_alert:
            _emit_count(f"{self._namespace}.alerts", 1, tags)

        # Optional: step count as a gauge (useful for debugging chain length)
        if hasattr(result, "step_count"):
            _emit_gauge(f"{self._namespace}.step_count", float(result.step_count), tags)

    def _build_tags(self, result, domain: str) -> List[str]:
        """Build the full tag list for this result."""
        tags = list(self._global_tags)
        tags.append(f"domain:{domain}")
        tags.append(f"tier:{result.confidence_tier.lower()}")
        if result.failure_mode:
            # Sanitise failure mode for Datadog tag constraints (max 200 chars, no commas)
            fm = result.failure_mode.replace(",", "_")[:100]
            tags.append(f"failure_mode:{fm}")
        tags.append(f"needs_alert:{'true' if result.needs_alert else 'false'}")
        return tags

    # ------------------------------------------------------------------
    # flush() — ensure delivery before process exit
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Flush buffered metrics to Datadog.

        The DogStatsD client uses UDP (fire-and-forget) by default, so flushing
        is usually unnecessary. If you use the datadog library with custom
        buffering, call this before process exit to ensure delivery.
        """
        if _BACKEND == "datadog":
            try:
                # datadog statsd client doesn't expose flush(), but we can
                # call socket close/reopen pattern if available
                if hasattr(_statsd_client, "flush"):
                    _statsd_client.flush()
            except Exception as exc:
                logger.debug("DatadogGuard.flush() encountered: %s", exc)
        elif _BACKEND == "statsd":
            try:
                if hasattr(_statsd_client, "pipeline"):
                    pass  # plain statsd sends synchronously; nothing to flush
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Transparent delegation to the underlying AgentGuard
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """
        Delegate any attribute not defined on DatadogGuard to the wrapped
        AgentGuard, making DatadogGuard a transparent drop-in replacement.
        """
        return getattr(self._guard, name)

    def __repr__(self) -> str:
        return (
            f"DatadogGuard(backend={_BACKEND!r}, "
            f"namespace={self._namespace!r}, "
            f"tags={self._global_tags!r})"
        )
