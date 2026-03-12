"""
llm-guard-kit framework integrations.

Drop-in callbacks and handlers for popular agent frameworks, observability
platforms, and metrics exporters. Each integration captures reasoning steps
automatically and scores the completed chain via AgentGuard.

Agent Framework Integrations
-----------------------------
    from llm_guard.integrations.langchain import AgentGuardCallback
    from llm_guard.integrations.llamaindex import AgentGuardEventHandler
    from llm_guard.integrations.crewai import AgentGuardCrewCallback

Observability / Tracing Integrations
--------------------------------------
    from llm_guard.integrations.langfuse_integration import LangfuseGuard, LangfuseGuardCallback
    from llm_guard.integrations.langsmith_integration import LangSmithGuardEvaluator

Metrics / Monitoring Integrations
------------------------------------
    from llm_guard.integrations.prometheus_integration import PrometheusMetricsExporter
    from llm_guard.integrations.prometheus_integration import make_grafana_dashboard_json
    from llm_guard.integrations.datadog_integration import DatadogGuard

All integrations lazy-import their optional dependencies and raise a clear
ImportError with install instructions when those dependencies are missing.
"""
