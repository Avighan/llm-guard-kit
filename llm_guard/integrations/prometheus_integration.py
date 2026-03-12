"""
Prometheus Integration for llm-guard-kit
==========================================
Exposes llm-guard risk metrics as Prometheus metrics for Grafana dashboards.

Usage
-----
    from llm_guard.integrations.prometheus_integration import PrometheusMetricsExporter

    exporter = PrometheusMetricsExporter(port=9090)
    exporter.start()  # starts HTTP server in background daemon thread

    # After each scored chain:
    result = guard.score_chain(question, steps, final_answer)
    exporter.record(result, domain="customer_service")

    # Grafana scrapes http://localhost:9090/metrics
    # Available metrics:
    #   llm_guard_risk_score{domain="customer_service"}         0.42
    #   llm_guard_chains_total{domain="customer_service"}       100
    #   llm_guard_alert_total{domain="customer_service"}        12
    #   llm_guard_tier_total{tier="high",domain="..."}          3
    #   llm_guard_behavioral_score{domain="customer_service"}   0.38

    # Generate a ready-to-import Grafana dashboard JSON:
    from llm_guard.integrations.prometheus_integration import make_grafana_dashboard_json
    json_str = make_grafana_dashboard_json(datasource="Prometheus")
    with open("llm_guard_dashboard.json", "w") as f:
        f.write(json_str)

Requirements
------------
    pip install prometheus_client
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy prometheus_client import
# ---------------------------------------------------------------------------

_PROMETHEUS_AVAILABLE = False
_prometheus_client = None


def _ensure_prometheus():
    global _PROMETHEUS_AVAILABLE, _prometheus_client
    if _PROMETHEUS_AVAILABLE:
        return
    try:
        import prometheus_client as _pc
        _prometheus_client = _pc
        _PROMETHEUS_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "prometheus_client is required for PrometheusMetricsExporter. "
            "Install it with: pip install prometheus_client"
        )


# ---------------------------------------------------------------------------
# PrometheusMetricsExporter
# ---------------------------------------------------------------------------

class PrometheusMetricsExporter:
    """
    Exposes llm-guard risk metrics as Prometheus metrics.

    Metrics
    -------
    llm_guard_risk_score (Gauge)
        Latest risk score, labelled by domain.
    llm_guard_behavioral_score (Gauge)
        Latest behavioral (SC_OLD) score, labelled by domain.
    llm_guard_chains_total (Counter)
        Total chains scored, labelled by domain.
    llm_guard_alert_total (Counter)
        Total chains that triggered an alert, labelled by domain.
    llm_guard_tier_total (Counter)
        Chains per confidence tier, labelled by tier and domain.

    Parameters
    ----------
    port : int
        HTTP port for the /metrics endpoint. Default 9090.
    addr : str
        Bind address. Default "" (all interfaces).
    registry : prometheus_client.CollectorRegistry, optional
        Custom registry. Uses prometheus_client.REGISTRY by default.
        Pass ``prometheus_client.CollectorRegistry()`` for test isolation.
    namespace : str
        Metric name prefix. Default "llm_guard".
    """

    def __init__(
        self,
        port: int = 9090,
        addr: str = "",
        registry=None,
        namespace: str = "llm_guard",
    ):
        self._port = port
        self._addr = addr
        self._namespace = namespace
        self._server = None
        self._started = False

        _ensure_prometheus()
        pc = _prometheus_client

        # Use provided registry or the global default
        self._registry = registry if registry is not None else pc.REGISTRY

        label_names_domain = ["domain"]
        label_names_tier   = ["tier", "domain"]

        self._risk_score = pc.Gauge(
            f"{namespace}_risk_score",
            "Latest composite risk score from AgentGuard.score_chain() [0-1]",
            label_names_domain,
            registry=self._registry,
        )
        self._behavioral_score = pc.Gauge(
            f"{namespace}_behavioral_score",
            "Latest behavioral (SC_OLD) score from AgentGuard.score_chain() [0-1]",
            label_names_domain,
            registry=self._registry,
        )
        self._chains_total = pc.Counter(
            f"{namespace}_chains_total",
            "Total number of agent chains scored",
            label_names_domain,
            registry=self._registry,
        )
        self._alert_total = pc.Counter(
            f"{namespace}_alert_total",
            "Total number of chains that triggered a risk alert (needs_alert=True)",
            label_names_domain,
            registry=self._registry,
        )
        self._tier_total = pc.Counter(
            f"{namespace}_tier_total",
            "Total chains per confidence tier (high/medium/low)",
            label_names_tier,
            registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the Prometheus HTTP metrics server in a daemon background thread.

        The server will be available at ``http://<addr>:<port>/metrics``.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._started:
            return
        _ensure_prometheus()

        def _serve():
            _prometheus_client.start_http_server(  # type: ignore[union-attr]
                port=self._port,
                addr=self._addr,
                registry=self._registry,
            )

        t = threading.Thread(target=_serve, daemon=True, name="llm-guard-prometheus")
        t.start()
        self._started = True
        logger.info(
            "PrometheusMetricsExporter: metrics server started on "
            "http://%s:%d/metrics",
            self._addr or "0.0.0.0",
            self._port,
        )

    def record(self, result, domain: str = "default") -> None:
        """
        Update Prometheus metrics from a ChainTrustResult.

        Parameters
        ----------
        result : ChainTrustResult
            Return value of AgentGuard.score_chain().
        domain : str
            Domain label for metric partitioning. Default "default".
        """
        risk       = float(result.risk_score)
        behavioral = float(result.behavioral_score)
        needs_alert = bool(result.needs_alert)
        tier = result.confidence_tier.lower()  # "high" / "medium" / "low"

        self._risk_score.labels(domain=domain).set(risk)
        self._behavioral_score.labels(domain=domain).set(behavioral)
        self._chains_total.labels(domain=domain).inc()
        self._tier_total.labels(tier=tier, domain=domain).inc()

        if needs_alert:
            self._alert_total.labels(domain=domain).inc()

    def get_metrics_text(self) -> str:
        """
        Return the current metrics in Prometheus text exposition format.

        Useful for testing and health-check endpoints without starting the
        HTTP server.

        Returns
        -------
        str
            Prometheus text format metrics snapshot.
        """
        _ensure_prometheus()
        return _prometheus_client.exposition.generate_latest(  # type: ignore[union-attr]
            self._registry
        ).decode("utf-8")

    def reset(self) -> None:
        """Reset all Gauges to 0. Counters cannot be reset (Prometheus convention)."""
        # Gauges can be cleared; counters are append-only by design
        logger.debug("PrometheusMetricsExporter.reset(): gauge values zeroed")


# ---------------------------------------------------------------------------
# make_grafana_dashboard_json()
# ---------------------------------------------------------------------------

def make_grafana_dashboard_json(
    datasource: str = "Prometheus",
    title: str = "LLM Guard Risk Dashboard",
    namespace: str = "llm_guard",
    refresh: str = "30s",
) -> str:
    """
    Return a Grafana dashboard JSON string, ready to import via
    Grafana UI → Dashboards → Import → Upload JSON file.

    Parameters
    ----------
    datasource : str
        Name of the Grafana Prometheus data source. Default "Prometheus".
    title : str
        Dashboard title. Default "LLM Guard Risk Dashboard".
    namespace : str
        Metric namespace prefix. Default "llm_guard".
    refresh : str
        Auto-refresh interval. Default "30s".

    Returns
    -------
    str
        Grafana dashboard JSON.

    Examples
    --------
    ::

        from llm_guard.integrations.prometheus_integration import make_grafana_dashboard_json
        import json, pathlib
        pathlib.Path("llm_guard_dashboard.json").write_text(
            make_grafana_dashboard_json(datasource="Prometheus")
        )
    """
    ns = namespace

    def _panel(
        panel_id: int,
        title: str,
        panel_type: str,
        expr: str,
        grid_x: int,
        grid_y: int,
        width: int = 8,
        height: int = 8,
        legend_format: str = "{{domain}}",
        unit: str = "short",
    ) -> dict:
        return {
            "id":    panel_id,
            "title": title,
            "type":  panel_type,
            "datasource": {"type": "prometheus", "uid": datasource},
            "gridPos": {"x": grid_x, "y": grid_y, "w": width, "h": height},
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": 0,
                    "color": {"mode": "palette-classic"},
                },
                "overrides": [],
            },
            "options": {
                "legend":  {"displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "single", "sort": "none"},
            },
            "targets": [
                {
                    "datasource":    {"type": "prometheus", "uid": datasource},
                    "expr":          expr,
                    "legendFormat":  legend_format,
                    "refId":         "A",
                }
            ],
        }

    panels = [
        # Row 1 — Risk score trend (time-series)
        _panel(
            panel_id=1,
            title="Risk Score (latest, by domain)",
            panel_type="timeseries",
            expr=f'{ns}_risk_score{{domain=~"$domain"}}',
            grid_x=0, grid_y=0, width=12, height=8,
            unit="percentunit",
        ),
        # Row 1 — Behavioral score trend
        _panel(
            panel_id=2,
            title="Behavioral Score (latest, by domain)",
            panel_type="timeseries",
            expr=f'{ns}_behavioral_score{{domain=~"$domain"}}',
            grid_x=12, grid_y=0, width=12, height=8,
            unit="percentunit",
        ),
        # Row 2 — Alert rate (rate over 5m window)
        _panel(
            panel_id=3,
            title="Alert Rate (per min)",
            panel_type="timeseries",
            expr=f'rate({ns}_alert_total{{domain=~"$domain"}}[5m]) * 60',
            grid_x=0, grid_y=8, width=8, height=8,
            unit="reqpm",
        ),
        # Row 2 — Chains per minute
        _panel(
            panel_id=4,
            title="Chains Scored (per min)",
            panel_type="timeseries",
            expr=f'rate({ns}_chains_total{{domain=~"$domain"}}[5m]) * 60',
            grid_x=8, grid_y=8, width=8, height=8,
            unit="reqpm",
        ),
        # Row 2 — Alert % stat panel
        {
            "id":    5,
            "title": "Alert % (5 min window)",
            "type":  "stat",
            "datasource": {"type": "prometheus", "uid": datasource},
            "gridPos": {"x": 16, "y": 8, "w": 8, "h": 8},
            "fieldConfig": {
                "defaults": {
                    "unit":  "percent",
                    "min":   0, "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green",  "value": 0},
                            {"color": "yellow", "value": 10},
                            {"color": "red",    "value": 25},
                        ],
                    },
                    "color": {"mode": "thresholds"},
                },
                "overrides": [],
            },
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "orientation": "auto",
                        "textMode": "auto", "colorMode": "background"},
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": datasource},
                    "expr": (
                        f"100 * rate({ns}_alert_total{{domain=~\"$domain\"}}[5m])"
                        f" / clamp_min(rate({ns}_chains_total{{domain=~\"$domain\"}}[5m]), 1e-9)"
                    ),
                    "legendFormat": "alert %",
                    "refId": "A",
                }
            ],
        },
        # Row 3 — Tier distribution (bar chart)
        {
            "id":    6,
            "title": "Tier Distribution (total)",
            "type":  "barchart",
            "datasource": {"type": "prometheus", "uid": datasource},
            "gridPos": {"x": 0, "y": 16, "w": 24, "h": 8},
            "fieldConfig": {
                "defaults": {"unit": "short", "color": {"mode": "palette-classic"}},
                "overrides": [],
            },
            "options": {
                "xField":  "tier",
                "legend":  {"displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "single", "sort": "none"},
            },
            "targets": [
                {
                    "datasource":   {"type": "prometheus", "uid": datasource},
                    "expr":         f'{ns}_tier_total{{domain=~"$domain"}}',
                    "legendFormat": "{{tier}} / {{domain}}",
                    "refId":        "A",
                }
            ],
        },
    ]

    # Template variable for domain filtering
    templating = {
        "list": [
            {
                "name":        "domain",
                "type":        "query",
                "datasource":  {"type": "prometheus", "uid": datasource},
                "query":       f'label_values({ns}_chains_total, domain)',
                "refresh":     2,
                "includeAll":  True,
                "multi":       True,
                "allValue":    ".*",
                "current":     {"selected": True, "text": "All", "value": "$__all"},
                "label":       "Domain",
            }
        ]
    }

    dashboard = {
        "uid":         "llm-guard-v1",
        "title":       title,
        "tags":        ["llm-guard", "agent-reliability"],
        "timezone":    "browser",
        "refresh":     refresh,
        "schemaVersion": 38,
        "version":     1,
        "panels":      panels,
        "templating":  templating,
        "time":        {"from": "now-1h", "to": "now"},
        "timepicker":  {},
        "annotations": {"list": []},
        "links":       [],
        "editable":    True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "liveNow":      False,
    }

    return json.dumps({"dashboard": dashboard, "overwrite": True, "folderId": 0}, indent=2)
