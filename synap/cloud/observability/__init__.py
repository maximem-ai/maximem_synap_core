"""Observability layer for telemetry, reporting, and alerts.

This module provides:
- Telemetry ingestion (fire-and-forget)
- Queue-based buffering (Redis Streams)
- Event and decision trace storage (TimescaleDB)
- Internal and client-facing reporting
- Alert management and evaluation

Only ``bootstrap`` and ``tracing`` load with the package. Every other name
below is imported from its module the first time it is asked for (PEP 562), so
Core, which ships just those two files, imports the package without the rest.
Cloud gets the same names, the same objects, and ``__all__`` is unchanged.
"""

import importlib
from .bootstrap import (
    ObservabilitySettings,
    ObservabilityBootstrapState,
    load_phase0_observability_settings,
    bootstrap_phase0_observability,
    shutdown_phase0_observability,
)
from .tracing import (
    TracingBootstrapState,
    bootstrap_phase1_tracing,
    shutdown_phase1_tracing,
    set_current_span_attributes,
    start_span,
    current_trace_id_hex,
)

# Name -> the module (inside this package) that defines it.
_LAZY_EXPORTS = {
    "ObservabilityError": "models",
    "ValidationError": "models",
    "IngestionError": "models",
    "QueueError": "models",
    "StoreError": "models",
    "AlertError": "models",
    "EventType": "models",
    "EventSource": "models",
    "TelemetryEvent": "models",
    "DecisionTrace": "models",
    "AlertSeverity": "models",
    "AlertScope": "models",
    "AlertCondition": "models",
    "AlertDefinition": "models",
    "TriggeredAlert": "models",
    "UsageMetrics": "models",
    "DecisionMetrics": "models",
    "ClientReport": "models",
    "QueueInterface": "queue",
    "RedisStreamQueue": "queue",
    "TelemetryIngestionService": "ingestion",
    "EventValidator": "ingestion",
    "EventEnricher": "ingestion",
    "TelemetryEventStore": "store",
    "DecisionTraceStore": "store",
    "ObservabilityTelemetrySQLStore": "store",
    "ObservabilityAlertsSQLStore": "store",
    "IngestionConsumer": "consumer",
    "AlertManager": "alerts",
    "AlertStore": "alerts",
    "InternalReporter": "reporting",
    "ClientReporter": "reporting",
    "MetricsBootstrapState": "metrics",
    "observe_http_request": "metrics",
    "record_telemetry_ingestion_result": "metrics",
    "render_prometheus_metrics": "metrics",
    "set_gauge": "metrics",
    "reset_phase2_metrics_for_tests": "metrics",
    "bootstrap_phase2_metrics": "metrics",
    "shutdown_phase2_metrics": "metrics",
    "Phase5RuntimeState": "phase5",
    "bootstrap_phase5_runtime": "phase5",
    "shutdown_phase5_runtime": "phase5",
    "get_phase5_runtime_state": "phase5",
    "ingest_phase5_batch": "phase5",
    "build_phase5_event": "phase5",
}

# The submodules were attributes of the package when it imported them all; keep that.
_LAZY_SUBMODULES = frozenset((
    "models",
    "queue",
    "ingestion",
    "store",
    "consumer",
    "alerts",
    "reporting",
    "metrics",
    "phase5",
))


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f".{module}", __name__), name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | _LAZY_SUBMODULES)


__all__ = [
    # Models - Errors
    "ObservabilityError",
    "ValidationError",
    "IngestionError",
    "QueueError",
    "StoreError",
    "AlertError",
    # Models - Events
    "EventType",
    "EventSource",
    "TelemetryEvent",
    "DecisionTrace",
    # Models - Alerts
    "AlertSeverity",
    "AlertScope",
    "AlertCondition",
    "AlertDefinition",
    "TriggeredAlert",
    # Models - Reports
    "UsageMetrics",
    "DecisionMetrics",
    "ClientReport",
    # Queue
    "QueueInterface",
    "RedisStreamQueue",
    # Ingestion
    "TelemetryIngestionService",
    "EventValidator",
    "EventEnricher",
    # Store
    "TelemetryEventStore",
    "DecisionTraceStore",
    "ObservabilityTelemetrySQLStore",
    "ObservabilityAlertsSQLStore",
    # Consumer
    "IngestionConsumer",
    # Alerts
    "AlertManager",
    "AlertStore",
    # Reporting
    "InternalReporter",
    "ClientReporter",
    # Bootstrap
    "ObservabilitySettings",
    "ObservabilityBootstrapState",
    "load_phase0_observability_settings",
    "bootstrap_phase0_observability",
    "shutdown_phase0_observability",
    # Tracing
    "TracingBootstrapState",
    "bootstrap_phase1_tracing",
    "shutdown_phase1_tracing",
    "set_current_span_attributes",
    "start_span",
    "current_trace_id_hex",
    # Metrics
    "MetricsBootstrapState",
    "observe_http_request",
    "record_telemetry_ingestion_result",
    "render_prometheus_metrics",
    "set_gauge",
    "reset_phase2_metrics_for_tests",
    "bootstrap_phase2_metrics",
    "shutdown_phase2_metrics",
    # Runtime pipeline
    "Phase5RuntimeState",
    "bootstrap_phase5_runtime",
    "shutdown_phase5_runtime",
    "get_phase5_runtime_state",
    "ingest_phase5_batch",
    "build_phase5_event",
]
