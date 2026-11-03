"""Tracing bootstrap and helpers.

This module is dependency-aware:
- If OpenTelemetry exporter/instrumentation packages are missing, tracing is
  skipped without breaking request handling.
- When dependencies are present, it configures OTLP trace export and FastAPI
  auto-instrumentation.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover - fallback for minimal environments
    class _NoOpSpanContext:
        is_valid = False
        trace_id = 0

    class _NoOpSpan:
        def is_recording(self) -> bool:
            return False

        def set_attribute(self, key: str, value: Any) -> None:
            _ = (key, value)

        def get_span_context(self) -> _NoOpSpanContext:
            return _NoOpSpanContext()

    class _NoOpSpanContextManager:
        def __enter__(self) -> _NoOpSpan:
            return _NoOpSpan()

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            return False

    class _NoOpTracer:
        def start_as_current_span(self, name: str) -> _NoOpSpanContextManager:
            _ = name
            return _NoOpSpanContextManager()

    class _NoOpTraceModule:
        def get_current_span(self) -> _NoOpSpan:
            return _NoOpSpan()

        def get_tracer(self, name: str) -> _NoOpTracer:
            _ = name
            return _NoOpTracer()

        def get_tracer_provider(self) -> Any:
            return None

        def set_tracer_provider(self, provider: Any) -> None:
            _ = provider

    trace = _NoOpTraceModule()

from .bootstrap import ObservabilityBootstrapState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TracingBootstrapState:
    """Runtime state for tracing bootstrap."""

    enabled: bool
    active: bool
    endpoint: Optional[str]
    sample_ratio: float
    reason: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_traces_endpoint(base_endpoint: str, protocol: str) -> Optional[str]:
    """Normalize collector endpoint to OTLP traces endpoint."""

    raw = (base_endpoint or "").strip()
    if not raw:
        return None

    # For OTLP HTTP exporter, endpoint must target the traces resource path.
    if protocol == "http/protobuf":
        if "://" in raw:
            parsed = urlparse(raw)
            if not parsed.scheme or not parsed.netloc:
                return None
            if parsed.path.endswith("/v1/traces"):
                return raw
            # Preserve base path if provided.
            base_path = parsed.path.rstrip("/")
            suffix = "/v1/traces"
            path = f"{base_path}{suffix}" if base_path else suffix
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        if raw.endswith("/v1/traces"):
            return raw
        return f"{raw.rstrip('/')}/v1/traces"

    # For OTLP gRPC exporter, keep authority endpoint and strip any path if present.
    if protocol == "grpc":
        if "://" in raw:
            parsed = urlparse(raw)
            if not parsed.scheme or not parsed.netloc:
                return None
            return f"{parsed.scheme}://{parsed.netloc}"
        return raw

    return None


def _build_span_exporter(*, protocol: str, endpoint: str, timeout_seconds: float) -> Any:
    """Build OTLP span exporter for requested protocol."""

    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(
            endpoint=endpoint,
            timeout=timeout_seconds,
        )

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        insecure_default = True
        if "://" in endpoint:
            scheme = urlparse(endpoint).scheme.strip().lower()
            if scheme in {"https", "grpcs"}:
                insecure_default = False
            elif scheme in {"http", "grpc"}:
                insecure_default = True
        insecure = _env_bool("SYNAP_OBSERVABILITY_OTLP_INSECURE", insecure_default)
        return OTLPSpanExporter(
            endpoint=endpoint,
            insecure=insecure,
            timeout=timeout_seconds,
        )

    raise ValueError(f"Unsupported tracing protocol: {protocol}")


def _parse_sample_ratio() -> float:
    raw = os.getenv("SYNAP_OBSERVABILITY_TRACE_SAMPLE_RATIO", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid SYNAP_OBSERVABILITY_TRACE_SAMPLE_RATIO=%s, falling back to 1.0",
            raw,
        )
        value = 1.0
    return min(max(value, 0.0), 1.0)


def _set_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    if span is None or not hasattr(span, "is_recording") or not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        span.set_attribute(str(key), value)


def _trace_export_logs_enabled() -> bool:
    """Enable per-span export logs for OTLP deliveries."""

    return _env_bool("SYNAP_OBSERVABILITY_TRACE_EXPORT_LOGS_ENABLED", False)


def _trace_export_logs_include_attributes() -> bool:
    """Enable attribute dump in per-span export logs."""

    return _env_bool("SYNAP_OBSERVABILITY_TRACE_EXPORT_LOGS_INCLUDE_ATTRIBUTES", False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _span_duration_ms(span: Any) -> Optional[float]:
    """Return span duration in milliseconds when start/end are available."""

    start_ns = getattr(span, "start_time", None)
    end_ns = getattr(span, "end_time", None)
    if start_ns is None or end_ns is None:
        return None
    try:
        delta_ns = int(end_ns) - int(start_ns)
    except Exception:
        return None
    if delta_ns < 0:
        return None
    return round(delta_ns / 1_000_000.0, 3)


def _readable_span_context_ids(span: Any) -> tuple[str, str, Optional[str]]:
    """Extract trace/span/parent ids from an OTel readable span."""

    trace_id = None
    span_id = None
    parent_span_id = None

    try:
        context = getattr(span, "context", None)
        trace_id_raw = getattr(context, "trace_id", 0) if context else 0
        span_id_raw = getattr(context, "span_id", 0) if context else 0
        if trace_id_raw:
            trace_id = f"{_safe_int(trace_id_raw):032x}"
        if span_id_raw:
            span_id = f"{_safe_int(span_id_raw):016x}"
    except Exception:
        trace_id = None
        span_id = None

    try:
        parent = getattr(span, "parent", None)
        parent_span_id_raw = getattr(parent, "span_id", 0) if parent else 0
        if parent_span_id_raw:
            parent_span_id = f"{_safe_int(parent_span_id_raw):016x}"
    except Exception:
        parent_span_id = None

    return (
        trace_id or "unknown",
        span_id or "unknown",
        parent_span_id,
    )


class _LoggingSpanExporter:
    """Decorator exporter that emits structured logs for every exported span."""

    def __init__(
        self,
        delegate: Any,
        *,
        service_name: str,
        endpoint: str,
        enable_logs: bool,
        include_attributes: bool,
    ) -> None:
        self._delegate = delegate
        self._service_name = service_name
        self._endpoint = endpoint
        self._enable_logs = enable_logs
        self._include_attributes = include_attributes

    def export(self, spans: Any) -> Any:
        if self._enable_logs:
            for span in spans:
                trace_id, span_id, parent_span_id = _readable_span_context_ids(span)
                kind = str(getattr(span, "kind", "unknown"))
                status = str(getattr(getattr(span, "status", None), "status_code", "unknown"))
                duration_ms = _span_duration_ms(span)
                attributes = dict(getattr(span, "attributes", {}) or {})
                attr_count = len(attributes)
                if self._include_attributes:
                    logger.info(
                        "cloud_observability_trace_export span_name=%s trace_id=%s span_id=%s parent_span_id=%s kind=%s status=%s duration_ms=%s service_name=%s destination=%s attribute_count=%s attributes=%s",
                        getattr(span, "name", "unknown"),
                        trace_id,
                        span_id,
                        parent_span_id,
                        kind,
                        status,
                        duration_ms,
                        self._service_name,
                        self._endpoint,
                        attr_count,
                        attributes,
                    )
                else:
                    logger.info(
                        "cloud_observability_trace_export span_name=%s trace_id=%s span_id=%s parent_span_id=%s kind=%s status=%s duration_ms=%s service_name=%s destination=%s attribute_count=%s",
                        getattr(span, "name", "unknown"),
                        trace_id,
                        span_id,
                        parent_span_id,
                        kind,
                        status,
                        duration_ms,
                        self._service_name,
                        self._endpoint,
                        attr_count,
                    )
            logger.info(
                "cloud_observability_trace_export_batch service_name=%s destination=%s span_count=%s",
                self._service_name,
                self._endpoint,
                len(spans),
            )

        return self._delegate.export(spans)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        delegate_force_flush = getattr(self._delegate, "force_flush", None)
        if callable(delegate_force_flush):
            return bool(delegate_force_flush(timeout_millis))
        return True

    def shutdown(self) -> Any:
        return self._delegate.shutdown()


def set_current_span_attributes(attributes: Dict[str, Any]) -> None:
    """Attach attributes to current span if one is active."""

    try:
        span = trace.get_current_span()
    except Exception:
        return
    _set_span_attributes(span, attributes)


@asynccontextmanager
async def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start an async context span from the shared tracer provider."""

    tracer = trace.get_tracer("synap.cloud.application")
    with tracer.start_as_current_span(name) as span:
        if attributes:
            _set_span_attributes(span, attributes)
        yield span


def current_trace_id_hex() -> Optional[str]:
    """Return current trace ID as 32-char hex string, if any."""

    try:
        ctx = trace.get_current_span().get_span_context()
    except Exception:
        return None
    if not ctx or not getattr(ctx, "is_valid", False):
        return None
    trace_id = getattr(ctx, "trace_id", 0)
    if not trace_id:
        return None
    return f"{trace_id:032x}"


async def bootstrap_phase1_tracing(
    app: Any,
    phase0_state: Optional[ObservabilityBootstrapState],
) -> TracingBootstrapState:
    """Initialize OTLP trace export and FastAPI tracing instrumentation."""

    enabled = _env_bool("SYNAP_OBSERVABILITY_TRACING_ENABLED", True)
    if not enabled:
        state = TracingBootstrapState(
            enabled=False,
            active=False,
            endpoint=None,
            sample_ratio=0.0,
            reason="feature_flag_disabled",
        )
        logger.info("cloud_observability_phase1_tracing_disabled reason=%s", state.reason)
        return state

    if phase0_state is None or not phase0_state.enabled:
        state = TracingBootstrapState(
            enabled=True,
            active=False,
            endpoint=None,
            sample_ratio=0.0,
            reason="phase0_not_enabled",
        )
        logger.info("cloud_observability_phase1_tracing_skipped reason=%s", state.reason)
        return state

    protocol = phase0_state.otlp_protocol
    traces_endpoint = os.getenv("SYNAP_OBSERVABILITY_TRACES_ENDPOINT", "").strip() or phase0_state.otlp_endpoint
    normalized_endpoint = _normalize_traces_endpoint(traces_endpoint, protocol)
    sample_ratio = _parse_sample_ratio()

    if protocol not in {"http/protobuf", "grpc"}:
        state = TracingBootstrapState(
            enabled=True,
            active=False,
            endpoint=normalized_endpoint,
            sample_ratio=sample_ratio,
            reason="protocol_not_supported_in_phase1",
        )
        logger.warning(
            "cloud_observability_phase1_tracing_skipped reason=%s protocol=%s",
            state.reason,
            protocol,
        )
        return state

    if not normalized_endpoint:
        state = TracingBootstrapState(
            enabled=True,
            active=False,
            endpoint=None,
            sample_ratio=sample_ratio,
            reason="invalid_traces_endpoint",
        )
        logger.warning(
            "cloud_observability_phase1_tracing_skipped reason=%s endpoint=%s",
            state.reason,
            traces_endpoint,
        )
        return state

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except Exception as exc:
        state = TracingBootstrapState(
            enabled=True,
            active=False,
            endpoint=normalized_endpoint,
            sample_ratio=sample_ratio,
            reason="missing_opentelemetry_dependencies",
        )
        logger.warning(
            "cloud_observability_phase1_tracing_skipped reason=%s error=%s",
            state.reason,
            exc,
        )
        return state

    try:
        resource = Resource.create(
            {
                "service.name": phase0_state.service_name,
                "deployment.environment": phase0_state.environment,
                "synap.observability.phase": "phase1",
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
        )
        exporter = _build_span_exporter(
            protocol=protocol,
            endpoint=normalized_endpoint,
            timeout_seconds=float(
                os.getenv("SYNAP_OBSERVABILITY_TRACES_EXPORT_TIMEOUT_SECONDS", "5.0") or 5.0
            ),
        )
        exporter = _LoggingSpanExporter(
            exporter,
            service_name=phase0_state.service_name,
            endpoint=normalized_endpoint,
            enable_logs=_trace_export_logs_enabled(),
            include_attributes=_trace_export_logs_include_attributes(),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        excluded_urls = os.getenv("SYNAP_OBSERVABILITY_FASTAPI_EXCLUDED_URLS", "").strip() or None
        FastAPIInstrumentor().instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls=excluded_urls,
        )
    except Exception as exc:
        state = TracingBootstrapState(
            enabled=True,
            active=False,
            endpoint=normalized_endpoint,
            sample_ratio=sample_ratio,
            reason="bootstrap_failed",
        )
        logger.warning(
            "cloud_observability_phase1_tracing_bootstrap_failed endpoint=%s error=%s",
            normalized_endpoint,
            exc,
            exc_info=True,
        )
        return state

    state = TracingBootstrapState(
        enabled=True,
        active=True,
        endpoint=normalized_endpoint,
        sample_ratio=sample_ratio,
        reason="ready",
    )
    logger.info(
        "cloud_observability_phase1_tracing_ready endpoint=%s sample_ratio=%s trace_export_logs_enabled=%s trace_export_logs_include_attributes=%s",
        normalized_endpoint,
        sample_ratio,
        _trace_export_logs_enabled(),
        _trace_export_logs_include_attributes(),
    )
    return state


async def shutdown_phase1_tracing(app: Any, state: Optional[TracingBootstrapState]) -> None:
    """Shutdown tracing exporter and uninstrument FastAPI app."""

    if state is None:
        return

    if state.active:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor().uninstrument_app(app)
        except Exception:
            pass

        try:
            provider = trace.get_tracer_provider()
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass

    logger.info(
        "cloud_observability_phase1_tracing_shutdown active=%s reason=%s endpoint=%s",
        state.active,
        state.reason,
        state.endpoint,
    )
