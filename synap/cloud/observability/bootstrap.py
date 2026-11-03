"""Observability bootstrap helpers.

This module intentionally keeps startup behavior non-blocking and resilient:
- disabled by default behind env flags
- never raises to callers during startup/shutdown
- optional connectivity probing to OTLP endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_SUPPORTED_PROTOCOLS = {"grpc", "http/protobuf"}


@dataclass(frozen=True)
class ObservabilitySettings:
    """Environment-derived settings for observability bootstrap."""

    enabled: bool
    service_name: str
    environment: str
    otlp_endpoint: str
    otlp_protocol: str
    connectivity_check: bool
    connect_timeout_seconds: float


@dataclass(frozen=True)
class ObservabilityBootstrapState:
    """Runtime state returned by bootstrap."""

    enabled: bool
    service_name: str
    environment: str
    otlp_endpoint: str
    otlp_protocol: str
    connectivity_check_enabled: bool
    connectivity_ok: Optional[bool]
    host: Optional[str]
    port: Optional[int]
    reason: str


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_phase0_observability_settings() -> ObservabilitySettings:
    """Load observability settings from environment."""

    protocol = os.getenv("SYNAP_OBSERVABILITY_OTLP_PROTOCOL", "grpc").strip().lower()
    if protocol not in _SUPPORTED_PROTOCOLS:
        logger.warning(
            "Unsupported SYNAP_OBSERVABILITY_OTLP_PROTOCOL=%s, falling back to grpc",
            protocol,
        )
        protocol = "grpc"

    return ObservabilitySettings(
        enabled=_env_bool("SYNAP_OBSERVABILITY_ENABLED", False),
        service_name=os.getenv("SYNAP_OBSERVABILITY_SERVICE_NAME", "synap-cloud").strip() or "synap-cloud",
        environment=os.getenv("SYNAP_OBSERVABILITY_ENV", os.getenv("ENVIRONMENT", "development")).strip()
        or "development",
        otlp_endpoint=os.getenv("SYNAP_OBSERVABILITY_OTLP_ENDPOINT", "http://localhost:4319").strip(),
        otlp_protocol=protocol,
        connectivity_check=_env_bool("SYNAP_OBSERVABILITY_CONNECTIVITY_CHECK", True),
        connect_timeout_seconds=max(
            float(os.getenv("SYNAP_OBSERVABILITY_CONNECT_TIMEOUT_SECONDS", "1.5") or 1.5),
            0.1,
        ),
    )


def _parse_host_port(endpoint: str, protocol: str) -> Optional[Tuple[str, int]]:
    """Resolve host/port from OTLP endpoint for lightweight TCP probing."""

    value = endpoint.strip()
    if not value:
        return None

    if "://" in value:
        parsed = urlparse(value)
        if not parsed.hostname:
            return None
        host = parsed.hostname
        port = parsed.port
        if port is None:
            if parsed.scheme in {"https", "wss"}:
                port = 443
            elif parsed.scheme in {"http", "ws"}:
                port = 80
            elif parsed.scheme == "grpc":
                port = 4317
            else:
                port = 4317 if protocol == "grpc" else 4318
    else:
        if ":" in value:
            host, raw_port = value.rsplit(":", 1)
            host = host.strip()
            if not host:
                return None
            try:
                port = int(raw_port)
            except ValueError:
                return None
        else:
            host = value
            port = 4317 if protocol == "grpc" else 4318

    if port < 1 or port > 65535:
        return None
    return host, port


async def _check_tcp_connectivity(
    host: str,
    port: int,
    timeout_seconds: float,
) -> tuple[bool, int, Optional[str]]:
    """Perform a non-blocking TCP probe against the configured OTLP endpoint."""

    start = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_seconds,
        )
        # Reader is intentionally unused for probe-only checks.
        _ = reader
        writer.close()
        await writer.wait_closed()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return True, elapsed_ms, None
    except Exception as exc:  # pragma: no cover - exercised via monkeypatched tests
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return False, elapsed_ms, str(exc)


async def bootstrap_phase0_observability() -> ObservabilityBootstrapState:
    """Initialize the observability runtime contract.

    This phase only establishes environment contract + endpoint reachability.
    It does not initialize OTel SDK exporters yet.
    """

    settings = load_phase0_observability_settings()
    if not settings.enabled:
        state = ObservabilityBootstrapState(
            enabled=False,
            service_name=settings.service_name,
            environment=settings.environment,
            otlp_endpoint=settings.otlp_endpoint,
            otlp_protocol=settings.otlp_protocol,
            connectivity_check_enabled=False,
            connectivity_ok=None,
            host=None,
            port=None,
            reason="feature_flag_disabled",
        )
        logger.info(
            "cloud_observability_phase0_disabled service=%s env=%s reason=%s",
            state.service_name,
            state.environment,
            state.reason,
        )
        return state

    endpoint_target = _parse_host_port(settings.otlp_endpoint, settings.otlp_protocol)
    if endpoint_target is None:
        state = ObservabilityBootstrapState(
            enabled=False,
            service_name=settings.service_name,
            environment=settings.environment,
            otlp_endpoint=settings.otlp_endpoint,
            otlp_protocol=settings.otlp_protocol,
            connectivity_check_enabled=settings.connectivity_check,
            connectivity_ok=False,
            host=None,
            port=None,
            reason="invalid_otlp_endpoint",
        )
        logger.warning(
            "cloud_observability_phase0_invalid_endpoint endpoint=%s protocol=%s",
            settings.otlp_endpoint,
            settings.otlp_protocol,
        )
        return state

    host, port = endpoint_target
    logger.info(
        "cloud_observability_phase0_start service=%s env=%s endpoint=%s protocol=%s connectivity_check=%s",
        settings.service_name,
        settings.environment,
        settings.otlp_endpoint,
        settings.otlp_protocol,
        settings.connectivity_check,
    )

    if not settings.connectivity_check:
        state = ObservabilityBootstrapState(
            enabled=True,
            service_name=settings.service_name,
            environment=settings.environment,
            otlp_endpoint=settings.otlp_endpoint,
            otlp_protocol=settings.otlp_protocol,
            connectivity_check_enabled=False,
            connectivity_ok=None,
            host=host,
            port=port,
            reason="connectivity_check_disabled",
        )
        logger.info(
            "cloud_observability_phase0_ready service=%s endpoint=%s reason=%s",
            state.service_name,
            state.otlp_endpoint,
            state.reason,
        )
        return state

    ok, elapsed_ms, error = await _check_tcp_connectivity(
        host=host,
        port=port,
        timeout_seconds=settings.connect_timeout_seconds,
    )
    if ok:
        logger.info(
            "cloud_observability_phase0_connectivity_ok host=%s port=%d latency_ms=%d",
            host,
            port,
            elapsed_ms,
        )
        return ObservabilityBootstrapState(
            enabled=True,
            service_name=settings.service_name,
            environment=settings.environment,
            otlp_endpoint=settings.otlp_endpoint,
            otlp_protocol=settings.otlp_protocol,
            connectivity_check_enabled=True,
            connectivity_ok=True,
            host=host,
            port=port,
            reason="connectivity_check_ok",
        )

    logger.warning(
        "cloud_observability_phase0_connectivity_failed host=%s port=%d latency_ms=%d error=%s",
        host,
        port,
        elapsed_ms,
        error,
    )
    return ObservabilityBootstrapState(
        enabled=True,
        service_name=settings.service_name,
        environment=settings.environment,
        otlp_endpoint=settings.otlp_endpoint,
        otlp_protocol=settings.otlp_protocol,
        connectivity_check_enabled=True,
        connectivity_ok=False,
        host=host,
        port=port,
        reason="connectivity_check_failed",
    )


async def shutdown_phase0_observability(state: Optional[ObservabilityBootstrapState]) -> None:
    """Shutdown hook for observability bootstrap."""

    if state is None:
        return
    logger.info(
        "cloud_observability_phase0_shutdown enabled=%s reason=%s endpoint=%s",
        state.enabled,
        state.reason,
        state.otlp_endpoint,
    )

