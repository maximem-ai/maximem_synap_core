"""Base worker class for ingestion pipeline."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from redis.asyncio import Redis

from synap.cloud.agents.ingestion.models import (
    IngestionTrace,
    PipelineStage,
    IngestStatus,
)
from synap.cloud.application.ingestion.status import StatusTracker
from synap.cloud.shared.step_logger import StepLogger
from opentelemetry import context as otel_context, propagate as otel_propagate
from synap.cloud.observability.tracing import start_span, set_current_span_attributes

if TYPE_CHECKING:  # type hint only: Core does not ship synap/providers/reporting (allowlist.yml)
    from synap.providers.reporting import ReportingStoreInterface


logger = logging.getLogger(__name__)


class BaseIngestionWorker(ABC):
    """
    Base class for ingestion pipeline workers.

    Handles:
    - Queue polling
    - Retry logic with exponential backoff
    - Circuit breaker for LLM failures
    - Trace emission to TimescaleDB
    - Status updates
    """

    # Queue names
    QUEUE_STAGE1 = "synap:ingestion:stage1"
    QUEUE_STAGE23 = "synap:ingestion:stage23"
    QUEUE_STAGE4 = "synap:ingestion:stage4"

    # Retry config.
    #
    # Read the arithmetic before changing any of it: with MAX_RETRIES = 3 the delays
    # are BASE * 2^0 and BASE * 2^1, so 1s and 2s. The total budget is THREE SECONDS,
    # and MAX_DELAY_SECONDS is unreachable — the largest delay computable here is 2.
    # It is left at 60 as a ceiling for anyone who raises MAX_RETRIES, not because
    # anything waits that long today.
    #
    # This budget is for a blip, and it should stay that way. It is not, and cannot
    # be, cover for a dependency being down: if the database is away for minutes, an
    # ingestion that has already had memories extracted from it is dead-lettered
    # seconds in.
    #
    # Raising MAX_RETRIES is the wrong fix and makes something else worse. The worker
    # loop is serial — `_process_with_retry` is awaited before the next `_dequeue` —
    # so every second of backoff here is a second this worker processes nothing else.
    # A budget long enough to ride out a restart would take the stage offline for the
    # duration, trading one lost item for a stalled pipeline.
    #
    # The way back in after an outage is `python -m synap.cloud.agents.ingestion.replay`,
    # which re-drives the dead-letter queue once the dependency is healthy again.
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 1
    MAX_DELAY_SECONDS = 60

    # Circuit breaker (sliding window: opens if >80% of last 10 items failed)
    CIRCUIT_WINDOW_SIZE = 10
    CIRCUIT_FAILURE_RATIO = 0.8
    CIRCUIT_RESET_SECONDS = 60

    def __init__(
        self,
        redis: Redis,
        status_tracker: StatusTracker,
        reporting: "ReportingStoreInterface",
        worker_id: str,
    ):
        self._redis = redis
        self._status = status_tracker
        self._reporting = reporting
        self._worker_id = worker_id

        # Circuit breaker state (per-worker-instance, sliding window).
        # Tracks last N item outcomes (True=failed, False=success).
        # Opens if >80% of recent items failed (detects systemic issues).
        self._recent_outcomes: deque = deque(maxlen=self.CIRCUIT_WINDOW_SIZE)
        self._circuit_open_until: Optional[datetime] = None

        self._running = False

    @property
    def tracing_span_name(self) -> str:
        """Parent OTel span name for this worker's process() call.

        Override in each concrete worker to set a clean span name
        (e.g. 'ingestion.stage1'). Default falls back to stage value.
        """
        return f"ingestion.{self.stage.value.lower()}"

    @property
    @abstractmethod
    def input_queue(self) -> str:
        """Queue to read from."""
        pass

    @property
    @abstractmethod
    def output_queue(self) -> Optional[str]:
        """Queue to write to (None for final stage)."""
        pass

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """Pipeline stage this worker handles."""
        pass

    @abstractmethod
    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item. Returns output for next stage."""
        pass

    async def run(self):
        """Main worker loop."""
        self._running = True
        logger.info(f"Worker {self._worker_id} starting for stage {self.stage.value}")

        while self._running:
            try:
                # Check circuit breaker
                if self._is_circuit_open():
                    await asyncio.sleep(1)
                    continue

                # Poll queue
                item = await self._dequeue()
                if not item:
                    continue

                # Process with retry
                result = await self._process_with_retry(item)

                if result and self.output_queue:
                    await self._enqueue_next(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info(f"Worker {self._worker_id} stopped")

    async def stop(self):
        """Stop the worker gracefully."""
        self._running = False

    async def _dequeue(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """Dequeue item from input queue."""
        import json
        import redis.exceptions

        try:
            result = await self._redis.brpop(self.input_queue, timeout=timeout)
        except redis.exceptions.TimeoutError:
            # A blocking-pop timeout means "no item available" — the same as an
            # empty return. redis-py 8.0.0 raises here instead of returning None
            # when the queue stays idle for the full blocking timeout, so treat
            # it as empty rather than letting run() log it as a worker error.
            return None
        if result:
            _, payload = result
            item = json.loads(payload)
            logger.info(
                "ingestion_worker_dequeued worker_id=%s stage=%s ingestion_id=%s input_queue=%s",
                self._worker_id,
                self.stage.value,
                item.get("ingestion_id"),
                self.input_queue,
            )
            return item
        return None

    async def _enqueue_next(self, payload: Dict[str, Any]):
        """Enqueue result to next stage's queue."""
        import json

        queue_len = await self._redis.lpush(self.output_queue, json.dumps(payload))
        logger.info(
            "ingestion_worker_enqueued_next worker_id=%s stage=%s ingestion_id=%s output_queue=%s output_queue_length=%s",
            self._worker_id,
            self.stage.value,
            payload.get("ingestion_id"),
            self.output_queue,
            queue_len,
        )

    async def _process_with_retry(
        self,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Process with exponential backoff retry."""
        ingestion_id = UUID(payload.get("ingestion_id"))
        last_error = None

        # Restore the trace context from the originating HTTP request so that
        # all worker spans appear as children of sdk.memory.create in the tracing backend.
        _trace_carrier = payload.get("_otel_context") or {}
        _parent_ctx = otel_propagate.extract(_trace_carrier)
        _ctx_token = otel_context.attach(_parent_ctx)

        # Calculate how long this item waited in the Redis queue before being picked up.
        _queue_wait_ms: Optional[float] = None
        _enqueued_at_str = payload.get("enqueued_at")
        if _enqueued_at_str:
            try:
                _enqueued_at = datetime.fromisoformat(_enqueued_at_str)
                if _enqueued_at.tzinfo is None:
                    _enqueued_at = _enqueued_at.replace(tzinfo=timezone.utc)
                _queue_wait_ms = round(
                    (datetime.now(timezone.utc) - _enqueued_at).total_seconds() * 1000, 2
                )
            except Exception:
                pass

        try:
            for attempt in range(self.MAX_RETRIES):
                trace_start = datetime.now(timezone.utc)
                t_attempt_start = time.monotonic()

                async with start_span(
                    self.tracing_span_name,
                    {
                        "synap.ingestion_id": str(ingestion_id),
                        "synap.stage": self.stage.value,
                        "synap.worker_id": self._worker_id,
                        "synap.attempt": attempt + 1,
                        "synap.max_retries": self.MAX_RETRIES,
                        "synap.queue_wait_ms": _queue_wait_ms if _queue_wait_ms is not None else -1,
                    },
                ):
                    try:
                        trace_fields = self._trace_fields(payload)
                        set_current_span_attributes({
                            "synap.request_id": trace_fields.get("request_id") or "",
                            "synap.correlation_id": trace_fields.get("correlation_id") or "",
                        })
                        logger.info(
                            "ingestion_worker_attempt_start worker_id=%s stage=%s attempt=%s ingestion_id=%s request_id=%s correlation_id=%s",
                            self._worker_id,
                            self.stage.value,
                            attempt + 1,
                            trace_fields["ingestion_id"],
                            trace_fields["request_id"],
                            trace_fields["correlation_id"],
                        )

                        # Step logging for request (on first attempt only)
                        if attempt == 0:
                            step_logger = StepLogger(f"IngestionWorker.{self.stage.value}")
                            step_logger.log_request(
                                payload=payload,
                                correlation_id=trace_fields["correlation_id"],
                            )

                        # Update status to processing
                        # Set started_at on first attempt only
                        status_updates = {"current_stage": self.stage.value}
                        if attempt == 0:
                            status_updates["started_at"] = datetime.now(timezone.utc)

                        await self._status.update(
                            ingestion_id,
                            IngestStatus.PROCESSING,
                            **status_updates,
                        )

                        # Process
                        result = await self.process(payload)

                        # Record success in sliding window
                        self._recent_outcomes.append(False)

                        attempt_ms = round((time.monotonic() - t_attempt_start) * 1000, 2)
                        set_current_span_attributes({
                            "synap.success": True,
                            "synap.attempt_duration_ms": attempt_ms,
                            "synap.has_output": result is not None,
                        })

                        # Emit success trace
                        await self._emit_trace(
                            ingestion_id=ingestion_id,
                            started_at=trace_start,
                            success=True,
                            payload=payload,
                            result=result,
                        )
                        logger.info(
                            "ingestion_worker_attempt_success worker_id=%s stage=%s attempt=%s ingestion_id=%s",
                            self._worker_id,
                            self.stage.value,
                            attempt + 1,
                            ingestion_id,
                        )

                        # Step logging for response (on first successful attempt)
                        if attempt == 0:
                            step_logger.log_response(
                                response=result,
                                status_code=200,
                                success=True,
                                correlation_id=trace_fields["correlation_id"],
                            )

                        # Propagate current span context into the next-stage payload
                        # so all stage spans appear as a connected waterfall in Tempo.
                        if result is not None:
                            _next_carrier: Dict[str, str] = {}
                            otel_propagate.inject(_next_carrier)
                            if _next_carrier:
                                result["_otel_context"] = _next_carrier

                        return result

                    except Exception as e:
                        last_error = e
                        trace_fields = self._trace_fields(payload)
                        attempt_ms = round((time.monotonic() - t_attempt_start) * 1000, 2)
                        set_current_span_attributes({
                            "synap.success": False,
                            "synap.error": True,
                            "synap.error_type": type(e).__name__,
                            "synap.attempt_duration_ms": attempt_ms,
                        })
                        logger.warning(
                            "ingestion_worker_attempt_failed worker_id=%s stage=%s attempt=%s ingestion_id=%s request_id=%s correlation_id=%s error=%s",
                            self._worker_id,
                            self.stage.value,
                            attempt + 1,
                            trace_fields["ingestion_id"],
                            trace_fields["request_id"],
                            trace_fields["correlation_id"],
                            e,
                        )

                        # Emit failure trace
                        await self._emit_trace(
                            ingestion_id=ingestion_id,
                            started_at=trace_start,
                            success=False,
                            payload=payload,
                            error=str(e),
                        )

                        # Calculate backoff
                        if attempt < self.MAX_RETRIES - 1:
                            delay = min(
                                self.BASE_DELAY_SECONDS * (2 ** attempt),
                                self.MAX_DELAY_SECONDS,
                            )
                            logger.info(
                                "ingestion_worker_retry_scheduled worker_id=%s stage=%s next_attempt=%s delay_seconds=%s ingestion_id=%s",
                                self._worker_id,
                                self.stage.value,
                                attempt + 2,
                                delay,
                                ingestion_id,
                            )
                            await asyncio.sleep(delay)

            # All retries failed - record failure in sliding window and check circuit breaker
            self._recent_outcomes.append(True)
            if (len(self._recent_outcomes) >= self.CIRCUIT_WINDOW_SIZE and
                    sum(self._recent_outcomes) / len(self._recent_outcomes) >= self.CIRCUIT_FAILURE_RATIO):
                self._circuit_open_until = datetime.now(timezone.utc) + \
                    timedelta(seconds=self.CIRCUIT_RESET_SECONDS)
                logger.warning(
                    "Circuit breaker opened for %s (%d/%d recent items failed)",
                    self.stage.value,
                    sum(self._recent_outcomes),
                    len(self._recent_outcomes),
                )

            # Send to DLQ
            await self._send_to_dlq(payload, str(last_error))
            await self._status.update(
                ingestion_id,
                IngestStatus.FAILED,
                error_message=str(last_error),
                error_stage=self.stage.value,
            )
            logger.error(
                "ingestion_worker_terminal_failed worker_id=%s stage=%s ingestion_id=%s error=%s",
                self._worker_id,
                self.stage.value,
                ingestion_id,
                last_error,
            )

            return None
        finally:
            # Always detach the propagated context — prevents leaking into
            # subsequent tasks running on the same event loop worker.
            otel_context.detach(_ctx_token)

    async def _emit_trace(
        self,
        ingestion_id: UUID,
        started_at: datetime,
        success: bool,
        payload: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Emit trace to TimescaleDB."""
        from uuid import uuid4
        import hashlib

        trace = {
            "trace_id": str(uuid4()),
            "ingestion_id": str(ingestion_id),
            "worker_id": self._worker_id,
            "stage": self.stage.value,
            "input_hash": hashlib.sha256(
                str(payload).encode()
            ).hexdigest()[:16],
            "output_hash": hashlib.sha256(
                str(result).encode()
            ).hexdigest()[:16] if result else None,
            "success": success,
            "error": error,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": int(
                (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
            ),
        }

        await asyncio.to_thread(
            self._reporting.write_metric,
            metric_name="ingestion_trace",
            value=1 if success else 0,
            tags=trace,
        )

    async def _send_to_dlq(self, payload: Dict[str, Any], error: str):
        """Send failed item to per-client dead letter queue."""
        import json

        client_id = (
            payload.get("client_id")
            or payload.get("request", {}).get("scope", {}).get("client_id", "_unknown")
        )
        dlq_key = f"synap:ingestion:dlq:{self.stage.value}:{client_id}"
        dlq_payload = {
            **payload,
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": self._worker_id,
        }
        dlq_len = await self._redis.lpush(dlq_key, json.dumps(dlq_payload))
        logger.error(
            "ingestion_worker_sent_dlq worker_id=%s stage=%s ingestion_id=%s dlq=%s dlq_length=%s error=%s",
            self._worker_id,
            self.stage.value,
            payload.get("ingestion_id"),
            dlq_key,
            dlq_len,
            error,
        )

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_open_until is None:
            return False
        if datetime.now(timezone.utc) > self._circuit_open_until:
            self._circuit_open_until = None
            self._recent_outcomes.clear()
            logger.info(f"Circuit breaker reset for {self.stage.value}")
            return False
        return True

    def _trace_fields(self, payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Best-effort extraction for correlation-friendly logs."""
        request = payload.get("request", {}) if isinstance(payload, dict) else {}
        metadata = request.get("metadata", {}) if isinstance(request, dict) else {}
        trace = metadata.get("_trace", {}) if isinstance(metadata, dict) else {}
        return {
            "ingestion_id": payload.get("ingestion_id") if isinstance(payload, dict) else None,
            "request_id": request.get("request_id") if isinstance(request, dict) else None,
            "correlation_id": trace.get("correlation_id") if isinstance(trace, dict) else None,
        }
