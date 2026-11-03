"""Tests for BaseIngestionWorker."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    PipelineStage,
    IngestStatus,
)
from synap.cloud.agents.ingestion.workers.base import BaseIngestionWorker


class ConcreteTestWorker(BaseIngestionWorker):
    """Concrete implementation for testing."""

    @property
    def input_queue(self) -> str:
        return self.QUEUE_STAGE1

    @property
    def output_queue(self) -> str:
        return self.QUEUE_STAGE23

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.STAGE_1_QUALIFICATION

    async def process(self, payload):
        """Test implementation."""
        if payload.get("should_fail"):
            raise ValueError("Simulated failure")
        return {"processed": True, **payload}


class TestBaseIngestionWorker:
    """Tests for BaseIngestionWorker."""

    @pytest.fixture
    def worker(self, redis, mock_status_tracker, mock_reporting):
        """Create test worker."""
        return ConcreteTestWorker(
            redis=redis,
            status_tracker=mock_status_tracker,
            reporting=mock_reporting,
            worker_id="test-worker-1",
        )

    @pytest.mark.asyncio
    async def test_dequeue_success(self, worker, redis, sample_payload):
        """Test successful dequeue from input queue."""
        # Enqueue item
        await redis.lpush(worker.input_queue, json.dumps(sample_payload))

        # Dequeue
        result = await worker._dequeue(timeout=1)

        assert result is not None
        assert result["ingestion_id"] == sample_payload["ingestion_id"]

    @pytest.mark.asyncio
    async def test_dequeue_timeout(self, worker):
        """Test dequeue timeout on empty queue."""
        result = await worker._dequeue(timeout=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_enqueue_next(self, worker, redis, sample_payload):
        """Test enqueueing to next stage."""
        await worker._enqueue_next(sample_payload)

        # Check item in output queue
        result = await redis.brpop(worker.output_queue, timeout=1)
        assert result is not None
        _, payload = result
        data = json.loads(payload)
        assert data["ingestion_id"] == sample_payload["ingestion_id"]

    @pytest.mark.asyncio
    async def test_process_with_retry_success(
        self, worker, sample_payload, mock_status_tracker, mock_reporting
    ):
        """Test successful processing on first attempt."""
        result = await worker._process_with_retry(sample_payload)

        assert result is not None
        assert result["processed"] is True

        # Verify status update
        mock_status_tracker.update.assert_called()
        calls = mock_status_tracker.update.call_args_list
        # Check if PROCESSING status was set (2nd positional arg)
        assert any(
            call[0][1] == IngestStatus.PROCESSING for call in calls
        )

        # Verify trace emission
        mock_reporting.write_metric.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_retry_exponential_backoff(
        self, worker, sample_payload, mock_status_tracker
    ):
        """Test exponential backoff on retries."""
        sample_payload["should_fail"] = True

        start_time = datetime.utcnow()
        result = await worker._process_with_retry(sample_payload)
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        assert result is None  # All retries failed

        # Should have retried 3 times with delays: 1s, 2s
        # Total delay should be at least 3 seconds (1 + 2)
        assert elapsed >= 3.0

        # Verify final status update to FAILED
        final_call = mock_status_tracker.update.call_args_list[-1]
        assert final_call[0][1] == IngestStatus.FAILED

    @pytest.mark.asyncio
    async def test_send_to_dlq(self, worker, redis, sample_payload):
        """Test sending failed items to DLQ."""
        error_msg = "Test error"
        await worker._send_to_dlq(sample_payload, error_msg)

        # DLQ keys are now per-client: synap:ingestion:dlq:{stage}:{client_id}.
        # client_id is resolved from the nested request scope (matching the
        # product's lookup order).
        client_id = sample_payload["request"]["scope"]["client_id"]
        dlq_key = f"synap:ingestion:dlq:{worker.stage.value}:{client_id}"
        result = await redis.brpop(dlq_key, timeout=1)

        assert result is not None
        _, payload = result
        data = json.loads(payload)
        assert data["ingestion_id"] == sample_payload["ingestion_id"]
        assert data["error"] == error_msg
        assert data["worker_id"] == "test-worker-1"
        assert "failed_at" in data

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, worker, sample_payload):
        """Test circuit breaker reports open while the open-until window is active."""
        sample_payload["should_fail"] = True

        # Circuit breaker is now a sliding window of recent outcomes; fill the
        # window with failures to mirror the systemic-failure condition.
        for _ in range(worker.CIRCUIT_WINDOW_SIZE):
            worker._recent_outcomes.append(True)

        # Manually open circuit (timezone-aware — the product compares against
        # datetime.now(timezone.utc)).
        worker._circuit_open_until = datetime.now(timezone.utc) + timedelta(seconds=10)

        # Circuit should be open
        assert worker._is_circuit_open() is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets(self, worker):
        """Test circuit breaker resets after timeout."""
        # Open circuit with a past timeout (timezone-aware).
        worker._circuit_open_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        # Populate the sliding window so we can confirm it is cleared on reset.
        for _ in range(5):
            worker._recent_outcomes.append(True)

        # Circuit should reset
        assert worker._is_circuit_open() is False
        assert len(worker._recent_outcomes) == 0
        assert worker._circuit_open_until is None

    @pytest.mark.asyncio
    async def test_emit_trace_success(
        self, worker, sample_payload, mock_reporting, ingestion_id
    ):
        """Test trace emission for successful processing."""
        started_at = datetime.now(timezone.utc)
        result = {"processed": True}

        await worker._emit_trace(
            ingestion_id=ingestion_id,
            started_at=started_at,
            success=True,
            payload=sample_payload,
            result=result,
        )

        # Verify write_metric called with correct parameters
        mock_reporting.write_metric.assert_called_once()
        call_args = mock_reporting.write_metric.call_args

        assert call_args[1]["metric_name"] == "ingestion_trace"
        assert call_args[1]["value"] == 1  # Success = 1
        
        tags = call_args[1]["tags"]
        assert tags["ingestion_id"] == str(ingestion_id)
        assert tags["worker_id"] == "test-worker-1"
        assert tags["stage"] == worker.stage.value
        assert tags["success"] is True
        assert "input_hash" in tags
        assert "output_hash" in tags

    @pytest.mark.asyncio
    async def test_emit_trace_failure(
        self, worker, sample_payload, mock_reporting, ingestion_id
    ):
        """Test trace emission for failed processing."""
        started_at = datetime.now(timezone.utc)
        error = "Test error"

        await worker._emit_trace(
            ingestion_id=ingestion_id,
            started_at=started_at,
            success=False,
            payload=sample_payload,
            error=error,
        )

        # Verify write_metric called with correct parameters
        mock_reporting.write_metric.assert_called_once()
        call_args = mock_reporting.write_metric.call_args

        assert call_args[1]["value"] == 0  # Failure = 0
        
        tags = call_args[1]["tags"]
        assert tags["success"] is False
        assert tags["error"] == error
        assert tags["output_hash"] is None

    @pytest.mark.asyncio
    async def test_worker_run_and_stop(self, worker, redis, sample_payload):
        """Test worker run loop and graceful stop."""
        # Enqueue one item
        await redis.lpush(worker.input_queue, json.dumps(sample_payload))

        # Run worker in background
        worker_task = asyncio.create_task(worker.run())

        # Give it time to process
        await asyncio.sleep(0.5)

        # Stop worker
        await worker.stop()
        await asyncio.sleep(0.1)

        # Should have processed the item
        result = await redis.brpop(worker.output_queue, timeout=1)
        assert result is not None

        # Clean up
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_worker_skips_processing_when_circuit_open(
        self, worker, redis, sample_payload
    ):
        """Test worker skips processing when circuit breaker is open."""
        # Open circuit breaker (timezone-aware to match the product).
        worker._circuit_open_until = datetime.now(timezone.utc) + timedelta(seconds=10)

        # Enqueue item
        await redis.lpush(worker.input_queue, json.dumps(sample_payload))

        # Run worker briefly
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.5)
        await worker.stop()

        # Item should still be in queue (not processed)
        queue_len = await redis.llen(worker.input_queue)
        assert queue_len == 1

        # Clean up
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
