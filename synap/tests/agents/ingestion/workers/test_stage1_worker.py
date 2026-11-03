"""Tests for Stage1Worker."""

import json
import pytest
from unittest.mock import AsyncMock

from synap.cloud.agents.ingestion.models import (
    PipelineStage,
    Stage1Output,
    ContentNature,
    DocumentType,
    DocumentSource,
)
from synap.cloud.agents.ingestion.workers.stage1_worker import Stage1Worker


class TestStage1Worker:
    """Tests for Stage1Worker."""

    @pytest.fixture
    def worker(
        self,
        redis,
        mock_status_tracker,
        mock_reporting,
        mock_qualifier,
        mock_config_loader,
    ):
        """Create Stage1Worker."""
        return Stage1Worker(
            redis=redis,
            status_tracker=mock_status_tracker,
            reporting=mock_reporting,
            worker_id="stage1-worker-1",
            qualifier=mock_qualifier,
            config_loader=mock_config_loader,
        )

    @pytest.mark.asyncio
    async def test_worker_properties(self, worker):
        """Test worker property values."""
        assert worker.input_queue == "synap:ingestion:stage1"
        assert worker.output_queue == "synap:ingestion:stage23"
        assert worker.stage == PipelineStage.STAGE_1_QUALIFICATION

    @pytest.mark.asyncio
    async def test_process_should_process_true(
        self,
        worker,
        sample_payload,
        mock_qualifier,
        mock_config_loader,
        ingestion_id,
    ):
        """Test processing when should_process is True."""
        # Mock qualifier output
        stage1_output = Stage1Output(
            input_id=str(ingestion_id),
            document_type=DocumentType.DOCUMENT,
            source=DocumentSource.API,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
        )
        mock_qualifier.qualify = AsyncMock(return_value=stage1_output)

        result = await worker.process(sample_payload)

        # Should return payload for next stage
        assert result is not None
        assert result["ingestion_id"] == sample_payload["ingestion_id"]
        assert "stage1_output" in result
        assert result["stage1_output"] == stage1_output.model_dump(mode="json")
        assert "maca_config_version" in result

        # Verify qualifier was called
        mock_qualifier.qualify.assert_called_once()

        # Verify config loader was called
        mock_config_loader.get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_should_process_false(
        self,
        worker,
        sample_payload,
        mock_qualifier,
        ingestion_id,
    ):
        """Test processing when should_process is False (skip)."""
        # Mock qualifier output with should_process=False
        stage1_output = Stage1Output(
            input_id=str(ingestion_id),
            document_type=DocumentType.DOCUMENT,
            source=DocumentSource.API,
            nature=ContentNature.TRIVIAL,
            should_process=False,
        )
        mock_qualifier.qualify = AsyncMock(return_value=stage1_output)

        worker._status.update = AsyncMock()

        result = await worker.process(sample_payload)

        # Should return None (skip further processing)
        assert result is None
        # and the status says why nothing was stored, not only "completed, 0 memories"
        kwargs = worker._status.update.await_args.kwargs
        assert kwargs["memories_created"] == 0
        assert "shorter than 100 characters" in kwargs["error_message"]

    @pytest.mark.asyncio
    async def test_end_to_end_processing(
        self,
        worker,
        redis,
        sample_payload,
        mock_qualifier,
        ingestion_id,
    ):
        """Test end-to-end processing through worker."""
        # Setup mock
        stage1_output = Stage1Output(
            input_id=str(ingestion_id),
            document_type=DocumentType.DOCUMENT,
            source=DocumentSource.API,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
        )
        mock_qualifier.qualify = AsyncMock(return_value=stage1_output)

        # Enqueue item
        await redis.lpush(worker.input_queue, json.dumps(sample_payload))

        # Process
        item = await worker._dequeue(timeout=1)
        result = await worker.process(item)

        # Enqueue to next stage
        if result:
            await worker._enqueue_next(result)

        # Verify item in stage23 queue
        next_item = await redis.brpop(worker.output_queue, timeout=1)
        assert next_item is not None
        _, payload = next_item
        data = json.loads(payload)
        assert data["ingestion_id"] == sample_payload["ingestion_id"]
        assert "stage1_output" in data
