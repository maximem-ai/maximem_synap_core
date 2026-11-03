"""Tests for Stage4Worker."""

import json
import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from synap.cloud.agents.ingestion.models import (
    PipelineStage,
    Stage3Output,
    Stage4Output,
    IngestStatus,
)
from synap.cloud.agents.ingestion.workers.stage4_worker import Stage4Worker


class TestStage4Worker:
    """Tests for Stage4Worker."""

    @pytest.fixture
    def worker(
        self,
        redis,
        mock_status_tracker,
        mock_reporting,
        mock_organizer,
    ):
        """Create Stage4Worker."""
        return Stage4Worker(
            redis=redis,
            status_tracker=mock_status_tracker,
            reporting=mock_reporting,
            worker_id="stage4-worker-1",
            organizer=mock_organizer,
        )

    @pytest.fixture
    def stage3_payload(self, sample_payload, ingestion_id):
        """Create payload with Stage 3 output."""
        stage3_output = Stage3Output(
            input_id=str(ingestion_id),
            memories=[],
            entities=[],
            relationships=[],
        )
        return {
            **sample_payload,
            "stage1_output": {},
            "stage2_output": {},
            "stage3_output": stage3_output.model_dump(mode="json"),
            "maca_config_version": "v1.0.0",
        }

    @pytest.mark.asyncio
    async def test_worker_properties(self, worker):
        """Test worker property values."""
        assert worker.input_queue == "synap:ingestion:stage4"
        assert worker.output_queue is None  # Final stage
        assert worker.stage == PipelineStage.STAGE_4_ORGANIZATION

    @pytest.mark.asyncio
    async def test_process_organizes_and_persists(
        self,
        worker,
        stage3_payload,
        mock_organizer,
        mock_status_tracker,
        ingestion_id,
    ):
        """Test that Stage 4 organizes and persists data."""
        # Mock Stage 4 output
        stage4_output = Stage4Output(
            input_id=str(ingestion_id),
            rdbms_memory_ids=["mem1", "mem2"],
            rdbms_candidate_ids=["cand1"],
            graph_entity_ids=["ent1", "ent2"],
            graph_relationship_ids=["rel1"],
            vector_ids=["vec1", "vec2"],
        )
        mock_organizer.organize_and_persist = AsyncMock(
            return_value=stage4_output
        )

        result = await worker.process(stage3_payload)

        # Verify organizer was called
        mock_organizer.organize_and_persist.assert_called_once()

        # Verify status updated to COMPLETED
        status_calls = mock_status_tracker.update.call_args_list
        final_call = status_calls[-1]
        assert final_call[0][1] == IngestStatus.COMPLETED  # 2nd positional arg is status
        assert final_call[1]["memories_created"] == 2  # kwargs
        assert final_call[1]["candidates_created"] == 1
        assert final_call[1]["entities_created"] == 2
        assert "completed_at" in final_call[1]

        # Final stage returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_process_with_empty_output(
        self,
        worker,
        stage3_payload,
        mock_organizer,
        mock_status_tracker,
        ingestion_id,
    ):
        """Test processing with no memories/entities extracted."""
        # Mock empty Stage 4 output
        stage4_output = Stage4Output(
            input_id=str(ingestion_id),
            rdbms_memory_ids=[],
            rdbms_candidate_ids=[],
            graph_entity_ids=[],
            graph_relationship_ids=[],
            vector_ids=[],
        )
        mock_organizer.organize_and_persist = AsyncMock(
            return_value=stage4_output
        )

        result = await worker.process(stage3_payload)

        # Still completes successfully
        status_calls = mock_status_tracker.update.call_args_list
        final_call = status_calls[-1]
        assert final_call[0][1] == IngestStatus.COMPLETED  # 2nd positional arg is status
        assert final_call[1]["memories_created"] == 0  # kwargs
        assert final_call[1]["candidates_created"] == 0
        assert final_call[1]["entities_created"] == 0

    @pytest.mark.asyncio
    async def test_end_to_end_processing(
        self,
        worker,
        redis,
        stage3_payload,
        mock_organizer,
        ingestion_id,
    ):
        """Test end-to-end processing through worker."""
        # Setup mock
        stage4_output = Stage4Output(
            input_id=str(ingestion_id),
            rdbms_memory_ids=["mem1"],
            rdbms_candidate_ids=[],
            graph_entity_ids=["ent1"],
            graph_relationship_ids=[],
            vector_ids=["vec1"],
        )
        mock_organizer.organize_and_persist = AsyncMock(
            return_value=stage4_output
        )

        # Enqueue item
        await redis.lpush(worker.input_queue, json.dumps(stage3_payload))

        # Process
        item = await worker._dequeue(timeout=1)
        result = await worker.process(item)

        # Final stage returns None
        assert result is None

        # No item in output queue (final stage)
        assert worker.output_queue is None
