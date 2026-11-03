"""Tests for Stage23Worker."""

import json
import pytest
from unittest.mock import AsyncMock

from synap.cloud.agents.ingestion.models import (
    PipelineStage,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    ContentNature,
    DocumentType,
    DocumentSource,
    MemoryType,
)
from synap.cloud.agents.ingestion.workers.stage23_worker import Stage23Worker


class TestStage23Worker:
    """Tests for Stage23Worker."""

    @pytest.fixture
    def worker(
        self,
        redis,
        mock_status_tracker,
        mock_reporting,
        mock_extractor,
        mock_config_loader,
    ):
        """Create Stage23Worker."""
        return Stage23Worker(
            redis=redis,
            status_tracker=mock_status_tracker,
            reporting=mock_reporting,
            worker_id="stage23-worker-1",
            extractor=mock_extractor,
            config_loader=mock_config_loader,
        )

    @pytest.fixture
    def stage1_payload(self, sample_payload, ingestion_id):
        """Create payload with Stage 1 output."""
        stage1_output = Stage1Output(
            input_id=str(ingestion_id),
            document_type=DocumentType.DOCUMENT,
            source=DocumentSource.API,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
        )
        return {
            **sample_payload,
            "stage1_output": stage1_output.model_dump(mode="json"),
            "maca_config_version": "v1.0.0",
        }

    @pytest.mark.asyncio
    async def test_worker_properties(self, worker):
        """Test worker property values."""
        assert worker.input_queue == "synap:ingestion:stage23"
        assert worker.output_queue == "synap:ingestion:stage4"
        assert worker.stage == PipelineStage.STAGE_3_EXTRACTION

    @pytest.mark.asyncio
    async def test_process_chains_stages(
        self,
        worker,
        stage1_payload,
        mock_extractor,
        ingestion_id,
    ):
        """Test that Stage 2 and Stage 3 are chained together.

        Extraction priorities are
        derived directly from the MACA config (see _build_stage2_from_config),
        so there is no Stage 2 LLM call. The stage2 -> stage3 chaining and the
        combined output payload remain the contract under test.
        """
        # Mock Stage 3 output
        stage3_output = Stage3Output(
            input_id=str(ingestion_id),
            memories=[],
            entities=[],
            relationships=[],
        )
        mock_extractor.extract = AsyncMock(return_value=stage3_output)

        result = await worker.process(stage1_payload)

        mock_extractor.extract.assert_called_once()

        # Verify output contains all stage outputs
        assert result is not None
        assert "stage1_output" in result
        assert "stage2_output" in result
        assert "stage3_output" in result
        # Stage 2 output is config-derived: no LLM categories, but extraction
        # priorities are populated from the config.
        assert result["stage2_output"]["categories"] == []
        assert len(result["stage2_output"]["extraction_priorities"]) > 0
        assert result["stage3_output"] == stage3_output.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_process_passes_correct_inputs(
        self,
        worker,
        stage1_payload,
        sample_request,
        mock_extractor,
        ingestion_id,
    ):
        """Test that correct inputs are passed to the extraction stage.

        Stage 2 priorities come from the MACA config,
        so the only LLM stage receiving inputs is the extractor. Verify it is
        called with the document content, a config-derived Stage2Output, and a
        document_id.
        """
        stage3_output = Stage3Output(
            input_id=str(ingestion_id),
            memories=[],
            entities=[],
            relationships=[],
        )
        mock_extractor.extract = AsyncMock(return_value=stage3_output)

        await worker.process(stage1_payload)


        # Verify Stage 3 inputs
        extract_call = mock_extractor.extract.call_args
        assert extract_call[1]["content"] == sample_request.content
        # stage2_output passed to the extractor is the config-derived one.
        passed_stage2 = extract_call[1]["stage2_output"]
        assert isinstance(passed_stage2, Stage2Output)
        assert passed_stage2.categories == []
        assert "document_id" in extract_call[1]

    @pytest.mark.asyncio
    async def test_end_to_end_processing(
        self,
        worker,
        redis,
        stage1_payload,
        mock_extractor,
        ingestion_id,
    ):
        """Test end-to-end processing through worker."""
        # Setup mocks
        stage2_output = Stage2Output(
            input_id=str(ingestion_id),
            categories=[],
            extraction_priorities=[],
        )

        stage3_output = Stage3Output(
            input_id=str(ingestion_id),
            memories=[],
            entities=[],
            relationships=[],
        )
        mock_extractor.extract = AsyncMock(return_value=stage3_output)

        # Enqueue item
        await redis.lpush(worker.input_queue, json.dumps(stage1_payload))

        # Process
        item = await worker._dequeue(timeout=1)
        result = await worker.process(item)

        # Enqueue to next stage
        if result:
            await worker._enqueue_next(result)

        # Verify item in stage4 queue
        next_item = await redis.brpop(worker.output_queue, timeout=1)
        assert next_item is not None
        _, payload = next_item
        data = json.loads(payload)
        assert data["ingestion_id"] == stage1_payload["ingestion_id"]
        assert "stage2_output" in data
        assert "stage3_output" in data
