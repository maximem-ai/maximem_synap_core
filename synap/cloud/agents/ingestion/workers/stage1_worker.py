"""Worker for Stage 1: Type/Source/Nature Qualification."""

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import UUID

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    IngestStatus,
    PipelineStage,
)
from synap.cloud.agents.ingestion.pipeline.stage1_qualification import Stage1Qualifier
from synap.cloud.application.ingestion.idempotency import IdempotencyChecker, normalized_content_hash
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from .base import BaseIngestionWorker

if TYPE_CHECKING:  # Cloud's table-backed loader; Core passes the file loader, so nothing is imported at run time
    from synap.cloud.application.memory_architecture.repository import MemoryArchitectureRepository

logger = logging.getLogger(__name__)


class Stage1Worker(BaseIngestionWorker):
    """Worker for Stage 1 qualification."""

    @property
    def tracing_span_name(self) -> str:
        return "ingestion.stage1"

    @property
    def input_queue(self) -> str:
        # Reads from priority-merged queue (managed by separate router)
        return self.QUEUE_STAGE1

    @property
    def output_queue(self) -> str:
        return self.QUEUE_STAGE23

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.STAGE_1_QUALIFICATION

    def __init__(
        self,
        qualifier: Stage1Qualifier,
        config_loader: "MemoryArchitectureRepository",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._qualifier = qualifier
        self._config_loader = config_loader

    async def _earlier_hashes(self, request: IngestDocumentRequest, ingestion_id: Optional[str]) -> set:
        """The content hash of this request, when another ingestion holds it in the same scope.

        Two requests with the same text can both be queued when they arrive
        together. The entry step records one of them under the scoped hash key;
        the other is the duplicate. Only the ingestion with the larger id yields
        to the recorded one, so the pair can never both be dropped. Text from
        another user or scope has a different key and is never matched.
        """
        if not ingestion_id or not isinstance(request.content, str):
            return set()
        try:
            owner = await IdempotencyChecker(self._redis).owner_of(
                normalized_content_hash(request.content), request.scope)
            if owner is not None and owner < UUID(ingestion_id):
                return {Stage1Qualifier.content_hash(request.content)}
        except Exception:
            logger.warning("ingestion_stage1_duplicate_lookup_failed ingestion_id=%s", ingestion_id, exc_info=True)
        return set()

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process Stage 1: Qualify input."""
        # Reconstruct request
        request = IngestDocumentRequest.model_validate(payload["request"])
        ingestion_id = payload.get("ingestion_id")
        logger.info(
            "ingestion_stage1_start ingestion_id=%s request_id=%s document_type=%s source=%s",
            ingestion_id,
            request.request_id,
            request.document_type.value if request.document_type else None,
            request.source.value if request.source else None,
        )

        # Load MACA config for this instance
        async with start_span("ingestion.stage1.load_config", {
            "synap.ingestion_id": ingestion_id or "",
            "synap.instance_id": str(request.scope.instance_id),
        }):
            t0 = time.monotonic()
            config = await self._config_loader.get_config(
                instance_id=str(request.scope.instance_id),
            )
            set_current_span_attributes({
                "synap.config_loaded": config is not None,
                "synap.config_version": config.metadata.config_version if config else None,
                "synap.load_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # Qualify
        async with start_span("ingestion.stage1.qualify", {
            "synap.ingestion_id": ingestion_id or "",
            "synap.instance_id": str(request.scope.instance_id),
            "synap.document_type": request.document_type.value if request.document_type else None,
            "synap.source": request.source.value if request.source else None,
            "synap.content_len": len(request.content) if request.content else 0,
        }):
            t0 = time.monotonic()
            output = await self._qualifier.qualify(
                request,
                existing_hashes=await self._earlier_hashes(request, ingestion_id),
            )
            set_current_span_attributes({
                "synap.should_process": output.should_process,
                "synap.nature": output.nature.value if output.nature else None,
                "synap.qualify_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        logger.info(
            "ingestion_stage1_qualified ingestion_id=%s request_id=%s should_process=%s nature=%s",
            ingestion_id,
            request.request_id,
            output.should_process,
            output.nature.value if output.nature else None,
        )

        if not output.should_process:
            # Skip - don't pass to next stage, but set terminal status
            ingestion_uuid = UUID(payload["ingestion_id"])
            try:
                await self._status.update(
                    ingestion_uuid,
                    IngestStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                    memories_created=0,
                    candidates_created=0,
                    entities_created=0,
                    # Said in the status, so a caller does not read "completed, 0 memories" as a bug.
                    error_message=Stage1Qualifier.skip_reason(output.document_type, output.nature),
                )
            except Exception as e:
                logger.warning(
                    "Failed to update skip status for %s: %s", ingestion_uuid, e
                )
            logger.info(
                "ingestion_stage1_skipped ingestion_id=%s request_id=%s nature=%s",
                ingestion_id,
                request.request_id,
                output.nature.value if output.nature else None,
            )
            return None

        # Populate chunk fields from Stage 0 metadata
        chunk_meta = request.metadata.get("_chunk")
        if chunk_meta:
            output.is_chunked = True
            output.chunk_index = chunk_meta.get("chunk_index")
            output.total_chunks = chunk_meta.get("total_chunks")
            output.preceding_summary = chunk_meta.get("preceding_summary") or None

        # Pass to Stage 2/3
        return {
            "ingestion_id": payload["ingestion_id"],
            "request": request.model_dump(mode="json"),
            "stage1_output": output.model_dump(mode="json"),
            "maca_config_version": config.metadata.config_version if config else None,
        }
