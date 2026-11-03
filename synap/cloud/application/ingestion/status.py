"""Job status tracking for ingestion."""

import json
import logging
from typing import List, Optional
from uuid import UUID

from redis.asyncio import Redis

from synap.cloud.agents.ingestion.models import IngestResponse, IngestStatus

logger = logging.getLogger(__name__)


class StatusTracker:
    """Track status of ingestion jobs."""

    STATUS_TTL = 86400 * 7  # 7 days

    def __init__(self, redis: Redis):
        """Initialize status tracker.
        
        Args:
            redis: Async Redis client
        """
        self._redis = redis

    def _key(self, ingestion_id: UUID) -> str:
        """Get Redis key for ingestion ID.
        
        Args:
            ingestion_id: Ingestion ID
            
        Returns:
            Redis key string
        """
        return f"ingestion:status:{ingestion_id}"

    def _children_key(self, ingestion_id: UUID) -> str:
        return f"ingestion:children:{ingestion_id}"

    async def create(
        self,
        ingestion_id: UUID,
        response: IngestResponse,
    ) -> None:
        """Create initial status.
        
        Args:
            ingestion_id: Ingestion ID
            response: Initial response to store
        """
        await self._redis.setex(
            self._key(ingestion_id),
            self.STATUS_TTL,
            response.model_dump_json(),
        )
        logger.info(
            "ingestion_status_create ingestion_id=%s status=%s request_id=%s",
            ingestion_id,
            response.status.value,
            response.request_id,
        )

    async def create_parent(
        self,
        ingestion_id: UUID,
        response: IngestResponse,
        child_ids: List[UUID],
    ) -> None:
        """Store the status of a document that Stage 0 split into chunks.

        Each chunk is its own job and the workers update only those. The id
        returned to the caller is this one, and `get` answers for it from the
        chunks, so polling it reports the whole document rather than the
        first chunk.
        """
        await self._redis.setex(
            self._children_key(ingestion_id),
            self.STATUS_TTL,
            json.dumps([str(c) for c in child_ids]),
        )
        await self.create(ingestion_id, response)

    async def get(self, ingestion_id: UUID) -> Optional[IngestResponse]:
        """Get current status. For a split document, the chunks' combined status."""
        response = await self._get_own(ingestion_id)
        if response is None:
            return None
        raw = await self._redis.get(self._children_key(ingestion_id))
        if not raw:
            return response
        children = [await self._get_own(UUID(c)) for c in json.loads(raw)]
        return _combine(response, [c for c in children if c is not None])

    async def _get_own(self, ingestion_id: UUID) -> Optional[IngestResponse]:
        data = await self._redis.get(self._key(ingestion_id))
        if data:
            response = IngestResponse.model_validate_json(data)
            logger.debug(
                "ingestion_status_get_hit ingestion_id=%s status=%s",
                ingestion_id,
                response.status.value,
            )
            return response
        logger.debug("ingestion_status_get_miss ingestion_id=%s", ingestion_id)
        return None

    async def update(
        self,
        ingestion_id: UUID,
        status: IngestStatus,
        **updates,
    ) -> None:
        """Update status with additional fields.
        
        Args:
            ingestion_id: Ingestion ID
            status: New status value
            **updates: Additional fields to update (e.g., completed_at, error_message)
        """
        response = await self._get_own(ingestion_id)
        if not response:
            logger.warning(
                "ingestion_status_update_missing ingestion_id=%s new_status=%s",
                ingestion_id,
                status.value,
            )
            return

        response.status = status
        for key, value in updates.items():
            if hasattr(response, key):
                setattr(response, key, value)

        await self._redis.setex(
            self._key(ingestion_id),
            self.STATUS_TTL,
            response.model_dump_json(),
        )
        logger.info(
            "ingestion_status_update ingestion_id=%s status=%s updates=%s",
            ingestion_id,
            status.value,
            sorted(list(updates.keys())),
        )


_TERMINAL = {IngestStatus.COMPLETED, IngestStatus.FAILED, IngestStatus.PARTIAL_SUCCESS}


def _combine(parent: IngestResponse, children: List[IngestResponse]) -> IngestResponse:
    """One status for a document from the statuses of its chunks."""
    if not children:
        return parent
    statuses = [c.status for c in children]
    if all(s == IngestStatus.QUEUED for s in statuses):
        status = IngestStatus.QUEUED
    elif not all(s in _TERMINAL for s in statuses):
        status = IngestStatus.PROCESSING
    elif all(s == IngestStatus.COMPLETED for s in statuses):
        status = IngestStatus.COMPLETED
    elif all(s == IngestStatus.FAILED for s in statuses):
        status = IngestStatus.FAILED
    else:
        status = IngestStatus.PARTIAL_SUCCESS
    started = [c.started_at for c in children if c.started_at]
    errors = [c.error_message for c in children if c.error_message]
    return parent.model_copy(update={
        "status": status,
        "memories_created": sum(c.memories_created for c in children),
        "candidates_created": sum(c.candidates_created for c in children),
        "entities_created": sum(c.entities_created for c in children),
        "relationships_created": sum(c.relationships_created for c in children),
        "memory_ids": [m for c in children for m in c.memory_ids],
        "started_at": min(started) if started else None,
        "completed_at": (
            max(c.completed_at for c in children if c.completed_at)
            if status in _TERMINAL and any(c.completed_at for c in children) else None
        ),
        "error_message": "; ".join(dict.fromkeys(errors)) or None,
        "error_stage": next((c.error_stage for c in children if c.error_stage), None),
    })
