"""Redis queue integration for ingestion pipeline with per-client fair scheduling."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
import json

from opentelemetry import propagate as otel_propagate
from redis.asyncio import Redis

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    IngestPriority,
)

logger = logging.getLogger(__name__)

_warned_otel_context_empty = False


class IngestionQueue:
    """
    Redis-based queue for ingestion pipeline with per-client fairness.

    Queue structure:
    - Per-client queues: synap:ingestion:{priority}:{client_id}
    - Client registry: synap:ingestion:clients:{priority} (SET of active client_ids)

    Round-robin dequeue ensures no single client can starve others.
    """

    QUEUE_PREFIX = "synap:ingestion"
    CLIENT_REGISTRY_PREFIX = "synap:ingestion:clients"

    # Legacy global queue names (for migration)
    LEGACY_QUEUES = [
        "synap:ingestion:realtime",
        "synap:ingestion:explicit",
        "synap:ingestion:bootstrap",
    ]

    def __init__(self, redis: Redis):
        """Initialize ingestion queue.

        Args:
            redis: Async Redis client
        """
        self._redis = redis
        # Round-robin cursor per priority level (index into sorted client list)
        self._cursors: Dict[str, int] = {}

    def _client_queue_name(self, priority: IngestPriority, client_id: str) -> str:
        """Get per-client queue name.

        Args:
            priority: Priority level
            client_id: Client identifier

        Returns:
            Queue name string like synap:ingestion:explicit:ba5a90c1-bbe6-...
        """
        return f"{self.QUEUE_PREFIX}:{priority.value}:{client_id}"

    def _client_registry_key(self, priority: IngestPriority) -> str:
        """Get client registry key for a priority level."""
        return f"{self.CLIENT_REGISTRY_PREFIX}:{priority.value}"

    async def enqueue(
        self,
        ingestion_id: UUID,
        request: IngestDocumentRequest,
        priority: IngestPriority,
    ) -> None:
        """Add item to appropriate per-client priority queue.

        Args:
            ingestion_id: Unique ingestion ID
            request: Document request to enqueue
            priority: Priority level for queue selection
        """
        client_id = str(request.scope.client_id)
        queue_name = self._client_queue_name(priority, client_id)
        registry_key = self._client_registry_key(priority)

        # Capture current OTel trace context so worker spans can be linked
        # as children of the originating sdk.memory.create trace.
        trace_carrier: Dict[str, str] = {}
        otel_propagate.inject(trace_carrier)
        if not trace_carrier:
            # Once per process: with no tracing set up (every Core install) this is true for every
            # item, and a warning per document buries the log.
            global _warned_otel_context_empty
            if not _warned_otel_context_empty:
                _warned_otel_context_empty = True
                logger.warning(
                    "ingestion_queue_otel_context_empty ingestion_id=%s — "
                    "W3C propagator may not be configured; worker spans will appear as separate traces "
                    "(logged once per process)",
                    ingestion_id,
                )

        payload = {
            "ingestion_id": str(ingestion_id),
            "client_id": client_id,
            "request": request.model_dump(mode="json"),
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
            "_otel_context": trace_carrier,
        }

        # Pipeline: push to client queue + register client atomically
        pipe = self._redis.pipeline()
        pipe.lpush(queue_name, json.dumps(payload))
        pipe.sadd(registry_key, client_id)
        results = await pipe.execute()
        queue_len = results[0]

        logger.info(
            "ingestion_queue_enqueued ingestion_id=%s request_id=%s queue=%s priority=%s queue_length=%s",
            ingestion_id,
            request.request_id,
            queue_name,
            priority.value,
            queue_len,
        )

    async def get_active_clients(self, priority: IngestPriority) -> List[str]:
        """Get list of clients with items in queue for a priority level.

        Args:
            priority: Priority level

        Returns:
            Sorted list of client_id strings
        """
        registry_key = self._client_registry_key(priority)
        clients = await self._redis.smembers(registry_key)
        return sorted(str(c) for c in clients)

    async def dequeue_for_client(
        self, priority: IngestPriority, client_id: str
    ) -> Optional[Dict[str, Any]]:
        """Dequeue one item from a specific client's queue.

        If the queue becomes empty, removes the client from the registry.

        Args:
            priority: Priority level
            client_id: Client identifier

        Returns:
            Dequeued payload dict or None if queue empty
        """
        queue_name = self._client_queue_name(priority, client_id)
        raw = await self._redis.rpop(queue_name)

        if raw is None:
            # Queue empty — clean up client from registry
            registry_key = self._client_registry_key(priority)
            await self._redis.srem(registry_key, client_id)
            return None

        item = json.loads(raw)
        logger.info(
            "ingestion_queue_dequeued ingestion_id=%s queue=%s client_id=%s",
            item.get("ingestion_id"),
            queue_name,
            client_id,
        )
        return item

    async def dequeue_round_robin(self, priority: IngestPriority) -> Optional[Dict[str, Any]]:
        """Dequeue one item using round-robin across clients for a priority level.

        Args:
            priority: Priority level

        Returns:
            Dequeued payload dict or None if no items available
        """
        clients = await self.get_active_clients(priority)
        if not clients:
            return None

        cursor_key = priority.value
        cursor = self._cursors.get(cursor_key, 0)

        # Try each client starting from cursor position
        for i in range(len(clients)):
            idx = (cursor + i) % len(clients)
            client_id = clients[idx]
            item = await self.dequeue_for_client(priority, client_id)
            if item is not None:
                # Advance cursor past this client for next call
                self._cursors[cursor_key] = (idx + 1) % max(len(clients), 1)
                return item

        return None

    async def dequeue(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Dequeue from highest priority queue with items, using round-robin.

        Checks queues in order: REALTIME > EXPLICIT > BOOTSTRAP
        Uses round-robin within each priority level for client fairness.

        Args:
            timeout: Not used (kept for API compatibility), polling is non-blocking

        Returns:
            Dequeued payload dict or None if all queues empty
        """
        for priority in [IngestPriority.REALTIME, IngestPriority.EXPLICIT, IngestPriority.BOOTSTRAP]:
            item = await self.dequeue_round_robin(priority)
            if item is not None:
                return item

        return None

    async def queue_lengths(self) -> Dict[str, int]:
        """Get total length of each priority level (sum of all client queues).

        Returns:
            Dict mapping priority name to total queue length
        """
        lengths = {}
        for priority in [IngestPriority.REALTIME, IngestPriority.EXPLICIT, IngestPriority.BOOTSTRAP]:
            clients = await self.get_active_clients(priority)
            total = 0
            for client_id in clients:
                queue_name = self._client_queue_name(priority, client_id)
                total += await self._redis.llen(queue_name)
            lengths[priority.value] = total
        return lengths
