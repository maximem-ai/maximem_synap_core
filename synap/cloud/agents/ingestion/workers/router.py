"""Routes items from per-client priority queues to stage queues with fair scheduling."""

import asyncio
import json
import logging
from redis.asyncio import Redis

from synap.cloud.agents.ingestion.models import IngestPriority
from synap.cloud.observability.tracing import start_span, set_current_span_attributes


logger = logging.getLogger(__name__)


class PriorityQueueRouter:
    """
    Routes items from per-client priority queues to Stage 1 queue.

    Uses round-robin scheduling within each priority level to ensure
    no single client can starve others. Checks priorities in order:
    REALTIME > EXPLICIT > BOOTSTRAP.
    """

    QUEUE_PREFIX = "synap:ingestion"
    CLIENT_REGISTRY_PREFIX = "synap:ingestion:clients"
    STAGE1_QUEUE = "synap:ingestion:stage1"

    # Legacy global queue names (pre-fairness)
    LEGACY_QUEUES = {
        "synap:ingestion:realtime": IngestPriority.REALTIME,
        "synap:ingestion:explicit": IngestPriority.EXPLICIT,
        "synap:ingestion:bootstrap": IngestPriority.BOOTSTRAP,
    }

    PRIORITIES = [IngestPriority.REALTIME, IngestPriority.EXPLICIT, IngestPriority.BOOTSTRAP]

    def __init__(self, redis: Redis):
        self._redis = redis
        self._running = False
        # Round-robin cursor per priority level
        self._cursors: dict[str, int] = {}

    async def _migrate_legacy_queues(self):
        """Drain old global queues into per-client queues (one-time migration).

        Reads client_id from each payload and pushes to the appropriate
        per-client queue. Safe to run multiple times (idempotent once drained).
        """
        total_migrated = 0

        for legacy_key, priority in self.LEGACY_QUEUES.items():
            count = 0
            while True:
                raw = await self._redis.rpop(legacy_key)
                if raw is None:
                    break

                try:
                    payload = json.loads(raw)
                    # Extract client_id from the nested request.scope
                    client_id = (
                        payload.get("client_id")
                        or payload.get("request", {}).get("scope", {}).get("client_id")
                    )
                    if not client_id:
                        # Can't determine client — push to a catch-all
                        client_id = "_unknown"

                    # Ensure client_id is in the payload at top level
                    payload["client_id"] = client_id

                    queue_name = f"{self.QUEUE_PREFIX}:{priority.value}:{client_id}"
                    registry_key = f"{self.CLIENT_REGISTRY_PREFIX}:{priority.value}"

                    pipe = self._redis.pipeline()
                    pipe.lpush(queue_name, json.dumps(payload))
                    pipe.sadd(registry_key, client_id)
                    await pipe.execute()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to migrate legacy queue item: {e}")
                    # Push back to avoid data loss
                    await self._redis.lpush(legacy_key, raw)
                    break

            if count > 0:
                logger.info(
                    "ingestion_router_migrated_legacy queue=%s count=%s priority=%s",
                    legacy_key, count, priority.value,
                )
                total_migrated += count

        if total_migrated > 0:
            logger.info("ingestion_router_legacy_migration_complete total=%s", total_migrated)

    async def _dequeue_round_robin(self, priority: IngestPriority) -> bytes | None:
        """Dequeue one item using round-robin across clients for a priority level.

        Returns:
            Raw JSON bytes payload or None
        """
        registry_key = f"{self.CLIENT_REGISTRY_PREFIX}:{priority.value}"
        clients = sorted(str(c) for c in await self._redis.smembers(registry_key))
        if not clients:
            return None

        cursor_key = priority.value
        cursor = self._cursors.get(cursor_key, 0)

        for i in range(len(clients)):
            idx = (cursor + i) % len(clients)
            client_id = clients[idx]
            queue_name = f"{self.QUEUE_PREFIX}:{priority.value}:{client_id}"

            raw = await self._redis.rpop(queue_name)
            if raw is not None:
                # Advance cursor past this client
                self._cursors[cursor_key] = (idx + 1) % max(len(clients), 1)

                # Check if client queue is now empty → remove from registry
                remaining = await self._redis.llen(queue_name)
                if remaining == 0:
                    await self._redis.srem(registry_key, client_id)

                return raw

        return None

    async def run(self):
        """Main routing loop."""
        self._running = True
        logger.info("PriorityQueueRouter starting")

        # Migrate any items from legacy global queues
        await self._migrate_legacy_queues()

        while self._running:
            try:
                routed = False
                for priority in self.PRIORITIES:
                    payload = await self._dequeue_round_robin(priority)
                    if payload is not None:
                        try:
                            parsed = json.loads(payload)
                            ingestion_id = parsed.get("ingestion_id")
                            client_id = parsed.get("client_id")
                        except Exception:
                            ingestion_id = None
                            client_id = None

                        async with start_span("ingestion.router", {
                            "synap.ingestion_id": ingestion_id or "",
                            "synap.source_priority": priority.value,
                            "synap.client_id": client_id or "",
                        }):
                            async with start_span("ingestion.router.enqueue_stage1", {
                                "synap.ingestion_id": ingestion_id or "",
                                "synap.destination_queue": self.STAGE1_QUEUE,
                            }):
                                stage1_len = await self._redis.lpush(self.STAGE1_QUEUE, payload)
                                set_current_span_attributes({
                                    "synap.stage1_queue_length": stage1_len,
                                })

                        logger.info(
                            "ingestion_router_routed ingestion_id=%s client_id=%s priority=%s destination_queue=%s destination_length=%s",
                            ingestion_id,
                            client_id,
                            priority.value,
                            self.STAGE1_QUEUE,
                            stage1_len,
                        )
                        routed = True
                        break  # Re-check from highest priority

                if not routed:
                    # No items in any queue — sleep briefly
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Router error: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info("PriorityQueueRouter stopped")

    async def stop(self):
        """Stop the router gracefully."""
        self._running = False
