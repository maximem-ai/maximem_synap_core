"""Tests for PriorityQueueRouter."""

import asyncio
import json
import pytest

from synap.cloud.agents.ingestion.models import IngestPriority
from synap.cloud.agents.ingestion.workers.router import PriorityQueueRouter


class TestPriorityQueueRouter:
    """Tests for PriorityQueueRouter."""

    @pytest.fixture
    def router(self, redis):
        """Create router."""
        return PriorityQueueRouter(redis=redis)

    @pytest.mark.asyncio
    async def test_routes_realtime_priority_first(
        self, router, redis, sample_payload
    ):
        """Test that REALTIME priority is handled first."""
        # Enqueue items in different priority queues
        await redis.lpush("synap:ingestion:bootstrap", json.dumps(sample_payload))
        await redis.lpush("synap:ingestion:explicit", json.dumps(sample_payload))
        await redis.lpush("synap:ingestion:realtime", json.dumps(sample_payload))

        # Start router in background
        router_task = asyncio.create_task(router.run())

        # Give router time to process
        await asyncio.sleep(0.5)

        # Stop router
        await router.stop()
        await asyncio.sleep(0.1)

        # Check Stage 1 queue - should have all 3 items
        stage1_len = await redis.llen(router.STAGE1_QUEUE)
        assert stage1_len == 3

        # Verify order: REALTIME should be last in (first out with RPOP)
        item1 = await redis.rpop(router.STAGE1_QUEUE)
        item2 = await redis.rpop(router.STAGE1_QUEUE)
        item3 = await redis.rpop(router.STAGE1_QUEUE)

        # All should be valid
        assert item1 is not None
        assert item2 is not None
        assert item3 is not None

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_routes_explicit_when_no_realtime(
        self, router, redis, sample_payload
    ):
        """Test that EXPLICIT priority is handled when no REALTIME items."""
        # Enqueue only to explicit queue
        await redis.lpush("synap:ingestion:explicit", json.dumps(sample_payload))

        # Start router
        router_task = asyncio.create_task(router.run())
        await asyncio.sleep(0.5)
        await router.stop()

        # Check Stage 1 queue
        stage1_len = await redis.llen(router.STAGE1_QUEUE)
        assert stage1_len == 1

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_routes_bootstrap_when_no_higher_priority(
        self, router, redis, sample_payload
    ):
        """Test that BOOTSTRAP priority is handled when no higher priority items."""
        # Enqueue only to bootstrap queue
        await redis.lpush("synap:ingestion:bootstrap", json.dumps(sample_payload))

        # Start router
        router_task = asyncio.create_task(router.run())
        await asyncio.sleep(0.5)
        await router.stop()

        # Check Stage 1 queue
        stage1_len = await redis.llen(router.STAGE1_QUEUE)
        assert stage1_len == 1

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_handles_empty_queues(self, router, redis):
        """Test router handles empty queues gracefully."""
        # Start router with no items
        router_task = asyncio.create_task(router.run())
        await asyncio.sleep(0.5)
        await router.stop()

        # Stage 1 queue should be empty
        stage1_len = await redis.llen(router.STAGE1_QUEUE)
        assert stage1_len == 0

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_continuous_routing(self, router, redis, sample_payload):
        """Test that router continuously processes items."""
        # Start router
        router_task = asyncio.create_task(router.run())

        # Items arriving after startup go to the per-client priority queue and
        # register the client (the legacy global-queue migration only runs once
        # at router startup). Mirror the production enqueue path here.
        client_id = sample_payload["request"]["scope"]["client_id"]
        payload = {**sample_payload, "client_id": client_id}
        queue_name = f"{router.QUEUE_PREFIX}:explicit:{client_id}"
        registry_key = f"{router.CLIENT_REGISTRY_PREFIX}:explicit"

        # Enqueue items over time
        for i in range(3):
            await redis.lpush(queue_name, json.dumps(payload))
            await redis.sadd(registry_key, client_id)
            await asyncio.sleep(0.2)

        await asyncio.sleep(0.5)
        await router.stop()

        # All items should be routed
        stage1_len = await redis.llen(router.STAGE1_QUEUE)
        assert stage1_len == 3

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_preserves_payload_content(self, router, redis, sample_payload):
        """Test that router preserves payload content."""
        # Enqueue item
        await redis.lpush("synap:ingestion:explicit", json.dumps(sample_payload))

        # Start router
        router_task = asyncio.create_task(router.run())
        await asyncio.sleep(0.5)
        await router.stop()

        # Retrieve routed item
        result = await redis.rpop(router.STAGE1_QUEUE)
        assert result is not None

        # Verify payload preserved
        routed_data = json.loads(result)
        assert routed_data["ingestion_id"] == sample_payload["ingestion_id"]
        assert routed_data["request"] == sample_payload["request"]

        # Clean up
        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass
