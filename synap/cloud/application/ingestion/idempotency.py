"""Idempotency handling for ingestion requests."""

import hashlib
from datetime import timedelta
from typing import Optional, Union
from uuid import UUID

from redis.asyncio import Redis

from synap.cloud.agents.ingestion.models import IngestResponse, Scope


def normalized_content_hash(content: Union[str, bytes]) -> str:
    """SHA-256 of the content with its whitespace collapsed: the hash the entry step keys on."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(b" ".join(content.split())).hexdigest()


class IdempotencyChecker:
    """
    Ensures idempotency via:
    - Request ID (network retry protection)
    - Content hash (duplicate content protection)
    """

    REQUEST_ID_TTL = timedelta(hours=24)
    CONTENT_HASH_TTL = timedelta(days=7)  # Longer for content dedup
    RESERVATION_TTL = 300  # 5 minutes for in-flight processing reservation

    def __init__(self, redis: Redis):
        """Initialize idempotency checker.

        Args:
            redis: Async Redis client
        """
        self._redis = redis

    def _hash_key(self, content_hash: str, scope: Scope) -> str:
        """Build content hash key scoped to instance+client+customer+user,
        plus the ladder node the write is filed at when it carries a chain.

        A write filed at an intermediate rung (`{"customer": "acme", "team":
        "payments"}`) has the same customer and no user as a customer-level
        write, so without the node the same text sent to two teams, or to a
        team and then its customer, returned the first ingestion and filed
        nothing at the second. Writes with no chain keep their old key.
        """
        leaf = f":{scope.chain[-1].node_id}" if scope.chain else ""
        return (
            f"idempotency:hash:{scope.client_id}"
            f":{scope.instance_id}"
            f":{scope.customer_id or '_none_'}"
            f":{scope.user_id or '_none_'}"
            f"{leaf}"
            f":{content_hash}"
        )

    async def check(
        self,
        request_id: str,
        content_hash: str,
        scope: Scope,
    ) -> Optional[IngestResponse]:
        """
        Check if this request/content has been processed.

        Returns cached response if found, None otherwise.

        Args:
            request_id: Request ID to check
            content_hash: Content hash to check
            scope: Request scope for content hash scoping

        Returns:
            Cached IngestResponse if duplicate detected, None otherwise
        """
        # Check request ID first (faster, more specific)
        request_key = f"idempotency:request:{request_id}"
        cached = await self._redis.get(request_key)
        if cached:
            return IngestResponse.model_validate_json(cached)

        # Check content hash (per-scope dedup)
        hash_key = self._hash_key(content_hash, scope)
        cached = await self._redis.get(hash_key)
        if cached and cached != b"processing" and cached != "processing":
            return IngestResponse.model_validate_json(cached)

        return None

    async def owner_of(self, content_hash: str, scope: Scope) -> Optional[UUID]:
        """The ingestion that holds this content hash in this scope, if one has been recorded.

        None while the hash is only reserved, when nothing is stored, or when the
        stored value cannot be read.
        """
        cached = await self._redis.get(self._hash_key(content_hash, scope))
        if not cached or cached in (b"processing", "processing"):
            return None
        try:
            return IngestResponse.model_validate_json(cached).ingestion_id
        except Exception:
            return None

    async def try_reserve(
        self,
        content_hash: str,
        scope: Scope,
    ) -> bool:
        """Atomically reserve a content hash to prevent duplicate processing.

        Uses Redis SETNX to ensure only one request processes a given content hash.
        The reservation expires after RESERVATION_TTL seconds.

        Args:
            content_hash: Content hash to reserve
            scope: Request scope for key scoping

        Returns:
            True if reservation was acquired, False if already reserved/processed
        """
        hash_key = self._hash_key(content_hash, scope)
        return await self._redis.set(
            hash_key, "processing", nx=True, ex=self.RESERVATION_TTL
        )

    async def store(
        self,
        request_id: str,
        response: IngestResponse,
        content_hash: Optional[str] = None,
        scope: Optional[Scope] = None,
    ) -> None:
        """Store response for idempotency.

        Args:
            request_id: Request ID
            response: Response to cache
            content_hash: Optional content hash for content-based dedup
            scope: Optional scope for content hash scoping
        """
        response_json = response.model_dump_json()

        # Store by request ID
        request_key = f"idempotency:request:{request_id}"
        await self._redis.setex(
            request_key,
            int(self.REQUEST_ID_TTL.total_seconds()),
            response_json,
        )

        # Store by content hash (overwrites "processing" reservation with actual response)
        if content_hash and scope:
            hash_key = self._hash_key(content_hash, scope)
            await self._redis.setex(
                hash_key,
                int(self.CONTENT_HASH_TTL.total_seconds()),
                response_json,
            )
