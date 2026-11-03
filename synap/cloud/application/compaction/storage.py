"""
Compaction Storage: Redis + Postgres backend for compacted context.

Implements dual-storage strategy per D6:
- Redis: Hot cache for fast access (1 hour TTL)
- Postgres: Persistent storage with versioning
"""
import json
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import redis.asyncio as redis
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.connection import get_session_factory
from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

from .models import (
    CompactedContext,
    ConversationMessage
)


logger = logging.getLogger(__name__)


class CompactionStorage:
    """
    Dual storage for compacted context.

    - Redis: Hot cache for fast access (1 hour TTL)
    - Postgres: Persistent storage with versioning
    """

    def __init__(
        self,
        redis_url: Optional[str] = None
    ):
        """
        Initialize storage.

        Args:
            redis_url: Redis connection URL. Defaults to localhost.
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    # ==========================================================================
    # Redis Operations (Hot Cache)
    # ==========================================================================

    async def get_from_cache(
        self,
        conversation_id: str,
    ) -> Optional[CompactedContext]:
        """Get compacted context from Redis cache."""
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            key = self._cache_key(conversation_id)

            data = await r.get(key)
            if data:
                return CompactedContext.from_dict(json.loads(data))
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")
        
        return None

    async def set_in_cache(
        self,
        context: CompactedContext,
        ttl_seconds: int = 3600  # 1 hour
    ):
        """Set compacted context in Redis cache."""
        try:
            r = await self._get_redis()
            key = self._cache_key(coerce_conversation_id(context.metadata.conversation_id))

            await r.setex(
                key,
                ttl_seconds,
                json.dumps(context.to_dict())
            )
        except Exception as e:
            logger.warning(f"Redis cache write failed (non-fatal): {e}")

    async def invalidate_cache(self, conversation_id: str):
        """Invalidate cached compacted context."""
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            key = self._cache_key(conversation_id)
            await r.delete(key)
        except Exception as e:
            logger.warning(f"Redis cache invalidation failed: {e}")

    def _cache_key(self, conversation_id: str) -> str:
        """Generate Redis cache key."""
        return f"compacted:{conversation_id}:latest"

    # ==========================================================================
    # Compaction Lock (prevent concurrent compaction)
    # ==========================================================================

    async def is_compaction_in_progress(self, conversation_id: str) -> bool:
        """Check if compaction is already in progress."""
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            key = f"compacting:{conversation_id}"
            result = await r.exists(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis lock check failed: {e}")
            return False

    async def set_compaction_in_progress(
        self,
        conversation_id: str,
        ttl_seconds: int = 120  # 2 minute timeout
    ) -> bool:
        """Atomically set compaction lock. Returns True if lock acquired, False if already held."""
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            key = f"compacting:{conversation_id}"
            result = await r.set(key, "1", ex=ttl_seconds, nx=True)
            return result is not None  # SET NX returns None if key already exists
        except Exception as e:
            logger.warning(f"Redis lock set failed: {e}")
            return False

    async def clear_compaction_in_progress(self, conversation_id: str):
        """Clear compaction in progress flag."""
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            key = f"compacting:{conversation_id}"
            await r.delete(key)
        except Exception as e:
            logger.warning(f"Redis lock clear failed: {e}")

    # ==========================================================================
    # Postgres Operations (Persistent Storage)
    # ==========================================================================

    async def save(self, context: CompactedContext):
        """Save compacted context to both Redis and Postgres in parallel.

        Redis failure is non-fatal (already logged inside set_in_cache).
        Postgres failure propagates since it's the persistent store.
        """
        import asyncio
        redis_result, postgres_result = await asyncio.gather(
            self.set_in_cache(context),
            self._save_to_postgres(context),
            return_exceptions=True,
        )
        # Postgres failure must propagate
        if isinstance(postgres_result, Exception):
            raise postgres_result

    async def _save_to_postgres(self, context: CompactedContext):
        """Save to Postgres with versioning."""
        session_factory = get_session_factory()
        # FK ``compacted_contexts.conversation_id`` references ``conversation``
        # whose rows are keyed by the coerced UUID5. Writes must use the same
        # shape — otherwise ``manual_api`` compactions (which arrive with the
        # raw ``pgv2_xxx`` form) violate the FK.
        conv_id = coerce_conversation_id(context.metadata.conversation_id)

        async with session_factory() as session:
            try:
                # Get next version number
                version = await self._get_next_version(
                    session,
                    conv_id,
                )

                # Store the record version in metadata so it's included in serialized JSON
                context.metadata.version = version

                # Ensure created_at is timezone-aware
                created_at = context.metadata.created_at
                if created_at and created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

                # Insert record (CAST for UUID/JSONB columns - asyncpg is strict)
                client_id_val = context.metadata.client_id if hasattr(context.metadata, 'client_id') else context.metadata.instance_id
                await session.execute(
                    text("""
                    INSERT INTO compacted_contexts (
                        id, instance_id, client_id, conversation_id, version,
                        compacted_content, original_token_count,
                        compacted_token_count, compression_ratio,
                        validation_score, trigger_type, created_at
                    ) VALUES (
                        CAST(:id AS uuid), :instance_id, :client_id, :conversation_id, :version,
                        CAST(:compacted_content AS jsonb), :original_token_count,
                        :compacted_token_count, :compression_ratio,
                        :validation_score, :trigger_type, :created_at
                    )
                    """),
                    {
                        "id": str(uuid4()),
                        "instance_id": context.metadata.instance_id,
                        "client_id": client_id_val,
                        "conversation_id": conv_id,
                        "version": version,
                        "compacted_content": json.dumps(context.to_dict()),
                        "original_token_count": context.metadata.original_token_count,
                        "compacted_token_count": context.metadata.compacted_token_count,
                        "compression_ratio": context.metadata.compression_ratio,
                        "validation_score": context.quality_assurance.validation_score,
                        "trigger_type": context.metadata.trigger_type.value,
                        "created_at": created_at
                    }
                )
                
                await session.commit()

                # Cleanup old versions (best-effort, failure doesn't fail the save)
                try:
                    await self._cleanup_old_versions(
                        session,
                        conv_id,
                    )
                except Exception as cleanup_err:
                    logger.warning(f"Version cleanup failed (non-fatal): {cleanup_err}")

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to save compacted context to Postgres: {e}")
                raise

    async def _cleanup_old_versions(
        self,
        session: AsyncSession,
        conversation_id: str,
        keep: int = 3
    ):
        """Delete compaction versions older than the latest `keep`.

        Runs after a successful save to prevent unbounded DB growth.
        """
        await session.execute(
            text("""
            DELETE FROM compacted_contexts
            WHERE conversation_id = :conversation_id
              AND version <= (
                  SELECT COALESCE(MAX(version), 0) - :keep
                  FROM compacted_contexts
                  WHERE conversation_id = :conversation_id
              )
            """),
            {"conversation_id": conversation_id, "keep": keep}
        )
        await session.commit()

    async def _get_next_version(
        self,
        session: AsyncSession,
        conversation_id: str,
    ) -> int:
        """Get next version number for a conversation's compaction."""
        result = await session.execute(
            text("""
            SELECT COALESCE(MAX(version), 0) + 1
            FROM compacted_contexts
            WHERE conversation_id = :conversation_id
            """),
            {"conversation_id": conversation_id}
        )

        version = result.scalar()
        return version or 1

    async def get_latest(
        self,
        conversation_id: str,
    ) -> Optional[CompactedContext]:
        """Get latest compacted context (cache first, then Postgres)."""
        conversation_id = coerce_conversation_id(conversation_id)
        # Try cache first
        cached = await self.get_from_cache(conversation_id)
        if cached:
            return cached

        # Fall back to Postgres
        return await self._get_from_postgres(conversation_id)

    async def _get_from_postgres(
        self,
        conversation_id: str,
    ) -> Optional[CompactedContext]:
        """Get latest from Postgres."""
        session_factory = get_session_factory()

        async with session_factory() as session:
            result = await session.execute(
                text("""
                SELECT compacted_content, version
                FROM compacted_contexts
                WHERE conversation_id = :conversation_id
                ORDER BY version DESC
                LIMIT 1
                """),
                {"conversation_id": conversation_id}
            )

            row = result.first()
            if row:
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                context = CompactedContext.from_dict(data)
                context.metadata.version = row[1]
                return context

        return None

    async def get_version(
        self,
        conversation_id: str,
        version: int,
    ) -> Optional[CompactedContext]:
        """Get specific version of compacted context."""
        conversation_id = coerce_conversation_id(conversation_id)
        session_factory = get_session_factory()

        async with session_factory() as session:
            result = await session.execute(
                text("""
                SELECT compacted_content, version
                FROM compacted_contexts
                WHERE conversation_id = :conversation_id
                  AND version = :version
                """),
                {
                    "conversation_id": conversation_id,
                    "version": version,
                }
            )

            row = result.first()
            if row:
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                context = CompactedContext.from_dict(data)
                context.metadata.version = row[1]
                return context

        return None

    # ==========================================================================
    # Worker Support Methods
    # ==========================================================================

    async def get_active_conversations(
        self,
        since: datetime
    ) -> List[Dict[str, Any]]:
        """Get conversations with recent activity."""
        session_factory = get_session_factory()
        
        async with session_factory() as session:
            try:
                result = await session.execute(
                    text("""
                    SELECT
                        c.conversation_id::text,
                        c.instance_id,
                        c.last_message_at,
                        c.message_count,
                        c.approximate_token_count as current_token_count,
                        c.user_id,
                        c.customer_id,
                        i.client_id,
                        -- The rung this conversation was opened at. Compaction
                        -- summarises a conversation and hands the result on,
                        -- so it has to know where the conversation sits, and
                        -- customer_id/user_id cannot say "the high-street
                        -- practice". NULL for every conversation that named no
                        -- rung, which is all of them today.
                        c.scope_path
                    FROM conversation c
                    JOIN instance i ON c.instance_id = i.instance_id
                    WHERE c.last_message_at >= :since
                      AND c.status = 'active'
                    ORDER BY c.last_message_at DESC
                    LIMIT 100
                    """),
                    {"since": since}
                )

                rows = result.fetchall()
                return [
                    {
                        "conversation_id": row[0],
                        "instance_id": row[1],
                        "last_message_at": row[2],
                        "message_count": row[3],
                        "current_token_count": row[4],
                        "user_id": row[5],
                        "customer_id": row[6],
                        "client_id": row[7],
                        "scope_path": row[8],
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.warning(f"Failed to get active conversations: {e}")
                return []

    async def get_unflushed_idle_conversations(
        self,
        before: datetime,
        after: Optional[tuple] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """One page of active conversations last touched before ``before`` that still hold
        content nothing has flushed: turns not yet promoted to long-term memory, or turns
        newer than the latest compaction. Newest first; ``after`` is the
        ``(last_message_at, conversation_id)`` of the last row of the previous page."""
        session_factory = get_session_factory()
        params: Dict[str, Any] = {"before": before, "lim": limit}
        cursor = ""
        if after is not None:
            cursor = "AND (c.last_message_at, c.conversation_id::text) < (:after_at, :after_id)"
            params["after_at"], params["after_id"] = after

        async with session_factory() as session:
            try:
                result = await session.execute(
                    text(f"""
                    SELECT
                        c.conversation_id::text,
                        c.instance_id,
                        c.last_message_at,
                        c.message_count,
                        c.approximate_token_count as current_token_count,
                        c.user_id,
                        c.customer_id,
                        i.client_id,
                        c.scope_path
                    FROM conversation c
                    JOIN instance i ON c.instance_id = i.instance_id
                    WHERE c.last_message_at < :before
                      AND c.status = 'active'
                      AND (
                          c.promoted_message_count < c.message_count
                          OR NOT EXISTS (
                              SELECT 1 FROM compacted_contexts cc
                              WHERE cc.conversation_id = c.conversation_id
                                AND cc.created_at >= c.last_message_at
                          )
                      )
                      {cursor}
                    ORDER BY c.last_message_at DESC, c.conversation_id::text DESC
                    LIMIT :lim
                    """),
                    params,
                )
                return [
                    {
                        "conversation_id": row[0],
                        "instance_id": row[1],
                        "last_message_at": row[2],
                        "message_count": row[3],
                        "current_token_count": row[4],
                        "user_id": row[5],
                        "customer_id": row[6],
                        "client_id": row[7],
                        "scope_path": row[8],
                    }
                    for row in result.fetchall()
                ]
            except Exception as e:
                logger.warning(f"Failed to get idle conversations: {e}")
                return []

    async def load_conversation_messages(
        self,
        conversation_id: str
    ) -> List[ConversationMessage]:
        """Load messages for a conversation."""
        conversation_id = coerce_conversation_id(conversation_id)
        session_factory = get_session_factory()
        
        async with session_factory() as session:
            try:
                result = await session.execute(
                    text("""
                    SELECT
                        message_id::text,
                        message_type::text as role,
                        content::text,
                        timestamp,
                        metadata::text
                    FROM message
                    WHERE conversation_id = :conversation_id
                      AND message_type IN ('user', 'assistant')
                    ORDER BY timestamp ASC, message_id ASC
                    """),
                    {"conversation_id": conversation_id}
                )

                rows = result.fetchall()
                messages = []
                for row in rows:
                    # Content is stored as JSONB. After ::text cast and
                    # json.loads it may be a str (if stored via json.dumps(str))
                    # or a dict (legacy/external inserts). Normalise to str.
                    raw_content = json.loads(row[2]) if row[2] else ""
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, dict):
                        content = raw_content.get("text", str(raw_content))
                    else:
                        content = str(raw_content)

                    messages.append(
                        ConversationMessage(
                            message_id=row[0],
                            role=row[1],
                            content=content,
                            timestamp=row[3],
                            metadata=json.loads(row[4]) if row[4] else None,
                        )
                    )
                return messages
            except Exception as e:
                logger.warning(f"Failed to load conversation messages: {e}")
                return []

    async def load_messages_since_compaction(
        self,
        conversation_id: str,
        cutoff_timestamp: str,
    ) -> List[ConversationMessage]:
        """Load messages that arrived after the last compaction cutoff.

        Args:
            conversation_id: The conversation to load messages for.
            cutoff_timestamp: ISO timestamp from source_message_range.end_timestamp.
                Only messages with timestamp strictly after this are returned.
        """
        conversation_id = coerce_conversation_id(conversation_id)
        session_factory = get_session_factory()

        async with session_factory() as session:
            try:
                result = await session.execute(
                    text("""
                    SELECT
                        message_id::text,
                        message_type::text as role,
                        content::text,
                        timestamp,
                        metadata::text
                    FROM message
                    WHERE conversation_id = :conversation_id
                      AND message_type IN ('user', 'assistant')
                      AND timestamp > CAST(:cutoff_timestamp AS timestamptz)
                    ORDER BY timestamp ASC, message_id ASC
                    """),
                    {
                        "conversation_id": conversation_id,
                        "cutoff_timestamp": cutoff_timestamp,
                    }
                )

                rows = result.fetchall()
                messages = []
                for row in rows:
                    raw_content = json.loads(row[2]) if row[2] else ""
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, dict):
                        content = raw_content.get("text", str(raw_content))
                    else:
                        content = str(raw_content)

                    messages.append(
                        ConversationMessage(
                            message_id=row[0],
                            role=row[1],
                            content=content,
                            timestamp=row[3],
                            metadata=json.loads(row[4]) if row[4] else None,
                        )
                    )
                return messages
            except Exception as e:
                logger.warning(f"Failed to load messages since compaction: {e}")
                return []

    # ==========================================================================
    # Redis Raw Message Buffer
    # ==========================================================================

    async def load_raw_messages_from_redis(
        self,
        conversation_id: str,
        after_timestamp: Optional[str] = None,
    ) -> List[ConversationMessage]:
        """Load raw buffered messages from Redis.

        The ConversationTracker RPUSHes each turn to
        ``conv_messages:{conversation_id}``.  This method reads them back,
        optionally filtering to only messages after *after_timestamp*
        (ISO 8601 string from the last compaction's source_message_range).

        Returns an empty list on any failure (non-fatal).
        """
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            redis_key = f"conv_messages:{conversation_id}"
            raw_entries = await r.lrange(redis_key, 0, -1)

            messages = []
            for entry_str in raw_entries:
                entry = json.loads(entry_str)
                msg_ts = entry.get("timestamp", "")

                if after_timestamp and msg_ts <= after_timestamp:
                    continue

                messages.append(ConversationMessage(
                    message_id=entry["message_id"],
                    role=entry["role"],
                    content=entry["content"],
                    timestamp=(
                        datetime.fromisoformat(msg_ts) if msg_ts
                        else datetime.now(timezone.utc)
                    ),
                    metadata=None,
                ))
            return messages
        except Exception as e:
            logger.warning(f"Failed to load raw messages from Redis: {e}")
            return []

    async def trim_raw_messages(
        self,
        conversation_id: str,
        up_to_timestamp: str,
    ) -> int:
        """Remove compacted messages from the Redis raw buffer.

        Trims entries whose timestamp <= *up_to_timestamp*.
        Returns the number of entries removed.
        """
        conversation_id = coerce_conversation_id(conversation_id)
        try:
            r = await self._get_redis()
            redis_key = f"conv_messages:{conversation_id}"
            raw_entries = await r.lrange(redis_key, 0, -1)

            trim_index = -1
            for i, entry_str in enumerate(raw_entries):
                entry = json.loads(entry_str)
                if entry.get("timestamp", "") <= up_to_timestamp:
                    trim_index = i

            if trim_index >= 0:
                await r.ltrim(redis_key, trim_index + 1, -1)
                return trim_index + 1
            return 0
        except Exception as e:
            logger.warning(f"Redis buffer trim failed (non-fatal): {e}")
            return 0

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    async def close(self):
        """Close connections."""
        if self._redis:
            await self._redis.close()
