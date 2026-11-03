"""Async wrapper for RDBMS provider."""

import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    ExtractedEntity,
    ExtractedRelationship,
    Scope,
)
from synap.providers.rdbms import RDBMSInterface
from synap.providers.rdbms.models import SQLQuery


class AsyncRDBMSWrapper:
    """
    Async wrapper around sync RDBMSInterface.
    
    Uses asyncio.to_thread() to make blocking calls non-blocking.
    Maps ingestion domain models to SQL queries.
    """

    def __init__(self, provider: RDBMSInterface):
        self._provider = provider

    async def insert_memory(
        self,
        memory: ExtractedMemory,
        scope: Scope,
        document_id: str,
    ) -> UUID:
        """Insert a memory into the memories table."""
        import logging
        from synap.cloud.shared.logging_utils import truncate_string

        logger = logging.getLogger(__name__)

        logger.debug(f"memory_insert_attempt memory_id={memory.memory_id} content={truncate_string(memory.content, 150)} instance_id={scope.instance_id} user_id={scope.user_id} document_id={document_id}")

        query = SQLQuery(
            sql="""
                INSERT INTO memories (
                    id, client_id, instance_id, customer_id, user_id,
                    memory_type, content, confidence,
                    category, subcategory, source_document_id,
                    pii_obfuscated, pii_detected, created_at, document_created_at,
                    temporal_category, valid_from, valid_until,
                    temporal_confidence, is_episode, episode_metadata,
                    assertion_source,
                    event_start, event_end, event_precision
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s,
                    %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    event_start = EXCLUDED.event_start,
                    event_end = EXCLUDED.event_end,
                    event_precision = EXCLUDED.event_precision,
                    temporal_category = EXCLUDED.temporal_category,
                    valid_from = EXCLUDED.valid_from,
                    valid_until = EXCLUDED.valid_until,
                    temporal_confidence = EXCLUDED.temporal_confidence,
                    is_episode = EXCLUDED.is_episode,
                    episode_metadata = EXCLUDED.episode_metadata,
                    document_created_at = EXCLUDED.document_created_at,
                    content = EXCLUDED.content,
                    confidence = EXCLUDED.confidence,
                    assertion_source = EXCLUDED.assertion_source
            """,
            params=(
                str(memory.memory_id),
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
                str(scope.user_id) if scope.user_id else None,
                memory.memory_type.value,
                memory.content,
                memory.confidence,
                memory.category,
                memory.subcategory,
                document_id,
                memory.pii_obfuscated,
                getattr(memory, "pii_detected", None),
                memory.extracted_at,
                memory.document_created_at,
                memory.temporal_category,
                memory.valid_from,
                memory.valid_until,
                memory.temporal_confidence,
                memory.is_episode,
                json.dumps(memory.episode_metadata) if memory.episode_metadata else None,
                memory.assertion_source,
                getattr(memory, "event_start", None),
                getattr(memory, "event_end", None),
                getattr(memory, "event_precision", None),
            )
        )

        try:
            await asyncio.to_thread(self._provider.execute, query)
            logger.info(f"memory_insert_success memory_id={memory.memory_id} content={truncate_string(memory.content, 100)}")
            return memory.memory_id
        except Exception as e:
            import traceback
            logger.error(f"memory_insert_failed memory_id={memory.memory_id} content={truncate_string(memory.content, 150)} error_type={type(e).__name__} error_message={e} stack_trace={traceback.format_exc()}", exc_info=True)
            raise

    async def insert_candidate(
        self,
        memory: ExtractedMemory,
        scope: Scope,
        document_id: str,
    ) -> UUID:
        """Insert a memory candidate into the candidates table."""
        query = SQLQuery(
            sql="""
                INSERT INTO candidates (
                    id, client_id, instance_id, customer_id, user_id,
                    memory_type, content, confidence,
                    category, subcategory, source_document_id,
                    pii_obfuscated, pii_detected, created_at, document_created_at,
                    temporal_category, valid_from, valid_until,
                    temporal_confidence, is_episode, episode_metadata,
                    assertion_source,
                    event_start, event_end, event_precision
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s,
                    %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    event_start = EXCLUDED.event_start,
                    event_end = EXCLUDED.event_end,
                    event_precision = EXCLUDED.event_precision,
                    temporal_category = EXCLUDED.temporal_category,
                    valid_from = EXCLUDED.valid_from,
                    valid_until = EXCLUDED.valid_until,
                    temporal_confidence = EXCLUDED.temporal_confidence,
                    is_episode = EXCLUDED.is_episode,
                    episode_metadata = EXCLUDED.episode_metadata,
                    document_created_at = EXCLUDED.document_created_at,
                    content = EXCLUDED.content,
                    confidence = EXCLUDED.confidence,
                    assertion_source = EXCLUDED.assertion_source
            """,
            params=(
                str(memory.memory_id),
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
                str(scope.user_id) if scope.user_id else None,
                memory.memory_type.value,
                memory.content,
                memory.confidence,
                memory.category,
                memory.subcategory,
                document_id,
                memory.pii_obfuscated,
                getattr(memory, "pii_detected", None),
                memory.extracted_at,
                memory.document_created_at,
                memory.temporal_category,
                memory.valid_from,
                memory.valid_until,
                memory.temporal_confidence,
                memory.is_episode,
                json.dumps(memory.episode_metadata) if memory.episode_metadata else None,
                memory.assertion_source,
                getattr(memory, "event_start", None),
                getattr(memory, "event_end", None),
                getattr(memory, "event_precision", None),
            )
        )

        await asyncio.to_thread(self._provider.execute, query)
        return memory.memory_id

    async def insert_entity_reference(
        self,
        memory_id: UUID,
        entity_id: str,
        scope: Scope,
    ) -> None:
        """Insert a memory-entity reference for audit."""
        query = SQLQuery(
            sql="""
                INSERT INTO memory_entity_links (
                    memory_id, entity_id, client_id, instance_id, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (memory_id, entity_id) DO NOTHING
            """,
            params=(
                str(memory_id),
                entity_id,
                str(scope.client_id),
                str(scope.instance_id),
                datetime.now(timezone.utc),
            )
        )

        await asyncio.to_thread(self._provider.execute, query)

    async def insert_client_knowledge(
        self,
        memory: ExtractedMemory,
        scope: Scope,
        document_id: str,
    ) -> UUID:
        """Insert client-level knowledge."""
        query = SQLQuery(
            sql="""
                INSERT INTO client_knowledge (
                    id, client_id, content, knowledge_type,
                    source_document_id, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """,
            params=(
                str(memory.memory_id),
                str(scope.client_id),
                memory.content,
                memory.memory_type.value,
                document_id,
                memory.extracted_at,
            )
        )
        
        await asyncio.to_thread(self._provider.execute, query)
        return memory.memory_id

    async def insert_customer_knowledge(
        self,
        memory: ExtractedMemory,
        scope: Scope,
        document_id: str,
    ) -> UUID:
        """Insert customer-level knowledge."""
        query = SQLQuery(
            sql="""
                INSERT INTO customer_knowledge (
                    id, client_id, customer_id, content,
                    knowledge_type, source_document_id, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """,
            params=(
                str(memory.memory_id),
                str(scope.client_id),
                str(scope.customer_id),
                memory.content,
                memory.memory_type.value,
                document_id,
                memory.extracted_at,
            )
        )
        
        await asyncio.to_thread(self._provider.execute, query)
        return memory.memory_id

    async def execute(self, query: SQLQuery) -> None:
        """Execute a raw SQL query asynchronously."""
        await asyncio.to_thread(self._provider.execute, query)

    async def fetch_one(self, query: SQLQuery) -> Optional[dict]:
        """Fetch a single row asynchronously."""
        return await asyncio.to_thread(self._provider.fetch_one, query)

    async def fetch_all(self, query: SQLQuery) -> List[dict]:
        """Fetch all rows asynchronously."""
        return await asyncio.to_thread(self._provider.fetch_all, query)

    async def list_memories(
        self,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """List memories matching filters."""
        where_clauses = []
        params = []

        if "client_id" in filters:
            where_clauses.append("client_id = %s")
            params.append(str(filters["client_id"]))

        if "customer_id" in filters and filters["customer_id"]:
            where_clauses.append("customer_id = %s")
            params.append(str(filters["customer_id"]))

        if "user_id" in filters and filters["user_id"]:
            where_clauses.append("user_id = %s")
            params.append(str(filters["user_id"]))

        where_clauses.append("superseded_by IS NULL")
        where_sql = " AND ".join(where_clauses)

        query = SQLQuery(
            sql=f"""
                SELECT
                    id, client_id, instance_id, customer_id, user_id,
                    memory_type, content, confidence,
                    category, subcategory, source_document_id,
                    pii_obfuscated, created_at,
                    temporal_category, valid_from, valid_until,
                    temporal_confidence, is_episode, episode_metadata
                FROM memories
                WHERE {where_sql}
                ORDER BY created_at DESC
            """,
            params=tuple(params),
        )

        return await self.fetch_all(query)

    async def insert_memory_action(
        self,
        *,
        correlation_id: str,
        ingestion_id,
        scope,
        action: str,
        memory_id: str,
        content,
        previous_content,
        memory_type,
        replaced_by_memory_id=None,
        reason: Optional[str] = None,
    ) -> None:
        """No-op: the sync RDBMS path is used by consolidation/short-term,
        neither of which has an HTTP correlation id or belongs in the Requests page ledger.
        """
        return None

    async def retire_memory_with_audit(self, **kwargs) -> bool:
        """Not available on the sync provider path, and loud about it.

        `insert_memory_action` above is a deliberate no-op here, which is fine
        while the caller is consolidation or short-term: neither retires a
        memory that inline KU would later be asked to undo. It is not fine for
        a retirement, because a no-op audit means a memory that can never be
        restored, and the only way to find that out would be to need the undo.
        The one caller that reached this path (admin_api ConsolidationManager)
        is retired at construction under D-023.
        """
        raise NotImplementedError(
            "retire_memory_with_audit is not implemented on the sync RDBMS "
            "path: its audit write is a no-op, so a retirement here would be "
            "unreversible. Use SQLAlchemyRDBMSWrapper.")

    async def soft_delete_memory(self, memory_id: str, superseded_by: str) -> bool:
        """Mark a memory as superseded by a newer memory."""
        from datetime import datetime, timezone
        query = SQLQuery(
            sql="""UPDATE memories
                   SET superseded_by = %s, superseded_at = %s
                   WHERE id = %s AND superseded_by IS NULL""",
            params=(str(superseded_by), datetime.now(timezone.utc), str(memory_id))
        )
        await self.execute(query)
        return True

    async def set_deprecated_at(self, memory_id: str) -> bool:
        """No-op: the sync RDBMS path is used by consolidation/short-term,
        neither of which goes through KU's deprecate-with-link flow."""
        return False

    async def value_retired_for_user(
        self,
        scope,
        value_token: str,
        exclude_memory_ids: list,
    ) -> bool:
        """No-op (never an echo): the sync RDBMS path never runs the KU echo
        guard — mirrors set_deprecated_at above."""
        return False

    async def current_memories_matching(
        self,
        scope,
        content_tokens: list,
        exclude_memory_ids: list = None,
        limit: int = 50,
    ) -> list:
        """No-op (nothing to sweep): the sync RDBMS path never runs the KU
        retraction sweep — mirrors value_retired_for_user above."""
        return []

    async def insert_memory_link(
        self,
        *,
        from_memory_id: str,
        to_memory_id: str,
        kind: str,
        reason: Optional[str] = None,
    ) -> None:
        """No-op: lineage is only written on the HTTP-originated ingestion path."""
        return None

    async def insert_memory_source_documents(
        self,
        memory_id: str,
        source_document_ids: "list[str]",
    ) -> None:
        """No-op: source attribution is only written on the HTTP-originated ingestion path."""
        return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory by ID."""
        query = SQLQuery(
            sql="DELETE FROM memories WHERE id = %s",
            params=(str(memory_id),)
        )

        await self.execute(query)
        return True
