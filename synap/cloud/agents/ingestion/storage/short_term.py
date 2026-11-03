"""Short-term storage for session context (Vector + RDBMS only, with TTL)."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from synap.cloud.agents.ingestion.interfaces import EmbeddingClient
from synap.cloud.agents.ingestion.models import Scope
from synap.providers.rdbms.models import SQLQuery
from .rdbms_wrapper import AsyncRDBMSWrapper
from .vector_wrapper import AsyncVectorWrapper, _nested_scope_metadata


class ShortTermStorage:
    """
    Handles short-term (session-scoped) storage.

    - Vector only (no graph - too transient)
    - RDBMS for metadata
    - TTL: session duration + max 24 hours
    """

    DEFAULT_TTL = timedelta(hours=24)

    def __init__(
        self,
        rdbms: AsyncRDBMSWrapper,
        vector: AsyncVectorWrapper,
        embedding_client: EmbeddingClient,
    ):
        self._rdbms = rdbms
        self._vector = vector
        self._embedding = embedding_client

    async def persist(
        self,
        content: str,
        scope: Scope,
        session_id: str,
        qualification_result: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Persist short-term context."""
        memory_id = uuid4()
        expires_at = datetime.now(timezone.utc) + self.DEFAULT_TTL

        # RDBMS persistence
        query = SQLQuery(
            sql="""
                INSERT INTO short_term_context (
                    id, client_id, instance_id, customer_id, user_id,
                    session_id, content, qualification_result,
                    created_at, expires_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
            """,
            params=(
                str(memory_id),
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
                str(scope.user_id) if scope.user_id else None,
                session_id,
                content,
                str(qualification_result) if qualification_result else None,
                datetime.now(timezone.utc),
                expires_at,
            )
        )
        
        await self._rdbms.execute(query)

        # Vector storage (session-scoped collection)
        # Use a separate namespace for short-term storage
        # A copy of the caller's scope, and it has to be a whole copy.
        # `scope_level` decides which namespace this lands in, and `chain`
        # decides which node id its records carry (S-03). Dropping either
        # writes the data somewhere the caller did not ask for, with no error:
        # a customer-shared turn filed as one person's, and a record invisible
        # to every nested read.
        short_term_scope = Scope(
            client_id=scope.client_id,
            instance_id=scope.instance_id,
            customer_id=scope.customer_id,
            user_id=scope.user_id,
            scope_level=scope.scope_level,
            chain=scope.chain,
        )
        
        # Generate embedding
        embedding = await self._embedding.embed(content)
        
        # Store in vector with session metadata
        await self._vector.ensure_namespace(short_term_scope)
        
        from synap.providers.vector.models import VectorDocument
        
        doc = VectorDocument(
            id=str(memory_id),
            embedding=embedding,
            metadata={
                "session_id": session_id,
                "instance_id": str(scope.instance_id),
                "client_id": str(scope.client_id),
                "customer_id": str(scope.customer_id),
                "user_id": str(scope.user_id) if scope.user_id else None,
                "expires_at": expires_at.isoformat(),
                "storage_type": "short_term",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.7,
                # S-03, the same stamp long-term records carry. The scope copy
                # above already keeps the chain, and it was being kept for a
                # reader that did not exist: nothing wrote the node id, so a
                # short-term turn recorded at a team was addressed as the
                # tenant's and a rung-scoped read could not tell it apart from
                # a sibling team's. Empty for every caller with no chain, which
                # is the three-rung shape and every caller today.
                **_nested_scope_metadata(
                    short_term_scope,
                    short_term_scope.scope_level or "user",
                ),
            },
            content=content,
        )
        
        namespace = f"synap_{scope.customer_id}_st_v2"

        # Create namespace if needed with custom dimension
        try:
            await self._vector._provider.create_namespace(namespace, self._embedding.dimension)
        except Exception:
            pass  # Already exists
        
        # Upsert document
        await asyncio.to_thread(
            self._vector._provider.upsert,
            namespace,
            [doc]
        )

        return memory_id

    async def get_by_session(
        self,
        session_id: str,
        scope: Scope,
    ) -> List[Dict[str, Any]]:
        """Get all short-term context for a session."""
        query = SQLQuery(
            sql="""
                SELECT * FROM short_term_context
                WHERE session_id = %s
                  AND client_id = %s
                  AND instance_id = %s
                  AND customer_id = %s
                  AND expires_at > %s
                ORDER BY created_at
            """,
            params=(
                session_id,
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
                datetime.now(timezone.utc),
            )
        )
        
        results = await self._rdbms.fetch_all(query)
        return results

    async def cleanup_expired(self) -> int:
        """
        Cleanup expired short-term context. Run periodically.

        Returns:
            Number of records deleted
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Fetch expired IDs before deletion (for vector cleanup)
        fetch_query = SQLQuery(
            sql="SELECT id, client_id FROM short_term_context WHERE expires_at < %s",
            params=(datetime.now(timezone.utc),),
        )
        expired_rows = await self._rdbms.fetch_all(fetch_query)

        if not expired_rows:
            return 0

        # RDBMS cleanup
        delete_query = SQLQuery(
            sql="DELETE FROM short_term_context WHERE expires_at < %s",
            params=(datetime.now(timezone.utc),),
        )
        await self._rdbms.execute(delete_query)

        # Best-effort vector cleanup (grouped by client_id namespace)
        try:
            from collections import defaultdict
            by_namespace = defaultdict(list)
            for row in expired_rows:
                ns = f"synap_{row['customer_id']}_st_v2"
                by_namespace[ns].append(str(row["id"]))

            for namespace, ids in by_namespace.items():
                try:
                    await asyncio.to_thread(self._vector._provider.delete, namespace, ids)
                except Exception as e:
                    _logger.warning("Short-term vector cleanup failed for namespace %s: %s", namespace, e)
        except Exception as e:
            _logger.warning("Short-term vector cleanup failed: %s", e)

        return len(expired_rows)

    async def extend_ttl(
        self,
        session_id: str,
        scope: Scope,
        additional_hours: int = 24,
    ) -> int:
        """
        Extend TTL for all context in a session.
        
        Returns:
            Number of records updated
        """
        new_expires_at = datetime.now(timezone.utc) + timedelta(hours=additional_hours)
        
        query = SQLQuery(
            sql="""
                UPDATE short_term_context
                SET expires_at = %s
                WHERE session_id = %s
                  AND client_id = %s
                  AND instance_id = %s
                  AND customer_id = %s
                  AND expires_at > %s
            """,
            params=(
                new_expires_at,
                session_id,
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
                datetime.now(timezone.utc),
            )
        )
        
        await self._rdbms.execute(query)
        
        return 0  # Would need to track actual update count

    async def delete_session(
        self,
        session_id: str,
        scope: Scope,
    ) -> int:
        """
        Delete all context for a session (immediate cleanup).
        
        Returns:
            Number of records deleted
        """
        # Get all IDs for vector deletion
        records = await self.get_by_session(session_id, scope)
        vector_ids = [str(r["id"]) for r in records]
        
        # Delete from RDBMS
        query = SQLQuery(
            sql="""
                DELETE FROM short_term_context
                WHERE session_id = %s
                  AND client_id = %s
                  AND instance_id = %s
                  AND customer_id = %s
            """,
            params=(
                session_id,
                str(scope.client_id),
                str(scope.instance_id),
                str(scope.customer_id),
            )
        )
        
        await self._rdbms.execute(query)
        
        # Delete from short-term vector collection (NOT the main memories collection)
        if vector_ids:
            namespace = f"synap_{scope.customer_id}_st_v2"
            await asyncio.to_thread(self._vector._provider.delete, namespace, vector_ids)
        
        return len(vector_ids)
