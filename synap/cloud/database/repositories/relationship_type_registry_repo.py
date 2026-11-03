"""Repository for the per-instance relationship type registry.

Uses raw SQL via AsyncSession.execute() because the table includes a
pgvector column (embedding vector(1024)) which doesn't have SQLAlchemy ORM
integration in this codebase. Follows the same raw-SQL pattern used by
the entity resolution tables (migration 010).
"""

import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class RelationshipTypeRegistryRepository:
    """CRUD + semantic lookup for the relationship_type_registry table.

    Each instance has its own set of relationship types with embeddings.
    Types are registered during ingestion (source='ingestion') or seeded
    from MACA config (source='maca').

    Retrieval uses the embeddings for semantic similarity expansion:
    given an inferred type, find stored types with cosine similarity > threshold.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(
        self,
        instance_id: str,
        relationship_type: str,
        embedding: List[float],
        source: str = "ingestion",
    ) -> None:
        """Insert a new type or increment occurrence_count if it already exists.

        Args:
            instance_id: The instance this type belongs to.
            relationship_type: UPPERCASE type name (e.g., "DIAGNOSED_WITH").
            embedding: Vector embedding of the type name (1024 dimensions).
            source: How this type was introduced ('ingestion', 'maca', 'manual').
        """
        normalized_type = relationship_type.strip().upper()
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        await self.session.execute(
            text("""
                INSERT INTO relationship_type_registry
                    (instance_id, relationship_type, embedding, source)
                VALUES
                    (:instance_id, :relationship_type, CAST(:embedding AS vector), :source)
                -- ⚠ CAST(...), never `:embedding::vector`. SQLAlchemy's text()
                -- does not recognise a bind parameter immediately followed by
                -- `::`, so it left `:embedding` in the SQL while binding the
                -- other three, and asyncpg received a mixed statement:
                --     VALUES ($1, $2, :embedding::vector, $3)
                -- Postgres then fails with `syntax error at or near ":"`, so
                -- every relationship type was registered with NO embedding and
                -- the semantic dedup of type names quietly had nothing to work
                -- with. The two `:query_embedding::vector` uses below had the
                -- identical shape and would have failed the same way.
                ON CONFLICT (instance_id, relationship_type)
                DO UPDATE SET
                    occurrence_count = relationship_type_registry.occurrence_count + 1
            """),
            {
                "instance_id": str(instance_id),
                "relationship_type": normalized_type,
                "embedding": embedding_str,
                "source": source,
            },
        )

    async def upsert_batch(
        self,
        instance_id: str,
        types_with_embeddings: List[Tuple[str, List[float]]],
        source: str = "ingestion",
    ) -> None:
        """Batch upsert multiple relationship types.

        Args:
            instance_id: The instance this batch belongs to.
            types_with_embeddings: List of (relationship_type, embedding) tuples.
            source: How these types were introduced.
        """
        for rel_type, embedding in types_with_embeddings:
            await self.upsert(instance_id, rel_type, embedding, source)

    async def get_all_for_instance(
        self,
        instance_id: str,
    ) -> List[Dict]:
        """Get all registered types for an instance.

        Returns:
            List of dicts with: relationship_type, embedding, occurrence_count, source.
        """
        result = await self.session.execute(
            text("""
                SELECT relationship_type, embedding::text, occurrence_count, source
                FROM relationship_type_registry
                WHERE instance_id = :instance_id
                ORDER BY occurrence_count DESC
            """),
            {"instance_id": str(instance_id)},
        )
        rows = result.fetchall()
        return [
            {
                "relationship_type": row[0],
                "embedding": _parse_vector(row[1]),
                "occurrence_count": row[2],
                "source": row[3],
            }
            for row in rows
        ]

    async def find_similar(
        self,
        instance_id: str,
        query_embedding: List[float],
        threshold: float = 0.8,
        limit: int = 5,
    ) -> List[Dict]:
        """Find stored types semantically similar to a query embedding.

        Uses pgvector cosine distance: 1 - (embedding <=> query) = cosine_similarity.

        Args:
            instance_id: The instance to search within.
            query_embedding: Embedding of the inferred type name.
            threshold: Minimum cosine similarity (0.0-1.0).
            limit: Maximum results to return.

        Returns:
            List of dicts with: relationship_type, similarity, occurrence_count.
        """
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        result = await self.session.execute(
            text("""
                SELECT
                    relationship_type,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                    occurrence_count
                FROM relationship_type_registry
                WHERE instance_id = :instance_id
                  AND 1 - (embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """),
            {
                "instance_id": str(instance_id),
                "query_embedding": embedding_str,
                "threshold": threshold,
                "limit": limit,
            },
        )
        rows = result.fetchall()
        return [
            {
                "relationship_type": row[0],
                "similarity": float(row[1]),
                "occurrence_count": row[2],
            }
            for row in rows
        ]

    async def get_type_names_for_instance(
        self,
        instance_id: str,
    ) -> List[str]:
        """Get just the type names for an instance (for cache population).

        Returns:
            List of UPPERCASE relationship type strings.
        """
        result = await self.session.execute(
            text("""
                SELECT relationship_type
                FROM relationship_type_registry
                WHERE instance_id = :instance_id
                ORDER BY occurrence_count DESC
            """),
            {"instance_id": str(instance_id)},
        )
        return [row[0] for row in result.fetchall()]

    async def delete_for_instance(self, instance_id: str) -> int:
        """Delete all registry entries for an instance (used during instance deletion).

        Returns:
            Number of rows deleted.
        """
        result = await self.session.execute(
            text("""
                DELETE FROM relationship_type_registry
                WHERE instance_id = :instance_id
            """),
            {"instance_id": str(instance_id)},
        )
        return result.rowcount


def _parse_vector(vector_str: str) -> List[float]:
    """Parse pgvector text representation '[0.1,0.2,...]' into a list of floats."""
    cleaned = vector_str.strip("[]")
    if not cleaned:
        return []
    return [float(v) for v in cleaned.split(",")]
