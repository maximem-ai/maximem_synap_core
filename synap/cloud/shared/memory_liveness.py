"""One definition of "a live memory", and the loader the graph uses to ask.

A memory is alive when its ``memories`` row exists and is neither superseded,
retired (deprecated) nor forgotten. Postgres is the authority; every other
store (Chroma vectors, Neo4j ``memory_ids``) only references it.

:func:`load_supporter_rows` fetches, in ONE query (``id = ANY``), the rows a
graph record's rule (``graph_lifecycle_rule.graph_record_state``) needs, plus
the dates of the memory that superseded each one.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import (
    SupporterRow,
    is_alive,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALIVE_SQL",
    "SupporterRow",
    "is_alive",
    "load_supporter_rows",
    "SUPPORTER_ROWS_SQL",
]

# The alive predicate as SQL, for a ``memories`` row aliased ``m``.
ALIVE_SQL = "m.superseded_by IS NULL AND m.deprecated_at IS NULL AND m.deleted_at IS NULL"

SUPPORTER_ROWS_SQL = """
    SELECT m.id::text AS id,
           m.superseded_by::text AS superseded_by,
           m.superseded_at, m.deprecated_at, m.deleted_at,
           COALESCE(m.is_episode, false) AS is_episode,
           m.temporal_category, m.document_created_at, m.created_at, m.event_start,
           n.document_created_at AS n_document_created_at,
           n.created_at AS n_created_at,
           n.event_start AS n_event_start
    FROM memories m
    LEFT JOIN memories n ON n.id = m.superseded_by
    WHERE m.id = ANY(CAST(:ids AS uuid[]))
"""


def _valid_uuid_strings(ids: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in ids:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s in seen:
            continue
        try:
            uuid.UUID(s)
        except (ValueError, AttributeError, TypeError):
            # Not a memory id: no memories row can match, so it reads as absent.
            continue
        seen.add(s)
        out.append(s)
    return out


def _row_to_supporter(m: Any) -> SupporterRow:
    get = m.get if hasattr(m, "get") else (lambda k: getattr(m, k))
    return SupporterRow(
        id=str(get("id")),
        superseded_by=get("superseded_by"),
        superseded_at=get("superseded_at"),
        deprecated_at=get("deprecated_at"),
        deleted_at=get("deleted_at"),
        is_episode=bool(get("is_episode")),
        temporal_category=get("temporal_category"),
        document_created_at=get("document_created_at"),
        created_at=get("created_at"),
        event_start=get("event_start"),
        superseder_document_created_at=get("n_document_created_at"),
        superseder_created_at=get("n_created_at"),
        superseder_event_start=get("n_event_start"),
    )


async def load_supporter_rows(
    memory_ids: Iterable[Any],
    session: Optional[Any] = None,
) -> Dict[str, SupporterRow]:
    """Load the Postgres state of ``memory_ids`` in one query.

    Returns ``{memory_id: SupporterRow}`` for every id that has a row. An id
    absent from the result was hard-deleted (or is not a memory id at all).

    ``session``: a SQLAlchemy ``AsyncSession``; when omitted one is opened from
    ``synap.cloud.database.connection.get_session_factory()``.

    Raises whatever the database raises. Callers on the read path must treat a
    failure as "liveness unknown" (do not serve unchecked graph statements).
    """
    ids = _valid_uuid_strings(memory_ids)
    if not ids:
        return {}

    from sqlalchemy import text

    async def _run(s: Any) -> Dict[str, SupporterRow]:
        result = await s.execute(text(SUPPORTER_ROWS_SQL), {"ids": ids})
        out: Dict[str, SupporterRow] = {}
        for m in result.mappings().all():
            row = _row_to_supporter(m)
            out[row.id] = row
        return out

    if session is not None:
        return await _run(session)

    from synap.cloud.database.connection import get_session_factory

    async with get_session_factory()() as s:
        return await _run(s)
