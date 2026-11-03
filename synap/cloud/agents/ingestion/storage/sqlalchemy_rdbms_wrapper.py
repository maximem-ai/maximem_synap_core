"""RDBMS wrapper using the app's existing SQLAlchemy async sessions.

Replaces the psycopg2-based AsyncRDBMSWrapper to avoid creating a
separate connection pool with potentially different credentials.

Used by the ingestion pipeline (StorageCoordinator) and retrieval layer.
"""

import json as _json
from datetime import datetime, timezone
from typing import Optional


class SQLAlchemyRDBMSWrapper:
    """RDBMS storage using SQLAlchemy async sessions.

    Uses :name parameter style with sqlalchemy.text() (asyncpg requirement).
    Sessions are obtained lazily from the app's session factory.
    """

    async def _execute(self, sql: str, params: dict):
        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.shared import event_window_schema as ews
        from sqlalchemy import text
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                await session.execute(text(ews.adapt(sql)), params)
                await session.commit()
        except Exception as exc:
            # A database without migration 154: store the row without its
            # event window rather than lose it. See event_window_schema.
            if not (ews.available() and ews.is_missing_column_error(exc)):
                raise
            ews.mark_missing(exc)
            async with session_factory() as session:
                await session.execute(text(ews.adapt(sql)), params)
                await session.commit()

    async def fetch_all(self, sql: str, params: dict) -> list:
        """Execute a SELECT query and return all rows as dicts.

        SQL must use :name params with text() (asyncpg requirement).
        """
        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.shared import event_window_schema as ews
        from sqlalchemy import text
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                result = await session.execute(text(ews.adapt(sql)), params)
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            if not (ews.available() and ews.is_missing_column_error(exc)):
                raise
            ews.mark_missing(exc)
            async with session_factory() as session:
                result = await session.execute(text(ews.adapt(sql)), params)
                return [dict(row._mapping) for row in result.fetchall()]

    # ── subject identity (migration 159) ────────────────────────────────────

    async def subjects_of(self, memory_ids: list) -> dict:
        """``{memory_id: (subject_id, subject_label)}`` for the ids that have one.

        Ids with no subject are absent rather than mapped to None, so a caller
        can tell "not grouped yet" from "grouped and we read it".
        """
        if not memory_ids:
            return {}
        rows = await self.fetch_all(
            """
            SELECT id::text AS id, subject_id::text AS subject_id, subject_label
            FROM memories
            WHERE id = ANY(CAST(:ids AS uuid[])) AND subject_id IS NOT NULL
            """,
            {"ids": [str(m) for m in memory_ids]},
        )
        return {r["id"]: (r["subject_id"], r["subject_label"]) for r in rows}

    async def assign_subject(self, memory_ids: list, subject_id: str,
                             subject_label: str) -> None:
        """Put these memories in one subject.

        Writes the label unconditionally for every id named, including ones that
        already carry the subject: a later member can state the thing more
        specifically than the first did, and the label is meant to be the most
        specific wording anyone used. Idempotent, so a retried ingest converges
        instead of accumulating.
        """
        if not memory_ids:
            return
        await self._execute(
            """
            UPDATE memories
               SET subject_id = CAST(:sid AS uuid), subject_label = :label
             WHERE id = ANY(CAST(:ids AS uuid[]))
            """,
            {"ids": [str(m) for m in memory_ids], "sid": str(subject_id),
             "label": subject_label},
        )

    async def subjects_in_scope(self, scope) -> dict:
        """``{subject_id: [member content, ...]}`` for one scope.

        Only rows that actually reach a reader: the same filter serving uses,
        `deleted_at IS NULL AND superseded_by IS NULL`. The first version
        filtered `deleted_at` alone, which let a RETIRED row stand in a group and
        speak for it.
        """
        rows = await self.fetch_all(
            """
            SELECT subject_id::text AS subject_id, content
            FROM memories
            WHERE user_id = CAST(:uid AS uuid)
              AND subject_id IS NOT NULL
              AND deleted_at IS NULL AND superseded_by IS NULL
            ORDER BY subject_id, created_at
            """,
            {"uid": str(getattr(scope, "user_id", None) or "")},
        )
        out: dict = {}
        for r in rows:
            out.setdefault(r["subject_id"], []).append(r["content"] or "")
        return out

    async def merge_subjects(self, from_subject: str, into_subject: str) -> int:
        """Repoint every member of one subject onto another. Returns rows moved.

        The convergence step. Two subjects for one thing form whenever two
        memories about it match different ungrouped partners, and nothing else
        in the pipeline ever joins them.
        """
        if not from_subject or not into_subject or from_subject == into_subject:
            return 0
        rows = await self.fetch_all(
            """
            UPDATE memories SET subject_id = CAST(:into AS uuid)
             WHERE subject_id = CAST(:frm AS uuid)
            RETURNING id::text AS id
            """,
            {"frm": str(from_subject), "into": str(into_subject)},
        )
        return len(rows)

    async def value_retired_for_user(
        self,
        scope,
        value_token: str,
        exclude_memory_ids: list,
    ) -> bool:
        """Echo resurrection guard lookup: True iff ``value_token`` appears in
        a RETIRED memory (superseded or deprecated) for this user AND in no
        CURRENT memory outside the just-persisted batch
        (``exclude_memory_ids``). A value in that state was retired by the
        user; an assistant re-asserting it is echoing stale context."""
        if not getattr(scope, "user_id", None):
            return False
        escaped = (
            value_token.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        rows = await self.fetch_all(
            r"""SELECT
                   bool_or(superseded_by IS NOT NULL
                           OR deprecated_at IS NOT NULL
                           OR deleted_at IS NOT NULL) AS retired,
                   bool_or(superseded_by IS NULL AND deprecated_at IS NULL
                           AND deleted_at IS NULL
                           AND NOT (id::text = ANY(:exclude))) AS live
               FROM memories
               WHERE user_id = CAST(:user_id AS uuid)
                 AND client_id = CAST(:client_id AS uuid)
                 AND content ILIKE :pattern ESCAPE '\'""",
            {
                "user_id": str(scope.user_id),
                "client_id": str(scope.client_id),
                "pattern": f"%{escaped}%",
                "exclude": list(exclude_memory_ids or []),
            },
        )
        if not rows:
            return False
        return bool(rows[0].get("retired")) and not bool(rows[0].get("live"))

    async def current_memories_matching(
        self,
        scope,
        content_tokens: list,
        exclude_memory_ids: list = None,
        limit: int = 50,
    ) -> list:
        """Retraction subject sweep lookup: CURRENT memories of this user
        whose content contains ALL of ``content_tokens`` (case-insensitive
        substring each). Used by execute_retraction_sweeps to retire every
        copy of a retracted fact, not just the one vector-paired row.
        Returns [{id, content, memory_type}, ...]."""
        if not getattr(scope, "user_id", None) or not content_tokens:
            return []
        params = {
            "user_id": str(scope.user_id),
            "client_id": str(scope.client_id),
            "exclude": [str(x) for x in (exclude_memory_ids or [])],
            "limit": int(limit),
        }
        clauses = []
        for i, token in enumerate(content_tokens):
            escaped = (
                str(token).replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            params[f"pat{i}"] = f"%{escaped}%"
            clauses.append(f"AND content ILIKE :pat{i} ESCAPE '\\'")
        rows = await self.fetch_all(
            "SELECT id, content, memory_type FROM memories"
            " WHERE user_id = CAST(:user_id AS uuid)"
            "   AND client_id = CAST(:client_id AS uuid)"
            "   AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL"
            "   AND NOT (id::text = ANY(:exclude)) "
            + " ".join(clauses)
            + " ORDER BY created_at LIMIT :limit",
            params,
        )
        return [
            {
                "id": str(r["id"]),
                "content": r.get("content") or "",
                "memory_type": r.get("memory_type"),
            }
            for r in rows
        ]

    async def current_memories_matching_any(
        self,
        scope,
        content_keywords: list,
        exclude_memory_ids: list = None,
        limit: int = 200,
    ) -> list:
        """Retraction sweep v2 (SYNAP_KU_RETRACTION_SWEEP_V2) class prefilter:
        CURRENT memories of this user whose content contains ANY of
        ``content_keywords`` (case-insensitive substring, OR-joined). This is a
        deliberately loose RECALL net — the caller re-confirms each row against
        the real attribute-class regexes before retiring it — so it is bounded
        by ``limit`` (200) rather than trusted on its own.
        Returns [{id, content, memory_type}, ...]."""
        if not getattr(scope, "user_id", None) or not content_keywords:
            return []
        params = {
            "user_id": str(scope.user_id),
            "client_id": str(scope.client_id),
            "exclude": [str(x) for x in (exclude_memory_ids or [])],
            "limit": int(limit),
        }
        clauses = []
        for i, keyword in enumerate(content_keywords):
            escaped = (
                str(keyword).replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            params[f"kw{i}"] = f"%{escaped}%"
            clauses.append(f"content ILIKE :kw{i} ESCAPE '\\'")
        rows = await self.fetch_all(
            "SELECT id, content, memory_type FROM memories"
            " WHERE user_id = CAST(:user_id AS uuid)"
            "   AND client_id = CAST(:client_id AS uuid)"
            "   AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL"
            "   AND NOT (id::text = ANY(:exclude))"
            "   AND (" + " OR ".join(clauses) + ")"
            " ORDER BY created_at LIMIT :limit",
            params,
        )
        return [
            {
                "id": str(r["id"]),
                "content": r.get("content") or "",
                "memory_type": r.get("memory_type"),
            }
            for r in rows
        ]

    async def retired_memories_with_value(
        self,
        scope,
        value_token: str,
        exclude_memory_ids: list = None,
        limit: int = 1,
    ) -> list:
        """Retired-value guard lookup (SYNAP_KU_RETIRED_VALUE_GUARD): RETIRED
        memories (superseded OR deprecated) of this user whose content contains
        ``value_token``.

        Unlike ``value_retired_for_user`` this does NOT require the value to be
        absent from current rows: the backwards-supersede case it defends
        against always has a live row carrying the value (the just-persisted
        new memory itself, plus any paraphrase siblings), so the
        "retired and nowhere current" test would never fire.
        Returns [{id, content, memory_type}, ...] (newest retirement first)."""
        if not getattr(scope, "user_id", None) or not value_token:
            return []
        escaped = (
            str(value_token).replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        rows = await self.fetch_all(
            r"""SELECT id, content, memory_type FROM memories
                 WHERE user_id = CAST(:user_id AS uuid)
                   AND client_id = CAST(:client_id AS uuid)
                   AND (superseded_by IS NOT NULL OR deprecated_at IS NOT NULL)
                   AND NOT (id::text = ANY(:exclude))
                   AND content ILIKE :pattern ESCAPE '\'
                 ORDER BY created_at DESC LIMIT :limit""",
            {
                "user_id": str(scope.user_id),
                "client_id": str(scope.client_id),
                "exclude": [str(x) for x in (exclude_memory_ids or [])],
                "pattern": f"%{escaped}%",
                "limit": int(limit),
            },
        )
        return [
            {
                "id": str(r["id"]),
                "content": r.get("content") or "",
                "memory_type": r.get("memory_type"),
            }
            for r in rows
        ]

    async def insert_memory(self, memory, scope, document_id: str):
        await self._execute(
            """INSERT INTO memories (
                id, client_id, instance_id, customer_id, user_id,
                memory_type, content, confidence,
                category, subcategory, source_document_id,
                pii_obfuscated, pii_detected, created_at,
                temporal_category, valid_from, valid_until,
                temporal_confidence, is_episode, episode_metadata,
                document_created_at, assertion_source,
                event_start, event_end, event_precision, event_anchor
            ) VALUES (
                CAST(:id AS uuid), :client_id, :instance_id, :customer_id, :user_id,
                :memory_type, :content, :confidence,
                :category, :subcategory, :source_document_id,
                :pii_obfuscated, :pii_detected, :created_at,
                :temporal_category, :valid_from, :valid_until,
                :temporal_confidence, :is_episode, :episode_metadata,
                :document_created_at, :assertion_source,
                :event_start, :event_end, :event_precision, CAST(:event_anchor AS JSONB)
            ) ON CONFLICT (id) DO UPDATE SET
                event_start = EXCLUDED.event_start,
                event_end = EXCLUDED.event_end,
                event_precision = EXCLUDED.event_precision,
                event_anchor = EXCLUDED.event_anchor,
                temporal_category = EXCLUDED.temporal_category,
                valid_from = EXCLUDED.valid_from,
                valid_until = EXCLUDED.valid_until,
                temporal_confidence = EXCLUDED.temporal_confidence,
                is_episode = EXCLUDED.is_episode,
                episode_metadata = EXCLUDED.episode_metadata,
                document_created_at = EXCLUDED.document_created_at,
                content = EXCLUDED.content,
                confidence = EXCLUDED.confidence,
                assertion_source = EXCLUDED.assertion_source""",
            {
                "id": str(memory.memory_id),
                "client_id": str(scope.client_id),
                "instance_id": str(scope.instance_id),
                "customer_id": str(scope.customer_id),
                "user_id": str(scope.user_id) if scope.user_id else None,
                "memory_type": memory.memory_type.value,
                "content": memory.content,
                "confidence": memory.confidence,
                "category": memory.category,
                "subcategory": memory.subcategory,
                "source_document_id": document_id,
                "pii_obfuscated": memory.pii_obfuscated,
                "pii_detected": getattr(memory, "pii_detected", None),
                "created_at": memory.extracted_at,
                "temporal_category": getattr(memory, 'temporal_category', 'perpetual'),
                "valid_from": getattr(memory, 'valid_from', None),
                "valid_until": getattr(memory, 'valid_until', None),
                "temporal_confidence": getattr(memory, 'temporal_confidence', 0.0),
                "is_episode": getattr(memory, 'is_episode', False),
                "episode_metadata": _json.dumps(memory.episode_metadata) if getattr(memory, 'episode_metadata', None) else None,
                "document_created_at": getattr(memory, 'document_created_at', None),
                "assertion_source": getattr(memory, 'assertion_source', None),
                "event_start": getattr(memory, 'event_start', None),
                "event_end": getattr(memory, 'event_end', None),
                "event_precision": getattr(memory, 'event_precision', None),
                # The relationship a still-unresolved date hangs off, so it can
                # be settled when the anchor event's own date arrives (mig 155).
                "event_anchor": (_json.dumps(memory.event_anchor)
                                 if getattr(memory, 'event_anchor', None) else None),
            },
        )
        return memory.memory_id

    async def insert_candidate(self, memory, scope, document_id: str):
        await self._execute(
            """INSERT INTO candidates (
                id, client_id, instance_id, customer_id, user_id,
                memory_type, content, confidence,
                category, subcategory, source_document_id,
                pii_obfuscated, pii_detected, created_at,
                temporal_category, valid_from, valid_until,
                temporal_confidence, is_episode, episode_metadata,
                document_created_at, assertion_source,
                event_start, event_end, event_precision, event_anchor
            ) VALUES (
                CAST(:id AS uuid), :client_id, :instance_id, :customer_id, :user_id,
                :memory_type, :content, :confidence,
                :category, :subcategory, :source_document_id,
                :pii_obfuscated, :pii_detected, :created_at,
                :temporal_category, :valid_from, :valid_until,
                :temporal_confidence, :is_episode, :episode_metadata,
                :document_created_at, :assertion_source,
                :event_start, :event_end, :event_precision, CAST(:event_anchor AS JSONB)
            ) ON CONFLICT (id) DO UPDATE SET
                event_start = EXCLUDED.event_start,
                event_end = EXCLUDED.event_end,
                event_precision = EXCLUDED.event_precision,
                event_anchor = EXCLUDED.event_anchor,
                temporal_category = EXCLUDED.temporal_category,
                valid_from = EXCLUDED.valid_from,
                valid_until = EXCLUDED.valid_until,
                temporal_confidence = EXCLUDED.temporal_confidence,
                is_episode = EXCLUDED.is_episode,
                episode_metadata = EXCLUDED.episode_metadata,
                document_created_at = EXCLUDED.document_created_at,
                content = EXCLUDED.content,
                confidence = EXCLUDED.confidence,
                assertion_source = EXCLUDED.assertion_source""",
            {
                "id": str(memory.memory_id),
                "client_id": str(scope.client_id),
                "instance_id": str(scope.instance_id),
                "customer_id": str(scope.customer_id),
                "user_id": str(scope.user_id) if scope.user_id else None,
                "memory_type": memory.memory_type.value,
                "content": memory.content,
                "confidence": memory.confidence,
                "category": memory.category,
                "subcategory": memory.subcategory,
                "source_document_id": document_id,
                "pii_obfuscated": memory.pii_obfuscated,
                "pii_detected": getattr(memory, "pii_detected", None),
                "created_at": memory.extracted_at,
                "temporal_category": getattr(memory, 'temporal_category', 'perpetual'),
                "valid_from": getattr(memory, 'valid_from', None),
                "valid_until": getattr(memory, 'valid_until', None),
                "temporal_confidence": getattr(memory, 'temporal_confidence', 0.0),
                "is_episode": getattr(memory, 'is_episode', False),
                "episode_metadata": _json.dumps(memory.episode_metadata) if getattr(memory, 'episode_metadata', None) else None,
                "document_created_at": getattr(memory, 'document_created_at', None),
                "assertion_source": getattr(memory, 'assertion_source', None),
                "event_start": getattr(memory, 'event_start', None),
                "event_end": getattr(memory, 'event_end', None),
                "event_precision": getattr(memory, 'event_precision', None),
                # The relationship a still-unresolved date hangs off, so it can
                # be settled when the anchor event's own date arrives (mig 155).
                "event_anchor": (_json.dumps(memory.event_anchor)
                                 if getattr(memory, 'event_anchor', None) else None),
            },
        )
        return memory.memory_id

    async def insert_entity_reference(self, memory_id, entity_id: str, scope):
        await self._execute(
            """INSERT INTO memory_entity_links (
                memory_id, entity_id, client_id, instance_id, created_at
            ) VALUES (
                CAST(:memory_id AS uuid), :entity_id, :client_id, :instance_id, :created_at
            ) ON CONFLICT (memory_id, entity_id) DO NOTHING""",
            {
                "memory_id": str(memory_id),
                "entity_id": entity_id,
                "client_id": str(scope.client_id),
                "instance_id": str(scope.instance_id),
                "created_at": datetime.now(timezone.utc),
            },
        )

    async def insert_client_knowledge(self, memory, scope, document_id: str):
        await self._execute(
            """INSERT INTO client_knowledge (
                id, client_id, content, knowledge_type,
                source_document_id, created_at
            ) VALUES (
                CAST(:id AS uuid), :client_id, :content, :knowledge_type,
                :source_document_id, :created_at
            ) ON CONFLICT (id) DO NOTHING""",
            {
                "id": str(memory.memory_id),
                "client_id": str(scope.client_id),
                "content": memory.content,
                "knowledge_type": memory.memory_type.value,
                "source_document_id": document_id,
                "created_at": memory.extracted_at,
            },
        )
        return memory.memory_id

    async def soft_delete_memory(self, memory_id: str, superseded_by: str) -> bool:
        """Mark a memory as superseded by a newer memory.

        Sets superseded_by and superseded_at. Does NOT physically delete the row.
        Returns True if the row was updated, False if memory_id not found or already superseded.
        """
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("""UPDATE memories
                        SET superseded_by = CAST(:superseded_by AS uuid),
                            superseded_at = :superseded_at
                        WHERE id = CAST(:id AS uuid)
                          AND superseded_by IS NULL"""),
                {
                    "id": str(memory_id),
                    "superseded_by": str(superseded_by),
                    "superseded_at": datetime.now(timezone.utc),
                },
            )
            await session.commit()
            return result.rowcount > 0

    async def set_deprecated_at(self, memory_id: str) -> bool:
        """KU v2 lifecycle marker: stamp `deprecated_at = NOW()` on an active memory.

        Idempotent on the active condition: returns True only if a currently-active
        row was just deprecated. Returns False if the memory is already deprecated
        or doesn't exist.

        This is the KU v2 replacement for `soft_delete_memory`. Both are written
        during the transition (legacy `superseded_by` for backward compat, new
        `deprecated_at` for the lifecycle-aware retrieval that lands later).
        """
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("""UPDATE memories
                        SET deprecated_at = :now
                        WHERE id = CAST(:id AS uuid)
                          AND deprecated_at IS NULL"""),
                {
                    "id": str(memory_id),
                    "now": datetime.now(timezone.utc),
                },
            )
            await session.commit()
            return result.rowcount > 0

    async def insert_memory_link(
        self,
        *,
        from_memory_id: str,
        to_memory_id: str,
        kind: str,
        reason: Optional[str] = None,
    ) -> None:
        """Insert one row in `memory_links`.

        Idempotent: ON CONFLICT (from_memory_id, to_memory_id, kind) DO NOTHING.
        `kind` must be one of DEPRECATES / ENRICHED / DISAMBIGUATES / CANCELS
        (DB CHECK constraint enforces this).
        """
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text
        import uuid as _uuid
        import logging as _logging
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                await session.execute(
                    text("""INSERT INTO memory_links (
                                id, from_memory_id, to_memory_id, kind, reason, created_at
                            ) VALUES (
                                CAST(:id AS uuid),
                                CAST(:from_memory_id AS uuid),
                                CAST(:to_memory_id AS uuid),
                                :kind, :reason, :created_at
                            )
                            ON CONFLICT (from_memory_id, to_memory_id, kind) DO NOTHING"""),
                    {
                        "id": str(_uuid.uuid4()),
                        "from_memory_id": str(from_memory_id),
                        "to_memory_id": str(to_memory_id),
                        "kind": kind,
                        "reason": reason,
                        "created_at": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as exc:
            # Lineage is metadata; never let a broken link write block the
            # surrounding memory operation. Match insert_memory_action's
            # warn-and-swallow stance.
            _logging.getLogger(__name__).warning(
                "memory_link_insert_failed from=%s to=%s kind=%s error=%s",
                from_memory_id, to_memory_id, kind, exc,
            )

    async def insert_memory_source_documents(
        self,
        memory_id: str,
        source_document_ids: "list[str]",
    ) -> None:
        """Insert M:N rows in `memory_source_documents` for one memory.

        Idempotent on (memory_id, source_document_id). For new memories during
        ingestion, called once with one source. For merged memories,
        called once with the union of all contributors' sources.
        """
        if not source_document_ids:
            return
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text
        import logging as _logging
        session_factory = get_session_factory()
        try:
            async with session_factory() as session:
                # One INSERT per row; tiny rowcount, simple correctness.
                # If we see this become hot, batch-insert with VALUES (...), (...)
                for source_id in source_document_ids:
                    if not source_id:
                        continue
                    await session.execute(
                        text("""INSERT INTO memory_source_documents
                                    (memory_id, source_document_id, contributed_at)
                                VALUES
                                    (CAST(:memory_id AS uuid), :source_document_id, :contributed_at)
                                ON CONFLICT (memory_id, source_document_id) DO NOTHING"""),
                        {
                            "memory_id": str(memory_id),
                            "source_document_id": source_id,
                            "contributed_at": datetime.now(timezone.utc),
                        },
                    )
                await session.commit()
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                "memory_source_documents_insert_failed memory=%s sources=%s error=%s",
                memory_id, source_document_ids, exc,
            )

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
        """Write one row to the memory_actions changelog.

        Failures are logged but do not raise: a broken audit row must not
        take down an otherwise successful memory write.
        """
        import uuid as _uuid
        import logging as _logging
        try:
            await self._execute(
                """INSERT INTO memory_actions (
                    id, correlation_id, ingestion_id,
                    client_id, instance_id, customer_id, user_id,
                    action, memory_id, replaced_by_memory_id,
                    previous_content, content, memory_type, reason, created_at
                ) VALUES (
                    CAST(:id AS uuid), :correlation_id,
                    CAST(:ingestion_id AS uuid),
                    :client_id, :instance_id, :customer_id, :user_id,
                    :action, CAST(:memory_id AS uuid),
                    CAST(:replaced_by_memory_id AS uuid),
                    :previous_content, :content, :memory_type, :reason, :created_at
                )""",
                {
                    "id": str(_uuid.uuid4()),
                    "correlation_id": correlation_id,
                    "ingestion_id": str(ingestion_id) if ingestion_id else None,
                    "client_id": str(scope.client_id),
                    "instance_id": str(scope.instance_id),
                    "customer_id": str(scope.customer_id),
                    "user_id": str(scope.user_id) if scope.user_id else None,
                    "action": action,
                    "memory_id": str(memory_id),
                    "replaced_by_memory_id": str(replaced_by_memory_id) if replaced_by_memory_id else None,
                    "previous_content": previous_content,
                    "content": content,
                    "memory_type": memory_type,
                    "reason": reason,
                    "created_at": datetime.now(timezone.utc),
                },
            )
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                "memory_action_insert_failed action=%s memory_id=%s correlation_id=%s error=%s",
                action, memory_id, correlation_id, exc,
            )

    # ── Retiring a memory and recording why, together or not at all ─────────
    #
    # Inline KU retires memories all day: a new fact supersedes an old one, a
    # restatement is dropped in favour of the row that already said it, a merge
    # absorbs its contributors. Meditation can undo any of those, but only from
    # the trail row, because the trail row is the only place the memory's
    # before-state survives. No trail row, no undo, ever.
    #
    # Written separately, that guarantee did not hold. The retire committed in
    # its own transaction and the trail row ran in a second one, so a failure
    # in between left a memory retired with nothing to restore it from. The
    # trail write also sat behind `if http_correlation_id:`, because
    # memory_actions.correlation_id is NOT NULL and an ingest without a request
    # id would have violated it — so the audit was skipped exactly where the
    # caller was least ordinary.
    #
    # This does the whole retirement in one transaction: the flags, the
    # lineage, and the trail row that makes it reversible. Either a memory is
    # retired AND undoable, or nothing happened. The audit is not optional and
    # a missing correlation id is not a reason to skip it — one is synthesised,
    # prefixed so it is always obvious which ids were minted here.
    _SYNTHETIC_CORRELATION_PREFIX = "inline-ku:"

    async def retire_memory_with_audit(
        self,
        *,
        memory_id: str,
        action: str,
        scope,
        previous_content,
        content=None,
        memory_type=None,
        superseded_by: Optional[str] = None,
        replaced_by_memory_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        ingestion_id=None,
        reason: Optional[str] = None,
        deprecate_with_link: bool = False,
        link_kind: str = "DEPRECATES",
        link_to_memory_id: Optional[str] = None,
        require_retire: bool = True,
    ) -> bool:
        """Retire one memory and write its trail row in a single transaction.

        ``superseded_by`` set means the legacy supersession column is stamped
        and the row must currently be un-superseded; ``None`` means this is a
        deprecation with no replacement (an explicit forget), and the row must
        currently be un-deprecated.

        Returns False without writing anything when the row is already retired
        — the caller's ``continue``. Returns True when the memory is retired
        and its trail row is committed with it.

        Raises if the trail row cannot be written. That is deliberate: a
        retirement nobody can undo is worse than an ingest that fails loudly,
        and the caller already treats KU failures as non-fatal.
        """
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text
        import uuid as _uuid

        now = datetime.now(timezone.utc)
        corr = correlation_id or (
            f"{self._SYNTHETIC_CORRELATION_PREFIX}"
            f"{ingestion_id or _uuid.uuid4()}"
        )

        session_factory = get_session_factory()
        async with session_factory() as session:
            if superseded_by is not None:
                retired = await session.execute(
                    text("""UPDATE memories
                            SET superseded_by = CAST(:superseded_by AS uuid),
                                superseded_at = :now
                            WHERE id = CAST(:id AS uuid)
                              AND superseded_by IS NULL"""),
                    {"id": str(memory_id),
                     "superseded_by": str(superseded_by),
                     "now": now},
                )
            else:
                retired = await session.execute(
                    text("""UPDATE memories
                            SET deprecated_at = :now
                            WHERE id = CAST(:id AS uuid)
                              AND deprecated_at IS NULL"""),
                    {"id": str(memory_id), "now": now},
                )

            if require_retire and retired.rowcount == 0:
                # Already retired by someone else. Nothing happened, so there
                # is nothing to record and nothing to roll back.
                await session.rollback()
                return False

            if deprecate_with_link and superseded_by is not None:
                await session.execute(
                    text("""UPDATE memories SET deprecated_at = :now
                            WHERE id = CAST(:id AS uuid)
                              AND deprecated_at IS NULL"""),
                    {"id": str(memory_id), "now": now},
                )
                await session.execute(
                    text("""INSERT INTO memory_links (
                                id, from_memory_id, to_memory_id, kind, reason, created_at
                            ) VALUES (
                                CAST(:id AS uuid), CAST(:frm AS uuid),
                                CAST(:to AS uuid), :kind, :reason, :created_at
                            )
                            ON CONFLICT (from_memory_id, to_memory_id, kind)
                            DO NOTHING"""),
                    {"id": str(_uuid.uuid4()),
                     "frm": str(memory_id),
                     "to": str(link_to_memory_id or superseded_by),
                     "kind": link_kind,
                     "reason": reason,
                     "created_at": now},
                )

            # The undo fuel. In the same transaction as the retirement above,
            # so the two cannot disagree.
            await session.execute(
                text("""INSERT INTO memory_actions (
                            id, correlation_id, ingestion_id,
                            client_id, instance_id, customer_id, user_id,
                            action, memory_id, replaced_by_memory_id,
                            previous_content, content, memory_type, reason, created_at
                        ) VALUES (
                            CAST(:id AS uuid), :correlation_id,
                            CAST(:ingestion_id AS uuid),
                            :client_id, :instance_id, :customer_id, :user_id,
                            :action, CAST(:memory_id AS uuid),
                            CAST(:replaced_by_memory_id AS uuid),
                            :previous_content, :content, :memory_type, :reason, :created_at
                        )"""),
                {"id": str(_uuid.uuid4()),
                 "correlation_id": corr,
                 "ingestion_id": str(ingestion_id) if ingestion_id else None,
                 "client_id": str(scope.client_id),
                 "instance_id": str(scope.instance_id),
                 "customer_id": str(scope.customer_id),
                 "user_id": str(scope.user_id) if scope.user_id else None,
                 "action": action,
                 "memory_id": str(memory_id),
                 "replaced_by_memory_id": (
                     str(replaced_by_memory_id) if replaced_by_memory_id
                     else (str(superseded_by) if superseded_by else None)),
                 "previous_content": previous_content,
                 "content": content,
                 "memory_type": memory_type,
                 "reason": reason,
                 "created_at": now},
            )
            await session.commit()
            return True

    async def insert_customer_knowledge(self, memory, scope, document_id: str):
        # Consolidation F11 (Tech Spec §3 M8, Option B): org knowledge unifies
        # into `memories` at customer scope (user_id NULL). One store means
        # inline KU fires on org ingest and the cycles curate org knowledge
        # like everything else. Rows are born-promoted (org knowledge is
        # curated by definition until a cycle says otherwise). Flag-gated;
        # the legacy table is FROZEN (no new writes) while the flag is on.
        import os
        if os.getenv("SYNAP_ORG_KNOWLEDGE_UNIFIED", "false").lower() == "true":
            await self._execute(
                """INSERT INTO memories (
                    id, client_id, instance_id, customer_id, user_id,
                    memory_type, content, confidence, source_document_id,
                    created_at, fidelity, promoted_at
                ) VALUES (
                    CAST(:id AS uuid), :client_id, :instance_id, :customer_id,
                    NULL, :memory_type, :content, :confidence,
                    :source_document_id, :created_at, 1, NOW()
                ) ON CONFLICT (id) DO NOTHING""",
                {
                    "id": str(memory.memory_id),
                    "client_id": str(scope.client_id),
                    "instance_id": str(scope.instance_id),
                    "customer_id": str(scope.customer_id),
                    "memory_type": memory.memory_type.value,
                    "content": memory.content,
                    "confidence": float(getattr(memory, "confidence", 0.9) or 0.9),
                    "source_document_id": document_id,
                    "created_at": memory.extracted_at,
                },
            )
            return memory.memory_id

        await self._execute(
            """INSERT INTO customer_knowledge (
                id, client_id, customer_id, content,
                knowledge_type, source_document_id, created_at
            ) VALUES (
                CAST(:id AS uuid), :client_id, :customer_id, :content,
                :knowledge_type, :source_document_id, :created_at
            ) ON CONFLICT (id) DO NOTHING""",
            {
                "id": str(memory.memory_id),
                "client_id": str(scope.client_id),
                "customer_id": str(scope.customer_id),
                "content": memory.content,
                "knowledge_type": memory.memory_type.value,
                "source_document_id": document_id,
                "created_at": memory.extracted_at,
            },
        )
        return memory.memory_id
