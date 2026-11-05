"""Outbox for the graph side of retiring or deleting a memory.

Postgres and Neo4j cannot share a transaction. A path that retires or deletes
a memory writes a ``graph_sync_pending`` row in the SAME Postgres transaction
(``enqueue``), so the intent to release the graph is committed exactly when the
memory change is. The graph step (``GraphLifecycle``) then runs from:

* ``drain_for_ingestion``: Stage 4, right after the document's graph write
  (supersessions are decided before the new memory's edges exist, so the rule
  can only judge them afterwards);
* ``drain_rows``: an inline caller (the delete route) right after its commit;
* ``run_drain_worker``: the background worker, for everything still pending
  (graph down, process killed), with exponential backoff.

A database without the table (migration 001 not applied) keeps working: the
enqueue is skipped with one warning and the reconciler (graph_reconcile)
remains the safety net.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

TABLE = "graph_sync_pending"
OPS = ("retire", "erase")
BASE_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 3600
LEASE_SECONDS = 120
BATCH = 50


def backoff_seconds(attempts: int) -> int:
    """5 s, 10 s, 20 s ... capped at one hour. A row is never given up: the
    graph must follow Postgres eventually, and the age is visible in the table."""
    return int(min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** max(0, attempts - 1))))


def ingestion_grace_seconds() -> int:
    """How long the background worker leaves a row written by an ingestion
    alone. Such a row belongs to Stage 4, which drains it right after the
    document's graph write (the rule must see the new memory's edges). The row
    was due at once, so the worker could take it first: it released BEFORE the
    graph write, raced Stage 4's own writes on the same records, and held the
    lease while Stage 4's drain looked for it (the claim skips leased rows), so
    Stage 4 found nothing to mark done and a failed worker attempt left the
    row pending. After the grace the worker is the backstop (graph write
    failed, process killed). SYNAP_GRAPH_SYNC_INGESTION_GRACE_SECONDS, default 300."""
    import os
    try:
        return max(0, int(os.getenv("SYNAP_GRAPH_SYNC_INGESTION_GRACE_SECONDS", "300")))
    except ValueError:
        return 300


async def table_available(session=None) -> bool:
    from synap.cloud.database import schema_compat
    ok = await schema_compat.has_table(TABLE, session)
    if not ok:
        schema_compat.warn_once(
            "graph_sync_pending_missing",
            "graph_sync_pending table is missing (core migration 001 not applied): retired and deleted "
            "memories are not queued for the graph; run graph_reconcile to repair the graph.")
    return ok


async def superseded_by_any(session, memory_ids: List[str]) -> Dict[str, List[str]]:
    """{N: [S, ...]}: the memories each of *memory_ids* had superseded.

    Read from ``memories.superseded_by`` (only correct BEFORE N is deleted: the
    foreign key clears it) and from the ``SUPERSEDE`` rows of ``memory_actions``
    (the audit trail, which has no foreign key and survives the delete). Must
    run in the caller's transaction, before its delete."""
    from sqlalchemy import text
    out: Dict[str, List[str]] = {}
    if not memory_ids:
        return out
    params = {"ids": [str(i) for i in memory_ids]}
    queries = [
        "SELECT superseded_by AS n, id AS s FROM memories "
        "WHERE superseded_by = ANY(CAST(:ids AS uuid[]))",
        "SELECT replaced_by_memory_id AS n, memory_id AS s FROM memory_actions "
        "WHERE action = 'SUPERSEDE' AND replaced_by_memory_id = ANY(CAST(:ids AS uuid[]))",
    ]
    for sql in queries:
        # No try/except: a failed statement aborts the caller's transaction,
        # and swallowing it would only move the error to the delete.
        rows = (await session.execute(text(sql), params)).fetchall()
        for n, s_id in rows or []:
            if n is None or s_id is None:
                continue
            lst = out.setdefault(str(n), [])
            if str(s_id) not in lst:
                lst.append(str(s_id))
    return out


async def enqueue(session, *, memory_ids: Iterable[Any], scope: Any, op: str, reason: str,
                  superseded_by: Optional[Any] = None, redact: bool = False,
                  ingestion_id: Optional[Any] = None,
                  restore_ids: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Insert one pending row per memory with *session*, WITHOUT committing:
    the caller commits it together with the memory change. Returns the row ids
    ([] when the table does not exist).

    For ``op="erase"`` each row also records the memories that memory had
    superseded (``restore_memory_ids``), so the graph can restore them once
    the delete clears their ``superseded_by``. *restore_ids* ({N: [S]}) is
    used when given; otherwise they are read here, so call this BEFORE the
    delete statement."""
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}, not {op!r}")
    ids = [str(m) for m in dict.fromkeys(memory_ids or []) if m]
    if not ids or not await table_available(session):
        return []
    from sqlalchemy import text
    from synap.cloud.agents.ingestion.storage.graph_lifecycle import release_namespaces
    from synap.cloud.database import schema_compat
    namespaces = release_namespaces(scope)
    with_restore = op == "erase" and await schema_compat.has_column(TABLE, "restore_memory_ids", session)
    if op == "erase" and not with_restore:
        schema_compat.warn_once(
            "graph_sync_restore_missing",
            "graph_sync_pending.restore_memory_ids is missing (core migration 002 not applied): "
            "deleting a memory does not restore the older memory it replaced in the graph; "
            "graph_reconcile repairs it.")
    if with_restore and restore_ids is None:
        restore_ids = await superseded_by_any(session, ids)
    restore_ids = restore_ids or {}
    row_ids: List[str] = []
    for mid in ids:
        rid = str(uuid.uuid4())
        await session.execute(
            text(f"""INSERT INTO {TABLE} (
                        id, memory_id, client_id, instance_id, customer_id, user_id, scope_level,
                        namespaces, op, reason, superseded_by, redact, ingestion_id, next_attempt_at
                        {', restore_memory_ids' if with_restore else ''})
                    VALUES (CAST(:id AS uuid), CAST(:memory_id AS uuid), :client_id, :instance_id,
                            :customer_id, :user_id, :scope_level, :namespaces, :op, :reason,
                            CAST(:superseded_by AS uuid), :redact, CAST(:ingestion_id AS uuid),
                            now() + make_interval(secs => CAST(:grace AS double precision))
                            {', CAST(:restore AS uuid[])' if with_restore else ''})"""),
            {"id": rid, "memory_id": mid,
             "client_id": str(scope.client_id),
             "instance_id": _s(getattr(scope, "instance_id", None)),
             "customer_id": _s(getattr(scope, "customer_id", None)),
             "user_id": _s(getattr(scope, "user_id", None)),
             "scope_level": getattr(scope, "scope_level", None),
             "namespaces": namespaces, "op": op, "reason": reason,
             "superseded_by": _s(superseded_by), "redact": bool(redact),
             "ingestion_id": _s(ingestion_id),
             "grace": float(ingestion_grace_seconds() if ingestion_id is not None else 0),
             "restore": [str(x) for x in restore_ids.get(mid, [])]})
        row_ids.append(rid)
    return row_ids


def _s(v: Any) -> Optional[str]:
    return str(v) if v not in (None, "") else None


# --- the store: SQL here, an in-memory twin in the tests ---------------------------

class PostgresOutboxStore:
    """Claims, completes and fails rows. Each call is its own short transaction."""

    def __init__(self, session_factory=None):
        self._factory = session_factory

    def _session(self):
        if self._factory is None:
            from synap.cloud.database.connection import get_session_factory
            self._factory = get_session_factory()
        return self._factory()

    async def available(self) -> bool:
        return await table_available()

    async def claim(self, *, limit: int = BATCH, ingestion_id: Optional[str] = None,
                    row_ids: Optional[List[str]] = None, lease_seconds: int = LEASE_SECONDS) -> List[Dict[str, Any]]:
        """Lease up to *limit* due rows (``FOR UPDATE SKIP LOCKED``, so two
        workers never take the same row) and count the attempt."""
        from sqlalchemy import text
        where = ["done_at IS NULL", "(leased_until IS NULL OR leased_until < now())"]
        params: Dict[str, Any] = {"limit": int(limit), "lease": float(lease_seconds)}
        if row_ids is not None:
            where.append("id = ANY(CAST(:row_ids AS uuid[]))")
            params["row_ids"] = [str(r) for r in row_ids]
        elif ingestion_id is not None:
            where.append("ingestion_id = CAST(:ingestion_id AS uuid)")
            params["ingestion_id"] = str(ingestion_id)
        else:
            where.append("next_attempt_at <= now()")
        from synap.cloud.database import schema_compat
        restore_col = ", restore_memory_ids" if await schema_compat.has_column(
            TABLE, "restore_memory_ids") else ""
        async with self._session() as session:
            rows = (await session.execute(text(f"""
                UPDATE {TABLE} SET attempts = attempts + 1,
                                   leased_until = now() + make_interval(secs => CAST(:lease AS double precision))
                WHERE id IN (SELECT id FROM {TABLE} WHERE {' AND '.join(where)}
                             ORDER BY created_at LIMIT :limit FOR UPDATE SKIP LOCKED)
                RETURNING id, memory_id, client_id, instance_id, customer_id, user_id, scope_level,
                          namespaces, op, reason, superseded_by, redact, ingestion_id, attempts{restore_col}"""),
                params)).mappings().all()
            await session.commit()
        out = []
        for r in rows:
            d = {k: (str(v) if k in ("id", "memory_id", "superseded_by", "ingestion_id") and v is not None else v)
                 for k, v in dict(r).items()}
            d["restore_memory_ids"] = [str(x) for x in (d.get("restore_memory_ids") or [])]
            out.append(d)
        return out

    async def complete(self, row_ids: List[str]) -> None:
        from sqlalchemy import text
        if not row_ids:
            return
        async with self._session() as session:
            await session.execute(text(f"""UPDATE {TABLE} SET done_at = now(), leased_until = NULL,
                                             last_error = NULL
                                           WHERE id = ANY(CAST(:ids AS uuid[]))"""),
                                  {"ids": [str(r) for r in row_ids]})
            await session.commit()

    async def fail(self, row_ids: List[str], error: str, attempts: int) -> None:
        from sqlalchemy import text
        if not row_ids:
            return
        async with self._session() as session:
            await session.execute(text(f"""UPDATE {TABLE} SET leased_until = NULL, last_error = :err,
                                             next_attempt_at = now() + make_interval(secs => CAST(:wait AS double precision))
                                           WHERE id = ANY(CAST(:ids AS uuid[]))"""),
                                  {"ids": [str(r) for r in row_ids], "err": error[:2000],
                                   "wait": float(backoff_seconds(attempts))})
            await session.commit()


# --- applying rows -------------------------------------------------------------

@dataclass
class DrainResult:
    claimed: int = 0
    done: int = 0
    failed: int = 0


def _scope_of(row: Dict[str, Any]):
    from uuid import UUID
    from synap.cloud.agents.ingestion.models import Scope

    def u(v):
        try:
            return UUID(str(v)) if v else None
        except ValueError:
            return None
    return Scope(client_id=u(row["client_id"]), instance_id=u(row.get("instance_id")) or u(row["client_id"]),
                 customer_id=u(row.get("customer_id")), user_id=u(row.get("user_id")),
                 scope_level=row.get("scope_level"))


async def apply_rows(rows: List[Dict[str, Any]], store, lifecycle) -> DrainResult:
    """Release each group of rows (same op, reason, superseder, redact and
    namespaces) through *lifecycle*, then mark them done or failed."""
    result = DrainResult(claimed=len(rows))
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["op"], r["reason"], r.get("superseded_by"), bool(r.get("redact")),
               tuple(r.get("restore_memory_ids") or ()),
               tuple(r.get("namespaces") or ()), r["client_id"], r.get("customer_id"), r.get("user_id"),
               r.get("scope_level"))
        groups.setdefault(key, []).append(r)
    for (op, reason, superseded_by, redact, restore, namespaces, *_), group in groups.items():
        ids = [r["memory_id"] for r in group]
        row_ids = [r["id"] for r in group]
        try:
            scope = _scope_of(group[0])
            for ns in namespaces:
                await lifecycle.release_in_namespace(ns, ids, mode=op, reason=reason,
                                                     superseded_by=superseded_by, redact=redact, scope=scope,
                                                     restore_ids=list(restore) or None)
            await store.complete(row_ids)
            result.done += len(group)
        except Exception as exc:  # noqa: BLE001 - the row stays pending and is retried
            attempts = max(int(r.get("attempts") or 1) for r in group)
            logger.warning("graph_sync_release_failed rows=%d op=%s attempts=%d error=%s",
                           len(group), op, attempts, exc)
            try:
                await store.fail(row_ids, f"{type(exc).__name__}: {exc}", attempts)
            except Exception:  # noqa: BLE001 - the lease expires and the row is retried anyway
                logger.warning("graph_sync could not record the failure", exc_info=True)
            result.failed += len(group)
    return result


def _lifecycle(graph, vector=None):
    from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
    return GraphLifecycle(graph, vector=vector)


async def drain_rows(row_ids: List[str], *, graph, vector=None, store=None) -> DrainResult:
    """Apply these rows now (an inline caller, right after its commit)."""
    if not row_ids or graph is None:
        return DrainResult()
    store = store or PostgresOutboxStore()
    rows = await store.claim(row_ids=list(row_ids), limit=len(row_ids))
    return await apply_rows(rows, store, _lifecycle(graph, vector))


async def drain_for_ingestion(ingestion_id: Any, *, graph, vector=None, store=None) -> int:
    """Stage 4, after the document's graph write: apply this ingestion's rows.
    Returns how many were applied. Never raises: what fails stays pending."""
    if graph is None or not ingestion_id:
        return 0
    store = store or PostgresOutboxStore()
    try:
        if not await store.available():
            return 0
        rows = await store.claim(ingestion_id=str(ingestion_id), limit=1000)
        return (await apply_rows(rows, store, _lifecycle(graph, vector))).done
    except Exception:  # noqa: BLE001
        logger.warning("graph_sync drain for ingestion %s failed; the worker retries", ingestion_id,
                       exc_info=True)
        return 0


async def drain_once(*, graph, vector=None, store=None, limit: int = BATCH) -> DrainResult:
    store = store or PostgresOutboxStore()
    rows = await store.claim(limit=limit)
    if not rows:
        return DrainResult()
    return await apply_rows(rows, store, _lifecycle(graph, vector))


async def run_drain_worker(*, graph, vector=None, store=None, interval: float = 10.0,
                           stop: Optional[asyncio.Event] = None) -> None:
    """Background loop: drain due rows until *stop* is set or the task is
    cancelled. Idles (with one warning) while the table is missing."""
    store = store or PostgresOutboxStore()
    stop = stop or asyncio.Event()
    logger.info("graph_sync drain worker started")
    while not stop.is_set():
        try:
            if await store.available():
                result = await drain_once(graph=graph, vector=vector, store=store)
                if result.claimed:
                    logger.info("graph_sync drained claimed=%d done=%d failed=%d",
                                result.claimed, result.done, result.failed)
                    if result.claimed >= BATCH and not result.failed:
                        continue  # more may be due right now
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop must outlive one bad pass
            logger.warning("graph_sync drain pass failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
