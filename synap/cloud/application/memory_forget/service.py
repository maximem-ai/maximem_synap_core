"""Preview, approve, reject and direct delete.

One forget is one judge call (one per 100 candidates, so one for nearly every request).
The user's memories and every other store that may hold a copy of the text are
GATHERED first and pooled; the judge reads the pool once and flags what to remove.

``preview``  gathers, judges once, and writes a pending request holding exactly what
             was flagged, memories and copies, each tagged with where it lives.
             Nothing is deleted.
``approve``  deletes exactly the items the request holds, by id. It neither searches
             nor judges again, so what a person reviewed is what goes and approving
             costs no JEV call.
``direct``   gathers, judges once and deletes in the same call, for an agent that
             cannot wait for a review. When it matches nothing it queues a retry
             (``retry.py``), because the note may still be being saved.

Nothing matching is a normal answer, not an error: nothing is written or deleted.
An unavailable judge raises ``JudgeUnavailable`` and also deletes nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sqlalchemy import text

from synap.cloud.application.memory_forget import requests_store as store
from synap.cloud.application.memory_forget.models import Candidate, ForgetOutcome
from synap.cloud.application.memory_forget.soft_delete import SoftDeleteResult, soft_delete_memories
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

REASON_DIRECT = "forgotten on request"
REASON_APPROVED = "forgotten on request, approved in the dashboard"

# Memories taken from the vector search. Inside the 50 to 100 asked for, and kept below
# the top of that range so the pool (memories plus a few of each other store) stays
# near one judge call.
MEMORY_PAGE = 50
# A flagged memory in the last TAIL ranks of a full page means the page may have been
# cut off while still finding matches.
TAIL = 10

JudgeFn = Callable[[str, Sequence[Candidate]], Awaitable[List[Candidate]]]
# (scope, flagged items, reason) -> result
DeleteFn = Callable[[Any, Sequence[Candidate], str], Awaitable[SoftDeleteResult]]


@dataclass
class Gathered:
    candidates: List[Candidate] = field(default_factory=list)
    # Stores that could not be searched, so a copy may be sitting in them unseen.
    unreadable: List[str] = field(default_factory=list)
    # The memories among the candidates, in rank order, and whether the search filled
    # its page (there may be more memories than were looked at).
    memory_ids_ranked: List[str] = field(default_factory=list)
    page_full: bool = False


@dataclass
class ForgetDeps:
    gather: Callable[[str], Awaitable[Gathered]]
    judge: JudgeFn
    delete: DeleteFn


def _incomplete(gathered: Gathered, picked: Sequence[Candidate]) -> bool:
    if gathered.unreadable:
        return True
    if not gathered.page_full:
        return False
    tail = set(gathered.memory_ids_ranked[-TAIL:])
    return any(c.store == "memory" and c.memory_id in tail for c in picked)


def _nothing_matches(gathered: Gathered) -> ForgetOutcome:
    """No match. If a store could not be searched, the text may still be in it,
    so say so: this used to answer `may_be_incomplete=False`, which a caller
    reads as "Synap holds nothing like this"."""
    return ForgetOutcome(status="nothing_matches",
                         may_be_incomplete=bool(gathered.unreadable))


async def _find(description: str, deps: ForgetDeps):
    gathered = await deps.gather(description)
    picked = await deps.judge(description, gathered.candidates) if gathered.candidates else []
    return gathered, picked


async def _outcome_from_delete(res: SoftDeleteResult, candidates: List[Candidate],
                               unreadable: Sequence[str] = (),
                               may_be_incomplete: bool = False,
                               request_id: Optional[str] = None,
                               retry_scheduled: bool = False) -> ForgetOutcome:
    # A skipped ID is one the UPDATE refused (already deleted, or not this user).
    # Already deleted is the state we wanted, so it only counts as failed when the
    # request asked for it and nothing marked it.
    still = list(res.still_present_in) + [u for u in unreadable
                                          if u not in res.still_present_in]
    return ForgetOutcome(
        status="partial" if still else "deleted",
        candidates=candidates, request_id=request_id,
        deleted_ids=res.deleted_ids, failed_ids=res.skipped_ids,
        still_present_in=still, may_be_incomplete=may_be_incomplete,
        retry_scheduled=retry_scheduled)


async def preview(session, *, scope, description: str, deps: ForgetDeps,
                  requested_by: Optional[str]) -> ForgetOutcome:
    gathered, picked = await _find(description, deps)
    if not picked:
        return _nothing_matches(gathered)
    request_id = await store.create_pending(
        session, scope=scope, query=description, candidates=picked,
        requested_by=requested_by)
    return ForgetOutcome(status="preview", candidates=list(picked), request_id=request_id,
                         may_be_incomplete=_incomplete(gathered, picked))


def _decided_status(candidates: Sequence[Candidate], res: SoftDeleteResult) -> str:
    asked_memories = any(c.store == "memory" for c in candidates)
    return (store.FAILED if asked_memories and not res.deleted_ids and res.skipped_ids
            else store.APPLIED)


async def direct(session, *, scope, description: str, deps: ForgetDeps,
                 requested_by: Optional[str] = None,
                 conversation_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 customer_id: Optional[str] = None) -> ForgetOutcome:
    """Find and delete now, and keep watching for what is still being saved.

    The search deletes what is saved and indexed. A note the person means may still be in
    flight (unsent in a batch of chat, or extracted but not yet stored), so the request is
    also remembered for each conversation that could hold it, and when a batch of such a
    conversation has been stored its new notes are checked against the request
    (retry.py, marker.py).

    Which conversations: the one in ``conversation_id`` if the caller gave it, otherwise the
    user's recently active conversations, found on the server from ``user_id`` and
    ``customer_id`` (the external ids the caller used). Nothing for the agent to send. If there
    is none, nothing could still be in flight: a request that finds nothing is answered
    "nothing found" and nothing waits.

    The rows are filed and committed BEFORE the search, so a batch that finishes storing while
    the search runs cannot be missed. They stay open whether the search finds something or
    nothing: finding something is not finding everything (a delete can remove
    other items while a related note is still being saved). Only a search that
    fails closes them.
    """
    from synap.cloud.application.memory_forget import retry

    waiting_ids = []
    for conv in await retry.conversations_to_watch(
            session, scope=scope, user_id=user_id, customer_id=customer_id,
            conversation_id=conversation_id):
        waiting_id = await retry.schedule_retry(
            session, scope=scope, description=description, requested_by=requested_by,
            conversation_id=conv)
        if waiting_id:
            waiting_ids.append(waiting_id)
    if waiting_ids:
        await _commit(session)
    try:
        gathered, picked = await _find(description, deps)
    except Exception:
        # Nothing was deleted and the caller is told so (or the search failed): do not leave a
        # request behind that would delete later.
        for waiting_id in waiting_ids:
            await retry.cancel_retry(session, waiting_id)
        raise
    if not picked:
        outcome = _nothing_matches(gathered)
        outcome.retry_scheduled = bool(waiting_ids)
        return outcome
    res = await deps.delete(scope, picked, REASON_DIRECT)
    request_id = None
    try:
        # A savepoint: the deletes are already committed, so failing to file the
        # record must not turn a done deletion into an error the caller retries.
        async with session.begin_nested():
            # The dashboard queue is Cloud's: None (no record) when no hook is registered.
            request_id = await edition.acall(
                "forget_dashboard_queue_write",
                session, scope=scope, query=description, candidates=picked,
                requested_by=requested_by, status=_decided_status(picked, res),
                result={"deleted": res.deleted_ids, "skipped": res.skipped_ids,
                        "still_present_in": res.still_present_in, "mode": "direct"})
    except Exception:
        logger.exception("forget_direct_record_failed deleted=%d", len(res.deleted_ids))
    return await _outcome_from_delete(res, list(picked), gathered.unreadable,
                                      _incomplete(gathered, picked), request_id=request_id,
                                      retry_scheduled=bool(waiting_ids))


async def _commit(session) -> None:
    """Make the waiting row visible to other workers before the search starts."""
    commit = getattr(session, "commit", None)
    if commit is not None:
        await commit()


async def approve(session, *, client_id: str, request_id: str, decided_by: str,
                  scope_of: Callable[[Dict[str, Any]], Any],
                  deps_for: Callable[[Any], ForgetDeps]) -> Optional[ForgetOutcome]:
    """Approve a pending request. None when it does not exist or is not pending."""
    req = await store.get_request(session, client_id=client_id, request_id=request_id)
    if req is None or req["status"] != store.PENDING:
        return None
    if not await store.claim_pending(session, client_id=client_id, request_id=request_id,
                                     decided_by=decided_by, new_status=store.APPLIED):
        return None                      # someone else decided it first
    candidates = store.to_candidates(req)
    scope = scope_of(req)
    deps = deps_for(scope)
    try:
        res = await deps.delete(scope, candidates, REASON_APPROVED)
    except Exception as exc:  # noqa: BLE001
        await store.finish(session, request_id=request_id, status=store.FAILED,
                           result={"error": f"{type(exc).__name__}"})
        raise
    status = _decided_status(candidates, res)
    await store.finish(session, request_id=request_id, status=status,
                       result={"deleted": res.deleted_ids, "skipped": res.skipped_ids,
                               "still_present_in": res.still_present_in})
    return await _outcome_from_delete(res, [], request_id=request_id)


async def search_or_nothing(vector, query_vec, scope, *, user_id):
    """Vector hits for a forget request; a user with no vector collection has none.

    A person who never stored anything has no collection, and Chroma raises on a search of one. For
    forget that is "nothing matches", not a failure. Only the missing-collection case is absorbed:
    any other error still raises, so a real wiring fault is not turned into an empty answer.
    """
    try:
        return await vector.search(query_vec, scope, top_k=MEMORY_PAGE, filters={"user_id": user_id})
    except Exception as exc:  # noqa: BLE001 - filtered to one known case below
        if "does not exist" not in str(exc).lower():
            raise
        return []


async def reject(session, *, client_id: str, request_id: str, decided_by: str) -> bool:
    if not await store.claim_pending(session, client_id=client_id, request_id=request_id,
                                     decided_by=decided_by, new_status=store.REJECTED):
        return False
    await store.finish(session, request_id=request_id, status=store.REJECTED, result={})
    return True


def scope_from_request(req: Dict[str, Any]):
    """The scope a stored request was made under, built from its own internal ids.

    Not `_resolve_scope`: these are already internal uuids, and running them through
    the external-id mapper hashes them a second time (see `_scope_of_stored_row`).
    """
    import uuid as _uuid

    from synap.cloud.agents.ingestion.models import Scope

    def _u(v):
        return _uuid.UUID(str(v)) if v else None

    return Scope(client_id=_u(req["client_id"]), instance_id=_u(req.get("instance_id")),
                 customer_id=_u(req.get("customer_id")), user_id=_u(req.get("user_id")),
                 scope_level=req.get("scope_level") or "user")


# --------------------------------------------------------------------------- wiring

def build_deps(session, *, scope, correlation_id: Optional[str] = None,
               judge: Optional[JudgeFn] = None) -> ForgetDeps:
    """The real stores behind the three functions above.

    Kept apart so the logic above is tested with fakes, and this is the only place
    that touches the providers.
    """
    from synap.cloud.application.memory_forget import judge as _judge
    from synap.cloud.application.memory_forget.erase_copies import (
        STORE_LABEL, erase_other_copies, gather_copies)
    from synap.cloud.application.profile.store import ProfileStore
    from synap.cloud.application.shared.providers import get_cached_providers

    providers = get_cached_providers()
    embedding = providers.get("embedding")
    vector = providers.get("vector")
    graph = providers.get("graph")
    user_id = str(scope.user_id) if scope.user_id else None

    async def _load_live(ids: Sequence[str]) -> List[Candidate]:
        # The vector hit is only a pointer. Every memory is re-read from Postgres,
        # scoped to the one user, live and not deleted, so a vector from another
        # user's bucket or an orphaned vector can never reach the judge or the delete.
        if not ids:
            return []
        rows = await session.execute(
            text("""
                SELECT id, content, memory_type FROM memories
                 WHERE id = ANY(CAST(:ids AS uuid[]))
                   AND client_id = CAST(:client_id AS uuid)
                   AND customer_id = CAST(:customer_id AS uuid)
                   AND user_id = CAST(:user_id AS uuid)
                   AND superseded_by IS NULL AND deprecated_at IS NULL
                   AND deleted_at IS NULL
            """),
            {"ids": list(ids), "client_id": str(scope.client_id),
             "customer_id": str(scope.customer_id), "user_id": user_id},
        )
        by_id = {str(r[0]): Candidate(str(r[0]), r[1] or "", r[2]) for r in rows.fetchall()}
        return [by_id[i] for i in ids if i in by_id]          # keep the vector's rank order

    async def gather(description: str) -> Gathered:
        if embedding is None or vector is None:
            raise RuntimeError("embedding or vector provider is not configured")
        query_vec = await embedding.embed(description)
        hits = await search_or_nothing(vector, query_vec, scope, user_id=user_id)
        ids = list(dict.fromkeys(str(h["id"]) for h in hits))
        memories = await _load_live(ids)
        found, unreadable = await gather_copies(
            session, scope, query_vec, vector=vector, profile_store=ProfileStore())
        # What a person sees as the "where" of a copy; memories keep their own type.
        copies = [Candidate(c.memory_id, c.content, STORE_LABEL.get(c.store, c.store), c.store)
                  for c in found]
        return Gathered(candidates=memories + copies, unreadable=unreadable,
                        memory_ids_ranked=[m.memory_id for m in memories],
                        page_full=len(ids) >= MEMORY_PAGE)

    async def _judge_fn(desc: str, cands: Sequence[Candidate]) -> List[Candidate]:
        # The judge reads the very text the user asked to have removed. Naming
        # the operation is what keeps that text out of the LLM call traces:
        # operations with "forget" in the name never have their text stored.
        from synap.cloud.infrastructure.llm.call_trace import trace_scope

        with trace_scope(operation="memory_forget.judge", correlation_id=correlation_id):
            return await _judge.pick(desc, cands, correlation_id=correlation_id)

    async def delete(for_scope, picked: Sequence[Candidate], reason: str) -> SoftDeleteResult:
        memories = [c for c in picked if c.store == "memory"]
        copies = [c for c in picked if c.store != "memory"]
        res = SoftDeleteResult()
        if memories:
            res = await soft_delete_memories(
                session, scope=for_scope, memory_ids=[c.memory_id for c in memories],
                reason=reason, correlation_id=correlation_id, vector=vector, graph=graph,
                invalidate=_make_invalidate(providers, for_scope), audit=_audit)
        # The text also sits in copies that carry no memory id; they were flagged in the
        # same judge call and are removed by exact id. A store that could not be
        # cleared is named, not hidden.
        failed = await erase_other_copies(
            session, scope=for_scope, memory_ids=res.deleted_ids, contents=res.contents,
            copies=copies, vector=vector, profile_store=ProfileStore())
        res.still_present_in.extend(f for f in failed if f not in res.still_present_in)
        return res

    return ForgetDeps(gather=gather, judge=judge or _judge_fn, delete=delete)


def _make_invalidate(providers, scope):
    async def _run():
        from synap.cloud.application.client_api.retrieval_manager import (
            invalidate_client_query_cache)
        from synap.cloud.shared import edition

        redis = providers.get("redis")
        if redis is not None:
            # Cloud: the consolidation invalidation (Redis tiers, local caches, publish).
            # Core registers nothing; the client's query cache is dropped just below.
            await edition.acall("forget_cache_invalidate", redis=redis,
                                client_id=str(scope.client_id),
                                customer_id=str(scope.customer_id),
                                user_id=str(scope.user_id) if scope.user_id else None,
                                instance_id=str(scope.instance_id) if scope.instance_id else None)
        # The L1 key is per client, so one user's cached bundle can sit under
        # another customer's prefix. Drop the whole client's query results.
        await invalidate_client_query_cache(str(scope.client_id))
    return _run


async def _audit(*, correlation_id: str, scope, memory_id: str, reason: str) -> None:
    from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper
    await SQLAlchemyRDBMSWrapper().insert_memory_action(
        correlation_id=correlation_id, ingestion_id=None, scope=scope, action="DELETE",
        memory_id=memory_id, content=None, previous_content=None, memory_type=None,
        reason=reason)
