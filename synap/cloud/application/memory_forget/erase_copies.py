"""Clear the copies of a forgotten memory's text that are not keyed by its id.

Marking the row deleted and removing its own vectors is not enough. The same text
sits in places that were written from it and that no read path filters on
``deleted_at``:

    memory_actions     audit rows carry the old and new text of a change
    loss_manifests     what a compression dropped, as text
    request_payloads   the response body of a fetch that served the memory
    candidates         below-threshold extractions, which nap can later promote
                       back into memories (so the value could return from here)
    profile            the per-user profile document, an LLM merge of memories
    chunks, sentences, entities, triples
                       raw source text and what was lifted out of it

The first three are keyed by the memory's id or hold its exact text, so they are
cleared directly. The rest carry no memory id and are found by what they SAY: this
module GATHERS them next to the memories (``gather_copies``), the caller asks the
judge once about the whole pool, and ``erase_other_copies`` then removes exactly the
ones that were picked, by id. Nothing here judges anything.

Each step stands alone. One failing is named in the returned list and does not stop
the others, so the caller can tell a person which store still holds the text.

Known limits, stated so nobody assumes more than this does:
  * Chunks are raw source text. A chunk that states the value is removed whole, which
    can take other sentences in the same chunk with it.
  * The raw conversation or document the user originally sent is not touched here.
    That includes the logged copy of an ingest request in request_payloads: those
    rows are skipped by path.
  * request_payloads rows carry only client_id and instance_id (customer_id and
    user_id are NULL on every real row), so they are scoped to the client and
    instance and matched on the memory id or its exact text, not on the user. A
    log row of another user that states the same text is blanked too.
  * These stores are region-local (only memories, memory_actions and request_payloads
    replicate, plus vector deletes through the vector write log). A request run in
    one region does not reach candidates, loss manifests or the profile in the other.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from synap.cloud.application.memory_forget.models import Candidate

logger = logging.getLogger(__name__)

# A memory text shorter than this is too short to match a payload on without
# catching unrelated rows, so only its id is used.
_MIN_PATTERN = 6
# How many hits each other store contributes to the one pooled judge call, and how
# many pending extractions. Small on purpose: the copies of one fact are few, and
# the pool has to stay near one judge call.
PER_STORE_HITS = 10
CANDIDATES_CAP = 25
# store label -> vector collection
COLLECTION_OF = {"chunk": "chunks", "sentence": "sentences",
                 "entity": "entities", "triple": "triples"}
VECTOR_COLLECTIONS = tuple(COLLECTION_OF.values())
# What a person sees as the "where" of a copy in the dashboard.
STORE_LABEL = {"chunk": "source text", "sentence": "source sentence", "entity": "entity",
               "triple": "relationship", "profile": "profile",
               "candidate": "pending extraction"}


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def payload_patterns(memory_ids: Sequence[str], contents: Dict[str, str]) -> List[str]:
    """LIKE patterns that find a saved request or response body naming these memories.

    The body is jsonb, so its text form escapes quotes and backslashes; both the raw
    and the JSON-escaped spelling of each memory text are tried.
    """
    pats: List[str] = [f"%{_like_escape(i)}%" for i in memory_ids]
    for content in contents.values():
        content = (content or "").strip()
        if len(content) < _MIN_PATTERN:
            continue
        for form in {content, json.dumps(content, ensure_ascii=False)[1:-1]}:
            pats.append(f"%{_like_escape(form)}%")
    return list(dict.fromkeys(pats))


async def _scrub_actions(session, scope, ids: Sequence[str]) -> None:
    await session.execute(
        text("""
            UPDATE memory_actions
               SET content = NULL, previous_content = NULL
             WHERE client_id = CAST(:client_id AS uuid)
               AND (memory_id = ANY(CAST(:ids AS uuid[]))
                    OR replaced_by_memory_id = ANY(CAST(:ids AS uuid[])))
               AND (content IS NOT NULL OR previous_content IS NOT NULL)
        """),
        {"client_id": str(scope.client_id), "ids": list(ids)})


async def _scrub_loss_manifests(session, scope, ids: Sequence[str]) -> None:
    await session.execute(
        text("""
            UPDATE loss_manifests
               SET active = FALSE, dropped_details = '[]'::jsonb
             WHERE client_id = CAST(:client_id AS uuid)
               AND customer_id = CAST(:customer_id AS uuid)
               AND (summary_memory_id = ANY(CAST(:ids AS uuid[]))
                    OR source_memory_ids && CAST(:ids AS uuid[]))
               AND (active OR dropped_details <> '[]'::jsonb)
        """),
        {"client_id": str(scope.client_id), "customer_id": str(scope.customer_id),
         "ids": list(ids)})


async def _scrub_payloads(session, scope, ids: Sequence[str],
                          contents: Dict[str, str]) -> None:
    pats = payload_patterns(ids, contents)
    if not pats:
        return
    await session.execute(
        text("""
            UPDATE request_payloads
               SET request_body = NULL, response_body = NULL
             WHERE client_id = CAST(:client_id AS uuid)
               AND (instance_id IS NULL OR instance_id = CAST(:instance_id AS uuid))
               AND path NOT LIKE '%/memories/create%'
               AND path NOT LIKE '%/memories/batch%'
               AND path NOT LIKE '%/memories/upload%'
               AND path NOT LIKE '%ingest%'
               AND (request_body IS NOT NULL OR response_body IS NOT NULL)
               AND (response_body::text LIKE ANY(CAST(:pats AS text[]))
                    OR request_body::text LIKE ANY(CAST(:pats AS text[])))
        """),
        {"client_id": str(scope.client_id),
         "instance_id": str(scope.instance_id) if scope.instance_id else None,
         "pats": pats})


async def _scrub_llm_traces(session, scope, ids: Sequence[str],
                            contents: Dict[str, str]) -> None:
    """Blank the stored text of model calls that quote these memories.

    The ``forget_llm_trace_scrub`` hook does it (Cloud writes LLM call traces; Core
    keeps none, so nothing is registered and there is nothing to blank).
    """
    from synap.cloud.shared import edition

    await edition.acall("forget_llm_trace_scrub", session, scope, ids, contents)


async def _load_candidates(session, scope) -> List[Candidate]:
    """The user's most recent below-threshold extractions, as judge candidates."""
    rows = await session.execute(
        text("""
            SELECT id, content, memory_type FROM candidates
             WHERE client_id = CAST(:client_id AS uuid)
               AND customer_id = CAST(:customer_id AS uuid)
               AND user_id = CAST(:user_id AS uuid)
             ORDER BY created_at DESC LIMIT :cap
        """),
        {"client_id": str(scope.client_id), "customer_id": str(scope.customer_id),
         "user_id": str(scope.user_id), "cap": CANDIDATES_CAP})
    return [Candidate(str(r[0]), r[1] or "", r[2], store="candidate")
            for r in rows.fetchall() if (r[1] or "").strip()]


async def _delete_candidates(session, scope, ids: Sequence[str]) -> None:
    await session.execute(
        text("""
            DELETE FROM candidates
             WHERE id = ANY(CAST(:ids AS uuid[]))
               AND client_id = CAST(:client_id AS uuid)
               AND user_id = CAST(:user_id AS uuid)
        """),
        {"ids": list(ids), "client_id": str(scope.client_id),
         "user_id": str(scope.user_id)})


def profile_candidates(profile: Dict[str, Any]) -> List[Candidate]:
    """The parts of a profile document a person could have asked to forget.

    Each attribute and each extra is one candidate, id'd by where it lives; the
    overview is one candidate. The judge reads them exactly as it reads memories.
    """
    out: List[Candidate] = []
    for name, attr in (profile.get("attributes") or {}).items():
        value = attr.get("value") if isinstance(attr, dict) else attr
        out.append(Candidate(f"attributes:{name}", f"{name}: {value}"))
    for name, value in (profile.get("extras") or {}).items():
        out.append(Candidate(f"extras:{name}", f"{name}: {value}"))
    overview = profile.get("overview")
    if isinstance(overview, str) and overview.strip():
        out.append(Candidate("overview", overview))
    return out


def strip_profile(profile: Dict[str, Any], picked_ids: Sequence[str]) -> Dict[str, Any]:
    """The profile without the picked parts. A blanked overview regrows at the next merge."""
    picked = set(picked_ids)
    new = dict(profile)
    new["attributes"] = {k: v for k, v in (profile.get("attributes") or {}).items()
                         if f"attributes:{k}" not in picked}
    new["extras"] = {k: v for k, v in (profile.get("extras") or {}).items()
                     if f"extras:{k}" not in picked}
    if "overview" in picked:
        new["overview"] = ""
    return new


async def _profile_owner_and_doc(scope, store):
    owner = await store.resolve_owner_by_scope_uuids(
        scope.client_id, scope.user_id, scope.customer_id)
    if owner is None:
        return None, None
    return owner, await store.read(owner)


async def _strip_profile_store(scope, picked_ids: Sequence[str], store) -> None:
    owner, profile = await _profile_owner_and_doc(scope, store)
    if owner is None or not profile:
        return
    # Applied to the FRESH document under the row lock, so a profile merge that landed
    # since the read keeps its other changes.
    await store.write(owner, lambda fresh: strip_profile(fresh or profile, picked_ids))


async def gather_copies(session, scope, query_vec: Sequence[float], *, vector=None,
                        profile_store=None) -> Tuple[List[Candidate], List[str]]:
    """Everything outside the memories table that might state what was asked to be
    forgotten, as judge candidates. Returns (candidates, stores that could not be read).

    Nothing is judged here. The caller pools these with the memories and asks the
    judge ONCE, so a forget costs one call however many stores hold a copy.
    """
    out: List[Candidate] = []
    unreadable: List[str] = []

    for label, collection in COLLECTION_OF.items():
        if vector is None:
            unreadable.append(collection)
            continue
        try:
            hits = await vector.search_in(query_vec, scope, collection, PER_STORE_HITS,
                                          {"user_id": str(scope.user_id)})
        except Exception as exc:  # noqa: BLE001 - one store failing must not stop the rest
            unreadable.append(collection)
            logger.error("forget_copy_search_failed store=%s: %s", collection, exc)
            continue
        out.extend(Candidate(str(h["id"]), h.get("content") or "", store=label)
                   for h in hits if (h.get("content") or "").strip())

    if profile_store is not None:
        try:
            _, profile = await _profile_owner_and_doc(scope, profile_store)
            if profile:
                out.extend(Candidate(c.memory_id, c.content, store="profile")
                           for c in profile_candidates(profile))
        except Exception as exc:  # noqa: BLE001
            unreadable.append("profile")
            logger.error("forget_copy_search_failed store=profile: %s", exc)

    try:
        out.extend(await _load_candidates(session, scope))
    except Exception as exc:  # noqa: BLE001
        unreadable.append("candidates")
        logger.error("forget_copy_search_failed store=candidates: %s", exc)
    return out, unreadable


async def erase_other_copies(
    session, *, scope, memory_ids: Sequence[str], contents: Dict[str, str],
    copies: Sequence[Candidate], vector=None, profile_store=None,
) -> List[str]:
    """Clear every other copy of the forgotten memories. No judging: ``copies`` are the
    ones already picked, and what each store is told to remove is an exact id.

    Returns the names of the stores that could not be cleared.
    """
    failed: List[str] = []
    ids = list(memory_ids)
    by_store: Dict[str, List[Candidate]] = {}
    for c in copies:
        by_store.setdefault(c.store, []).append(c)

    async def step(label: str, coro_fn):
        try:
            await coro_fn()
        except Exception as exc:  # noqa: BLE001 - one store failing must not stop the rest
            failed.append(label)
            logger.error("forget_copy_cleanup_failed store=%s: %s", label, exc)

    # The source text of what was removed, for the payload pass below. It is the text a
    # fetch of a SURVIVING memory quotes as evidence, so it is not the memory's own.
    removed_texts = [c.content for s in COLLECTION_OF for c in by_store.get(s, [])]

    await step("audit log", lambda: _scrub_actions(session, scope, ids))
    await step("loss manifests", lambda: _scrub_loss_manifests(session, scope, ids))
    extra = {f"removed:{n}": t for n, t in enumerate(dict.fromkeys(removed_texts))}
    await step("saved payloads",
               lambda: _scrub_payloads(session, scope, ids, {**contents, **extra}))
    if by_store.get("candidate"):
        await step("candidates", lambda: _delete_candidates(
            session, scope, [c.memory_id for c in by_store["candidate"]]))
    try:
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        failed.append("database copies")
        logger.error("forget_copy_commit_failed: %s", exc)

    # After the commit, not before it: this step reads the client row on the
    # same session and then writes to the telemetry database, and a failure in
    # that read must not take the scrubs above down with it.
    await step("LLM call traces",
               lambda: _scrub_llm_traces(session, scope, ids, {**contents, **extra}))

    if by_store.get("profile") and profile_store is not None:
        await step("profile", lambda: _strip_profile_store(
            scope, [c.memory_id for c in by_store["profile"]], profile_store))

    for label, collection in COLLECTION_OF.items():
        picked = by_store.get(label)
        if not picked:
            continue
        if vector is None:
            failed.append(collection)
            continue
        await step(collection, lambda c=collection, p=picked: vector.delete_in(
            scope, c, [x.memory_id for x in p]))
    return failed
