"""One retention cycle: find memories past their age limit and delete them.

Expiring a memory is the explicit delete-by-id route, called as a function
(``sdk_memories.delete_memory``). This module holds no deletion SQL and no store
calls of its own for the first attempt, so what retention removes from
Postgres, the vector store and the graph is exactly what an API delete removes.

The one thing it adds is a retry for a half-finished delete. That route removes
the Postgres row first and then reports, in its answer, which store still holds
the record. The row is gone by then, so a "what is still to delete" query cannot
find the record again; the worker remembers it instead (``CleanupLedger``) and
repeats the same two store calls on later cycles. The ledger lives in this
process: a restart forgets it (the failure is logged at error level when it
happens, and again when the retries are given up).

Logs carry counts, ids and exception class names only: a database error message
can quote the row, so it is never logged.
"""

from __future__ import annotations

import logging
import os
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FLAG = "SYNAP_GUARDRAIL_RETENTION"

DEFAULT_BATCH_SIZE = 200
DEFAULT_INTERVAL_SECONDS = 3600
# Per instance and cycle: batch size times this. A first run over years of
# memories then spreads across cycles instead of holding the database.
MAX_BATCHES_PER_CYCLE = 5
# After this many cycles a leftover copy is reported and no longer retried.
MAX_CLEANUP_ATTEMPTS = 5

# The two stores the delete route reports on that a retry can repeat. A leftover
# "source text" is not retried: the route purges it only after counting the other
# memories that rest on the same document, and that row is gone by the retry.
RETRYABLE_STORES = ("search index", "graph")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


def batch_size() -> int:
    return _positive_int_env("SYNAP_RETENTION_BATCH_SIZE", DEFAULT_BATCH_SIZE)


def interval_seconds() -> int:
    return _positive_int_env("SYNAP_RETENTION_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS)


def retention_wanted() -> bool:
    """Whether to start the worker at all. True when the switch is on for the
    fleet or named for some instances; each cycle still asks per instance."""
    from synap.cloud.shared.instance_flags import enabled_for

    return enabled_for(FLAG) or bool(os.environ.get(f"{FLAG}_INSTANCES", "").strip())


@dataclass(frozen=True)
class Candidate:
    memory_id: str
    created_at: datetime
    client_id: str
    customer_id: Optional[str]
    user_id: Optional[str]


@dataclass
class DeleteOutcome:
    """What the delete route answered. ``gone`` is True when there was nothing to delete."""
    still_present_in: List[str] = field(default_factory=list)
    gone: bool = False


@dataclass
class Pending:
    candidate: Candidate
    instance_id: str
    stores: List[str]
    attempts: int = 0


class CleanupLedger:
    """Memories whose Postgres row is gone but whose vector or graph copy is not."""

    def __init__(self) -> None:
        self.items: Dict[str, Pending] = {}


@dataclass
class Deps:
    list_instances: Callable[[], Awaitable[List[str]]]
    get_config: Callable[[str], Awaitable[Any]]
    find_expired: Callable[[str, datetime, int, Optional[Tuple[datetime, str]]],
                           Awaitable[List[Candidate]]]
    delete_by_id: Callable[[str, Candidate], Awaitable[DeleteOutcome]]
    # Returns the stores that still hold the record after another attempt.
    retry_cleanup: Callable[[Pending], Awaitable[List[str]]]
    flag_enabled: Callable[[str], bool]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def _was_set(model: Any, name: str) -> bool:
    return name in (getattr(model, "model_fields_set", None) or ())


def max_age_days(config: Any) -> Optional[int]:
    """The age limit the developer wrote, or None.

    The schema gives ``max_memory_age_days`` a default of 365, so the value alone
    cannot tell "chose a year" from "said nothing". Acting on the default would
    delete every memory after a year for anyone who never wrote the setting. So
    this answers only when each level of the path was set explicitly: pydantic
    records that, and both loaders build the model from the developer's (or the
    stored) dict, which keeps it. A config object that does not carry that record
    is treated as not set.
    """
    guardrails = getattr(config, "guardrails", None)
    retention = getattr(guardrails, "retention", None)
    if not (_was_set(config, "guardrails") and _was_set(guardrails, "retention")
            and _was_set(retention, "max_memory_age_days")):
        return None
    days = getattr(retention, "max_memory_age_days", None)
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        return None
    return days


async def _retry_pending(deps: Deps, ledger: CleanupLedger, instance_id: str) -> Tuple[int, int]:
    retried_ok = gave_up = 0
    for mid, item in list(ledger.items.items()):
        if item.instance_id != instance_id:
            continue
        item.attempts += 1
        try:
            left = await deps.retry_cleanup(item)
        except Exception as exc:  # noqa: BLE001
            left = list(item.stores)
            logger.warning("retention_cleanup_retry_failed memory_id=%s error=%s", mid,
                           type(exc).__name__)
        if not left:
            del ledger.items[mid]
            retried_ok += 1
        elif item.attempts >= MAX_CLEANUP_ATTEMPTS:
            del ledger.items[mid]
            gave_up += 1
            logger.error(
                "retention_cleanup_given_up memory_id=%s still_present_in=%s: the row is "
                "deleted but a copy remains; remove it by hand", mid, ",".join(left))
        else:
            item.stores = left
    return retried_ok, gave_up


async def _sweep_instance(deps: Deps, ledger: CleanupLedger, instance_id: str,
                          size: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"age_days": None, "expired": 0, "failed": 0, "partial": 0,
                               "retried_ok": 0, "gave_up": 0, "capped": False, "skipped": ""}
    # Before the config is read: a leftover copy is owed a retry even if the
    # limit was removed or the config cannot be read this cycle.
    summary["retried_ok"], summary["gave_up"] = await _retry_pending(deps, ledger, instance_id)
    config = await deps.get_config(str(instance_id))
    days = max_age_days(config)
    if days is None:
        summary["skipped"] = "no_age_limit"
        return summary
    summary["age_days"] = days

    cutoff = deps.now() - timedelta(days=days)
    after: Optional[Tuple[datetime, str]] = None
    for batch_no in range(MAX_BATCHES_PER_CYCLE):
        batch = await deps.find_expired(instance_id, cutoff, size, after)
        if not batch:
            return summary
        for cand in batch:
            try:
                outcome = await deps.delete_by_id(instance_id, cand)
            except Exception as exc:  # noqa: BLE001
                # The row is still there, so the next cycle finds it again.
                summary["failed"] += 1
                logger.warning("retention_delete_failed memory_id=%s error=%s", cand.memory_id,
                               type(exc).__name__)
                continue
            if outcome.gone:
                continue
            summary["expired"] += 1
            if outcome.still_present_in:
                summary["partial"] += 1
                retry = [s for s in outcome.still_present_in if s in RETRYABLE_STORES]
                logger.error(
                    "retention_partial_delete memory_id=%s still_present_in=%s",
                    cand.memory_id, ",".join(outcome.still_present_in))
                if retry:
                    ledger.items[cand.memory_id] = Pending(cand, instance_id, retry)
        # A cursor, not an offset: a memory that failed stays in the table and
        # must not be picked up again in this same cycle.
        last = batch[-1]
        after = (last.created_at, last.memory_id)
        if len(batch) < size:
            return summary
    summary["capped"] = True
    return summary


async def run_cycle(deps: Deps, ledger: CleanupLedger,
                    size: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """One pass over every instance the switch is on for. Returns each instance's
    counts; the same counts are logged, one line per instance."""
    size = size or batch_size()
    results: Dict[str, Dict[str, Any]] = {}
    for instance_id in await deps.list_instances():
        if not deps.flag_enabled(instance_id):
            continue
        try:
            summary = await _sweep_instance(deps, ledger, instance_id, size)
        except Exception as exc:  # noqa: BLE001
            logger.error("retention_instance_failed instance=%s error=%s", instance_id,
                         type(exc).__name__)
            summary = {"error": True}
        results[instance_id] = summary
        logger.info("retention_sweep instance=%s %s", instance_id,
                    " ".join(f"{k}={v}" for k, v in summary.items()))
    return results


# --- the real dependencies ---------------------------------------------------


def _real_deps() -> Deps:
    from sqlalchemy import text

    from synap.cloud.database.connection import get_db_session
    from synap.cloud.shared.instance_flags import enabled_for
    from synap.cloud.shared.instance_limits import LIVE_INSTANCE_SQL
    from synap.cloud.shared.scope import parse_or_deterministic_uuid

    async def list_instances() -> List[str]:
        async with get_db_session() as session:
            rows = await session.execute(
                text(f"SELECT instance_id FROM instance WHERE {LIVE_INSTANCE_SQL}"))
            return [str(r[0]) for r in rows.fetchall()]

    async def get_config(instance_id: str) -> Any:
        from synap.cloud.memory_config.loader import get_config_loader

        return await get_config_loader().get_config(instance_id)

    async def find_expired(instance_id, cutoff, limit, after) -> List[Candidate]:
        params: Dict[str, Any] = {"iid": str(parse_or_deterministic_uuid(instance_id)),
                                  "cutoff": cutoff, "n": limit}
        keyset = ""
        if after is not None:
            keyset = ("AND (created_at, id) > (CAST(:after_ts AS timestamptz), "
                      "CAST(:after_id AS uuid))")
            params["after_ts"], params["after_id"] = after
        async with get_db_session() as session:
            rows = await session.execute(text(f"""
                SELECT id, created_at, client_id, customer_id, user_id
                  FROM memories
                 WHERE instance_id = CAST(:iid AS uuid)
                   AND deleted_at IS NULL
                   AND created_at < :cutoff
                   {keyset}
                 ORDER BY created_at, id
                 LIMIT :n
            """), params)
            return [Candidate(str(r[0]), r[1], str(r[2]),
                              str(r[3]) if r[3] else None, str(r[4]) if r[4] else None)
                    for r in rows.fetchall()]

    async def delete_by_id(instance_id: str, cand: Candidate) -> DeleteOutcome:
        from fastapi import HTTPException

        from synap.cloud.application.sdk_api import sdk_memories
        from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext

        # The row's own client uuid goes in as the client id: the route parses a
        # uuid string as itself, so the delete finds the row it was given.
        auth = SDKAuthContext(client_id=cand.client_id, instance_id=instance_id,
                              credential_id=_uuid.uuid4())
        try:
            async with get_db_session() as session:
                answer = await sdk_memories.delete_memory(cand.memory_id, auth, session)
        except HTTPException as exc:
            if exc.status_code == 404:
                return DeleteOutcome(gone=True)
            raise
        return DeleteOutcome(still_present_in=list(answer.still_present_in))

    async def retry_cleanup(item: Pending) -> List[str]:
        from types import SimpleNamespace

        from synap.cloud.application.sdk_api import sdk_memories
        from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext
        from synap.cloud.application.shared.providers import get_cached_providers

        cand = item.candidate
        auth = SDKAuthContext(client_id=cand.client_id, instance_id=item.instance_id,
                              credential_id=_uuid.uuid4())
        scope = sdk_memories._scope_of_stored_row(
            auth, cand.client_id,
            SimpleNamespace(customer_id=cand.customer_id, user_id=cand.user_id))
        providers = get_cached_providers()
        left: List[str] = []
        if "search index" in item.stores:
            try:
                await providers["vector"].delete_memory_vectors(
                    memory_ids=[cand.memory_id], scope=scope)
            except Exception as exc:  # noqa: BLE001
                left.append("search index")
                logger.warning("retention_vector_retry_failed memory_id=%s error=%s",
                               cand.memory_id, type(exc).__name__)
        if "graph" in item.stores:
            try:
                await providers["graph"].soft_delete_memory_from_edges(
                    memory_id=cand.memory_id, scope=scope)
            except Exception as exc:  # noqa: BLE001
                left.append("graph")
                logger.warning("retention_graph_retry_failed memory_id=%s error=%s",
                               cand.memory_id, type(exc).__name__)
        return left

    return Deps(list_instances=list_instances, get_config=get_config,
                find_expired=find_expired, delete_by_id=delete_by_id,
                retry_cleanup=retry_cleanup,
                flag_enabled=lambda iid: enabled_for(FLAG, iid))
