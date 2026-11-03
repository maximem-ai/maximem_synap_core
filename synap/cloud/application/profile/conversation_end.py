"""Core's user-profile update, run when a conversation ends or goes idle.

Cloud updates a user's profile inside the consolidation cycle (the "nap"). Core has
no such cycle, so when a conversation ends or goes idle, long-term promotion calls
``spawn_on_conversation_end`` and this module runs the SAME merge,
``profile.updater.merge_profile``, once for that user.

Which of the two happens is decided by one plain edition value,
``edition.profile_at_conversation_end``, read in ``conversation/promotion.py``.
Nothing here looks at the edition.

The watermark
-------------
"New since the last update" is a watermark stored in the profile itself, as
``_meta.last_memory_at``: the ``created_at`` of the newest memory the last merge
saw. A run reads the memories created after it, merges them, and writes the new
watermark in the same locked write as the profile. So:

* a second end of the same conversation finds no newer memory and does nothing;
* two ends at once for one user run one after the other (an in-process lock), and
  across processes the store's row lock plus the watermark check inside the write
  drop the loser;
* a failed merge writes nothing, so the same memories are tried again next time.

It is not the nap's mark. The nap uses ``promoted_at >= run start`` and a run id;
Core has no run, and a conversation end can arrive twice.

Limit, stated plainly: a memory whose ``created_at`` is older than the watermark but
which was committed after the merge read the table would be missed. Stage 4 stamps
``created_at`` at write time and the trigger first waits for the batch just sent to
finish, so this needs two conversations of one user ending inside the same few
seconds.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy import text

from synap.cloud.application.profile.updater import (
    _MAX_MEMORIES_IN_PROMPT,
    _as_datetime,
    memory_lines_can_change_profile,
    merge_profile,
    warn_owner_not_found,
)

logger = logging.getLogger(__name__)

_TERMINAL = {"completed", "failed", "partial_success"}
_WAIT_SECONDS_VAR = "SYNAP_PROFILE_WAIT_SECONDS"
_DEFAULT_WAIT_SECONDS = 300.0
_POLL_SECONDS = 1.0

_COMMON = """
FROM memories
WHERE client_id = CAST(:cl AS uuid) AND customer_id = CAST(:cu AS uuid)
  AND user_id IS NOT DISTINCT FROM CAST(:u AS uuid)
  AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL
"""
_SELECT = "SELECT memory_type, content, created_at "
_NEW_SINCE_SQL = _SELECT + _COMMON + "  AND created_at > :since\nORDER BY created_at DESC\nLIMIT :lim"
_ALL_SQL = _SELECT + _COMMON + "ORDER BY created_at DESC\nLIMIT :lim"

# One merge at a time per user inside this process.
_user_locks: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
# Conversations this process has already settled, so an idle sweep that comes round
# again with nothing new to send does not repeat the lookups. Bounded.
_settled: "OrderedDict[str, None]" = OrderedDict()
_BOUND = 2048
_tasks: Set["asyncio.Task[Any]"] = set()


def _lock_for(key: str) -> asyncio.Lock:
    lock = _user_locks.get(key)
    if lock is None:
        lock = _user_locks[key] = asyncio.Lock()
        while len(_user_locks) > _BOUND:
            _user_locks.popitem(last=False)
    else:
        _user_locks.move_to_end(key)
    return lock


def _remember(conversation_id: str) -> None:
    _settled[conversation_id] = None
    _settled.move_to_end(conversation_id)
    while len(_settled) > _BOUND:
        _settled.popitem(last=False)


async def fetch_new_memories(*, scope, since, session_factory) -> Tuple[List[str], Optional[Any]]:
    """The user's live memories created after *since*, newest first, as
    ``"[type] content"`` lines, and the newest ``created_at`` among them.

    Same scope filter and line format as the nap's read, capped at the same
    number of memories the merge prompt takes.
    """
    params: Dict[str, Any] = {
        "cl": str(scope.client_id),
        "cu": str(scope.customer_id) if scope.customer_id else None,
        "u": str(scope.user_id) if scope.user_id else None,
        "lim": _MAX_MEMORIES_IN_PROMPT,
    }
    sql = _ALL_SQL
    if since is not None:
        params["since"] = since
        sql = _NEW_SINCE_SQL
    async with session_factory() as session:
        rows = (await session.execute(text(sql), params)).fetchall()
    lines: List[str] = []
    newest = None
    for memory_type, content, created_at in rows:
        if created_at is not None and (newest is None or created_at > newest):
            newest = created_at
        content = (content or "").strip()
        if content:
            lines.append(f"[{memory_type or 'memory'}] {content}")
    return lines, newest


async def run_for_user(
    *,
    scope,
    conversation_id: Optional[str],
    config_loader,
    store,
    fetch: Callable[..., Awaitable[Tuple[List[str], Optional[Any]]]],
    complete: Callable[..., Awaitable[Any]],
) -> str:
    """Merge the memories created since this user's last profile update, once.

    Returns the merge result (``updated`` or ``skipped:<reason>``). Never raises.
    """
    try:
        return await _run_for_user(
            scope=scope, conversation_id=conversation_id, config_loader=config_loader,
            store=store, fetch=fetch, complete=complete,
        )
    except Exception as exc:  # noqa: BLE001 -- a profile failure never fails a conversation
        logger.warning(
            "profile_conversation_end_error conversation_id=%s error=%s",
            conversation_id, exc, exc_info=True,
        )
        return f"error:{type(exc).__name__}"


async def _run_for_user(*, scope, conversation_id, config_loader, store, fetch, complete) -> str:
    if not scope.user_id and not scope.customer_id:
        return "skipped:no_user_scope"

    config = await config_loader.get_config(instance_id=str(scope.instance_id))
    up_cfg = getattr(config, "user_profile", None)
    if up_cfg is None or not up_cfg.enabled:
        return "skipped:disabled"

    key = f"{scope.client_id}:{scope.customer_id}:{scope.user_id}"
    async with _lock_for(key):
        owner = await store.resolve_owner_by_scope_uuids(
            client_uuid=scope.client_id, user_uuid=scope.user_id, customer_uuid=scope.customer_id,
        )
        log_ref = f"conversation_id={conversation_id}"
        if owner is None:
            warn_owner_not_found(scope, log_ref=log_ref)
            return "skipped:owner_not_found"

        current = await store.read(owner)
        stored = ((current or {}).get("_meta") or {}).get("last_memory_at")
        since = _as_datetime(stored)

        lines, newest = await fetch(scope=scope, since=since)
        if not lines or newest is None:
            return "skipped:no_new_memories"
        if not memory_lines_can_change_profile(lines):
            return "skipped:not_profile_relevant"

        watermark = newest.isoformat() if hasattr(newest, "isoformat") else str(newest)
        result, _ = await merge_profile(
            owner=owner, up_cfg=up_cfg, current=current, memory_lines=lines,
            analysis=None, conversation_id=conversation_id,
            idempotency_key=f"conversation_end:{watermark}",
            written_by="conversation_end", complete=complete, store=store,
            log_ref=log_ref, watermark=watermark,
        )
        return result


def _wait_seconds() -> float:
    try:
        return float(os.environ.get(_WAIT_SECONDS_VAR, _DEFAULT_WAIT_SECONDS))
    except ValueError:
        return _DEFAULT_WAIT_SECONDS


async def wait_for_ingestion(
    ingestion, ingestion_id, *, timeout: Optional[float] = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> bool:
    """Wait until the batch just sent has been through ingestion (its memories exist).

    True when it reached a final status, False on timeout or when it cannot be
    looked up; the caller merges what exists either way, and the rest is picked up
    by the next end because of the watermark.
    """
    if ingestion is None or ingestion_id is None:
        return True
    limit = _wait_seconds() if timeout is None else timeout
    waited = 0.0
    while True:
        try:
            status = await ingestion.get_status(ingestion_id)
        except Exception:  # noqa: BLE001
            logger.debug("profile wait: status lookup failed", exc_info=True)
            return False
        value = getattr(getattr(status, "status", None), "value", getattr(status, "status", None))
        if status is None or str(value).lower() in _TERMINAL:
            return True
        if waited >= limit:
            return False
        await sleep(_POLL_SECONDS)
        waited += _POLL_SECONDS


async def _merge_for_scope(scope, conversation_id: Optional[str], label: Optional[str] = None) -> str:
    """Wire the real store, config loader, model and database, then run the merge once
    for the user *scope* names. Never raises. *label* names the cause in logs when there
    is no conversation."""
    from synap.cloud.application.ingestion.startup import _get_llm_client
    from synap.cloud.application.profile.store import ProfileStore
    from synap.cloud.database.connection import get_session_factory
    from synap.cloud.memory_config.loader import get_config_loader

    ref = conversation_id or label or "unknown"
    llm = _get_llm_client("ingestion_agent_light")

    async def complete(*, prompt: str, max_tokens: int) -> Any:
        return await llm.complete(
            prompt=prompt, json_mode=True, telemetry_scope=scope,
            telemetry_operation="ingestion.profile_merge",
            telemetry_correlation_id=f"profile_conversation_end:{ref}",
            max_tokens=max_tokens, temperature=0.0,
        )

    session_factory = get_session_factory()

    async def fetch(*, scope, since):
        return await fetch_new_memories(scope=scope, since=since, session_factory=session_factory)

    return await run_for_user(
        scope=scope, conversation_id=conversation_id, config_loader=get_config_loader(),
        store=ProfileStore(), fetch=fetch, complete=complete,
    )


async def on_memory_created(*, scope, ingestion=None, ingestion_id=None) -> str:
    """The trigger for memories that arrive without a conversation (``memories.create``).

    Waits for the batch just queued to finish ingestion, then runs the same merge as
    the end of a conversation, once for the user in *scope*, over the memories created
    since their last profile update. Never raises: a profile failure, or a user with no
    row to hold a profile, is logged and skipped and never touches the ingestion.
    """
    try:
        await wait_for_ingestion(ingestion, ingestion_id)
        result = await _merge_for_scope(scope, None, label=f"ingestion:{ingestion_id}")
        logger.info("profile_memory_created result=%s ingestion_id=%s", result, ingestion_id)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("profile_memory_created_error ingestion_id=%s error=%s",
                       ingestion_id, exc, exc_info=True)
        return f"error:{type(exc).__name__}"


async def on_conversation_end(
    *,
    conversation_id: str,
    instance_id: str,
    user_id: Optional[str],
    customer_id: Optional[str],
    client_id: Optional[str],
    scope_path: Optional[Dict[str, str]] = None,
    ingestion=None,
    ingestion_id=None,
    sent: int = 0,
) -> str:
    """The trigger. Wires the real store, config loader, model and database, then
    calls ``run_for_user``. Never raises."""
    try:
        if not sent and not ingestion_id and conversation_id in _settled:
            return "skipped:already_settled"
        await wait_for_ingestion(ingestion, ingestion_id)

        from synap.cloud.application.shared.providers import resolve_compaction_promotion_scope

        event = SimpleNamespace(
            conversation_id=conversation_id, instance_id=instance_id, user_id=user_id,
            customer_id=customer_id, client_id=client_id, scope_path=scope_path,
            trigger_type=None,
        )
        scope = await resolve_compaction_promotion_scope(event)
        if scope is None:
            return "skipped:no_scope"

        result = await _merge_for_scope(scope, conversation_id)
        if not result.startswith("error:"):
            _remember(conversation_id)
        logger.info("profile_conversation_end result=%s conversation_id=%s", result, conversation_id)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("profile_conversation_end_error conversation_id=%s error=%s",
                       conversation_id, exc, exc_info=True)
        return f"error:{type(exc).__name__}"


def spawn_on_conversation_end(**kwargs: Any) -> Optional["asyncio.Task[Any]"]:
    """Start the trigger in the background and return at once. Never raises."""
    try:
        task = asyncio.get_running_loop().create_task(on_conversation_end(**kwargs))
    except RuntimeError:
        return None
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def spawn_on_memory_created(**kwargs: Any) -> Optional["asyncio.Task[Any]"]:
    """Start ``on_memory_created`` in the background and return at once. Never raises."""
    try:
        task = asyncio.get_running_loop().create_task(on_memory_created(**kwargs))
    except RuntimeError:
        return None
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
