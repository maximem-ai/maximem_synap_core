"""The ONE check every instance-creating path goes through.

``InstanceRepository.create_instance`` calls ``assert_can_create_instance`` before
it inserts, so no caller can create a live instance without passing it (the one
deliberate exception is the five system playground agents, which pass
``enforce_limit=False`` and are not counted).

Three limits, checked in this order:

1. session   instances created by ONE onboarding key (one agent-driven session)
2. workspace live instances in the workspace (default, or ``client.instance_limit``)
3. rate      instances created in the last hour by the workspace

Rules:

* Race safe: a transaction-scoped advisory lock on the workspace is taken before
  counting and held until the creating transaction commits or rolls back, so two
  creates at cap-1 are serialised and the second sees the first.
* Blocks NEW creation only. A workspace already above its cap keeps everything.
* "Live" is not deleted (deleted_at / deleted / deletion_state=completed) and not a
  playground agent. Soft-deleted instances do not count.
* Counts only what THIS region's database sees. Instance rows replicate between
  regions, so a workspace that creates in both regions at once can exceed the cap
  by the rows not yet replicated. Accepted.
* An idempotent replay of an already-created instance never reaches this check
  (the replay returns the existing row before any create).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE_LIMIT = 20
DEFAULT_SESSION_INSTANCE_LIMIT = 5
DEFAULT_CREATES_PER_HOUR = 10
RATE_WINDOW_SECONDS = 3600

LIMIT_CODE = "instance_limit_reached"
RATE_CODE = "instance_create_rate_limited"

#: Live = not deleted in any of the three recorded ways, and not a playground agent.
LIVE_INSTANCE_SQL = (
    "(instance_metadata->>'deleted_at') IS NULL "
    "AND COALESCE(instance_metadata->>'deleted', 'false') <> 'true' "
    "AND COALESCE(instance_metadata->>'deletion_state', '') <> 'completed' "
    "AND COALESCE(instance_metadata->>'is_playground', 'false') <> 'true'"
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return value if value >= 0 else default


def default_instance_limit() -> int:
    return _env_int("SYNAP_INSTANCE_LIMIT_DEFAULT", DEFAULT_INSTANCE_LIMIT)


def session_instance_limit() -> int:
    return _env_int("SYNAP_INSTANCE_LIMIT_PER_SESSION", DEFAULT_SESSION_INSTANCE_LIMIT)


def creates_per_hour_limit() -> int:
    return _env_int("SYNAP_INSTANCE_CREATES_PER_HOUR", DEFAULT_CREATES_PER_HOUR)


class InstanceLimitReached(Exception):
    """A create was refused by a limit. ``scope`` is workspace, session or rate."""

    def __init__(self, scope: str, limit: int, current: int, retry_after: Optional[int] = None):
        self.scope = scope
        self.limit = limit
        self.current = current
        self.retry_after = retry_after
        super().__init__(instance_limit_message(scope, limit, current))

    @property
    def code(self) -> str:
        return RATE_CODE if self.scope == "rate" else LIMIT_CODE

    @property
    def status_code(self) -> int:
        return 429 if self.scope == "rate" else 409


def instance_limit_message(scope: str, limit: int, current: int) -> str:
    if scope == "rate":
        return (
            f"This workspace has already created {current} instances in the last hour "
            f"(the limit is {limit} per hour). Stop, and ask the person to try again later. "
            f"To raise the limit, contact support."
        )
    if scope == "session":
        return (
            f"This setup session has already created {current} instances (the limit is "
            f"{limit} per session). Stop and ask the person. To raise it, they can contact "
            f"support, or create further instances in the dashboard themselves."
        )
    return (
        f"This workspace already has {current} instances and the limit is {limit}. "
        f"No instance was created and nothing was changed. Stop and ask the person. "
        f"To raise the limit, contact support."
    )


def instance_limit_body(exc: InstanceLimitReached) -> Dict[str, Any]:
    """The JSON body of the refusal, shared by the management and dashboard routes."""
    body: Dict[str, Any] = {
        "error": exc.code,
        "code": exc.code,
        "message": str(exc),
        "limit": exc.limit,
        "current": exc.current,
        "scope": exc.scope,
        "action": "stop_and_ask_human",
    }
    if exc.retry_after is not None:
        body["retry_after_seconds"] = exc.retry_after
    return body


def instance_limit_response(exc: InstanceLimitReached):
    from fastapi.responses import JSONResponse

    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(status_code=exc.status_code, content=instance_limit_body(exc), headers=headers)


async def effective_instance_limit(session, client_id: str) -> int:
    """The workspace's cap: ``client.instance_limit`` when set, else the default."""
    from synap.cloud.database import schema_compat

    if await schema_compat.has_column("client", "instance_limit", session):
        row = (await session.execute(
            text("SELECT instance_limit FROM client WHERE client_id = :c"), {"c": client_id}
        )).first()
        if row is not None and row.instance_limit is not None:
            return int(row.instance_limit)
    return default_instance_limit()


async def count_live_instances(session, client_id: str) -> int:
    return int((await session.execute(
        text(f"SELECT count(*) FROM instance WHERE client_id = :c AND {LIVE_INSTANCE_SQL}"),
        {"c": client_id},
    )).scalar() or 0)


async def instance_usage(session, client_id: str) -> Dict[str, int]:
    """Read-only ``{"count", "limit"}`` for display (admin view). Takes no lock."""
    return {
        "count": await count_live_instances(session, client_id),
        "limit": await effective_instance_limit(session, client_id),
    }


async def assert_can_create_instance(
    session,
    client_id: str,
    *,
    onboarding_session_key: Optional[str] = None,
) -> None:
    """Raise InstanceLimitReached if one more instance may not be created.

    ``onboarding_session_key`` is the credential id of the onboarding key making
    the request (None for every other caller); it selects the per-session cap.
    On refusal the transaction is rolled back, which releases the lock.
    """
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
        {"k": f"instance_cap:{client_id}"},
    )

    refusal: Optional[InstanceLimitReached] = None

    if onboarding_session_key:
        cap = session_instance_limit()
        made = int((await session.execute(
            text(
                f"SELECT count(*) FROM instance WHERE client_id = :c AND {LIVE_INSTANCE_SQL} "
                "AND instance_metadata->>'created_by' = :k"
            ),
            {"c": client_id, "k": str(onboarding_session_key)},
        )).scalar() or 0)
        if made >= cap:
            refusal = InstanceLimitReached("session", cap, made)

    if refusal is None:
        cap = await effective_instance_limit(session, client_id)
        live = await count_live_instances(session, client_id)
        if live >= cap:
            refusal = InstanceLimitReached("workspace", cap, live)

    if refusal is None:
        cap = creates_per_hour_limit()
        recent = int((await session.execute(
            text(
                "SELECT count(*) FROM instance WHERE client_id = :c "
                "AND COALESCE(instance_metadata->>'is_playground', 'false') <> 'true' "
                "AND created_at > clock_timestamp() - make_interval(secs => :w)"
            ),
            {"c": client_id, "w": RATE_WINDOW_SECONDS},
        )).scalar() or 0)
        if recent >= cap:
            refusal = InstanceLimitReached("rate", cap, recent, retry_after=RATE_WINDOW_SECONDS // 4)

    if refusal is not None:
        await session.rollback()  # release the lock; nothing was written
        logger.warning(
            "instance create refused client_id=%s scope=%s limit=%d current=%d",
            client_id, refusal.scope, refusal.limit, refusal.current,
        )
        raise refusal
