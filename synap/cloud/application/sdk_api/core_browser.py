"""The Core memory browser: a read-only page and the GET routes behind it.

Served by the Core server at ``/browser``. ``sdk_api/router.py`` includes this
router, and Cloud never includes that router, so Cloud serves none of it.

* Nothing here writes. Every route is a ``GET``, and a test asserts it.
* The API routes depend on ``get_sdk_auth``, as every SDK route does: with
  ``SYNAP_API_TOKEN`` unset they are open, with it set they need the bearer
  token. There is no login of any kind.
* The page's four files are served from a fixed table, never by joining a path
  from the request, so no other file in the folder can be fetched.
* Nothing Cloud only is read: no credits, usage, PII, scope ladder, review queue
  or Cloud config store. The tables are ``memories``, ``memory_actions``,
  ``customer`` and ``"user"``, and the config is the YAML file.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.application.sdk_api.core_identity import load_scoping_mode
from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.database.connection import get_async_session
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/browser", tags=["Memory browser"])

BROWSER_DIR_ENV = "SYNAP_BROWSER_DIR"
DEFAULT_BROWSER_DIR = Path(__file__).resolve().parents[2] / "ui" / "core-browser"

# The only files served. A name that is not a key here is a 404.
PAGE_FILES: Dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "logo.svg": "image/svg+xml",
}

# No inline script, no outside host. Memory text is untrusted.
PAGE_HEADERS: Dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}

MAX_PAGE_SIZE = 100
MAX_CONFIG_BYTES = 256 * 1024
#: The first line of the sample config that ships with Core. A file that still starts with it has not been
#: changed by the developer, and the Config page says so and shows how to make it their own.
STARTER_MARKER = "# Synap Core memory config: the quick-start sample."

# What "active" means: not retired by a knowledge update, and not forgotten.
_ACTIVE = "m.deprecated_at IS NULL AND m.superseded_by IS NULL AND m.deleted_at IS NULL"
_RETIRED = "m.deleted_at IS NULL AND (m.deprecated_at IS NOT NULL OR m.superseded_by IS NOT NULL)"
_NOT_FORGOTTEN = "m.deleted_at IS NULL"
_STATUS_SQL = {"active": _ACTIVE, "retired": _RETIRED, "all": _NOT_FORGOTTEN}

# A memory names its owner by the owner's uuid5 hash (the column `uuid5_hash` of the customer and
# "user" tables), not by the table's own `internal_*_id`. Checked against a real Core database.
_MEMORY_FROM = (
    "FROM memories m "
    "LEFT JOIN customer c ON c.uuid5_hash = m.customer_id "
    'LEFT JOIN "user" u ON u.uuid5_hash = m.user_id '
)

Status = Literal["active", "retired", "all"]
Kind = Literal["customer", "user"]


def _browser_dir() -> Path:
    return Path(os.environ.get(BROWSER_DIR_ENV) or DEFAULT_BROWSER_DIR)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _like(value: str) -> str:
    """A LIKE pattern that matches *value* anywhere, with its wildcards taken literally."""
    escaped = value.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return f"%{escaped}%"


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _scope_of(customer_ext: Optional[str], user_ext: Optional[str]) -> str:
    if user_ext:
        return "user"
    if customer_ext:
        return "customer"
    return "install"


def _memory_row(row: Any) -> Dict[str, Any]:
    retired = row.deprecated_at is not None or row.superseded_by is not None
    return {
        "id": str(row.id),
        "memory_type": row.memory_type,
        "category": row.category,
        "content": row.content,
        "confidence": row.confidence,
        "created_at": _iso(row.created_at),
        "status": "retired" if retired else "active",
        "scope": _scope_of(row.customer_ext, row.user_ext),
        "customer_id": row.customer_ext,
        "user_id": row.user_ext,
    }


def _memory_filters(
    status_value: str, search: Optional[str], entity: Optional[str]
) -> tuple[str, Dict[str, Any]]:
    clauses = [_STATUS_SQL[status_value]]
    params: Dict[str, Any] = {}
    if search and search.strip():
        clauses.append("m.content ILIKE :search ESCAPE '!'")
        params["search"] = _like(search.strip())
    if entity and entity.strip():
        clauses.append("(c.customer_id = :entity OR u.user_id = :entity)")
        params["entity"] = entity.strip()
    return " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

@router.get("", include_in_schema=False)
async def browser_redirect() -> RedirectResponse:
    return RedirectResponse(url="/browser/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


def _page_file(name: str) -> FileResponse:
    media_type = PAGE_FILES.get(name)
    path = _browser_dir() / name if media_type else None
    if media_type is None or path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type=media_type, headers=PAGE_HEADERS)


@router.get("/", include_in_schema=False)
async def browser_index() -> FileResponse:
    return _page_file("index.html")


# ---------------------------------------------------------------------------
# The API. Declared before the file route so "api" is never read as a file name.
# ---------------------------------------------------------------------------

@router.get("/api/overview")
async def overview(
    response: Response,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    mode = await load_scoping_mode(auth.instance_id)
    counts = (
        await session.execute(
            text(
                "SELECT "
                f"(SELECT count(*) FROM memories m WHERE {_ACTIVE}) AS active, "
                f"(SELECT count(*) FROM memories m WHERE {_RETIRED}) AS retired, "
                "(SELECT count(*) FROM customer) AS customers, "
                '(SELECT count(*) FROM "user") AS users'
            )
        )
    ).one()
    return {
        "edition": edition.name,
        "client_id": auth.client_id,
        "instance_id": auth.instance_id,
        "scoping_mode": mode,
        "memories_active": counts.active,
        "memories_retired": counts.retired,
        "customers": counts.customers,
        "users": counts.users,
        # Each user is their own customer in this mode, so there is one list of people.
        "entity_kinds": ["customer"] if mode == "equals_customer" else ["customer", "user"],
    }


@router.get("/api/memories")
async def list_memories(
    response: Response,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=MAX_PAGE_SIZE),
    search: Optional[str] = Query(None, max_length=200),
    type: Optional[str] = Query(None, max_length=50),
    entity: Optional[str] = Query(None, max_length=255, description="A user id or a customer id"),
    status_filter: Status = Query("active", alias="status"),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    where, params = _memory_filters(status_filter, search, entity)

    by_type_rows = (
        await session.execute(
            text(f"SELECT m.memory_type AS mtype, count(*) AS n {_MEMORY_FROM} WHERE {where} GROUP BY m.memory_type ORDER BY n DESC"),
            params,
        )
    ).all()
    by_type = [{"memory_type": r.mtype, "count": r.n} for r in by_type_rows]

    page_where, page_params = where, dict(params)
    if type and type.strip():
        page_where += " AND m.memory_type = :type"
        page_params["type"] = type.strip()
    total = (await session.execute(text(f"SELECT count(*) {_MEMORY_FROM} WHERE {page_where}"), page_params)).scalar_one()

    rows = (
        await session.execute(
            text(
                "SELECT m.id, m.memory_type, m.category, m.content, m.confidence, m.created_at, "
                "m.deprecated_at, m.superseded_by, c.customer_id AS customer_ext, u.user_id AS user_ext "
                f"{_MEMORY_FROM} WHERE {page_where} "
                "ORDER BY m.created_at DESC, m.id LIMIT :limit OFFSET :offset"
            ),
            {**page_params, "limit": page_size, "offset": (page - 1) * page_size},
        )
    ).all()
    return {
        "items": [_memory_row(r) for r in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, -(-total // page_size)),
        "by_type": by_type,
    }


@router.get("/api/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    response: Response,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    try:
        mid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="No such memory") from None

    row = (
        await session.execute(
            text(
                "SELECT m.id, m.memory_type, m.category, m.subcategory, m.content, m.confidence, "
                "m.created_at, m.updated_at, m.deprecated_at, m.superseded_by, m.superseded_at, "
                "m.temporal_category, m.valid_from, m.valid_until, m.document_created_at, "
                "m.source_document_id, m.is_episode, c.customer_id AS customer_ext, u.user_id AS user_ext, "
                "r.content AS replacement_content "
                f"{_MEMORY_FROM} LEFT JOIN memories r ON r.id = m.superseded_by "
                f"WHERE m.id = :id AND {_NOT_FORGOTTEN}"
            ),
            {"id": mid},
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="No such memory")

    history = (
        await session.execute(
            text(
                "SELECT id, action, content, previous_content, replaced_by_memory_id, reason, created_at "
                "FROM memory_actions WHERE memory_id = :id ORDER BY created_at, id"
            ),
            {"id": mid},
        )
    ).all()

    out = _memory_row(row)
    out.update(
        {
            "subcategory": row.subcategory,
            "updated_at": _iso(row.updated_at),
            "superseded_by": str(row.superseded_by) if row.superseded_by else None,
            "superseded_at": _iso(row.superseded_at),
            "deprecated_at": _iso(row.deprecated_at),
            "replacement_content": row.replacement_content,
            "temporal_category": row.temporal_category,
            "valid_from": _iso(row.valid_from),
            "valid_until": _iso(row.valid_until),
            "document_created_at": _iso(row.document_created_at),
            "source_document_id": row.source_document_id,
            "is_episode": bool(row.is_episode),
            "history": [
                {
                    "action": h.action,
                    "content": h.content,
                    "previous_content": h.previous_content,
                    "replaced_by_memory_id": str(h.replaced_by_memory_id) if h.replaced_by_memory_id else None,
                    "reason": h.reason,
                    "created_at": _iso(h.created_at),
                }
                for h in history
            ],
        }
    )
    return out


async def _memory_counts(session: AsyncSession, column: str, ids: List[uuid.UUID]) -> Dict[uuid.UUID, Dict[str, int]]:
    """Active and retired memory counts for each id, in one query for the page."""
    if not ids:
        return {}
    stmt = text(
        f"SELECT m.{column} AS owner, "
        f"count(*) FILTER (WHERE {_ACTIVE}) AS active, "
        f"count(*) FILTER (WHERE {_RETIRED}) AS retired "
        f"FROM memories m WHERE m.{column} IN :ids AND {_NOT_FORGOTTEN} GROUP BY m.{column}"
    ).bindparams(bindparam("ids", expanding=True))
    rows = (await session.execute(stmt, {"ids": ids})).all()
    return {r.owner: {"active": r.active, "retired": r.retired} for r in rows}


@router.get("/api/entities")
async def list_entities(
    response: Response,
    kind: Kind = Query("customer"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=MAX_PAGE_SIZE),
    search: Optional[str] = Query(None, max_length=200),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    params: Dict[str, Any] = {"limit": page_size, "offset": (page - 1) * page_size}
    like_sql = ""
    if search and search.strip():
        params["search"] = _like(search.strip())

    if kind == "customer":
        if "search" in params:
            like_sql = " WHERE (customer_id ILIKE :search ESCAPE '!' OR customer_name ILIKE :search ESCAPE '!')"
        total = (await session.execute(text(f"SELECT count(*) FROM customer{like_sql}"), params)).scalar_one()
        rows = (
            await session.execute(
                text(
                    "SELECT uuid5_hash AS internal_id, customer_id AS external_id, customer_name AS name, "
                    f"customer_short_description AS description, created_at, NULL AS parent FROM customer{like_sql} "
                    "ORDER BY created_at DESC, customer_id LIMIT :limit OFFSET :offset"
                ),
                params,
            )
        ).all()
        counts = await _memory_counts(session, "customer_id", [r.internal_id for r in rows])
    else:
        if "search" in params:
            like_sql = ' WHERE (user_id ILIKE :search ESCAPE \'!\' OR user_name ILIKE :search ESCAPE \'!\')'
        total = (await session.execute(text(f'SELECT count(*) FROM "user"{like_sql}'), params)).scalar_one()
        rows = (
            await session.execute(
                text(
                    "SELECT uuid5_hash AS internal_id, user_id AS external_id, user_name AS name, "
                    f'user_short_description AS description, created_at, customer_id AS parent FROM "user"{like_sql} '
                    "ORDER BY created_at DESC, user_id LIMIT :limit OFFSET :offset"
                ),
                params,
            )
        ).all()
        counts = await _memory_counts(session, "user_id", [r.internal_id for r in rows])

    items = []
    for r in rows:
        n = counts.get(r.internal_id, {"active": 0, "retired": 0})
        items.append(
            {
                "kind": kind,
                "external_id": r.external_id,
                "name": r.name or None,
                "description": r.description or None,
                "parent_customer_id": r.parent,
                "created_at": _iso(r.created_at),
                "memories_active": n["active"],
                "memories_retired": n["retired"],
            }
        )
    return {
        "items": items,
        "kind": kind,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, -(-total // page_size)),
    }


@router.get("/api/entities/{kind}/{external_id:path}")
async def get_entity(
    kind: Kind,
    external_id: str,
    response: Response,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    if kind == "customer":
        row = (
            await session.execute(
                text(
                    "SELECT uuid5_hash AS internal_id, customer_id AS external_id, customer_name AS name, "
                    "customer_short_description AS description, is_org, created_at, updated_at, NULL AS parent "
                    "FROM customer WHERE customer_id = :ext LIMIT 1"
                ),
                {"ext": external_id},
            )
        ).first()
        owner_column = "customer_id"
    else:
        row = (
            await session.execute(
                text(
                    "SELECT uuid5_hash AS internal_id, user_id AS external_id, user_name AS name, "
                    "user_short_description AS description, false AS is_org, created_at, updated_at, "
                    'customer_id AS parent FROM "user" WHERE user_id = :ext LIMIT 1'
                ),
                {"ext": external_id},
            )
        ).first()
        owner_column = "user_id"
    if row is None:
        raise HTTPException(status_code=404, detail="No such customer or user")

    by_type = (
        await session.execute(
            text(
                f"SELECT m.memory_type AS mtype, count(*) AS n FROM memories m WHERE m.{owner_column} = :id AND {_ACTIVE} "
                "GROUP BY m.memory_type ORDER BY n DESC"
            ),
            {"id": row.internal_id},
        )
    ).all()
    counts = (await _memory_counts(session, owner_column, [row.internal_id])).get(row.internal_id, {"active": 0, "retired": 0})
    latest = (
        await session.execute(
            text(
                "SELECT m.id, m.memory_type, m.category, m.content, m.confidence, m.created_at, "
                "m.deprecated_at, m.superseded_by, c.customer_id AS customer_ext, u.user_id AS user_ext "
                f"{_MEMORY_FROM} WHERE m.{owner_column} = :id AND {_ACTIVE} "
                "ORDER BY m.created_at DESC, m.id LIMIT 10"
            ),
            {"id": row.internal_id},
        )
    ).all()

    return {
        "kind": kind,
        "external_id": row.external_id,
        "name": row.name or None,
        "description": row.description or None,
        "is_org": bool(row.is_org),
        "parent_customer_id": row.parent,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "memories_active": counts["active"],
        "memories_retired": counts["retired"],
        "by_type": [{"memory_type": r.mtype, "count": r.n} for r in by_type],
        "latest_memories": [_memory_row(r) for r in latest],
        "profile": await _profile_of(auth, kind, row.external_id, row.parent),
    }


async def _profile_of(auth: SDKAuthContext, kind: str, external_id: str, parent: Optional[str]) -> Optional[Dict[str, Any]]:
    """The stored profile, or None. A profile belongs to a user. With each user
    their own customer, the customer row is that user, so it is asked for as one."""
    try:
        from synap.cloud.application.profile.store import ProfileStore

        mode = await load_scoping_mode(auth.instance_id)
        if kind == "customer" and mode != "equals_customer":
            return None
        return await ProfileStore().read_profile(
            client_id=auth.client_id,
            external_user_id=external_id,
            external_customer_id=parent if kind == "user" else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("browser: no profile for one entity (%s: %s)", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Analytics and requests. Both read what Core already stores: `memories` and `memory_actions`.
# Core records no API request log and no usage, so nothing here is a metering figure.
# ---------------------------------------------------------------------------

_ACTION_FROM = (
    "FROM memory_actions a "
    "LEFT JOIN customer c ON c.uuid5_hash = a.customer_id "
    'LEFT JOIN "user" u ON u.uuid5_hash = a.user_id '
)
_OWNER_EXT = "COALESCE(u.user_id, c.customer_id)"
MAX_ACTIONS_SHOWN = 200
MAX_DAYS = 365


def _utc_days(days: int) -> List[date]:
    today = datetime.now(timezone.utc).date()
    return [today - timedelta(days=n) for n in range(days - 1, -1, -1)]


@router.get("/api/analytics")
async def analytics(
    response: Response,
    days: int = Query(30, ge=1, le=MAX_DAYS),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Counts and trends from the stored memories and the write log. UTC days."""
    _no_store(response)
    window = _utc_days(days)
    since = datetime.combine(window[0], datetime.min.time(), tzinfo=timezone.utc)

    made = (
        await session.execute(
            text(
                "SELECT (m.created_at AT TIME ZONE 'UTC')::date AS day, count(*) AS n FROM memories m "
                f"WHERE {_NOT_FORGOTTEN} AND m.created_at >= :since GROUP BY 1"
            ),
            {"since": since},
        )
    ).all()
    asked = (
        await session.execute(
            text(
                "SELECT (a.created_at AT TIME ZONE 'UTC')::date AS day, count(DISTINCT a.correlation_id) AS n "
                "FROM memory_actions a WHERE a.created_at >= :since GROUP BY 1"
            ),
            {"since": since},
        )
    ).all()
    outcomes = (
        await session.execute(
            text("SELECT a.action AS kind, count(*) AS n FROM memory_actions a WHERE a.created_at >= :since GROUP BY a.action ORDER BY n DESC"),
            {"since": since},
        )
    ).all()
    by_type = (
        await session.execute(
            text(f"SELECT m.memory_type AS mtype, count(*) AS n FROM memories m WHERE {_ACTIVE} GROUP BY m.memory_type ORDER BY n DESC")
        )
    ).all()
    by_scope = (
        await session.execute(
            text(
                "SELECT CASE WHEN u.user_id IS NOT NULL THEN 'user' WHEN c.customer_id IS NOT NULL THEN 'customer' "
                f"ELSE 'install' END AS scope, count(*) AS n {_MEMORY_FROM} WHERE {_ACTIVE} GROUP BY 1 ORDER BY n DESC"
            )
        )
    ).all()
    top = (
        await session.execute(
            text(
                "SELECT COALESCE(u.user_id, c.customer_id) AS ext, "
                "CASE WHEN u.user_id IS NOT NULL THEN 'user' ELSE 'customer' END AS kind, count(*) AS n "
                f"{_MEMORY_FROM} WHERE {_ACTIVE} AND COALESCE(u.user_id, c.customer_id) IS NOT NULL "
                "GROUP BY 1, 2 ORDER BY n DESC, 1 LIMIT 5"
            )
        )
    ).all()
    counts = (
        await session.execute(
            text(
                f"SELECT (SELECT count(*) FROM memories m WHERE {_ACTIVE}) AS active, "
                f"(SELECT count(*) FROM memories m WHERE {_RETIRED}) AS retired, "
                "(SELECT count(DISTINCT correlation_id) FROM memory_actions WHERE created_at >= :since) AS requests"
            ),
            {"since": since},
        )
    ).one()

    made_by_day = {r.day: r.n for r in made}
    asked_by_day = {r.day: r.n for r in asked}
    return {
        "days": days,
        "memories_active": counts.active,
        "memories_retired": counts.retired,
        "memories_created": sum(made_by_day.values()),
        "requests": counts.requests,
        "per_day": [{"day": d.isoformat(), "memories": made_by_day.get(d, 0), "requests": asked_by_day.get(d, 0)} for d in window],
        "outcomes": [{"action": r.kind, "count": r.n} for r in outcomes],
        "by_type": [{"memory_type": r.mtype, "count": r.n} for r in by_type],
        "by_scope": [{"scope": r.scope, "count": r.n} for r in by_scope],
        "top_entities": [{"external_id": r.ext, "kind": r.kind, "memories_active": r.n} for r in top],
    }


async def _request_outcomes(session: AsyncSession, ids: List[str]) -> Dict[str, Dict[str, int]]:
    if not ids:
        return {}
    stmt = text(
        "SELECT a.correlation_id AS rid, a.action AS kind, count(*) AS n FROM memory_actions a "
        "WHERE a.correlation_id IN :ids GROUP BY a.correlation_id, a.action"
    ).bindparams(bindparam("ids", expanding=True))
    out: Dict[str, Dict[str, int]] = {}
    for r in (await session.execute(stmt, {"ids": ids})).all():
        out.setdefault(r.rid, {})[r.kind] = r.n
    return out


@router.get("/api/requests")
async def list_requests(
    response: Response,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=MAX_PAGE_SIZE),
    entity: Optional[str] = Query(None, max_length=255, description="A user id or a customer id"),
    days: Optional[int] = Query(None, ge=1, le=MAX_DAYS),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """Memory writes, one row per request: what each call that stored text did to the memories."""
    _no_store(response)
    clauses, params = ["TRUE"], {}
    if entity and entity.strip():
        clauses.append("(c.customer_id = :entity OR u.user_id = :entity)")
        params["entity"] = entity.strip()
    if days:
        clauses.append("a.created_at >= :since")
        params["since"] = datetime.combine(_utc_days(days)[0], datetime.min.time(), tzinfo=timezone.utc)
    where = " AND ".join(clauses)

    total = (
        await session.execute(text(f"SELECT count(DISTINCT a.correlation_id) {_ACTION_FROM} WHERE {where}"), params)
    ).scalar_one()
    rows = (
        await session.execute(
            text(
                "SELECT a.correlation_id AS rid, min(a.created_at) AS started, count(*) AS actions, "
                f"min({_OWNER_EXT}) AS owner, count(DISTINCT a.customer_id) AS owners "
                f"{_ACTION_FROM} WHERE {where} GROUP BY a.correlation_id "
                "ORDER BY min(a.created_at) DESC, a.correlation_id LIMIT :limit OFFSET :offset"
            ),
            {**params, "limit": page_size, "offset": (page - 1) * page_size},
        )
    ).all()
    outcomes = await _request_outcomes(session, [r.rid for r in rows])
    return {
        "items": [
            {
                "request_id": r.rid,
                "started_at": _iso(r.started),
                "actions": r.actions,
                "entity": r.owner,
                "entities": r.owners,
                "outcomes": outcomes.get(r.rid, {}),
            }
            for r in rows
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, -(-total // page_size)),
    }


@router.get("/api/requests/{request_id}")
async def get_request(
    request_id: str,
    response: Response,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    _no_store(response)
    head = (
        await session.execute(
            text(
                "SELECT min(a.created_at) AS started, max(a.created_at) AS ended, count(*) AS actions, "
                f"min({_OWNER_EXT}) AS owner, count(DISTINCT a.customer_id) AS owners "
                f"{_ACTION_FROM} WHERE a.correlation_id = :rid"
            ),
            {"rid": request_id},
        )
    ).one()
    if not head.actions:
        raise HTTPException(status_code=404, detail="No such request")
    rows = (
        await session.execute(
            text(
                "SELECT a.action AS kind, a.memory_id, a.memory_type, a.content, a.previous_content, a.reason, a.created_at "
                "FROM memory_actions a WHERE a.correlation_id = :rid ORDER BY a.created_at, a.id LIMIT :limit"
            ),
            {"rid": request_id, "limit": MAX_ACTIONS_SHOWN},
        )
    ).all()
    return {
        "request_id": request_id,
        "started_at": _iso(head.started),
        "ended_at": _iso(head.ended),
        "actions": head.actions,
        "entity": head.owner,
        "entities": head.owners,
        "outcomes": (await _request_outcomes(session, [request_id])).get(request_id, {}),
        "items": [
            {
                "action": r.kind,
                "memory_id": str(r.memory_id),
                "memory_type": r.memory_type,
                "content": r.content,
                "previous_content": r.previous_content,
                "reason": r.reason,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ],
        "truncated": head.actions > MAX_ACTIONS_SHOWN,
    }


def _leaves(value: Any) -> int:
    """How many settings a block holds: leaves of nested mappings, items of a list."""
    if isinstance(value, dict):
        return sum(_leaves(v) for v in value.values())
    if isinstance(value, list):
        return len(value)
    return 1


@router.get("/api/config")
async def get_config(
    response: Response,
    auth: SDKAuthContext = Depends(get_sdk_auth),
) -> Dict[str, Any]:
    """The YAML file as the server reads it, whether it is valid, and the samples next to it.
    It is the developer's own file; the Synap Cloud key lives in the environment, not here."""
    _no_store(response)
    from synap.cloud.memory_config.file_loader import MemoryConfigError, parse_and_validate, shared_loader

    path = Path(shared_loader().path)
    samples: List[str] = []
    try:
        samples = sorted(p.name for p in (path.parent / "samples").glob("*.yml"))[:20]
    except OSError:
        pass
    base: Dict[str, Any] = {
        "path": str(path),
        "samples": samples,
    }
    try:
        st = path.stat()
    except OSError:
        return {**base, "exists": False, "starter": False, "text": None, "truncated": False, "size_bytes": None,
                "modified_at": None, "valid": None, "problems": [], "sections": []}
    with path.open("rb") as fh:
        raw_bytes = fh.read(MAX_CONFIG_BYTES + 1)
    truncated = len(raw_bytes) > MAX_CONFIG_BYTES
    body = raw_bytes[:MAX_CONFIG_BYTES].decode("utf-8", errors="replace")

    valid: Optional[bool] = None
    problems: List[str] = []
    sections: List[Dict[str, Any]] = []
    if not truncated:
        try:
            written, _ = parse_and_validate(body, source=path.name)
            valid = True
            sections = [{"name": k, "settings": _leaves(v)} for k, v in written.items() if k != "metadata"]
        except MemoryConfigError as exc:
            valid, problems = False, str(exc).splitlines()
        except Exception as exc:  # noqa: BLE001
            valid, problems = False, [f"{path.name}: {type(exc).__name__}: {exc}"]
    return {
        **base,
        "exists": True,
        "starter": body.lstrip().startswith(STARTER_MARKER),
        "text": body,
        "truncated": truncated,
        "size_bytes": st.st_size,
        "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "valid": valid,
        "problems": problems,
        "sections": sections,
    }


@router.get("/{name}", include_in_schema=False)
async def browser_file(name: str) -> FileResponse:
    return _page_file(name)
