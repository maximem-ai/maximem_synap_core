"""Tolerate a database that has not run the newest migrations (166-174) yet.

Migrations are applied BY HAND per region and the deploy scripts do not run
them, so new code can start against an older schema. The rule: a missing table
or column may only degrade the NEW feature, never an existing customer path.

Three small tools:

* ``warn_once(key, msg)``      one WARNING per process per key (no log floods).
* ``has_column`` / ``has_table``  a catalogue probe, cached per process. A
  positive answer is kept for good (a column is never dropped under us); a
  negative one for ``NEGATIVE_TTL_SECONDS`` so a migration applied by hand is
  noticed without a restart. A probe that itself fails answers False and is not
  cached.
* ``is_missing_schema_error``  recognises "undefined table / column" errors so
  a caller can treat them as "feature unavailable" instead of a failure.

``supervision_available()`` is the single question the build code asks: are
``maca_build`` (168), ``maca_build.task_started_at`` (171) and
``maca_config.build_id`` (168) all present?
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: How long "this object is missing" is believed before the catalogue is asked again.
NEGATIVE_TTL_SECONDS = 60.0

_warned: set = set()
# key -> (exists, checked_monotonic)
_probe_cache: Dict[Tuple[str, str], Tuple[bool, float]] = {}

# SQLSTATE 42P01 undefined_table, 42703 undefined_column.
_MISSING_SQLSTATES = {"42P01", "42703"}


def reset_for_tests() -> None:
    """Forget every cached probe and every "already warned" key."""
    _warned.clear()
    _probe_cache.clear()


def warn_once(key: str, message: str, *args) -> bool:
    """Log ``message`` at WARNING the first time ``key`` is seen in this process."""
    if key in _warned:
        return False
    _warned.add(key)
    logger.warning(message, *args)
    return True


def is_missing_schema_error(exc: BaseException) -> bool:
    """True for an undefined table / undefined column error (any driver wrapper)."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        code = getattr(cur, "sqlstate", None) or getattr(cur, "pgcode", None)
        if code in _MISSING_SQLSTATES:
            return True
        if type(cur).__name__ in ("UndefinedTableError", "UndefinedColumnError", "UndefinedTable", "UndefinedColumn"):
            return True
        cur = getattr(cur, "orig", None) or cur.__cause__ or cur.__context__
    return False


async def _probe(sql: str, params: dict, session=None) -> Optional[bool]:
    try:
        if session is not None:
            return bool((await session.execute(text(sql), params)).scalar())
        from synap.cloud.database.connection import get_session_factory

        async with get_session_factory()() as own:
            return bool((await own.execute(text(sql), params)).scalar())
    except Exception as exc:  # noqa: BLE001 - a failed probe must not break the caller
        logger.debug("schema probe failed (%s)", exc)
        return None


async def _cached(key: Tuple[str, str], sql: str, params: dict, session=None) -> bool:
    now = time.monotonic()
    hit = _probe_cache.get(key)
    if hit is not None and (hit[0] or now - hit[1] < NEGATIVE_TTL_SECONDS):
        return hit[0]
    answer = await _probe(sql, params, session)
    if answer is None:
        return False
    _probe_cache[key] = (answer, now)
    return answer


async def has_table(table: str, session=None) -> bool:
    """Does ``table`` exist on this connection's search_path?"""
    return await _cached(
        ("table", table), "SELECT to_regclass(:t) IS NOT NULL", {"t": table}, session
    )


async def has_column(table: str, column: str, session=None) -> bool:
    """Does ``table.column`` exist on this connection's search_path?"""
    return await _cached(
        (table, column),
        "SELECT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = to_regclass(:t) "
        "AND attname = :c AND NOT attisdropped AND attnum > 0)",
        {"t": table, "c": column},
        session,
    )


async def supervision_available(session=None) -> bool:
    """True when every object the supervised build path needs exists (168 + 171)."""
    ok = (
        await has_column("maca_build", "task_started_at", session)
        and await has_column("maca_config", "build_id", session)
    )
    if not ok:
        warn_once(
            "supervision_unavailable",
            "supervised MACA builds are unavailable: maca_build / maca_build.task_started_at / "
            "maca_config.build_id are missing (migrations 168 and 171 not applied in this "
            "database). New instances use the legacy unsupervised build; the sweeper idles. "
            "Re-checked every %ds.", int(NEGATIVE_TTL_SECONDS),
        )
    return ok
