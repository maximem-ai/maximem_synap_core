"""Per-instance cap on context fetches: ``guardrails.rate_limits.max_retrieval_calls_per_minute``.

A route dependency, so it runs before the handler and the retrieval manager
never sees it. Off unless ``SYNAP_GUARDRAIL_RETRIEVAL_RATE`` is on for the
instance; off, nothing is read and no Redis call is made.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from fastapi import Depends, HTTPException, status

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.shared.instance_flags import enabled_for

logger = logging.getLogger(__name__)

FLAG = "SYNAP_GUARDRAIL_RETRIEVAL_RATE"
_WINDOW_SECONDS = 60
# Longer than the window so a slow INCR/EXPIRE pair can never leave the key
# alive for good, short enough that idle instances leave nothing behind.
_KEY_TTL_SECONDS = 2 * _WINDOW_SECONDS

_redis_singleton: Any = None


def _now() -> float:
    return time.time()


async def _get_redis():
    global _redis_singleton
    if _redis_singleton is None:
        import redis.asyncio as aioredis

        _redis_singleton = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
    return _redis_singleton


def _limit_from(config: Any) -> int:
    """The configured ceiling, or 0 for no limit.

    The schema fills in 60 when the developer wrote nothing, so "absent" is read
    from ``model_fields_set``: a limit the developer did not write is not enforced.
    """
    rate_limits = getattr(getattr(config, "guardrails", None), "rate_limits", None)
    if rate_limits is None:
        return 0
    written = getattr(rate_limits, "model_fields_set", None)
    if written is not None and "max_retrieval_calls_per_minute" not in written:
        return 0
    value = getattr(rate_limits, "max_retrieval_calls_per_minute", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return value


async def _count_this_call(instance_id: str, window: int) -> Optional[int]:
    """Calls so far in this minute including this one; None when Redis fails (allow)."""
    try:
        redis = await _get_redis()
        key = f"retrieval_rate:{instance_id}:{window}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, _KEY_TTL_SECONDS)
        count, _ = await pipe.execute()
        return int(count)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrieval_rate_limit_redis_failed instance_id=%s error=%s: allowing the call",
            instance_id, exc,
        )
        return None


async def enforce_retrieval_rate_limit(
    auth: SDKAuthContext = Depends(get_sdk_auth),
) -> None:
    if not enabled_for(FLAG, auth.instance_id):
        return

    try:
        from synap.cloud.memory_config.loader import get_config_loader

        config = await get_config_loader().get_config(str(auth.instance_id))
        limit = _limit_from(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrieval_rate_limit_config_failed instance_id=%s error=%s: allowing the call",
            auth.instance_id, exc,
        )
        return
    if limit <= 0:
        return

    now = _now()
    count = await _count_this_call(str(auth.instance_id), int(now // _WINDOW_SECONDS))
    if count is None or count <= limit:
        return

    retry_after = _WINDOW_SECONDS - int(now % _WINDOW_SECONDS)
    logger.info(
        "retrieval_rate_limited instance_id=%s limit=%d retry_after=%d",
        auth.instance_id, limit, retry_after,
    )
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Retrieval rate limit of {limit} per minute exceeded. "
            f"Retry in {retry_after} seconds."
        ),
        headers={"Retry-After": str(retry_after)},
    )
