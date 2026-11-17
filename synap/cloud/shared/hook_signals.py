"""Metrics counters and security log lines, through hooks.

Engine files that used to import ``inc_counter``, ``log_rate_limited`` or
``log_pool_timeout`` from the metrics modules import the same names from here
instead. Cloud registers hooks that call the originals with the same arguments; in
Core nothing is registered and each call does nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from synap.cloud.shared import edition

logger = logging.getLogger(__name__)


def _signal(hook: str, *args: Any, **kwargs: Any) -> None:
    """A metric or log line must never break the request or hide the real error."""
    try:
        edition.call(hook, *args, **kwargs)
    except Exception as exc:
        logger.debug("hook %s failed (ignored): %s", hook, exc)


def inc_counter(*args: Any, **kwargs: Any) -> None:
    """The metrics counter hook (name, amount, labels, help_text)."""
    _signal("metrics_counter", *args, **kwargs)


def log_rate_limited(*args: Any, **kwargs: Any) -> None:
    """``observability.request_fields.log_rate_limited``."""
    _signal("security_log", "rate_limited", *args, **kwargs)


def log_pool_timeout(*args: Any, **kwargs: Any) -> None:
    """``observability.request_fields.log_pool_timeout``."""
    _signal("security_log", "pool_timeout", *args, **kwargs)
