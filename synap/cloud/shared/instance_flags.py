"""A feature flag that can be on for the fleet or for named instances.

    SYNAP_GRAPH_PER_SCOPE=true                    every instance
    SYNAP_GRAPH_PER_SCOPE_INSTANCES=inst_a,inst_b those instances only ("*" = all)

The same idiom as GRAPH_EXTRACTION_INSTANCE_ALLOWLIST and
STAGE3_EXTRACTION_MODE_BY_INSTANCE: an instance is named by either of its id
forms. A B2C and a B2B instance want different graph settings, and a flag that
changes where records are stored has to give the write and the read of one
instance the same answer, which the instance id does.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Set

_TRUTHY = {"1", "true", "yes", "on"}


def _named(raw: str) -> Set[str]:
    out: Set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(item.lower())
        try:
            from synap.cloud.shared.scope import parse_or_deterministic_uuid
            out.add(str(parse_or_deterministic_uuid(item)).lower())
        except Exception:  # noqa: BLE001
            pass
    return out


def enabled_for(flag_env: str, instance_id: Optional[Any] = None) -> bool:
    """True when *flag_env* is set, or *instance_id* is listed in
    ``<flag_env>_INSTANCES``."""
    if os.environ.get(flag_env, "").strip().lower() in _TRUTHY:
        return True
    raw = os.environ.get(f"{flag_env}_INSTANCES", "")
    if not raw.strip():
        return False
    named = _named(raw)
    if "*" in named:
        return True
    return instance_id is not None and str(instance_id).strip().lower() in named
