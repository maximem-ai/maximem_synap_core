"""Server-owned keys of ``instance.instance_metadata``.

Every key starting with ``maca_`` (build marker, owner region, build outcome and
attempts, finish time) is written by the build code only. A caller-supplied
metadata update, whole-bag replace or merge, must neither set, change nor remove
them: wiping ``maca_region`` would leave the instance "unstamped" so either region
could act on it, and wiping ``maca_build_failed`` would hide a failed build.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

RESERVED_PREFIX = "maca_"


def keep_reserved_keys(new_bag: Optional[Dict[str, Any]], stored_bag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``new_bag`` with every reserved key replaced by the stored value (or absent)."""
    out = {k: v for k, v in (new_bag or {}).items() if not str(k).startswith(RESERVED_PREFIX)}
    for k, v in (stored_bag or {}).items():
        if str(k).startswith(RESERVED_PREFIX):
            out[k] = v
    return out
