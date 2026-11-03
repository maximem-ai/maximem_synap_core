"""Turn a Cloud-built memory config into one Core's loader accepts.

MACA writes configs for Synap Cloud. Core's loader is strict about a hand-written file
(``core_rules.py``) and rejects the fields of features Core does not have: cross-user grants,
the scope ladder proposal, anticipation, the store choice, and so on. Run on 16 real MACA
configs, the loader rejected every one, 14 to 19 times each. The fields are not wrong for
Cloud; Core simply has nothing that reads them.

``to_core_config`` removes exactly what Core's own rules name as removed, plus keys the schema
does not know (which Cloud drops silently), and returns the list so the developer can see what
was left out. It never touches a value: a ``maca_suggests`` scoping mode, a partial set of
ranking weights or a wrong type is still reported by the loader, because guessing there would
change what the config means.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from . import core_rules

_REMOVED_FIELD = "this field was removed from Core"
_UNKNOWN_KEY = "unknown key "


def _drop(root: Any, path: Tuple[Any, ...]) -> bool:
    cur = root
    for key in path[:-1]:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return False
    last = path[-1]
    if isinstance(cur, dict) and last in cur:
        del cur[last]
        return True
    return False


def to_core_config(raw: Any) -> Tuple[Any, List[str]]:
    """Return ``(config without the fields Core does not read, the dotted paths removed)``.

    The input is not changed. Anything that is not a mapping is returned as it is, for the
    loader to report.
    """
    if not isinstance(raw, dict):
        return raw, []
    cfg: Dict[str, Any] = copy.deepcopy(raw)
    removed: List[str] = []
    for _ in range(8):  # removing a block can expose nothing new, but never loop forever
        changed = False
        for path, message in core_rules.find_problems(cfg):
            if not path:
                continue
            if _REMOVED_FIELD in message or message.startswith(_UNKNOWN_KEY):
                if _drop(cfg, tuple(path)):
                    removed.append(".".join(str(p) for p in path))
                    changed = True
        if not changed:
            break
    return cfg, sorted(set(removed))
