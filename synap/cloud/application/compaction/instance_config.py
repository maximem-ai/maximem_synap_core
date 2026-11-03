"""The memory config for compaction and the conversation tracker, where this edition reads a file.

Cloud keeps its config in Postgres and its readers (the compaction manager, the idle
worker, the conversation tracker) load it themselves, as they always have. Core keeps one
YAML file (``synap/cloud/memory_config/file_loader.py``). ``file_config``
returns that file's content as the developer wrote it, and ``None`` when the edition
reads Postgres, so each reader runs its old path unchanged.

``automatic_compaction_enabled`` is the on/off switch for automatic compaction
(``compaction_config.automatic``). It stops compaction only. Promotion to
long-term memory has its own schedule and never calls it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from synap.cloud.shared import edition


async def file_config(instance_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The mounted config file's content (a copy), or ``None`` when this edition reads Postgres."""
    if edition.config_source != "file":
        return None
    from synap.cloud.memory_config.loader import get_config_loader

    return await get_config_loader().get_raw_config(instance_id)


async def automatic_compaction_enabled(instance_id: Optional[str] = None) -> bool:
    """False when the config file turns automatic compaction off. True otherwise.

    Automatic means the four triggers nobody asks for: a recorded message passing its
    size test, the idle sweep, the end of a session, and a user opening a new
    conversation. The compact route and the transcript route are requests and ignore it.
    Cloud reads no file, so for Cloud this is always True.
    """
    raw = await file_config(instance_id)
    if raw is None:
        return True
    return (raw.get("compaction_config") or {}).get("automatic", True) is not False
