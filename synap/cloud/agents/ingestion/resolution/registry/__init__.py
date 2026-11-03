"""Synap's entity registry.

The registry is Synap's: the ``registry_entries`` table, every write to it,
the in-process copy each worker keeps, and how a change on one worker reaches
the others. synap-er, the resolution library, only reads a registry. It says
what a registry has to answer (its ``Registry`` protocol) and ships an
in-memory index that does so; Synap fills that index from its own table and
keeps it current.

    models         the tables: registry_entries, review_queue_entries
    write_through  every change to an entry: database, then index, then
                   a notification to the other workers
    listener       applies those notifications to this worker's index
    types          the entry's enums, from whichever synap-er is installed
"""

from synap.cloud.agents.ingestion.resolution.registry.listener import ChangeListener
from synap.cloud.agents.ingestion.resolution.registry.models import (
    RegistryEntry,
    ReviewQueueEntry,
    ReviewStatus,
)
from synap.cloud.agents.ingestion.resolution.registry.types import EntityType, Scope, Source
from synap.cloud.agents.ingestion.resolution.registry.write_through import WriteThroughManager

__all__ = [
    "ChangeListener",
    "EntityType",
    "RegistryEntry",
    "ReviewQueueEntry",
    "ReviewStatus",
    "Scope",
    "Source",
    "WriteThroughManager",
]
