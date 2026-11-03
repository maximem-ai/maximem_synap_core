"""Marks vector writes that are replicas of a write from the other region.

The replication applier writes through the same adapter as everything else,
so the store stays swappable and a move off Chroma is one adapter away rather
than a second HTTP client to port. But its writes must not be recorded in the
vector write log: the log is what the shipper drains, so logging a received
vector would send it straight back where it came from and the two regions
would trade the same vector forever.

Postgres has the identical problem and solves it identically, with
``SET synap.replication_apply = 'on'`` so the outbox trigger ignores the
applier's own writes. This is that flag, for the vector store.

**This module deliberately imports nothing.** It sits between the applier and
the ingestion stack, and both have to be able to import it. Putting the flag
in ``vector_wrapper`` instead pulls that module's whole dependency chain --
langgraph included -- into the replication process, which is a lot of surface
for a boolean, and a lot of ways for an unrelated import error to stop
replication.

A ContextVar rather than an attribute or a global: the adapter is shared and
concurrent, and a flag on the instance would leak across tasks and silently
stop logging real ingestion writes happening at the same moment.
"""

from __future__ import annotations

import contextlib
import contextvars

_REPLICATION_APPLY: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "synap_replication_apply", default=False
)


def is_replication_apply() -> bool:
    """True when the current task is applying a write from the other region."""
    return _REPLICATION_APPLY.get()


@contextlib.contextmanager
def applying_replication():
    """Mark every vector write in this block as a replica, so it is not logged."""
    token = _REPLICATION_APPLY.set(True)
    try:
        yield
    finally:
        _REPLICATION_APPLY.reset(token)
