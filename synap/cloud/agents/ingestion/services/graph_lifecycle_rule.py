"""Whether a graph record (an edge or a node) still states something current.

One pure rule, shared by the graph writer, the repair job and the read path, so
the three can never disagree about what the graph is allowed to serve.

A graph record is *supported* by the memories whose ids it carries in
``memory_ids`` (references mode) or in its ``memory_contents`` entries (copies
mode). Postgres is the authority on whether each of those memories is still
alive; the caller loads one :class:`SupporterRow` per id (see
``synap.cloud.shared.memory_liveness.load_supporter_rows``) and passes the
rows here. An id with no row means the memory was hard-deleted.

The rule (design dossier issue12_graph.md, section 5.1):

* "Said at" of a memory = ``document_created_at``, else ``created_at``.
* *Current support* = alive supporters that are not episodes
  (``is_episode`` false and ``temporal_category != 'episode'``). Episodes are
  dated history: they never keep a present-tense line current.
* **T2** - some supporter S was superseded by a memory N, N is not itself a
  supporter of this record, and no current supporter was said at or after N
  -> Closed, ``valid_to = N.event_start or N said-at``, ``by_memory = N``.
* **T1** - the record once had support and no current support is left
  -> Closed. Reason ``deleted`` when every non-alive supporter is absent or
  ``deleted_at`` (erased), ``superseded`` when one of them was superseded,
  else ``unsupported`` (only episodes or explicitly retired memories left).
* A record that never carried a memory id ("unbacked", e.g. document ABOUT
  edges) is Open (owner decision 2026-11-05: keep serving them).
* The writer and customer nodes ("the user") are always Open.

When T1 and T2 both hold, the T2 result is returned because it carries the
precise end of the state.

Tie rule for T2: a current supporter said at the *same* time as N (same
source document date) keeps the record open. The knowledge-update step saw
both in one document and kept the supporter, so the document re-asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "SupporterRow",
    "GraphRecord",
    "Open",
    "Closed",
    "GraphRecordState",
    "graph_record_state",
    "is_alive",
    "is_current_support",
    "said_at",
    "REASON_DELETED",
    "REASON_SUPERSEDED",
    "REASON_UNSUPPORTED",
    "REASON_UPDATED",
]

REASON_DELETED = "deleted"
REASON_SUPERSEDED = "superseded"
REASON_UNSUPPORTED = "unsupported"
# Not produced by graph_record_state; written by the lifecycle service when a
# memory's text is edited and its old graph support is released.
REASON_UPDATED = "updated"


@dataclass(frozen=True)
class SupporterRow:
    """The Postgres state of one memory that a graph record references.

    Columns of ``memories`` for the memory itself, plus the dates of the
    memory that superseded it (LEFT JOIN on ``superseded_by``), which T2 needs.
    """

    id: str
    superseded_by: Optional[str] = None
    superseded_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    is_episode: bool = False
    temporal_category: Optional[str] = None
    document_created_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    event_start: Optional[datetime] = None
    # The superseding memory's dates (None when not superseded, or when the
    # superseding row is gone).
    superseder_document_created_at: Optional[datetime] = None
    superseder_created_at: Optional[datetime] = None
    superseder_event_start: Optional[datetime] = None

    @property
    def alive(self) -> bool:
        return is_alive(self)

    @property
    def superseder_said_at(self) -> Optional[datetime]:
        return self.superseder_document_created_at or self.superseder_created_at


@dataclass(frozen=True)
class GraphRecord:
    """What the rule needs to know about one edge or node.

    ``memory_ids``: ids currently on the record (``memory_ids`` property, or
    the ids of ``memory_contents`` entries in copies mode).
    ``released_memory_ids``: ids the lifecycle service already removed from
    support (history). They are never support, even when their memory is
    alive, but their rows are judged for T2 and T1 exactly like those of
    ``memory_ids``, they count for "once had support" and for "N is a
    supporter of this record". Callers pass the two lists as stored; they do
    not merge them. ``supporters`` must hold the rows of BOTH lists.
    ``is_fixed_entity``: the writer node or the customer node, always Open.
    """

    memory_ids: Sequence[str] = ()
    released_memory_ids: Sequence[str] = ()
    is_fixed_entity: bool = False


@dataclass(frozen=True)
class Open:
    """The record states something current and may be served.

    ``alive_memory_ids``: supporters alive in Postgres (episodes included);
    these are the ``backing_memory_ids`` exposed on served items.
    ``current_memory_ids``: alive, non-episode supporters.
    ``unbacked``: the record never carried a memory id.
    """

    alive_memory_ids: Tuple[str, ...] = ()
    current_memory_ids: Tuple[str, ...] = ()
    unbacked: bool = False
    is_open: bool = field(default=True, init=False)


@dataclass(frozen=True)
class Closed:
    """The record no longer states something current and must not be served.

    ``rule``: "T1" or "T2". ``reason``: one of REASON_*. ``by_memory``: the
    superseding memory id when known. ``valid_to``: end of the stated state
    when known (T2, or T1 by supersession).
    """

    rule: str
    reason: str
    valid_to: Optional[datetime] = None
    by_memory: Optional[str] = None
    alive_memory_ids: Tuple[str, ...] = ()
    is_open: bool = field(default=False, init=False)


GraphRecordState = Union[Open, Closed]


def is_alive(row: Optional[SupporterRow]) -> bool:
    """Alive = the row exists, is not superseded, retired or forgotten.

    Same predicate as ``synap.cloud.shared.memory_liveness.ALIVE_SQL``.
    """
    if row is None:
        return False
    return row.superseded_by is None and row.deprecated_at is None and row.deleted_at is None


def is_current_support(row: Optional[SupporterRow]) -> bool:
    """Alive and not a dated episode."""
    if not is_alive(row):
        return False
    return not bool(row.is_episode) and (row.temporal_category or "") != "episode"


def said_at(row: SupporterRow) -> Optional[datetime]:
    """When the memory was said: its document's date, else its row's creation."""
    return row.document_created_at or row.created_at


def _cmp_key(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Postgres timestamptz always arrives aware; a naive value is UTC.
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _dedupe(ids: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    out = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return tuple(out)


def graph_record_state(
    record: GraphRecord,
    supporters: Mapping[str, SupporterRow],
) -> GraphRecordState:
    """Decide whether ``record`` is Open or Closed.

    ``supporters`` maps memory id -> its Postgres row, for the ids of
    ``record.memory_ids`` and ``record.released_memory_ids``. A missing id is
    treated as hard-deleted. The function is pure: no I/O, no clock.
    """
    if record.is_fixed_entity:
        return Open(unbacked=not record.memory_ids)

    released = _dedupe(record.released_memory_ids)
    released_set = set(released)
    # An id in both lists has been released: it is no longer support.
    ids = tuple(i for i in _dedupe(record.memory_ids) if i not in released_set)
    all_supporter_ids = set(ids) | released_set

    if not ids and not released:
        return Open(unbacked=True)

    # Support comes only from memory_ids. History (T2, T1 reasons) is judged
    # over memory_ids AND released_memory_ids: a released id's row (who
    # superseded it, when) still decides whether the record ended.
    support_rows = [(i, supporters.get(i)) for i in ids]
    rows = support_rows + [(i, supporters.get(i)) for i in released]
    alive_ids = tuple(i for i, r in support_rows if is_alive(r))
    current = [(i, r) for i, r in support_rows if is_current_support(r)]
    current_ids = tuple(i for i, _ in current)

    # T2: a supersession by a non-supporter, with no current support said at
    # or after the superseding memory.
    t2: Optional[Closed] = None
    t2_key: Optional[float] = None
    for i, r in rows:
        if r is None or r.superseded_by is None:
            continue
        n_id = r.superseded_by
        if n_id in all_supporter_ids:
            continue
        n_said = _cmp_key(r.superseder_said_at)
        if n_said is None:
            # The superseding row is gone or undated: T2 cannot order it.
            continue
        reasserted = any(
            (k := _cmp_key(said_at(cr))) is not None and k >= n_said for _, cr in current
        )
        if reasserted:
            continue
        # Several supersessions can end the record; the latest one defines
        # when the state ended.
        if t2_key is None or n_said > t2_key:
            t2_key = n_said
            t2 = Closed(
                rule="T2",
                reason=REASON_SUPERSEDED,
                valid_to=r.superseder_event_start or r.superseder_said_at,
                by_memory=n_id,
                alive_memory_ids=alive_ids,
            )
    if t2 is not None:
        return t2

    if current_ids:
        return Open(alive_memory_ids=alive_ids, current_memory_ids=current_ids)

    # T1: once supported, no current support left. A released id whose memory
    # is still alive (an edited memory) is neither support nor a cause.
    not_alive = [(i, r) for i, r in rows if not is_alive(r)]
    if not_alive and all(r is None or r.deleted_at is not None for _, r in not_alive):
        return Closed(rule="T1", reason=REASON_DELETED, alive_memory_ids=alive_ids)
    superseded = [r for _, r in not_alive if r is not None and r.superseded_by is not None]
    if superseded:
        latest = max(
            superseded,
            key=lambda r: _cmp_key(r.superseder_said_at) or float("-inf"),
        )
        return Closed(
            rule="T1",
            reason=REASON_SUPERSEDED,
            valid_to=latest.superseder_event_start or latest.superseder_said_at,
            by_memory=latest.superseded_by,
            alive_memory_ids=alive_ids,
        )
    return Closed(rule="T1", reason=REASON_UNSUPPORTED, alive_memory_ids=alive_ids)
