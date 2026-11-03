"""A waiting forget is run the moment the batch it was about has been stored."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.memory_forget import marker, retry
from synap.cloud.application.memory_forget import requests_store as store
from synap.cloud.application.memory_forget import retry_store as rstore
from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeUnavailable
from synap.cloud.application.memory_forget.models import Candidate
from synap.cloud.application.memory_forget.soft_delete import SoftDeleteResult
from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

PHONE = Candidate("11111111-1111-1111-1111-111111111111", "The user's contact number is 512-555-0142", "facts")
EMAIL = Candidate("22222222-2222-2222-2222-222222222222", "The user's email is a@b.example", "facts")
CONV = coerce_conversation_id("pgv2_abc123")


def _scope():
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id=uuid.uuid4(),
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4(), scope_level="user")


def _batch_meta(first=61, last=65, conv=CONV):
    return {"conversation_id": conv, "source_message_range": {"from_position": first, "to_position": last}}


def _marker(query="my contact number", position=63):
    return {"id": str(uuid.uuid4()), "query": query, "requested_by": "sdk:test",
            "request_position": position, "attempt": 0}


class _Savepoint:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Ctx:
    async def __aenter__(self):
        return SimpleNamespace(begin_nested=lambda: _Savepoint())

    async def __aexit__(self, *exc):
        return False


class World:
    """The stores, faked: what is waiting, what the batch stored, and what was done."""

    def __init__(self, markers=(), notes=(), flag=None, judge_raises=False, close_wins=True):
        self.markers, self.notes, self.flag = list(markers), list(notes), flag
        self.judge_raises, self.close_wins = judge_raises, close_wins
        self.marker_queries, self.note_reads = [], 0
        self.judge_calls, self.deleted, self.closed, self.recorded = [], [], [], []
        self.covered = []

    async def markers_for_batch(self, session, *, scope, conversation_id, first_position, window_seconds):
        self.marker_queries.append((conversation_id, first_position, window_seconds))
        return list(self.markers)

    async def load_notes_by_id(self, session, *, scope, memory_ids):
        self.note_reads += 1
        return [n for n in self.notes if n.memory_id in set(memory_ids)]

    async def mark_covered(self, session, *, retry_ids):
        self.covered.extend(retry_ids)

    async def close_if_waiting(self, session, *, retry_id, status, attempt, result):
        self.closed.append((retry_id, status, result))
        return self.close_wins

    async def record_direct(self, session, *, scope, query, candidates, requested_by, status, result):
        self.recorded.append((query, [c.memory_id for c in candidates], requested_by))
        return "r"

    def deps(self):
        async def judge(desc, cands):
            self.judge_calls.append((desc, [c.memory_id for c in cands]))
            if self.judge_raises:
                raise JudgeUnavailable("down")
            return [c for c in cands if self.flag is None or c.memory_id in self.flag]

        async def delete(scope, picked, reason):
            self.deleted.append(([c.memory_id for c in picked], reason))
            return SoftDeleteResult(deleted_ids=[c.memory_id for c in picked])

        return service.ForgetDeps(gather=None, judge=judge, delete=delete)


@pytest.fixture
def world(monkeypatch):
    def make(**kw):
        w = World(**kw)
        for name in ("markers_for_batch", "load_notes_by_id", "close_if_waiting", "mark_covered"):
            monkeypatch.setattr(rstore, name, getattr(w, name))
        monkeypatch.setattr(store, "record_direct", w.record_direct)
        monkeypatch.setattr(service, "build_deps", lambda session, scope, correlation_id=None: w.deps())
        monkeypatch.setattr("synap.cloud.database.connection.get_db_session", lambda: Ctx())
        return w
    monkeypatch.delenv("SYNAP_FORGET_RETRY", raising=False)
    monkeypatch.delenv("SYNAP_FORGET_RETRY_NOTE_WINDOW_SECONDS", raising=False)
    return make


async def _run(w, ids=(PHONE.memory_id, EMAIL.memory_id), meta=None, scope=None):
    return await marker.after_persist(scope=scope or _scope(), metadata=meta if meta is not None else _batch_meta(),
                                      memory_ids=list(ids))


async def test_a_waiting_request_removes_the_matching_note_when_its_batch_is_stored(world):
    m = _marker()
    w = world(markers=[m], notes=[PHONE, EMAIL], flag={PHONE.memory_id})
    removed = await _run(w)
    assert removed == 1
    assert w.judge_calls == [("my contact number", [PHONE.memory_id, EMAIL.memory_id])]
    assert w.deleted == [([PHONE.memory_id], marker.REASON_MARKER)]      # only the flagged one
    assert w.closed and w.closed[0][0] == m["id"] and w.closed[0][1] == "done"
    assert w.recorded == [("my contact number", [PHONE.memory_id], "sdk:test")]


async def test_the_lookup_uses_the_normalised_conversation_and_the_batch_start(world):
    w = world(markers=[], notes=[PHONE])
    await _run(w, meta=_batch_meta(first=66, last=70, conv="pgv2_abc123"))
    # an un-normalised id from the batch is normalised the same way the message table is
    assert w.marker_queries == [(CONV, 66, retry.note_window_seconds())]


async def test_no_waiting_request_means_no_judge_call_and_no_note_read(world):
    w = world(markers=[], notes=[PHONE, EMAIL])
    assert await _run(w) == 0
    assert w.judge_calls == [] and w.note_reads == 0 and w.deleted == []


@pytest.mark.parametrize("meta", [
    None, {}, {"conversation_id": CONV},                      # not a conversation batch
    {"source_message_range": {"from_position": 1}},           # no conversation
])
async def test_a_document_that_is_not_a_conversation_batch_is_ignored(world, meta):
    w = world(markers=[_marker()], notes=[PHONE])
    assert await _run(w, meta=meta if meta is not None else {}) == 0
    assert w.marker_queries == []


async def test_a_batch_that_stored_nothing_still_covers_the_request_it_held(world):
    """The batch holding the request stored no notes, so nothing was judged,
    and the request was never marked covered. It must be, or the timed retries keep reading
    notes from later batches."""
    m = _marker(position=63)
    w = world(markers=[m], notes=[PHONE])
    assert await _run(w, ids=[], meta=_batch_meta(first=61, last=65)) == 0
    assert w.judge_calls == [] and w.deleted == []       # no note, no model call
    assert w.covered == [m["id"]]


async def test_a_request_is_covered_even_when_an_earlier_request_used_up_the_notes(world):
    first, second = _marker("my contact number", position=63), _marker("my phone", position=63)
    w = world(markers=[first, second], notes=[PHONE], flag={PHONE.memory_id})
    assert await _run(w, ids=[PHONE.memory_id], meta=_batch_meta(first=61, last=65)) == 1
    assert [c[0] for c in w.closed] == [first["id"]]     # the first one removed it
    assert w.covered == [second["id"]]                   # the second still held by this batch


async def test_nothing_runs_when_the_switch_is_off(world, monkeypatch):
    w = world(markers=[_marker()], notes=[PHONE])
    monkeypatch.setenv("SYNAP_FORGET_RETRY", "false")
    assert await _run(w) == 0
    assert w.marker_queries == []


async def test_a_judge_outage_leaves_the_request_waiting_and_never_raises(world):
    w = world(markers=[_marker()], notes=[PHONE], judge_raises=True)
    assert await _run(w) == 0
    assert w.deleted == [] and w.closed == []         # the timed retries are still queued


async def test_a_failing_store_never_breaks_the_batch(world, monkeypatch):
    w = world(markers=[_marker()], notes=[PHONE])

    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(rstore, "markers_for_batch", boom)
    assert await _run(w) == 0


async def test_notes_the_judge_does_not_flag_are_left_alone(world):
    m = _marker()
    w = world(markers=[m], notes=[EMAIL], flag=set())
    assert await _run(w, ids=[EMAIL.memory_id]) == 0
    assert w.deleted == [] and w.closed == []         # still waiting for a later batch


async def test_two_requests_do_not_re_judge_a_note_the_first_one_removed(world):
    w = world(markers=[_marker("my contact number"), _marker("my phone")],
              notes=[PHONE, EMAIL], flag={PHONE.memory_id})
    assert await _run(w) == 1
    assert w.judge_calls[0][1] == [PHONE.memory_id, EMAIL.memory_id]
    assert w.judge_calls[1][1] == [EMAIL.memory_id]


async def test_the_dashboard_row_is_filed_once_when_another_run_closed_the_request(world):
    w = world(markers=[_marker()], notes=[PHONE], close_wins=False)
    assert await _run(w) == 1          # the note is removed either way
    assert w.recorded == []            # but only the run that closed the row records it


async def test_a_scope_without_a_user_is_ignored(world):
    w = world(markers=[_marker()], notes=[PHONE])
    scope = _scope()
    scope.user_id = None
    assert await _run(w, scope=scope) == 0 and w.marker_queries == []


# ------------------------------------------------------------------ what the request stores

async def test_direct_hands_the_conversation_id_to_the_queue(monkeypatch):
    seen = {}

    async def enqueue(session, *, scope, query, requested_by, first_attempt_in_seconds,
                      conversation_id=None, unsent_wait_seconds=None):
        seen.update(conversation_id=conversation_id, query=query)
        return "id"

    class Sess:
        async def commit(self):
            pass

        def begin_nested(self):
            return _Savepoint()

    monkeypatch.setattr(rstore, "enqueue", enqueue)
    monkeypatch.delenv("SYNAP_FORGET_RETRY", raising=False)

    async def gather(desc):
        return service.Gathered()

    deps = service.ForgetDeps(gather=gather, judge=None, delete=None)
    out = await service.direct(Sess(), scope=_scope(), description="my contact number", deps=deps,
                               requested_by="sdk:t", conversation_id=CONV)
    assert out.retry_scheduled is True
    assert seen == {"conversation_id": CONV, "query": "my contact number"}


# ------------------------------------------------------------------ "covered": where the timers stop

async def test_a_batch_that_holds_the_request_marks_it_covered_when_nothing_matched(world):
    m = _marker(position=63)                     # asked at message 63
    w = world(markers=[m], notes=[EMAIL], flag=set())
    await _run(w, ids=[EMAIL.memory_id], meta=_batch_meta(first=61, last=65))   # 61..65 holds 63
    assert w.covered == [m["id"]]


async def test_an_earlier_batch_does_not_cover_a_request_made_after_it(world):
    m = _marker(position=70)                     # asked at message 70, in a later batch
    w = world(markers=[m], notes=[EMAIL], flag=set())
    await _run(w, ids=[EMAIL.memory_id], meta=_batch_meta(first=61, last=65))   # ends at 65
    assert w.covered == []                        # the batch holding message 70 is still to come


async def test_a_request_that_was_removed_is_closed_not_covered(world):
    m = _marker(position=63)
    w = world(markers=[m], notes=[PHONE], flag={PHONE.memory_id})
    await _run(w, ids=[PHONE.memory_id], meta=_batch_meta(first=61, last=65))
    assert w.closed and w.covered == []


async def test_missing_columns_are_reported_once_and_never_break_the_batch(world, monkeypatch, caplog):
    from sqlalchemy.exc import ProgrammingError
    w = world(markers=[_marker()], notes=[PHONE])

    async def missing(*a, **k):
        raise ProgrammingError("select", {}, Exception('column "conversation_id" does not exist'))

    monkeypatch.setattr(rstore, "markers_for_batch", missing)
    monkeypatch.setattr(marker, "_columns_missing_logged", False)
    with caplog.at_level("ERROR"):
        assert await _run(w) == 0
        assert await _run(w) == 0
    assert sum("forget_marker_columns_missing" in r.message for r in caplog.records) == 1


async def test_a_batch_request_is_filed_in_the_dashboard_with_a_scope_level():
    """A conversation batch's scope has no scope_level, and deletion_requests.scope_level is NOT
    NULL: the first marker delete logged forget_retry_record_failed because of it."""
    seen = []

    class Sess:
        async def execute(self, statement, params=None):
            seen.append(params)

    scope = _scope()
    scope.scope_level = None
    await store.record_direct(Sess(), scope=scope, query="my contact number", candidates=[PHONE],
                              requested_by="sdk:t", status="applied", result={})
    await store.create_pending(Sess(), scope=scope, query="my contact number", candidates=[PHONE],
                               requested_by="sdk:t")
    assert [p["scope_level"] for p in seen] == ["user", "user"]
