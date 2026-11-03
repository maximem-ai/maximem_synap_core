"""Settling open anchors after a document persists.

The case that drives this: session 1 says the Haight-Ashbury stay was "booked three
months in advance", session 2 dates the wedding to March 2023, and session 1 was
ingested FIRST. Nothing running only at session 1's ingest could resolve it, which
is why this runs after every document and looks in both directions.
"""

import json
from datetime import datetime

import pytest

from synap.cloud.agents.ingestion.services import anchor_resolver
from synap.cloud.agents.ingestion.services.event_anchor import AnchorCandidate

AIRBNB_ANCHOR = {
    "anchor_text": "best friend's wedding",
    "offset": {"n": 3, "unit": "months", "direction": "before"},
    "source_phrase": "had to book three months in advance",
}


class Scope:
    user_id = "11111111-1111-1111-1111-111111111111"
    instance_id = "inst_test"
    customer_id = "22222222-2222-2222-2222-222222222222"
    client_id = "33333333-3333-3333-3333-333333333333"
    scope_level = "user"


class FakeResult:
    def __init__(self, rows=(), rowcount=1):
        self._rows = list(rows)
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class FakeSession:
    """Answers the resolver's three statements in the order it sends them."""

    def __init__(self, open_rows, pool_rows, *, update_rowcount=1):
        self.open_rows = open_rows
        self.pool_rows = pool_rows
        self.update_rowcount = update_rowcount
        self.updates = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        if sql.startswith("SELECT id::text, content, event_anchor"):
            return FakeResult(self.open_rows)
        if sql.startswith("SELECT id::text, content, event_start"):
            return FakeResult(self.pool_rows)
        if sql.startswith("UPDATE memories SET event_start"):
            assert "AND event_start IS NULL" in sql, \
                "the fill-only guard must be in the SQL, not in Python"
            self.updates.append(params)
            return FakeResult(rowcount=self.update_rowcount)
        raise AssertionError(f"unexpected statement: {sql[:90]}")

    async def commit(self):
        self.committed = True



async def judge_admits_all(anchor_text, candidates, *, correlation_id=None):
    """A stand-in for the Jev screen.

    These tests are about the resolver's SQL, its fill-only UPDATE and its vector
    push, not about who picks the candidates. Injecting the verdict keeps them
    testing their own subject, and keeps them off the network.
    """
    return set(range(len(candidates)))


async def judge_unavailable(anchor_text, candidates, *, correlation_id=None):
    """What a down or unkeyed Jev returns. The anchor must be DEFERRED, not
    settled by some other means."""
    return None


def factory(session):
    return lambda: session


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("SYNAP_RESOLVE_OPEN_ANCHORS", "on")
    from synap.cloud.shared import event_window_schema as ews
    ews._reset_for_tests()
    yield
    ews._reset_for_tests()


# ── the case it exists for ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_airbnb_booking_is_settled_from_a_later_session():
    session = FakeSession(
        open_rows=[("m-booking",
                    "The user stayed in Haight-Ashbury for their best friend's "
                    "wedding and had to book three months in advance.",
                    json.dumps(AIRBNB_ANCHOR))],
        pool_rows=[("m-wedding",
                    "The user visited San Francisco for their best friend's "
                    "wedding in March 2023.",
                    datetime(2023, 3, 1), datetime(2023, 3, 31), "month")])
    counts = await anchor_resolver.resolve_open_anchors(factory(session), Scope(),
                                                  screen_fn=judge_admits_all)
    assert counts == {"open": 1, "resolved": 1, "refused": 0, "screened": 1, "deferred": 0, "vectors_updated": 0}
    assert session.updates[0]["s"].date() == datetime(2022, 12, 1).date()
    assert session.committed


# ── the guarantee that bounds every mistake ───────────────────────────────────

@pytest.mark.asyncio
async def test_a_date_already_set_is_never_overwritten():
    """The UPDATE's own WHERE clause does this, so a bug in this module cannot
    undo a correct resolution. rowcount 0 means someone dated it first and their
    value stands."""
    session = FakeSession(
        open_rows=[("m-booking", "booked three months in advance for the wedding",
                    json.dumps(AIRBNB_ANCHOR))],
        pool_rows=[("m-wedding", "the best friend's wedding in March",
                    datetime(2023, 3, 1), None, "month")],
        update_rowcount=0)
    counts = await anchor_resolver.resolve_open_anchors(factory(session), Scope(),
                                                  screen_fn=judge_admits_all)
    assert counts["resolved"] == 0
    assert counts["refused"] == 1


@pytest.mark.asyncio
async def test_running_twice_changes_nothing_the_second_time():
    """Idempotent by construction: a resolved row no longer has event_start NULL,
    so the first SELECT stops returning it."""
    session = FakeSession(open_rows=[], pool_rows=[])
    counts = await anchor_resolver.resolve_open_anchors(factory(session), Scope(),
                                                  screen_fn=judge_admits_all)
    assert counts == {"open": 0, "resolved": 0, "refused": 0, "screened": 0, "deferred": 0, "vectors_updated": 0}
    assert session.updates == []


# ── matching ──────────────────────────────────────────────────────────────────

def test_word_containment_is_gone_and_nothing_silently_replaces_it():
    """It used to pick the candidates, then it was kept as a fallback, now it is
    deleted.

    It required every significant word of the anchor phrase to appear in the
    candidate, so "a wedding for his best friend" missed "best friend's wedding"
    over one apostrophe and "the user's best friend got married" could never match
    "wedding" at all. Keeping it for when the judge is unavailable meant the same
    document ingested twice could get different dates depending on whether Jev
    happened to be up, with nothing in the output saying which path ran.

    Deleting it is cheap because this work is RETRYABLE: the resolver runs after
    every document of the scope persists and its UPDATE carries
    `AND event_start IS NULL`, so an unavailable judge costs a deferral, not a date.
    """
    from synap.cloud.agents.ingestion.services import anchor_resolver
    assert not hasattr(anchor_resolver, "candidates_for")
    assert not hasattr(anchor_resolver, "significant_words")
    import inspect
    src = inspect.getsource(anchor_resolver)
    assert "deferred" in src, (
        "an unavailable judge must be counted and logged, not silently absorbed")

def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_RESOLVE_OPEN_ANCHORS", raising=False)
    monkeypatch.delenv("SYNAP_RESOLVE_OPEN_ANCHORS_BY_INSTANCE", raising=False)
    assert anchor_resolver.enabled_for(None) is False


def test_per_instance_override(monkeypatch):
    monkeypatch.setenv("SYNAP_RESOLVE_OPEN_ANCHORS", "off")
    monkeypatch.setenv("SYNAP_RESOLVE_OPEN_ANCHORS_BY_INSTANCE", "inst_test=on")
    assert anchor_resolver.enabled_for("inst_test") is True
    assert anchor_resolver.enabled_for("inst_other") is False


# ── the vector half, which is where this becomes visible ──────────────────────

@pytest.mark.asyncio
async def test_the_new_date_is_pushed_into_the_vector_metadata():
    """Retrieval reads a memory's date off its VECTOR metadata, so a
    Postgres-only update leaves the whole fix invisible."""
    class Provider:
        def __init__(self):
            self.calls = []

        def update_metadata(self, namespace, updates):
            self.calls.append((namespace, updates))
            return len(updates)

    class VectorWrapper:
        def __init__(self, provider):
            self._provider = provider

        def _get_namespace(self, scope, scope_level=None):
            return f"synap_{scope.customer_id}_v2"

    provider = Provider()
    session = FakeSession(
        open_rows=[("m-booking", "booked three months in advance for the wedding",
                    json.dumps(AIRBNB_ANCHOR))],
        pool_rows=[("m-wedding", "the best friend's wedding in March 2023",
                    datetime(2023, 3, 1), datetime(2023, 3, 31), "month")])
    counts = await anchor_resolver.resolve_open_anchors(
        factory(session), Scope(), vector=VectorWrapper(provider), screen_fn=judge_admits_all)
    assert counts["resolved"] == 1
    assert counts["vectors_updated"] == 1
    namespace, updates = provider.calls[0]
    assert namespace.endswith("_v2")
    assert updates["m-booking"]["event_start"].startswith("2022-12-01")


@pytest.mark.asyncio
async def test_a_store_without_update_metadata_still_writes_postgres():
    """The dates land in Postgres either way; the warning says they will not be
    served until those vectors are rewritten."""
    class VectorWrapper:
        _provider = object()

        def _get_namespace(self, scope, scope_level=None):
            return "ns_v2"

    session = FakeSession(
        open_rows=[("m-booking", "booked three months in advance for the wedding",
                    json.dumps(AIRBNB_ANCHOR))],
        pool_rows=[("m-wedding", "the best friend's wedding in March 2023",
                    datetime(2023, 3, 1), None, "month")])
    counts = await anchor_resolver.resolve_open_anchors(
        factory(session), Scope(), vector=VectorWrapper(), screen_fn=judge_admits_all)
    assert counts["resolved"] == 1
    assert counts["vectors_updated"] == 0


@pytest.mark.asyncio
async def test_an_unavailable_judge_DEFERS_rather_than_guessing():
    """The whole reason deleting the fallback is safe.

    This runs after every document of the scope persists and its UPDATE carries
    `AND event_start IS NULL`, so a judge that is down costs a deferral, not a
    date: the anchor stays open and the next document asks again. The alternative,
    keeping word containment for these moments, meant the same document ingested
    twice could get different dates depending on whether Jev happened to be up.
    """
    session = FakeSession(
        open_rows=[("m-booking", "booked three months in advance for the wedding",
                    json.dumps({"anchor_text": "best friend's wedding",
                                "offset": {"n": 3, "unit": "months",
                                           "direction": "before"}}))],
        pool_rows=[("m-wedding", "the user's best friend's wedding",
                    datetime(2023, 5, 20), None, "day")])
    counts = await anchor_resolver.resolve_open_anchors(
        factory(session), Scope(), screen_fn=judge_unavailable)
    assert counts["open"] == 1
    assert counts["deferred"] == 1
    assert counts["resolved"] == 0
    assert counts["screened"] == 0
    assert session.updates == [], "nothing may be written when the judge is absent"


@pytest.mark.asyncio
async def test_with_the_screen_off_nothing_is_settled_by_other_means():
    """The coupling this introduces, pinned so it cannot drift back.

    Deciding WHICH stored memory is the event is now the judge's job and nobody
    else's, so the resolver depends on the screen being on for the instance.
    """
    session = FakeSession(
        open_rows=[("m-booking", "booked three months in advance for the wedding",
                    json.dumps({"anchor_text": "best friend's wedding",
                                "offset": {"n": 3, "unit": "months",
                                           "direction": "before"}}))],
        pool_rows=[("m-wedding", "the user's best friend's wedding",
                    datetime(2023, 5, 20), None, "day")])
    counts = await anchor_resolver.resolve_open_anchors(factory(session), Scope())
    assert counts["resolved"] == 0
    assert counts["deferred"] == 1
    assert session.updates == []
