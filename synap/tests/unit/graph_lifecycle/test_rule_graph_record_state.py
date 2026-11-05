"""graph_record_state: the one rule that decides whether a graph record is current.

The 4b fixture is the live check of 2026-11-05 (user r12a_5a60d, the a_job
documents of evidence/c4_graph/e4.py), rows copied from
synap_test/plan/live_check/pg.txt:

    3c1f4817 facts, not episode, temporal_fact, doc 2026-09-06, SUPERSEDED by 0874ae8a
    58bee2aa facts, not episode, temporal_fact, doc 2026-09-06, LIVE ("manager at Acme is Priya")
    6336821f temporal_events, EPISODE,            doc 2026-09-06, LIVE ("started at Acme ...")
    0874ae8a facts, episode,                       doc 2026-11-03, event_start 2026-10-26, LIVE ("resigned")

Edges: WORKS_AT [3c1f, 58be, 6336], HAS_ROLE [3c1f, 6336], REPORTS_TO [58be, 6336].
Before this rule existed nothing judged these records at all, and all three
were served as present-tense statements.
"""
from __future__ import annotations

from datetime import datetime, timezone

from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import (
    REASON_DELETED,
    REASON_SUPERSEDED,
    REASON_UNSUPPORTED,
    Closed,
    GraphRecord,
    Open,
    SupporterRow,
    graph_record_state,
    is_alive,
    is_current_support,
)


def _d(y, m, d, h=0, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


M_3C1F = "3c1f4817-d3c8-4ad1-8fee-b26d9cfbe082"
M_58BE = "58bee2aa-8733-4d93-b167-38e0d6210781"
M_6336 = "6336821f-d3b7-4068-901e-ff93bc331d65"
M_0874 = "0874ae8a-d213-44a2-8b72-9bd5cc83ffed"

ROWS_4B = {
    M_3C1F: SupporterRow(
        id=M_3C1F, superseded_by=M_0874, superseded_at=_d(2026, 11, 5, 7, 24),
        is_episode=False, temporal_category="temporal_fact",
        document_created_at=_d(2026, 9, 6), created_at=_d(2026, 11, 5, 7, 22),
        superseder_document_created_at=_d(2026, 11, 3),
        superseder_created_at=_d(2026, 11, 5, 7, 23),
        superseder_event_start=_d(2026, 10, 26),
    ),
    M_58BE: SupporterRow(
        id=M_58BE, is_episode=False, temporal_category="temporal_fact",
        document_created_at=_d(2026, 9, 6), created_at=_d(2026, 11, 5, 7, 22),
    ),
    M_6336: SupporterRow(
        id=M_6336, is_episode=True, temporal_category="episode",
        document_created_at=_d(2026, 9, 6), created_at=_d(2026, 11, 5, 7, 22),
        event_start=_d(2026, 9, 6),
    ),
    M_0874: SupporterRow(
        id=M_0874, is_episode=True, temporal_category="episode",
        document_created_at=_d(2026, 11, 3), created_at=_d(2026, 11, 5, 7, 23),
        event_start=_d(2026, 10, 26),
    ),
}


# --- the 4b live fixture -----------------------------------------------------

def test_4b_works_at_acme_closed_by_t2():
    state = graph_record_state(GraphRecord(memory_ids=[M_3C1F, M_58BE, M_6336]), ROWS_4B)
    assert isinstance(state, Closed)
    assert state.rule == "T2"
    assert state.reason == REASON_SUPERSEDED
    assert state.by_memory == M_0874
    assert state.valid_to == _d(2026, 10, 26)
    assert set(state.alive_memory_ids) == {M_58BE, M_6336}


def test_4b_has_role_data_engineer_closed():
    state = graph_record_state(GraphRecord(memory_ids=[M_3C1F, M_6336]), ROWS_4B)
    assert isinstance(state, Closed)
    assert state.by_memory == M_0874
    assert state.valid_to == _d(2026, 10, 26)


def test_4b_reports_to_priya_stays_open():
    state = graph_record_state(GraphRecord(memory_ids=[M_58BE, M_6336]), ROWS_4B)
    assert isinstance(state, Open)
    assert state.current_memory_ids == (M_58BE,)
    assert set(state.alive_memory_ids) == {M_58BE, M_6336}


def test_4b_unbacked_edge_open():
    state = graph_record_state(GraphRecord(memory_ids=[]), ROWS_4B)
    assert isinstance(state, Open)
    assert state.unbacked is True


# --- T1 ----------------------------------------------------------------------

def test_t1_all_superseded_closes_with_superseder():
    rows = {"m1": SupporterRow(id="m1", superseded_by="m2", document_created_at=_d(2026, 1, 1),
                               superseder_document_created_at=_d(2026, 2, 1))}
    # m2 is a supporter, so T2 does not apply; T1 does (no current support).
    state = graph_record_state(GraphRecord(memory_ids=["m1", "m2"]), rows)
    assert isinstance(state, Closed)
    assert state.rule == "T1"
    assert state.reason == REASON_SUPERSEDED
    assert state.by_memory == "m2"


def test_t1_hard_deleted_ids_close_as_deleted():
    state = graph_record_state(GraphRecord(memory_ids=["gone-1", "gone-2"]), {})
    assert isinstance(state, Closed)
    assert state.rule == "T1"
    assert state.reason == REASON_DELETED


def test_t1_forgotten_closes_as_deleted():
    rows = {"m1": SupporterRow(id="m1", deleted_at=_d(2026, 3, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1", "absent"]), rows)
    assert isinstance(state, Closed) and state.reason == REASON_DELETED


def test_t1_deprecated_closes_as_unsupported():
    rows = {"m1": SupporterRow(id="m1", deprecated_at=_d(2026, 3, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1"]), rows)
    assert isinstance(state, Closed) and state.reason == REASON_UNSUPPORTED


def test_episode_alone_does_not_keep_edge_current():
    rows = {"e1": SupporterRow(id="e1", is_episode=True, document_created_at=_d(2026, 1, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["e1"]), rows)
    assert isinstance(state, Closed)
    assert state.rule == "T1" and state.reason == REASON_UNSUPPORTED
    assert state.alive_memory_ids == ("e1",)


def test_episode_by_temporal_category_alone_does_not_keep_edge_current():
    rows = {"e1": SupporterRow(id="e1", is_episode=False, temporal_category="episode")}
    assert isinstance(graph_record_state(GraphRecord(memory_ids=["e1"]), rows), Closed)


def test_released_only_record_is_closed():
    state = graph_record_state(GraphRecord(memory_ids=[], released_memory_ids=["m1"]), {})
    assert isinstance(state, Closed) and state.rule == "T1"


# --- T2 ----------------------------------------------------------------------

def _superseded_m1(**superseder):
    base = dict(superseder_document_created_at=_d(2026, 5, 1), superseder_event_start=_d(2026, 4, 20))
    base.update(superseder)
    return SupporterRow(id="m1", superseded_by="n1", document_created_at=_d(2026, 1, 1), **base)


def test_mixed_support_said_after_superseder_stays_open():
    rows = {"m1": _superseded_m1(),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 6, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows)
    assert isinstance(state, Open)
    assert state.alive_memory_ids == ("m3",)


def test_mixed_support_said_before_superseder_closes():
    rows = {"m1": _superseded_m1(),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 3, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows)
    assert isinstance(state, Closed) and state.rule == "T2"
    assert state.valid_to == _d(2026, 4, 20)


def test_t2_tie_same_document_date_stays_open():
    rows = {"m1": _superseded_m1(),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 5, 1))}
    assert isinstance(graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows), Open)


def test_t2_superseder_that_supports_the_record_does_not_close_it():
    rows = {"m1": _superseded_m1(),
            "n1": SupporterRow(id="n1", document_created_at=_d(2026, 5, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1", "n1"]), rows)
    assert isinstance(state, Open)


def test_t2_superseder_in_released_ids_counts_as_supporter():
    rows = {"m1": _superseded_m1(),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 3, 1))}
    state = graph_record_state(
        GraphRecord(memory_ids=["m1", "m3"], released_memory_ids=["n1"]), rows)
    assert isinstance(state, Open)


def test_t2_valid_to_falls_back_to_superseder_said_at():
    rows = {"m1": _superseded_m1(superseder_event_start=None),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 3, 1))}
    state = graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows)
    assert isinstance(state, Closed) and state.valid_to == _d(2026, 5, 1)


def test_t2_skipped_when_superseder_dates_unknown():
    rows = {"m1": SupporterRow(id="m1", superseded_by="n1"),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 3, 1))}
    assert isinstance(graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows), Open)


def test_said_at_falls_back_to_created_at():
    rows = {"m1": _superseded_m1(),
            "m3": SupporterRow(id="m3", created_at=_d(2026, 7, 1))}
    assert isinstance(graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows), Open)


def test_naive_and_aware_datetimes_compare():
    rows = {"m1": _superseded_m1(superseder_document_created_at=datetime(2026, 5, 1)),
            "m3": SupporterRow(id="m3", document_created_at=_d(2026, 6, 1))}
    assert isinstance(graph_record_state(GraphRecord(memory_ids=["m1", "m3"]), rows), Open)


# --- fixed entities, predicates ---------------------------------------------

def test_writer_and_customer_nodes_always_open():
    state = graph_record_state(GraphRecord(memory_ids=["gone"], is_fixed_entity=True), {})
    assert isinstance(state, Open)


def test_alive_predicate():
    assert is_alive(SupporterRow(id="a"))
    assert not is_alive(None)
    assert not is_alive(SupporterRow(id="a", superseded_by="b"))
    assert not is_alive(SupporterRow(id="a", deprecated_at=_d(2026, 1, 1)))
    assert not is_alive(SupporterRow(id="a", deleted_at=_d(2026, 1, 1)))
    assert is_current_support(SupporterRow(id="a"))
    assert not is_current_support(SupporterRow(id="a", is_episode=True))


def test_duplicate_ids_are_ignored():
    state = graph_record_state(GraphRecord(memory_ids=[M_58BE, M_58BE]), ROWS_4B)
    assert isinstance(state, Open) and state.alive_memory_ids == (M_58BE,)


# --- released ids (lead decision 2026-11-05): judged for history, never support ---

def test_t2_closed_record_stays_closed_after_release():
    # The write side released the superseded id 3c1f; the other supporters stay.
    state = graph_record_state(
        GraphRecord(memory_ids=[M_58BE, M_6336], released_memory_ids=[M_3C1F]), ROWS_4B)
    assert isinstance(state, Closed)
    assert state.rule == "T2" and state.by_memory == M_0874
    assert state.valid_to == _d(2026, 10, 26)


def test_released_record_reopens_on_new_supporter_said_after_superseder():
    rows = dict(ROWS_4B)
    m_new = "11111111-2222-4333-8444-555555555555"
    rows[m_new] = SupporterRow(id=m_new, document_created_at=_d(2026, 12, 1))
    state = graph_record_state(
        GraphRecord(memory_ids=[M_58BE, M_6336, m_new], released_memory_ids=[M_3C1F]), rows)
    assert isinstance(state, Open)
    assert set(state.current_memory_ids) == {M_58BE, m_new}


def test_released_alive_memory_is_not_support():
    # An edited memory: released, still alive. It cannot keep the record open.
    rows = {"m1": SupporterRow(id="m1", document_created_at=_d(2026, 1, 1))}
    state = graph_record_state(GraphRecord(memory_ids=[], released_memory_ids=["m1"]), rows)
    assert isinstance(state, Closed) and state.rule == "T1"
    state = graph_record_state(GraphRecord(memory_ids=["m1"], released_memory_ids=["m1"]), rows)
    assert isinstance(state, Closed)


def test_released_deleted_only_closes_as_deleted():
    state = graph_record_state(GraphRecord(memory_ids=[], released_memory_ids=["gone"]), {})
    assert isinstance(state, Closed) and state.reason == REASON_DELETED
