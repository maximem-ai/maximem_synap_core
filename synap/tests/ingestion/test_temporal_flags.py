"""One switch for the temporal set, and the per-feature escape hatch.

Four flags for one feature was the mistake this module exists to undo, so the
test that matters most is that ONE name turns all four on.
"""

import pytest

from synap.cloud.agents.ingestion.services import (
    anchor_resolver, date_prepass, event_anchor, prior_context, temporal_flags)

ALL = {"date_prepass": date_prepass, "event_anchor": event_anchor,
       "prior_context": prior_context, "resolve_open_anchors": anchor_resolver}

EVERY_ENV = [
    "SYNAP_TEMPORAL_V2", "SYNAP_TEMPORAL_V2_BY_INSTANCE",
    "SYNAP_STAGE3_DATE_PREPASS", "SYNAP_STAGE3_DATE_PREPASS_BY_INSTANCE",
    "SYNAP_STAGE3_PRIOR_CONTEXT", "SYNAP_STAGE3_PRIOR_CONTEXT_BY_INSTANCE",
    "SYNAP_EVENT_ANCHOR", "SYNAP_EVENT_ANCHOR_BY_INSTANCE",
    "SYNAP_RESOLVE_OPEN_ANCHORS", "SYNAP_RESOLVE_OPEN_ANCHORS_BY_INSTANCE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in EVERY_ENV:
        monkeypatch.delenv(name, raising=False)


def test_everything_is_off_when_nothing_is_set():
    for name, mod in ALL.items():
        assert mod.enabled_for(None) is False, name
        assert mod.enabled_for("inst_abc") is False, name


def test_one_name_turns_the_whole_set_on(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_V2", "on")
    for name, mod in ALL.items():
        assert mod.enabled_for("inst_abc") is True, name


def test_one_name_turns_it_on_for_a_single_instance(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_V2_BY_INSTANCE", "inst_bench=on")
    for name, mod in ALL.items():
        assert mod.enabled_for("inst_bench") is True, name
        assert mod.enabled_for("inst_real_client") is False, name
        assert mod.enabled_for(None) is False, name


def test_a_single_feature_can_be_held_back(monkeypatch):
    """The reason to keep the per-feature names: switch the set on, then take one
    piece out when it misbehaves, without touching the other three."""
    monkeypatch.setenv("SYNAP_TEMPORAL_V2", "on")
    monkeypatch.setenv("SYNAP_RESOLVE_OPEN_ANCHORS", "off")
    assert anchor_resolver.enabled_for("inst_abc") is False
    assert date_prepass.enabled_for("inst_abc") is True
    assert event_anchor.enabled_for("inst_abc") is True
    assert prior_context.enabled_for("inst_abc") is True


def test_a_single_feature_can_be_let_through_on_one_instance(monkeypatch):
    monkeypatch.setenv("SYNAP_EVENT_ANCHOR_BY_INSTANCE", "inst_one=on")
    assert event_anchor.enabled_for("inst_one") is True
    assert event_anchor.enabled_for("inst_two") is False
    assert date_prepass.enabled_for("inst_one") is False


def test_the_most_specific_setting_wins(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_V2", "on")
    monkeypatch.setenv("SYNAP_STAGE3_DATE_PREPASS", "off")
    monkeypatch.setenv("SYNAP_STAGE3_DATE_PREPASS_BY_INSTANCE", "inst_yes=on")
    assert date_prepass.enabled_for("inst_yes") is True     # 1 beats 2
    assert date_prepass.enabled_for("inst_no") is False     # 2 beats 4


def test_off_is_spelled_several_ways_and_junk_is_not_a_yes(monkeypatch):
    for word in ("off", "0", "false", "no", "OFF", "False"):
        monkeypatch.setenv("SYNAP_TEMPORAL_V2", word)
        assert date_prepass.enabled_for(None) is False, word
    for word in ("", "maybe", "later"):
        monkeypatch.setenv("SYNAP_TEMPORAL_V2", word)
        assert date_prepass.enabled_for(None) is False, word


def test_describe_answers_is_it_actually_on(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_V2_BY_INSTANCE", "inst_bench=on")
    monkeypatch.setenv("SYNAP_RESOLVE_OPEN_ANCHORS", "off")
    d = temporal_flags.describe("inst_bench")
    assert d["master"] is True
    assert d["date_prepass"] is True
    assert d["event_anchor"] is True
    assert d["prior_context"] is True
    assert d["resolve_open_anchors"] is False
    # The resolver's own flag is off here, so the screen is not what blocks it.
    assert d["resolve_open_anchors_flag"] is False
    assert d["resolve_open_anchors_blocked_by"] is None
    assert temporal_flags.describe("inst_other")["master"] is False
