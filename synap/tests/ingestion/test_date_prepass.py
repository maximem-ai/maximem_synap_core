"""The pre-pass's calendar arithmetic, against the addendum's own rule table.

These cases are the table in ``prompts/extraction/temporal_addendum.py`` turned
into assertions. If one fails, the pre-pass and the extraction prompts have
drifted apart, which puts the two halves of Stage 3 back into the disagreement
this work exists to remove.

DCT is 2023-05-27, a Saturday — the real Airbnb scope's conversation date, so the
interesting cases are the ones that actually cost us a question.
"""

from datetime import datetime

import pytest

from synap.cloud.agents.ingestion.services.date_prepass import (
    PhraseReading,
    add_months,
    enabled_for,
    render_resolved_block,
    resolve_all,
    resolve_reading,
    week_start_sunday,
)

DCT = datetime(2023, 5, 27)          # Saturday


def r(kind, **kw):
    return PhraseReading(phrase=kw.pop("phrase", kind), kind=kind, **kw)


def win(kind, **kw):
    """(start, end, precision) with the dates as plain dates for readability."""
    start, end, precision = resolve_reading(r(kind, **kw), DCT)
    return (start.date() if start else None,
            end.date() if end else None,
            precision)


# ── points ────────────────────────────────────────────────────────────────────

def test_today_yesterday_tomorrow():
    assert win("today") == (datetime(2023, 5, 27).date(), None, "day")
    assert win("yesterday") == (datetime(2023, 5, 26).date(), None, "day")
    assert win("tomorrow") == (datetime(2023, 5, 28).date(), None, "day")


def test_counted_offsets():
    assert win("n_days_ago", n=3) == (datetime(2023, 5, 24).date(), None, "day")
    assert win("n_days_ahead", n=3) == (datetime(2023, 5, 30).date(), None, "day")
    assert win("n_weeks_ago", n=2) == (datetime(2023, 5, 13).date(), None, "approximate")
    assert win("n_weeks_ahead", n=2) == (datetime(2023, 6, 10).date(), None, "approximate")


def test_two_months_ago_is_the_airbnb_wedding():
    """"exactly two months ago" on 27 May is 27 March, and the 5-day trip that
    follows is what dates the booking three months before it."""
    assert win("n_months_ago", n=2) == (datetime(2023, 3, 27).date(), None, "approximate")


def test_a_month_ago_keeps_the_day_number():
    """The addendum is explicit: on 2023-05-30 "a month ago" is 2023-04-30, NOT
    2023-04-01. The old first-of-month rule turned one week into five."""
    start, _, _ = resolve_reading(r("n_months_ago", n=1), datetime(2023, 5, 30))
    assert start.date() == datetime(2023, 4, 30).date()


def test_month_arithmetic_clamps_to_the_shorter_month():
    start, _, _ = resolve_reading(r("n_months_ago", n=1), datetime(2023, 3, 31))
    assert start.date() == datetime(2023, 2, 28).date()


def test_month_arithmetic_handles_a_leap_day():
    start, _, _ = resolve_reading(r("n_months_ago", n=1), datetime(2024, 3, 29))
    assert start.date() == datetime(2024, 2, 29).date()


def test_add_months_is_symmetric_across_a_year_boundary():
    assert add_months(datetime(2023, 1, 15), -1).date() == datetime(2022, 12, 15).date()
    assert add_months(datetime(2022, 12, 15), 1).date() == datetime(2023, 1, 15).date()


def test_duration_ongoing_counts_back_in_weeks():
    """"taking lessons for six weeks" on 27 May started around 15 April."""
    assert win("duration_ongoing", n=6) == (datetime(2023, 4, 15).date(), None,
                                           "approximate")


def test_duration_ongoing_counts_months_as_months():
    """Measured on a real ingest: "six months now", said on
    2023-05-20, was stored as starting 2023-04-08, which is six WEEKS back."""
    start, _, _ = resolve_reading(
        r("duration_ongoing", phrase="six months now", n=6, unit="months"),
        datetime(2023, 5, 20))
    assert start.date() == datetime(2022, 11, 20).date()


def test_duration_ongoing_counts_years_as_years():
    assert win("duration_ongoing", phrase="the past five years", n=5,
               unit="years")[0] == datetime(2018, 5, 27).date()


def test_duration_ongoing_counts_days_as_days():
    assert win("duration_ongoing", phrase="for ten days now", n=10,
               unit="days")[0] == datetime(2023, 5, 17).date()


def test_the_phrase_overrules_a_model_that_already_converted():
    """A reading of n 12 for "three months now" is the old prompt's conversion
    (a month = 4 weeks) arriving with the new unit. Counting 12 MONTHS back
    would date the thing a year ago. The phrase says three, so three it is."""
    start, _, _ = resolve_reading(
        r("duration_ongoing", phrase="three months now", n=12, unit="months"),
        datetime(2023, 5, 30))
    assert start.date() == datetime(2023, 2, 28).date()


def test_a_phrase_with_no_unit_still_counts_in_weeks():
    """Nothing in the phrase or the reading names a unit, so the arithmetic is
    the one this branch has always done, and a model that ignores the new field
    costs nothing."""
    assert win("duration_ongoing", phrase="for a while now", n=6)[0] == \
        datetime(2023, 4, 15).date()


def test_a_few_months_is_three_months_not_one():
    """"I have been using Apple TV+ for a few months now", said 2023-05-22, is
    about February. The phrase contains the word "a", so a quantity lookup that
    tries the articles first reads it as ONE month and overrules the model's
    correct 3. gpt4_2f56ae70."""
    start, _, _ = resolve_reading(
        r("duration_ongoing", phrase="for a few months now", n=3, unit="months"),
        datetime(2023, 5, 22))
    assert start.date() == datetime(2023, 2, 22).date()


def test_a_phrase_with_no_number_keeps_the_readings_own_n():
    """"for months now" states a unit and no quantity, so the model's n stands."""
    assert win("duration_ongoing", phrase="for months now", n=4)[0] == \
        datetime(2023, 1, 27).date()


# ── weekdays ──────────────────────────────────────────────────────────────────

def test_weekday_past():
    assert win("weekday_past", weekday="tuesday") == (datetime(2023, 5, 23).date(),
                                                      None, "day")


def test_weekday_past_is_never_today():
    """DCT is a Saturday, so "last Saturday" is the one a week earlier. The
    addendum says strictly before, and a reading that returned today would date
    an event to the day it was merely mentioned."""
    assert win("weekday_past", weekday="saturday") == (datetime(2023, 5, 20).date(),
                                                       None, "day")


def test_weekday_future_may_be_today():
    assert win("weekday_future", weekday="monday") == (datetime(2023, 5, 29).date(),
                                                       None, "day")
    assert win("weekday_future", weekday="saturday") == (datetime(2023, 5, 27).date(),
                                                         None, "day")


# ── windows ───────────────────────────────────────────────────────────────────

def test_weeks_run_sunday_to_saturday():
    assert week_start_sunday(DCT).date() == datetime(2023, 5, 21).date()
    assert win("this_week") == (datetime(2023, 5, 21).date(),
                                datetime(2023, 5, 27).date(), "week")
    assert win("last_week") == (datetime(2023, 5, 14).date(),
                               datetime(2023, 5, 20).date(), "week")


def test_weekends():
    assert win("last_weekend") == (datetime(2023, 5, 20).date(),
                                   datetime(2023, 5, 21).date(), "weekend")
    assert win("this_weekend") == (datetime(2023, 5, 27).date(),
                                   datetime(2023, 5, 28).date(), "weekend")


def test_months():
    assert win("last_month") == (datetime(2023, 4, 1).date(),
                                 datetime(2023, 4, 30).date(), "month")
    assert win("this_month") == (datetime(2023, 5, 1).date(),
                                 datetime(2023, 5, 31).date(), "month")


def test_named_month_without_a_year_picks_the_most_recent_one():
    # March has already happened this year
    assert win("named_month", month=3) == (datetime(2023, 3, 1).date(),
                                           datetime(2023, 3, 31).date(), "month")
    # August has not, so it means last August
    assert win("named_month", month=8) == (datetime(2022, 8, 1).date(),
                                           datetime(2022, 8, 31).date(), "month")


def test_named_month_with_a_year_is_taken_as_given():
    assert win("named_month", month=8, year=2021) == (datetime(2021, 8, 1).date(),
                                                      datetime(2021, 8, 31).date(),
                                                      "month")


def test_years():
    assert win("last_year") == (datetime(2022, 1, 1).date(),
                                datetime(2022, 12, 31).date(), "year")
    assert win("this_year") == (datetime(2023, 1, 1).date(),
                                datetime(2023, 12, 31).date(), "year")
    assert win("named_year", year=2019) == (datetime(2019, 1, 1).date(),
                                            datetime(2019, 12, 31).date(), "year")
    assert win("n_years_ago", n=1) == (datetime(2022, 1, 1).date(),
                                       datetime(2022, 12, 31).date(), "year")


def test_since_a_year_has_no_end():
    assert win("since_year", year=2019) == (datetime(2019, 1, 1).date(), None, "year")


def test_seasons():
    # Summer 2023 has not started on 27 May, so "last summer" is 2022's
    assert win("last_season", season="summer") == (datetime(2022, 6, 1).date(),
                                                   datetime(2022, 8, 31).date(),
                                                   "season")
    # winter straddles the new year
    assert win("named_season", season="winter", year=2022) == (
        datetime(2022, 12, 1).date(), datetime(2023, 2, 28).date(), "season")


def test_explicit_date_and_range():
    """A year the phrase states is the year we keep."""
    assert win("explicit_date", phrase="February 24 2023",
               date="2023-02-24") == (datetime(2023, 2, 24).date(), None, "day")
    assert win("explicit_range", phrase="May 3 to May 7, 2023", date="2023-05-03",
               end_date="2023-05-07") == (datetime(2023, 5, 3).date(),
                                          datetime(2023, 5, 7).date(), "range")


def test_a_bare_month_and_day_takes_its_year_from_the_tense():
    """DCT is 2023-05-27. The year the model wrote is ignored when the phrase
    did not state one, so a finished event can never land after the day it was
    said. gpt4_731e37d7 and a9f6b44c are both this case."""
    # said as done, and the model's year would put it 18 days ahead
    assert win("explicit_date", phrase="on June 14th", date="2023-06-14",
               tense="past") == (datetime(2022, 6, 14).date(), None, "day")
    # said as done, already behind DCT: the year is unchanged
    assert win("explicit_date", phrase="on February 24th", date="2023-02-24",
               tense="past") == (datetime(2023, 2, 24).date(), None, "day")
    # still to come
    assert win("explicit_date", phrase="on June 14th", date="2023-06-14",
               tense="future") == (datetime(2023, 6, 14).date(), None, "day")
    # the real workshop, as it was said: "on March 15-16", past, DCT 2023-05-27
    assert win("explicit_range", phrase="on March 15-16", date="2023-03-15",
               end_date="2023-03-16", tense="past") == (
        datetime(2023, 3, 15).date(), datetime(2023, 3, 16).date(), "range")
    # a range the tense pushes back a year keeps its own length
    assert win("explicit_range", phrase="from June 14 to June 20",
               date="2023-06-14", end_date="2023-06-20", tense="past") == (
        datetime(2022, 6, 14).date(), datetime(2022, 6, 20).date(), "range")
    # and one that straddles the new year stays eleven days, not minus 354
    assert win("explicit_range", phrase="from Dec 28 to Jan 3",
               date="2023-12-28", end_date="2024-01-03", tense="past") == (
        datetime(2022, 12, 28).date(), datetime(2023, 1, 3).date(), "range")


def test_with_no_tense_only_a_date_past_the_speaking_day_is_refused():
    """DCT is 2023-05-27. A date at or before it cannot be the defect, so the
    model's year is kept. One after it is where a mis-yeared past event and a
    real plan look identical, so it is refused.

    The first version of this refused BOTH, and the re-ingest on 2026-10-08
    showed that cost four correct dates ("February 22", "December 12",
    "on February 20th", "March 10th", all untensed, all already right) to fix
    two wrong ones."""
    # at or before the speaking day: kept exactly as the model wrote it
    assert win("explicit_date", phrase="on February 22",
               date="2023-02-22") == (datetime(2023, 2, 22).date(), None, "day")
    assert win("explicit_date", phrase="on December 12",
               date="2022-12-12") == (datetime(2022, 12, 12).date(), None, "day")
    assert win("explicit_date", phrase="today's date",
               date="2023-05-27") == (datetime(2023, 5, 27).date(), None, "day")
    # a range we are keeping keeps the model's end too
    assert win("explicit_range", phrase="from May 3 to May 7", date="2023-05-03",
               end_date="2023-05-07") == (datetime(2023, 5, 3).date(),
                                          datetime(2023, 5, 7).date(), "range")
    # after the speaking day with no tense: refused
    assert win("explicit_date", phrase="on June 14th",
               date="2023-06-14") == (None, None, None)
    assert win("explicit_range", phrase="on June 14-15", date="2023-06-14",
               end_date="2023-06-15") == (None, None, None)


# ── refusing to guess ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,kw", [
    ("n_days_ago", {}),            # no count
    ("n_months_ago", {}),
    ("duration_ongoing", {}),
    ("weekday_past", {}),          # no weekday
    ("weekday_past", {"weekday": "frideay"}),
    ("explicit_date", {}),         # no date
    ("explicit_date", {"date": "not-a-date"}),
    # a bare month-and-day with no tense: see the test above
    ("explicit_date", {"phrase": "on the 14th", "date": "2023-06-14"}),
    ("named_month", {"month": 13}),
    ("named_season", {"season": "monsoon"}),
    ("since_year", {}),
])
def test_an_incomplete_reading_resolves_to_nothing(kind, kw):
    """All None is the safe answer: the phrase is left out of the table and the
    per-type prompts behave exactly as they do today."""
    assert win(kind, **kw) == (None, None, None)


def test_an_unknown_kind_is_dropped_not_guessed():
    assert PhraseReading.from_raw({"phrase": "x", "kind": "next_fortnight"}) is None
    assert PhraseReading.from_raw({"phrase": "x", "kind": "none"}) is None
    assert PhraseReading.from_raw({"kind": "today"}) is None          # no phrase
    assert PhraseReading.from_raw("today") is None                    # not a dict


def test_from_raw_does_not_read_a_bool_as_a_count():
    assert PhraseReading.from_raw({"phrase": "x", "kind": "n_days_ago", "n": True}).n is None


# ── the table and the block ───────────────────────────────────────────────────

def test_resolve_all_keeps_only_what_resolved():
    table = resolve_all([r("last_month"),
                         r("n_days_ago", phrase="three days ago", n=3),
                         r("weekday_past", phrase="last Frideay", weekday="frideay")],
                        DCT)
    assert set(table) == {"last_month", "three days ago"}
    # validate_event_window closes an inclusive window on the last second
    assert table["last_month"]["event_end"].hour == 23


def test_an_empty_table_renders_nothing():
    """Empty string, so the per-type prompt is byte-identical to today's."""
    assert render_resolved_block({}) == ""


def test_the_block_names_the_phrase_and_forbids_recomputing():
    block = render_resolved_block(resolve_all([r("last_month")], DCT))
    assert '"last_month"' in block
    assert "2023-04-01" in block and "2023-04-30" in block
    assert "do NOT" in block


# ── the flag ──────────────────────────────────────────────────────────────────

def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_STAGE3_DATE_PREPASS", raising=False)
    monkeypatch.delenv("SYNAP_STAGE3_DATE_PREPASS_BY_INSTANCE", raising=False)
    assert enabled_for(None) is False
    assert enabled_for("inst_abc") is False


def test_per_instance_override_beats_the_fleet_flag(monkeypatch):
    monkeypatch.setenv("SYNAP_STAGE3_DATE_PREPASS", "on")
    monkeypatch.setenv("SYNAP_STAGE3_DATE_PREPASS_BY_INSTANCE", "inst_abc=off")
    assert enabled_for(None) is True
    assert enabled_for("inst_other") is True
    assert enabled_for("inst_abc") is False
    assert enabled_for("INST_ABC") is False          # ids match case-insensitively


# ── the no-op guarantee ───────────────────────────────────────────────────────

def test_a_missing_table_leaves_the_prompt_byte_identical():
    """The one assertion that keeps a flag-off or failed pre-pass free.

    Asserted on the STRING, not on behaviour: a shipped temporal filter was once
    a guaranteed no-op because nobody compared the thing that actually goes out.
    """
    from synap.cloud.agents.ingestion.pipeline.stage3_extraction import (
        _maybe_add_resolved_dates,
    )
    prompt = "Content to analyze:\n---\nUser: hello\n---\n"
    assert _maybe_add_resolved_dates(prompt, None) == prompt
    assert _maybe_add_resolved_dates(prompt, {}) == prompt


def test_a_resolved_table_is_prepended_and_keeps_the_original_intact():
    from synap.cloud.agents.ingestion.pipeline.stage3_extraction import (
        _maybe_add_resolved_dates,
    )
    prompt = "Content to analyze:\n---\nUser: hello\n---\n"
    out = _maybe_add_resolved_dates(prompt, resolve_all([r("last_month")], DCT))
    assert out.endswith(prompt)          # the original is untouched, not rewritten
    assert out != prompt
    assert "2023-04-01" in out


@pytest.mark.asyncio
async def test_run_prepass_returns_an_empty_table_on_every_failure_path():
    """A timeout, a raising client, a non-JSON answer and a missing DCT all have
    to come back empty rather than raise, because the caller treats empty as
    "behave exactly as today" and anything else costs memories."""
    from synap.cloud.agents.ingestion.services.date_prepass import run_prepass

    class Raises:
        async def complete(self, **kw):
            raise RuntimeError("provider down")

    class Garbage:
        async def complete(self, **kw):
            return {"content": "not json at all"}

    class WrongShape:
        async def complete(self, **kw):
            return {"content": '{"phrases": "should have been a list"}'}

    assert await run_prepass(content="x", dct=DCT, llm_client=Raises()) == {}
    assert await run_prepass(content="x", dct=DCT, llm_client=Garbage()) == {}
    assert await run_prepass(content="x", dct=DCT, llm_client=WrongShape()) == {}
    assert await run_prepass(content="x", dct=None, llm_client=Garbage()) == {}
    assert await run_prepass(content="", dct=DCT, llm_client=Garbage()) == {}
    assert await run_prepass(content="x", dct=DCT, llm_client=None) == {}


@pytest.mark.asyncio
async def test_run_prepass_resolves_a_well_formed_reading():
    from synap.cloud.agents.ingestion.services.date_prepass import run_prepass

    class Good:
        async def complete(self, **kw):
            return {"content": '{"phrases": [{"phrase": "two months ago",'
                               ' "kind": "n_months_ago", "n": 2},'
                               ' {"phrase": "recently", "kind": "none"}]}'}

    table = await run_prepass(content="x", dct=DCT, llm_client=Good())
    assert list(table) == ["two months ago"]          # "none" is dropped, not dated
    assert table["two months ago"]["event_start"].date() == datetime(2023, 3, 27).date()


# ── the origin of a phrase, added 2026-10-03 ──────────────────────────────────
#
# Measured failure this pins. The pre-pass read "three months in advance" as
# n_months_ago and resolved it to DCT minus three months, then the resolved-dates
# block told extraction to "use it exactly and do NOT recompute". That overrode
# the event-anchor rule in the same prompt saying not to guess a date from DCT, so
# the Airbnb question stored 2023-02-27 when the answer was 2022-12-27, five
# months before the question instead of three. Two halves of one prompt
# contradicted each other and the wrong half won because it was at the top.

class TestOriginKinds:
    def test_an_offset_from_another_event_is_never_dated_here(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        r = dp.PhraseReading(phrase="three months in advance",
                             kind="offset_from_event", n=3)
        assert dp.resolve_reading(r, datetime(2023, 5, 27)) == (None, None, None)

    def test_a_length_of_time_is_never_dated_here(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        r = dp.PhraseReading(phrase="up to 2 weeks", kind="duration_property", n=2)
        assert dp.resolve_reading(r, datetime(2023, 5, 27)) == (None, None, None)

    def test_an_ongoing_state_IS_still_dated(self):
        """The neighbour of duration_property, which must keep working: a person
        who has been doing something for six weeks started six weeks ago."""
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        start, _end, _p = dp.resolve_reading(
            dp.PhraseReading(phrase="for six weeks", kind="duration_ongoing", n=6),
            datetime(2023, 5, 27))
        assert start == datetime(2023, 4, 15)

    def test_both_kinds_are_accepted_readings(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        for kind in ("offset_from_event", "duration_property"):
            assert dp.PhraseReading.from_raw({"phrase": "x", "kind": kind}) is not None, kind
            assert kind in dp.ALL_KINDS

    def test_the_phrase_is_KEPT_not_dropped(self):
        """Dropping it is the trap: the block tells the extractor that a phrase
        missing from the list should be dated by the rules it was given, which is
        the guess this change exists to stop."""
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        table = dp.resolve_all([
            dp.PhraseReading(phrase="three months in advance",
                             kind="offset_from_event", n=3),
            dp.PhraseReading(phrase="two months ago", kind="n_months_ago", n=2),
        ], datetime(2023, 5, 27))
        assert set(table) == {"three months in advance", "two months ago"}
        assert table["three months in advance"]["event_start"] is None
        assert table["three months in advance"]["no_dct_origin"] == "offset_from_event"
        assert table["two months ago"]["event_start"] == datetime(2023, 3, 27)

    def test_the_block_tells_the_extractor_NOT_to_date_it(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        block = dp.render_resolved_block(dp.resolve_all([
            dp.PhraseReading(phrase="three months in advance",
                             kind="offset_from_event", n=3),
            dp.PhraseReading(phrase="two months ago", kind="n_months_ago", n=2),
        ], datetime(2023, 5, 27)))
        assert "NOT DATED FROM THIS CONVERSATION" in block
        assert '"three months in advance"' in block
        assert "event_anchor" in block                      # points at the anchor path
        assert "2023-03-27" in block                        # the dated one still resolves
        # and the phrase must never appear with a date beside it
        for line in block.splitlines():
            if "three months in advance" in line:
                assert "event_start" not in line

    def test_a_block_with_only_undated_phrases_still_renders(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        block = dp.render_resolved_block(dp.resolve_all(
            [dp.PhraseReading(phrase="in advance", kind="offset_from_event")],
            datetime(2023, 5, 27)))
        assert "NOT DATED FROM THIS CONVERSATION" in block
        assert "RESOLVED DATES" not in block

    def test_the_reading_prompt_teaches_the_origin_test(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        p = dp.READING_SYSTEM_PROMPT
        assert "offset_from_event" in p and "duration_property" in p
        assert "counted from WHAT" in p
        assert "MUST report these two kinds" in p


# ── a bare month or season needs a tense, added 2026-10-04 ────────────────────
#
# Found by hand-checking the first 33 scopes of a full run. Three of 18 sampled
# dated rows were a year out, all the same way, and two of them were the SAME TRIP
# contradicting itself in the store:
#
#   said 2023-05-05 -> 2022-06-01  "planning a road trip to the mountains in June 2022"
#   said 2023-05-05 -> 2023-06-01  "planning a road trip to the mountains in June 2023"
#   said 2023-05-05 -> 2022-06-01  "goal of reaching 1000 miles by the end of summer"
#   said 2023-05-21 -> 2022-10-01  "Fort Lauderdale Boat Show takes place in October"
#
# resolve_reading hardcoded "the most recent occurrence at or before DCT's month",
# ignoring tense, while the extraction prompt four lines away said "that month, in
# the year that fits the TENSE". The pre-pass block is framed as already decided,
# so it won. Fourth instance in one day of two halves of one prompt disagreeing.

class TestABareMonthNeedsATense:
    DCT = datetime(2023, 5, 5)

    def _start(self, **kw):
        from synap.cloud.agents.ingestion.services.date_prepass import (
            PhraseReading, resolve_reading)
        s, _e, _p = resolve_reading(PhraseReading(**kw), self.DCT)
        return s

    def test_a_future_month_takes_the_coming_year(self):
        """The measured row: planning a trip 'in June', said on 5 May 2023."""
        assert self._start(phrase="in June", kind="named_month", month=6,
                           tense="future") == datetime(2023, 6, 1)

    def test_a_future_month_already_past_this_year_rolls_over(self):
        assert self._start(phrase="in February", kind="named_month", month=2,
                           tense="future") == datetime(2024, 2, 1)

    def test_a_past_month_is_unchanged(self):
        assert self._start(phrase="in March", kind="named_month", month=3,
                           tense="past") == datetime(2023, 3, 1)

    def test_no_tense_keeps_the_old_behaviour(self):
        """A reading without tense resolves exactly as before, so a reader that
        does not report it cannot make anything worse than today."""
        assert self._start(phrase="in June", kind="named_month",
                           month=6) == datetime(2022, 6, 1)

    def test_a_future_season_takes_this_year(self):
        """"by the end of summer", said in May, is THIS summer."""
        assert self._start(phrase="end of summer", kind="named_season",
                           season="summer", tense="future") == datetime(2023, 6, 1)

    def test_last_season_ignores_a_future_tense(self):
        """"last summer" is past by its own words; a stray tense must not push it
        forward."""
        assert self._start(phrase="last summer", kind="last_season",
                           season="summer", tense="future") == datetime(2022, 6, 1)

    def test_an_explicit_year_always_wins_over_tense(self):
        assert self._start(phrase="in June 2019", kind="named_month", month=6,
                           year=2019, tense="future") == datetime(2019, 6, 1)

    def test_the_reader_is_asked_for_tense(self):
        from synap.cloud.agents.ingestion.services import date_prepass as dp
        assert '"tense"' in dp.READING_SYSTEM_PROMPT
        assert ("REQUIRED for named_month, named_season,\n             explicit_date "
                "and explicit_range") in dp.READING_SYSTEM_PROMPT
        assert "lands a year out" in dp.READING_SYSTEM_PROMPT
        # The field list is the half the model follows literally, so a bare
        # month-and-day has to be named there and in the JSON example, not only
        # in prose. See EVENT_ANCHOR_ADDENDUM for the time prose lost 15 of 15.
        assert '"tense": "past"' in dp.READING_SYSTEM_PROMPT
        assert "ALWAYS set tense when the\n                   phrase states no year" \
            in dp.READING_SYSTEM_PROMPT

    def test_tense_survives_a_raw_reading(self):
        from synap.cloud.agents.ingestion.services.date_prepass import PhraseReading
        r = PhraseReading.from_raw({"phrase": "in June", "kind": "named_month",
                                    "month": 6, "tense": "FUTURE"})
        assert r.tense == "future"


# ── the assistant was not there ───────────────────────────────────────────────

def test_a_relative_phrase_said_by_the_assistant_is_not_dated():
    """gpt4_85da3956. The user said "I just got back" from the festival on
    2023-07-15; the assistant replied "I didn't know you went three weeks ago",
    and that invented phrase was the only datable one, so the festival was
    stored as 2023-06-24 and the reader answered six weeks instead of three."""
    table = resolve_all(
        [PhraseReading(phrase="three weeks ago", kind="n_weeks_ago", n=3,
                       said_by="assistant")],
        datetime(2023, 7, 15))
    assert table["three weeks ago"]["event_start"] is None
    assert table["three weeks ago"]["assistant_only"] is True


def test_the_same_phrase_from_the_user_is_dated_as_before():
    table = resolve_all(
        [PhraseReading(phrase="three weeks ago", kind="n_weeks_ago", n=3,
                       said_by="user")],
        datetime(2023, 7, 15))
    assert table["three weeks ago"]["event_start"].date() == datetime(2023, 6, 24).date()


def test_a_date_the_assistant_states_outright_is_still_used():
    """An agent reading a record ("your order shipped on 3 July") is not doing
    arithmetic, so only the counted-from-today kinds are refused."""
    table = resolve_all(
        [PhraseReading(phrase="on July 3rd", kind="explicit_date", date="2023-07-03",
                       tense="past", said_by="assistant")],
        datetime(2023, 7, 15))
    assert table["on July 3rd"]["event_start"].date() == datetime(2023, 7, 3).date()


def test_the_block_tells_the_extractor_whose_words_those_were():
    block = render_resolved_block(resolve_all(
        [PhraseReading(phrase="three weeks ago", kind="n_weeks_ago", n=3,
                       said_by="assistant")],
        datetime(2023, 7, 15)))
    assert "SAID BY THE ASSISTANT" in block
    assert "three weeks ago" in block
    assert "Do NOT date anything the user did from it." in block


@pytest.mark.asyncio
async def test_run_prepass_returns_an_empty_table_when_resolving_raises(monkeypatch):
    """The docstring says "Never raises" and stage 3 depends on it. A bug in the
    resolver (not only a bad model reply) must also come back as an empty table."""
    from synap.cloud.agents.ingestion.services import date_prepass

    class Fine:
        async def complete(self, **kw):
            return {"content": '{"phrases": []}'}

    def boom(readings, dct):
        raise ValueError("resolver bug")

    monkeypatch.setattr(date_prepass, "resolve_all", boom)
    assert await date_prepass.run_prepass(content="x", dct=DCT, llm_client=Fine()) == {}
