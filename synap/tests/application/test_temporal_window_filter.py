"""The question's period, the arithmetic over it, and what it is never allowed to drop.

Every date here is checked by hand against a real question in the LongMemEval
lme3 cohort, so a failure means the behaviour changed, not that a fixture drifted.
"""
import os
from datetime import datetime, timezone

import pytest

from synap.cloud.application.client_api.temporal_question import (
    item_is_outside,
    parse_reading,
    resolve_window,
)

# 0bc8ad93 was asked at this moment and says "two months ago".
ASKED = datetime(2023, 3, 11, 21, 26, tzinfo=timezone.utc)


def _r(**kw):
    base = {"restricts_to_period": True, "kind": "none", "unit": None, "n": None,
            "month": None, "year": None, "weekday": None, "date": None, "phrase": ""}
    base.update(kw)
    return base


class _Item:
    def __init__(self, start, end=None, said_at=None):
        self.event_date = start
        self.event_end = end
        self.said_at = said_at


def _d(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


class TestTheQuestionMustRestrictSomething:
    def test_a_question_asking_for_a_duration_yields_no_window(self):
        # "How many weeks ago did I attend the baking class?" names a period only
        # in the sense that the ANSWER is one. 982b5123 is the case that proves
        # filtering these is wrong: its evidence sits outside any such window.
        assert resolve_window(_r(restricts_to_period=False), ASKED) is None

    def test_an_empty_reading_yields_no_window(self):
        assert resolve_window({}, ASKED) is None
        assert resolve_window(None, ASKED) is None

    def test_an_unknown_kind_yields_no_window(self):
        assert resolve_window(_r(kind="vibes"), ASKED) is None


class TestTheArithmetic:
    def test_two_months_ago_spans_the_month_either_side(self):
        # The answer-bearing memory on 0bc8ad93 is month-precision January 2023.
        # A single-day window would exclude it, so the window is deliberately wide.
        w = resolve_window(_r(kind="relative_past", unit="month", n=2), ASKED)
        assert (w.start.date().isoformat(), w.end.date().isoformat()) == ("2022-12-11", "2023-02-11")
        assert w.precision == "month"

    def test_a_named_month_with_no_year_takes_the_most_recent_one(self):
        w = resolve_window(_r(kind="named_month", month=3), ASKED)
        assert (w.start.date().isoformat(), w.end.date().isoformat()) == ("2023-03-01", "2023-03-31")

    def test_a_named_month_still_ahead_this_year_takes_last_year(self):
        # Asked in March, "in December" means last December, not this one.
        w = resolve_window(_r(kind="named_month", month=12), ASKED)
        assert w.start.year == 2022 and w.end.date().isoformat() == "2022-12-31"

    def test_last_weekday_never_resolves_to_today(self):
        # ASKED is itself a Saturday. "Last Saturday" is the one before.
        assert ASKED.strftime("%A") == "Saturday"
        w = resolve_window(_r(kind="weekday_past", weekday=5), ASKED)
        assert w.start.date().isoformat() == "2023-03-04"

    def test_the_past_two_weeks_is_a_span_ending_today(self):
        w = resolve_window(_r(kind="relative_past", unit="week", n=2), ASKED)
        assert w.end.date() == ASKED.date()
        # 14 days back is 25 Feb; the window carries three days of slack past it.
        assert w.start.date().isoformat() == "2023-02-22"

    def test_a_week_window_keeps_a_memory_three_days_past_the_boundary(self):
        # gpt4_59149c78: asked 2023-02-01, "two weeks ago", and the Met visit that
        # answers it is dated 2023-01-15. A one-day pad deleted it by two days.
        asked = datetime(2023, 2, 1, tzinfo=timezone.utc)
        w = resolve_window(_r(kind="relative_past", unit="week", n=2), asked)
        assert w.start.date().isoformat() == "2023-01-15"
        assert item_is_outside(_Item(_d(2023, 1, 15)), w) is False

    def test_month_arithmetic_clamps_a_day_the_target_month_lacks(self):
        # 31 March minus one month must not produce 31 February.
        w = resolve_window(_r(kind="relative_past", unit="month", n=1), _d(2023, 3, 31))
        assert w is not None

    def test_a_naive_query_date_is_treated_as_utc(self):
        w = resolve_window(_r(kind="named_month", month=3), datetime(2023, 3, 11))
        assert w is not None and w.start.tzinfo is not None


class TestWhatIsNeverDropped:
    WINDOW = resolve_window(_r(kind="relative_past", unit="month", n=2), ASKED)

    def test_an_item_with_no_date_is_kept(self):
        assert item_is_outside(_Item(None), self.WINDOW) is False

    def test_an_item_whose_only_date_was_the_conversations_is_kept(self):
        # 982b5123: "had to book three months in advance" carries no real date,
        # and is the memory the answer needs. It used to arrive here carrying
        # the session date with `event_date_is_mention` set, so this function
        # had to test that flag to avoid dropping it; `event_date` is now
        # simply empty and the no-date test above catches it.
        kept = _Item(None, said_at=_d(2023, 5, 27))
        assert item_is_outside(kept, self.WINDOW) is False

    def test_the_same_date_IS_dropped_once_it_is_real(self):
        assert item_is_outside(_Item(_d(2022, 1, 1)), self.WINDOW) is True

    def test_an_overlapping_window_is_kept(self):
        # Month-precision January 2023 overlaps 11 Dec .. 11 Feb.
        jan = _Item(_d(2023, 1, 1), _d(2023, 1, 31))
        assert item_is_outside(jan, self.WINDOW) is False

    def test_an_item_touching_the_boundary_is_kept(self):
        assert item_is_outside(_Item(self.WINDOW.end), self.WINDOW) is False
        assert item_is_outside(_Item(self.WINDOW.start), self.WINDOW) is False

    def test_no_window_means_nothing_is_outside(self):
        assert item_is_outside(_Item(_d(1999, 1, 1)), None) is False

    def test_naive_item_dates_do_not_raise(self):
        assert item_is_outside(_Item(datetime(2022, 1, 1)), self.WINDOW) is True


class TestTheRealZeroBcOneCase:
    """0bc8ad93, with the dates actually stored for it."""

    WINDOW = resolve_window(
        _r(kind="relative_past", unit="month", n=2, phrase="two months ago"), ASKED)

    def test_the_january_answer_survives(self):
        assert item_is_outside(_Item(_d(2023, 1, 1), _d(2023, 1, 31)), self.WINDOW) is False

    def test_the_february_distractor_is_dropped(self):
        # 17 Feb is three weeks before the question, not two months.
        assert item_is_outside(_Item(_d(2023, 2, 17)), self.WINDOW) is True

    def test_the_october_distractor_is_kept_because_it_has_no_real_date(self):
        # Honest about the limit: this one SHOULD go and cannot, because
        # extraction stored no real date for it. It was spoken about on
        # 22 Oct 2022, which is now carried as said_at and not as an event date.
        assert item_is_outside(_Item(None, said_at=_d(2022, 10, 22)),
                               self.WINDOW) is False


class TestReadingParser:
    def test_a_bare_object(self):
        assert parse_reading('{"restricts_to_period": false}') == {"restricts_to_period": False}

    def test_a_fenced_object(self):
        assert parse_reading('```json\n{"restricts_to_period": true}\n```')["restricts_to_period"] is True

    def test_prose_around_an_object(self):
        assert parse_reading('Sure!\n{"restricts_to_period": true}\nHope that helps')["restricts_to_period"] is True

    def test_rubbish_is_none(self):
        for bad in ("", None, "no json here", "[1,2,3]"):
            assert parse_reading(bad) is None


class TestTheCacheMustNotLoseTheSaidDate:
    """Found by the benchmark team, 2026-10-01, on the field that preceded this.

    The retrieval cache round-trips every item through to_dict/from_dict. The
    `event_date_is_mention` flag was missing from to_dict, so a cache hit turned
    a borrowed date into a real one: the prefix rendered bare on a hit and
    "(said ...)" live for the same memory inside the 300s TTL, and the filter
    lost the one property that kept it safe.

    That flag is gone, because the read path no longer borrows the date in the
    first place. The day the words were said now travels as `said_at`, and the
    same round trip has to carry it or a cache hit loses the date entirely.
    """

    def test_the_said_date_survives_a_round_trip(self):
        from synap.cloud.agents.models.bundles import ContextItem
        a = ContextItem(item_id="x", content="c", context_type="facts", source="vector",
                        said_at=_d(2022, 10, 22))
        back = ContextItem.from_dict(a.to_dict())
        assert back.said_at == _d(2022, 10, 22)
        assert back.event_date is None

    def test_a_real_event_date_also_survives(self):
        from synap.cloud.agents.models.bundles import ContextItem
        a = ContextItem(item_id="x", content="c", context_type="facts", source="vector",
                        event_date=_d(2022, 10, 22), said_at=_d(2023, 5, 27))
        back = ContextItem.from_dict(a.to_dict())
        assert back.event_date == _d(2022, 10, 22)
        assert back.said_at == _d(2023, 5, 27)

    def test_a_cache_entry_with_no_said_at_does_not_crash(self):
        from synap.cloud.agents.models.bundles import ContextItem
        a = ContextItem(item_id="x", content="c", context_type="facts", source="vector",
                        event_date=_d(2022, 10, 22))
        legacy = {k: v for k, v in a.to_dict().items() if k != "said_at"}
        assert ContextItem.from_dict(legacy).said_at is None

    def test_an_item_with_no_event_date_is_never_droppable_through_the_cache(self):
        # The consequence the old flag existed for, restated: with the date no
        # longer borrowed, there is nothing for a cache hit to promote into a
        # real event date, on either side of the round trip.
        from synap.cloud.agents.models.bundles import ContextItem
        win = resolve_window(_r(kind="relative_past", unit="month", n=2), ASKED)
        a = ContextItem(item_id="x", content="c", context_type="facts", source="vector",
                        said_at=_d(2022, 1, 1))
        assert item_is_outside(a, win) is False
        assert item_is_outside(ContextItem.from_dict(a.to_dict()), win) is False
        legacy = {k: v for k, v in a.to_dict().items() if k != "said_at"}
        assert item_is_outside(ContextItem.from_dict(legacy), win) is False
