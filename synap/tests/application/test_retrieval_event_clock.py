"""An event_date is an event date, or it is empty. It is never the day the
words were said standing in for one.

The read path used to fill `event_date` with the source document's own date
whenever ingestion had stored no event window and no validity start -- which is
the documented, correct thing for ingestion to store when the turn gives no
time at all ("the user likes jazz", "recently", "the other day"). So the field
was never empty: on the 500-question run, 5,601 of 10,443 served items (53.6%)
carried the conversation's own clock in it, and a client had no way to tell
those from the 35.4% that were real. The substitution was flagged
`event_date_is_mention` so four separate readers could each remember to
distrust it; this suite exists so the substitution cannot come back under the
flag's successor.

The day the words were said keeps its own field, `said_at`, and ranking still
prefers it to our ingest clock -- which is what the deleted `_event_anchor`
helper was for. Reading the corpus by ingest time mis-ranked 31% of it
(measured 2026-09-08), so that precedence is asserted here too.
"""
import inspect
from datetime import datetime

from synap.cloud.application.client_api import retrieval_manager
from synap.cloud.application.client_api.retrieval_manager import _event_fields

EVENT_START = "2026-06-01T00:00:00"
VF = "2026-08-01T00:00:00"
DCT = "2026-07-01T09:30:00"
ROW_WRITTEN = "2026-10-09T21:04:00"


def _fields(**row):
    return _event_fields(lambda k: row.get(k))


def test_a_stored_event_window_is_the_event_date():
    out = _fields(event_start=EVENT_START, valid_from=VF, document_created_at=DCT)
    assert out["event_date"] == datetime(2026, 6, 1)


def test_a_validity_start_is_NOT_an_event_date_when_there_is_no_window():
    """This test used to assert the opposite, and the opposite was the bug.

    It said a `valid_from` was "a date somebody stated" and so was a fine event
    date. It is not: `valid_from` is when a fact BEGAN BEING TRUE, and ingestion
    sets it to the conversation's date for almost every time-bound fact. On a
    25-question benchmark run, 140 of 140 time-bound facts with no event window
    carried a `valid_from` equal to the day of the conversation, and all 140 were
    served as "happened <that day>": "has been doing 30-minute walks 3 times a
    week", "once spent $60 on coffee mugs", "is planning a private wine and
    cheese tasting". None of those happened on the day they were mentioned.

    What the old test protected, a stated date reaching the client, is still
    protected, but through `event_start`: see
    test_a_stored_event_window_is_the_event_date. The day the words were said
    still reaches the client too, under its own name `said_at`.
    """
    out = _fields(valid_from=VF, document_created_at=DCT)
    assert out["event_date"] is None
    assert out["event_end"] is None and out["event_precision"] is None
    assert out["said_at"] == datetime(2026, 7, 1, 9, 30)


def test_valid_from_equal_to_the_conversation_day_is_not_served_as_an_event():
    """The measured shape: valid_from == the day of the conversation, no window."""
    out = _fields(valid_from=DCT, document_created_at=DCT)
    assert out["event_date"] is None
    assert out["said_at"] == datetime(2026, 7, 1, 9, 30)


def test_a_window_still_wins_and_valid_from_never_leaks_into_it():
    """With a window, event_date is the window's start whatever valid_from says,
    and the end and precision come through unchanged."""
    out = _fields(event_start=EVENT_START, event_end="2026-06-30T23:59:59",
                  event_precision="month", valid_from=VF, document_created_at=DCT)
    assert out["event_date"] == datetime(2026, 6, 1)
    assert out["event_end"] == datetime(2026, 6, 30, 23, 59, 59)
    assert out["event_precision"] == "month"
    assert out["said_at"] == datetime(2026, 7, 1, 9, 30)


def test_no_window_and_no_validity_start_means_NO_event_date():
    # The whole point. This is the 53.6%-of-served-items case.
    out = _fields(document_created_at=DCT, created_at=ROW_WRITTEN)
    assert out["event_date"] is None


def test_the_day_it_was_said_is_still_carried_separately():
    out = _fields(document_created_at=DCT, created_at=ROW_WRITTEN)
    assert out["said_at"] == datetime(2026, 7, 1, 9, 30)
    # and it is not quietly the same object as the (absent) event date
    assert out["event_date"] is None


def test_the_substitution_flag_is_gone_not_renamed():
    out = _fields(document_created_at=DCT)
    assert "event_date_is_mention" not in out
    src = inspect.getsource(_event_fields)
    assert "document_created_at" in src, "said_at still reads the document date"
    assert 'get("document_created_at")' not in src.split("start =", 1)[1], (
        "event_date is reaching for the document date again"
    )


def test_ranking_prefers_when_it_was_said_over_when_we_wrote_the_row():
    # event_date, then said_at, then created_at. said_at is a real date the
    # client gave us; created_at is our own ingest clock.
    src = inspect.getsource(retrieval_manager)
    # the statement, not up to the first ")" -- getattr() has one of its own
    block = "\n".join(
        src.split("temporal_anchor = ", 1)[1].splitlines()[:3]
    )
    assert "item.event_date" in block
    assert "said_at" in block
    assert "item.created_at" in block
    assert block.index("item.event_date") < block.index("said_at") < block.index("item.created_at")


def test_the_day_it_was_said_is_never_the_minute_we_extracted():
    """`said_at` used to fall back to `created_at`.

    The reasoning in the code was that "on the vector path created_at carries
    the document date too", and it was wrong: every Chroma writer sets
    `metadata["created_at"] = memory.extracted_at` (vector_wrapper.py:395, :487
    and :849), so on both paths `created_at` is the minute WE ran extraction.
    Serving it answered "when did I say this" with a timestamp from our own
    pipeline, which is the one date the client has no way to recognise as ours.

    A transcript ingested with no dates
    anywhere used to serve an `extracted_at` of the ingest day on every item for a
    2023 conversation. `said_at` must be empty there instead.
    """
    out = _fields(created_at=ROW_WRITTEN)
    assert out["said_at"] is None
    src = inspect.getsource(_event_fields)
    said_stmt = src.split("said = ", 1)[1].split("\n", 1)[0]
    # The bare field, not the substring inside `document_created_at`.
    assert 'get("created_at")' not in said_stmt, (
        "said_at is reaching for the extraction clock again: %s" % said_stmt
    )


def test_the_day_it_was_said_is_the_document_s_own_date_when_there_is_one():
    out = _fields(document_created_at=DCT, created_at=ROW_WRITTEN)
    assert out["said_at"] == datetime(2026, 7, 1, 9, 30)
