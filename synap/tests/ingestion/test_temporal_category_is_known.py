"""`temporal_category` may only ever be one of three strings.

WHY THIS EXISTS. The extractor once invented a `temporal_category` longer than
the column it is written into (character varying(20)). The write failed after
the rest of a document's work was done, and one scope lost both of its sessions.
Other invented values were short enough to be stored silently, so the crash was
the visible half of a quieter problem.

The values exercised here are real ones the extractor produced, kept in
fixtures/invented_temporal_categories.json. None was made up.
"""

import json
import uuid
from pathlib import Path

import pytest

from synap.cloud.agents.ingestion.models.enums import MemoryDestination, MemoryType
from synap.cloud.agents.ingestion.models.outputs import (
    TEMPORAL_CATEGORIES,
    ExtractedMemory,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "invented_temporal_categories.json").read_text()
)
PROD_INVALID = sorted(FIXTURE["invalid"])
PROD_VALID = sorted(FIXTURE["valid"])

# The column the value is written into. Nothing longer than this can be stored,
# and the fix is only a fix if it can never produce something longer.
COLUMN_WIDTH = 20


def _memory(**over):
    kwargs = dict(
        memory_id=uuid.uuid4(),
        memory_type=MemoryType.FACTS,
        content="The user's friend Rachel has moved back to the suburbs.",
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="longterm:830ce83f_answer_0b1a0942_1",
    )
    kwargs.update(over)
    return ExtractedMemory(**kwargs)


def test_the_allowed_set_is_exactly_three():
    assert TEMPORAL_CATEGORIES == ("perpetual", "temporal_fact", "episode")


@pytest.mark.parametrize("value", PROD_VALID)
def test_every_valid_prod_value_survives_untouched(value):
    assert _memory(temporal_category=value).temporal_category == value


@pytest.mark.parametrize("value", PROD_INVALID)
def test_every_invented_prod_value_becomes_the_default(value):
    """All 34 of them, including 'PAST' and 'once or twice a week'."""
    assert _memory(temporal_category=value).temporal_category == "perpetual"


@pytest.mark.parametrize("value", PROD_INVALID + PROD_VALID)
def test_nothing_can_exceed_the_column_width(value):
    assert len(_memory(temporal_category=value).temporal_category) <= COLUMN_WIDTH


def test_the_value_that_killed_the_document_no_longer_kills_it():
    """A category longer than the column must not raise, and must not persist.

    The exact string is not recoverable from the log, which truncated the
    parameter list, so this uses the longest shape the same extractor produced
    ("once or twice a week", 20 characters) pushed past the limit the same way
    the real one was.
    """
    too_long = "once or twice a week, sometimes more"
    assert len(too_long) > COLUMN_WIDTH
    memory = _memory(temporal_category=too_long)
    assert memory.temporal_category == "perpetual"
    assert len(memory.temporal_category) <= COLUMN_WIDTH


def test_case_and_space_are_forgiven_for_real_values():
    """The legacy text parser lowercases, the JSON parser does not."""
    for raw, want in (
        ("Episode", "episode"),
        ("  temporal_fact  ", "temporal_fact"),
        ("PERPETUAL", "perpetual"),
    ):
        assert _memory(temporal_category=raw).temporal_category == want


def test_an_explicit_null_becomes_the_default():
    """Same treatment a missing key gets, which is what the 2026-10-05 prompt
    change made common: every key present 'with null where you do not know'."""
    assert _memory(temporal_category=None).temporal_category == "perpetual"


def test_a_missing_key_still_defaults():
    assert _memory().temporal_category == "perpetual"


def test_an_invented_value_is_logged_so_it_shows_up_as_a_rate(caplog):
    with caplog.at_level("WARNING"):
        _memory(temporal_category="once or twice a week")
    assert "extraction_temporal_category_invalid" in caplog.text
    assert "once or twice a week" in caplog.text


def test_a_valid_value_is_not_logged(caplog):
    with caplog.at_level("WARNING"):
        _memory(temporal_category="episode")
    assert "extraction_temporal_category_invalid" not in caplog.text


def test_is_episode_is_still_derivable_after_normalisation():
    """'episode' is the only value that may drive is_episode, and the three
    survivors keep that distinction intact."""
    assert _memory(temporal_category="episode").temporal_category == "episode"
    for value in PROD_INVALID:
        assert _memory(temporal_category=value).temporal_category != "episode"
