"""Tests for the named-entity verbatim guardrail (KU v2 merge follow-up).

The guardrail rejects a merge if any named entity, quoted phrase, or
literal numeric token from any contributor is missing (case-sensitive
substring) from the merged content. This file tests:

  1. extract_must_preserve_tokens — pure regex extractor
  2. find_dropped_entities       — diff helper
  3. synthesize_merges integration — guardrail-tripping cluster gets skipped
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    EnrichmentAction,
    KnowledgeUpdateDetector,
    extract_must_preserve_tokens,
    find_dropped_entities,
    record_identifiers,
)


# ---------------------------------------------------------------------------
# 1. extract_must_preserve_tokens
# ---------------------------------------------------------------------------


def test_extracts_quoted_strings():
    text = 'Melanie loved reading "Charlotte\'s Web" as a child.'
    tokens = extract_must_preserve_tokens(text)
    assert "Charlotte's Web" in tokens


def test_extracts_smart_quoted_strings():
    text = 'The piece was called “Finding Freedom” and won first place.'
    tokens = extract_must_preserve_tokens(text)
    assert "Finding Freedom" in tokens


def test_extracts_multi_word_proper_nouns():
    text = "Caroline moved from Stockholm to the Bay Area in 2018."
    tokens = extract_must_preserve_tokens(text)
    # "Bay Area" is a multi-word capitalized phrase
    assert "Bay Area" in tokens


def test_extracts_single_word_proper_nouns():
    text = "Caroline mentors a transgender teen named Riley."
    tokens = extract_must_preserve_tokens(text)
    assert "Caroline" in tokens
    assert "Riley" in tokens


def test_extracts_product_names_with_digits():
    text = "She started with a Nikon D40 and upgraded to a Sony A7 III."
    tokens = extract_must_preserve_tokens(text)
    assert "Nikon D40" in tokens
    assert "Sony A7 III" in tokens


def test_extracts_camelcase_brands():
    text = "DoorDash laid Gina off in January."
    tokens = extract_must_preserve_tokens(text)
    assert "DoorDash" in tokens


def test_extracts_lowercase_i_brand():
    text = "She just got the iPhone 14 Pro."
    tokens = extract_must_preserve_tokens(text)
    # Either "iPhone" alone or "iPhone 14 Pro" should appear as a token.
    assert any("iPhone" in t for t in tokens)


def test_extracts_numeric_patterns():
    text = "The user practices yoga 3x/week and spent $120 on the new mat."
    tokens = extract_must_preserve_tokens(text)
    assert "3x/week" in tokens
    assert "$120" in tokens


def test_extracts_blood_pressure_pattern():
    text = "Blood pressure reading is 120/80, down from 140/90 last month."
    tokens = extract_must_preserve_tokens(text)
    assert "120/80" in tokens
    assert "140/90" in tokens


def test_extracts_year():
    text = "She started photography in 2018 and entered her first show in 2023."
    tokens = extract_must_preserve_tokens(text)
    assert "2018" in tokens
    assert "2023" in tokens


def test_filters_sentence_starter_when_short():
    text = "The user enjoys hiking on weekends."
    tokens = extract_must_preserve_tokens(text)
    # "The" is in the stoplist so shouldn't appear, and short sentence-initial
    # capitalized words shouldn't either.
    assert "The" not in tokens


def test_filters_pronouns_and_days():
    text = "On Monday she went to the park. They had ice cream."
    tokens = extract_must_preserve_tokens(text)
    assert "Monday" not in tokens, "days are stoplisted"
    assert "She" not in tokens
    assert "They" not in tokens


def test_skips_context_brackets():
    text = "Caroline is starting therapy.\n\n[Context: person:Caroline]"
    tokens = extract_must_preserve_tokens(text)
    # Internal context tag should NOT be required in merged content.
    assert "Context: person:Caroline" not in tokens
    assert "Caroline" in tokens


def test_handles_empty_or_none_text():
    assert extract_must_preserve_tokens("") == set()
    assert extract_must_preserve_tokens(None) == set()


# ---------------------------------------------------------------------------
# 2. find_dropped_entities
# ---------------------------------------------------------------------------


def test_no_dropped_when_merged_carries_all_entities():
    contributors = [
        "Melanie loved reading \"Charlotte's Web\" as a child.",
        "Melanie's favorite childhood book was \"Charlotte's Web\".",
    ]
    merged = (
        "Melanie loved reading \"Charlotte's Web\" as a child and considers "
        "it her favorite childhood book."
    )
    assert find_dropped_entities(contributors, merged) == set()


def test_dropped_when_title_paraphrased_away():
    """The classic failure mode: 'Charlotte's Web' becomes 'the classic'."""
    contributors = [
        "Melanie loved reading \"Charlotte's Web\" as a child.",
        "Melanie's favorite childhood book was \"Charlotte's Web\".",
    ]
    merged = "Melanie's favorite childhood book is a children's classic she loves to revisit."
    dropped = find_dropped_entities(contributors, merged)
    assert "Charlotte's Web" in dropped


def test_dropped_when_product_name_generalized():
    contributors = [
        "She started with a Nikon D40 in 2018.",
        "Her camera collection began with a Nikon D40.",
    ]
    merged = "She started photography with her first DSLR camera in 2018."
    dropped = find_dropped_entities(contributors, merged)
    assert "Nikon D40" in dropped


def test_dropped_when_year_dropped():
    contributors = [
        "She started photography in 2018.",
        "Her photography hobby began with a Nikon D40 in 2018.",
    ]
    merged = "She started photography with a Nikon D40 several years ago."
    dropped = find_dropped_entities(contributors, merged)
    assert "2018" in dropped


def test_dropped_includes_only_what_is_actually_missing():
    """Verify the diff is precise — entities present in merged shouldn't be flagged."""
    contributors = [
        "Caroline lives in Stockholm and works at Spotify.",
        "Caroline moved to Stockholm five years ago.",
    ]
    merged = "Caroline lives in Stockholm, having moved there five years ago."
    dropped = find_dropped_entities(contributors, merged)
    # Spotify was dropped; Stockholm + Caroline preserved.
    assert dropped == {"Spotify"}


def test_dropped_empty_for_merged_content_none():
    assert find_dropped_entities(["anything"], "") == set()
    assert find_dropped_entities([""], "") == set()


# ---------------------------------------------------------------------------
# 2b. D-032 — false positives the guard used to report (a small fraction of alive
#     memories, all this shape), and the record-id family Nap needs.
#     Every "not falsely dropped" case below is proven a false positive by the
#     self-merge invariant: merging a memory into ITSELF cannot lose anything.
# ---------------------------------------------------------------------------


def test_possessive_phrase_not_falsely_dropped():
    """A proper-noun phrase spanning a possessive ("Sarah's Bakery") is
    extracted possessive-stripped ("Sarah Bakery"); the presence test must not
    then report it dropped from content that keeps the possessive. Mutation
    guard: the pre-fix substring test returns {"Sarah Bakery"} here."""
    contributor = "The user's favourite spot is Sarah's Bakery on Oak Street."
    merged = "The user often visits Sarah's Bakery on Oak Street for coffee."
    assert find_dropped_entities([contributor], merged) == set()


def test_nonbreaking_space_not_falsely_dropped():
    """Content carrying U+202F (narrow no-break space) between the words of a
    name must still match a phrase joined on a plain space."""
    c = "The user tracks the Nifty 50 Index and reads the New York Times."
    assert find_dropped_entities([c], c) == set()


def test_redaction_tag_join_not_falsely_dropped():
    """An internal [redaction] tag between two words makes the extractor join
    them into a phrase ("Motilal Midcap") that is not a verbatim substring of
    the raw content; that is not data loss and must not be reported."""
    c = "Review of Motilal [a Person's name was given and not kept] Midcap fund."
    assert find_dropped_entities([c], c) == set()


def test_self_merge_never_reports_a_drop():
    """The invariant behind the measurement: for any real memory, merging it
    into itself drops nothing. Any non-empty result is a pure false positive."""
    corpus = [
        "Meera Krishnan Nair's Aadhaar number is 7890 1234 5678.",
        "The user holds a Motilal [a name kept out] Nifty 500 Index Fund.",
        'Vanguard S&P 500 ETF (VOO) 5-year return is ~15.66%.',
        "Shailee requested a refund for 'The52 Women's Beautiful Kurta'.",
        'The Nightingale became a New York Times bestseller in 2015.',
        "Order #844908 placed for 'Sharmayu Ayurveda Virya Stambhan Bati'.",
    ]
    for c in corpus:
        assert find_dropped_entities([c], c) == set(), c


def test_real_drop_still_caught_after_normalisation():
    """The leniency must be one-directional: a genuinely paraphrased-away title
    is still reported even though the target is possessive-normalised."""
    contributor = "Melanie's favourite book was \"Charlotte's Web\"."
    merged = "Melanie's favourite book is a beloved children's classic."
    assert "Charlotte's Web" in find_dropped_entities([contributor], merged)


def test_single_quoted_phrase_dropped_quotes_not_falsely_dropped():
    """A phrase single-quoted for emphasis in the source ('Dermadew Glow Face
    Wash', 'In Transit') is captured by the proper-noun scan with the closing
    quote glued on (_QUOTED_SPAN_RE only lifts DOUBLE-quoted spans, and
    _GUARD_WORD_RE allows a trailing quote). A merge that writes the phrase
    without the decorative quotes preserves the entity, so it must NOT be
    reported dropped. Mutation guard: before the edge-quote normalisation the
    token "...Wash'" is not a substring of the quote-free merge and this returns
    a non-empty set. This case used to defer redundant
    deep-clean merges into escalated adjudication cases."""
    contributor = "The order is for 'Dermadew Glow Face Wash' and is 'In Transit'."
    merged = "The order is for Dermadew Glow Face Wash and is in transit."
    assert find_dropped_entities([contributor], merged) == set()


def test_single_quoted_phrase_genuinely_dropped_still_caught():
    """The edge-quote leniency is one-directional, same as the possessive one:
    if the single-quoted phrase is actually paraphrased away, it is still
    reported. Normalising the haystack cannot conjure a needle."""
    contributor = "The order is for 'Dermadew Glow Face Wash' and is 'In Transit'."
    merged = "The order is for a face wash product that is on its way."
    assert find_dropped_entities([contributor], merged), \
        "a genuinely dropped quoted phrase must still be caught"


# ---------------------------------------------------------------------------
# 2c. Record-identifier family — a merge that drops a record id has collapsed
#     two distinct records; that is the order-collapse the fast path had.
# ---------------------------------------------------------------------------


def test_record_identifiers_extractor():
    ids = record_identifiers("Order #844908 refunded; ticket 833137 opened.")
    assert ids == {"844908", "833137"}
    # 4-digit years are numeric anchors, never record ids
    assert record_identifiers("She joined in 2018 and left in 2023.") == set()
    # '#' and spacing normalise away so "#844908" == "844908"
    assert record_identifiers("#844908") == record_identifiers("844908")


def test_record_id_family_blocks_distinct_records():
    """Two contributors about DIFFERENT orders; whatever the merge says, both
    order ids cannot survive, so the merge must be refused."""
    contributors = [
        "Order #844908 placed for the Ayurveda item; status processing.",
        "Order #833137 placed for the Ayurveda item; status processing.",
    ]
    merged = "Order #844909 placed for the Ayurveda item; status processing."
    dropped = find_dropped_entities(contributors, merged)
    assert "#844908" in dropped and "#833137" in dropped


def test_record_id_family_allows_same_record():
    """Two memories about the SAME order restated — the id survives, so the
    id family adds no refusal (the merge may still proceed)."""
    contributors = [
        "Order #844908 was placed on September 2.",
        "Order #844908 shipped the next day.",
    ]
    merged = "Order #844908 was placed on September 2 and shipped the next day."
    assert find_dropped_entities(contributors, merged) == set()


# ---------------------------------------------------------------------------
# 3. synthesize_merges integration — guardrail must reject lossy merges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_merges_skips_when_entity_dropped():
    """End-to-end: cluster is valid, length guardrail passes, but the
    merge LLM dropped a named entity → cluster must NOT produce a MergeAction."""
    detector = KnowledgeUpdateDetector(
        vector=AsyncMock(),
        llm_client=AsyncMock(),
    )
    a, b = str(uuid4()), str(uuid4())
    enrichments = [
        EnrichmentAction(
            existing_memory_id=a,
            existing_memory_content="Melanie loved reading \"Charlotte's Web\" as a child.",
            new_memory_id=b,
            new_memory_content="Melanie's favorite childhood book was \"Charlotte's Web\".",
            reason="adds favorite-status detail",
            memory_type="facts",
            new_memory_type="facts",
        ),
    ]
    # Mock LLM produces a long-enough but entity-dropping merge.
    bad_merge = (
        "Melanie has a deep nostalgic connection to a beloved children's book "
        "she read repeatedly as a kid and still considers her favorite from "
        "childhood, often associated with feelings of comfort and warmth."
    )
    detector._call_merge_llm = AsyncMock(
        return_value={
            "merged_content": bad_merge,
            "rationale": "combined nostalgic + favorite",
        }
    )

    scope = Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )
    result = await detector.synthesize_merges(enrichments=enrichments, scope=scope)
    assert result == [], "merge that dropped 'Charlotte's Web' must be rejected"


@pytest.mark.asyncio
async def test_synthesize_merges_proceeds_when_entities_preserved():
    """Sanity: a merge that DOES carry every entity verbatim still produces
    a MergeAction (proves the guardrail doesn't false-reject good merges)."""
    detector = KnowledgeUpdateDetector(
        vector=AsyncMock(),
        llm_client=AsyncMock(),
    )
    a, b = str(uuid4()), str(uuid4())
    enrichments = [
        EnrichmentAction(
            existing_memory_id=a,
            existing_memory_content="Melanie loved reading \"Charlotte's Web\" as a child.",
            new_memory_id=b,
            new_memory_content="Melanie's favorite childhood book was \"Charlotte's Web\".",
            reason="adds favorite-status detail",
            memory_type="facts",
            new_memory_type="facts",
        ),
    ]
    good_merge = (
        "Melanie loved reading \"Charlotte's Web\" as a child and considers "
        "it her favorite childhood book."
    )
    detector._call_merge_llm = AsyncMock(
        return_value={
            "merged_content": good_merge,
            "rationale": "combined love-of-reading and favorite-status",
        }
    )

    scope = Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )
    result = await detector.synthesize_merges(enrichments=enrichments, scope=scope)
    assert len(result) == 1
    assert "Charlotte's Web" in result[0].merged_content


@pytest.mark.asyncio
async def test_synthesize_merges_skips_when_year_dropped():
    """Numeric anchor (year) preservation — same guardrail."""
    detector = KnowledgeUpdateDetector(
        vector=AsyncMock(),
        llm_client=AsyncMock(),
    )
    a, b = str(uuid4()), str(uuid4())
    enrichments = [
        EnrichmentAction(
            existing_memory_id=a,
            existing_memory_content="The user started photography in 2018 with a Nikon D40.",
            new_memory_id=b,
            new_memory_content="The user has been taking photos since 2018.",
            reason="adds equipment detail",
            memory_type="facts",
            new_memory_type="facts",
        ),
    ]
    detector._call_merge_llm = AsyncMock(
        return_value={
            "merged_content": (
                "The user has been an avid photographer for years, having "
                "started with her first DSLR camera and continued the hobby ever since."
            ),
            "rationale": "combined start-date and equipment",
        }
    )
    scope = Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )
    result = await detector.synthesize_merges(enrichments=enrichments, scope=scope)
    assert result == [], "merge that dropped '2018' and 'Nikon D40' must be rejected"


# ---------------------------------------------------------------------------
# 4. Quantities. Measured on the exact details the LME3 failures turn on: the
#    guard protected the proper nouns beside every one of these numbers and not
#    one of the numbers themselves, and "needs a total of 300 points" had an
#    EMPTY protected set, so a merge could have rewritten it to anything.
# ---------------------------------------------------------------------------

MEASURED_FAILURES = [
    ("The user has 1250 followers on Instagram", "1250"),
    ("The user needs a total of 300 points to redeem a free skincare product", "300"),
    ("The user spent around 70 hours playing Assassin's Creed Odyssey", "70"),
    ("The user goes to the gym at 6:00 pm on Mondays, Wednesdays and Fridays", "6:00"),
    ("It took the user four hours to drive to the Outer Banks", "four"),
    ("The user is building a 1/16 scale German Tiger I tank", "1/16"),
]


@pytest.mark.parametrize("text,number", MEASURED_FAILURES)
def test_the_number_each_failing_question_turns_on_is_protected(text, number):
    assert number in extract_must_preserve_tokens(text)


def test_the_300_points_memory_no_longer_has_an_empty_protected_set():
    """The worst case found: not one token, so no merge of it could ever be
    refused however much it changed."""
    tokens = extract_must_preserve_tokens(
        "The user needs a total of 300 points to redeem a free skincare product")
    assert tokens, "an empty set means this memory is unguarded entirely"


@pytest.mark.parametrize("contributor,merged,label", [
    ("The user has 1250 followers on Instagram",
     "The user has 1300 followers on Instagram", "a changed count"),
    ("The user needs a total of 300 points",
     "The user needs points to redeem a product", "a dropped count"),
    ("It took the user four hours to drive",
     "It took the user a while to drive", "a dropped spelled-out quantity"),
    ("The user is building a 1/16 scale tank",
     "The user is building a scale tank", "a dropped scale"),
    ("The user goes to the gym at 6:00 pm",
     "The user goes to the gym at 7:00 pm", "a changed clock time"),
])
def test_a_silent_number_change_is_caught(contributor, merged, label):
    assert find_dropped_entities([contributor], merged), label


def test_a_quantity_is_boundary_matched_not_substring_matched():
    """The name families substring-test, which would let "170 hours" satisfy a
    contributor's "70 hours" — exactly the silent change this family exists to
    stop."""
    dropped = find_dropped_entities(
        ["The user spent around 70 hours playing"],
        "The user spent around 170 hours playing")
    assert "70" in dropped


def test_four_is_not_satisfied_by_fourteen():
    assert "four" in find_dropped_entities(
        ["It took four hours"], "It took fourteen hours")


@pytest.mark.parametrize("contributor,merged,label", [
    ("It took the user four hours to drive to the Outer Banks",
     "It took the user 4 hours to drive to the Outer Banks", "four -> 4"),
    ("The user spent 70 hours on the game",
     "The user spent seventy hours on the game", "70 -> seventy"),
    ("The user has 1250 followers on Instagram",
     "On Instagram the user has 1250 followers", "reordered, number kept"),
])
def test_rewriting_a_quantity_the_other_way_is_not_a_loss(contributor, merged, label):
    """A merge may reword. The quantity survived; only its spelling changed.
    A guard that refuses this ends up refusing everything, which is the failure
    the "4.3 Cr" note in the source already records."""
    assert find_dropped_entities([contributor], merged) == set(), label


@pytest.mark.parametrize("article", ["one", "a", "an"])
def test_articles_are_not_treated_as_quantities(article):
    """"one", "a" and "an" are articles far more often than quantities, and a
    token present in nearly every sentence guards nothing while refusing
    everything."""
    tokens = extract_must_preserve_tokens(f"The user bought {article} ticket")
    assert article not in {t.lower() for t in tokens}


# --- dates must not shred into bare numbers --------------------------------

def test_a_date_reduces_to_one_canonical_token_however_it_is_written():
    """Two things at once. The date is consumed WHOLE, so it does not shred into
    the bare quantities "04" and "15"; and it reduces to one canonical token, so
    the same day written either way is the same token."""
    iso = extract_must_preserve_tokens(
        "The user acquired a silver necklace on 2023-04-15.")
    written = extract_must_preserve_tokens(
        "The user acquired a silver necklace on April 15, 2023.")
    assert iso == {"date:2023-04-15"}
    assert written == {"date:2023-04-15"}
    assert "04" not in iso and "15" not in iso


def test_the_same_date_written_two_ways_is_not_a_loss_in_EITHER_direction():
    """The second direction used to fail: the proper-noun scan read "April 15"
    as a phrase, so restyling it as ISO reported that phrase dropped even though
    it is the same day. Canonicalising both sides is what closes it."""
    iso = "The user acquired a silver necklace on 2023-04-15."
    written = "The user acquired a silver necklace on April 15, 2023."
    assert find_dropped_entities([iso], written) == set()
    assert find_dropped_entities([written], iso) == set()


def test_a_changed_date_is_caught_at_DAY_precision_not_just_the_year():
    """The failure this family exists for: a four-digit year anchor read
    2022-03-01 and 2022-02-01 as the same thing, and meditation merged
    "finished listening" into "started listening" a month earlier."""
    a = "The user finished listening to Sapiens on 2022-03-01"
    b = "The user started listening to Sapiens on 2022-02-01."
    assert "date:2022-03-01" in find_dropped_entities([a], b)


def test_an_all_numeric_date_falls_back_to_its_year():
    """04/15/2023 is month-first or day-first depending on locale. Guessing
    would either invent losses or hide them, so it is consumed (no shredding
    into bare quantities) and only its year is protected."""
    tokens = extract_must_preserve_tokens("The trip was on 04/15/2023.")
    assert tokens == {"2023"}


def test_a_bare_year_is_still_a_year_not_a_date():
    tokens = extract_must_preserve_tokens("She joined in 2018 and left in 2023.")
    assert tokens == {"2018", "2023"}


def test_dates_false_restores_the_year_only_anchor():
    """Compression passes this, with quantities=False, so its anchor set is
    exactly what it was before either family existed."""
    iso = "The user acquired a silver necklace on 2023-04-15."
    assert extract_must_preserve_tokens(iso, dates=False) == {"2023"}
    # and then a changed date is invisible again, which is the old behaviour
    b = "The user acquired a silver necklace on 2023-11-02."
    assert find_dropped_entities([iso], b, dates=False) == set()
    assert find_dropped_entities([iso], b) == {"date:2023-04-15"}


# --- the opt-out compression relies on ------------------------------------

def test_quantities_false_restores_the_pre_quantity_token_set():
    """Compression passes this. Its contract is that every anchor lands in the
    summary or the loss manifest, so an anchor it can never restate would wedge
    a group on every future pass, not merely refuse one merge."""
    text = "The user has 1250 followers on Instagram and needs 300 points"
    with_q = extract_must_preserve_tokens(text)
    without_q = extract_must_preserve_tokens(text, quantities=False)
    assert "1250" in with_q and "300" in with_q
    assert "1250" not in without_q and "300" not in without_q
    assert "Instagram" in without_q, "the name families are untouched"


def test_quantities_false_also_reaches_the_diff_helper():
    contributor = "The user has 1250 followers on Instagram"
    merged = "The user has 1300 followers on Instagram"
    assert find_dropped_entities([contributor], merged)
    assert find_dropped_entities([contributor], merged, quantities=False) == set()


def test_a_record_id_is_still_caught_with_quantities_off():
    """Record identifiers are their own family and must not depend on the
    quantity switch."""
    dropped = find_dropped_entities(
        ["Order #844908 was refunded"], "An order was refunded",
        quantities=False)
    assert "#844908" in dropped


# ---------------------------------------------------------------------------
# 5. The event window on a merge.
#
# resolve_merge_metadata used to answer NULL whenever the windowed contributors
# were not byte-identical. A merge asserts the contributors describe ONE fact, so
# different windows mean one of three things, and NULL was wrong for all three:
# overlapping windows are the same event at two precisions (a date destroyed for
# nothing); non-overlapping windows mean the merge's premise is false (so do not
# merge); and nobody having a window is the only case NULL is honest for.
#
# NULL is also not neutral downstream: the row falls back to valid_from / DCT and
# serves as though it happened on the day it was mentioned.
# ---------------------------------------------------------------------------

from datetime import datetime as _dt

from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    resolve_merge_metadata,
)


def _contrib(start=None, end=None, precision=None, **over):
    base = {"memory_type": "facts", "confidence": 0.9,
            "temporal_category": "perpetual", "event_start": start,
            "event_end": end, "event_precision": precision}
    base.update(over)
    return base


def test_overlapping_windows_keep_the_most_precise_one():
    """"March 2023" and "Mar 27-31" are the same event said twice. The old rule
    saw two distinct tuples and answered NULL, throwing away a date it had."""
    resolved = resolve_merge_metadata([
        _contrib(_dt(2023, 3, 1), _dt(2023, 3, 31), "month"),
        _contrib(_dt(2023, 3, 27), _dt(2023, 3, 31), "range"),
    ])
    assert resolved["merge_refused_reason"] is None
    assert resolved["event_start"] == _dt(2023, 3, 27)
    assert resolved["event_precision"] == "range"


def test_identical_windows_still_survive():
    w = _dt(2023, 5, 4)
    resolved = resolve_merge_metadata([_contrib(w, None, "day"),
                                       _contrib(w, None, "day")])
    assert resolved["event_start"] == w
    assert resolved["merge_refused_reason"] is None


def test_non_overlapping_windows_REFUSE_the_merge_rather_than_null_it():
    """Two windows that cannot both be one event mean these are not one fact.
    Refusing keeps both rows and both dates; the old answer merged and left the
    merged row with no date at all."""
    resolved = resolve_merge_metadata([
        _contrib(_dt(2023, 2, 1), _dt(2023, 2, 3), "range"),
        _contrib(_dt(2023, 9, 1), _dt(2023, 9, 3), "range"),
    ])
    assert resolved["merge_refused_reason"], "the merge must be refused"
    assert "one event" in resolved["merge_refused_reason"]


def test_a_known_window_is_never_replaced_by_null():
    """The invariant. Whatever happens, a window some contributor had does not
    become NULL on the merged row: either it survives, or there is no merge."""
    for contributors in (
        [_contrib(_dt(2023, 3, 1), None, "day"), _contrib(_dt(2023, 3, 1), None, "day")],
        [_contrib(_dt(2023, 3, 1), _dt(2023, 3, 31), "month"), _contrib(_dt(2023, 3, 5), None, "day")],
        [_contrib(_dt(2023, 2, 1), None, "day"), _contrib(_dt(2023, 9, 1), None, "day")],
        [_contrib(_dt(2023, 2, 1), None, "day"), _contrib()],
    ):
        resolved = resolve_merge_metadata(contributors)
        had_one = any(c["event_start"] for c in contributors)
        if had_one and not resolved["merge_refused_reason"]:
            assert resolved["event_start"] is not None, (
                "a merge that proceeds must carry a window some contributor had")


def test_no_contributor_has_a_window_is_the_one_honest_null():
    resolved = resolve_merge_metadata([_contrib(), _contrib()])
    assert resolved["event_start"] is None
    assert resolved["merge_refused_reason"] is None, (
        "nothing to disagree about, so nothing to refuse")


def test_one_windowed_contributor_among_several_still_gives_its_window():
    resolved = resolve_merge_metadata([_contrib(_dt(2023, 7, 4), None, "day"),
                                       _contrib(), _contrib()])
    assert resolved["event_start"] == _dt(2023, 7, 4)
    assert resolved["merge_refused_reason"] is None


def test_the_caller_drops_a_refused_merge_rather_than_executing_it():
    """Read from source. execute_merges consumes whatever _attach_metadata_to_merges
    returns, so a refused merge must LEAVE the list, not merely carry a flag."""
    import inspect
    from synap.cloud.agents.ingestion.pipeline import stage4_organization
    src = inspect.getsource(stage4_organization.Stage4Organizer._attach_metadata_to_merges)
    assert "merge.merge_refused_reason" in src
    assert "if not m.merge_refused_reason" in src, (
        "the refused merges must be filtered out of the returned list")
