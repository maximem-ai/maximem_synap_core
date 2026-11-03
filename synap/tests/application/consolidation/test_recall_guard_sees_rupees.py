"""The merge recall guard could only see dollars.

`find_dropped_entities` is what stops a merge silently losing a fact. Its
numeric anchors matched `\\$` alone, so an amount in ANY other currency was
never extracted and therefore never protected.

Found by sweeping every deep-clean merge: exactly one loss — "₹5.3 Cr", one of
three overlapping corpus estimates in a five-memory financial merge. The guard reported nothing,
because `extract_must_preserve_tokens("₹4.3 Cr to ₹5.3 Cr")` returned only
{"Cr"}. The dollar control in the same sweep caught $120, which isolated the
gap to the symbol set rather than the mechanism.

THE FIX IS THE PATTERN, NOT THE INSTANCE. Hand-listing "$₹€£¥" would have
fixed the rupee and left won, ruble, lira, baht and shekel exactly as broken,
and would need editing again the first time a client invoices in one of them.
The symbol set is derived from Unicode's own currency category (Sc), so these
tests assert the PROPERTY — every currency Unicode knows about is protected —
rather than re-listing the ones someone happened to hit.
"""
import re
import unicodedata

import pytest

from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    _build_currency_class,
    extract_must_preserve_tokens,
    find_dropped_entities,
)

SRC = "The user is considering a corpus of ₹4.3 Cr to ₹5.3 Cr by age 35."
MERGED = ("The user plans to have a financial corpus of approximately "
          "₹4.3–₹5.5 Cr by age 35.")

# Derived the same way the code derives it, so this test cannot silently
# shrink to whatever the implementation happens to support.
ALL_CURRENCY = [chr(c) for c in range(0x20, 0x2500)
                if unicodedata.category(chr(c)) == "Sc"]


def test_the_symbol_set_is_derived_and_broad():
    """A hand-written list would be a handful of symbols. Unicode knows many,
    and the guard must cover them without anyone editing a list."""
    assert len(ALL_CURRENCY) > 40, (
        f"only {len(ALL_CURRENCY)} currency symbols found; the derivation is "
        f"probably wrong, which would make every test below vacuous")
    cls = _build_currency_class()
    for sym in ("$", "₹", "€", "£", "¥", "₩", "₽", "₺", "฿", "₪"):
        assert re.search(cls, sym), f"{sym} missing from the derived class"


@pytest.mark.parametrize("symbol", ALL_CURRENCY)
def test_every_unicode_currency_amount_is_an_anchor(symbol):
    """The property, asserted across all of them rather than a chosen few."""
    tokens = extract_must_preserve_tokens(f"the fee was {symbol}250 per month")
    assert any("250" in t for t in tokens), (
        f"{symbol}250 is not protected: {sorted(tokens)}")


def test_the_real_prod_loss_is_caught():
    assert "₹5.3" in find_dropped_entities([SRC], MERGED)


def test_a_merge_that_kept_everything_is_not_flagged():
    """A false positive is safe for the DATA (it refuses the merge) but
    expensive for the product: enough of them and nothing merges at all,
    which is the entire benefit of deep clean."""
    assert find_dropped_entities([SRC], SRC) == set()


@pytest.mark.parametrize("text,expected", [
    ("cost ₹500 total", "₹500"),            # no space
    ("cost ₹ 500 total", "₹ 500"),          # space — a mutation slipped past
    ("cost ₹1,00,000 total", "₹1,00,000"),  # Indian digit grouping
    ("cost $1,200.50 total", "$1,200.50"),  # thousands + decimals
    ("cost ₹4.3 Cr", "₹4.3"),               # decimal with a magnitude word
])
def test_amount_shapes_that_actually_occur(text, expected):
    """Written after a mutation that removed the optional space passed all
    seven of the first tests: they only covered shapes I had personally seen."""
    assert expected in extract_must_preserve_tokens(text), (
        sorted(extract_must_preserve_tokens(text)))


def test_dollars_still_work():
    """The control that isolated the bug. If this fails, the fix broke the one
    case that already worked, and a green suite would be hiding it."""
    assert find_dropped_entities(["total was $120"], "the order was placed") == {"$120"}


def test_the_guard_can_still_return_nothing():
    """Prove absence is reportable, so a clean result means something."""
    assert find_dropped_entities(["nothing numeric here"], "still nothing") == set()
