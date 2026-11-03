"""Put back the detail the five type calls dropped, without storing it twice.

WHY THIS EXISTS, with the measurement.

A detail a person says in passing, inside a message about something else, often
reaches no memory at all. Measured on the 500-question benchmark: 42 questions
are right ONLY because the raw conversation chunk we serve beside the memories
carries the answer, and 21 of those 42 need a price, a duration, a count, a
percentage, a clock time or a proper name that exists in chunk text and in no
memory line. Until those land as memories, the raw chunks cannot be switched
off, and the chunks cost 10.4 characters of context for every 1 character of
memory.

The five type calls cannot fix this by being told again. The same instruction
has been in facts.py since 2026-10-05 for the boots case and the obligation
still vanished; 341ef76f7 added a rule of this shape and reverted it after
measuring no change; and the same scope ingested twice on the same build stored
"was told it would cost around $60" once and "finds the taxi fare expensive" the
other time. Each of those five calls reads a ~9,000-character chunk through a
10,000 to 25,000 character system prompt and returns what it judges the main
content. An aside is nobody's main content, so it falls through all five.

So this does not instruct harder. It checks, with one short call whose only job
is to say what the text says, and then puts what is missing into the row that
already covers it.

MEASURED, 2026-10-15, through the real extractor on 9 scopes x 3 runs:

    arm                       rows   rows/chunk on CONTROL scopes   needles
    base                       389              5.60                12/27
    sweep+attach, unfiltered   443              5.77                23/27
    sweep+attach, filtered     403              5.13                23/27

    the $60 taxi fare        base 0/3 -> 3/3    via EXTEND
    the Mar 26 miss's cause  base 0/3 -> 3/3    via EXTEND
    left home at 7 AM        base 0/3 -> 3/3    via ADD
    album was DOWNLOADED     base 0/3 -> 2/3    via EXTEND
    a contrastive limit      base 0/3 -> 0/3    never emitted by the sweep

Duplicate pairs 50 against base's 52: this rewrites 101 rows rather than adding
101, which is the whole point. 3 of the 4 recovered details arrived by EXTEND
and the 7 AM arrived by ADD, so neither path can be removed.

⚠ WHAT IT DOES NOT DO. The sweep does not catch a contrastive limit: "I enjoy X
but I want to branch out" came back as the enjoyment alone, 0 of 3, which is the
exact failure this pass exists to prevent, on that one class. Do not claim that
class until it is measured again.
"""

import json
import logging
import os
import re
from typing import List, Optional, Sequence, Tuple

from synap.cloud.agents.ingestion.prompts.sweep_repair import (
    SWEEP_SYSTEM,
    ATTACH_SYSTEM,
    build_attach_user_prompt,
    build_sweep_user_prompt,
)

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_INGEST_SWEEP_REPAIR"
_BY_INSTANCE_ENV = "SYNAP_INGEST_SWEEP_REPAIR_BY_INSTANCE"


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether the repair pass runs for this instance.

    ⚠ ``instance_id`` has no default, for the reason temporal_flags.enabled
    records: a caller that omits it gets the fleet answer for a per-instance
    flag, which is how SYNAP_EVENT_ANCHOR ended up off at the only place that
    mattered. Passing None explicitly still means "the fleet".

    Deliberately NOT routed through temporal_flags: that helper falls back to
    SYNAP_TEMPORAL_V2, and this feature has nothing to do with dates.
    """
    by_instance = (os.environ.get(_BY_INSTANCE_ENV) or "").strip()
    if by_instance and instance_id is not None:
        want = str(instance_id).strip().lower()
        for pair in by_instance.split(","):
            key, _, value = pair.partition("=")
            if key.strip().lower() == want:
                return value.strip().lower() in ("1", "true", "yes", "on")
    return (os.environ.get(_FLAG_ENV) or "").strip().lower() in (
        "1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────────────
# Guard A: the quote must be in the text. No model.
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


MIN_QUOTE_CHARS = 8
MIN_QUOTE_WORDS = 3


def _anchors_enough(quote: str) -> bool:
    """Whether a quote is specific enough that matching it is not an accident.

    Three whitespace words, OR eight characters, OR any character outside Basic
    Latin. The last clause is not decoration: "東京" is two characters and one
    "word" and is a perfectly specific quote, while the attack this floor exists
    to stop was the single letter "a". Word counting does not work for scripts
    without spaces, so high-entropy characters earn their own clause.
    """
    if len(quote.split()) >= MIN_QUOTE_WORDS:
        return True
    if len(quote) >= MIN_QUOTE_CHARS:
        return True
    return any(ord(ch) > 0x24F for ch in quote)


def surviving_statements(raw_statements, content: str) -> List[dict]:
    """Keep only statements whose quote is really in *content*, and long enough.

    The model has to hand back the words it read the statement from, and we
    check them. 98% of quotes came back verbatim in the measured run, so the
    cost of the check is a couple of statements, not the pass.

    ⚠ The floor rests on WORD COUNT, not length: "I paid $60" is a perfectly
    good anchor at 10 characters and an earlier 12-character floor rejected it.
    Three words is what stops an accidental match; the character floor is only
    a belt against three one-letter words.
    A one-character quote is a substring of
    almost any chunk, so without it "a" anchors any content at all: an
    adversarial pass got "bought a yacht for $2000" past this check against a
    chunk about tea. A quote has to be substantial enough to actually anchor
    something.

    Whitespace is normalised on both sides because the chunk arrives with its own
    line breaks and a model reflowing a quote is not a fabrication.
    """
    hay = _normalise_space(content).lower()
    out: List[dict] = []
    for s in raw_statements or []:
        if not isinstance(s, dict):
            continue
        quote = _normalise_space(str(s.get("quote") or ""))
        body = _normalise_space(str(s.get("content") or ""))
        if not quote or not body:
            continue
        if not _anchors_enough(quote):
            logger.debug("sweep_quote_too_short quote=%r", quote[:40])
            continue
        # ⚠ The content is checked against its OWN quote, not just the quote
        # against the chunk. Quote "I paid $60 for the taxi" is real while
        # content "paid $600 for a limousine" is not, and the quote check alone
        # waves that through. Every number the content states must appear in the
        # words it claims to come from.
        if not _keeps_every_number(body, quote):
            logger.debug("sweep_content_invents_a_number content=%r", body[:60])
            continue
        # A statement may not negate where the words it came from do not.
        # Quote "My cat Miso is 3" with content "does not own 3 cats" passes
        # every other check: the number is there and the quote is real. This is
        # free to refuse: 0 of the 277 statements in the measured run negate
        # where their quote does not.
        if _NEGATION.search(body) and not _NEGATION.search(quote):
            logger.debug("sweep_content_negates_its_quote content=%r", body[:60])
            continue
        if quote.lower() not in hay:
            logger.debug("sweep_quote_not_in_text quote=%r", quote[:80])
            continue
        out.append({"quote": quote, "content": body,
                    "said_by": str(s.get("said_by") or "").strip().lower()})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Guard B: an ADD must carry something a question can be asked about. No model.
# ──────────────────────────────────────────────────────────────────────────────
#
# Without this, 19% of adds qualified and the rest were intentions and
# pleasantries: "thinking of setting up a small studio", "would like to explore
# more about the Climate and Sea-Level Rise Initiative", "has heard that it's
# important to sign and date the painting". The filter keeps 8 of 48 adds and
# loses NO recovered detail, which is the only reason it is safe to be this
# strict.
#
# EXTEND is deliberately not filtered: it adds no row, and the detail it carries
# has already been judged to belong to a row we decided to keep.

_VALUE = re.compile(
    r"\$\s?\d"                                  # a price
    r"|\b\d{1,2}:\d{2}\b"                       # a clock time
    r"|\b\d{1,2}\s*[ap]\.?m\.?\b"               # 7 AM
    r"|\b(?:19|20)\d{2}\b"                      # a year
    r"|\b\d[\d,.]*\s*(?:dozen|pages?|miles?|minutes?|mins?|hours?|hrs?|days?"
    r"|weeks?|months?|years?|percent|%|times|gallons?|pounds?|lbs?|kgs?|kms?)\b",
    re.I,
)
_ACQUIRED = re.compile(
    r"\b(bought|buy|purchas\w*|downloaded|replac\w*|fixed|repair\w*|return\w*"
    r"|donat\w*|lent|sold|owns?|needs? to|has to|must)\b",
    re.I,
)


def worth_adding(text: str) -> bool:
    """Whether a statement no row covers may become a row of its own."""
    return bool(_VALUE.search(text or "") or _ACQUIRED.search(text or ""))


# ──────────────────────────────────────────────────────────────────────────────
# Applying the decisions
# ──────────────────────────────────────────────────────────────────────────────

MAX_ADDS_PER_CHUNK = 6
MAX_TEXT_CHARS = 600
_MIN_STATEMENT_OVERLAP = 0.5

_NEGATION = re.compile(
    r"\b(not|never|cannot|nor|no longer|none|neither)\b|n't\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9']+")
_SMALL = frozenset("""the and for with that this from they them their have has had was
were are is been your you about into over after before a an of to in on at it its as by
if can could would should will shall may might must not no too very just also more most
some any each other such same what which who when where why how all both one two""".split())


def _content_words(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower())
            if len(w) > 3 and w not in _SMALL}


def _numbers(text: str) -> List[str]:
    return [m.group(0) for m in _NUMBER.finditer(text or "")]


def _keeps_every_number(original: str, candidate: str) -> bool:
    """Every number in *original* must survive in *candidate* as a WHOLE token.

    ⚠ This exists because find_dropped_entities does not catch it: "$60"
    rewritten to "$600" passes that guard, since "60" is a substring of "600".
    A token-boundary check refuses it. Measured: the adversarial pass got
    "paid $60" rewritten to "paid $600" through the old guard.
    """
    for num in _numbers(original):
        esc = re.escape(num)
        if not re.search(rf"(?<!\d){esc}(?!\d)", candidate or ""):
            return False
    return True


def _says_what_the_statement_says(statement: str, text: str) -> bool:
    """The written text must actually be about the statement that justified it.

    ⚠ Nothing used to tie them together: `statement_index` was read only to
    look up `said_by`, so a decision could cite a statement about a taxi and
    write a row about buying a house. Both halves are checked: every number the
    statement carries has to appear, and at least half its content words.
    """
    if not _keeps_every_number(statement, text):
        return False
    want = _content_words(statement)
    if not want:
        return True
    have = _content_words(text)
    return len(want & have) / len(want) >= _MIN_STATEMENT_OVERLAP


def extend_is_safe(row: str, text: str) -> Tuple[bool, str]:
    """Whether rewriting *row* as *text* can only have ADDED to it.

    find_dropped_entities is necessary and NOT sufficient: it catches a dropped
    name, quantity or date, and it cannot see any of these, all of which an
    adversarial pass got through:
      - "The user owns 3 cats" -> "The user does not own 3 cats"   (negation)
      - a long row paraphrased down to "user likes walks"          (shrinkage)
      - a row replaced wholesale by unrelated text
      - a row reduced to "."                                       (punctuation)
      - "paid $60" -> "paid $600"                                  (number edit)
    So an extend must be at least as long as the row, must keep every number as
    a whole token, and must not introduce a negation the row did not have.
    """
    row = row or ""
    text = text or ""
    if len(text) < len(row):
        return False, "shorter_than_the_row"
    if not _keeps_every_number(row, text):
        return False, "changed_a_number"
    if _NEGATION.search(text) and not _NEGATION.search(row):
        return False, "introduced_a_negation"
    kept = _content_words(row)
    if kept and len(kept & _content_words(text)) / len(kept) < _MIN_STATEMENT_OVERLAP:
        return False, "lost_most_of_the_row"
    return True, ""


def _valid_index(value, upper: int) -> bool:
    """A real int index. ⚠ bool is an int in Python, and `row_index: true`
    rewrote row 1 before this check existed."""
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < upper)


def plan_from_decisions(decisions, statements: Sequence[dict],
                        row_contents: Sequence[str]
                        ) -> Tuple[List[Tuple[int, str, str]], List[dict]]:
    """Turn the model's decisions into (extends in order, adds).

    Everything malformed, out of range, unjustified by its statement, or failing
    a guard is dropped here, so the caller only ever sees changes it is allowed
    to make.

    Extends come back as an ORDERED LIST of (row index, text, the statement that
    justified it), because the caller still has to run the two provenance checks
    that need the statement. Keeping only the first candidate per row here meant
    a refused first extend silently blocked a valid second one.
    """
    row_count = len(row_contents)
    extends: List[Tuple[int, str, str]] = []
    adds: List[dict] = []
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        action = str(d.get("action") or "").strip().lower()
        text = _normalise_space(str(d.get("text") or ""))
        if not text or len(text) > MAX_TEXT_CHARS:
            continue
        # Both actions write something, so both must name the statement that
        # justifies it, and the text must be about that statement.
        si = d.get("statement_index")
        if not _valid_index(si, len(statements)):
            continue
        statement = str(statements[si].get("content") or "")
        if not _says_what_the_statement_says(statement, text):
            continue
        said_by = statements[si].get("said_by") or ""
        if action == "extend":
            ri = d.get("row_index")
            if not _valid_index(ri, row_count):
                continue
            ok, why = extend_is_safe(row_contents[ri], text)
            if not ok:
                logger.debug("sweep_extend_refused row=%d why=%s", ri, why)
                continue
            extends.append((ri, text, statement))
        elif action == "new":
            if worth_adding(text) and len(adds) < MAX_ADDS_PER_CHUNK:
                adds.append({"content": text, "said_by": said_by,
                             "statement": statement})
    return extends, adds


def loads_json(raw) -> dict:
    """Tolerant load, matching what the extraction path already accepts."""
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001 — a bad reply loses the pass, never the ingest
        return {}
    return parsed if isinstance(parsed, dict) else {}
