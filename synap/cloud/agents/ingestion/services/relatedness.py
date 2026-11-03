"""Which stored memories are worth comparing a new one against.

Knowledge update answers "what should happen between these two memories". It is
good at that. What it was bad at is being handed the right two memories, because
the choosing was done by cosine distance and four stacked floors:

    threshold          0.55, env-tunable
    retraction_floor   lower, for "forget my address"
    value_floor        lower again, when both sides name the same attribute class
    plus a top_k of 5

Each floor exists because the one above it was missing a case. The Rachel scope is
the proof that the shape is wrong rather than the numbers: every live "moved to
Chicago" row scored 0.595 to 0.762 against the gold "moved back to the suburbs"
row, and all four were ranks 1 to 4 of 20. The budget already held them. The floor
was the only thing hiding them, and the cost of that one miss was two contradictory
rows served live.

So this asks a judge instead of a distance. Vector search goes wide and stops
deciding anything; Jev reads the new memory against each candidate and says whether
the pair is worth comparing; knowledge update still makes every decision it makes
today. Nothing here can supersede, merge, date or drop a memory. It can only
decide what knowledge update gets to look at.

⚠ One number survives, and it is not the same KIND of number. ``_ADMIT_P`` is "more
likely than not that this pair is worth comparing", read off a stated rubric. A
cosine floor asks how similar two strings are, which is not the question. Honest
accounting: this replaces four thresholds with one probability, it does not remove
thresholds.

⚠ Jev's scores are NOT stable. A prior measurement found 81% of scores moving
between identical calls. That is survivable here and would not be if this decided
anything: a noisy screen feeding a deterministic classifier costs an occasional
borderline candidate, while a noisy decider would write a different memory graph on
every run. It is also why this never has the last word.

Fails to today's behaviour, not open. An unavailable Jev, a timeout or a malformed
reply returns None, and the caller runs the floors it runs now. Failing open would
hand the classifier 150 unscreened candidates per memory, which is a cost incident.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set

from synap.cloud.agents.ingestion.prompts.relatedness import (
    RUBRIC as _RUBRIC,
    PURPOSE as _PURPOSE,
    SUBJECT_CRITERIA as _SUBJECT_CRITERIA,
    SUBJECT_PURPOSE as _SUBJECT_PURPOSE,
    ANCHOR_RUBRIC as _ANCHOR_RUBRIC,
    ANCHOR_PURPOSE as _ANCHOR_PURPOSE,
    CONVERSATION_CONTEXT_NOTE,
    SUBJECT_GROUP_INSTRUCTIONS,
    SUBJECT_ONE_INSTRUCTIONS,
    rate_instruction,
)

logger = logging.getLogger(__name__)


class JevShapeError(ValueError):
    """A reply arrived but is not the shape the API documents.

    Its own type so it is caught by the same handler as a timeout and reported
    as "could not run", rather than being mistaken for a verdict.
    """

_FLAG_ENV = "SYNAP_KU_JEV_SCREEN"
_BY_INSTANCE_ENV = "SYNAP_KU_JEV_SCREEN_BY_INSTANCE"
_POOL_ENV = "SYNAP_KU_CANDIDATE_POOL"
_TRUTHY = {"1", "true", "on", "yes"}

# Subject identity (migration 159). Separate variables from the screen's own,
# so the screen can run without grouping anything, which is what every client
# but the benchmark instance does until this is measured.
_SUBJECT_FLAG_ENV = "SYNAP_SUBJECT_LINK"
_SUBJECT_BY_INSTANCE_ENV = "SYNAP_SUBJECT_LINK_BY_INSTANCE"

# Jev's own re-ranker truncates at 150 per call rather than batching, so that is
# the working ceiling for one state. Wide enough that the budget stops being the
# thing that hides a pair.
DEFAULT_POOL = 150
MAX_POOL = 150

# P(worth comparing) at or above which knowledge update gets to see the pair.
_ADMIT_P = 0.5

_ADMIT_FROM = 2          # levels 2 and 3 are "worth comparing"


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether Jev screens candidates for this instance.

    Its own pair of variables, deliberately NOT routed through temporal_flags:
    this is not part of the temporal set and must not switch on with it.
    """
    raw_inst = os.environ.get(_BY_INSTANCE_ENV, "")
    if instance_id is not None and raw_inst.strip():
        for part in raw_inst.split(","):
            key, _, value = part.strip().partition("=")
            if key.strip().lower() == str(instance_id).strip().lower():
                return value.strip().lower() in _TRUTHY
    return os.environ.get(_FLAG_ENV, "").strip().lower() in _TRUTHY


def pool_size() -> int:
    """How many candidates the vector search should return when screening is on."""
    try:
        n = int(os.environ.get(_POOL_ENV) or DEFAULT_POOL)
    except (TypeError, ValueError):
        return DEFAULT_POOL
    return max(1, min(n, MAX_POOL))


# How much of the source conversation to show for the NEW memory. Free to include
# (Stage 4 already holds it for entity resolution) and it is the side that needs
# context most, since it is being judged against everything else.
NEW_CHUNK_CHARS = 2000


def build_state(new_content: str, candidates: Sequence[Any], *,
                new_chunk: Optional[str] = None) -> str:
    """The state Jev reads.

    ⚠ Why the candidates get a DATE and not their source conversation. The bug a
    chunk would disambiguate is the one recorded in knowledge_update's own comment,
    "two distinct concerts get superseded into one", and what separates two concerts
    is when they happened, which is already in each candidate's metadata. Sending
    every candidate's chunk instead would be roughly 225k characters for a pool of
    150, against about 15k for their texts, so it would force the pool back down to
    a handful and give away the recall this whole change exists to buy. The date is
    the part of the chunk that does the work.

    A candidate may be a plain string or a mapping with ``content`` and optionally
    ``document_created_at``, ``event_start`` and ``memory_type``.
    """
    lines = []
    for i, c in enumerate(candidates):
        if isinstance(c, dict):
            text = str(c.get("content") or "")
            bits = []
            for key, label in (("event_start", "happened"),
                               ("document_created_at", "said on")):
                v = c.get(key)
                if v:
                    bits.append(f"{label} {str(v)[:10]}")
            if c.get("memory_type"):
                bits.append(str(c["memory_type"]))
            tag = f"  ({', '.join(bits)})" if bits else ""
        else:
            text, tag = str(c), ""
        lines.append(f"[{i}] {text}{tag}")

    head = f"{_PURPOSE}\n\nTHE NEW MEMORY:\n{new_content}\n"
    if new_chunk and new_chunk.strip():
        head += (CONVERSATION_CONTEXT_NOTE
                 + f"{new_chunk.strip()[:NEW_CHUNK_CHARS]}\n")
    return head + "\nMEMORIES ALREADY STORED ABOUT THIS PERSON:\n" + "\n".join(lines)


def build_questions(count: int, rubric: Optional[List[str]] = None
                    ) -> Dict[str, Dict[str, Any]]:
    return {f"s{i}": {"type": "score",
                      "instructions": rate_instruction(i),
                      # criteria must be an ORDERED LIST for a score question;
                      # the dict shape choice questions use is rejected 422.
                      "criteria": rubric or _RUBRIC}
            for i in range(count)}


def admitted_from_answers(answers: Dict[str, Any], count: int, *,
                          admit_from: Optional[int] = None,
                          levels: Optional[int] = None) -> Set[int]:
    """Indices whose P(level >= admit_from) reaches _ADMIT_P.

    A missing or unparsable answer is NOT admitted: the whole point is to stop
    handing the classifier pairs nothing vouched for, so silence means no.
    """
    out: Set[int] = set()
    for i in range(count):
        answer = answers.get(f"s{i}")
        if not isinstance(answer, dict):
            continue                     # a string, a list, None: all mean no
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            continue
        total = 0.0
        first = _ADMIT_FROM if admit_from is None else admit_from
        last = (levels if levels is not None
                else max(len(_RUBRIC), (admit_from or 0) + 1))
        for level in range(first, last):
            try:
                total += float(probs.get(str(level), 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        if total >= _ADMIT_P:
            out.add(i)
    return out


def _probability(value: Any) -> Optional[float]:
    """A reply's number as a probability, or None if it is not one.

    NaN is the dangerous value and the reason this exists. Every comparison
    against NaN is False, so a NaN score slipped past BOTH gates in
    any gate written as a comparison. The floor and margin that once stood here
    are gone and the subject question is a boolean now, but a NaN would still
    make "true" and "false" incomparable, so it is rejected at the door.

    Returning None rather than 0.0 keeps "we could not read it" distinct from
    "the model said no", which is the whole point of omitting unparsable
    answers in the first place.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):   # NaN or an infinity
        return None
    return min(max(f, 0.0), 1.0)


def same_subject_scores(answers: Dict[str, Any], count: int) -> Dict[int, float]:
    """P(this candidate is the same thing in the world) per candidate index.

    The same sum ``admitted_from_answers`` computes and then throws away. It is
    kept here because levels 2 and 3 are the two readings of "one thing" and the
    boolean that replaced them is the reason nothing downstream can tell eight
    sentences about one coffee table from eight different pieces of furniture.

    A missing or unparsable answer is absent from the result rather than scored
    zero: "we could not read it" and "the model said no" are different, and only
    the second is evidence.
    """
    out: Dict[int, float] = {}
    for i in range(count):
        answer = answers.get(f"s{i}")
        if not isinstance(answer, dict):
            continue
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            continue
        total = 0.0
        readable = True
        for level in range(_ADMIT_FROM, len(_RUBRIC)):
            raw = probs.get(str(level))
            if raw is None:
                # An absent level is mass the model put elsewhere, not an
                # unreadable answer. All of it on level 0 means "a different
                # thing", which scores 0.0 and must stay distinguishable from
                # a reply we could not read at all.
                continue
            p = _probability(raw)
            if p is None:
                # A present but unreadable VALUE is different: a partial total
                # would understate the pair, and a NaN would bypass every gate
                # downstream, so the whole answer is treated as unread.
                readable = False
                break
            total += p
        if readable:
            out[i] = min(total, 1.0)
    return out


def subject_link_enabled(instance_id: Optional[object]) -> bool:
    """Whether a screened pair may be recorded as the same subject.

    Rides on the screen: with no screen there are no per-level scores and
    nothing to decide from. Its own pair of variables so it can be turned on for
    one instance without turning it on everywhere.
    """
    raw_inst = os.environ.get(_SUBJECT_BY_INSTANCE_ENV, "")
    if instance_id is not None and raw_inst.strip():
        for part in raw_inst.split(","):
            key, _, value = part.strip().partition("=")
            if key.strip().lower() == str(instance_id).strip().lower():
                return value.strip().lower() in _TRUTHY
    return os.environ.get(_SUBJECT_FLAG_ENV, "").strip().lower() in _TRUTHY


async def screen(new_content: str, candidates: Sequence[Any], *,
                 new_chunk: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 ask_fn=None) -> Optional[Set[int]]:
    """Indices of candidates worth comparing, or None if the screen could not run.

    None means "fall back to the floors", never "admit everything".
    """
    admitted, _ = await screen_scored(
        new_content, candidates, new_chunk=new_chunk,
        correlation_id=correlation_id, ask_fn=ask_fn)
    return admitted


async def screen_scored(new_content: str, candidates: Sequence[Any], *,
                        new_chunk: Optional[str] = None,
                        correlation_id: Optional[str] = None,
                        ask_fn=None):
    """``(admitted indices, same-subject score per index)``.

    One Jev call answering both questions, because they are the same question
    read at two thresholds: "worth comparing at all" is P(level >= 2) >= 0.5,
    and "the same thing in the world" is the argmax of that same number. Asking
    twice would double the cost and could disagree with itself.

    On failure: ``(None, {})`` — None still means "fall back to the floors",
    never "admit everything", and an empty score map means nothing is grouped.
    """
    if not new_content or not candidates:
        return set(), {}
    if ask_fn is None:
        from synap.cloud.shared.jev_client import ask as _ask, jev_configured
        if not jev_configured():
            logger.info("ku_jev_screen_unavailable reason=no_api_key correlation_id=%s",
                        correlation_id or "")
            return None, {}
        ask_fn = _ask
    head = list(candidates)[:MAX_POOL]
    try:
        state = build_state(new_content, head, new_chunk=new_chunk)
        answers = await ask_fn(state, build_questions(len(head)))
    except Exception as exc:  # noqa: BLE001 — degrade to the floors, never to "all"
        logger.warning("ku_jev_screen_failed %s: %s correlation_id=%s",
                       type(exc).__name__, str(exc)[:160], correlation_id or "")
        return None, {}
    answers = answers or {}
    admitted = admitted_from_answers(answers, len(head))
    scores = same_subject_scores(answers, len(head))
    logger.info("ku_jev_screen candidates=%d admitted=%d scored=%d state_chars=%d "
                "chunk=%s correlation_id=%s", len(head), len(admitted), len(scores),
                len(state), bool(new_chunk), correlation_id or "")
    return admitted, scores

# ── the same judge, asked whether two things are ONE thing ───────────────────
#
# A third question of the same shape, and the reason it is a question rather
# than a number is measured. The first attempt decided this from the screen's
# own score with a floor and a margin, and both numbers were wrong in ways no
# number could fix:
#
#   the margin  blocked a group BECAUSE the thing was mentioned often. Ten
#               statements about one coffee table all score 0.86-0.99, so the
#               gap between best and runner-up is 0.000 and every pick was
#               refused. Five of 25 scopes grouped nothing.
#   the floor   decided correctness from a threshold, which is the thing the
#               screen exists to stop doing. "A close second because it is a
#               rival" and "a close second because it is the same table" are
#               the same number and different answers.
#
# So this asks, and takes the model's OWN answer: a boolean question, decided
# by whichever way it leans. There is no tuned number anywhere in this path.
# The only number left is the screen's existing admit threshold, which now
# chooses which candidates are worth ASKING about, a cost bound rather than a
# claim about what counts as one thing.

# ⚠ WHAT CHANGED 2026-10-13, and why it was not a wording tweak.
#
# lme42 scored 15/25 against 17/25, and the diagnosis blamed one clause in the
# false side, "two occasions of the same kind". That diagnosis was WRONG, and an
# offline harness over real sentences proved it: deleting the clause moved
# the road-bike pair from 0.29 to 0.32, still a no, and five different wordings
# all scored 29/31 with the SAME two misses.
#
# The reason is that the old question asked what each SENTENCE IS ABOUT. "Cleaned
# the road bike's chain on March 2" is about a chain cleaning and "serviced the
# road bike at Pedal Power on March 10" is about a service, so "the same single
# thing" was honestly no. But the benchmark question counts BIKES.
#
# So the QUESTION changed, from "is this about the same single thing" to "do
# these two statements concern the same one thing of this person's". That
# separates the thing from the events that happen to it. Measured on 70
# hand-labelled assertions taken verbatim from the lme42 corpus, 31 from the
# scopes the failures came from, 28 held out from five scopes never used to tune
# it, and 11 in the group form:
#
#                             combined   tuned   held-out   group   margin
#   the old question            57/70     29/31    28/28     ?      +0.01
#   object framing alone        67/70     31/31    27/28    10/11   +0.10
#   this wording                70/70     31/31    28/28    11/11   +0.19
#
# Margin is the lowest TRUE minus the highest FALSE. The old question's margin
# was SMALLER THAN ITS OWN RUN-TO-RUN WOBBLE (~0.07), so its held-out 28/28 was
# not reproducible: a repeat run flipped a pair from 0.54 to 0.47. This wording
# held 70/70 across three trials with the largest single movement at 0.07.
#
# Three clauses in the false side each came from a measured failure, so do not
# trim one without re-running the harness (scratchpad/v4.py):
#   "a separate event connected to a thing"   merged Rachel's baby SHOWER with
#                                             Rachel's BABY at 0.60
#   "a taste for other things of the same kind"  two preferences that merely
#                                             NAME Avengers: Endgame sat at
#                                             0.50 and 0.49, coin flips
#   "something that happens over and over"    the GROUP form merged two
#                                             instances of one weekly fun run
#                                             at 0.58, because the group held a
#                                             sentence about the run SERIES.
#                                             The per-statement form never did.
#
# Still unsolved and deliberately untouched: "a fact about a thing and a fact
# about its category" is what keeps the 4 CATEGORY questions (cuisines, citrus
# types) wrong, and nothing measured has needed that clause. Test it on its own.


def build_subject_state(new_content: str, candidates: Sequence[Any],
                        groups: Sequence[Any], *,
                        new_chunk: Optional[str] = None) -> str:
    """The state for the subject question.

    Carries the new statement, the individual statements worth asking about, and
    every GROUP the system has already decided is one thing. Groups are in the
    same state as the singles so one call answers both "does this belong with
    that statement" and "does this belong with that group", which is what lets
    two rival groups for one thing be noticed and merged.
    """
    lines = [_SUBJECT_PURPOSE, "", "THE NEW STATEMENT:", new_content, ""]
    if new_chunk and new_chunk.strip():
        lines += ["THE CONVERSATION IT CAME FROM, for context only:",
                  new_chunk.strip()[:NEW_CHUNK_CHARS], ""]
    if candidates:
        lines.append("INDIVIDUAL STATEMENTS ALREADY STORED ABOUT THIS PERSON:")
        for i, c in enumerate(candidates):
            text = c.get("content") if isinstance(c, dict) else str(c)
            lines.append("  [c%d] %s" % (i, text))
        lines.append("")
    if groups:
        lines.append("GROUPS THE SYSTEM HAS ALREADY DECIDED ARE ONE THING EACH:")
        for i, g in enumerate(groups):
            members = g.get("members") if isinstance(g, dict) else list(g)
            lines.append("  [g%d] a group of %d statements:" % (i, len(members)))
            for m in members[:8]:
                lines.append("        - %s" % m)
            if len(members) > 8:
                lines.append("        - ... and %d more" % (len(members) - 8))
        lines.append("")
    return "\n".join(lines)


def build_subject_questions(n_candidates: int, n_groups: int) -> Dict[str, Dict[str, Any]]:
    """One boolean question per individual statement and per group."""
    qs: Dict[str, Dict[str, Any]] = {}
    for i in range(n_candidates):
        qs["c%d" % i] = {
            "type": "noul",
            "instructions": SUBJECT_ONE_INSTRUCTIONS % i,
            "criteria": _SUBJECT_CRITERIA,
        }
    for i in range(n_groups):
        qs["g%d" % i] = {
            "type": "noul",
            "instructions": SUBJECT_GROUP_INSTRUCTIONS % i,
            "criteria": _SUBJECT_CRITERIA,
        }
    return qs


def subject_verdicts(answers: Dict[str, Any], n_candidates: int, n_groups: int):
    """``(candidate indices answered yes, group indices answered yes)``.

    Decided by whichever way the model leans, never by a threshold anybody
    tuned: "true" wins when it carries more of the probability than "false".
    An answer we cannot read is a no, because silence is not evidence.
    """
    def _yes(key: str) -> bool:
        a = answers.get(key)
        if not isinstance(a, dict):
            return False
        # A boolean answer comes back as ONE number: {"type": "noul",
        # "noul": 0.93}, the probability that the answer is "true". There is no
        # true/false map and no `probabilities` key, which is what the first
        # version of this guessed; it read nothing, returned False for every
        # pair, and grouped nothing at all. Verified against the live API: 0.93
        # for two statements about one coffee table, 0.03 for a coffee table
        # against a bookshelf.
        p = _probability(a.get("noul"))
        if p is None:
            # Some replies carry a plain boolean or a string instead.
            v = a.get("answer", a.get("value"))
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            return str(v).strip().lower() == "true"
        # More likely than not. This is the model's own answer read the only way
        # a probability of "true" can be read, not a threshold anybody tuned:
        # there is no number here to raise or lower in search of a better score.
        return p > 0.5

    return ({i for i in range(n_candidates) if _yes("c%d" % i)},
            {i for i in range(n_groups) if _yes("g%d" % i)})


async def ask_subject(new_content: str, candidates: Sequence[Any],
                      groups: Sequence[Any], *,
                      new_chunk: Optional[str] = None,
                      correlation_id: Optional[str] = None,
                      ask_fn=None):
    """``(yes candidates, yes groups)``, or ``None`` if the judge could not run.

    None means "group nothing for this memory", never "group everything": an
    unavailable judge must leave the corpus exactly as it would have been
    before this feature existed.
    """
    if not new_content or (not candidates and not groups):
        return set(), set()
    if ask_fn is None:
        from synap.cloud.shared.jev_client import ask as _ask, jev_configured
        if not jev_configured():
            return None
        ask_fn = _ask
    cands = list(candidates)[:MAX_POOL]
    grps = list(groups)[:MAX_POOL]
    try:
        state = build_subject_state(new_content, cands, grps, new_chunk=new_chunk)
        answers = await ask_fn(state, build_subject_questions(len(cands), len(grps)))
        # Parsed INSIDE the try. It used to sit after it, so a reply that was
        # truthy but not a dict (a bare list, say) raised out of a function whose
        # docstring promises None. The coordinator's own catch hid it, which is
        # the worst version: the contract was wrong and nothing said so.
        #
        # A reply we cannot read is None, not an empty verdict. "The judge could
        # not run" and "the judge said no to everything" are different claims,
        # and only the second is evidence.
        if not isinstance(answers, dict):
            raise JevShapeError("reply was %s, not a mapping" % type(answers).__name__)
        yes_c, yes_g = subject_verdicts(answers, len(cands), len(grps))
    except Exception as exc:  # noqa: BLE001 — degrade to no grouping, never to all
        logger.warning("subject_question_failed %s: %s correlation_id=%s",
                       type(exc).__name__, str(exc)[:160], correlation_id or "")
        return None
    logger.info("subject_question candidates=%d groups=%d yes_candidates=%d "
                "yes_groups=%d state_chars=%d correlation_id=%s",
                len(cands), len(grps), len(yes_c), len(yes_g), len(state),
                correlation_id or "")
    return yes_c, yes_g


# ── the same judge, asked the referential question ───────────────────────────
#
# An open event_anchor names the event its date hangs off, in a few words the model
# chose: "best friend's wedding". Finding which stored memory IS that event was done
# by requiring every significant word of the phrase to appear in the candidate. That
# fails on one apostrophe: "The user went to a wedding for his best friend" is
# unmistakably the same event and misses, because "friend's" is not "friend". It
# also cannot see "The user's best friend got married", which shares no word with
# "wedding" at all.
#
# Same mechanism as above, different question. Not a new one: one module asks Jev to
# narrow candidates, and this is the second thing it narrows.


def build_anchor_state(anchor_text: str, candidates: Sequence[Any]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        if isinstance(c, dict):
            text, when = str(c.get("content") or ""), c.get("event_start")
        else:
            text, when = str(getattr(c, "content", c)), getattr(c, "event_start", None)
        tag = f"  (dated {str(when)[:10]})" if when else ""
        lines.append(f"[{i}] {text}{tag}")
    return (f"{_ANCHOR_PURPOSE}\n\nTHE EVENT BEING REFERRED TO:\n{anchor_text}\n\n"
            f"DATED MEMORIES ALREADY STORED ABOUT THIS PERSON:\n" + "\n".join(lines))


async def screen_anchor(anchor_text: str, candidates: Sequence[Any], *,
                        correlation_id: Optional[str] = None,
                        ask_fn=None) -> Optional[Set[int]]:
    """Indices of candidates that ARE the event named, or None if it could not run.

    Only the top rubric level is admitted, unlike the relatedness screen. A wrong
    match here WRITES A DATE onto a memory, where a wrong match there only costs a
    classifier call, so this end has to be the strict one. None means the caller
    falls back to word containment.
    """
    if not anchor_text or not candidates:
        return set()
    if ask_fn is None:
        from synap.cloud.shared.jev_client import ask as _ask, jev_configured
        if not jev_configured():
            return None
        ask_fn = _ask
    head = list(candidates)[:MAX_POOL]
    try:
        answers = await ask_fn(build_anchor_state(anchor_text, head),
                               build_questions(len(head), rubric=_ANCHOR_RUBRIC))
    except Exception as exc:  # noqa: BLE001 — degrade to containment, never to "all"
        logger.warning("anchor_jev_screen_failed %s: %s correlation_id=%s",
                       type(exc).__name__, str(exc)[:160], correlation_id or "")
        return None
    admitted = admitted_from_answers(answers or {}, len(head),
                                     admit_from=len(_ANCHOR_RUBRIC) - 1)
    logger.info("anchor_jev_screen anchor=%r candidates=%d admitted=%d correlation_id=%s",
                anchor_text[:60], len(head), len(admitted), correlation_id or "")
    return admitted
