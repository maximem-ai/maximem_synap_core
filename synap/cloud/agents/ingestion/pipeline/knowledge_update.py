"""Knowledge-update detection for ingestion pipeline (v2).

When new memories arrive, this stage compares each one against semantically
similar existing memories for the same user and classifies the relationship
into one of four outcomes:

- **SUPERSEDE** — the existing memory describes a state attribute whose value
  changed; deprecate the old, link to the new (handled by caller as a
  soft-delete + memory_actions row, eventually a memory_links row).
- **NONE** — the new memory is a true duplicate of an existing one with no
  added detail; drop the new memory before persistence.
- **ENRICH** — both memories describe the same underlying fact but each carries
  unique specifics that the other lacks (or the new memory would supersede but
  drops detail). The classifier keeps both alive and emits a memory_actions
  row tagged ENRICH; the merge step can then synthesize a merged richer memory.
- **UNRELATED** — the memories describe different facts; keep both, no link.

Feature-gated by ENABLE_KNOWLEDGE_UPDATE_DETECTION env var (default: true).

Design notes:
- The cascade phase from v1 was removed; empirical evidence showed it
  amplified per-decision errors 5-14x via top-15 neighbour sweeps with no
  analog in any comparable system. Removed with no replacement.
- The classifier prompt is dataset-neutral: every example uses placeholder
  names ("Person A", "the user", "Acme Corp") and rotates across domains
  (medical, software, finance, work, hobby) to validate that the rule
  generalises. Empirically reaches ~90%+ on a held-out 83-case test set
  with cross-domain coverage.
"""

import asyncio
import json
import logging
import os
import re
import unicodedata
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from synap.cloud.agents.ingestion.interfaces import LLMClient
from synap.cloud.infrastructure.llm.models import LLMTimeoutError
from synap.cloud.agents.ingestion.models import ExtractedMemory, MemoryType, Scope
from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper
from synap.cloud.shared import edition
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL
from synap.cloud.agents.ingestion.prompts.knowledge_update import (
    CLASSIFICATION_SYSTEM_PROMPT,
    BATCH_CLASSIFICATION_ADDENDUM,
    CLASSIFICATION_USER_PROMPT,
    DELETE_ADDENDUM as _DELETE_ADDENDUM,
    PROVENANCE_ADDENDUM as _PROVENANCE_ADDENDUM,
    build_batch_user_prompt,
    provenance_note,
)

logger = logging.getLogger(__name__)


def _build_dedup_scope_filter(scope: Scope) -> Optional[Dict[str, Any]]:
    """ChromaDB ``where`` filter restricting dedup candidates to the new
    memory's own owner scope.

    Without this filter the candidate search spans the entire
    ``synap_{customer_id}`` collection, so a new memory is compared against
    EVERY user's vectors in the customer. A second user ingesting content
    similar to a first user then has every memory classified NONE/SUPERSEDE
    against the *other* user's data and silently dropped from the vector
    index (multi-tenant isolation bug). Mirrors the read-side scoping in
    ``retrieval_manager._build_customer_scope_filter``:

    - user-scope ingest:     own memories + customer-shared (sentinel)
    - customer-scope ingest: customer-shared only, never individual users'
    - client-scope ingest:   namespace is already the client collection, so
                             no user_id filter is needed.

    Note: unlike retrieval, this intentionally ignores
    ``allow_cross_user_retrieval``. Dedup DROPS a brand-new memory; doing that
    based on another user's data is silent data loss and must stay
    per-user-isolated even when cross-user *reads* are permitted.
    """
    # ── the ladder ─────────────────────────────────────────────────────────
    #
    # Everything below this comment is the two-level model, and on a deeper
    # ladder it is not merely incomplete, it reopens the exact bug the
    # docstring above describes as fixed.
    #
    # A memory filed at an intermediate rung, a team or a practice or a region,
    # has no user of its own, so `derive_scope_level` calls it "customer" and
    # the write stamps `user_id = SENTINEL`. Every team under one tenant
    # therefore looks identical to the filter below, and they all live in the
    # same `synap_{customer_id}` collection. Team A's new memory is then
    # compared against Team B's, and KU can classify it NONE or SUPERSEDE and
    # drop it. That is the multi-tenant isolation bug named above, one rung up,
    # and it is silent: no error, no 4xx, just a memory that stopped existing.
    #
    # The chain answers it exactly. `build_scope_filter` is the same builder the
    # read path compiles, so dedup and retrieval agree by construction rather
    # than by two people remembering to change both. S-08 exists because they
    # did not.
    #
    # Cross-user stays OFF here whatever the instance allows, for the reason
    # the docstring gives: dedup DROPS a brand-new memory, and doing that on
    # the strength of somebody else's data is silent data loss.
    chain = getattr(scope, "chain", None)
    if chain:
        try:
            from synap.cloud.shared.scope_filter import (
                build_scope_filter,
                to_chroma_where,
            )

            node_ids = [str(r.node_id) for r in chain if getattr(r, "node_id", None)]
            if node_ids:
                by_role = {}
                for r in chain:
                    for name in (getattr(r, "roles", None)
                                 or ([r.role] if getattr(r, "role", None) else [])):
                        by_role.setdefault(name, str(r.node_id))
                # ⚠ The legacy arm is included only when the LEAF is a rung the
                # two-level model can actually express, and that condition is
                # the whole difference between this and the read path.
                #
                # A legacy record carries `customer_id` and `user_id` and no
                # rung. At the identity or tenant rung those two ids say
                # exactly where it sits, so matching it is right. At an
                # INTERMEDIATE rung they do not: every team under one tenant
                # writes `customer_id = <tenant>, user_id = SENTINEL`, so a
                # legacy arm there matches every sibling team and Team B's
                # memory becomes a dedup candidate for Team A's ingest. That is
                # the bug this whole change exists to close, arriving through
                # the back door.
                #
                # A read may include it, because a read only ever returns more
                # of what the client could already see. Dedup DROPS a
                # brand-new memory, so the same widening destroys data.
                #
                # The cost of leaving it out is that unstamped records at an
                # intermediate rung are not dedup candidates, so duplicates
                # accumulate there until the backfill stamps them. Failing to
                # dedup is recoverable. Wrongly deduping is not.
                #
                # Found by the test for the sibling case failing against the
                # first version of this fix, which included the arm
                # unconditionally.
                leaf = chain[-1]
                leaf_roles = (getattr(leaf, "roles", None)
                              or ([leaf.role] if getattr(leaf, "role", None) else []))
                leaf_is_addressable = any(
                    r in ("root", "tenant_of_record", "identity") for r in leaf_roles
                )

                # ⚠ The sentinel is ambiguous the moment a ladder has an
                # intermediate rung, and that is a SECOND way into the same
                # bug. On a five-rung
                # ladder, with the arm as it first shipped, an ingest at the
                # tenant rung AND an ingest at a patient both matched a
                # sibling care team's memory, because a team's record is
                # written `customer_id = <tenant>, user_id = <sentinel>` and
                # so is a tenant-shared record. Chroma cannot ask whether a
                # key is absent, so the two cannot be told apart inside the
                # filter.
                #
                # Which arm is safe depends on the CLIENT's ladder, never on
                # this request's chain: a tenant-scoped request carries the
                # same two-rung chain on a three-rung ladder and on a
                # seven-rung one, and the sibling teams that make the sentinel
                # ambiguous sit below the chain, not on it.
                #
                #   no intermediate rung  -> exactly today's filter. This is
                #       the default three-rung clients, and their compiled
                #       predicate is unchanged.
                #   leaf is the identity  -> keep the arm, drop the sentinel.
                #       `user_id == this person` is unambiguous whatever else
                #       the ladder holds, so unstamped records for that person
                #       stay dedup candidates while siblings do not.
                #   anything else         -> no legacy arm. Unstamped records
                #       under a deep ladder stop being candidates until the
                #       backfill stamps them.
                #
                # The asymmetry from the note above still governs the choice:
                # failing to dedup leaves a duplicate, which is recoverable.
                # Deduping across a rung deletes a memory, which is not.
                deep = any(
                    bool(getattr(r, "ladder_has_intermediate_rungs", False)) for r in chain
                )
                if not deep:
                    include_legacy, roll_up = leaf_is_addressable, True
                elif "identity" in leaf_roles:
                    include_legacy, roll_up = True, False
                else:
                    include_legacy, roll_up = False, True

                return to_chroma_where(build_scope_filter(
                    node_ids,
                    tenant_node_id=(str(scope.customer_id)
                                    if getattr(scope, "customer_id", None)
                                    else by_role.get("tenant_of_record")),
                    identity_node_id=(str(scope.user_id)
                                      if getattr(scope, "user_id", None)
                                      else None),
                    roll_up_to_customer=roll_up,
                    cross_user=False,
                    cross_customer=False,
                    include_legacy=include_legacy,
                ))
        except Exception as exc:  # noqa: BLE001
            # Never fail an ingest over the dedup filter. Falling through to the
            # two-level filter is what this code did before the ladder existed,
            # and it is narrow rather than wrong for the three-rung shape that
            # is still almost every client.
            logger.warning(
                "dedup scope filter fell back to the two-level form (%s: %s)",
                type(exc).__name__, exc,
            )

    if getattr(scope, "user_id", None):
        return {"user_id": {"$in": [str(scope.user_id), CUSTOMER_SCOPE_SENTINEL]}}
    if getattr(scope, "customer_id", None):
        return {"user_id": CUSTOMER_SCOPE_SENTINEL}
    return None


# Cosine similarity floor for candidate pairs. batch_search returns similarity
# (1 - distance). This is a BUDGET HINT, not a truth claim: it says which rows
# are cheap enough to be worth asking the classifier about. TOP_K_CANDIDATES is
# what actually bounds the cost.
#
# It was 0.82 until 2026-10-03, on the theory that two rows below 82% similarity
# cannot be the same fact. That theory is false, and the measurement is specific.
#
# M3, over the 278 within-document pairs a real model called one fact on 40 lme3
# documents: 31.7% sit BELOW 0.82. Median 0.880, minimum 0.470.
#
# The case that forced it, scope lme3_830ce83f, "where did Rachel move":
#
#   gold, document _2 : The user has a friend named Rachel who moved back to the
#                       suburbs.
#   document _1       : Rachel moved to Chicago.                        cos 0.639
#                       ...planning to visit Rachel, who recently moved
#                       to a new apartment in Chicago.                  cos 0.762
#                       ...prefers to ask Rachel about her neighborhood
#                       and favorite spots in Chicago.                  cos 0.643
#                       The user plans to visit Rachel in Chicago.      cos 0.595
#
# Those four are ranks 1, 2, 3 and 4 against the gold row out of 20 candidates,
# so they are ALREADY inside the top-5 budget. Not one clears 0.82. Shown the
# pair directly, this file's own classifier answers SUPERSEDE at 0.95 confidence.
# The classifier was never wrong; it was never asked. Three rows said Chicago,
# one said the suburbs, and the reader believed the majority.
#
# 0.55 is where all four become visible. 0.60 reaches three of them.
#
# The cost, measured rather than guessed, over 354 live rows in six lme3 scopes:
# 0.82 gives 389 classifier calls, 0.55 gives 1501, so 3.86x the CALLS. The
# dollar figure rises by much less, because a KU classify prompt is 96% fixed
# text and prompt caching on it was measured at 92.4%; what grows is the two
# short memory texts and the output. TOP_K_CANDIDATES still caps the worst case
# at 5 per new memory, so nothing here is unbounded.
#
# This changes merge behaviour for every client, deliberately and without a flag.
#
# TODO(similarity-calibrator): one number for every client and every embedding
# model. To be supplied per client by SimilarityCalibrator
# (synap/cloud/application/consolidation/similarity_calibrator.py). Decisions and
# build list: synap/docs/architecture/similarity_threshold_calibrator.md
SIMILARITY_THRESHOLD = 0.55


def _similarity_threshold() -> float:
    """Effective candidate-pair similarity gate.

    The gate can filter out a genuine state update BEFORE it reaches the
    classifier, so the old value is never superseded. That is not a hypothetical:
    see SIMILARITY_THRESHOLD above for the measured case. The env override lets
    ops move it without a code change, clamped to a sane band so a fat-fingered
    value can neither disable dedup nor admit everything.
    """
    raw = os.environ.get("SYNAP_KU_SIMILARITY_THRESHOLD")
    if raw is None:
        return SIMILARITY_THRESHOLD
    try:
        return min(max(float(raw), 0.5), 0.99)
    except (TypeError, ValueError):
        return SIMILARITY_THRESHOLD


def _delete_enabled() -> bool:
    """Gap 6: natural-language deletion. Default OFF so the "system never
    deletes" invariant holds until explicitly enabled (gated on the delete
    track's own ingestion eval). When on, the classifier may emit DELETE and
    the caller soft-deletes the target — deprecate + drop from retrieval, with
    the audit trail preserved (never a hard destroy)."""
    return os.environ.get("ENABLE_KU_DELETE", "false").lower() in ("true", "1", "yes")


def _batch_classify_enabled() -> bool:
    """One classify call per new memory instead of one per candidate pair.

    The classifier prompt is ~2,900 tokens and is re-sent on every pair, so it
    is 93% of KU's input spend. Batching the candidates for one new memory into
    a single call pays it once. Measured on 251 real pairs with gpt-6.1-sol at
    reasoning_effort=medium: input tokens 687k -> 211k, calls 251 -> 60, and
    correct effect on stored data 88.0% -> 93.8% with the Pattern B fix in.

    Default OFF: this changes the request shape for every client.
    """
    return os.environ.get("ENABLE_KU_BATCH_CLASSIFY", "false").lower() in ("true", "1", "yes")


def _ku_model_name() -> str:
    """The model the KU classifier runs on, resolved exactly as startup.py
    resolves it so a blank or padded value cannot yield an empty name."""
    return os.environ.get("SYNAP_KU_CLASSIFY_MODEL", "gpt-4o").strip() or "gpt-4o"


def _ku_reasoning_effort() -> Optional[str]:
    """Reasoning effort for the KU calls, or None on a chat model.

    Chat models reject the field outright, so it is only sent when the
    configured model is one the provider treats as a reasoning model.
    """
    from synap.cloud.infrastructure.llm.providers.openai_provider import (
        _COMPLETION_TOKEN_MODELS,
    )
    if not _ku_model_name().lower().startswith(_COMPLETION_TOKEN_MODELS):
        return None
    return os.environ.get("SYNAP_KU_REASONING_EFFORT", "medium").strip() or "medium"


def _ku_effort_kwargs() -> Dict[str, str]:
    effort = _ku_reasoning_effort()
    return {"reasoning_effort": effort} if effort else {}


# gpt-4o returns HTTP 400 above 16,384 completion tokens, which silently killed
# every batch of five or more in an earlier run and left only the easy small
# ones. Prefix-matched so the dated aliases (gpt-4o-2024-08-06) are covered.
_MODEL_COMPLETION_CAP = (("gpt-4o-mini", 16384), ("gpt-4o", 16384))
_DEFAULT_COMPLETION_CAP = 16384
_PROBED_LARGE_CAP = ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-astra", "gpt-6-luna")


def _classify_budget(model: str, n_candidates: int) -> int:
    """Reasoning tokens are drawn from the completion budget, so this covers
    the thinking as well as the answer."""
    m = (model or "").strip().lower()
    cap = _DEFAULT_COMPLETION_CAP
    for prefix, c in _MODEL_COMPLETION_CAP:
        if m.startswith(prefix):
            cap = c
            break
    else:
        if m.startswith(_PROBED_LARGE_CAP):
            cap = 128000
    return min(2000 * max(1, n_candidates) + 8192, cap)


def _echo_guard_enabled() -> bool:
    """Echo guard (WS2 of the echo-poisoning plan): an assistant-sourced
    memory may not SUPERSEDE a user-sourced (or unattributed) memory with a
    different value. Default OFF; rides on ENABLE_SPEAKER_ATTRIBUTION having
    populated assertion_source at extraction time — without attribution every
    memory is unattributed and the guard never fires."""
    return os.environ.get("ENABLE_ECHO_GUARD", "false").lower() in ("true", "1", "yes")


def _direction_by_document_date_enabled() -> bool:
    """Decide which of two conflicting memories is newer by the date of the
    CONVERSATION each came from, not by which document we happened to ingest
    first. Default OFF.

    The direction of a SUPERSEDE is set by arrival order: the memories of the
    document being ingested are the "new" side and whatever the vector search
    returns is the "existing" side. The comment on the sibling path below
    already states the intended rule -- "across documents the order comes from
    document_created_at" -- but nothing checked it, so a conversation ingested
    out of order retired a statement the user made LATER.

    Measured on the benchmark instance on 2026-10-08, counting only rows whose
    memory_actions.action is SUPERSEDE: of 507 supersessions, 54 (10.7%) had
    the winner's source conversation EARLIER than the loser's, and 105 more
    were same-day. (Do not count memories.superseded_by on its own: NONE writes
    that column too, for duplicate drops, and reading it undivided gives 818 of
    5,004, which is not this defect.) One of the 54 answers a graded question:
    the user said on 21 May that their friend Rachel had moved to Chicago and
    on 26 May that she "just moved back
    to the suburbs again"; the 26 May session was ingested first, so Chicago
    arrived as the "new" value and retired the suburbs, and the classifier's
    own reason reads "the new memory updating the location from the suburbs to
    Chicago". Retrieval hides superseded rows, so "where did Rachel move to"
    could only be answered Chicago. Others lose a current count the same way:
    a to-watch list went from 25 titles back to 20.

    When on, an inverted pair is re-classified with the sides swapped, so the
    model is asked the question in the right time order with every one of its
    information-loss guardrails applying to the right side. Only a SUPERSEDE
    on that second answer retires anything, and what it retires is the
    earlier-said memory, which is the arriving one. Re-run over all 54 real
    inverted pairs with the real prompt and gpt-4o: 21 retire the earlier
    statement, 33 retire nothing and serve both, 0 keep today's direction."""
    return os.environ.get(
        "SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "false"
    ).lower() in ("true", "1", "yes")


def _parse_document_date(value) -> Optional[datetime]:
    """The source conversation's date, from a datetime or an ISO string.

    Chroma metadata holds it as an ISO string and ExtractedMemory holds a
    datetime, so both shapes arrive here. A naive value is read as UTC, which
    is what the ingestion path writes. Anything unparseable returns None, and
    None means the guard does not fire -- never that the pair is ordered.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# Concrete value tokens (addresses, urls, digit runs) — used by the echo
# guard to tell "assistant restated the SAME value" (legal dedup supersession)
# from "assistant asserted a DIFFERENT value" (echo — must not retire what the
# user said). Keyword-only matches like the word "email" are deliberately not
# values.
_VALUE_TOKEN_PATTERN = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+"      # email addresses
    r"|https?://\S+"                  # urls
    r"|\+?\d[\d\s().-]{6,}\d"        # phone-shaped digit runs
    r"|\b\d{4,}\b",                   # card fragments / numeric ids
    re.IGNORECASE,
)


def _extract_value_tokens(text: str) -> frozenset:
    """Normalized concrete values mentioned in a memory. Emails/urls are
    lowercased (trailing sentence punctuation trimmed); digit runs (phones,
    cards) get separators stripped. Empty set = no comparable value."""
    tokens = set()
    for m in _VALUE_TOKEN_PATTERN.findall(text or ""):
        if "@" in m or "://" in m:
            tokens.add(m.lower().rstrip(".,;:"))
        else:
            tokens.add(re.sub(r"[\s().-]", "", m).lower())
    return frozenset(tokens)


def _value_aware_recall_enabled() -> bool:
    """Gap 5 recall fix: admit sub-threshold candidate pairs when BOTH texts
    mention the same volatile-value attribute class (email, phone, address,
    payment card, URL). A changed email/address is a high-entropy token swap
    that drags whole-sentence cosine below the gate (measured with the
    arctic-l-v2.0 model: distinct-email update 0.78, address update 0.71 vs
    the 0.82 gate) even though the pair is exactly the last-write-wins case
    the classifier exists for. Class-matched admission recovers those pairs
    while cross-attribute noise stays excluded (email-vs-phone raw ~0.42-0.75
    but never shares a class). Default OFF per the off-by-default rollout
    convention; the LLM classifier remains the precision layer."""
    return os.environ.get(
        "ENABLE_KU_VALUE_AWARE_RECALL", "false"
    ).lower() in ("true", "1", "yes")


# Floor for class-matched sub-threshold admission. Measured same-attribute
# value updates land at 0.71-0.78; same-user cross-attribute pairs that also
# share a class are rare, and the classifier resolves them. Clamped so the
# floor can never exceed the main gate nor drop into noise territory.
VALUE_RECALL_FLOOR = 0.55


def _value_recall_floor(threshold: float) -> float:
    raw = os.environ.get("SYNAP_KU_VALUE_RECALL_FLOOR")
    try:
        floor = float(raw) if raw is not None else VALUE_RECALL_FLOOR
    except (TypeError, ValueError):
        floor = VALUE_RECALL_FLOOR
    return min(max(floor, 0.40), threshold)


# Floor for retraction-shaped new memories (Gap 6 recall). A retraction is a
# *reference* to a fact, not a restatement of it, so it embeds far from its
# target: measured "asked not to keep their payment card" vs the stored card
# memory = 0.55, "asked to delete their shipping address" vs the address =
# 0.55, while retraction-vs-unrelated noise sits at 0.27-0.30. Without this
# floor the DELETE outcome can never fire at any usable global threshold —
# the pair that should be classified never forms.
RETRACTION_RECALL_FLOOR = 0.45


def _retraction_recall_floor(threshold: float) -> float:
    raw = os.environ.get("SYNAP_KU_RETRACTION_RECALL_FLOOR")
    try:
        floor = float(raw) if raw is not None else RETRACTION_RECALL_FLOOR
    except (TypeError, ValueError):
        floor = RETRACTION_RECALL_FLOOR
    return min(max(floor, 0.30), threshold)


# Attribute classes for value-aware recall. A class matches on a concrete
# value pattern (an email address, a phone-shaped number, a URL) or on the
# attribute's keyword (so "shipping address is <new address>" pairs with the
# old address memory even though street strings share no tokens). Both texts
# must share at least one class for a sub-threshold pair to be admitted.
_ATTRIBUTE_CLASS_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\be-?mail\b", re.IGNORECASE),
    "phone": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        r"|\b(?:phone|mobile|cell)\b",
        re.IGNORECASE,
    ),
    "address": re.compile(
        # "address" alone also appears in "email address" / "IP address" —
        # exclude those so an email memory doesn't inherit the postal class.
        r"\b(?:(?<!e-mail )(?<!email )(?<!ip )address|street|avenue|boulevard"
        r"|apartment|apt\.?|suite)\b",
        re.IGNORECASE,
    ),
    "payment": re.compile(
        r"\b(?:card|payment method|iban|account number)\b", re.IGNORECASE
    ),
    "url": re.compile(r"https?://\S+", re.IGNORECASE),
}


def _attribute_classes(text: str) -> frozenset:
    if not text:
        return frozenset()
    return frozenset(
        name for name, pat in _ATTRIBUTE_CLASS_PATTERNS.items() if pat.search(text)
    )


# Retraction-intent lexicon over extracted (third-person) memory text.
# Deliberately recall-oriented: a false positive only lowers the candidate
# gate for that one memory, and the classifier still has to answer DELETE.
#
# v2 — generalized after end-to-end testing: the extractor NORMALIZES retraction
# requests, e.g. "please don't keep my payment card on file" was stored as
# "The user prefers not to have their payment card kept on file", which the
# v1 verb-phrase list missed. Two semantic shapes now cover the space:
#   (a) negated retention — a negation token within a clause of a
#       keep/store/save/remember/on-file verb ("prefers NOT to have it KEPT",
#       "doesN'T want it STORED", "should NOT be ON FILE");
#   (b) removal request — a desire/request verb within a clause of a
#       removal outcome ("WANTS it REMOVED", "REQUESTED the DELETION",
#       "ASKED to FORGET"), or a direct removal verb on a possessive.
_RETRACTION_PATTERNS = [
    # (a) negated retention: not / n't / never / no longer / stop / without
    #     ... keep|kept|store(d)|save(d)|retain(ed)|remember(ed)|on file
    re.compile(r"(?:\b(?:not|never|no\s+longer|stops?|stopped|without)\b|n['’‘`]?t\b)"
               r"[^.!?]{0,60}?"
               r"\b(?:keep(?:s|ing)?|kept|stor(?:e|es|ed|ing|age)|sav(?:e|es|ed|ing)"
               r"|retain(?:s|ed|ing)?|remember(?:s|ed|ing)?|on\s+file)\b",
               re.IGNORECASE),
    # (b) removal request: want/prefer/ask/request/would like ... removed/
    #     deleted/forgotten/erased/discarded/wiped/removal/deletion/gone
    re.compile(r"\b(?:wants?|wanted|prefers?|preferred|asks?|asked|requests?"
               r"|requested|would\s+like|likes?)\b"
               r"[^.!?]{0,80}?"
               r"\b(?:remov(?:ed|al)|delet(?:ed|ion)|forgotten|forget|eras(?:ed|ure)"
               r"|discarded|wiped|scrubbed|gone)\b",
               re.IGNORECASE),
    # direct removal verb on a possessive ("delete my address",
    # "remove the user's card")
    re.compile(r"\b(?:delete|remove|forget|erase|discard|wipe|scrub)\s+"
               r"(?:my|their|his|her|its|the\s+user['’‘`]?s?)\b", re.IGNORECASE),
    re.compile(r"\bretract(?:s|ed|ion)?\b", re.IGNORECASE),
]


def _is_retraction_request(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _RETRACTION_PATTERNS)


def _delete_subject_guard_enabled() -> bool:
    """Subject guard for DELETE verdicts (default OFF). The retraction recall
    floor (0.45) admits ANY vector-similar memory as a candidate pair, and the
    DELETE branch used to trust the paired memory as the deletion target
    unchecked, which mis-targeted: "delete my home address"
    deleted "The user lives in Texas" while the address fact survived. When
    on: (a) a DELETE verdict only executes when the paired memory matches the
    retraction's subject; (b) the retraction instead drives a subject-directed
    sweep over ALL of the user's current memories, so every copy of the
    retracted fact is retired, not just the one vector-paired row."""
    return os.environ.get(
        "ENABLE_KU_DELETE_SUBJECT_GUARD", "false"
    ).lower() in ("true", "1", "yes")


# Scaffolding words that carry no retraction *subject*: pronouns/aux verbs,
# the retraction verbs themselves (delete/remove/forget/... and their
# negated-retention counterparts keep/store/save/...), and generic nouns the
# extractor's normalization adds ("from the system", "records"). What remains
# after filtering is the subject ("home address", "payment card", "jazz").
_SUBJECT_STOPWORDS = frozenset("""
    the user users they their them theirs this that these those from with your
    you and for not any all about into onto been being will would should could
    can may might shall must have has had having was were are is be its his
    her also just now here there when what which who how
    prefers prefer preferred wants want wanted wishes wish wished asks ask
    asked requests request requested likes like liked longer anymore no
    delete deleted deleting deletion remove removed removing removal forget
    forgotten forgetting forgets erase erased erasing erasure discard
    discarded wipe wiped scrub scrubbed retract retracted retraction gone
    keep keeps kept keeping store stored storing storage save saved saving
    retain retained retaining remember remembered remembering file filed
    system records record data information info please dont stop stopped
""".split())

_SUBJECT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _subject_tokens(text: str) -> frozenset:
    """Content-bearing tokens of a memory text: lowercase alnum runs, len>=3,
    minus scaffolding/retraction stopwords and pure digit runs (those are
    value tokens, handled by _extract_value_tokens)."""
    tokens = set()
    for t in _SUBJECT_TOKEN_RE.findall((text or "").lower()):
        if len(t) < 3 or t.isdigit() or t in _SUBJECT_STOPWORDS:
            continue
        tokens.add(t)
    return frozenset(tokens)


def _tokens_overlap(a: frozenset, b: frozenset) -> bool:
    """Loose singular/plural-tolerant overlap: exact match, or one token is a
    prefix of the other with >=4 shared chars ("live"/"lives",
    "address"/"addresses")."""
    for x in a:
        for y in b:
            if x == y:
                return True
            if len(x) >= 4 and len(y) >= 4 and (
                x.startswith(y) or y.startswith(x)
            ):
                return True
    return False


def _delete_subject_match(new_text: str, old_content: str) -> bool:
    """Does the paired existing memory actually talk about what the retraction
    retracts? Match on any of: shared concrete value token (email/number),
    subject-token overlap, or shared attribute class (email/phone/address/
    payment/url — covers paraphrase within a class). A retraction with NO
    extractable signal can't be validated — pass it through unblocked (the
    guard is a precision layer; the classifier already answered DELETE)."""
    new_vals = _extract_value_tokens(new_text)
    if new_vals and (new_vals & _extract_value_tokens(old_content)):
        return True
    subj = _subject_tokens(new_text)
    if subj and _tokens_overlap(subj, _subject_tokens(old_content)):
        return True
    new_classes = _attribute_classes(new_text)
    if new_classes and (new_classes & _attribute_classes(old_content)):
        return True
    return not new_vals and not subj and not new_classes


def _retraction_sweep_v2_enabled() -> bool:
    """Retraction sweep v2 (default OFF). The v1 sweep matches candidate rows
    by requiring ALL subject tokens as ILIKE substrings, which fails on the
    case "don't keep my card on file after this transaction" is
    extracted as "The user prefers not to have their payment card kept on file
    after the current transaction" -> subject tokens {payment, card, current,
    after, transaction}, while the stored row reads "The user's card ends in
    4242". Zero rows match, the card is never retired. v2 adds two things:
    (a) the conditional tail ("after the current transaction") is cut before
    tokenizing, and (b) rows sharing the retraction's ATTRIBUTE CLASS also
    qualify, so a payment retraction reaches any payment row regardless of
    wording. Off = byte-identical v1 behaviour."""
    return os.environ.get(
        "SYNAP_KU_RETRACTION_SWEEP_V2", "false"
    ).lower() in ("true", "1", "yes")


# Connectives that open a *conditional/temporal qualifier* on a retraction
# ("...kept on file AFTER the current transaction", "...deleted ONCE the order
# ships"). Everything past them describes WHEN the retraction applies, not
# WHAT is being retracted, so those words are noise in a subject-token match.
_CONDITIONAL_TAIL_RE = re.compile(
    r"\s(?:after|once|until|when|as\s+soon\s+as|following)\s",
    re.IGNORECASE,
)


def _truncate_conditional_tail(text: str) -> str:
    """Text up to (not including) the first conditional/temporal connective."""
    if not text:
        return text or ""
    match = _CONDITIONAL_TAIL_RE.search(text)
    return text[: match.start()] if match else text


# Cheap SQL prefilter keywords per attribute class (retraction sweep v2). The
# ILIKE fetch these drive is a RECALL net only — every fetched row is then
# re-confirmed in Python against the real _ATTRIBUTE_CLASS_PATTERNS before it
# can be retired, so a loose keyword here cannot widen what gets swept.
_CLASS_SQL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "payment": ("card", "payment"),
    "address": ("address", "street"),
    "email": ("email", "@"),
    "phone": ("phone", "mobile", "number"),
    "url": ("http", "www", "site"),
}


def _class_sql_keywords(classes) -> List[str]:
    """Deduplicated ILIKE prefilter keywords for a set of attribute classes."""
    keywords: List[str] = []
    for name in sorted(classes or ()):
        for keyword in _CLASS_SQL_KEYWORDS.get(name, ()):
            if keyword not in keywords:
                keywords.append(keyword)
    return keywords


def _retraction_class_match(retraction_classes, row_content: str) -> bool:
    """Does a candidate row carry the attribute class the retraction is about?

    Email/address precedence: "the user's email address is x@y" matches the
    `email` class, and an ADDRESS retraction must not sweep it — otherwise
    "forget my home address" would take the email with it. The postal class
    only reaches rows that are not email rows (unless the retraction itself is
    about email too, e.g. "delete my email address")."""
    if not retraction_classes:
        return False
    row_classes = _attribute_classes(row_content)
    shared = retraction_classes & row_classes
    if not shared:
        return False
    if (
        "address" in shared
        and "email" in row_classes
        and "email" not in retraction_classes
    ):
        return False
    return True


def _retired_value_guard_enabled() -> bool:
    """Retired-value guard (default OFF). A later extraction that re-asserted
    the user's OLD notification email SUPERSEDED the row carrying the NEW
    one ("updates the domain of the user's notification email address"), silently rolling
    the email back. When
    on, a SUPERSEDE whose NEW content carries a high-entropy value that was
    already RETIRED for this user is blocked — unless the text explicitly
    signals a flip ("switched back to old@x"), which is a legitimate revert."""
    return os.environ.get(
        "SYNAP_KU_RETIRED_VALUE_GUARD", "false"
    ).lower() in ("true", "1", "yes")


def _value_sibling_sweep_enabled() -> bool:
    """Sibling value sweep (default OFF). Superseding the primary email row
    leaves paraphrase siblings ("the user prefers to be contacted at old@x")
    alive as CURRENT, so retrieval serves two emails at once (the
    base email was re-CREATEd after the change). When on, a successful SUPERSEDE
    also retires every other current row carrying the superseded value."""
    return os.environ.get(
        "SYNAP_KU_VALUE_SIBLING_SWEEP", "false"
    ).lower() in ("true", "1", "yes")


# High-entropy values only: a full email address, a URL, or a phone-shaped run
# with >=7 digits. Deliberately NARROWER than _VALUE_TOKEN_PATTERN, which also
# admits any 4+ digit run ("ends in 4242", "Suite 1200") — those repeat across
# unrelated facts and must never drive a retire-by-value decision.
_HIGH_ENTROPY_VALUE_PATTERN = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+"      # email addresses
    r"|https?://\S+"                  # urls
    r"|\+?\d[\d\s().-]{5,}\d",       # phone-shaped digit runs
    re.IGNORECASE,
)


def _extract_high_entropy_values(text: str) -> frozenset:
    """Lowercased surface forms of the high-entropy values in a memory.

    The surface form (not a normalized one) is kept because these values are
    used as ILIKE needles against stored content — stripping a phone's
    separators would stop it matching the text it was found in."""
    values = set()
    for match in _HIGH_ENTROPY_VALUE_PATTERN.findall(text or ""):
        candidate = match.strip().rstrip(".,;:")
        if not candidate:
            continue
        if "@" in candidate or "://" in candidate:
            values.add(candidate.lower())
            continue
        if len(re.sub(r"\D", "", candidate)) >= 7:
            values.add(candidate.lower())
    return frozenset(values)


# Explicit "the value changed" markers. Their presence means a re-assertion of
# a previously retired value is an intentional flip back, not an echo of stale
# context — the retired-value guard stands down.
_VALUE_CHANGE_MARKER_RE = re.compile(
    r"\b(?:chang\w*|switch\w*|now\s+use[sd]?|instead|updat\w*"
    r"|back\s+to|revert\w*|actually)\b",
    re.IGNORECASE,
)


def _has_value_change_marker(text: str) -> bool:
    return bool(text) and bool(_VALUE_CHANGE_MARKER_RE.search(text))


# Types that are append-only — events and emotions don't
# decay or get superseded; KU should not classify them at all.
# Gated by ENABLE_KU_TYPE_AWARE_DISPATCH (default: false until rolled out).
APPEND_ONLY_TYPES = frozenset({MemoryType.EPISODES, MemoryType.EMOTIONS})

# Maximum existing memories to compare per new memory.
TOP_K_CANDIDATES = 5


@dataclass
class SupersessionAction:
    """A single supersession to execute (action='SUPERSEDE')."""
    old_memory_id: str          # UUID string of the existing memory to deprecate
    old_memory_content: str     # For changelog
    new_memory_id: str          # UUID string of the new memory replacing it
    new_memory_content: str     # For changelog
    reason: str                 # LLM-generated rationale (surfaced to clients)
    # KU v2: type of the memory whose ID is in the action row
    # (= old_memory_id for SUPERSEDE). Carried via ChromaDB metadata so the
    # changelog can group by type without an extra Postgres lookup.
    memory_type: Optional[str] = None
    # Raw utterance the new memory was extracted from, when available. Only
    # read by the retired-value guard (SYNAP_KU_RETIRED_VALUE_GUARD), which
    # looks for explicit change markers ("switched back to ...") that the
    # third-person extraction can drop. Optional so every existing
    # construction/consumer is unchanged.
    new_memory_source_span: Optional[str] = None


@dataclass
class SkipAction:
    """A new memory classified as a true duplicate (action='NONE').

    The new memory should be removed before vector persistence; an existing
    memory already carries the same fact. We point `superseded_by` at the
    survivor so the audit trail makes sense ("new was a duplicate, existing
    is the survivor").
    """
    new_memory_id: str          # UUID of the new memory to drop
    new_memory_content: str
    existing_memory_id: str     # UUID of the existing memory that survives
    existing_memory_content: str
    reason: str
    # Type of the dropped new memory (= action row's memory_id).
    memory_type: Optional[str] = None
    # What the changelog row calls this. "NONE" (the default) is a duplicate
    # drop. The out-of-order direction guard reuses this action because the
    # database outcome is identical -- retire the new row, point it at the
    # existing survivor -- but it is a real supersession, not a duplicate, and
    # a client reading its changelog has to be told which one happened.
    action_label: str = "NONE"


@dataclass
class EnrichmentAction:
    """Two memories that share an underlying fact but each carry unique
    specifics (action='ENRICH').

    The classifier keeps both memories alive (no link generation, no merge synthesis)
    and emits a memory_actions row tagged ENRICH for client visibility.
    The merge step uses these rows as input to the link generation + merged-content
    synthesis path.
    """
    existing_memory_id: str
    existing_memory_content: str
    new_memory_id: str
    new_memory_content: str
    reason: str
    # Type of the existing memory (= action row's memory_id).
    memory_type: Optional[str] = None
    # Lossless-merge guardrail: type of the new memory too, so
    # synthesize_merges can refuse clusters containing APPEND_ONLY types
    # even when ENABLE_KU_TYPE_AWARE_DISPATCH is off (defense in depth).
    new_memory_type: Optional[str] = None


@dataclass
class SubjectCandidates:
    """A new memory and the stored statements worth asking the subject question about.

    Deliberately NOT a decision. The first version of this decided here, from
    the screen's own score with a floor and a margin, and both numbers were
    wrong in ways no number could fix: the margin refused a group BECAUSE the
    thing was mentioned often (ten statements about one coffee table all score
    0.86-0.99, so the gap is 0.000), and the floor put correctness back on a
    threshold, which is the thing the screen exists to stop doing.

    So knowledge update hands over a shortlist and nothing more. The storage
    coordinator asks the question, because the answer also depends on the groups
    that already exist and those live in Postgres, which this stage cannot read.

    The shortlist is the screen's admitted set, which is a COST bound: it says
    which pairs are cheap enough to ask about, not which pairs are one thing.
    """
    new_memory_id: str
    new_memory_content: str
    #: ``[(memory_id, content), ...]`` in the order the screen saw them.
    candidates: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class DeletionAction:
    """A memory the user explicitly asked to forget/retract (action='DELETE').

    Gap 6, flag-gated by ENABLE_KU_DELETE. Executed as a SOFT delete: the
    target is deprecated + removed from vector retrieval, never hard-destroyed,
    so the audit trail (RDBMS row + changelog) is preserved. ``new_memory_*``
    is the retraction memory that triggered the deletion.
    """
    old_memory_id: str          # UUID of the existing memory to remove
    old_memory_content: str
    new_memory_id: str          # UUID of the retraction memory that triggered it
    new_memory_content: str
    reason: str
    memory_type: Optional[str] = None


@dataclass
class RetractionSweep:
    """A confirmed retraction whose subject should retire ALL matching current
    memories, not just the one vector-paired row (ENABLE_KU_DELETE_SUBJECT_GUARD).

    Built in the DELETE branch whenever the classifier answers DELETE for a
    retraction-shaped new memory — including when the subject guard blocked
    the paired target as a mis-match. ``subject_tokens``/``value_tokens``
    drive an ILIKE sweep in the coordinator; entries are only emitted when the
    signal is strong enough to be precise (a concrete value token, or >=2
    subject tokens).
    """
    new_memory_id: str          # UUID of the retraction memory
    new_memory_content: str
    subject_tokens: tuple       # e.g. ("home", "address")
    value_tokens: tuple         # e.g. ("casey.old@example.com",)
    reason: str                 # classifier rationale, for changelog rows


@dataclass
class DeferAction:
    """Consolidation D2/D-003: the classifier's call landed in the defer band
    (uncertain but plausible). Neither memory is touched; an adjudication
    case carries the question to the next cycle. Only emitted when
    SYNAP_KU_SHARED_BANDS is on."""
    case_type: str                       # supersession | enrichment
    proposed_action: str                 # SUPERSEDE | ENRICH
    trigger_memory_id: str               # the NEW memory
    candidate_memory_id: str             # the existing memory
    confidence: float
    reason: str
    memory_type: Optional[str] = None


@dataclass
class KnowledgeUpdateResult:
    """Aggregated outcomes from one detect_updates run."""
    supersessions: List[SupersessionAction]
    skips: List[SkipAction]
    enrichments: List[EnrichmentAction]
    # Gap 6 — populated only when ENABLE_KU_DELETE is on; empty otherwise, so
    # existing callers/constructions are unchanged.
    deletions: List[DeletionAction] = field(default_factory=list)
    # Subject guard — populated only when ENABLE_KU_DELETE_SUBJECT_GUARD is
    # also on; empty otherwise.
    retraction_sweeps: List[RetractionSweep] = field(default_factory=list)
    # D2 shared bands — populated only when SYNAP_KU_SHARED_BANDS is on.
    defers: List[DeferAction] = field(default_factory=list)
    # Migration 159 — populated only when the Jev screen AND subject linking are
    # both on for the instance; empty otherwise, so every existing caller and
    # every existing construction is unchanged.
    subject_candidates: List[SubjectCandidates] = field(default_factory=list)


def _lexicon_retraction_sweep(memory) -> Optional[RetractionSweep]:
    """Sweep for a lexicon-detected retraction, independent of pair recall.

    The paired DELETE path only runs when vector recall admits a candidate
    (retraction floor 0.45) AND the classifier answers DELETE. A retraction
    phrased unlike its target ("prefers not to keep their card on file" vs
    "provided a payment card ending in 4242") embeds below the floor, no
    pair forms, and nothing is retired. The
    subject/class sweep does its own precise targeting in the RDBMS, so it
    can be emitted from the lexicon signal alone. v2-only (both call sites
    are gated on SYNAP_KU_RETRACTION_SWEEP_V2).
    """
    parts = [
        t for t in (memory.content, getattr(memory, "source_span", None)) if t
    ]
    if not parts:
        return None
    guard_text = " ".join(parts)
    if not _is_retraction_request(guard_text):
        return None
    subject_source = " ".join(_truncate_conditional_tail(p) for p in parts)
    subj = _subject_tokens(subject_source)
    vals = _extract_value_tokens(guard_text)
    # Strong-signal gate: a concrete value token, >=2 subject tokens, or a
    # single subject token backed by an attribute class ("their card" ->
    # {card} + payment). The coordinator's class confirmation keeps the
    # one-token form precise.
    if not (
        vals
        or len(subj) >= 2
        or (len(subj) >= 1 and _attribute_classes(guard_text))
    ):
        return None
    return RetractionSweep(
        new_memory_id=str(memory.memory_id),
        new_memory_content=memory.content,
        subject_tokens=tuple(sorted(subj)),
        value_tokens=tuple(sorted(vals)),
        reason="lexicon-detected retraction (standalone sweep)",
    )


# Hierarchy used when contributors disagree on memory_type (modal-tied case).
# Listed most → least informational; the first match in the cluster wins.
_MEMORY_TYPE_HIERARCHY = (
    "temporal_events",
    "temporal_facts",
    "emotions",
    "episodes",
    "preferences",
    "facts",
)


def resolve_merge_metadata(
    contributors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the merged-memory metadata from N contributors' DB rows.

    Pure function — easy to unit-test. Caller (Stage 4) loads each contributor's
    row from `memories` (one batched SELECT), passes the list of dicts in here,
    receives back the resolved field values to attach to the MergeAction.

    Each contributor dict is expected to expose (None-safe):
      memory_type, category, subcategory, confidence,
      temporal_category, valid_from, valid_until, temporal_confidence,
      is_episode, pii_obfuscated,
      document_created_at, extracted_at, event_date

    Policies (one per metadata field, see KU v2 lossless-merge design):
      memory_type      modal; tied → first match in _MEMORY_TYPE_HIERARCHY
      category         modal; NULL on tie
      subcategory      modal among matching-category contributors; NULL otherwise
      confidence       1 - prod(1 - c.conf), capped 0.99   (independent-OR)
      temporal_category episode > temporal_fact > perpetual; episode demotes to temporal_fact
      valid_from       min(valid_from); fallback min(event_date)
      valid_until      NULL if any None; else max
      temporal_confidence inherited from contributor whose valid_from we used
      is_episode       derived from final temporal_category
      pii_obfuscated   OR across contributors
      document_created_at  min — earliest known mention
      document_last_observed_at  max — latest known mention
      extracted_at     min — when the system first knew

    Returns a dict ready to splat into MergeAction(**resolved).
    """
    n = len(contributors)
    if n == 0:
        return {}

    # --- memory_type (modal w/ hierarchy tiebreak; demote episode below) ---
    type_counts = Counter(
        c.get("memory_type") for c in contributors if c.get("memory_type")
    )
    if type_counts:
        top_count = type_counts.most_common(1)[0][1]
        tied = [t for t, k in type_counts.items() if k == top_count]
        if len(tied) == 1:
            primary_type = tied[0]
        else:
            primary_type = next(
                (t for t in _MEMORY_TYPE_HIERARCHY if t in tied),
                tied[0],
            )
    else:
        primary_type = None

    # --- category (modal; NULL on tie) ----------------------------------
    cat_counts = Counter(c.get("category") for c in contributors if c.get("category"))
    category = None
    if cat_counts:
        top = cat_counts.most_common(1)[0][1]
        tied_cats = [c for c, k in cat_counts.items() if k == top]
        category = tied_cats[0] if len(tied_cats) == 1 else None

    # --- subcategory (modal among matching-category contribs; NULL on tie) ---
    subcategory = None
    if category is not None:
        sub_counts = Counter(
            c.get("subcategory") for c in contributors
            if c.get("category") == category and c.get("subcategory")
        )
        if sub_counts:
            top = sub_counts.most_common(1)[0][1]
            tied_subs = [s for s, k in sub_counts.items() if k == top]
            subcategory = tied_subs[0] if len(tied_subs) == 1 else None

    # --- confidence (independent-OR combination, capped 0.99) -----------
    # 1 - product(1 - c.conf). Treats each contributor as independent evidence.
    prob_none_true = 1.0
    for c in contributors:
        conf = c.get("confidence")
        if conf is None:
            continue
        try:
            prob_none_true *= max(0.0, 1.0 - float(conf))
        except (TypeError, ValueError):
            pass
    confidence = round(min(0.99, 1.0 - prob_none_true), 4)

    # --- temporal_category (most-temporal wins; episode demotes) -------
    cats_present = {c.get("temporal_category") for c in contributors}
    if "episode" in cats_present:
        # 3 episodes describing recurring activity → it's no longer a single
        # episode, it's a temporal fact about a recurring pattern.
        temporal_category = "temporal_fact"
    elif "temporal_fact" in cats_present:
        temporal_category = "temporal_fact"
    else:
        temporal_category = "perpetual"

    # --- valid_from (min(valid_from); fallback min(event_date)) ---------
    vfs = [c.get("valid_from") for c in contributors if c.get("valid_from")]
    if not vfs:
        vfs = [c.get("event_date") for c in contributors if c.get("event_date")]
    valid_from = min(vfs) if vfs else None

    # --- valid_until (NULL if any None; else max) -----------------------
    vu_values = [c.get("valid_until") for c in contributors]
    if any(v is None for v in vu_values):
        valid_until = None
    else:
        valid_until = max(vu_values) if vu_values else None

    # --- temporal_confidence (inherit from chosen-valid_from contributor) ---
    temporal_confidence = 0.0
    if valid_from is not None:
        for c in contributors:
            if c.get("valid_from") == valid_from:
                tc = c.get("temporal_confidence")
                if tc is not None:
                    try:
                        temporal_confidence = float(tc)
                    except (TypeError, ValueError):
                        temporal_confidence = 0.0
                break
    else:
        # No valid_from chosen; take max temporal_confidence as defensive default.
        tcs = [c.get("temporal_confidence") for c in contributors
               if c.get("temporal_confidence") is not None]
        if tcs:
            try:
                temporal_confidence = float(max(tcs))
            except (TypeError, ValueError):
                temporal_confidence = 0.0

    # --- is_episode (derived from final temporal_category) -------------
    is_episode = (temporal_category == "episode")

    # --- pii_obfuscated (OR) ------------------------------------------
    pii_obfuscated = any(bool(c.get("pii_obfuscated")) for c in contributors)

    # --- pii_detected (AND-of-checked, then OR-of-found) --------------
    # Not an OR, because this column answers "did a detector run", and a
    # merged row is only checked if every contributor was. One contributor
    # written before the ingress gate shipped means part of what this row now
    # says was never looked at, and the honest answer is NULL rather than a
    # confident FALSE (migration 083).
    _detected = [c.get("pii_detected") for c in contributors]
    if any(d is None for d in _detected) or not _detected:
        pii_detected = None
    else:
        pii_detected = any(bool(d) for d in _detected)

    # --- DCT range -----------------------------------------------------
    dcts = [c.get("document_created_at") for c in contributors
            if c.get("document_created_at")]
    document_created_at = min(dcts) if dcts else None
    document_last_observed_at = max(dcts) if dcts else None

    # --- extracted_at (min; "first observed by system") ----------------
    eas = [c.get("extracted_at") for c in contributors if c.get("extracted_at")]
    extracted_at = min(eas) if eas else None

    # --- event window ---------------------------------------------------------
    # This used to answer NULL whenever the windowed contributors were not
    # byte-identical, and that was wrong in all three cases it covered.
    #
    # A merge asserts the contributors describe ONE fact. So different windows
    # mean one of exactly three things:
    #
    #   1. They OVERLAP: the same event at two precisions. "March 2023" and
    #      "Mar 27-31" are not a disagreement. NULLing there destroyed a date we
    #      had, for nothing.
    #   2. They do NOT overlap: they cannot both be true of one event, so the
    #      premise of the merge is FALSE. The right answer is not to merge. The
    #      old answer merged anyway and wrote NULL over two real dates.
    #   3. Nobody has a window: we never knew. NULL, honestly.
    #
    # And NULL is not neutral. Downstream it does not read as "unknown": the row
    # falls back to valid_from / DCT and starts serving as though the thing
    # happened on the day it was mentioned, which is the exact defect this whole
    # workstream exists to remove.
    #
    # ⚠ The same rule reads CORRECTLY in consolidation/executors._inherited_event_date
    # and must stay there: that path DERIVES a new fact from several memories, and
    # a derived row spanning several events genuinely has no single event date.
    # The premise differs, so the answer differs. This is a merge.
    windowed = [c for c in contributors if c.get("event_start")]
    merge_refused_reason: Optional[str] = None
    event_start = event_end = event_precision = None
    if windowed:
        from synap.cloud.agents.ingestion.services.event_anchor import (
            precision_order, windows_conflict,
        )
        if windows_conflict([(c.get("event_start"), c.get("event_end"))
                             for c in windowed]):
            merge_refused_reason = (
                "contributors give event windows that cannot all describe one "
                "event, so they are not one fact")
        else:
            # Most precise wins; a month window containing a day range is the
            # same event said twice and the day range is the better answer.
            best = min(windowed,
                       key=lambda c: (precision_order(c.get("event_precision")),
                                      c.get("event_start")))
            event_start = best.get("event_start")
            event_end = best.get("event_end")
            event_precision = best.get("event_precision")

    return {
        "merge_refused_reason": merge_refused_reason,
        "primary_memory_type": primary_type,
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "temporal_category": temporal_category,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "temporal_confidence": temporal_confidence,
        "is_episode": is_episode,
        "pii_obfuscated": pii_obfuscated,
        "pii_detected": pii_detected,
        "document_created_at": document_created_at,
        "document_last_observed_at": document_last_observed_at,
        "extracted_at": extracted_at,
        "event_start": event_start,
        "event_end": event_end,
        "event_precision": event_precision,
    }


# ---------------------------------------------------------------------------
# Named-entity verbatim guardrail (merge follow-up; consolidation F2 shares
# this exact mechanism — one guardrail, N call sites).
#
# The classic lossy-merge failure: the merge LLM turns "Charlotte's Web" into
# "a children's classic", or drops "Nikon D40"/"2018" for "her first DSLR
# camera years ago". Length guardrails don't catch this — the output is long
# enough, it just paraphrased the anchors away. So: extract every token that
# must survive verbatim (quoted phrases, proper-noun phrases, numeric
# anchors), then reject any merge whose output lacks one (case-sensitive
# substring). Cheap regex, zero LLM calls, deliberately conservative: a
# false reject keeps contributors unmerged (safe); a false accept loses data.
# ---------------------------------------------------------------------------

_MUST_PRESERVE_STOPWORDS = frozenset({
    # articles / determiners / conjunctions / prepositions
    "the", "a", "an", "this", "that", "these", "those", "and", "but", "or",
    "so", "if", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "to", "with", "over", "under", "about", "after", "before", "when",
    "while", "then", "there", "here", "however", "also", "now", "not",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "her", "him", "his",
    "hers", "its", "our", "ours", "their", "theirs", "them", "me", "my",
    "mine", "your", "yours", "us",
    # days — relative anchors, never load-bearing verbatim
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
})

_QUOTED_SPAN_RE = re.compile(
    r'"([^"\n]{2,120})"|“([^”\n]{2,120})”')
_BRACKET_META_RE = re.compile(r"\[[^\]]*\]")   # [Context: ...] internal tags
def _build_currency_class() -> str:
    """Every Unicode currency symbol, as a regex character class.

    The numeric anchors used to match `\\$` alone, which made the merge recall
    guard BLIND to every rupee figure in the corpus. For example,
    a deep-clean merge dropped "₹5.3 Cr" and `find_dropped_entities` reported
    nothing, because the amount was never extracted as an anchor in the first
    place. The dollar control in the same sweep caught $120 correctly, which
    is how the gap was isolated to the symbol set rather than the mechanism.

    Deriving the set instead of listing it is the point. Hand-listing "$₹€£¥"
    would have fixed the rupee and left won, ruble, lira, baht and shekel
    exactly as broken, and would need editing again the first time a client
    invoices in one of them. Unicode already maintains this list under the
    general category Sc (Symbol, currency), so we read it rather than guess it.
    """
    symbols = [chr(c) for c in range(0x20, 0x2500)
               if unicodedata.category(chr(c)) == "Sc"]
    return "[" + re.escape("".join(symbols)) + "]"


_CURRENCY_CLASS = _build_currency_class()

_NUMERIC_ANCHOR_RES = (
    # Any currency amount, in any currency. See _CURRENCY_CLASS: the symbol set
    # is DERIVED from Unicode, not listed by hand, so a currency nobody here
    # has thought of is covered the same as the dollar.
    re.compile(rf"{_CURRENCY_CLASS}\s?\d[\d,. ]*\d|{_CURRENCY_CLASS}\s?\d"),
    # A "4.3 Cr" magnitude-word pattern was tried here and removed: on real
        # text it flagged "4.3 Cr" as dropped from a merge that plainly kept
    # "₹4.3", because the survivor wrote the range as "₹4.3–₹5.5 Cr" and the
    # digits were no longer adjacent to the unit. A false positive is safe for
    # DATA (it refuses the merge) but expensive for the product: enough of them
    # and nothing merges at all, which is the entire benefit of deep clean.
    re.compile(r"\b\d+x/\w+"),                  # 3x/week
    # Clock times. There was NO clock pattern here, so a merge could move a gym
    # session from 6:00 to any other hour and the guard allowed it.
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),      # 6:00, 14:30
    # Ratios and scales, widened from \d{2,3}/\d{2,3}: that form required two to
    # three digits on EACH side, so "120/80" was covered and the "1/16 scale"
    # of a model kit was not, because "1" is one digit.
    re.compile(r"\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\b"),   # 1/16, 120/80, 3/4
    re.compile(r"\b\d+\.\d+\b"),                      # 4.5, 1.75
    re.compile(r"\b(?:19|20)\d{2}\b"),                # years
)

# What the structured patterns above leave behind: a bare quantity. These are
# the numbers the failing questions actually turn on, and the guard protected
# NONE of them — "1250 followers", "300 points", "70 hours" all passed through,
# and the "needs a total of 300 points" memory had an EMPTY protected set, so a
# merge could have rewritten it to any number at all.
#
# Two digits or more only. A single bare digit is a list marker or a sentence's
# "1." about as often as it is a quantity, and every token that matches
# everywhere costs a merge without protecting anything. The consequence is a
# known, deliberate gap: "3 kids" is not protected as a digit. It IS protected
# when written "three kids" (see _NUMERAL_WORDS), which is the commoner form in
# extracted memory text.
_BARE_CARDINAL_RE = re.compile(r"\b\d{2,}\b")

# Spelled-out quantities, because extraction writes them as often as digits:
# "It took the user four hours to drive to the Outer Banks" protected only
# "Outer Banks", so the four was free to change.
#
# "one", "a" and "an" are deliberately absent. They are articles far more often
# than quantities, and a token present in nearly every sentence is not a guard.
_NUMERAL_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000",
}
_NUMERAL_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_NUMERAL_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)
_NUMERAL_WORDS_BY_DIGITS = {v: k for k, v in _NUMERAL_WORDS.items()}

_MONTH_NAMES = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may"
                r"|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?"
                r"|nov(?:ember)?|dec(?:ember)?")

# A whole date, consumed as one unit BEFORE bare quantities are read. Without
# this, "2023-04-15" shreds into the bare cardinals "04" and "15", and then the
# same day written "April 15, 2023" looks like it lost the "04" — which broke
# tidy's identity check for a purely cosmetic restyling.
#
# Only the YEAR is emitted, which is exactly what this guard protected before
# the quantity family existed. Whole-date equality is deliberately somebody
# else's job: `_canonical_dates` in consolidation/tidy.py already normalises
# both spellings to one `date:YYYY-MM-DD` token and compares those.
_DATE_SHAPE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"
    rf"|\b(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*(?:19|20)\d{{2}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})\.?,?\s*(?:19|20)\d{{2}}\b",
    re.IGNORECASE)
_YEAR_IN_DATE_RE = re.compile(r"(?:19|20)\d{2}")

# Canonical dates. Lifted here from consolidation/tidy.py, which has had them
# since 2026-09-16 and is the only caller that ever compared dates properly.
# This is now the one definition, like record_identifiers: tidy imports it.
#
# The failure it was written for: a four-digit YEAR anchor and
# nothing finer meant "2022-03-01" and "2022-02-01" both reduced to {"2022"} and
# compared equal, so meditation merged "finished listening ... on 2022-03-01"
# into "started listening ... on 2022-02-01" — two events a month apart, folded
# into one. The temporal-reasoning questions are answered off exactly these
# dates.
#
# Every full date reduces to ONE token whatever its spelling, so "2023-04-15"
# and "April 15, 2023" both give date:2023-04-15 and a pure restyling is not read
# as a loss. That is the point of canonicalising rather than comparing raw text.
#
# ⚠ All-numeric forms (04/15/2023) are deliberately NOT parsed: month-first and
# day-first cannot be told apart without a locale, and a guess would either
# invent losses or hide them. _DATE_SHAPE_RE still consumes them so they do not
# shred into bare quantities, and they fall back to their year.
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), start=1)}
_MONTHS.update({name[:3]: num for name, num in list(_MONTHS.items())})
_MONTHS["sept"] = 9

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_FIRST_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b")
_DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})\b")


def canonical_dates(content: Optional[str]) -> set:
    """Every year-month-day date in ``content`` as one ``date:YYYY-MM-DD``
    token, independent of how it was written."""
    out: set = set()
    if not content:
        return out

    def add(year: int, month: int, day: int) -> None:
        if 1 <= month <= 12 and 1 <= day <= 31:
            out.add(f"date:{year:04d}-{month:02d}-{day:02d}")

    for year, month, day in _ISO_DATE_RE.findall(content):
        add(int(year), int(month), int(day))
    for name, day, year in _MONTH_FIRST_RE.findall(content):
        month = _MONTHS.get(name.lower())
        if month:
            add(int(year), month, int(day))
    for day, name, year in _DAY_FIRST_RE.findall(content):
        month = _MONTHS.get(name.lower())
        if month:
            add(int(year), month, int(day))
    return out


# A token made only of digits and numeric punctuation: 1250, 6:00, 1/16, 4.5.
# Currency amounts ($120) and frequencies (3x/week) are deliberately NOT in this
# family — they keep the original substring rule, so their behaviour is unchanged.
_NUMERIC_TOKEN_RE = re.compile(r"^[\d.:/,]+$")


def _quantity_alias(token: str) -> Optional[str]:
    """The same quantity written the other way round, or None.

    A merge is allowed to write "4 hours" where a contributor said "four
    hours". The quantity survived; only its spelling changed. Without this the
    guard would refuse that merge, and a guard that refuses correct merges ends
    up refusing all of them, which is the failure mode the "4.3 Cr" note above
    already records.
    """
    low = token.lower()
    if low in _NUMERAL_WORDS:
        return _NUMERAL_WORDS[low]
    return _NUMERAL_WORDS_BY_DIGITS.get(low)


def _is_quantity_token(token: str) -> bool:
    return (token.lower() in _NUMERAL_WORDS
            or bool(_NUMERIC_TOKEN_RE.match(token)))


def _quantity_present(token: str, target: str) -> bool:
    """Is this quantity in the merged text, as a quantity?

    Boundary-matched, unlike the name families, which use a plain substring
    test. A substring test lets a merge satisfy a contributor's "70 hours" with
    "170 hours", and lets "four" be satisfied by "fourteen" — which is exactly
    the silent number change this family exists to stop. Either spelling counts.
    """
    forms = [token]
    alias = _quantity_alias(token)
    if alias:
        forms.append(alias)
    return any(
        re.search(rf"(?<!\w){re.escape(form)}(?!\w)", target, re.IGNORECASE)
        for form in forms
    )
_GUARD_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’$/-]*")
_CAMEL_BRAND_RE = re.compile(r"^[a-z]+[A-Z]")   # iPhone, eBay

# Record identifiers: #-prefixed digit runs and standalone runs of 5+ digits
# (order / ticket / invoice numbers). 4-digit years are excluded — they are
# already numeric anchors above. Normalised so "#844908" and "844908" are the
# same token. Canonical home (tidy.py imports it): a record id is identity, and
# a merge that drops one has collapsed two distinct records into one, which is
# not a paraphrase the entity guard should let through.
_RECORD_ID_RE = re.compile(r"#\s?\d{2,}|\b\d{5,}\b")
_POSSESSIVE_DROP_RE = re.compile(r"['’]s\b")
# Decorative quote characters at a WORD EDGE (straight and curly, single and
# double). A source phrase quoted for emphasis — 'Dermadew Glow Face Wash', "In
# Transit" — is captured by the proper-noun scan with the closing quote glued to
# its last word (_GUARD_WORD_RE allows a trailing quote), because _QUOTED_SPAN_RE
# only lifts DOUBLE-quoted spans. A merge that writes the phrase without the
# decorative quotes then reads as dropping it. Stripping edge quotes from BOTH
# the token and the target (they share _normalize_presence_target) removes that
# asymmetry only: an internal apostrophe (O'Brien) is preceded AND followed by a
# word char, so neither alternative matches and it is kept. Normalising the
# haystack cannot conjure a needle, so a genuinely missing quoted title is still
# missing after normalisation — the guard keeps its full power (see the
# find_dropped_entities tests that assert a paraphrased-away title is still
# caught). This case used to defer redundant merges.
_EDGE_QUOTE_RE = re.compile(
    r"(?<!\w)['\"’‘“”]+|['\"’‘“”]+(?!\w)")


def record_identifiers(content: Optional[str]) -> set:
    return {re.sub(r"[#\s]", "", m.group(0))
            for m in _RECORD_ID_RE.finditer(content or "")}


def _normalize_presence_target(text: str) -> str:
    """Normalise merged content the SAME way must-preserve tokens are derived,
    so the substring presence test does not report a drop over a purely cosmetic
    difference that is not data loss:
      - internal ``[redaction]`` tags are stripped (a token can be built across
        a stripped tag, e.g. "Motilal [a name] Midcap" -> phrase "Motilal
        Midcap", which then is not a verbatim substring of the raw content);
      - possessive ``'s`` is dropped (the phrase is possessive-stripped at
        extraction, so "Sarah's Bakery" yields "Sarah Bakery");
      - non-breaking / repeated whitespace collapses to one plain space (phrases
        join on a single space, but content may carry U+202F etc.);
      - decorative quote characters at a word edge are dropped (a phrase quoted
        for emphasis in the source keeps a glued trailing quote at extraction,
        which a merge that writes it bare would appear to lose).
    Each rule WIDENS what counts as present, and the widening is deliberately
    narrow: it forgives edge quotes, possessives, redaction tags and whitespace,
    nothing more. A multi-word entity must still appear with its words adjacent
    and in order, so the paraphrase-away drops the guard exists to catch are
    still reported (asserted by test_real_drop_still_caught_after_normalisation
    and test_single_quoted_phrase_genuinely_dropped_still_caught). Leniency is
    not free of risk — it could mask a contrived split, e.g. a source 'Shop
    Verify' against a merge that wrote 'Shop' and 'Verify' as two separate
    quoted items, since both normalise to the same string — but that is a
    narrow, engineered case, not the entity loss this guards against."""
    t = _BRACKET_META_RE.sub(" ", text)
    t = _POSSESSIVE_DROP_RE.sub("", t)
    t = _EDGE_QUOTE_RE.sub("", t)
    return re.sub(r"\s+", " ", t)


def _strip_possessive(word: str) -> str:
    for suffix in ("'s", "’s"):
        if word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _is_guard_stopword(word: str) -> bool:
    return _strip_possessive(word).lower() in _MUST_PRESERVE_STOPWORDS


def _is_name_word(word: str) -> bool:
    """Can this word START a must-preserve phrase?"""
    if _is_guard_stopword(word):
        return False
    first = word[:1]
    if first.isupper() and any(c.isalpha() for c in word):
        return True
    return bool(_CAMEL_BRAND_RE.match(word))    # lowercase-i brands


def _is_name_continuation(word: str) -> bool:
    """Can this word EXTEND a phrase? (model numbers, numerals, Caps words)"""
    if _is_guard_stopword(word):
        return False
    if any(c.isdigit() for c in word):
        return True                              # D40, A7, 14, 120/80
    return word[:1].isupper() and len(word) > 1  # Area, Web, Pro, III


def _is_sentence_start(text: str, offset: int) -> bool:
    before = text[:offset].rstrip(" \t\"'“”")
    return not before or before[-1] in ".!?\n"


def extract_must_preserve_tokens(text: Optional[str], *,
                                 quantities: bool = True,
                                 dates: bool = True) -> set:
    """Extract the tokens a merge must carry verbatim.

    ``quantities=False`` drops the bare-quantity and spelled-numeral families,
    and ``dates=False`` drops canonical dates in favour of the bare year.
    Together they leave the token set exactly as it was before either family
    existed. Compression passes False to both: its contract is that every anchor
    must land in the summary or in the loss manifest, and an anchor it can never
    restate would wedge the group forever rather than merely refuse one merge.
    Both of compression's call sites pass the same values on purpose, because two
    disagreeing definitions of "anchor" between the check and the tripwire is its
    own bug.

    Five families: quoted phrases (straight or smart quotes), numeric anchors
    ($amounts, clock times, ratios, decimals, frequencies, years), canonical
    dates, bare quantities and spelled-out numerals, and proper-noun phrases
    (consecutive capitalized/model-number words, e.g. "Sony A7 III"). Internal
    ``[Context: ...]`` tags are stripped first; stopwords, pronouns, days, and
    short sentence-starters never qualify.

    The quantity family was added after measuring this function on the exact
    details the LME3 failures turn on. It protected ``Instagram`` but not the
    ``1250`` beside it, ``Outer Banks`` but not the ``four hours``, and nothing
    whatsoever in "needs a total of 300 points". A merge is allowed to reword;
    it is not allowed to change a number, and until now only years, currency
    and a couple of ratio shapes said so.
    """
    if not text:
        return set()
    tokens: set = set()
    work = _BRACKET_META_RE.sub(" ", text)

    # 1. Quoted spans, verbatim — then removed so the noun scan can't
    #    re-tokenize their words into a possessive-stripped phrase that the
    #    merged content would legitimately lack.
    for match in _QUOTED_SPAN_RE.finditer(work):
        span = (match.group(1) or match.group(2) or "").strip()
        if span:
            tokens.add(span)
    work = _QUOTED_SPAN_RE.sub(" ", work)

    # 2. Numeric anchors, in two passes over a SEPARATE copy.
    #
    # Each structured match is blanked out before the next pattern runs, so a
    # clock time yields "6:00" and not also a stray bare "00", and a "1/16"
    # scale yields the scale and not also "16". Only what survives that blanking
    # is a genuine standalone quantity.
    #
    # The copy matters: the proper-noun scan reads digits too ("Sony A7
    # III", "Nikon D40"), so blanking numbers out of `work` itself would quietly
    # stop protecting product names that carry a model number.
    numeric_work = work
    # Whole dates first. Each one reduces to a single canonical token whatever
    # its spelling, so restyling a date is not a loss while CHANGING one is. A
    # shape canonical_dates cannot parse (the all-numeric 04/15/2023, ambiguous
    # without a locale) falls back to its year, which is what this guard
    # protected before.
    #
    # The date text is blanked out of BOTH copies, and `work` is the reason: the
    # proper-noun scan reads "April 15, 2023" as the phrase "April 15", so the
    # same day written in ISO form looked like it had lost that phrase. Blanking
    # it here is what makes the two spellings compare equal in both directions.
    for match in _DATE_SHAPE_RE.finditer(numeric_work):
        found = canonical_dates(match.group(0)) if dates else set()
        if found:
            tokens |= found
        else:
            year = _YEAR_IN_DATE_RE.search(match.group(0))
            if year:
                tokens.add(year.group(0))
    numeric_work = _DATE_SHAPE_RE.sub(" ", numeric_work)
    if dates:
        work = _DATE_SHAPE_RE.sub(" ", work)
    for regex in _NUMERIC_ANCHOR_RES:
        for match in regex.finditer(numeric_work):
            tokens.add(match.group(0))
        numeric_work = regex.sub(" ", numeric_work)
    if quantities:
        for match in _BARE_CARDINAL_RE.finditer(numeric_work):
            tokens.add(match.group(0))
        for match in _NUMERAL_WORD_RE.finditer(numeric_work):
            tokens.add(match.group(0))

    # 3. Proper-noun phrases. Words join only across plain spaces, so
    #    punctuation and sentence boundaries break phrases.
    words = [(m.group(0), m.start(), m.end()) for m in _GUARD_WORD_RE.finditer(work)]
    i = 0
    while i < len(words):
        word, start, end = words[i]
        if not _is_name_word(word):
            i += 1
            continue
        parts = [_strip_possessive(word)]
        j = i + 1
        last_end = end
        while j < len(words):
            nxt, nxt_start, nxt_end = words[j]
            gap = work[last_end:nxt_start]
            if gap.strip() != "" or not _is_name_continuation(nxt):
                break
            parts.append(_strip_possessive(nxt))
            last_end = nxt_end
            j += 1
        phrase = " ".join(parts)
        single = len(parts) == 1
        if single and len(phrase) <= 3 and _is_sentence_start(work, start):
            i = j                                # "But", "Yet" style starters
            continue
        if len(phrase) >= 2:
            tokens.add(phrase)
        i = j
    return tokens


def find_dropped_entities(contributors: Optional[List[str]],
                          merged_content: Optional[str], *,
                          quantities: bool = True,
                          dates: bool = True) -> set:
    """Which must-preserve tokens from any contributor are missing from the
    merged content? Multi-word phrases and anchors match case-sensitively
    (the anchor must survive verbatim, not merely be alluded to); single-word
    tokens match case-insensitively, because a sentence-initial "Favorite"
    legitimately becomes "favorite" mid-sentence in a merge. Empty merged
    content returns an empty set (there is no merge to guard).

    Two guards, one direction each:
      - MUST-PRESERVE tokens (names, quotes, anchors) tested against a merged
        content NORMALISED the same way the tokens were derived (see
        ``_normalize_presence_target``); this only removes false drops, so the
        guard stops refusing merges that lose nothing.
      - RECORD IDENTIFIERS (order / ticket numbers): a contributor's record id
        that the merge did not carry is a distinct record collapsed away, not a
        paraphrase. Reported as ``#<id>``. This adds refusals, so a live merge
        can no longer fold two different records into one (the failure the fast
        path had before D-029; now enforced for Nap and ingestion alike)."""
    if not merged_content:
        return set()
    required: set = set()
    contributor_ids: set = set()
    for contributor in contributors or []:
        required |= extract_must_preserve_tokens(contributor,
                                                 quantities=quantities,
                                                 dates=dates)
        contributor_ids |= record_identifiers(contributor)
    # Compare BOTH sides through the same normalisation. A quoted token keeps its
    # possessive ("Charlotte's Web") while a proper-noun phrase is stripped
    # ("Sarah Bakery"); normalising both the token and the target makes the test
    # uniform, so it neither false-drops the phrase nor false-passes a genuinely
    # missing quoted title.
    target = _normalize_presence_target(merged_content)
    target_lower = target.lower()
    dropped = set()
    merged_dates = canonical_dates(merged_content) if dates else set()
    for token in required:
        if token.startswith("date:"):
            # Compared canonically, so a restyled date is not a loss and a
            # CHANGED date is. The year-only anchor this replaces read
            # 2022-03-01 and 2022-02-01 as the same thing.
            if token not in merged_dates:
                dropped.add(token)
            continue
        norm = _normalize_presence_target(token)
        if _is_quantity_token(norm):
            # Quantities get their own rule: boundary-matched, and either
            # spelling counts. See _quantity_present.
            if not _quantity_present(norm, target):
                dropped.add(token)
        elif " " in norm:
            if norm not in target:
                dropped.add(token)
        elif norm.lower() not in target_lower:
            dropped.add(token)
    merged_ids = record_identifiers(merged_content)
    for rid in contributor_ids:
        if rid not in merged_ids:
            dropped.add("#" + rid)
    return dropped


@dataclass
class MergeAction:
    """A cluster of N memories to be merged into one.

    Built by `synthesize_merges` from the ENRICH classifications. Consumed by
    `StorageCoordinator.execute_merges`, which:
      - inserts a new memory row carrying every resolved metadata field
      - inserts N rows in `memory_source_documents` (union of contributors')
      - inserts entity link rows (union of contributors' memory_entity_links)
      - upserts the new vector + HQE in ChromaDB with full metadata mirror
      - deprecates each contributor (set deprecated_at + memory_links(kind='ENRICHED'))
      - hard-deletes contributors from ChromaDB (memory + HQE)

    Every metadata field below is *resolved from contributors* (not hard-coded).
    See knowledge_update._resolve_merge_metadata for the policies.
    """
    # --- The merge itself ---------------------------------------------------
    contributor_memory_ids: List[str]      # Existing memory IDs (≥2)
    contributor_contents: List[str]        # For changelog + audit
    contributor_source_documents: List[str]  # Union of source_document_ids
    merged_content: str                    # LLM-synthesized merged memory text
    rationale: str                         # LLM's one-line explanation

    # --- Resolved metadata (lossless merge) ---------------------
    # These are populated by Stage4Organizer._attach_metadata_to_merges via
    # one batched SELECT against the contributors' memories rows. Defaults
    # are set to safe NULL/zero so test code that doesn't fill them still
    # produces a valid INSERT.
    primary_memory_type: Optional[str] = None     # modal of contributors w/ hierarchy tiebreak
    category: Optional[str] = None                # modal; NULL on tie
    subcategory: Optional[str] = None             # modal among matching-category; NULL otherwise
    confidence: float = 0.9                       # 1 - prod(1 - c.conf), capped 0.99
    temporal_category: str = "perpetual"          # most-temporal wins; episode demotes to temporal_fact
    valid_from: Optional[datetime] = None         # min across contributors (or earliest event_date)
    valid_until: Optional[datetime] = None        # NULL if any still-valid; else max
    temporal_confidence: float = 0.0              # inherited from the contributor we leaned on
    is_episode: bool = False                      # derived from temporal_category
    pii_obfuscated: bool = False                  # OR across contributors
    pii_detected: Optional[bool] = None            # NULL if any contributor unchecked
    document_created_at: Optional[datetime] = None        # min(contributors.DCT)
    document_last_observed_at: Optional[datetime] = None  # max(contributors.DCT)
    extracted_at: Optional[datetime] = None       # min(contributors.extracted_at)
    event_start: Optional[datetime] = None        # most precise of the OVERLAPPING contributor windows
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    # Set when the contributors' windows cannot all describe one event, which
    # means the premise of the merge is false. The caller drops the merge rather
    # than executing it. See resolve_merge_metadata.
    merge_refused_reason: Optional[str] = None
    contributor_entity_ids: List[str] = field(default_factory=list)  # union for memory_entity_links


class KnowledgeUpdateDetector:
    """Detects how new memories relate to similar existing ones.

    Pipeline:
    1. Batch ChromaDB search for similar existing memories per new memory
       (top-K, similarity >= SIMILARITY_THRESHOLD).
    2. Pair-wise LLM classification into SUPERSEDE / NONE / UNRELATED / ENRICH.
    3. Aggregate into a KnowledgeUpdateResult; the caller routes each outcome
       to the right storage operation.
    """

    def __init__(
        self,
        vector: AsyncVectorWrapper,
        llm_client: LLMClient,
    ):
        self._vector = vector
        self._llm = llm_client

    async def detect_updates(
        self,
        memories: List[ExtractedMemory],
        memory_embeddings: List,
        scope: Scope,
        correlation_id: Optional[str] = None,
        document_content: Optional[str] = None,
    ) -> KnowledgeUpdateResult:
        """Detect how each new memory relates to existing ones.

        Args:
            memories: New memories about to be persisted.
            memory_embeddings: Pre-computed embeddings (same order as memories).
            scope: Tenant scope for namespace resolution.
            correlation_id: The ingestion request id. Stamped on the usage
                rows for every classification call so the tokens land under
                the request that caused them. Without it these rows carry a
                NULL correlation_id and no per-request cost view can see
                them — pairwise classification is the largest LLM consumer
                in the pipeline, so that hole was most of the ingestion's
                token spend.
            document_content: The conversation these memories came from. Used ONLY
                by the Jev candidate screen, as context for judging the new memory
                against stored ones, and ignored when that screen is off. Optional
                so a caller that does not have it loses the context, not the call.

        Returns:
            KnowledgeUpdateResult with one list per outcome class.
        """
        empty_result = KnowledgeUpdateResult(supersessions=[], skips=[], enrichments=[])

        if not memories or not memory_embeddings:
            return empty_result

        if memory_embeddings is None or len(memory_embeddings) == 0:
            return empty_result

        async with start_span("ingestion.knowledge_update.detect", {
            "synap.new_memory_count": len(memories),
        }):
            t0 = time.time()

            # Convert ndarray to list if needed (ChromaDB expects List[List[float]])
            converted_embeddings = []
            for emb in memory_embeddings:
                if hasattr(emb, 'tolist'):
                    converted_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    converted_embeddings.append(emb)
                else:
                    converted_embeddings.append(list(emb))

            # 1. Batch search for similar existing memories.
            #
            # Two failure modes need different handling:
            #
            # (a) Collection doesn't exist → genuine first ingestion for this
            #     user. Falling through with empty_result is correct: there's
            #     nothing older to supersede / enrich / dedupe against.
            #
            # (b) Transient ChromaDB error (read timeout, pool exhaustion,
            #     server 5xx) → NOT first ingestion. The collection may have
            #     plenty of prior memories that should have triggered
            #     SUPERSEDE / NONE / ENRICH decisions, but we couldn't reach
            #     them. Silently falling through to empty_result here is what
            #     caused the playground demo to surface only CREATE rows even
            #     for verbatim-duplicate user turns (see #739 verification
            #     run, 10/10 CREATE under sustained Chroma pressure).
            #
            # Distinguish by inspecting the error: chromadb / httpcore expose
            # "timed out", "pool", or HTTP status codes in str(e) for
            # transient errors; "does not exist" / "not found" / "no
            # collection" for missing-collection. Retry once with a short
            # backoff on transient errors before falling through.
            def _is_transient_chroma_error(err: Exception) -> bool:
                s = str(err).lower()
                if not s:
                    return False
                transient_markers = (
                    "timed out",
                    "timeout",
                    "pool timed out",
                    "connection refused",
                    "connection reset",
                    "service unavailable",
                    "bad gateway",
                    "internal server error",
                )
                return any(m in s for m in transient_markers)

            # Restrict candidates to the new memory's owner scope so dedup can
            # never match another user's vectors (multi-tenant isolation).
            # Gated for instant rollback; default on.
            dedup_filter = None
            if os.environ.get(
                "ENABLE_KU_USER_SCOPED_DEDUP", "true"
            ).lower() in ("true", "1", "yes"):
                dedup_filter = _build_dedup_scope_filter(scope)

            # Does a judge choose the candidates, or do the cosine floors?
            # Resolved before the search because it changes how wide the pull is.
            from synap.cloud.agents.ingestion.services import relatedness
            _screen_on = relatedness.enabled_for(getattr(scope, "instance_id", None))

            search_results = None
            last_err: Optional[Exception] = None
            for attempt in range(2):
                try:
                    search_results = await self._vector.batch_search(
                        embeddings=converted_embeddings,
                        scope=scope,
                        # Wide when a judge is going to read the pool, narrow when
                        # cosine floors are the only thing choosing. 5 exists
                        # because the classifier cost scales with what reaches it;
                        # once Jev screens, the pool is cheap and the floors are
                        # the thing hiding pairs (see services/relatedness).
                        top_k=(relatedness.pool_size() if _screen_on
                               else TOP_K_CANDIDATES),
                        filters=dedup_filter,
                    )
                    break
                except Exception as e:
                    last_err = e
                    if attempt == 0 and _is_transient_chroma_error(e):
                        # Brief backoff so a momentarily-saturated Chroma pool
                        # has a chance to drain before retry.
                        await asyncio.sleep(1.0)
                        continue
                    break

            if search_results is None:
                if last_err is not None and _is_transient_chroma_error(last_err):
                    logger.warning(
                        "Knowledge-update detection: ChromaDB transient error "
                        "after retry — KU will skip this batch and the "
                        "pipeline will emit CREATE for every candidate. "
                        "SUPERSEDE / ENRICH / NONE decisions are LOST for "
                        "this turn. Error: %s",
                        last_err,
                    )
                else:
                    logger.info(
                        "Knowledge-update detection: ChromaDB search failed "
                        "(likely first ingestion for this user): %s",
                        last_err,
                    )
                return empty_result

            # 2. Build candidate pairs: (new_idx, existing_id, existing_content,
            # existing_memory_type). The memory_type is carried via ChromaDB
            # metadata so SUPERSEDE/ENRICH action rows can be tagged
            # with the OLD memory's type without an extra Postgres round-trip.
            #
            # Type-aware dispatch (gated by ENABLE_KU_TYPE_AWARE_DISPATCH):
            # EPISODES and EMOTIONS are append-only — events don't decay, emotions
            # are bound to their moment. Filter them out at the new-memory level
            # before classification, so they pass through to the default ADD path
            # without ever invoking the LLM. Saves both cost and a
            # bug where two distinct concerts get superseded into one.
            type_aware = os.environ.get(
                "ENABLE_KU_TYPE_AWARE_DISPATCH", "false"
            ).lower() in ("true", "1", "yes")

            # (new_idx, existing_id, existing_content, existing_memory_type,
            #  existing_assertion_source, existing_document_created_at). The
            # last element is the date of the CONVERSATION the stored memory
            # came from, which is what orders two conflicting memories in time;
            # see _direction_by_document_date_enabled.
            candidate_pairs: List[Tuple[
                int, str, str, Optional[str], Optional[str], Any]] = []
            new_memory_ids = {str(m.memory_id) for m in memories}
            type_skipped_indices: set[int] = set()
            threshold = _similarity_threshold()  # Gap 5: env-tunable, default 0.55
            value_aware = _value_aware_recall_enabled()
            value_floor = _value_recall_floor(threshold)
            delete_on = _delete_enabled()
            retraction_floor = _retraction_recall_floor(threshold)

            # One Jev call per new memory, all of them at once. Each returns the
            # indices (into that memory's non-self candidate list, in order) that
            # are worth classifying, or None if the screen could not run, in which
            # case that memory falls back to the floors below. Never "all".
            # ── this document's own siblings ────────────────────────────────
            #
            # Knowledge update runs at stage4:639 and the new vectors are written
            # at 895, so a memory has only ever been compared against OLDER
            # DOCUMENTS. Its own document's other memories are invisible to it,
            # and coalesce only reaches the same-fact ones. A state change stated
            # inside one conversation therefore leaves two live rows.
            #
            # ⚠ The hard part is direction, not plumbing. Deprecating the wrong one
            # retires a true fact, which is worse than not looking. There is no
            # positional signal to order siblings by: `source_span` is always None
            # because the memory prompts stopped asking for it, and they stopped
            # because citing a span made the model refuse to build one memory from
            # two separate turns.
            #
            # But text position was never the right signal. SUPERSEDE needs to know
            # which STATE is more recent, and that is `event_start`, which is
            # already on the row and which the date pre-pass fills for exactly the
            # statement that needs it ("actually I moved last month"). Across
            # documents the order comes from document_created_at; for siblings that
            # value is identical, so event_start is what is left.
            #
            # So: a sibling pair is offered ONLY when both carry an event_start and
            # the dates differ, and it is always built with the EARLIER one as the
            # `existing` side, which is the side that gets deprecated. Equal dates,
            # or either missing, and the pair is not offered at all, because then we
            # genuinely do not know and refusing is the right answer.
            def _all_siblings(idx: int) -> List[int]:
                """Every other memory from this same document, in index order.

                The screen sees ALL of them, because the two questions asked of
                a sibling pair need different things. "Which of these is newer,
                so the older can be retired" needs dates and needs them to
                differ. "Are these two about the same thing" does not care about
                dates at all.

                The narrower list used to be built here and shared, and the
                consequence was measured: the three statements about one IKEA
                bookshelf all carry the same date, so the pair was never offered
                to the judge for EITHER question, and the bookshelf stayed
                ungrouped while the reader counted its statements separately.

                Deterministic, so the screen's indices and the pair builder's
                agree by construction rather than by two people remembering.
                """
                if idx >= len(memories):
                    return []
                return [j for j in range(len(memories)) if j != idx]

            def _is_earlier_sibling(idx: int, j: int) -> bool:
                """Whether sibling j's event is strictly earlier than idx's.

                The supersession gate, unchanged: a sibling pair is offered to
                the CLASSIFIER only when both carry an event_start and the dates
                differ, and always with the earlier one as the side that may be
                deprecated. Equal dates, or either missing, and we genuinely do
                not know which is newer, so refusing is the right answer.
                """
                mine = getattr(memories[idx], "event_start", None)
                theirs = getattr(memories[j], "event_start", None)
                return mine is not None and theirs is not None and theirs < mine

            screened: Dict[int, Optional[set]] = {}
            # The (memory_id, content) behind each candidate index, in the SAME
            # order the screen saw them, so an admitted index can be turned back
            # into a memory id. Empty unless the screen ran.
            #
            # The per-candidate SCORES are deliberately not kept any more. They
            # were what the floor and the margin decided from, and both were
            # wrong in ways a number could not fix; the subject question now
            # asks instead, and the scores only choose what is worth asking.
            subject_scores: Dict[int, Dict[int, float]] = {}
            cand_identity: Dict[int, List[Tuple[str, str]]] = {}
            if _screen_on:
                async def _screen_one(idx: int, results: Any):
                    if not isinstance(results, list) or idx >= len(memories):
                        return idx, set(), {}, []
                    # Each candidate carries the date that tells two similar
                    # events apart, straight out of the Chroma metadata it already
                    # has. See services/relatedness.build_state for why this is
                    # the date and not the whole source conversation.
                    cands = []
                    # Who each index IS, built in lockstep with cands so the
                    # screen's indices can be turned back into memory ids
                    # without a second pass that could drift out of order.
                    ident: List[Tuple[str, str]] = []
                    for r in results:
                        if r.get("id", "") in new_memory_ids:
                            continue
                        meta = r.get("metadata") or {}
                        cands.append({
                            "content": r.get("content", ""),
                            "memory_type": meta.get("memory_type"),
                            "event_start": meta.get("event_start"),
                            "document_created_at": meta.get("document_created_at"),
                        })
                        ident.append((str(r.get("id", "")), r.get("content", "")))
                    # Siblings go in the SAME call, appended after the stored ones,
                    # so one Jev request covers both and the index order is fixed.
                    for j in _all_siblings(idx):
                        sib = memories[j]
                        cands.append({
                            "content": sib.content or "",
                            "memory_type": getattr(
                                getattr(sib, "memory_type", None), "value", None),
                            "event_start": getattr(sib, "event_start", None),
                        })
                        ident.append((str(sib.memory_id), sib.content or ""))
                    adm, scores = await relatedness.screen_scored(
                        memories[idx].content or "", cands,
                        new_chunk=document_content,
                        correlation_id=correlation_id)
                    return idx, adm, scores, ident
                for _idx, _adm, _sc, _id in await asyncio.gather(*[
                        _screen_one(i, r) for i, r in enumerate(search_results)]):
                    screened[_idx] = _adm
                    subject_scores[_idx] = _sc
                    cand_identity[_idx] = _id
                _ran = sum(1 for v in screened.values() if v is not None)
                logger.info(
                    "Knowledge-update screen: jev ran for %d of %d new memories, "
                    "admitted %d pairs total, pool=%d",
                    _ran, len(screened),
                    sum(len(v) for v in screened.values() if v), relatedness.pool_size())

            # ── the shortlist for the subject question ─────────────────────
            #
            # No decision here. The screen already said which pairs are worth
            # looking at; this only records them, with the identity behind each
            # index so the coordinator can turn an answer back into a memory id.
            # The coordinator asks the question, because the answer also depends
            # on the groups that already exist and those live in Postgres.
            #
            # ⚠ An entry is emitted for EVERY surviving new memory, including
            # one whose shortlist is empty. It used to be emitted only when the
            # shortlist had something in it, and that silently cost recall: a
            # memory the screen matched to no individual statement was never
            # handed over, so the groups that already exist were never asked
            # about it. "The coffee table has metal legs" could score below the
            # screen's admit level against every stored statement and still
            # belong to the coffee-table GROUP, and nothing ever asked.
            #
            # Costs nothing when there is nothing to ask: ask_subject returns
            # without a call when the candidates AND the groups are both empty,
            # so a scope with no groups yet pays for no extra request.
            subject_candidates: List[SubjectCandidates] = []
            if _screen_on and relatedness.subject_link_enabled(
                    getattr(scope, "instance_id", None)):
                for new_idx, admitted in screened.items():
                    if new_idx >= len(memories):
                        continue
                    ident = cand_identity.get(new_idx) or []
                    # `admitted` is None when the screen could not run for this
                    # memory. That is not "nothing matched", but the shortlist is
                    # unknown either way, so it is treated as empty and the
                    # groups are still asked about.
                    idxs = sorted(admitted) if admitted else []
                    shortlist = [ident[i] for i in idxs
                                 if i < len(ident) and ident[i][0]]
                    # Never offer a memory against itself.
                    me = str(memories[new_idx].memory_id)
                    shortlist = [(mid, txt) for mid, txt in shortlist if mid != me]
                    subject_candidates.append(SubjectCandidates(
                        new_memory_id=me,
                        new_memory_content=memories[new_idx].content or "",
                        candidates=shortlist,
                    ))
                if subject_candidates:
                    _with = sum(1 for s in subject_candidates if s.candidates)
                    logger.info(
                        "Knowledge-update subject: %d new memories offered, %d "
                        "with a shortlist, %d pairs in total",
                        len(subject_candidates), _with,
                        sum(len(s.candidates) for s in subject_candidates))

            for new_idx, results_for_memory in enumerate(search_results):
                if not isinstance(results_for_memory, list):
                    continue
                screen_set = screened.get(new_idx) if _screen_on else None
                cand_i = -1

                # Type-aware short-circuit: skip the whole new memory if its
                # type is append-only. The new memory still gets persisted by
                # the rest of the pipeline (CREATE row in changelog) — KU just
                # doesn't try to relate it to anything older.
                if type_aware and new_idx < len(memories):
                    new_mem_type = getattr(memories[new_idx], "memory_type", None)
                    if new_mem_type in APPEND_ONLY_TYPES:
                        type_skipped_indices.add(new_idx)
                        continue

                new_content = (
                    memories[new_idx].content if new_idx < len(memories) else ""
                ) or ""
                # Gap 6 recall: a retraction references its target rather than
                # restating it, so the pair sits far below the main gate
                # (~0.50-0.55 measured). Lower the gate for this one new memory
                # so the DELETE-capable classifier actually gets to see the pair.
                is_retraction = delete_on and _is_retraction_request(new_content)
                new_classes = (
                    _attribute_classes(new_content) if value_aware else frozenset()
                )

                for result in results_for_memory:
                    existing_id = result.get("id", "")
                    # Skip self-comparison (re-ingestion of an already-stored memory).
                    if existing_id in new_memory_ids:
                        continue
                    cand_i += 1
                    similarity = result.get("score", 0.0)
                    if screen_set is not None:
                        # A judge chose this pool. The floors below are what the
                        # judge replaces, so none of them runs: stacking a cosine
                        # gate on top of a semantic verdict would reinstate exactly
                        # the recall ceiling this exists to remove.
                        if cand_i in screen_set:
                            meta = result.get("metadata") or {}
                            candidate_pairs.append((
                                new_idx, existing_id, result.get("content", ""),
                                meta.get("memory_type"), meta.get("assertion_source"),
                                meta.get("document_created_at"),
                            ))
                        continue
                    admit = similarity >= threshold
                    if not admit and is_retraction and similarity >= retraction_floor:
                        admit = True
                    if (
                        not admit
                        and value_aware
                        and new_classes
                        and similarity >= value_floor
                    ):
                        # Gap 5 recall: same attribute class on both sides means
                        # this is likely the same fact with a changed value —
                        # exactly what the gate was dropping. The classifier
                        # keeps precision (UNRELATED/ENRICH stay safe outcomes).
                        admit = bool(
                            new_classes & _attribute_classes(result.get("content", ""))
                        )
                    if admit:
                        meta = result.get("metadata") or {}
                        candidate_pairs.append((
                            new_idx,
                            existing_id,
                            result.get("content", ""),
                            meta.get("memory_type"),
                            meta.get("assertion_source"),
                            meta.get("document_created_at"),
                        ))

            # Sibling pairs, built from the same screen verdict. The screen saw
            # them at indices [n_stored ... n_stored + n_siblings), in the order
            # _earlier_siblings returns, so the mapping is recomputed not stored.
            sibling_pairs = 0
            if _screen_on:
                for new_idx in range(len(memories)):
                    screen_set = screened.get(new_idx)
                    if not screen_set:
                        continue
                    results = (search_results[new_idx]
                               if new_idx < len(search_results) else [])
                    n_stored = len([r for r in results
                                    if isinstance(r, dict)
                                    and r.get("id", "") not in new_memory_ids]
                                   ) if isinstance(results, list) else 0
                    for offset, j in enumerate(_all_siblings(new_idx)):
                        if (n_stored + offset) not in screen_set:
                            continue
                        # The screen scored every sibling; only the ones whose
                        # event is strictly earlier may be CLASSIFIED, because
                        # only those have a direction a supersession could use.
                        if not _is_earlier_sibling(new_idx, j):
                            continue
                        sib = memories[j]
                        candidate_pairs.append((
                            new_idx,                       # the LATER state
                            str(sib.memory_id),            # the earlier one, deprecated
                            sib.content or "",
                            getattr(getattr(sib, "memory_type", None), "value", None),
                            getattr(sib, "assertion_source", None),
                            # Same document on both sides, so the document-date
                            # guard can never fire on a sibling pair; these are
                            # already ordered by event_start above.
                            getattr(sib, "document_created_at", None),
                        ))
                        sibling_pairs += 1
                if sibling_pairs:
                    logger.info(
                        "Knowledge-update: %d in-session sibling pairs offered "
                        "(earlier event superseded by later, never the reverse)",
                        sibling_pairs)

            if type_aware and type_skipped_indices:
                logger.info(
                    "Knowledge-update detection: type-aware dispatch skipped %d new memories "
                    "(EPISODES/EMOTIONS are append-only)",
                    len(type_skipped_indices),
                )

            if not candidate_pairs:
                elapsed = (time.time() - t0) * 1000
                logger.info(
                    "Knowledge-update detection: no candidates found elapsed_ms=%.1f",
                    elapsed,
                )
                set_current_span_attributes({
                    "synap.ku_candidates": 0,
                    "synap.ku_elapsed_ms": round(elapsed, 1),
                })
                # v2: a lexicon-detected retraction must still sweep even when
                # vector recall found nothing to pair it with (a retraction
                # phrased unlike its target embeds below the 0.45 floor).
                if _delete_enabled() and _retraction_sweep_v2_enabled():
                    standalone = [
                        s for s in (
                            _lexicon_retraction_sweep(m) for m in memories
                        ) if s is not None
                    ]
                    if standalone:
                        logger.info(
                            "Knowledge-update detection: no candidate pairs, "
                            "emitting %d standalone retraction sweep(s)",
                            len(standalone),
                        )
                        return KnowledgeUpdateResult(
                            supersessions=[], skips=[], enrichments=[],
                            retraction_sweeps=standalone,
                        )
                return empty_result

            logger.info(
                "Knowledge-update detection: %d candidate pairs to classify",
                len(candidate_pairs),
            )

            # 3. Classify each pair with the LLM.
            supersessions: List[SupersessionAction] = []
            skips: List[SkipAction] = []
            enrichments: List[EnrichmentAction] = []
            deletions: List[DeletionAction] = []
            defers: List[DeferAction] = []
            delete_enabled = _delete_enabled()  # Gap 6, default OFF
            subject_guard_on = _delete_subject_guard_enabled()  # default OFF
            sweep_v2_on = _retraction_sweep_v2_enabled()  # default OFF
            # D2 (parity-gated, default OFF): route SUPERSEDE/ENRICH through
            # the SHARED authority bands — act >=0.85 executes as today, the
            # 0.50-0.85 band opens an adjudication case (both memories keep
            # serving), below 0.50 nothing happens. Flips ON only after the
            # LoCoMo/LongMemEval parity run clears (<=0.5pt regression rule).
            shared_bands_on = os.environ.get(
                "SYNAP_KU_SHARED_BANDS", "false").lower() in ("true", "1", "yes")
            bands_cfg = {
                "inline.act_threshold": float(
                    os.environ.get("SYNAP_KU_ACT_THRESHOLD", "0.85")),
                "inline.defer_floor": float(
                    os.environ.get("SYNAP_KU_DEFER_FLOOR", "0.50")),
            }
            # One sweep per retraction memory, keyed by new_id (multiple
            # candidate pairs for the same retraction must not sweep twice).
            sweeps_by_new_id: Dict[str, RetractionSweep] = {}

            # Track outcomes already attached to a given new memory so we don't
            # emit duplicate rows when multiple existing candidates match.
            new_id_outcomes: Dict[str, str] = {}
            # And track which old memories are already targets of a supersession
            # in this batch, to avoid double-superseding.
            seen_old_ids = set()

            echo_guard_on = _echo_guard_enabled()  # WS2, default OFF

            direction_guard_on = _direction_by_document_date_enabled()
            out_of_order_pairs = 0        # inverted pairs the guard caught
            out_of_order_retired = 0      # of those, the ones re-decided SUPERSEDE

            # One classify call per NEW memory instead of one per
            # pair. Skipped when the echo guard is on, because that path feeds
            # per-pair provenance into the prompt which the batch shape has no
            # place for. Anything the batch does not return falls through to
            # the per-pair call below, so this can only cost latency.
            batch_verdicts: Dict[Tuple[int, Any], Dict] = {}
            if _batch_classify_enabled() and candidate_pairs and not echo_guard_on:
                by_new: Dict[int, List[Tuple[Any, str]]] = {}
                for _cp in candidate_pairs:
                    by_new.setdefault(_cp[0], []).append((_cp[1], _cp[2]))
                for _ni, _cands in by_new.items():
                    _got = await self._classify_candidates(
                        new_content=memories[_ni].content,
                        candidates=_cands,
                        scope=scope,
                        correlation_id=correlation_id,
                    )
                    for _oid, _verdict in _got.items():
                        batch_verdicts[(_ni, _oid)] = _verdict

            for (new_idx, old_id, old_content, old_memory_type,
                 old_assertion_source, old_document_created_at) in candidate_pairs:
                new_memory = memories[new_idx]
                new_id_str = str(new_memory.memory_id)
                # Type of the new memory, for action rows whose
                # memory_id = new_id (NONE only).
                new_memory_type = (
                    new_memory.memory_type.value
                    if new_memory.memory_type is not None else None
                )

                # If this new memory already received a NONE verdict, the new
                # memory is being dropped — further pair classifications about
                # it are moot.
                if new_id_outcomes.get(new_id_str) == "NONE":
                    continue
                # If this old memory was already superseded by another new
                # memory in this batch, skip — newer wins.
                if old_id in seen_old_ids:
                    continue

                classification = batch_verdicts.get((new_idx, old_id))
                if classification is None:
                    classification = await self._classify_pair(
                        existing_content=old_content,
                        new_content=new_memory.content,
                        scope=scope,
                        correlation_id=correlation_id,
                        new_assertion_source=getattr(new_memory, "assertion_source", None),
                        existing_assertion_source=old_assertion_source,
                    )
                if not classification:
                    continue
                verdict = classification.get("classification", "")
                reason = classification.get("reason", "")

                if shared_bands_on and verdict in ("SUPERSEDE", "ENRICH"):
                    # One mechanism, N call sites: the same band_verdict the
                    # cycles use. Missing confidence lands mid defer band —
                    # abstain and adjudicate, never guess (D-019 rule).
                    try:
                        raw_confidence = classification.get("confidence")
                        confidence = (float(raw_confidence)
                                      if raw_confidence is not None else 0.6)
                    except (TypeError, ValueError):
                        confidence = 0.6
                    # None when consolidation is absent (Core): no banding.
                    banded = await edition.acall(
                        "consolidation_links", "band_verdict",
                        verdict, confidence, bands_cfg, reason=reason)
                    if banded is not None and banded.route == "defer":
                        defers.append(DeferAction(
                            case_type=banded.case_type or "supersession",
                            proposed_action=verdict,
                            trigger_memory_id=new_id_str,
                            candidate_memory_id=old_id,
                            confidence=confidence, reason=reason,
                            memory_type=old_memory_type))
                        continue
                    if banded is not None and banded.route == "none":
                        continue        # below the floor: both memories stay

                if verdict == "SUPERSEDE" and direction_guard_on:
                    # Which of the two is actually newer: ask the conversations,
                    # not the order we happened to ingest them in.
                    #
                    # Nothing below runs unless BOTH sides know their source
                    # conversation's date and the stored one is strictly later.
                    # Equal dates (the same document, 73.5% of pairs on the
                    # benchmark instance) and a missing date on either side both
                    # fall through to the behaviour above, because then we do not
                    # know the order and refusing to guess is the right answer.
                    _old_dct = _parse_document_date(old_document_created_at)
                    _new_dct = _parse_document_date(
                        getattr(new_memory, "document_created_at", None))
                    if _old_dct and _new_dct and _old_dct > _new_dct:
                        out_of_order_pairs += 1
                        # Ask again with the sides swapped. This is the whole
                        # fix: the model is handed the pair in the right time
                        # order, so "does the NEW side preserve everything the
                        # EXISTING side had" -- the information-loss procedure,
                        # the dropped-qualifier check, the subject-inversion
                        # rule -- all apply to the side that should be tested.
                        # Re-asking rather than simply flipping the ids matters
                        # because the first verdict only ever checked that the
                        # ARRIVING memory loses nothing, and here it is the
                        # arriving memory that would be retired.
                        swapped = await self._classify_pair(
                            existing_content=new_memory.content,   # said earlier
                            new_content=old_content,               # said later
                            scope=scope,
                            correlation_id=correlation_id,
                            new_assertion_source=old_assertion_source,
                            existing_assertion_source=getattr(
                                new_memory, "assertion_source", None),
                        )
                        swapped_verdict = (swapped or {}).get("classification", "")
                        swapped_reason = (swapped or {}).get("reason", "")
                        echo_blocks = (
                            echo_guard_on
                            and old_assertion_source == "assistant"
                            and getattr(new_memory, "assertion_source", None)
                            != "assistant"
                        )
                        if swapped_verdict == "SUPERSEDE" and not echo_blocks:
                            # The arriving memory is the older statement and the
                            # store already holds the later one. Retire the
                            # arriving row and point it at the survivor: the
                            # same database outcome a forward supersession has,
                            # with the two sides the right way round.
                            out_of_order_retired += 1
                            skips.append(SkipAction(
                                new_memory_id=new_id_str,
                                new_memory_content=new_memory.content,
                                existing_memory_id=old_id,
                                existing_memory_content=old_content,
                                reason=(
                                    "out-of-order ingest: this memory's "
                                    f"conversation ({_new_dct.date()}) is older "
                                    "than the stored memory's "
                                    f"({_old_dct.date()}), so the stored value "
                                    f"stays current. {swapped_reason}"
                                ),
                                memory_type=new_memory_type,
                                action_label="SUPERSEDE",
                            ))
                            new_id_outcomes[new_id_str] = "NONE"
                            logger.info(
                                "Knowledge-update out-of-order supersession "
                                "inverted: retired the ARRIVING memory new=%s "
                                "(said %s) in favour of stored old=%s (said %s)",
                                new_id_str, _new_dct.date(), old_id, _old_dct.date(),
                            )
                        else:
                            # Re-asked in the right order the model does not
                            # call it a clean replacement, so nothing is
                            # retired and both memories keep serving. The
                            # served line carries "(said ...)" on both, which
                            # is what lets a reader order them.
                            logger.info(
                                "Knowledge-update out-of-order supersession "
                                "BLOCKED: arriving new=%s (said %s) is older "
                                "than stored old=%s (said %s); swapped verdict "
                                "%s%s",
                                new_id_str, _new_dct.date(), old_id,
                                _old_dct.date(), swapped_verdict or "none",
                                " (echo guard)" if echo_blocks else "",
                            )
                        continue

                if verdict == "SUPERSEDE":
                    if (
                        echo_guard_on
                        and getattr(new_memory, "assertion_source", None) == "assistant"
                        and old_assertion_source != "assistant"
                    ):
                        # Echo guard: the agent's restatement may not retire
                        # what the user (or an unattributed legacy row)
                        # asserted — UNLESS both sides carry the same concrete
                        # value (then it's a plain duplicate and dedup is
                        # safe). Blocking with a NONE drop keeps the truth
                        # current AND keeps the echo out of retrieval, with
                        # the audit trail in memory_actions.
                        new_vals = _extract_value_tokens(new_memory.content)
                        old_vals = _extract_value_tokens(old_content)
                        same_value_dup = bool(new_vals) and new_vals == old_vals
                        if not same_value_dup:
                            skips.append(SkipAction(
                                new_memory_id=new_id_str,
                                new_memory_content=new_memory.content,
                                existing_memory_id=old_id,
                                existing_memory_content=old_content,
                                reason=(
                                    "echo-guard: assistant-asserted value may not "
                                    "supersede a user/unattributed memory "
                                    f"(classifier said SUPERSEDE: {reason})"
                                ),
                                memory_type=new_memory_type,
                            ))
                            new_id_outcomes[new_id_str] = "NONE"
                            logger.info(
                                "Knowledge-update echo guard: blocked assistant-sourced "
                                "supersession new=%s old=%s",
                                new_id_str, old_id,
                            )
                            continue
                    supersessions.append(SupersessionAction(
                        old_memory_id=old_id,
                        old_memory_content=old_content,
                        new_memory_id=new_id_str,
                        new_memory_content=new_memory.content,
                        reason=reason,
                        memory_type=old_memory_type,
                        new_memory_source_span=getattr(
                            new_memory, "source_span", None
                        ),
                    ))
                    seen_old_ids.add(old_id)
                    new_id_outcomes[new_id_str] = "SUPERSEDE"
                elif verdict == "NONE":
                    skips.append(SkipAction(
                        new_memory_id=new_id_str,
                        new_memory_content=new_memory.content,
                        existing_memory_id=old_id,
                        existing_memory_content=old_content,
                        reason=reason,
                        memory_type=new_memory_type,
                    ))
                    new_id_outcomes[new_id_str] = "NONE"
                elif verdict == "ENRICH":
                    enrichments.append(EnrichmentAction(
                        existing_memory_id=old_id,
                        existing_memory_content=old_content,
                        new_memory_id=new_id_str,
                        new_memory_content=new_memory.content,
                        reason=reason,
                        memory_type=old_memory_type,
                        new_memory_type=new_memory_type,
                    ))
                    # Don't mark new_id as taken — multiple existing candidates
                    # could each be an ENRICH partner; keep them all logged.
                elif verdict == "DELETE" and delete_enabled:
                    # Gap 6: explicit retraction. Soft-delete the target (the
                    # caller deprecates it + drops it from retrieval). Only
                    # possible when the flag is on — the classifier isn't even
                    # offered DELETE otherwise. Guard old_id like SUPERSEDE so
                    # one target isn't deleted twice in a batch.
                    if subject_guard_on:
                        # The retraction subject drives a coordinator-side
                        # sweep over ALL current matching rows — record it
                        # once per retraction memory, whether or not this
                        # particular pair survives the guard. Sweep only on
                        # strong signal (value token, or >=2 subject tokens)
                        # so a vague retraction can't mass-retire by ILIKE.
                        guard_parts = [
                            t for t in (
                                new_memory.content,
                                getattr(new_memory, "source_span", None),
                            ) if t
                        ]
                        guard_text = " ".join(guard_parts)
                        if new_id_str not in sweeps_by_new_id:
                            # v2: drop each part's conditional tail before
                            # tokenizing ("...kept on file AFTER the current
                            # transaction" -> "...kept on file"), per part so
                            # a qualifier in the extracted text can't swallow
                            # the raw source span behind it.
                            subject_source = (
                                " ".join(
                                    _truncate_conditional_tail(p)
                                    for p in guard_parts
                                )
                                if sweep_v2_on else guard_text
                            )
                            subj = _subject_tokens(subject_source)
                            vals = _extract_value_tokens(guard_text)
                            if vals or len(subj) >= 2:
                                sweeps_by_new_id[new_id_str] = RetractionSweep(
                                    new_memory_id=new_id_str,
                                    new_memory_content=new_memory.content,
                                    subject_tokens=tuple(sorted(subj)),
                                    value_tokens=tuple(sorted(vals)),
                                    reason=reason,
                                )
                        if not _delete_subject_match(guard_text, old_content):
                            # Mis-target: the vector-paired memory isn't about
                            # the retracted subject (retraction floor 0.45
                            # admits loose pairs). Don't delete it — the
                            # subject sweep retires the right rows instead.
                            logger.info(
                                "Knowledge-update delete guard: blocked "
                                "subject-mismatched DELETE new=%s old=%s "
                                "old_content=%r",
                                new_id_str, old_id, old_content[:80],
                            )
                            continue
                    deletions.append(DeletionAction(
                        old_memory_id=old_id,
                        old_memory_content=old_content,
                        new_memory_id=new_id_str,
                        new_memory_content=new_memory.content,
                        reason=reason,
                        memory_type=old_memory_type,
                    ))
                    seen_old_ids.add(old_id)
                    new_id_outcomes[new_id_str] = "DELETE"
                # UNRELATED (or DELETE with the flag off): no action.

            # v2: retraction memories whose pairs never reached a DELETE
            # verdict (classifier said NONE/UNRELATED, or every pair was for
            # a different new memory) still sweep on the lexicon signal —
            # same standalone emission as the no-candidates early return.
            if delete_enabled and sweep_v2_on:
                for m in memories:
                    mid = str(m.memory_id)
                    if mid in sweeps_by_new_id:
                        continue
                    sweep = _lexicon_retraction_sweep(m)
                    if sweep is not None:
                        sweeps_by_new_id[mid] = sweep

            elapsed = (time.time() - t0) * 1000
            set_current_span_attributes({
                "synap.ku_candidates": len(candidate_pairs),
                "synap.ku_supersessions": len(supersessions),
                "synap.ku_skips": len(skips),
                "synap.ku_enrichments": len(enrichments),
                "synap.ku_deletions": len(deletions),
                "synap.ku_elapsed_ms": round(elapsed, 1),
                "synap.ku_out_of_order_pairs": out_of_order_pairs,
                "synap.ku_out_of_order_retired": out_of_order_retired,
            })
            logger.info(
                "Knowledge-update detection: supersede=%d skip=%d enrich=%d "
                "delete=%d defer=%d out_of_order=%d of_which_retired=%d "
                "elapsed_ms=%.1f",
                len(supersessions), len(skips), len(enrichments), len(deletions),
                len(defers), out_of_order_pairs, out_of_order_retired, elapsed,
            )

            return KnowledgeUpdateResult(
                supersessions=supersessions,
                skips=skips,
                enrichments=enrichments,
                deletions=deletions,
                retraction_sweeps=list(sweeps_by_new_id.values()),
                defers=defers,
                # Every shortlisted pair is one the screen admitted, so it
                # also reached the classifier, and the two early returns above
                # cannot drop one.
                subject_candidates=subject_candidates,
            )

    async def _classify_pair(
        self,
        existing_content: str,
        new_content: str,
        scope: Scope,
        new_assertion_source: Optional[str] = None,
        existing_assertion_source: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Classify the relationship between an existing and new memory."""
        user_prompt = CLASSIFICATION_USER_PROMPT.format(
            existing_content=existing_content,
            new_content=new_content,
        )
        system_prompt = CLASSIFICATION_SYSTEM_PROMPT
        if _delete_enabled():
            system_prompt = system_prompt + _DELETE_ADDENDUM
        if _echo_guard_enabled() and (new_assertion_source or existing_assertion_source):
            # Provenance context (precision aid — the hard guarantee is the
            # code-level echo guard in the pair loop). Kept out of the base
            # prompt so the default classifier distribution is unchanged.
            system_prompt = system_prompt + _PROVENANCE_ADDENDUM
            user_prompt = user_prompt + provenance_note(
                existing_assertion_source, new_assertion_source)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self._llm.complete(
                messages=messages,
                telemetry_scope=scope,
                telemetry_operation="ingestion.knowledge_update.classify",
                telemetry_correlation_id=correlation_id,
                max_tokens=200,
                temperature=0.0,
                json_mode=True,
                **_ku_effort_kwargs(),
            )
            response_text = response if isinstance(response, str) else getattr(response, 'content', str(response))
            result = json.loads(response_text)
            classification = result.get("classification", "")
            valid_classifications = ("SUPERSEDE", "NONE", "UNRELATED", "ENRICH")
            if _delete_enabled():
                # DELETE is only a legal verdict when the flag (and therefore
                # the prompt addendum that defines it) is active. Without this,
                # a DELETE answer was rejected here as "invalid classification"
                # and the DELETE branch downstream could never be reached.
                valid_classifications = valid_classifications + ("DELETE",)
            if classification in valid_classifications:
                return result
            logger.warning(
                "Knowledge-update LLM response invalid classification '%s': %s",
                classification, response_text[:200],
            )
            return None
        except json.JSONDecodeError as e:
            logger.warning("Knowledge-update LLM response not valid JSON: %s", e)
            return None
        except LLMTimeoutError:
            # The model did not answer in time. Returning None here would save
            # the new memory with no supersession check; fail the stage instead
            # so the worker retries it (and dead-letters it after the last try).
            logger.warning("Knowledge-update classification timed out; failing the stage so it is retried")
            raise
        except Exception as e:
            logger.warning("Knowledge-update classification failed: %s", e)
            return None

    async def _classify_candidates(
        self,
        new_content: str,
        candidates: List[Tuple[Any, str]],
        scope: Scope,
        correlation_id: Optional[str] = None,
    ) -> Dict[Any, Dict]:
        """Classify ONE new memory against several candidates in one call.

        Returns {old_id: result}. A candidate missing from the return is NOT an
        error: the caller falls back to `_classify_pair` for it, so a malformed
        or partial response costs latency, never a lost verdict.
        """
        if not candidates:
            return {}
        system_prompt = CLASSIFICATION_SYSTEM_PROMPT
        if _delete_enabled():
            system_prompt = system_prompt + _DELETE_ADDENDUM
        system_prompt = system_prompt + BATCH_CLASSIFICATION_ADDENDUM

        user_prompt = build_batch_user_prompt(new_content, candidates)

        model = _ku_model_name()
        valid = ("SUPERSEDE", "NONE", "UNRELATED", "ENRICH")
        if _delete_enabled():
            valid = valid + ("DELETE",)
        try:
            response = await self._llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                telemetry_scope=scope,
                telemetry_operation="ingestion.knowledge_update.classify_batch",
                telemetry_correlation_id=correlation_id,
                max_tokens=_classify_budget(model, len(candidates)),
                temperature=0.0,
                json_mode=True,
                **({"reasoning_effort": _ku_reasoning_effort()}
                   if _ku_reasoning_effort() else {}),
            )
            text = response if isinstance(response, str) else getattr(
                response, "content", str(response))
            payload = json.loads(text)
            out: Dict[Any, Dict] = {}
            for entry in payload.get("results", []) or []:
                idx = entry.get("index")
                if not isinstance(idx, int) or not (1 <= idx <= len(candidates)):
                    continue
                if entry.get("classification") not in valid:
                    continue
                out[candidates[idx - 1][0]] = entry
            if len(out) != len(candidates):
                logger.warning(
                    "knowledge_update batch classify returned %d of %d verdicts "
                    "— the rest fall back to per-pair calls",
                    len(out), len(candidates),
                )
            return out
        except json.JSONDecodeError as e:
            logger.warning("knowledge_update batch classify not valid JSON: %s", e)
            return {}
        except LLMTimeoutError:
            # Same rule as _classify_pair. Not falling back to per-pair calls
            # either: each would wait out the same read limit again.
            logger.warning("knowledge_update batch classify timed out; failing the stage so it is retried")
            raise
        except Exception as e:
            logger.warning("knowledge_update batch classify failed: %s", e)
            return {}

    async def synthesize_merges(
        self,
        enrichments: List[EnrichmentAction],
        scope: Scope,
        max_cluster_size: int = 5,
        correlation_id: Optional[str] = None,
    ) -> List[MergeAction]:
        """Turn ENRICH-classified pairs into merged memories.

        Algorithm:
        1. Build a graph from the ENRICH pairs (each pair = an undirected edge
           between memory_ids).
        2. Find connected components — each component is a cluster of memories
           that should be merged into one.
        3. If a cluster exceeds `max_cluster_size`, take the first N members
           (the ones the classifier was most confident about, since they
           appeared in the result list earlier). Remaining members stay
           unmerged and re-trigger ENRICH on the next ingestion.
        4. Per cluster, call the merge LLM with all contributors' content;
           apply guardrails (length, named entities); emit one MergeAction.

        Returns: list of MergeAction (may be shorter than enrichment count
        because of clustering).
        """
        if not enrichments:
            return []

        from synap.cloud.agents.ingestion.prompts.merge import (
            MERGE_SYSTEM_PROMPT,
            build_merge_user_prompt,
        )

        async with start_span("ingestion.knowledge_update.synthesize_merges", {
            "synap.enrichment_pair_count": len(enrichments),
        }):
            # 1+2. Union-find clustering.
            parent: Dict[str, str] = {}

            def find(x: str) -> str:
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x

            def union(a: str, b: str) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            # Index enrichment metadata by memory_id for later lookup.
            content_by_id: Dict[str, str] = {}
            type_by_id: Dict[str, Optional[str]] = {}
            for e in enrichments:
                content_by_id[e.existing_memory_id] = e.existing_memory_content
                content_by_id[e.new_memory_id] = e.new_memory_content
                # Track types so the cluster guardrail can refuse to merge
                # APPEND_ONLY memories (episodes, emotions). The
                # type-aware dispatch is the primary line of defense; this
                # is the secondary check, hot when the dispatch flag is off.
                type_by_id[e.existing_memory_id] = e.memory_type
                type_by_id[e.new_memory_id] = e.new_memory_type
                parent.setdefault(e.existing_memory_id, e.existing_memory_id)
                parent.setdefault(e.new_memory_id, e.new_memory_id)
                union(e.existing_memory_id, e.new_memory_id)

            # Collect cluster members keyed by root.
            clusters: Dict[str, List[str]] = {}
            for mid in list(parent.keys()):
                clusters.setdefault(find(mid), []).append(mid)

            logger.info(
                "Knowledge-update merge synthesis: %d enrichment pairs -> %d clusters",
                len(enrichments), len(clusters),
            )

            # 3. Cap cluster size + 4. call merge LLM per cluster.
            append_only_type_values = {t.value for t in APPEND_ONLY_TYPES}
            merges: List[MergeAction] = []
            for root, member_ids in clusters.items():
                if len(member_ids) < 2:
                    continue  # not really a cluster

                # Guardrail: refuse to merge if any member is an APPEND_ONLY
                # type. Episodes and emotions are bound to their moment and
                # collapsing them into a single "perpetual" statement loses
                # the per-occurrence detail that's the whole point of the
                # type. The contributors stay alive — KU just declines the merge.
                cluster_types = [type_by_id.get(mid) for mid in member_ids]
                if any(t in append_only_type_values for t in cluster_types if t):
                    logger.info(
                        "Merge cluster %s skipped: contains APPEND_ONLY type "
                        "(%s) — episodes/emotions don't merge",
                        root[:8],
                        sorted({t for t in cluster_types if t in append_only_type_values}),
                    )
                    continue

                if len(member_ids) > max_cluster_size:
                    logger.info(
                        "Merge cluster %s capped: %d members -> taking first %d, rest stay unmerged",
                        root[:8], len(member_ids), max_cluster_size,
                    )
                    member_ids = member_ids[:max_cluster_size]

                contents = [content_by_id.get(mid, "") for mid in member_ids]
                merge_result = await self._call_merge_llm(
                    contents, scope=scope, correlation_id=correlation_id,
                )
                if merge_result is None:
                    logger.warning(
                        "Merge LLM returned no result for cluster of %d members; skipping merge",
                        len(member_ids),
                    )
                    continue

                merged_content = (merge_result.get("merged_content") or "").strip()
                rationale = merge_result.get("rationale", "")

                # Guardrail: merged content must be at least as long as the
                # longest contributor (otherwise the LLM probably summarized
                # away detail rather than synthesizing).
                longest = max((len(c) for c in contents), default=0)
                if not merged_content or len(merged_content) < longest:
                    logger.warning(
                        "Merge guardrail tripped for cluster %s: merged_len=%d longest_contributor=%d — skipping merge",
                        root[:8], len(merged_content), longest,
                    )
                    continue

                # Guardrail: every named entity, quoted phrase, and numeric
                # anchor from every contributor must survive verbatim. A
                # paraphrased-away anchor ("Charlotte's Web" → "the classic")
                # is silent data loss; the cluster stays unmerged instead.
                dropped_entities = find_dropped_entities(contents, merged_content)
                if dropped_entities:
                    logger.warning(
                        "Merge guardrail tripped for cluster %s: entities dropped %s — skipping merge",
                        root[:8], sorted(dropped_entities),
                    )
                    continue

                merges.append(MergeAction(
                    contributor_memory_ids=member_ids,
                    contributor_contents=contents,
                    contributor_source_documents=[],  # caller fills from DB lookup
                    merged_content=merged_content,
                    rationale=rationale,
                ))

            set_current_span_attributes({
                "synap.merge_clusters": len(clusters),
                "synap.merge_actions": len(merges),
            })
            return merges

    async def _call_merge_llm(
        self,
        contents: List[str],
        scope: Scope,
        correlation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Call the merge prompt for one cluster. Returns parsed JSON or None on failure."""
        from synap.cloud.agents.ingestion.prompts.merge import (
            MERGE_SYSTEM_PROMPT,
            build_merge_user_prompt,
        )
        messages = [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_merge_user_prompt(contents)},
        ]
        try:
            response = await self._llm.complete(
                messages=messages,
                telemetry_scope=scope,
                telemetry_operation="ingestion.knowledge_update.merge_synthesis",
                telemetry_correlation_id=correlation_id,
                max_tokens=400,  # merged content + rationale; cluster of 5 fits easily
                **_ku_effort_kwargs(),
                temperature=0.0,
                json_mode=True,
            )
            text = response if isinstance(response, str) else getattr(response, "content", str(response))
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Merge LLM response not valid JSON: %s", e)
            return None
        except Exception as e:
            logger.warning("Merge LLM call failed: %s", e)
            return None
