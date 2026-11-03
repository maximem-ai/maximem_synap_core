"""SDK context fetch routes.

Provides endpoints for retrieving memory context at different scopes:
- conversation, user, customer, client.

Uses RetrievalManager for hybrid retrieval with caching, compaction integration,
and 4-check filtering.
"""

import asyncio
import json as _json
import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from synap.cloud.application.sdk_api.sdk_auth import (
    SDKAuthContext,
    get_sdk_auth,
)
from synap.cloud.application.sdk_api.readiness import require_instance_ready
from synap.cloud.application.sdk_api.retrieval_rate_limit import enforce_retrieval_rate_limit
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from synap.cloud.shared.scope import (
    collapse_b2c_identity,
    parse_or_deterministic_uuid as _parse_or_deterministic_uuid,
    build_scope_with_maca,
    ScopeHierarchy,
    ScopePathRejected,
)
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, _failure_code
from synap.cloud.application.profile.store import ProfileStore
from synap.cloud.shared import edition
from synap.cloud.application.shared.providers import get_cached_providers
from synap.cloud.agents.models.decisions import (
    ContextDecision,
    RetrievalMode,
    TriggerType,
)
from synap.cloud.shared.step_logger import StepLogger

def get_usage_tracker():
    """Cloud's usage tracker (the ``usage_tracking`` hook); None in Core, which has none."""
    return edition.call("usage_tracking", "tracker")


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/context", tags=["SDK Context"])

# ── Token budget, derived from the caller's top-K ────────────────────────────
#
# `max_results` and a fixed token budget are two knobs expressing the same
# intent, and when they disagree the budget wins SILENTLY. Until this change the
# budget was hardcoded to 2000 at all four fetch routes, which is why asking for
# more than about nine items did nothing: `_apply_token_budget` trimmed the
# bundle back and only said so in a log line.
#
# Sizes are measured, not guessed — the app's own token counter over a sample of real
# response items:
#
#     item without source_evidence   mean 222 tokens   (p90 280)
#     item with source_evidence      mean 693 tokens   (p90 771)
#     source_evidence alone          mean 473 tokens
#
# So evidence is worth more than twice everything else combined, and the budget
# has to know whether the caller asked for it. The per-item figures below round
# the measured means up toward p90 so a typical top-K is not trimmed by noise.
#
# FLOOR: keep the historical 2000 so small requests behave exactly as before.
# CEILING: a derivation with no cap lets `max_results=10000` ask for seven
# million tokens. The cap is generous (top-100 with evidence fits) but finite.
TOKENS_PER_ITEM_WITH_EVIDENCE = 750
TOKENS_PER_ITEM_BARE = 280
TOKEN_BUDGET_FLOOR = 2000
TOKEN_BUDGET_CEILING = 120_000


def derive_token_budget(max_results: int, include_source_evidence: bool) -> int:
    """Budget large enough to actually return `max_results` items.

    The caller says how many items they want; the budget is an implementation
    detail sized to deliver that, not a second control that can contradict it.

    ⚠ This changes what EVERY client receives, not just the one that asked for
    it. A default call (max_results=10, evidence on) moves from 2000 to 7500
    tokens, so a client who was quietly getting ~3 items now gets the 10 they
    asked for — correct, but roughly 3x the response bytes and 3x the tokens in
    whatever prompt they feed it into. SYNAP_DERIVE_TOKEN_BUDGET=false restores
    the old fixed 2000 instantly, without a deploy, if that lands badly for
    someone.
    """
    if os.environ.get("SYNAP_DERIVE_TOKEN_BUDGET", "true").lower() in ("false", "0", "no"):
        return TOKEN_BUDGET_FLOOR
    try:
        k = int(max_results or 0)
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        return TOKEN_BUDGET_FLOOR
    per_item = (
        TOKENS_PER_ITEM_WITH_EVIDENCE if include_source_evidence
        else TOKENS_PER_ITEM_BARE
    )
    return max(TOKEN_BUDGET_FLOOR, min(k * per_item, TOKEN_BUDGET_CEILING))




# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ContextFetchRequest(BaseModel):
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    search_query: Optional[List[str]] = None
    max_results: int = 10
    # How many items each search query may contribute when ``search_query``
    # holds more than one. ``max_results`` stays the ceiling on the whole
    # response: three queries at five each under max_results=10 return ten, not
    # fifteen. With one query, or left out, the response is the best
    # ``max_results`` items overall, as it always was. A value below 1 is
    # treated as left out rather than refused.
    #
    # Added 2026-10-08. An older server ignores the field.
    max_results_per_query: Optional[int] = None
    types: List[str] = ["all"]
    mode: str = "fast"
    # SDK-authoritative ST: SDKs with a warm
    # ShortTermContextStore for this conversation pass False to skip the
    # server-side ST assembly and merge their own cached block client-side.
    # Defaults True for backwards compatibility — older SDKs and any
    # non-SDK callers keep getting ST in the response.
    include_conversation_context: bool = True
    # Metadata-keyed lookups (e.g. LangGraph checkpoint reads keyed by
    # thread_id) don't need semantic query expansion or LLM relevance
    # filtering — the caller selects the row by exact metadata afterward.
    # These let such callers skip the two Groq round-trips (sub-query
    # generation + relevance filter) that dominate accurate/fast latency.
    # Both default False → zero behaviour change for existing callers.
    skip_subquery_gen: bool = False
    skip_llm_filter: bool = False
    # When the caller knows the moment the question is being asked, it belongs
    # here rather than smuggled into `search_query`. The benchmark harness
    # proved the smuggled form hurts: prefixing "[asked on <date>]" onto the
    # query text scored 85.53% against 87.23% for the same run without it.
    #
    # Absent means "now", so every existing caller is byte-unchanged. Anything
    # unparseable is ignored with a warning rather than refused: this is an
    # enhancement, and a read that 400s over a cosmetic field is worse than a
    # read that falls back to today.
    query_date: Optional[str] = None
    # `source_evidence` carries the verbatim source text behind each memory.
    # Measured over a sample of real responses: it is about **41.5% of the
    # response body**, roughly 400 bytes per item, present on nearly
    # every item. A caller that pipes our response straight into an LLM
    # pays for all of it in every prompt.
    #
    # It is NOT a latency knob. After the chunk-evidence batching fix the whole
    # enrichment is p50 under 20ms over a sample of requests, under
    # 1% of a fetch. Skipping it saves bytes and tokens, not time.
    #
    # Defaults True, so nothing changes for existing callers.
    include_source_evidence: bool = True
    # G-03. Where this conversation is happening, in the client's own level
    # names, e.g. {"customer": "acme", "team": "payments", "user": "d-42"}.
    #
    # Optional, and None is every request today: callers send ids and the
    # legacy adapter maps them onto the default three rungs. Supplying a path
    # is the only way to address a ladder deeper than that, which is the whole
    # point of the feature and was previously unreachable from any transport.
    #
    # Strict when present. An unknown level, a path that skips a rung, or the
    # feature being off for this instance is a 400, never a quiet fall back to
    # the legacy chain: falling back would serve a WIDER scope than was asked
    # for, and P-01 exists because a wrong inference at or above the tenant of
    # record is a cross-tenant leak.
    scope: Optional[Dict[str, str]] = None
    # Public speed/precision knob. "high" (default) keeps the server-side
    # relevance-refinement pass; "medium" skips it for faster responses —
    # the same candidate memories are searched, results are just less
    # precisely filtered. Equivalent to skip_llm_filter=True but validated
    # and documented; skip_llm_filter stays as the internal escape hatch.
    # Unlike `mode`, bad values are rejected (422) instead of silently
    # coerced, so typos can't flip retrieval behavior.
    precision_level: Literal["high", "medium"] = "high"
    # C4 — conversation-summary fetch (spec §6.1). "in-conversation" (default)
    # is the existing retrieval path; "conversation-summary" is a route-level
    # branch (fetch_user_context only) that returns the caller profile + the
    # last N conversation summaries and NEVER enters RetrievalManager (§6.2).
    # Bad values are rejected at parsing (422) — same posture as precision_level.
    context_mode: Literal["in-conversation", "conversation-summary"] = "in-conversation"
    # conversation-summary mode only: whether to include the caller profile
    # document (§5.2) alongside the conversation summaries.
    include_profile: bool = True
    # conversation-summary mode only: how many most-recent conversations to
    # summarise (0 is valid — with include_profile=false it is a well-formed
    # empty response, spec §6.1). Bounds enforced at parsing (422 out of range).
    last_n_conversations: int = Field(1, ge=0, le=20)
    # How the items come back.
    #
    # "by_type" (default) is the response every published SDK reads: five lists,
    # one per memory type. Grouping by type loses the order the engine ranked
    # across types, so a caller who wants the best items first has to merge the
    # lists and sort on `metadata.rank_score` itself.
    #
    # "ranked" returns that order directly: ONE list, `context.items`, best
    # first, each item carrying its type (`category`), its place (`rank`, from
    # 1) and `metadata.rank_score`. Chunks, the quotes from source documents,
    # are placed by their own score there and typed "excerpts"; in "by_type"
    # they stay "facts" at their fixed place, as before. The five lists are not
    # sent, so nothing is sent twice.
    #
    # Opt-in so that no published SDK changes underneath its users: an SDK that
    # predates this never sends it and gets byte-for-byte the response it always
    # did. The default moves to "ranked" later, with notice, once callers have
    # moved. Bad values are rejected (422), like `precision_level`.
    response_shape: Literal["by_type", "ranked"] = "by_type"
    # NOTE (spec §6.1): in conversation-summary mode `search_query`, `mode`,
    # `precision_level` and `response_shape` are IGNORED — no retrieval
    # decision is built.


class ContextItem(BaseModel):
    id: str
    content: str
    confidence: float = 0.0
    source: str = ""
    category: str = ""
    metadata: Dict[str, Any] = {}
    # Temporal fields
    #
    # Two dates and no more, because two is what a reading agent can use: when
    # the thing happened, and when the user said it. `event_date` is now empty
    # unless somebody stated one, so `said_at` has to be here or an item with no
    # stated event date arrives with no machine-readable date at all. It did for
    # 2,017 of 10,443 served items on the 500-question run (19.3%): the day they
    # were said reached the reader only inside the content string, as "(said
    # ...)", where a client's code cannot get at it.
    #
    # There used to be an `extracted_at` here, carrying the row's `created_at`.
    # It is the minute OUR pipeline ran, and it is gone for the simple reason
    # that it is of no use outside Synap: no client agent can do anything with
    # the time we happened to process a document. It was also not reliably the
    # same day as `said_at`, which is how it got defended before: a transcript
    # ingested with no dates in it served an `extracted_at` of the ingest day
    # on every item for a conversation from 2023. The field is still in the published SDKs,
    # where it now simply
    # parses to None, so removing it here needs no SDK release.
    said_at: Optional[str] = None
    event_date: Optional[str] = None
    # The rest of the event window. `event_date` is its START, and without
    # these two a window is served as though it were a single day: "last month"
    # is stored correctly as 2023-04-01 to 2023-04-30 with precision "month",
    # and a client was handed only 2023-04-01, which reads as the 1st. Any
    # arithmetic from it is then wrong by up to the width of the window.
    #
    # Both are already computed by retrieval_manager._event_fields and were
    # being dropped here. 102 of the 221 dated rows in one ingest carry a real
    # end: whole months ("married in December 2022"), ranges ("the festival
    # June 16-19"), weekends, weeks, seasons and years ("The Tipping Point
    # (2000)").
    #
    # `event_precision` matters on its own, because "approximate" is stored as
    # a bare point with no end: "subscribed since around November 20, 2022" and
    # "subscribed on November 20, 2022" are identical in the payload, and the
    # only thing that tells them apart is this label.
    event_end: Optional[str] = None
    event_precision: Optional[str] = None
    valid_until: Optional[str] = None
    temporal_category: Optional[str] = None
    temporal_confidence: float = 0.0
    # Source tracking
    source_document_id: Optional[str] = None
    source_evidence: Optional[List[str]] = None
    # Which of the request's ``search_query`` strings found this item, in the
    # order they were sent. A fetch returns one set for all of its queries, so
    # this is how a caller tells them apart. A result of a subquery or template
    # the server wrote names the query it was written from. Empty when the
    # request named no query, and for an item no single query accounts for.
    #
    # Added 2026-10-08. Additive: a client that does not read it sees the
    # response it always did.
    matched_queries: List[str] = []


class RankedContextItem(ContextItem):
    """An item in the ranked shape: the same fields, plus its place.

    A subclass rather than a field on ContextItem, because a field there would
    print `"rank": null` on every item of every default-shape response, and the
    default shape is promised byte-identical to what published SDKs read.
    """
    # 1 is the best item. Consecutive, no gaps.
    rank: int


# REVIEW BY 2027-04-08 (internal; six months from 2026-10-08).
#
# This payload is one set for the whole fetch, grouped by memory type. A fetch
# can now carry up to five search queries, give each a share of the response
# (max_results_per_query), and is charged per query, so "one flat set" is the
# part of the contract that no longer matches how the request is shaped. For
# now each item says which queries found it (ContextItem.matched_queries) and
# the caller groups them. That was chosen because it is purely additive: no
# published SDK or raw REST caller breaks.
#
# The candidate successor is a response grouped by query. Before building it,
# check what the six months showed:
#   - do callers actually group by matched_queries, or ignore it;
#   - how many fetches carry more than one query, and how many set a share;
#   - whether items with an empty matched_queries on a multi-query fetch (a
#     Neo4j record with no embedding of its own) are common enough to matter.
# If grouped-by-query wins, the flat shape is what gets deprecated, with both
# served side by side for a release first. Nothing here is deprecated today.
class ContextPayload(BaseModel):
    facts: List[ContextItem] = []
    preferences: List[ContextItem] = []
    episodes: List[ContextItem] = []
    emotions: List[ContextItem] = []
    temporal_events: List[ContextItem] = []
    # Date range across all items — helps LLM calibrate temporal references
    earliest_event_date: Optional[str] = None
    latest_event_date: Optional[str] = None


# The type of a chunk in the ranked shape. A chunk is a quote from a source
# document, not an extracted memory, and the default shape files it under
# "facts" only because it had to go in one of the five lists.
EXCERPT_CATEGORY = "excerpts"


class RankedContextPayload(BaseModel):
    """`response_shape="ranked"`: one list, best first, nothing sent twice.

    `extra="forbid"` and the required `items` are what keep the two shapes apart
    when the response is validated against `ContextFetchResponse`: a by-type
    payload has neither, so it can never be read as this one.
    """
    model_config = {"extra": "forbid"}

    items: List[RankedContextItem]
    earliest_event_date: Optional[str] = None
    latest_event_date: Optional[str] = None

    # Read-only views for this module's own logging, which counts per type.
    # Properties are not serialised, so they add nothing to the response.
    def _of(self, category: str) -> List[RankedContextItem]:
        return [it for it in self.items if it.category == category]

    @property
    def facts(self) -> List[RankedContextItem]:
        # Excerpts count as facts here, as they are filed in the by-type shape,
        # so the per-type counts the routes log (and the customer route's
        # "nothing came back" check) agree between the two shapes.
        return [it for it in self.items if it.category in ("facts", EXCERPT_CATEGORY)]

    @property
    def preferences(self) -> List[RankedContextItem]:
        return self._of("preferences")

    @property
    def episodes(self) -> List[RankedContextItem]:
        return self._of("episodes")

    @property
    def emotions(self) -> List[RankedContextItem]:
        return self._of("emotions")

    @property
    def temporal_events(self) -> List[RankedContextItem]:
        return self._of("temporal_events")


class ConversationContextPayload(BaseModel):
    """Current session context from compaction + raw buffer."""
    summary: Optional[str] = None
    current_state: Dict[str, Any] = {}
    key_extractions: Dict[str, List[Dict[str, Any]]] = {}
    recent_turns: List[Dict[str, Any]] = []
    compaction_id: Optional[str] = None
    compacted_at: Optional[str] = None
    conversation_id: Optional[str] = None


class ConversationSummaryPayload(BaseModel):
    """One returned conversation summary in conversation-summary mode (spec §6.3).

    ``summary``/``classification`` are pulled from the latest ``compacted_contexts``
    row; ``analysis`` is the client's own per-call JSON echoed back verbatim.
    ``summary_status`` is the read-time view of the three-vocabulary mapping
    (spec §4.1): ``available`` (a compaction row exists), ``pending`` (none yet),
    ``failed`` (a ``compaction_failed:{cid}`` Redis marker is set).
    """
    conversation_id: str
    external_conversation_id: Optional[str] = None
    conversation_type: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    summary_status: Literal["available", "pending", "failed"]
    summary: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    compaction_version: Optional[int] = None
    compacted_at: Optional[datetime] = None


class RetrievalHealth(BaseModel):
    """Why this answer may be thinner than the stores can support.

    Present ONLY when a leg of the retrieval failed, or when the request held
    more search queries than one fetch runs (``MAX_SEARCH_QUERIES``; that case
    alone leaves ``degraded`` false). A healthy fetch leaves the field null, so
    the response shape is unchanged for every existing caller, the same
    convention ``scope_note`` follows.

    It exists because the opposite was true and cost us a benchmark: on a
    LongMemEval run, a few retrievals lost a vector search arm to
    a store error and still returned 200 with a smaller context, and the harness
    scored every one of them as a clean success. A client cannot retry, alert or
    discount a failure it cannot see.
    """

    degraded: bool = True
    # "partial" (some stores answered), "minimal" (cache only) or "failed".
    level: Optional[str] = None
    # WHY, for code rather than for a human: "timeout", "overloaded",
    # "unavailable" or "internal". `level` says how thin the answer is, `code`
    # says whether retrying is worth it. Never names a service of ours.
    code: Optional[str] = None
    # One line per leg that failed, already de-duplicated.
    warnings: List[str] = []


class ContextFetchResponse(BaseModel):
    # One of the two shapes, chosen by the request's `response_shape`.
    # left_to_right: the ranked shape is tried first and refuses anything that
    # is not one (see RankedContextPayload), so a by-type payload always lands
    # on ContextPayload and serialises exactly as it did before the union.
    context: Union[RankedContextPayload, ContextPayload] = Field(
        default_factory=ContextPayload, union_mode="left_to_right")
    conversation_context: Optional[ConversationContextPayload] = None
    ttl_seconds: int = 300
    # Null unless a leg of the retrieval failed. See RetrievalHealth.
    retrieval: Optional[RetrievalHealth] = None
    # C4 — populated only in conversation-summary mode (spec §6.3). ``profile``
    # is the §5.2 document (or null); ``conversations`` is the list of last-N
    # summaries (possibly empty). Both stay None in in-conversation mode so the
    # existing response shape is byte-identical for old callers.
    profile: Optional[Dict[str, Any]] = None
    conversations: Optional[List["ConversationSummaryPayload"]] = None
    # Set only by fetch_customer_context, only when the fetch came back empty
    # AND the reason is structural rather than "no data yet": see
    # `customer_scope_is_shared_material_only`. None for every other route and
    # every other outcome of this one, so the response shape is unchanged for
    # every existing caller who doesn't read this field.
    scope_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# How many search queries one fetch runs. Every query is an embedding plus a
# search of each collection on the scope chain, a hypothetical-question search
# and a short-term search, and there was no bound: one request could fan out as
# wide as its caller cared to write it. Five covers a planner that splits one
# turn into a handful of searches. A caller with more sends a second fetch.
#
# Applied on the public fetch routes only. The anticipation agent builds its
# own decision and has its own cap (anticipation_agent/tools.py).
MAX_SEARCH_QUERIES = 5


def _per_query_share(raw) -> Optional[int]:
    """``max_results_per_query`` as retrieval reads it: a positive int, or None."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _cap_search_queries(decision: ContextDecision) -> Optional[str]:
    """Trim ``decision.search_queries`` to the cap. Returns what to tell the caller.

    The queries past the cap are not searched, and the response says so: this
    route spent eight months dropping a caller's second query in ACCURATE mode
    without a word (see ``accurate_searches_all_queries``). None when nothing
    was trimmed, so the response is unchanged for every request under the cap.
    """
    queries = list(decision.search_queries or [])
    if len(queries) <= MAX_SEARCH_QUERIES:
        return None
    decision.search_queries = queries[:MAX_SEARCH_QUERIES]
    return (
        f"search_query held {len(queries)} queries; only the first "
        f"{MAX_SEARCH_QUERIES} were searched. Send the rest in another fetch."
    )


_UNSET = object()  # Sentinel to distinguish "not passed" from "explicitly None"


async def _build_scope(
    auth: SDKAuthContext,
    user_id: Optional[str] = None,
    customer_id=_UNSET,
    maca_scoping: Optional[dict] = None,
    scope_path: Optional[Dict[str, str]] = None,
) -> ScopeHierarchy:
    """Build ScopeHierarchy using MACA-aware scope resolution.

    - B2C (user_context_isolation == "equals_customer"): customer_id = user_id
    - When customer_id is explicitly None (client-level fetch), no customer scoping
    - Uses IdMapper for persistent mapping when available, falls back to
      parse_or_deterministic_uuid via build_scope_with_maca.
    """
    raw_customer = None if customer_id is _UNSET else customer_id

    # Resolve isolation ONCE, here, and hand the same answer to both branches.
    # The fallback below is synchronous and so cannot ask; left to itself it
    # defaulted to "strict", which turns an IdMapper blip on a B2C instance
    # into a B2B rejection of a correctly-formed request. Filling it here also
    # means `resolve_identity` finds the key present and does not look again.
    from synap.cloud.shared.scope import effective_user_context_isolation
    maca_scoping = dict(maca_scoping or {})
    maca_scoping["user_context_isolation"] = await effective_user_context_isolation(
        auth.instance_id, maca_scoping,
    )

    try:
        from synap.cloud.shared.id_mapper import get_id_mapper
        from synap.cloud.shared.scope import build_scope_with_maca_mapped
        mapper = get_id_mapper()
        return await build_scope_with_maca_mapped(
            mapper=mapper,
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=user_id,
            customer_id=raw_customer,
            maca_scoping=maca_scoping,
            scope_path=scope_path,
        )
    except ScopePathRejected:
        # NEVER falls back. The fallback below builds a scope that ignores
        # `scope_path` entirely, so swallowing this would answer a caller who
        # asked to be narrowed to one rung with the whole tenant instead. That
        # is the precise failure the strict contract exists to prevent, and the
        # bare handler below would have caused it. Let it reach the route and
        # become a 400.
        raise
    except Exception as exc:  # noqa: BLE001
        # This fallback is not equivalent to the path it replaces, and it was
        # silent. Both builders derive the same uuids, so the response looks
        # identical and the data stays readable, but only the mapped one goes
        # through IdMapper, which is what creates the `customer` and `user`
        # rows. Falling back therefore leaves a scope that owns data and has no
        # control-plane row, which can leave many scopes owning memory rows
        # (SHIP-22).
        #
        # Logging it is the difference between a measurable rate and a mystery:
        # until now nothing recorded that the degradation had happened at all.
        #
        # With a `scope_path` there is no fallback. `build_scope_with_maca`
        # takes no path, so falling back read the default levels for a caller
        # who named a narrower one: the widening ScopePathRejected refuses
        # above, reached by a different exception. A ValueError is the
        # caller's input (a B2B user with no customer) and stays a 400; any
        # other failure is ours, so it is a 503 the SDKs retry.
        if scope_path:
            if isinstance(exc, ValueError):
                raise
            logger.warning(
                "scope path not honoured, refusing instead of widening (%s: %s)",
                type(exc).__name__, exc,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "The scope path could not be resolved right now, so the "
                    "request was refused rather than served at a wider scope. "
                    "Retry it."
                ),
            ) from exc
        logger.warning(
            "scope built WITHOUT id mapping (%s: %s); "
            "customer/user control-plane rows will not be created for this request",
            type(exc).__name__, exc,
        )
        return build_scope_with_maca(
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=user_id,
            customer_id=raw_customer,
            maca_scoping=maca_scoping,
        )


async def _b2c_check(auth: SDKAuthContext, customer_id, *, where: str, user_id=None) -> None:
    """Apply the B2C single-identifier contract to an inbound request.

    Thin on purpose: the policy, the MACA read and the message all live in
    `shared.b2c_contract`, so every route asks the same question and a change of
    rollout state touches one file.
    """
    from synap.cloud.shared.b2c_contract import check_customer_id
    await check_customer_id(auth.instance_id, customer_id, where=where, user_id=user_id)


async def _b2c_reject_customer_route(auth: SDKAuthContext, *, where: str) -> None:
    from synap.cloud.shared.b2c_contract import reject_customer_route
    await reject_customer_route(auth.instance_id, where=where)


def customer_scope_reaches_an_identity(chain_rungs) -> Optional[str]:
    """Why a customer-scope read must not run, or None when it may.

    Module level and public so the rule is testable by calling it. Inline in
    the route it could only be tested by a copy of itself, and a copy passes
    while the original is broken.

    A customer-scope read is a read at the tenant of record: shared material,
    never some individual's personal memories. `_build_customer_scope_filter`
    has always said exactly that. The nested arm can break it in two ways, and
    both are the same shape: the resolved chain contains a node that holds the
    `identity` role, and `node_id IN chain` then matches every record that
    person ever wrote.

    ⚠ The live one was B2C. There the tenant node IS the identity node, so a
    chain resolved from nothing but a `customer_id` already contains a person.
    `/v1/context/customer/fetch` with
    `customer_id=<a person's id>` returned that person's private memories.
    Before nested scoping the same route compiled only the legacy arm, which
    asks for `user_id == '__CUSTOMER_SCOPE__'`, a value B2C never writes, so it
    returned nothing at all. `b2c_contract.reject_customer_route` cites that
    "returns nothing" property as the reason its `off` and `warn` rollout
    states are safe to pass through, and `warn` is the default. The property is
    gone, so the guard cannot depend on the rollout state any more.

    The second way is a caller sending a `scope` path that names the identity
    rung. That is a well-formed path and a perfectly good `/user/fetch`; it is
    not a customer-scope read, and answering it here would return one person
    under a tenant-shaped question.

    Reads the roles a rung actually holds rather than its single `role`,
    because in B2C the tenant rung holds two and looking at only the first
    concludes the request has no identity.
    """
    for rung in chain_rungs or ():
        roles = rung.get("roles") or ([rung.get("role")] if rung.get("role") else [])
        if "identity" not in roles:
            continue
        if "tenant_of_record" in roles:
            return (
                "This instance has user_context_isolation=equals_customer, so the "
                "customer and the person are one scope and there is no customer "
                "level to read at. Fetch at user scope instead: "
                "POST /v1/context/user/fetch with user_id."
            )
        return (
            "The scope path names the identity rung, which addresses one person "
            "rather than the tenant they belong to. A customer-scope fetch reads "
            "shared material only. Send this path to POST /v1/context/user/fetch, "
            "or stop the path above the identity rung."
        )
    return None


def customer_scope_is_shared_material_only(
    *, user_id: Optional[str], allow_cross_user_retrieval: bool,
) -> bool:
    """True when this customer-scope fetch can only ever return tenant-wide
    shared material, never an individual user's own memories.

    `_build_customer_scope_filter` (mirrored in
    `scope_filter.legacy_scope_predicate`) has always narrowed a request with
    no `user_id` to `user_id == CUSTOMER_SCOPE_SENTINEL` — material stored with
    no owner — unless `allow_cross_user_retrieval` is on. That flag is False by
    default for EVERY relationship shape (`DEFAULT_INSTANCE_TYPE_CONFIG`: b2c,
    b2b, internal, agent_to_agent all ship `allow_cross_user_within_customer:
    False`), and turning it on is a deliberate, per-instance privacy decision,
    never something this route can decide for a caller. So "aggregate every
    user under this customer" — the outcome the route's own docstring used to
    promise — is not what a bare `customer_id` fetch does today, and never
    silently was.

    This is not "the customer scope is empty right now": a caller who genuinely
    stores shared, ownerless knowledge (an FAQ, a company-wide fact) gets real
    results here, as a client using exactly that shape does.
    It is "this fetch cannot have returned anything other than that narrow
    shared bucket", which is the fact a caller needs to debug a permanent 200
    empty, and which was previously nowhere in the response, a log line, or a
    docstring that agreed with the code.

    A client called this route thousands of times and every call came back
    empty. Every one of that instance's stored memories belongs to a specific
    end-user; none were ever written as shared, ownerless material.
    `allow_cross_user_retrieval` was False the whole time. Neither the
    B2C-identity-collapse guard above nor an id-derivation defect explains
    this: both were checked directly against that client's requests and
    ruled out.
    """
    return not user_id and not allow_cross_user_retrieval


_DCT_RANGE_THRESHOLD_DAYS = 30  # only show "(Jan – Jul 2023)" if range > this


def _format_day_precise(d, weekday: bool = False) -> str:
    """Render a single date at the precision we actually know it to.

    Day-of-month 1 is the MONTH-PRECISION SENTINEL produced by temporal
    resolution: when a source only said "in July 2023" the resolver anchors the
    date to the 1st, so a rendered "Jul 1, 2023" would be a fabricated day. Any
    other day-of-month came from a real day-level mention and is rendered in
    full ("Jul 14, 2026") — month-only rendering used to collapse every date in
    a month onto one label, which made "when did I …?" and same-month ordering
    unanswerable from the context text.

    Formatting is portable on purpose: ``%-d`` is glibc-only (it raises on
    Windows and is undefined elsewhere), so the day is interpolated directly.

    *weekday* names the day of the week in front of it, for the reader that is
    asked what happened "the past weekend". Only reachable on a day-precise
    date: the month-precision sentinel above returns before this line.
    """
    stem = f"{d.strftime('%b')} {d.day}, {d.year}"
    return f"{d.strftime('%a')} {stem}" if weekday else stem


def _parse_query_date(raw):
    """The moment the question is being asked, as an aware datetime, or None.

    Lenient on purpose. A parenthesised weekday is stripped first because that is
    the shape the benchmark harness already holds ("2023/03/11 (Sat) 21:26"), and
    making the caller reformat a string we can read is friction for no gain.
    Anything still unreadable logs and returns None, which means "use now" and is
    exactly today's behaviour.
    """
    if not raw:
        return None
    import re as _re
    from datetime import timezone as _tz
    text = _re.sub(r"\s*\([A-Za-z]{3,9}\)\s*", " ", str(raw)).strip()
    try:
        from dateutil import parser as _dp
        dt = _dp.parse(text)
    except Exception:
        logger.warning(f"query_date could not be parsed, falling back to now: {str(raw)[:60]!r}")
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=_tz.utc)


def _said_with_event_enabled() -> bool:
    """Whether every memory says when it was said, and labels the event date too.

    A row with a stored window renders bare, "(2022) ... five sessions". A row
    without one renders "(said May 11, 2023) ... three sessions". Served together
    those are two statements about one event, and the reader has nothing to order
    them by: "May 11, 2023" looks later than "2022". It is not. On f9e8c073 the
    five-sessions row was spoken on Oct 30, 2023, five months AFTER the other,
    and both a frontier reader and a mini reader answered three.

    So this prints both when they differ: "(2022, said Oct 30, 2023)". Measured on
    the lme14 run it would change about 11% of served memories, and it
    only ever ADDS a date.

    Both halves are labelled and "said" is always first, so every served memory
    opens the same way and no date is ever bare. Half-labelled was the problem:
    one row saying "(2022)" and another saying "(said May 11, 2023)" gives a
    reader two dates it cannot compare. The event half is dropped only when it
    would repeat the speaking date exactly, where one date says both and no
    reading can be wrong.

    Off by default, same reason as the prefix above: it changes the content string
    every SDK caller receives. Enable with
    ``SYNAP_RETRIEVAL_SAID_WITH_EVENT_DATE=on``.

    Deployment-wide rather than per instance, because this function is handed
    one item and never the request's instance id. Adding a date is additive for
    every caller, which is why that is an acceptable shape here and would not be
    for anything that removes or rewrites text.
    """
    import os
    return os.environ.get("SYNAP_RETRIEVAL_SAID_WITH_EVENT_DATE", "").strip().lower() in {"1", "true", "on", "yes"}


def _weekday_in_dates_enabled() -> bool:
    """Whether a day-precise date also names its day of the week.

    "Sun Mar 19, 2023" instead of "Mar 19, 2023". The reader is routinely asked
    which thing happened "the past weekend", "last Friday", "last Saturday", and
    the date-to-weekday arithmetic is arithmetic it gets wrong. 21 of the 470
    scored LongMemEval questions name a weekday or a weekend, and those fail at
    15.0% against a 6.7% baseline.

    On gpt4_e414231f ("which bike did I fix or service the past weekend?", gold
    road bike) we served a mountain-bike repair dated Mar 15, a WEDNESDAY, next
    to the Mar 19 road bike, and both readers answered with both bikes. On
    gpt4_fa19884d ("what artist did I start listening to last Friday?") the right
    memory is dated Fri Mar 31 and the one the reader picked is dated Sat Mar 25.

    Derived from the date we already hold, so it cannot disagree with anything
    and adds roughly one token per date. Off by default and enabled per
    deployment, same as the two flags above, because it changes the content
    string every SDK caller receives.
    """
    import os
    return os.environ.get("SYNAP_RETRIEVAL_WEEKDAY_IN_DATES", "").strip().lower() in {"1", "true", "on", "yes"}


def _age_in_dates_enabled() -> bool:
    """Whether a date also says how long before the question it was.

    "(said Sat Apr 15, 2023 19:13, 7 days ago)". 32 of the 470 scored questions
    point at a past moment with a phrase rather than a date ("two months ago",
    "last month", "most recently"), and those fail at 12.9% against a 6.7%
    baseline. The reader has the question date and every memory date and still
    has to subtract: on 0bc8ad93 ("I mentioned visiting a museum two months ago")
    it chose a visit 22 days back over the one 2 months back, and on
    gpt4_59149c78 ("an art event two weeks ago") it chose 24 days back over 17.

    Off by default, same reason as the flags above.
    """
    import os
    return os.environ.get("SYNAP_RETRIEVAL_AGE_IN_DATES", "").strip().lower() in {"1", "true", "on", "yes"}


_AGE_SUPPRESS_DAYS = 365


def _age_reference(decision, bundle):
    """The date an "N days ago" is counted back from, or None to print no age.

    The caller's own question date when it sent one (``query_date``), because
    that is when the question is being asked; this server's clock otherwise,
    because for a live conversation that IS the present.

    None when the reference is more than a year later than the newest memory in
    the bundle. That is a replay of old data against today's clock, a harness
    re-asking 2023 questions in 2026, where every memory would read "3 years
    ago", which is uniform and useless. Measured on the 470 scored questions: the
    newest served memory is a median of 1 day and at most 122 days before the
    question date, so a real reference never trips this.
    """
    if not _age_in_dates_enabled():
        return None
    reference = getattr(decision, "query_date", None)
    if reference is None or not hasattr(reference, "date"):
        reference = datetime.now(timezone.utc).replace(tzinfo=None)
    newest = None
    try:
        for items in bundle.items_by_type.values():
            for item in items:
                for attr in ("said_at", "event_date", "created_at"):
                    dt = getattr(item, attr, None)
                    if dt is not None and hasattr(dt, "date"):
                        if newest is None or dt.date() > newest:
                            newest = dt.date()
                        break
    except Exception:  # noqa: BLE001 - an age never fails a response
        return None
    if newest is None:
        return None
    try:
        ref_day = reference.date()
    except AttributeError:
        return None
    if (ref_day - newest).days > _AGE_SUPPRESS_DAYS:
        logger.info("age_in_dates_suppressed reference=%s newest_memory=%s",
                    ref_day, newest)
        return None
    return reference


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit} ago" if n == 1 else f"{n} {unit}s ago"


def _age_phrase(days: int) -> str:
    """How long ago, in the unit the question is likely to use.

    Rounded to the unit a person would say, because that is the unit the
    question asks in: "two weeks ago" has to match 17 days and "two months ago"
    has to match 59. Days up to a fortnight stay exact so "the past weekend" and
    "last Friday" are not rounded away. "" for a same-day age, which is a
    dangling word beside the date the prefix already printed.

    The week-to-month boundary sits at 5 weeks rather than at 10, so 59 days
    reads "2 months ago" and not "8 weeks ago". 0bc8ad93 asks about a visit "two
    months ago" and the two candidates are 59 days and 22 days back; in weeks
    those are 8 and 3, and neither matches the words in the question.
    """
    if days <= 0:
        return ""
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 35:
        return _plural(round(days / 7), "week")
    if days < 365:
        return _plural(round(days / 30.44), "month")
    return _plural(round(days / 365.25), "year")


def _age_months_phrase(target, reference) -> str:
    """The age of a date we only know to the month, counted in whole months.

    A month-precision event is stored on the 1st (the sentinel in
    ``_format_day_precise``), so counting its age in days or weeks would publish
    a precision the source never gave: "happened Jan 2023, 10 weeks ago" asserts
    a day that nobody said. Whole months is the coarseness we actually have, and
    it is also the coarseness the question uses.
    """
    months = ((reference.year - target.year) * 12
              + (reference.month - target.month))
    if months <= 0:
        return ""
    if months >= 12 and months % 12 == 0:
        return _plural(months // 12, "year")
    return _plural(months, "month")


def _age_suffix(target, reference, coarse: bool = False) -> str:
    """", 7 days ago", or "" when there is nothing safe to say.

    The flag is re-read here and not only in ``_age_reference``, so that one
    switch is the single answer to "does this deployment print an age". Gating it
    at the reference alone left any later caller that passes a date of its own
    able to turn the age on without the flag.

    Empty for a date AFTER the reference: an age counted the wrong way round
    would read as though the thing had already happened.

    *coarse* is for a date known only to the month, season or year.
    """
    if not _age_in_dates_enabled():
        return ""
    if reference is None or target is None or not hasattr(target, "date"):
        return ""
    try:
        days = (reference.date() - target.date()).days
    except (AttributeError, TypeError):
        return ""
    if days < 0:
        return ""
    # Nothing is said about a zero-length age. The reference IS the day the
    # prefix just printed, so ", today" and ", this month" are a dangling word
    # that adds no fact. Both helpers return "" for it.
    phrase = (_age_months_phrase(target, reference) if coarse
              else _age_phrase(days))
    return f", {phrase}" if phrase else ""


def _event_half_is_coarse(item) -> bool:
    """Whether the event half names something wider than one day.

    A month, a season, a year or an "around" date. Checked so the age beside it
    is stated at the same coarseness, rather than turning "Jan 2023" into a
    number of days counted from the 1st.
    """
    precision = (getattr(item, "event_precision", None) or "").lower()
    if precision in {"month", "season", "year", "approximate"}:
        return True
    if precision:
        return False
    dt = getattr(item, "event_date", None) or getattr(item, "created_at", None)
    # No stored precision: the day-1 sentinel is what month precision looks like.
    return bool(dt is not None and getattr(dt, "day", 0) == 1)


def _event_half(item, weekday: bool = False) -> str:
    """The "happened ..." half of the prefix: the event window, or "" .

    Everything the legacy prefix used to print BARE, now returned as a bare
    string so the caller can label it. Order matches the legacy branches so the
    two paths cannot disagree about what the event date is.
    """
    # Only a real event date. The `created_at` arm below used to be
    # unreachable, because the read path filled `event_date` for every row, and
    # the rows it filled were caught by the `event_date_is_mention` test on the
    # next line. `event_date` is now empty for them, so an empty event half is
    # the whole answer and our own ingest clock never stands in for one.
    dt = item.event_date if (item.event_date
                             and hasattr(item.event_date, "strftime")) else None
    if dt is None:
        return ""
    try:
        if getattr(item, "event_precision", None):
            from synap.cloud.agents.ingestion.services.event_window import format_event_window
            return format_event_window(dt, getattr(item, "event_end", None),
                                       item.event_precision, weekday=weekday) or ""
        last_obs = getattr(item, "document_last_observed_at", None)
        if (last_obs is not None and hasattr(last_obs, "strftime")
                and (last_obs - dt).days > _DCT_RANGE_THRESHOLD_DAYS):
            start, end = dt.strftime("%b %Y"), last_obs.strftime("%b %Y")
            if start != end:
                return f"{start} – {end}"
        if getattr(dt, "day", 1) == 1:
            return dt.strftime("%b %Y")
        return _format_day_precise(dt, weekday=weekday)
    except Exception:  # noqa: BLE001 — a prefix never fails a response
        return ""


def _format_day_precise_always(dt, weekday: bool = False) -> str:
    """"May 27, 2023", or "May 28, 2023 06:23" when the clock time is real.

    The sentinel elsewhere exists because month-precision resolution emits the
    1st, so day == 1 means "sometime that month". A document date is a real
    timestamp, so the 1st means the 1st and must not be widened to a month.

    ⚠ The time is kept because dropping it made same-day statements unorderable.
    a2f3aa27 asks "how many followers do I have NOW". The user said 1250 in a
    conversation at 06:23 and "close to 1300" in a DIFFERENT conversation at
    14:10 the same day. Both rendered as "(said May 28, 2023)", so the reader had
    nothing to order them by and answered 1250; the answer is 1300.

    Midnight is left bare on purpose: a source that supplied only a date lands on
    00:00, and printing it would assert a precision nobody gave us.
    """
    try:
        day = dt.strftime("%b %-d, %Y")
    except ValueError:
        day = dt.strftime("%b %d, %Y")
    if weekday:
        day = f"{dt.strftime('%a')} {day}"
    try:
        if (dt.hour, dt.minute) != (0, 0):
            return f"{day} {dt.hour:02d}:{dt.minute:02d}"
    except AttributeError:
        pass
    return day


def _happened_or_planned(item, said) -> str:
    """"happened" for a past event, "planned for" one still ahead of the speaker.

    Reads the event window's start against the day the user spoke. Both are
    already on the item, so this adds no field and no lookup, and it changes
    only the word: a wrong date stays exactly as wrong as it was.
    """
    start = getattr(item, "event_date", None)
    if start is None or said is None:
        return "happened"
    try:
        if start.date() > said.date():
            return "planned for"
    except AttributeError:
        return "happened"
    return "happened"


def _format_date_prefix(item, reference=None) -> str:
    """Build a date prefix like '(Jul 14, 2026)' from the item's temporal metadata.

    Priority: event_date > created_at. Returns empty string if no date available.
    This makes dates visible in the content text so the LLM can reason about
    temporal ordering without needing to parse metadata fields.

    Precision follows the day-1 sentinel rule in ``_format_day_precise``:
    day != 1 renders "(Jul 14, 2026)", day == 1 stays month-only "(Jul 2026)"
    because the 1st is what month-precision resolution emits.

    For KU v2 merged memories that carry `document_last_observed_at` distinct
    from `created_at` by more than 30 days, render a range like
    "(Jan 2023 – Jul 2023)" so the LLM sees the merged memory describes a
    pattern observed across that span — not a single point in time. Ranges stay
    month-granular: the span, not the endpoints, is the signal there.
    """
    # ── the uniform form ──────────────────────────────────────────────────────
    #
    # EVERY memory says when the user said it, and adds when the thing happened
    # whenever that is a different time. No date in the prefix is ever
    # unlabelled, so a reader never has to infer which kind it is looking at.
    #
    # The half-and-half version this replaced is what cost f9e8c073: one row
    # printed "(2022) ... five sessions" and another "(said May 11, 2023) ...
    # three sessions", and both a frontier reader and a mini reader answered
    # three, because "May 11, 2023" looks later than "2022". It is not — the
    # five-sessions row was spoken on Oct 30, 2023. One row labelled its date and
    # the other did not, so there was nothing to compare.
    weekday = _weekday_in_dates_enabled()
    said = getattr(item, "said_at", None)
    if _said_with_event_enabled() and said is not None and hasattr(said, "strftime"):
        said_str = _format_day_precise_always(said, weekday=weekday)
        happened = _event_half(item, weekday=weekday)
        # Dropped only when it would repeat the speaking date word for word: the
        # event was that same single day, so one date says both and no reading of
        # it can be wrong. Compared as rendered strings on purpose — a window
        # like "Mar 27 – Mar 31, 2023" or "around May 13, 2023" never matches,
        # so a span or an approximation is always kept.
        if happened and happened != said_str:
            # "happened" or "planned for". A date after the day the user spoke
            # has not happened yet, and saying it happened makes an intention
            # countable as an occurrence. On 88432d0a ("how many times did I
            # bake something in the past two weeks", gold 4) we served "plans
            # to try a new recipe for rustic Italian bread this weekend
            # (May 27-28)" alongside five completed bakes, with the same word
            # in front of both.
            #
            # Compared by calendar day, not by instant: an event stored at
            # midnight on the day it was spoken about is earlier as a timestamp
            # and is not a plan. "planned for" rather than "will happen"
            # because by the time anyone reads this the day may have passed —
            # what we know is that it had not happened when the user said it.
            #
            # The age counts back to the EVENT, because that is the date the
            # question's "two months ago" is pointing at.
            #
            # Nothing is counted back to a date the speaker had not reached yet.
            # "planned for Sat Mar 18, 3 days ago" would read as an occurrence,
            # which is the exact confusion the "planned for" word exists to stop:
            # on 88432d0a a planned bake was counted among five completed ones.
            tense = _happened_or_planned(item, said)
            age = ""
            if tense == "happened":
                age = _age_suffix(
                    getattr(item, "event_date", None) or said, reference,
                    coarse=_event_half_is_coarse(item))
            return f"(said {said_str}, {tense} {happened}{age}) "
        return f"(said {said_str}{_age_suffix(said, reference)}) "

    # ── the legacy form, for a deployment with the flag off ───────────────────
    dt = item.event_date if (item.event_date
                             and hasattr(item.event_date, 'strftime')) else None
    if dt is None:
        # No event date. Say when it was said, or say nothing -- never our own
        # ingest clock, which is what the `created_at` arm here used to reach
        # for once `event_date` stopped being filled unconditionally.
        said_only = getattr(item, "said_at", None)
        if said_only is not None and hasattr(said_only, "strftime"):
            return f"(said {_format_day_precise_always(said_only, weekday=weekday)}) "
        return ""

    try:
        # A stored event window (migration 154) is rendered as the window it
        # is: "(Feb 2023)", "(Apr 2 – Apr 8, 2023)", "(around May 13, 2023)".
        # Rows without one fall through to the formats below, unchanged.
        if item.event_date and getattr(item, "event_precision", None):
            from synap.cloud.agents.ingestion.services.event_window import format_event_window
            label = format_event_window(item.event_date, getattr(item, "event_end", None),
                                        item.event_precision)
            if label:
                return f"({label}) "
        last_obs = getattr(item, "document_last_observed_at", None)
        if (
            last_obs is not None
            and hasattr(last_obs, "strftime")
            and (last_obs - dt).days > _DCT_RANGE_THRESHOLD_DAYS
        ):
            start = dt.strftime("%b %Y")
            end = last_obs.strftime("%b %Y")
            if start != end:
                return f"({start} – {end}) "
        if getattr(dt, "day", 1) == 1:
            return f"({dt.strftime('%b %Y')}) "
        return f"({_format_day_precise(dt, weekday=weekday)}) "
    except Exception:
        return ""


def stream_date_prefixes(items_by_type: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    """The date label of every item in a bundle the SDK stream carries, in the
    bundle's own shape: ``{type: [label, ...]}``, one per item, "" for none.

    An SDK answering from its anticipation cache never reaches
    ``_bundle_item_to_context_item``, which is the only place the label used to
    be written, so the same memory read "(said May 11, 2023) ..." from the
    server and bare from the cache. The label now rides with the item
    (``ContextItemProto.date_prefix``) and the SDK puts it in front of the
    content when it serves a hit. Written here, by ``_format_date_prefix``
    itself, so the wording has one source and the SDKs never rebuild it.

    The age ("7 days ago") counts back from now, the push, as a live reply
    with no ``query_date`` does; decided once for the bundle by
    ``_age_reference``, as the HTTP reply decides it once for its own. A cache
    entry lives 30 minutes at most, so that is how stale the age can get.

    The "[thing N]" marker is left out on purpose: its numbers are counted
    across one reply, and a cache hit merges items from several bundles.

    Takes item dicts (``ContextItem.to_dict``) or items. Never raises: an item
    it cannot read gets no label, which is what it had before this.
    """
    from types import SimpleNamespace
    from synap.cloud.agents.models.bundles import ContextItem as _BundleItem

    parsed: Dict[str, List[Any]] = {}
    for ctx_type, entries in (items_by_type or {}).items():
        row: List[Any] = []
        for entry in entries or []:
            try:
                row.append(_BundleItem.from_dict(entry) if isinstance(entry, dict) else entry)
            except Exception:  # noqa: BLE001 - a missing key costs the label only
                row.append(None)
        parsed[ctx_type] = row

    holder = SimpleNamespace(items_by_type={
        t: [i for i in row if i is not None] for t, row in parsed.items()})
    try:
        reference = _age_reference(None, holder)
    except Exception:  # noqa: BLE001
        reference = None

    def _label(item: Any) -> str:
        if item is None:
            return ""
        try:
            return _format_date_prefix(item, reference) or ""
        except Exception:  # noqa: BLE001 - a label never fails a push
            return ""

    return {t: [_label(i) for i in row] for t, row in parsed.items()}


def _queries_as_sent(item, as_sent: Optional[Dict[str, str]], order: Optional[Dict[str, int]]) -> List[str]:
    """The item's matched queries, in the caller's words and the caller's order.

    Retrieval tags an item with the string it searched on. When a privacy policy
    rewrote the query before the search, that string is the rewritten one, and
    it is the rewritten one that sits in the cache, which more than one caller
    can read. The caller's own wording is put back here, after the cache, the
    same side of it that ``_reveal_for_caller`` works on.
    """
    tags = list(getattr(item, "matched_queries", None) or [])
    if as_sent:
        tags = [as_sent.get(t, t) for t in tags]
    tags = list(dict.fromkeys(tags))
    if order:
        tags.sort(key=lambda t: order.get(t, len(order)))
    return tags


def _subject_marker_enabled() -> bool:
    """Whether a served item says which thing it is about, in its text.

    The grouping is on the row either way and always rides in metadata. This
    flag decides only whether it also enters the content string, which is the
    one place a reader is certain to look.

    Off by default and deployment-wide, like the three date flags above, and for
    the same reason: it changes the content string every SDK caller receives.
    Enable with ``SYNAP_RETRIEVAL_SUBJECT_MARKER=on``.
    """
    import os
    return os.environ.get("SYNAP_RETRIEVAL_SUBJECT_MARKER", "").strip().lower() in {"1", "true", "on", "yes"}


def _subject_ordinals(bundle) -> dict:
    """``{subject_id: 1..N}`` over one bundle, in the order subjects first appear.

    A short ordinal, not the label, is what goes in the text. The label is 160
    characters at worst and repeating it on eight items about one coffee table
    spends the budget that the eighth item needed; "thing 3" costs three tokens
    and says the only thing the reader has to know, which is that these items
    and no others are about one thing. The label still travels in metadata, and
    that is where we read it back when a grouping looks wrong.

    Numbered per response and never stored, so nothing here can disagree with
    the write-side grouping: this only renames it for one reader.
    """
    order: dict = {}
    try:
        for _type, items_list in (bundle.items_by_type or {}).items():
            for it in items_list or []:
                sid = getattr(it, "subject_id", None)
                if sid and sid not in order:
                    order[sid] = len(order) + 1
    except Exception:  # noqa: BLE001 — a marker never fails a response
        return {}
    return order


def _bundle_item_to_context_item(item, reference=None, as_sent=None, query_order=None,
                                 subject_ordinals=None, rank: Optional[int] = None) -> ContextItem:
    """Convert internal ContextItem (bundles.py) to SDK ContextItem (response model).

    Internal ContextItem fields: item_id, content, context_type, source,
    similarity_score, relevance_score, confidence, scope, entity_id, created_at.
    It does NOT have 'metadata' or 'category'.

    Prepends a date prefix to content (e.g., '(May 2023) user did X') so
    the downstream LLM can see temporal context directly in the text.

    ``rank`` set means the ranked shape (`response_shape="ranked"`): the item
    comes back as a RankedContextItem with its place, and a chunk is typed
    "excerpts" and carries its rank_score. Left None, the item is exactly what
    the default shape has always served.
    """
    ranked = rank is not None
    is_chunk = getattr(item, "source", None) == "chunks"
    date_prefix = _format_date_prefix(item, reference)

    # Which thing this item is about (migration 159), as a short ordinal after
    # the date and before the text: "(said May 24, 2023) [thing 1] The user got
    # a new coffee table ...". Items sharing a number are statements about one
    # real-world thing, which is what a reader counting them has to know.
    #
    # Only when ingestion actually grouped it. An ungrouped item gets no marker
    # rather than a marker of its own, because "thing 7" on a lone item reads as
    # a seventh thing and would invent exactly the count we are trying to fix.
    #
    # ⚠ Added to `prefix`, never to `content` afterwards, because the PII spans
    # below are moved by len(prefix) and the comment there asks every later step
    # that rewrites the content to move them too. Two separate concatenations
    # would shift them by the date alone and quietly highlight the wrong words.
    subject_id = getattr(item, "subject_id", None)
    subject_ordinal = (subject_ordinals or {}).get(subject_id) if subject_id else None
    prefix = date_prefix
    if subject_ordinal and _subject_marker_enabled():
        prefix = f"{date_prefix}[thing {subject_ordinal}] "
    content = f"{prefix}{item.content or ''}" if prefix else (item.content or "")

    metadata = {
        "scope": item.scope,
        "similarity_score": item.similarity_score,
        "relevance_score": item.relevance_score,
    }
    # Always in metadata, flag or no flag: the grouping is a fact about the row
    # and a caller that wants to count distinct things should not have to parse
    # it back out of a sentence. Only when known, like assertion_source below,
    # so a response for ungrouped memories stays byte-identical to before.
    if subject_id:
        metadata["subject_id"] = subject_id
        _label = getattr(item, "subject_label", None)
        if _label:
            metadata["subject_label"] = _label
        if subject_ordinal:
            metadata["subject_ordinal"] = subject_ordinal
    # Provenance rides in metadata only when known — None/legacy rows keep
    # the response shape byte-identical to pre-071 payloads.
    assertion_source = getattr(item, "assertion_source", None)
    if assertion_source:
        metadata["assertion_source"] = assertion_source
    # The custom category a memory was extracted under: additive and only
    # when the retrieval step found it among the instance's configured names.
    custom_category = getattr(item, "custom_category", None)
    if custom_category:
        metadata["custom_category"] = custom_category

    # THE SCORE WE RANKED BY, as one opaque number.
    #
    # Until this was added, nothing in the response could reproduce the order we
    # served. `relevance_score` above is the raw vector cosine: both fields are
    # set from the same number at construction (`similarity_score=score,
    # relevance_score=score`), and measured on thousands of items served by one
    # benchmark run they were equal on 100.0% of them. The final sort, meanwhile,
    # runs on `_ranking_key`, so the signal the order depends on was unpublished.
    #
    # That is not only cosmetic. A caller that wants a top-5, or that merges two
    # fetches, needs ONE comparable number, and the only two we gave it were the
    # cosine and the ingestion `confidence` — which looks like a relevance and is
    # not. The benchmark harness duly sorts on `confidence`, which across the
    # same 6,196 items took 10 values, 88% of them 0.9/0.85/0.95, and ran
    # BACKWARDS against relevance: the items labelled 1.0 had the lowest median
    # cosine of any bucket, 0.242. On question 830ce83f the Chicago memory and an
    # unrelated hurricane-season fact were both served at confidence 0.95 with
    # cosines of 0.5955 and 0.0453.
    #
    # Deliberately ONE number with a name that says what it is for, not how it
    # is made: no component names, no per-signal breakdown. What the caller is
    # promised is only that higher is better and that sorting on it descending
    # reproduces the order we sent. How we arrive at it stays ours to change.
    #
    # Additive and only when known, like `assertion_source` above, so a client
    # reading neither key sees an unchanged payload. `relevance_score` keeps its
    # existing meaning rather than being redefined under clients already reading
    # it. It is a stashed attribute rather than a declared field, so it is not
    # guaranteed to be a number and a bad stash must not 500 a whole fetch.
    #
    # Read from the declared `rank_score` first. It is what assembly published
    # and, unlike the stashed `_final_score`, it survives the L1 cache, so a
    # cache hit now carries the same rank_score as the live fetch it copies.
    #
    # ⚠ Chunks are scored too now, so the ranked shape can place them, but the
    # default shape must not start publishing that score: a chunk there sits at
    # a fixed place, and a rank_score beside it would say the opposite.
    _rank = getattr(item, "rank_score", None)
    if _rank is None:
        _rank = getattr(item, "_final_score", None)
    if _rank is not None and (ranked or not is_chunk):
        try:
            metadata["rank_score"] = float(_rank)
        except (TypeError, ValueError):
            pass

    # PII spans, SHIFTED BY THE DATE PREFIX.
    #
    # The offsets the read path recorded index `item.content`. This function
    # ships `date_prefix + item.content`, so every span is wrong by exactly
    # len(date_prefix) unless it is moved — and wrong quietly: the highlight
    # still renders, just over the neighbouring words, which reads as a bug in
    # the aliasing rather than in a string concatenation.
    #
    # This is why `pii_revealed` is a declared field on the internal item. Any
    # future step that rewrites `content` has to move these too, and a field is
    # visible to whoever writes that step in a way a stashed attribute is not.
    spans = getattr(item, "pii_revealed", None)
    if spans:
        shift = len(prefix)
        metadata["pii_revealed"] = [
            {
                "start": sp["start"] + shift,
                "end": sp["end"] + shift,
                "field_type": sp.get("field_type", ""),
            }
            for sp in spans
            if isinstance(sp, dict) and "start" in sp and "end" in sp
        ]

    _said = getattr(item, "said_at", None)
    said_at_out = _said.isoformat() if hasattr(_said, "isoformat") else None

    fields = dict(
        id=item.item_id or "",
        content=content,
        confidence=item.confidence or 0.0,
        source=item.source or "memory",
        category=(EXCERPT_CATEGORY if ranked and is_chunk else item.context_type or ""),
        metadata=metadata,
        said_at=said_at_out,
        event_date=item.event_date.isoformat() if item.event_date and hasattr(item.event_date, 'isoformat') else None,
        event_end=item.event_end.isoformat() if getattr(item, 'event_end', None) and hasattr(item.event_end, 'isoformat') else None,
        event_precision=getattr(item, 'event_precision', None) or None,
        valid_until=item.valid_until.isoformat() if item.valid_until and hasattr(item.valid_until, 'isoformat') else None,
        temporal_category=item.temporal_category,
        temporal_confidence=item.temporal_confidence or 0.0,
        source_document_id=item.source_document_id if hasattr(item, 'source_document_id') else None,
        source_evidence=item.source_evidence if hasattr(item, 'source_evidence') else None,
        matched_queries=_queries_as_sent(item, as_sent, query_order),
    )
    if ranked:
        return RankedContextItem(**fields, rank=rank)
    return ContextItem(**fields)


async def _resolve_scope_from_conversation(
    conversation_id: str,
    instance_id: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, str]]]:
    """
    Resolve user_id and customer_id from a stored conversation record.

    Fallback only. Post migration 063, conversation_id is opaque text and
    the gRPC tracker writes the row asynchronously, so this lookup can race
    against the tracker write. Callers should prefer scope ids passed in
    the request body and only call this when none are provided.

    Args:
        conversation_id: Conversation id (opaque string supplied by the SDK)
        instance_id: The caller's instance. The row must belong to it: a
            conversation id from another instance resolves to nothing, not to
            that instance's user, customer and rung.

    Returns:
        (user_id, customer_id, scope_path) — any may be None if the
        conversation row doesn't exist yet or has no scope ids.

    ⚠ `scope_path` is the third element and it is the reason this signature
    changed. Returning only the two ids meant a fetch that auto-resolved from
    the conversation lost the rung, even for a conversation that was opened at
    one: the caller had named a practice on the write and got the whole tenant
    back on the read. The two ids cannot carry a rung, so the path has to come
    out of the row with them.
    """
    from synap.cloud.database.connection import get_session_factory
    from sqlalchemy import text
    from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

    # Match writer-side coercion: tracker.record_message stores the
    # ``conversation`` row at the coerced UUID5 form, so a raw lookup
    # (e.g. ``pgv2_xxx``) here misses every time.
    conversation_id = coerce_conversation_id(conversation_id)

    # The scope path stored on the row belongs to the scope ladder. An edition
    # without the ladder (scope_levels is set) does not read it, so it never
    # reaches the scope that is built for the fetch.
    _read_scope_path = edition.scope_levels is None

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT user_id, customer_id, scope_path
                    FROM conversation
                    WHERE conversation_id = :conversation_id
                      AND instance_id = :instance_id
                """ if _read_scope_path else """
                    SELECT user_id, customer_id, NULL
                    FROM conversation
                    WHERE conversation_id = :conversation_id
                      AND instance_id = :instance_id
                """),
                {"conversation_id": conversation_id, "instance_id": instance_id},
            )
            row = result.first()
            if row:
                user_id = str(row[0]) if row[0] else None
                customer_id = str(row[1]) if row[1] else None
                # JSONB comes back decoded on this driver, but a row written
                # before the column existed, or by a path that stored a string,
                # must not turn a fetch into a 500. A path we cannot read is a
                # path we do not have.
                raw = row[2]
                if isinstance(raw, str):
                    try:
                        raw = _json.loads(raw)
                    except ValueError:
                        raw = None
                scope_path = raw if isinstance(raw, dict) and raw else None
                return user_id, customer_id, scope_path
            return None, None, None
    except Exception as e:
        logger.warning(
            f"Failed to resolve scope from conversation_id={conversation_id}: {e}",
            exc_info=True
        )
        return None, None, None


def _record_fetch_telemetry(
    *,
    decision: ContextDecision,
    bundle,
    fetch_latency_ms: float,
    conv_ctx_present: bool,
) -> None:
    """Attribute a cold-path SDK fetch to the active turn's telemetry row.

    Server-side fetches always take the "cloud" transport — the
    anticipation transport is decided in the SDK's in-process cache before
    the request even reaches us. We still record the cloud fetch so the
    dashboard's "Fetches" panel shows what the customer's agent actually
    asked for and what came back.

    Items_picked is a short preview of what we returned (first item per
    type) so an operator can sanity-check against the agent's eventual
    response without storing the full payload again.

    Best-effort: never raises.
    """
    if not decision or not decision.conversation_id:
        return
    instance = str(decision.instance_id or "")
    conv = str(decision.conversation_id or "")
    if not (instance and conv):
        return

    try:
        from synap.cloud.observability.anticipation_telemetry import (
            lookup_for_conversation,
        )
    except Exception:
        return
    collector = lookup_for_conversation(instance, conv)
    if collector is None:
        return

    try:
        bundle_items = getattr(bundle, "items_by_type", {}) or {}
        items_preview = []
        for item_type, items in bundle_items.items():
            if not items:
                continue
            head = items[0]
            content = getattr(head, "content", "") or ""
            items_preview.append({
                "item_index": 0,
                "bundle_id": getattr(bundle, "bundle_id", "") or "",
                "item_type": item_type,
                "content_first_240": str(content)[:240],
                "bm25_score": 0.0,  # server-side fetch — no BM25 ranking, just retrieval
                "confidence": float(getattr(head, "confidence", 0.0) or 0.0),
            })

        retrieval_mode = (
            decision.retrieval_mode.value
            if hasattr(decision.retrieval_mode, "value")
            else str(decision.retrieval_mode)
        )

        collector.record_fetch(
            scope=_scope_label_for_decision(decision),
            search_query=list(decision.search_queries or []),
            transport="cloud",
            anticipation_event_fired=None,
            wait_ms=0,
            latency_ms=int(fetch_latency_ms or 0),
            item_count=int(getattr(bundle, "item_count", 0) or 0),
            has_conversation_context=conv_ctx_present,
            cache_state_at_lookup={
                "server_side": True,
                "cache_hit": bool(getattr(bundle, "cache_hit", False)),
                "degradation_level": (
                    bundle.degradation_level.value
                    if hasattr(bundle, "degradation_level") and hasattr(bundle.degradation_level, "value")
                    else None
                ),
                "mode": retrieval_mode,
            },
            scope_filter_request={
                "user_id": decision.user_id,
                "customer_id": decision.customer_id,
                "conversation_id": decision.conversation_id,
            },
            scope_filter_accepted=[_scope_label_for_decision(decision)],
            items_picked=items_preview,
            items_rejected=[],
        )
    except Exception:
        logger.debug("record_fetch failed", exc_info=True)


def _scope_label_for_decision(decision: ContextDecision) -> str:
    """Best label for the fetch's scope — matches the dashboard funnel."""
    # User scope is the narrowest; customer broadens it; conversation is
    # always within a customer; client is the broadest.
    if decision.user_id:
        return "user"
    if decision.customer_id:
        return "customer"
    if decision.conversation_id:
        return "conversation"
    return "client"


def _empty_fetch_response(request: "ContextFetchRequest") -> "ContextFetchResponse":
    """The response for a fetch that names nobody to fetch for, in the shape
    the caller asked for. An empty ranked payload is ``{"items": []}``, so a
    ranked caller never has to handle the by-type keys."""
    if getattr(request, "response_shape", "by_type") == "ranked":
        return ContextFetchResponse(context=RankedContextPayload(items=[]))
    return ContextFetchResponse()


def _ranked_order(bundle) -> List[Any]:
    """The bundle's items, best first, for the ranked shape.

    The order is ``bundle.ranked_item_ids``, written by assembly. The items
    themselves come from ``items_by_type``, which stays the record of what is
    served: an id whose item a later step dropped (the hard token budget) is
    skipped, and an item the id list does not name goes after the named ones,
    by rank_score. A bundle from before the id list existed, which is any L1
    cache entry written by an older build, has no ids and is sorted on
    rank_score alone. Unscored items keep their relative order after the
    scored ones.
    """
    present: Dict[str, Any] = {}
    for items_list in (bundle.items_by_type or {}).values():
        for it in items_list or []:
            present.setdefault(getattr(it, "item_id", None) or str(id(it)), it)

    ordered: List[Any] = []
    for iid in getattr(bundle, "ranked_item_ids", None) or []:
        it = present.pop(iid, None)
        if it is not None:
            ordered.append(it)

    def _score(it):
        r = getattr(it, "rank_score", None)
        if r is None:
            r = getattr(it, "_final_score", None)
        try:
            return float(r) if r is not None else None
        except (TypeError, ValueError):
            return None

    rest = sorted(present.values(),
                  key=lambda it: (_score(it) is None, -(_score(it) or 0.0)))
    return ordered + rest


async def _fetch_via_retrieval_manager(
    retrieval_manager: RetrievalManager,
    decision: ContextDecision,
    scope: ScopeHierarchy,
    response_shape: str = "by_type",
) -> tuple:
    """Fetch context using RetrievalManager and convert to SDK payload format.

    Returns:
        Tuple of (ContextPayload or RankedContextPayload,
        Optional[ConversationContextPayload], Optional[RetrievalHealth]). The
        first is a RankedContextPayload only when ``response_shape`` is
        "ranked". The third element is None on a healthy fetch and is what the
        caller puts in ``ContextFetchResponse.retrieval``.
    """
    from synap.cloud.shared.logging_utils import truncate_string

    query_cap_warning = _cap_search_queries(decision)
    if query_cap_warning:
        logger.warning(f"search_query_cap_applied decision_id={decision.decision_id} {query_cap_warning}")
    # The queries as the caller wrote them. fetch_context may replace
    # ``decision.search_queries`` in place (privacy rewriting), and the items
    # come back tagged with the replacements.
    queries_as_sent = list(decision.search_queries or [])

    trigger_type = decision.trigger_type.value if hasattr(decision.trigger_type, 'value') else str(decision.trigger_type)
    retrieval_mode = decision.retrieval_mode.value if hasattr(decision.retrieval_mode, 'value') else str(decision.retrieval_mode)
    search_queries = [truncate_string(q, 100) for q in decision.search_queries] if decision.search_queries else None
    scope_info = f"client_id={scope.client_id} instance_id={scope.instance_id} customer_id={scope.customer_id} user_id={scope.user_id}" if scope else "scope=None"
    logger.info(
        f"retrieval_manager_fetch_start decision_id={decision.decision_id} trigger_type={trigger_type} retrieval_mode={retrieval_mode} context_types={decision.context_types} search_queries={search_queries} instance_id={decision.instance_id} user_id={decision.user_id} customer_id={decision.customer_id} conversation_id={decision.conversation_id} max_results={decision.max_results} token_budget={decision.token_budget} {scope_info}",
    )

    async with start_span("sdk.context.fetch", {
        "synap.decision_id": decision.decision_id,
        "synap.retrieval_mode": retrieval_mode,
        "synap.trigger_type": trigger_type,
        "synap.context_types": ",".join(decision.context_types or []),
        "synap.query_count": len(decision.search_queries or []),
        "synap.max_results": decision.max_results,
        "synap.instance_id": str(scope.instance_id) if scope else "",
        "synap.transport": "http",
        # Which response shape was asked for. Read before moving the default to
        # "ranked": the move waits until callers have stopped asking for by_type.
        "synap.response_shape": response_shape,
    }):
        try:
            t0 = time.monotonic()
            bundle = await retrieval_manager.fetch_context(decision, scope=scope)

            _searched = list(decision.search_queries or [])
            _as_sent = (
                dict(zip(_searched, queries_as_sent))
                if len(_searched) == len(queries_as_sent) and _searched != queries_as_sent
                else None
            )
            _query_order = {q: i for i, q in reversed(list(enumerate(queries_as_sent)))}

            bundle_id = bundle.bundle_id if hasattr(bundle, 'bundle_id') else None
            item_count = bundle.item_count if hasattr(bundle, 'item_count') else 0
            total_tokens = bundle.total_tokens if hasattr(bundle, 'total_tokens') else 0
            retrieval_time_ms = bundle.retrieval_time_ms if hasattr(bundle, 'retrieval_time_ms') else 0
            cache_hit = bundle.cache_hit if hasattr(bundle, 'cache_hit') else False
            degradation_level = bundle.degradation_level.value if hasattr(bundle, 'degradation_level') and hasattr(bundle.degradation_level, 'value') else None
            items_by_type_counts = {key: len(items) for key, items in bundle.items_by_type.items()} if hasattr(bundle, 'items_by_type') else {}
            fetch_latency_ms = round((time.monotonic() - t0) * 1000, 2)
            logger.info(
                f"retrieval_manager_fetch_success decision_id={decision.decision_id} response_shape={response_shape} bundle_id={bundle_id} item_count={item_count} total_tokens={total_tokens} retrieval_time_ms={retrieval_time_ms} cache_hit={cache_hit} degradation_level={degradation_level} items_by_type_counts={items_by_type_counts} fetch_latency_ms={fetch_latency_ms}",
            )

            set_current_span_attributes({
                "synap.bundle_id": bundle_id or "",
                "synap.items_returned": item_count,
                "synap.total_tokens": total_tokens,
                "synap.cache_hit": cache_hit,
                "synap.degradation_level": degradation_level or "",
                "synap.fetch_latency_ms": fetch_latency_ms,
            })

            payload = ContextPayload()
            ranked_payload = None

            # One reference for the whole bundle, decided once: every item's
            # "N days ago" has to count back to the same day or they cannot be
            # compared with each other.
            age_reference = _age_reference(decision, bundle)

            # Subjects numbered once for the whole bundle, for the same reason
            # the age reference is decided once: a marker only means anything if
            # the same thing carries the same number everywhere in one response.
            _subjects = _subject_ordinals(bundle)

            if response_shape == "ranked":
                ranked_payload = RankedContextPayload(items=[
                    _bundle_item_to_context_item(item, age_reference, _as_sent, _query_order,
                                                 _subjects, rank=n)
                    for n, item in enumerate(_ranked_order(bundle), start=1)
                ])

            for item in bundle.items_by_type.get("facts", []):
                payload.facts.append(_bundle_item_to_context_item(item, age_reference, _as_sent, _query_order, _subjects))

            for item in bundle.items_by_type.get("preferences", []):
                payload.preferences.append(_bundle_item_to_context_item(item, age_reference, _as_sent, _query_order, _subjects))

            for item in bundle.items_by_type.get("episodes", []):
                payload.episodes.append(_bundle_item_to_context_item(item, age_reference, _as_sent, _query_order, _subjects))

            for item in bundle.items_by_type.get("emotions", []):
                payload.emotions.append(_bundle_item_to_context_item(item, age_reference, _as_sent, _query_order, _subjects))

            for item in bundle.items_by_type.get("temporal_events", []):
                payload.temporal_events.append(_bundle_item_to_context_item(item, age_reference, _as_sent, _query_order, _subjects))

            # Compute date range across all items for temporal framing
            all_internal_items = []
            for items_list in bundle.items_by_type.values():
                all_internal_items.extend(items_list)
            event_dates = [
                item.event_date for item in all_internal_items
                if item.event_date and hasattr(item.event_date, 'isoformat')
            ]
            if not event_dates:
                # ⚠ This fell back to `created_at`, with the comment
                # "document_created_at flows through here". That is only
                # sometimes true: on the vector path `created_at` is
                # `memory.extracted_at`, the minute WE ran. So the range served
                # to the client could be our own processing time, and since
                # `event_date` is now empty whenever nobody stated one, this
                # fallback fires far more often than it used to. `said_at` is
                # the same intent under a name that is true, and when nothing
                # said when the conversation happened there is no range to give.
                event_dates = [
                    item.said_at for item in all_internal_items
                    if getattr(item, 'said_at', None)
                    and hasattr(item.said_at, 'isoformat')
                ]
            if event_dates:
                payload.earliest_event_date = min(event_dates).isoformat()
                payload.latest_event_date = max(event_dates).isoformat()
            if ranked_payload is not None:
                ranked_payload.earliest_event_date = payload.earliest_event_date
                ranked_payload.latest_event_date = payload.latest_event_date
                # The by-type payload above is still built, for nothing but the
                # per-type counts the routes log. It is never sent.
                payload = ranked_payload

            # Extract conversation_context from bundle
            conv_ctx_payload = None
            conv_ctx = getattr(bundle, 'conversation_context', None)
            if conv_ctx is not None:
                conv_dict = conv_ctx.to_dict() if hasattr(conv_ctx, 'to_dict') else {}
                conv_ctx_payload = ConversationContextPayload(
                    summary=conv_dict.get("summary"),
                    current_state=conv_dict.get("current_state", {}),
                    key_extractions=conv_dict.get("key_extractions", {}),
                    recent_turns=conv_dict.get("recent_turns", []),
                    compaction_id=conv_dict.get("compaction_id"),
                    compacted_at=conv_dict.get("compacted_at"),
                    conversation_id=conv_dict.get("conversation_id"),
                )

            # Best-effort: attribute this fetch to the active turn so the
            # dashboard's Fetches panel populates for SDK traffic. Inspired
            # by playground_grpc.py:140 which does the same for playground
            # fetches via the SDK cache lookup hook.
            try:
                _record_fetch_telemetry(
                    decision=decision,
                    bundle=bundle,
                    fetch_latency_ms=fetch_latency_ms,
                    conv_ctx_present=conv_ctx_payload is not None,
                )
            except Exception:
                logger.debug("fetch telemetry record failed", exc_info=True)
            # A fetch that reached this route was not answered from the SDK's
            # anticipation cache: a miss, for the hit rate. Off the request path.
            try:
                from synap.cloud.application.client_api.retrieval_manager import _run_detached
                _miss = edition.call(
                    "anticipation_miss_record",
                    decision=decision,
                    scope=_scope_label_for_decision(decision),
                    item_count=int(getattr(bundle, "item_count", 0) or 0),
                )
                if _miss is not None:
                    _run_detached(_miss)
            except Exception:
                logger.debug("anticipation outcome record failed", exc_info=True)

            # Degradation the bundle reached, plus any leg that failed
            # underneath it (RetrievalManager folds those into warnings).
            health = None
            if degradation_level and degradation_level != "none":
                health = RetrievalHealth(
                    degraded=True,
                    level=degradation_level,
                    code=getattr(bundle, "failure_code", None),
                    warnings=list(getattr(bundle, "warnings", None) or []),
                )
            if query_cap_warning:
                # Not a failed leg, so not ``degraded`` on its own: every store
                # answered. The caller still has to hear that some of what it
                # asked for was never searched.
                if health is None:
                    health = RetrievalHealth(degraded=False, warnings=[query_cap_warning])
                else:
                    health.warnings.append(query_cap_warning)

            return payload, conv_ctx_payload, health

        except Exception as e:
            decision_id = decision.decision_id if hasattr(decision, 'decision_id') else None
            instance_id = scope.instance_id if scope else None
            set_current_span_attributes({
                "synap.error": True,
                "synap.error_type": type(e).__name__,
            })
            logger.error(
                f"RetrievalManager fetch failed, returning empty payload error_type={type(e).__name__} error_message={e} decision_id={decision_id} instance_id={instance_id}",
                exc_info=True,
            )
            # An empty payload and a 200 is indistinguishable from "this user
            # has no memories yet" unless we say which it was.
            # The class and message are on the logger.error line above. The
            # client gets only the closed-vocabulary code: a raw exception text
            # here named internal services and gave the caller nothing to do.
            _code = _failure_code(e)
            empty = (RankedContextPayload(items=[]) if response_shape == "ranked"
                     else ContextPayload())
            return empty, None, RetrievalHealth(
                degraded=True,
                level="failed",
                code=_code,
                warnings=[f"Retrieval failed ({_code})"],
            )


# ---------------------------------------------------------------------------
# C4 — conversation-summary fetch (spec §6, route-level; never RetrievalManager)
# ---------------------------------------------------------------------------

_SUMMARY_FETCH_MODE = "conversation-summary"
_SUMMARY_TTL_SECONDS = 60

# spec-mandated single batched read (§6.2): one conversation query LEFT JOINed
# to the latest compaction per conversation via DISTINCT ON (no per-conversation
# get_latest N+1). Customer-scoped (§6.2/BE-7): the same external user_id under
# two B2B customers is two different people, so customer_id is always filtered.
_SUMMARY_CONVERSATIONS_SQL = """
    SELECT
        c.conversation_id::text    AS conversation_id,
        c.external_conversation_id AS external_conversation_id,
        c.conversation_type        AS conversation_type,
        c.start_timestamp          AS started_at,
        c.end_timestamp            AS ended_at,
        c.last_message_at          AS last_message_at,
        c.message_count            AS message_count,
        c.analysis                 AS analysis,
        cc.compacted_content       AS compacted_content,
        cc.version                 AS compaction_version,
        cc.created_at              AS compacted_at
    FROM conversation c
    LEFT JOIN (
        SELECT DISTINCT ON (conversation_id)
            conversation_id, compacted_content, version, created_at
        FROM compacted_contexts
        WHERE instance_id = :iid
        ORDER BY conversation_id, version DESC
    ) cc ON cc.conversation_id = c.conversation_id
    WHERE c.instance_id = :iid
      AND c.user_id = :external_uid
      AND c.customer_id = :external_cid
      AND c.message_count > 0
    ORDER BY c.last_message_at DESC
    LIMIT :n
"""


def get_profile_store() -> ProfileStore:
    """Dependency seam (overridable in tests, mirrors get_retrieval_manager)."""
    return ProfileStore()


async def _load_maca_scoping(instance_id: str) -> Optional[dict]:
    """MACA scoping dict for scope resolution — B2C detection lives here.

    Returns ``{"user_context_isolation": ...}`` or None.
    Kept module-level so tests can monkeypatch B2C/B2B without a DB.
    """
    try:
        from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
        maca = await _load_maca_config(instance_id)
        if maca:
            return {"user_context_isolation": maca.user_context_isolation}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to load MACA scoping for summary fetch: {e}")
    return None


def _as_jsonb_dict(value: Any) -> Optional[dict]:
    """JSONB defensively — asyncpg returns dict, some drivers a str."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import json as _json
            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def _summary_from_compacted(cc: dict) -> dict:
    """Build the §6.3 ``summary`` dict from a compacted_contexts payload.

    narrative overview + current_state + key extractions — the same shape the
    in-conversation ConversationContext exposes (retrieval_manager
    _format_compacted_for_internal_use), so downstream consumers see one model.
    """
    narrative = cc.get("narrative_summary") or {}
    extractions = cc.get("structured_extractions") or {}

    def _facts(key: str):
        return extractions.get(key) or []

    return {
        "overview": narrative.get("overview", ""),
        "current_state": narrative.get("current_state", {}),
        "key_extractions": {
            "facts": _facts("explicit_facts"),
            "decisions": _facts("decisions"),
            "preferences": _facts("user_preferences"),
            "constraints": _facts("constraints"),
            "emotional_states": _facts("emotional_states"),
        },
    }


def _classification_from_compacted(cc: dict) -> dict:
    """{primary_category, subcategory, objective} from Pass-1 (spec §6.3)."""
    cls = cc.get("conversation_classification") or {}
    return {
        "primary_category": cls.get("primary_category"),
        "subcategory": cls.get("subcategory"),
        "objective": cls.get("objective"),
    }


_summary_redis_singleton = None


async def _get_summary_redis():
    """Lazily create the Redis client used to read compaction failure markers.

    Mirrors CompactionStorage._get_redis (same REDIS_URL, decode_responses=True)
    so the ``compaction_failed:{cid}`` key written by the compaction path
    (sdk_compaction.py:661-670) is read back verbatim.
    """
    global _summary_redis_singleton
    if _summary_redis_singleton is None:
        import os
        import redis.asyncio as aioredis
        _summary_redis_singleton = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
    return _summary_redis_singleton


async def _has_compaction_failed_marker(redis_client, conversation_id: str) -> bool:
    """True iff a ``compaction_failed:{cid}`` marker exists (spec §4.1/§6.3)."""
    try:
        if redis_client is None:
            redis_client = await _get_summary_redis()
        return bool(await redis_client.get(f"compaction_failed:{conversation_id}"))
    except Exception:  # pragma: no cover - Redis best-effort, never fails a read
        return False


async def fetch_conversation_summary_context(
    *,
    store: ProfileStore,
    client_id: str,
    instance_id: str,
    external_user_id: str,
    external_customer_id: Optional[str],
    include_profile: bool,
    last_n: int,
    session_factory=None,
    redis_client=None,
) -> Tuple[Optional[dict], List[ConversationSummaryPayload]]:
    """Route-level helper for conversation-summary mode (spec §6.2).

    Reads pure Postgres + Redis — no embeddings, no vector/graph, no Groq, no
    ContextDecision, no L1 cache. Returns ``(profile_doc, conversation_payloads)``.

    The profile read and the conversations query are independent (different
    tables, no data dependency) and each opens its own session, so they run
    concurrently: total wall-clock is max(profile, conversations), not the
    sum — which matters when every DB round trip crosses cloud boundaries.
    """
    async def _load_profile() -> Optional[dict]:
        if not include_profile:
            return None
        return await store.read_profile(
            client_id=client_id,
            external_user_id=external_user_id,
            external_customer_id=external_customer_id,
        )

    async def _load_conversation_rows() -> list:
        if last_n <= 0:
            return []
        from sqlalchemy import text
        params = {
            "iid": instance_id,
            "external_uid": external_user_id,
            "external_cid": external_customer_id,
            "n": last_n,
        }
        if session_factory is not None:
            # Injected factory (tests) keeps the session path.
            async with session_factory() as session:
                result = await session.execute(text(_SUMMARY_CONVERSATIONS_SQL), params)
                return result.mappings().all()

        # Production path: single-statement read on the shared AUTOCOMMIT
        # engine — no BEGIN/COMMIT wire round trips.
        from synap.cloud.database.connection import get_autocommit_engine
        async with get_autocommit_engine().connect() as conn:
            result = await conn.execute(text(_SUMMARY_CONVERSATIONS_SQL), params)
            return result.mappings().all()

    profile, rows = await asyncio.gather(_load_profile(), _load_conversation_rows())

    conversations: List[ConversationSummaryPayload] = []
    if rows:
        for row in rows:
            cid = row["conversation_id"]
            compacted = _as_jsonb_dict(row["compacted_content"])
            if compacted:
                summary_status = "available"
                summary = _summary_from_compacted(compacted)
                classification = _classification_from_compacted(compacted)
                compaction_version = row["compaction_version"]
                compacted_at = row["compacted_at"]
            else:
                # No compaction row: failed marker => "failed", else "pending".
                failed = await _has_compaction_failed_marker(redis_client, cid)
                summary_status = "failed" if failed else "pending"
                summary = None
                classification = None
                compaction_version = None
                compacted_at = None

            conversations.append(
                ConversationSummaryPayload(
                    conversation_id=cid,
                    external_conversation_id=row["external_conversation_id"],
                    conversation_type=row["conversation_type"],
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    last_message_at=row["last_message_at"],
                    message_count=row["message_count"] or 0,
                    summary_status=summary_status,
                    summary=summary,
                    classification=classification,
                    analysis=_as_jsonb_dict(row["analysis"]),
                    compaction_version=compaction_version,
                    compacted_at=compacted_at,
                )
            )

    return profile, conversations


async def _track_summary_fetch(
    scope: Optional[ScopeHierarchy],
    item_count: int,
    latency_ms: float,
    correlation_id: Optional[str],
) -> None:
    """Metering shim for summary fetches (spec §6.2 step 4).

    Borrowed from RetrievalManager._track_context_fetch: records a context-fetch
    usage event labelled ``mode='conversation-summary'`` (never fast/accurate) so
    the fetch is billed/observed as its own thing. Self-contained; never raises
    into the read path — dispatched fire-and-forget by the route.
    """
    if scope is None:
        return
    try:
        tracker = get_usage_tracker()
        if tracker is None:
            return
        await tracker.track_context_fetch(
            client_id=str(scope.client_id),
            instance_id=str(scope.instance_id),
            mode=_SUMMARY_FETCH_MODE,
            latency_ms=int(latency_ms or 0),
            item_count=int(item_count or 0),
            cache_hit=False,
            user_id=str(scope.user_id) if scope.user_id else None,
            conversation_id=None,
            correlation_id=correlation_id,
        )
        logger.info(
            "cloud_usage_track_summary_fetch context_mode=%s mode=%s item_count=%s "
            "client_id=%s instance_id=%s correlation_id=%s",
            _SUMMARY_FETCH_MODE,
            _SUMMARY_FETCH_MODE,
            item_count,
            scope.client_id,
            scope.instance_id,
            correlation_id,
        )
    except Exception as e:  # pragma: no cover - metering best-effort
        logger.warning(
            "cloud_usage_track_summary_fetch_failed client_id=%s instance_id=%s error=%s",
            getattr(scope, "client_id", None),
            getattr(scope, "instance_id", None),
            e,
        )


def _dispatch_summary_metering(
    scope: Optional[ScopeHierarchy],
    item_count: int,
    latency_ms: float,
    correlation_id: Optional[str],
) -> None:
    """Fire the metering shim detached (same pattern as _track_context_fetch)."""
    try:
        from synap.cloud.application.client_api.retrieval_manager import _run_detached
        _run_detached(_track_summary_fetch(scope, item_count, latency_ms, correlation_id))
    except Exception:  # pragma: no cover - metering never fails a read
        logger.debug("summary metering dispatch failed", exc_info=True)


def _reject_summary_mode(request: ContextFetchRequest) -> None:
    """Handler-level guard for the three non-user routes (spec §6.1).

    ``ContextFetchRequest`` is shared across all four fetch routes, so model
    validation cannot carry the per-route rule; each non-user handler calls
    this first (before its blanket try/except, so the 422 isn't masked to 500).
    """
    if request.context_mode == _SUMMARY_FETCH_MODE:
        raise HTTPException(
            status_code=422,
            detail=(
                "context_mode='conversation-summary' is only supported on the "
                "user-scope route (/v1/context/user/fetch)."
            ),
        )


async def _fetch_user_summary(
    request: ContextFetchRequest,
    auth: SDKAuthContext,
    store: ProfileStore,
) -> ContextFetchResponse:
    """conversation-summary branch of fetch_user_context (spec §6.2).

    Runs BEFORE any ContextDecision construction and never touches
    RetrievalManager — no return channel, no L1-cache collision, no mislabelled
    metering (all three documented in §6.2). HTTPExceptions raised here are
    intentionally OUTSIDE the caller's blanket try/except so 400/422 aren't
    masked to 500.
    """
    if not request.user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required for context_mode='conversation-summary'.",
        )

    maca_scoping = await _load_maca_scoping(auth.instance_id)
    is_b2c = bool(
        maca_scoping
        and maca_scoping.get("user_context_isolation") == "equals_customer"
    )

    # Resolve scope exactly as the existing path does. B2B without customer_id
    # raises ValueError -> 400 (spec §6.2); B2C collapses customer_id = user_id.
    try:
        scope = await _build_scope(
            auth,
            user_id=request.user_id,
            customer_id=request.customer_id,
            maca_scoping=maca_scoping,
            scope_path=request.scope,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # The conversation table stores raw external ids (both writer paths). B2C
    # collapses the customer scope onto the user id; B2B uses the (now-required)
    # customer_id. Same function the writers call, so the id looked up here is
    # by construction the id they stored.
    _, external_customer_id = collapse_b2c_identity(
        request.user_id, request.customer_id, is_b2c=is_b2c,
        where="sdk_context._fetch_user_summary",
    )

    t0 = time.monotonic()
    profile, conversations = await fetch_conversation_summary_context(
        store=store,
        client_id=auth.client_id,
        instance_id=auth.instance_id,
        external_user_id=request.user_id,
        external_customer_id=external_customer_id,
        include_profile=request.include_profile,
        last_n=request.last_n_conversations,
    )
    latency_ms = round((time.monotonic() - t0) * 1000, 2)

    # Metering: detached, mode='conversation-summary' (spec §6.2 step 4).
    _dispatch_summary_metering(scope, len(conversations), latency_ms, auth.correlation_id)

    user_log_id = _uuid.uuid5(_uuid.NAMESPACE_URL, request.user_id)
    logger.info(
        "summary_fetch_complete context_mode=%s client_id=%s instance_id=%s "
        "user_uuid5=%s is_b2c=%s profile_hit=%s n_returned=%s latency_ms=%s "
        "correlation_id=%s",
        _SUMMARY_FETCH_MODE,
        auth.client_id,
        auth.instance_id,
        user_log_id,
        is_b2c,
        profile is not None,
        len(conversations),
        latency_ms,
        auth.correlation_id,
    )

    # In summary mode the five item lists are empty and conversation_context is
    # None; ttl_seconds is the 60s freshness hint (spec §6.2/§6.3).
    return ContextFetchResponse(
        context=ContextPayload(),
        conversation_context=None,
        ttl_seconds=_SUMMARY_TTL_SECONDS,
        profile=profile,
        conversations=conversations,
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

_retrieval_manager_singleton: Optional[RetrievalManager] = None

def get_retrieval_manager() -> RetrievalManager:
    """Get singleton RetrievalManager with all available providers.

    Returns the same instance across all requests so that the L2 in-memory
    adjacency cache persists between requests (the whole point of L2).

    Note: ScopeResolver is not needed here — scope is pre-built by
    endpoint handlers using _build_scope() with ingestion-matching UUIDs.

    Returns:
        RetrievalManager instance (singleton)
    """
    global _retrieval_manager_singleton
    if _retrieval_manager_singleton is not None:
        return _retrieval_manager_singleton

    providers = get_cached_providers()

    _retrieval_manager_singleton = RetrievalManager(
        cache_provider=providers["redis"],
        vector_provider=providers["vector"],
        graph_provider=providers["graph"],
        postgres_provider=providers["postgres"],
        embedding_client=providers.get("embedding"),
        llm_client=providers.get("llm"),
        compaction_storage=providers.get("compaction_storage"),
        compaction_manager=providers.get("compaction_manager"),
    )
    return _retrieval_manager_singleton


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_TYPES = ["facts", "preferences", "episodes", "emotions", "temporal_events"]


@router.post("/conversation/fetch", response_model=ContextFetchResponse, dependencies=[Depends(require_instance_ready), Depends(enforce_retrieval_rate_limit)])
async def fetch_conversation_context(
    request: ContextFetchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    retrieval_manager: RetrievalManager = Depends(get_retrieval_manager),
):
    """Fetch context for a specific conversation.

    Uses RetrievalManager for:
    - Multi-layer caching (L1/L2/L3)
    - Compacted context integration
    - Hybrid vector+graph retrieval (ACCURATE mode)
    - 4-check filtering
    - Token budget enforcement
    """
    # conversation-summary mode is user-scope only (spec §6.1). Reject before
    # the blanket try/except below so the 422 isn't masked to 500.
    _reject_summary_mode(request)

    from synap.cloud.shared.logging_utils import log_data_sample, truncate_string

    _handler_entry_ts = time.monotonic()

    # Step logging for API request
    step_logger = StepLogger("API.fetch_conversation_context")
    step_logger.log_request(
        payload={
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "customer_id": request.customer_id,
            "search_query": request.search_query,
            "max_results": request.max_results,
            "types": request.types,
            "mode": request.mode,
            "auth": {
                "client_id": auth.client_id,
                "instance_id": auth.instance_id,
            },
        },
        correlation_id=None,
    )

    search_query_preview = [truncate_string(q, 100) for q in request.search_query] if request.search_query else None
    logger.info(
        f"fetch_conversation_context_start conversation_id={request.conversation_id} user_id={request.user_id} customer_id={request.customer_id} mode={request.mode} types={request.types} search_query={search_query_preview} max_results={request.max_results}",
    )

    try:
        if not request.conversation_id:
            return _empty_fetch_response(request)

        # Auto-resolve user_id, customer_id and the RUNG from the conversation
        # when the caller did not send them.
        #
        # The rung is looked up even when both ids ARE supplied, and that is
        # deliberate. A caller who opened the conversation at a practice and
        # then fetches with just their two ids used to get the whole tenant
        # back: the write named a rung, the read could not, and neither end
        # raised. Sending both ids is not a statement that the rung does not
        # matter, it is the ordinary shape of a fetch.
        resolved_user_id = request.user_id
        resolved_customer_id = request.customer_id
        resolved_scope_path = request.scope
        if not resolved_user_id or not resolved_customer_id or not resolved_scope_path:
            conv_user_id, conv_customer_id, conv_scope_path = await _resolve_scope_from_conversation(request.conversation_id, auth.instance_id)
            # An explicit path on the request always wins. It is an
            # instruction, and the stored one is a default.
            if not resolved_scope_path and conv_scope_path:
                resolved_scope_path = conv_scope_path
                logger.info(
                    f"Auto-resolved scope path {sorted(conv_scope_path)} from "
                    f"conversation_id={request.conversation_id}"
                )
            if not resolved_user_id and conv_user_id:
                resolved_user_id = conv_user_id
                logger.info(
                    f"Auto-resolved user_id={resolved_user_id} from "
                    f"conversation_id={request.conversation_id}"
                )
            if not resolved_customer_id and conv_customer_id:
                resolved_customer_id = conv_customer_id
                logger.info(
                    f"Auto-resolved customer_id={resolved_customer_id} from "
                    f"conversation_id={request.conversation_id}"
                )
            if not resolved_user_id:
                logger.warning(
                    f"Could not resolve user_id for conversation_id={request.conversation_id}, "
                    "retrieval may return limited results"
                )

        # Load MACA config to detect B2C scoping (same as fetch_user_context)
        maca_scoping = None
        try:
            from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
            maca = await _load_maca_config(auth.instance_id)
            if maca:
                maca_scoping = {"user_context_isolation": maca.user_context_isolation}
        except Exception as e:
            logger.warning(f"Failed to load MACA scoping for conversation context: {e}")

        try:
            await _b2c_check(auth, request.customer_id,
                             where="POST /v1/context/conversation/fetch",
                             user_id=request.user_id)
            scope = await _build_scope(auth, user_id=resolved_user_id, customer_id=resolved_customer_id, maca_scoping=maca_scoping, scope_path=resolved_scope_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        context_types = (
            request.types if "all" not in request.types
            else DEFAULT_CONTEXT_TYPES
        )

        decision = ContextDecision(
            decision_id=str(_uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=resolved_user_id,
            customer_id=resolved_customer_id,
            conversation_id=request.conversation_id,
            query_date=_parse_query_date(getattr(request, "query_date", None)),
            retrieval_mode=RetrievalMode.FAST if request.mode == "fast" else RetrievalMode.ACCURATE,
            trigger_payload={
                "skip_subquery_gen": request.skip_subquery_gen,
                "skip_llm_filter": request.skip_llm_filter,
                "precision_level": request.precision_level,
                "include_source_evidence": request.include_source_evidence,
            },
            context_types=context_types,
            search_queries=request.search_query or [],
            max_results=request.max_results,
            max_results_per_query=_per_query_share(request.max_results_per_query),
            token_budget=derive_token_budget(
                request.max_results, request.include_source_evidence
            ),
            include_conversation_context=request.include_conversation_context,
        )

        logger.info(
            f"pre_retrieval_manager_call elapsed_since_handler_entry_ms={round((time.monotonic() - _handler_entry_ts) * 1000, 1)} user_id_resolved={resolved_user_id} conversation_id={request.conversation_id} instance_id={auth.instance_id}",
        )
        payload, conv_ctx, retrieval_health = await _fetch_via_retrieval_manager(
            retrieval_manager, decision, scope, response_shape=request.response_shape)

        # Calculate total items from all context types
        total_items = (
            len(payload.facts) +
            len(payload.preferences) +
            len(payload.episodes) +
            len(payload.emotions) +
            len(payload.temporal_events)
        ) if payload else 0

        # Collect all items for preview
        all_items = []
        if payload:
            all_items.extend([{"type": "fact", "content": truncate_string(item.content, 80), "confidence": item.confidence} for item in payload.facts[:2]])
            all_items.extend([{"type": "pref", "content": truncate_string(item.content, 80), "confidence": item.confidence} for item in payload.preferences[:2]])
            all_items.extend([{"type": "episode", "content": truncate_string(item.content, 80), "confidence": item.confidence} for item in payload.episodes[:2]])

        items_preview = log_data_sample(all_items, sample_size=6, label="context_items") if all_items else None
        logger.info(
            f"fetch_conversation_context_complete conversation_id={request.conversation_id} result_count={total_items} facts_count={len(payload.facts) if payload else 0} preferences_count={len(payload.preferences) if payload else 0} episodes_count={len(payload.episodes) if payload else 0} emotions_count={len(payload.emotions) if payload else 0} temporal_events_count={len(payload.temporal_events) if payload else 0} has_conversation_context={conv_ctx is not None} items_preview={items_preview}",
        )

        # Step logging for API response
        step_logger.log_response(
            response={
                "total_items": total_items,
                "facts_count": len(payload.facts) if payload else 0,
                "preferences_count": len(payload.preferences) if payload else 0,
                "episodes_count": len(payload.episodes) if payload else 0,
                "emotions_count": len(payload.emotions) if payload else 0,
                "temporal_events_count": len(payload.temporal_events) if payload else 0,
                "has_conversation_context": conv_ctx is not None,
                "facts": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.facts[:3] if payload else [])
                ],
                "preferences": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.preferences[:3] if payload else [])
                ],
                "episodes": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.episodes[:3] if payload else [])
                ],
                "ttl_seconds": 300,
            },
            status_code=200,
            success=True,
        )

        return ContextFetchResponse(context=payload, conversation_context=conv_ctx, ttl_seconds=300, retrieval=retrieval_health)

    except HTTPException:
        # Let a deliberate status through. The handler below catches Exception,
        # and HTTPException is an Exception, so every 4xx these routes raise on
        # purpose was being re-wrapped as `500 Retrieval failed: HTTPException`.
        # An audit reported a rejected `scope` path returning
        # 500 instead of the promised 400, and the exception class was only half
        # the reason. The other half is this: the B2B "customer_id is required"
        # 400 was reaching callers as a 500 as well.
        raise
    except Exception as e:
        logger.exception(f"Error fetching conversation context: {e}")

        # Step logging for error response
        step_logger.log_response(
            response={"error": str(e)},
            status_code=500,
            success=False,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}")


@router.post("/user/fetch", response_model=ContextFetchResponse, dependencies=[Depends(require_instance_ready), Depends(enforce_retrieval_rate_limit)])
async def fetch_user_context(
    request: ContextFetchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    retrieval_manager: RetrievalManager = Depends(get_retrieval_manager),
    store: ProfileStore = Depends(get_profile_store),
):
    """Fetch context for a specific user.

    In the default ``context_mode="in-conversation"`` this uses RetrievalManager
    for hybrid retrieval with caching and filtering. In
    ``context_mode="conversation-summary"`` a route-level branch (spec §6.2)
    returns the caller profile + last-N conversation summaries and NEVER enters
    RetrievalManager.
    """
    from synap.cloud.shared.logging_utils import log_data_sample, truncate_string

    # Step logging for API request
    step_logger = StepLogger("API.fetch_user_context")
    step_logger.log_request(
        payload={
            "user_id": request.user_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "search_query": request.search_query,
            "max_results": request.max_results,
            "types": request.types,
            "mode": request.mode,
            "context_mode": request.context_mode,
            "auth": {
                "client_id": auth.client_id,
                "instance_id": auth.instance_id,
            },
        },
        correlation_id=None,
    )

    # C4 — conversation-summary branch runs BEFORE the blanket try/except (so
    # its 400/422 aren't masked to 500) and BEFORE any ContextDecision, so it
    # never reaches RetrievalManager (spec §6.2).
    if request.context_mode == _SUMMARY_FETCH_MODE:
        # Checked here as well as below: this branch returns before the block
        # that guards the in-conversation path, and it is a branch real
        # callers use, so leaving it unguarded would exempt a busy caller.
        await _b2c_check(auth, request.customer_id,
                         where="POST /v1/context/user/fetch (conversation-summary)",
                         user_id=request.user_id)
        return await _fetch_user_summary(request, auth, store)

    await _b2c_check(auth, request.customer_id, where="POST /v1/context/user/fetch",
                     user_id=request.user_id)

    try:
        if not request.user_id:
            return _empty_fetch_response(request)

        # Load MACA config to detect B2C scoping
        maca_scoping = None
        try:
            from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
            maca = await _load_maca_config(auth.instance_id)
            if maca:
                maca_scoping = {"user_context_isolation": maca.user_context_isolation}
        except Exception as e:
            logger.warning(f"Failed to load MACA scoping for scope build: {e}")

        try:
            scope = await _build_scope(auth, user_id=request.user_id, customer_id=request.customer_id, maca_scoping=maca_scoping, scope_path=request.scope)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        context_types = (
            request.types if "all" not in request.types
            else DEFAULT_CONTEXT_TYPES
        )

        decision = ContextDecision(
            decision_id=str(_uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=request.user_id,
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            query_date=_parse_query_date(getattr(request, "query_date", None)),
            retrieval_mode=RetrievalMode.FAST if request.mode == "fast" else RetrievalMode.ACCURATE,
            trigger_payload={
                "skip_subquery_gen": request.skip_subquery_gen,
                "skip_llm_filter": request.skip_llm_filter,
                "precision_level": request.precision_level,
                "include_source_evidence": request.include_source_evidence,
            },
            context_types=context_types,
            search_queries=request.search_query or [],
            max_results=request.max_results,
            max_results_per_query=_per_query_share(request.max_results_per_query),
            token_budget=derive_token_budget(
                request.max_results, request.include_source_evidence
            ),
            include_conversation_context=request.include_conversation_context,
        )

        payload, conv_ctx, retrieval_health = await _fetch_via_retrieval_manager(
            retrieval_manager, decision, scope, response_shape=request.response_shape)

        # Calculate total items
        total_items = (
            len(payload.facts) +
            len(payload.preferences) +
            len(payload.episodes) +
            len(payload.emotions) +
            len(payload.temporal_events)
        ) if payload else 0

        # Step logging for API response
        step_logger.log_response(
            response={
                "total_items": total_items,
                "facts_count": len(payload.facts) if payload else 0,
                "preferences_count": len(payload.preferences) if payload else 0,
                "episodes_count": len(payload.episodes) if payload else 0,
                "emotions_count": len(payload.emotions) if payload else 0,
                "temporal_events_count": len(payload.temporal_events) if payload else 0,
                "has_conversation_context": conv_ctx is not None,
                "facts": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.facts[:3] if payload else [])
                ],
                "ttl_seconds": 300,
            },
            status_code=200,
            success=True,
        )

        return ContextFetchResponse(context=payload, conversation_context=conv_ctx, ttl_seconds=300, retrieval=retrieval_health)

    except HTTPException:
        # Let a deliberate status through. The handler below catches Exception,
        # and HTTPException is an Exception, so every 4xx these routes raise on
        # purpose was being re-wrapped as `500 Retrieval failed: HTTPException`.
        # An audit reported a rejected `scope` path returning
        # 500 instead of the promised 400, and the exception class was only half
        # the reason. The other half is this: the B2B "customer_id is required"
        # 400 was reaching callers as a 500 as well.
        raise
    except Exception as e:
        logger.exception(f"Error fetching user context: {e}")

        # Step logging for error response
        step_logger.log_response(
            response={"error": str(e)},
            status_code=500,
            success=False,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}")


@router.post("/customer/fetch", response_model=ContextFetchResponse, dependencies=[Depends(require_instance_ready), Depends(enforce_retrieval_rate_limit)])
async def fetch_customer_context(
    request: ContextFetchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    retrieval_manager: RetrievalManager = Depends(get_retrieval_manager),
):
    """Fetch context for a specific customer.

    Uses RetrievalManager for hybrid retrieval with caching and filtering.

    This does NOT aggregate all of the customer's users' memories. It reads
    tenant-wide shared material — knowledge stored with no owning user — and
    nothing else, unless the instance has `allow_cross_user_retrieval` on. See
    `customer_scope_is_shared_material_only` for why, and for the case
    this docstring used to misdescribe.
    """
    # conversation-summary mode is user-scope only (spec §6.1).
    _reject_summary_mode(request)

    from synap.cloud.shared.logging_utils import truncate_string

    step_logger = StepLogger("API.fetch_customer_context")
    step_logger.log_request(
        payload={
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "search_query": request.search_query,
            "max_results": request.max_results,
            "types": request.types,
            "mode": request.mode,
            "auth": {
                "client_id": auth.client_id,
                "instance_id": auth.instance_id,
            },
        },
        correlation_id=None,
    )

    try:
        if not request.customer_id:
            return _empty_fetch_response(request)

        # This was "the one context route that does not load MACA before
        # resolving scope", which `b2c_contract.reject_customer_route` records
        # as the reason the route returns nothing on a B2C instance. Resolving
        # as B2B on a B2C instance is not a safety property, it is an accident
        # that happened to be safe while the only filter was the legacy one. It
        # stopped being safe when the nested arm arrived. Loading it here means
        # the chain knows the tenant rung is also the identity rung, which is
        # what the guard below reads.
        maca_scoping = None
        maca = None
        try:
            from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
            maca = await _load_maca_config(auth.instance_id)
            if maca:
                maca_scoping = {"user_context_isolation": maca.user_context_isolation}
        except Exception as e:
            logger.warning(f"Failed to load MACA scoping for scope build: {e}")

        try:
            await _b2c_reject_customer_route(
                auth, where="POST /v1/context/customer/fetch")
            # G-03. The caller's own path, carried like every other context
            # route carries it. It used to be dropped here, and because `scope`
            # lives on the shared request model the route ACCEPTED it and threw
            # it away: measured on a four-rung ladder, a caller saying
            # {"customer":"acme","team":"payments"} was resolved to
            # [client, acme], the rung ABOVE the one they named. The field's
            # contract is that it is honoured exactly or refused, and dropping
            # it is neither.
            scope = await _build_scope(
                auth, customer_id=request.customer_id, maca_scoping=maca_scoping,
                scope_path=request.scope)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # A customer-scope read must not resolve to a person. See
        # `customer_scope_reaches_an_identity` for the two ways it could, and
        # for the measurement that showed the B2C one was live.
        refusal = customer_scope_reaches_an_identity(getattr(scope, "chain_rungs", None))
        if refusal:
            raise HTTPException(status_code=400, detail=refusal)

        context_types = (
            request.types if "all" not in request.types
            else DEFAULT_CONTEXT_TYPES
        )

        decision = ContextDecision(
            decision_id=str(_uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=None,
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            query_date=_parse_query_date(getattr(request, "query_date", None)),
            retrieval_mode=RetrievalMode.FAST if request.mode == "fast" else RetrievalMode.ACCURATE,
            trigger_payload={
                "skip_subquery_gen": request.skip_subquery_gen,
                "skip_llm_filter": request.skip_llm_filter,
                "precision_level": request.precision_level,
                "include_source_evidence": request.include_source_evidence,
            },
            context_types=context_types,
            search_queries=request.search_query or [],
            max_results=request.max_results,
            max_results_per_query=_per_query_share(request.max_results_per_query),
            token_budget=derive_token_budget(
                request.max_results, request.include_source_evidence
            ),
            include_conversation_context=request.include_conversation_context,
        )

        payload, conv_ctx, retrieval_health = await _fetch_via_retrieval_manager(
            retrieval_manager, decision, scope, response_shape=request.response_shape)

        total_items = (
            len(payload.facts) +
            len(payload.preferences) +
            len(payload.episodes) +
            len(payload.emotions) +
            len(payload.temporal_events)
        ) if payload else 0

        # See `customer_scope_is_shared_material_only`. A caller who sent no
        # user_id (every real call to this route today, since the field isn't
        # even accepted here) can only ever be reading the tenant-shared,
        # no-owner bucket unless cross-user retrieval is on. When that's true
        # and the fetch came back empty, say so instead of handing back an
        # indistinguishable, silent 200 — which is exactly what let a client
        # call this route thousands of times for nothing.
        shared_only = customer_scope_is_shared_material_only(
            user_id=None,
            allow_cross_user_retrieval=bool(maca and maca.allow_cross_user_retrieval),
        )
        scope_note = None
        if total_items == 0 and shared_only:
            scope_note = (
                "This customer-scope fetch found no tenant-wide shared knowledge "
                "for this customer, and by design it does not aggregate "
                "individual users' own memories (see the route docstring). If "
                "you want that user's own history, call POST "
                "/v1/context/user/fetch with their user_id. If you want an "
                "aggregate view across every user under this customer, ask "
                "Synap to enable cross-user retrieval for this instance."
            )
            logger.warning(
                "customer_fetch_empty_shared_only_scope client_id=%s "
                "instance_id=%s customer_id=%s",
                auth.client_id, auth.instance_id, request.customer_id,
            )

        step_logger.log_response(
            response={
                "total_items": total_items,
                "facts_count": len(payload.facts) if payload else 0,
                "preferences_count": len(payload.preferences) if payload else 0,
                "episodes_count": len(payload.episodes) if payload else 0,
                "emotions_count": len(payload.emotions) if payload else 0,
                "temporal_events_count": len(payload.temporal_events) if payload else 0,
                "has_conversation_context": conv_ctx is not None,
                "facts": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.facts[:3] if payload else [])
                ],
                "ttl_seconds": 300,
                "scope_note": scope_note,
            },
            status_code=200,
            success=True,
        )

        return ContextFetchResponse(
            context=payload, conversation_context=conv_ctx, ttl_seconds=300,
            scope_note=scope_note, retrieval=retrieval_health,
        )

    except HTTPException:
        # Let a deliberate status through. The handler below catches Exception,
        # and HTTPException is an Exception, so every 4xx these routes raise on
        # purpose was being re-wrapped as `500 Retrieval failed: HTTPException`.
        # An audit reported a rejected `scope` path returning
        # 500 instead of the promised 400, and the exception class was only half
        # the reason. The other half is this: the B2B "customer_id is required"
        # 400 was reaching callers as a 500 as well.
        raise
    except Exception as e:
        logger.exception(f"Error fetching customer context: {e}")

        step_logger.log_response(
            response={"error": str(e)},
            status_code=500,
            success=False,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}")


@router.post("/client/fetch", response_model=ContextFetchResponse, dependencies=[Depends(require_instance_ready), Depends(enforce_retrieval_rate_limit)])
async def fetch_client_context(
    request: ContextFetchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    retrieval_manager: RetrievalManager = Depends(get_retrieval_manager),
):
    """Fetch context for the authenticated client (org-level).

    Uses RetrievalManager for hybrid retrieval with caching and filtering.
    Client-level fetch aggregates all memories across all customers and users.
    """
    # conversation-summary mode is user-scope only (spec §6.1).
    _reject_summary_mode(request)

    from synap.cloud.shared.logging_utils import truncate_string

    step_logger = StepLogger("API.fetch_client_context")
    step_logger.log_request(
        payload={
            "conversation_id": request.conversation_id,
            "search_query": request.search_query,
            "max_results": request.max_results,
            "types": request.types,
            "mode": request.mode,
            "auth": {
                "client_id": auth.client_id,
                "instance_id": auth.instance_id,
            },
        },
        correlation_id=None,
    )

    try:
        # Client-level: explicitly pass customer_id=None to avoid defaulting
        try:
            # This route ignores customer_id entirely (client scope has no
            # customer), but it still ACCEPTS one, because it shares
            # ContextFetchRequest. A caller sending it is misunderstanding the
            # contract, and silently discarding the field is how that
            # misunderstanding survives.
            await _b2c_check(auth, request.customer_id,
                             where="POST /v1/context/client/fetch",
                             user_id=request.user_id)
            # G-03, same reasoning as the customer route. Dropped here it was
            # worse than on that route: with no customer id either, the request
            # resolved NO chain at all, so a caller who said which rung they
            # were standing at had their position ignored completely and the
            # nested arm never applied. `customer_id` stays None because client
            # scope is the root; the path says where the caller is, not what
            # they are asking for.
            scope = await _build_scope(auth, customer_id=None, scope_path=request.scope)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        context_types = (
            request.types if "all" not in request.types
            else DEFAULT_CONTEXT_TYPES
        )

        decision = ContextDecision(
            decision_id=str(_uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            user_id=None,
            customer_id=None,
            conversation_id=request.conversation_id,
            query_date=_parse_query_date(getattr(request, "query_date", None)),
            retrieval_mode=RetrievalMode.FAST if request.mode == "fast" else RetrievalMode.ACCURATE,
            trigger_payload={
                "skip_subquery_gen": request.skip_subquery_gen,
                "skip_llm_filter": request.skip_llm_filter,
                "precision_level": request.precision_level,
                "include_source_evidence": request.include_source_evidence,
            },
            context_types=context_types,
            search_queries=request.search_query or [],
            max_results=request.max_results,
            max_results_per_query=_per_query_share(request.max_results_per_query),
            token_budget=derive_token_budget(
                request.max_results, request.include_source_evidence
            ),
            include_conversation_context=request.include_conversation_context,
        )

        payload, conv_ctx, retrieval_health = await _fetch_via_retrieval_manager(
            retrieval_manager, decision, scope, response_shape=request.response_shape)

        total_items = (
            len(payload.facts) +
            len(payload.preferences) +
            len(payload.episodes) +
            len(payload.emotions) +
            len(payload.temporal_events)
        ) if payload else 0

        step_logger.log_response(
            response={
                "total_items": total_items,
                "facts_count": len(payload.facts) if payload else 0,
                "preferences_count": len(payload.preferences) if payload else 0,
                "episodes_count": len(payload.episodes) if payload else 0,
                "emotions_count": len(payload.emotions) if payload else 0,
                "temporal_events_count": len(payload.temporal_events) if payload else 0,
                "has_conversation_context": conv_ctx is not None,
                "facts": [
                    {
                        "id": item.id,
                        "content": truncate_string(item.content, 100),
                        "confidence": item.confidence,
                    }
                    for item in (payload.facts[:3] if payload else [])
                ],
                "ttl_seconds": 1800,
            },
            status_code=200,
            success=True,
        )

        return ContextFetchResponse(context=payload, conversation_context=conv_ctx, ttl_seconds=1800, retrieval=retrieval_health)  # 30min TTL for client-level

    except HTTPException:
        # Let a deliberate status through. The handler below catches Exception,
        # and HTTPException is an Exception, so every 4xx these routes raise on
        # purpose was being re-wrapped as `500 Retrieval failed: HTTPException`.
        # An audit reported a rejected `scope` path returning
        # 500 instead of the promised 400, and the exception class was only half
        # the reason. The other half is this: the B2B "customer_id is required"
        # 400 was reaching callers as a 500 as well.
        raise
    except Exception as e:
        logger.exception(f"Error fetching client context: {e}")

        step_logger.log_response(
            response={"error": str(e)},
            status_code=500,
            success=False,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}")
