"""Stage 3: Memory Extraction."""

import asyncio
import json
from contextvars import ContextVar
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import logging
import re

from synap.cloud.agents.ingestion.interfaces import EntityExtractor, LLMClient
from synap.cloud.agents.ingestion.models import (
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelationship,
    MemoryDestination,
    MemoryType,
    Stage2Output,
    Stage3Output,
)
from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import TURN_PATTERN
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.shared.logging_utils import truncate_string
from synap.cloud.shared.tokens import get_token_counter
from synap.cloud.observability.tracing import start_span, set_current_span_attributes

logger = logging.getLogger(__name__)

from synap.cloud.agents.ingestion.prompts.extraction import assistant_recall
from synap.cloud.agents.ingestion.services import (
    coalesce, custom_categories, date_prepass, guardrails, prior_context, sweep_repair,
)
from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
    PROHIBITED_CONTENT, REQUIRE_ENTITY_GROUNDING,
)
from synap.cloud.agents.ingestion.prompts.stage3_extraction import (
    SPEAKER_ATTRIBUTION_ADDENDUM as _SPEAKER_ATTRIBUTION_ADDENDUM,
    CUSTOMER_WE_SECTION,
    ANALYSIS_HINTS_HEADER as _ANALYSIS_HINTS_HEADER,
)


# --------------------------------------------------------------------------
# MACA-steered extraction
# --------------------------------------------------------------------------
# Every extractor is built ONCE at worker startup with a blank default config.
# The worker loaded each instance's real MACA config per document, used it to
# choose which memory types run, and never passed it on, so every field meant
# to steer the prompt (focus areas, avoid list, category rationale and examples,
# priority, aggressiveness, industry, key entities and relationships, graph
# schemas, role semantics) rendered from the blank config: none of the instance-only phrases reached
# the prompts until the config was passed.
#
# The config for the document being extracted right now lives in a ContextVar,
# not on the shared extractor: one extractor serves concurrent documents from
# different instances, and every task created while a call is running (the
# per-type fan-out, the entity task) inherits that call's value.
_ACTIVE_MACA_CONFIG: ContextVar[Optional[MemoryArchitectureConfig]] = ContextVar(
    "stage3_active_maca_config", default=None)


def maca_extraction_instances() -> Set[str]:
    """Instances whose extraction prompts are steered by their own MACA config.

    ``SYNAP_MACA_EXTRACTION_INSTANCES`` is a comma list of instance UUIDs (the
    ingestion Scope carries the UUID, not the ``inst_...`` id), or ``*`` for all.
    Empty or unset means off everywhere: MACA configs are LLM-generated and have
    been wrong before, so switching a client onto its own config is a decision,
    not a side effect of deploying this.
    """
    raw = os.environ.get("SYNAP_MACA_EXTRACTION_INSTANCES", "") or ""
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def maca_extraction_enabled_for(instance_id: Optional[object]) -> bool:
    allow = maca_extraction_instances()
    if not allow:
        return False
    if "*" in allow:
        return True
    return instance_id is not None and str(instance_id).strip().lower() in allow


def _instance_in_allowlist(raw: str, instance_id: Optional[object]) -> bool:
    """Match an instance against a comma list holding either id form."""
    allowed = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        allowed.add(item)
        try:
            from synap.cloud.shared.scope import parse_or_deterministic_uuid
            allowed.add(str(parse_or_deterministic_uuid(item)))
        except Exception:
            pass
    return instance_id is not None and str(instance_id).strip() in allowed


def graph_extraction_enabled_for(instance_id: Optional[object]) -> bool:
    """Whether to spend calls on entities and relationships for this instance.

    Entity and relationship extraction is two extra LLM calls per document on the
    baseline and hybrid paths, one on hybrid_graph. The only thing that reads
    what they produce is graph traversal at retrieval time, and that is gated by
    ENABLE_GRAPH_TRAVERSAL plus GRAPH_TRAVERSAL_INSTANCE_ALLOWLIST, which in practice
    names only a few instances. Every other client pays for a graph nobody reads.

    Same idiom as the read side: ENABLE_GRAPH_EXTRACTION is the global switch
    (default true, so deploying this changes nothing) and
    GRAPH_EXTRACTION_INSTANCE_ALLOWLIST narrows it or, when the global switch is
    off, re-admits named instances. Keep this allowlist in step with the
    traversal one: an instance that traverses the graph must keep filling it.
    """
    allow = os.environ.get("GRAPH_EXTRACTION_INSTANCE_ALLOWLIST", "").strip()
    globally_on = os.environ.get("ENABLE_GRAPH_EXTRACTION", "true").strip().lower() in (
        "true", "1", "yes")
    if not globally_on:
        return bool(allow) and _instance_in_allowlist(allow, instance_id)
    if not allow:
        return True
    return _instance_in_allowlist(allow, instance_id)


def recall_in_baseline_enabled() -> bool:
    """Run the assistant-recall call alongside the five baseline calls.

    Off by default: it is an extra LLM call per document and it re-extracts user
    facts that the per-type calls already found, so it trades duplicates for the
    assistant content the five calls are told to ignore.
    """
    return os.environ.get("SYNAP_ASSISTANT_RECALL_IN_BASELINE", "").strip().lower() in (
        "true", "1", "yes")


def _without_avoid(config: MemoryArchitectureConfig) -> MemoryArchitectureConfig:
    """The same config with the ingestion avoid list emptied.

    Used only on the assistant-recall path. That profile's system prompt exists
    to store what the ASSISTANT said, while a MACA avoid list can say the
    opposite (the LongMemEval instance's includes "Agent/assistant responses
    that do not contain user-stated facts"). Sending both in one call hands the
    model two contradictory instructions; the profile defines its own scope, so
    its call omits the avoid list and keeps everything else.
    """
    ingestion = config.agent_hints.ingestion_agent.model_copy(update={"avoid": []})
    agent_hints = config.agent_hints.model_copy(update={"ingestion_agent": ingestion})
    return config.model_copy(update={"agent_hints": agent_hints})


# --------------------------------------------------------------------------
# Extraction mode resolution
# --------------------------------------------------------------------------
_VALID_EXTRACTION_MODES = ("baseline", "hybrid", "hybrid_graph")


def _instance_mode_overrides() -> Dict[str, str]:
    """``STAGE3_EXTRACTION_MODE_BY_INSTANCE`` parsed into {instance_id: mode}.

    Format: ``id=mode,id=mode``. Ids are matched case-insensitively and
    trimmed, so a copy-pasted list with spaces or mixed case still binds.
    """
    raw = os.environ.get("STAGE3_EXTRACTION_MODE_BY_INSTANCE", "") or ""
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip().lower(), value.strip().lower()
        if key and value:
            out[key] = value
    return out


def resolve_extraction_mode(instance_id: Optional[object]) -> str:
    """The Stage-3 mode for *instance_id*.

    Precedence: a per-instance override, then the fleet-wide
    ``STAGE3_EXTRACTION_MODE``, then ``baseline``.

    An unrecognised mode NEVER silently becomes something else: it is logged
    and the fleet default is used, because the failure this guards against is
    a typo in an env var quietly changing how every client's memories are
    extracted, which no test and no dashboard would show.
    """
    fleet = (os.environ.get("STAGE3_EXTRACTION_MODE", "baseline") or "baseline").strip().lower()
    if fleet not in _VALID_EXTRACTION_MODES:
        logger.warning(
            "STAGE3_EXTRACTION_MODE=%r is not one of %s — using baseline",
            fleet, _VALID_EXTRACTION_MODES,
        )
        fleet = "baseline"
    if instance_id is None:
        return fleet
    override = _instance_mode_overrides().get(str(instance_id).strip().lower())
    if override is None:
        return fleet
    if override not in _VALID_EXTRACTION_MODES:
        logger.warning(
            "STAGE3_EXTRACTION_MODE_BY_INSTANCE gives instance %s the unknown "
            "mode %r — falling back to the fleet mode %r",
            instance_id, override, fleet,
        )
        return fleet
    if override != fleet:
        logger.info(
            "stage3_extraction_mode_override instance=%s mode=%s fleet=%s",
            instance_id, override, fleet,
        )
    return override

# Token budget per chunk when Stage 3 splits large content internally.
# Larger than Stage 0's 3000 because extraction prompts are expensive —
# fewer chunks = fewer LLM calls.
STAGE3_CHUNK_BUDGET = 6000


def _speaker_attribution_enabled() -> bool:
    """Speaker attribution is opt-in: OFF keeps extraction prompts and the
    resulting memory rows byte-identical to today (assertion_source stays
    None/NULL everywhere)."""
    return os.environ.get("ENABLE_SPEAKER_ATTRIBUTION", "false").lower() in ("true", "1", "yes")


# Accepted values after normalization; common speaker-label aliases fold in.
_ASSERTION_SOURCE_ALIASES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "agent": "assistant",
    "system": "system",
    "tool": "system",
}


def _clean_assertion_source(value: Any) -> Optional[str]:
    """Normalize an LLM-emitted assertion_source to user|assistant|system,
    or None for anything unexpected (never raise on junk output)."""
    if not isinstance(value, str):
        return None
    return _ASSERTION_SOURCE_ALIASES.get(value.strip().lower())


def _maybe_add_customer_we(system_prompt: str, content: str, request_scope) -> str:
    """When "we" means the writer's customer (SYNAP_CUSTOMER_ENTITY), memories
    say so. Without it "we were charged twice" is stored as "The user was
    charged twice": the wrong one of the two, and a memory that never names the
    customer's node in the graph."""
    from synap.cloud.agents.ingestion.services import writer_entity
    customer = writer_entity.customer_label_for(content, request_scope)
    if not customer:
        return system_prompt
    return system_prompt + CUSTOMER_WE_SECTION.format(
        customer=customer, customer_cap=customer[0].upper() + customer[1:])


def _maybe_add_speaker_attribution(system_prompt: str, content: str) -> str:
    """Append the attribution addendum when enabled and the content has
    speaker-labelled turns (stage-0 grammar). No-op otherwise, so
    non-conversation documents and flag-off runs keep today's prompts."""
    if _speaker_attribution_enabled() and TURN_PATTERN.search(content):
        return system_prompt + _SPEAKER_ATTRIBUTION_ADDENDUM
    return system_prompt


def _analysis_hints_enabled() -> bool:
    """Kill-switch for client-provided analysis hints (spec §4.5).

    Default ON — the transcript-ingest endpoint stores/forwards analysis
    only when the client supplied it, so flag-off is the emergency lever,
    not the steady state (contrast ENABLE_SPEAKER_ATTRIBUTION's opt-in)."""
    return os.environ.get("ENABLE_ANALYSIS_HINTS", "true").lower() in ("true", "1", "yes")


# Cap on the serialized analysis JSON appended to extraction prompts.
ANALYSIS_HINTS_MAX_CHARS = 2000


def _maybe_add_analysis_hints(user_prompt: str, analysis_hints: Optional[Dict[str, Any]]) -> str:
    """Append client-provided analysis JSON as extraction *hints* (spec §4.5).

    No-op when hints are absent/non-dict, when the kill-switch is off, or
    when the hints cannot be serialized — flag-off runs and non-transcript
    documents keep today's prompts byte-identical."""
    if not analysis_hints or not isinstance(analysis_hints, dict) or not _analysis_hints_enabled():
        return user_prompt
    try:
        compact = json.dumps(
            analysis_hints, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), default=str,
        )
    except Exception:
        return user_prompt
    if len(compact) > ANALYSIS_HINTS_MAX_CHARS:
        compact = compact[:ANALYSIS_HINTS_MAX_CHARS]
    return user_prompt + _ANALYSIS_HINTS_HEADER + compact


def _maybe_add_resolved_dates(user_prompt: str,
                              resolved_dates: Optional[Dict[str, Any]]) -> str:
    """Prepend the document's already-resolved dates (services/date_prepass).

    ⚠ An earlier version of this docstring said the defect was the five per-type
    calls computing DIFFERENT dates from the same words. Measured on the lme3
    failure set, that is almost never what happens: of 218 duplicate-sentence
    groups only 1 held two different dates, while 9 held a date on one copy and
    NULL on another. The defect is omission, not disagreement, and it is large:
    555 of 703 undated non-perpetual memories (79%) came out of a document that
    successfully dated other memories, touching 59 of 76 failing questions.

    Measured effect of this block, 480 real gpt-4o calls over 24 of those
    documents, each arm run twice:
      dated share of non-perpetual memories  75.3% -> 84.0%  (+8.6pp)
      non-perpetual memories returned          223 -> 315     (+41%)
      within-arm spread on the identical prompt        0.9pp
    So the signal is about ten times the noise. The volume rise is not padding:
    on an 8-document content check, near-duplicate rate did not get worse
    (26.5% -> 24.0%) and 57 of the extra memories were new content.

    An empty or absent table returns the prompt unchanged, so a disabled flag or
    a failed pre-pass leaves the prompt byte-identical to today's. That is the
    common case more often than is comfortable: the pre-pass found no phrase at
    all in 48 of 172 lme3 documents (28%), and the median document yields 1
    phrase, so this cannot be the whole answer to dating a document's memories."""
    if not resolved_dates:
        return user_prompt
    block = date_prepass.render_resolved_block(resolved_dates)
    return (block + user_prompt) if block else user_prompt


# A source_phrase counted from the CONVERSATION, not from another event. The
# addendum says plainly that these belong to section 1 and must not produce an
# anchor, but a measured run returned one anyway: content "drove to their
# sister's place ... around April 15", source_phrase "about six weeks ago",
# anchor_text "trip to San Francisco", offset six weeks AFTER. The phrase means
# six weeks before today; resolving it as six weeks after the San Francisco trip
# would have manufactured a confident wrong date out of an unrelated event.
_DCT_RELATIVE_SOURCE = re.compile(
    r"\b(?:ago|from\s+now|yesterday|today|tonight|tomorrow|last\s+night"
    r"|this\s+(?:morning|afternoon|evening|week|weekend|month|year)"
    r"|last\s+(?:week|weekend|month|year|summer|winter|spring|fall|autumn)"
    r"|next\s+(?:week|weekend|month|year))\b", re.IGNORECASE)

# An explicit calendar date already written into the memory TEXT. When the model
# has committed to a date in the sentence a reader will see, a derived event_start
# can only agree (and add nothing) or disagree (and produce a memory that
# contradicts itself). Measured: 2 of 4 anchors returned on the lme3 anchor
# sessions did exactly this, one of them with a date that was itself wrong.
_DATE_IN_CONTENT = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b",
    re.IGNORECASE)


def _anchor_for(raw: Dict[str, Any], content: str) -> Optional[Dict[str, Any]]:
    """The model's event_anchor, or one read out of the memory's own words.

    ⚠ The fallback is the point. Asking the model for this field has been tried
    twice and produces nothing on real conversations: the extraction prompt's
    "event_anchor" came back 0 times in 5,220 asks on the lme14 run, and
    date_prepass's "offset_from_event" kind was never classified
    in a live ingest. Meanwhile the distance is sitting in the text we store, in
    plain words: "booking three months in advance".

    Only the number, unit and direction are read here. WHICH stored memory is the
    event stays with anchor_resolver's screen, which already searches the whole
    scope across documents and is the reason this can reach an event dated in a
    DIFFERENT conversation. 982b5123 is exactly that shape: session 2 says
    "booked three months in advance", session 1 dates the wedding, and nothing
    joined them.

    A model-supplied anchor still wins when there is one, so this can only add.
    """
    given = _clean_event_anchor(raw.get("event_anchor"), content)
    if given:
        return given
    # Nothing to resolve against an anchor when the memory is already dated.
    if raw.get("event_start"):
        return None
    from synap.cloud.agents.ingestion.services.event_anchor import offset_from_text
    derived = offset_from_text(content)
    if not derived:
        return None
    # Through the SAME cleaner as a model-supplied anchor, deliberately. Its
    # guards are not about who produced the anchor: a zero offset is still not a
    # relationship, a DCT-relative phrase still belongs to date_prepass, and a
    # sentence that already states a date must still not get one derived, or the
    # same interval is counted twice.
    cleaned = _clean_event_anchor(derived, content)
    if cleaned is None:
        logger.info("event_anchor_derived_then_refused phrase=%r",
                    derived["source_phrase"][:120])
        return None
    logger.info("event_anchor_derived_from_text offset=%s phrase=%r",
                cleaned["offset"], cleaned["source_phrase"][:120])
    return cleaned


def _clean_event_anchor(raw: Any,
                        content: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """An event_anchor worth storing, or None.

    Rejects here rather than at resolution time so a malformed anchor never
    reaches a writer: a row carrying junk in that column would be re-examined by
    the resolver on every subsequent document of the scope, forever.

    *content* is the memory's own text. It is checked, not just carried: an anchor
    is only safe on a sentence that does NOT already state a date.
    """
    # ⚠ Every rejection below used to be silent, and over 444 sessions the model
    # returned 40 anchors with a VALUE that something here threw away. Neither of
    # the two logged paths further down had fired, so those 40 died on one of
    # these three and nobody could say which. A diagnostic was added in the
    # extractor to catch them and could never fire, because by then the value had
    # already been cleaned off the item it was reading. This is the place that
    # knows, so this is the place that says so.
    if not isinstance(raw, dict):
        logger.info("event_anchor_rejected reason=not_an_object value=%r",
                    str(raw)[:120])
        return None
    anchor_text = str(raw.get("anchor_text") or "").strip()
    source_phrase = str(raw.get("source_phrase") or "").strip()
    from synap.cloud.agents.ingestion.services.event_anchor import parse_offset
    offset = parse_offset(raw.get("offset"))
    if not anchor_text or offset is None:
        logger.info("event_anchor_rejected reason=%s value=%r",
                    "no_anchor_text" if not anchor_text else "offset_unparseable",
                    str(raw)[:200])
        return None
    n, unit, direction = offset
    if n == 0:
        # not a relationship; see event_anchor guardrail 1
        logger.info("event_anchor_rejected reason=zero_offset anchor_text=%r",
                    anchor_text[:120])
        return None
    if source_phrase and _DCT_RELATIVE_SOURCE.search(source_phrase):
        logger.info("event_anchor_rejected reason=source_phrase_is_dct_relative "
                    "source_phrase=%r", source_phrase[:120])
        return None
    if content and _DATE_IN_CONTENT.search(content):
        logger.info("event_anchor_rejected reason=content_already_states_a_date "
                    "anchor_text=%r", anchor_text[:120])
        return None
    return {"anchor_text": anchor_text[:300],
            "offset": {"n": n, "unit": unit, "direction": direction},
            "source_phrase": source_phrase[:300]}


class Stage3Extractor:
    """
    Extracts memories using category-specific prompts.

    Different prompts for each memory type:
    - FACTS: Concrete, verifiable information
    - PREFERENCES: Likes, dislikes, choices
    - EPISODES: Events, interactions with temporal grounding
    - EMOTIONS: Sentiment, feelings (if enabled)
    - TEMPORAL_EVENTS: Dates, deadlines, scheduled items
    """

    # Tier 1 memory types: pattern-matching tasks suitable for lighter models (gpt-4o)
    LIGHT_MEMORY_TYPES = {MemoryType.FACTS, MemoryType.EMOTIONS}

    def __init__(
        self,
        config: MemoryArchitectureConfig,
        llm_client: LLMClient,
        entity_extractor: Optional[EntityExtractor] = None,
        llm_client_light: Optional[LLMClient] = None,
        prior_context_fn=None,
        earlier_reference_resolver=None,
        llm_client_graph: Optional[LLMClient] = None,
    ):
        self._config = config
        self._llm = llm_client
        self._llm_light = llm_client_light
        # The client for the graph calls (entities, relationships, graph v1/v2). The
        # one the light tier gets unless SYNAP_ER_MODEL gives the graph its own model.
        self._llm_graph = llm_client_graph
        self._entity_extractor = entity_extractor
        # async fn(scope, document_id) -> rendered context block. Injected so the
        # extractor keeps its config-and-LLM-only shape; absent means the feature
        # is off however SYNAP_STAGE3_PRIOR_CONTEXT is set, rather than Stage 3
        # quietly acquiring a database dependency.
        self._prior_context_fn = prior_context_fn
        # services/earlier_reference.EarlierReferenceResolver, or None. Finds who
        # "he" is when the document names no one. Injected for the same reason
        # as the fetcher above; off unless SYNAP_STAGE3_EARLIER_REFERENCES is set.
        self._earlier_references = earlier_reference_resolver

    @staticmethod
    def _scope_kind(request_scope) -> str:
        """The kind of note this is: "user" (a person wrote it), "customer" (a record
        kept for one customer) or "client" (the client's own document). Read by the
        graph prompt when the instance's MACA config carries extraction patterns."""
        if getattr(request_scope, "user_id", None):
            return "user"
        if getattr(request_scope, "customer_id", None):
            return "customer"
        return "client"

    @property
    def _cfg(self) -> MemoryArchitectureConfig:
        """The MACA config steering THIS call: the instance's own config when
        the call opted in, else the startup config. Read this, never
        ``self._config``, anywhere a prompt or an extraction decision is made."""
        return _ACTIVE_MACA_CONFIG.get() or self._config

    async def extract(
        self,
        content: str,
        stage2_output: Stage2Output,
        document_id: str,
        dct: Optional[datetime] = None,
        has_temporal_content: bool = False,
        request_scope=None,
        correlation_id: Optional[str] = None,
        analysis_hints: Optional[Dict[str, Any]] = None,
        config: Optional[MemoryArchitectureConfig] = None,
    ) -> Stage3Output:
        """Extract memories; see ``_extract_impl``.

        *config*: the MACA config of the instance this document belongs to. It
        steers the prompts only when ``SYNAP_MACA_EXTRACTION_INSTANCES`` enables
        the instance. Otherwise the startup config is used and every prompt is
        byte-identical to what was sent before this was wired.
        """
        instance_id = getattr(request_scope, "instance_id", None)
        active = config if (config is not None and maca_extraction_enabled_for(instance_id)) else None
        if active is not None:
            logger.info(
                "stage3_maca_extraction_active instance=%s config_version=%s correlation_id=%s",
                instance_id, getattr(active.metadata, "config_version", None), correlation_id or "")
        token = _ACTIVE_MACA_CONFIG.set(active)
        # Read from `config` and not `active`: this guardrail does not wait for
        # the MACA prompt switch. Empty when its own switch is off.
        prohibited_token = PROHIBITED_CONTENT.set(
            guardrails.prohibited_content_for(config, instance_id))
        # Same for the custom categories: their own switch, read from `config`.
        custom_token = custom_categories.ACTIVE.set(
            custom_categories.custom_categories_for(config, instance_id))
        grounding_token = REQUIRE_ENTITY_GROUNDING.set(
            guardrails.entity_grounding_required(config, instance_id))
        try:
            return await self._extract_impl(
                content=content, stage2_output=stage2_output, document_id=document_id,
                dct=dct, has_temporal_content=has_temporal_content,
                request_scope=request_scope, correlation_id=correlation_id,
                analysis_hints=analysis_hints)
        finally:
            REQUIRE_ENTITY_GROUNDING.reset(grounding_token)
            custom_categories.ACTIVE.reset(custom_token)
            PROHIBITED_CONTENT.reset(prohibited_token)
            _ACTIVE_MACA_CONFIG.reset(token)

    async def _extract_impl(
        self,
        content: str,
        stage2_output: Stage2Output,
        document_id: str,
        dct: Optional[datetime] = None,
        has_temporal_content: bool = False,
        request_scope=None,
        correlation_id: Optional[str] = None,
        analysis_hints: Optional[Dict[str, Any]] = None,
    ) -> Stage3Output:
        """
        Extract memories based on categorization priorities.

        For each priority memory type:
        1. Run type-specific extraction prompt
        2. (PII is handled at the ingress gate, before any model call)
        3. Score confidence
        4. Determine destination (memories vs candidates)

        *analysis_hints*: client-provided call-analysis JSON (transcript
        ingest, spec §4.5) — appended to the extraction user prompt as
        hints, never as extraction source.
        """
        all_memories: List[ExtractedMemory] = []
        all_entities: List[ExtractedEntity] = []
        all_relationships: List[ExtractedRelationship] = []

        # Split content if it exceeds the chunk budget (handles bypassed Stage 0)
        chunks = self._chunk_content(content)
        is_chunked = len(chunks) > 1

        if is_chunked:
            logger.info(
                f"Stage 3: Content splitting activated chunks={len(chunks)} "
                f"content_length={len(content)} correlation_id={correlation_id}"
            )

        # Extraction mode, per instance where an override says so.
        # The fleet-wide variable alone could not answer "baseline for one
        # instance, hybrid_graph for everyone else", which matters because
        # baseline costs ~3 more model calls per chunk and every client would
        # have paid for an experiment run on one of them.
        mode = resolve_extraction_mode(getattr(request_scope, "instance_id", None))

        # Resolve this document's temporal phrases ONCE, before any per-type call.
        # On the FULL content, not per chunk: a phrase and the event it anchors to
        # can sit either side of a chunk boundary, and the table is the document's,
        # not a chunk's. Off by default; never raises; an empty table leaves every
        # prompt below byte-identical to today's.
        resolved_dates: Dict[str, Any] = {}
        if date_prepass.enabled_for(getattr(request_scope, "instance_id", None)):
            resolved_dates = await date_prepass.run_prepass(
                content=content, dct=dct, llm_client=self._llm,
                correlation_id=correlation_id,
            )
            # dated vs not_dated, not one "resolved" count: the table carries both
            # phrases we worked out and phrases we deliberately refused to date.
            _dated = sum(1 for w in resolved_dates.values()
                         if w.get("event_start") is not None)
            logger.info(
                "stage3_date_prepass dated=%d not_dated=%d instance=%s mode=%s "
                "correlation_id=%s", _dated, len(resolved_dates) - _dated,
                getattr(request_scope, "instance_id", None), mode, correlation_id or "")

        # What we already know about this person, for resolving a reference the
        # document cannot resolve alone. Read once per document, never per chunk.
        # Measured safe (0 leaks in 63 planted opportunities) but recall is NOT
        # settled — see services/prior_context.
        prior_block = ""
        if (self._prior_context_fn is not None
                and prior_context.enabled_for(
                    getattr(request_scope, "instance_id", None))):
            try:
                prior_block = await self._prior_context_fn(request_scope, document_id)
            except Exception:  # noqa: BLE001 — an enhancement, never a gate
                logger.warning("stage3_prior_context_failed correlation_id=%s",
                               correlation_id or "", exc_info=True)
                prior_block = ""

        # Entity extraction runs on FULL content — start in parallel with chunk
        # processing. Skipped in hybrid_graph mode (one merged call extracts
        # entities AND relationships together after the memory pass).
        graph_on = graph_extraction_enabled_for(getattr(request_scope, "instance_id", None))
        if not graph_on:
            logger.info(
                "stage3_graph_extraction_skipped instance=%s mode=%s correlation_id=%s",
                getattr(request_scope, "instance_id", None), mode, correlation_id or "")

        # SYNAP_GRAPH_EXTRACTION_V2: one graph call for every mode, so
        # ``baseline`` no longer makes its own entity and relationship calls.
        from synap.cloud.agents.ingestion.services import graph_v2
        graph_v2_on = graph_v2.enabled(getattr(request_scope, "instance_id", None))

        entity_task = None
        if self._entity_extractor and mode != "hybrid_graph" and graph_on and not graph_v2_on:
            enabled_types = [mt for mt in stage2_output.extraction_priorities if self._is_type_enabled(mt)]
            logger.info(f"Stage 3: Starting parallel extraction (types + entities) type_tasks={len(enabled_types)} content_length={len(content)}")
            entity_task = asyncio.create_task(
                self._extract_entities(
                    content,
                    [],
                    request_scope=request_scope,
                    correlation_id=correlation_id,
                )
            )

        # Process chunks sequentially (summary chains), types parallel within each chunk.
        #
        # STAGE3_EXTRACTION_MODE=hybrid (Option D): extract the three "simple"
        # types (FACTS, PREFERENCES, EMOTIONS) in ONE merged call instead of
        # three, keeping EPISODES / TEMPORAL_EVENTS as their own calls. Any
        # other value keeps the original per-type fan-out (baseline).
        preceding_summary = ""
        SIMPLE_TYPES = {MemoryType.FACTS, MemoryType.PREFERENCES, MemoryType.EMOTIONS}
        for chunk_idx, chunk in enumerate(chunks):
            _summary = preceding_summary if is_chunked else ""
            enabled = [
                mt
                for mt in stage2_output.extraction_priorities
                if self._is_type_enabled(mt)
            ]

            def _type_task(mt):
                return self._extract_type(
                    content=chunk,
                    memory_type=mt,
                    categories=stage2_output.categories,
                    document_id=document_id,
                    dct=dct,
                    has_temporal_content=has_temporal_content,
                    request_scope=request_scope,
                    correlation_id=correlation_id,
                    preceding_summary=_summary,
                    analysis_hints=analysis_hints,
                    resolved_dates=resolved_dates,
                    prior_block=prior_block,
                )

            if mode in ("hybrid", "hybrid_graph"):
                simple = [mt for mt in enabled if mt in SIMPLE_TYPES]
                rest = [mt for mt in enabled if mt not in SIMPLE_TYPES]
                type_tasks = []
                # A custom type rides the merged call, so it runs even when the
                # three simple types are all switched off.
                if simple or custom_categories.ACTIVE.get():
                    type_tasks.append(
                        self._extract_simple_group(
                            content=chunk,
                            memory_types=simple,
                            categories=stage2_output.categories,
                            document_id=document_id,
                            dct=dct,
                            has_temporal_content=has_temporal_content,
                            request_scope=request_scope,
                            correlation_id=correlation_id,
                            preceding_summary=_summary,
                            analysis_hints=analysis_hints,
                            resolved_dates=resolved_dates,
                            prior_block=prior_block,
                        )
                    )
                type_tasks.extend(_type_task(mt) for mt in rest)
            else:
                if chunk_idx == 0 and custom_categories.ACTIVE.get():
                    logger.warning(
                        "stage3_custom_categories_skipped_in_baseline instance=%s "
                        "correlation_id=%s", getattr(request_scope, "instance_id", None),
                        correlation_id or "")
                type_tasks = [_type_task(mt) for mt in enabled]
                # Assistant recall only ever rode the merged call, so an
                # instance pinned to baseline (the 4 July reproduction) gets
                # none of it, even when it is named in
                # SYNAP_ASSISTANT_RECALL_INSTANCES. That profile is what took
                # LongMemEval's single-session-assistant category from 13/56 to
                # 53/56. SYNAP_ASSISTANT_RECALL_IN_BASELINE adds it as one extra
                # call beside the five, leaving those five untouched, so the
                # baseline reproduction and the recall profile can run in one
                # ingest instead of two phases. Off by default; the extra call
                # also re-extracts user facts, so it trades duplicates for
                # coverage.
                simple = [mt for mt in enabled if mt in SIMPLE_TYPES]
                if (simple
                        and recall_in_baseline_enabled()
                        and assistant_recall.enabled_for(
                            getattr(request_scope, "instance_id", None))):
                    logger.info(
                        "stage3_assistant_recall_extra_call instance=%s correlation_id=%s",
                        getattr(request_scope, "instance_id", None), correlation_id or "")
                    type_tasks.append(
                        self._extract_simple_group(
                            content=chunk,
                            memory_types=simple,
                            categories=stage2_output.categories,
                            document_id=document_id,
                            dct=dct,
                            has_temporal_content=has_temporal_content,
                            request_scope=request_scope,
                            correlation_id=correlation_id,
                            preceding_summary=_summary,
                            analysis_hints=analysis_hints,
                            resolved_dates=resolved_dates,
                            prior_block=prior_block,
                        )
                    )

            type_results = await asyncio.gather(*type_tasks)
            for batch in type_results:
                all_memories.extend(batch)

            # Build preceding summary for next chunk
            if is_chunked and chunk_idx < len(chunks) - 1:
                preceding_summary = self._generate_preceding_summary(preceding_summary, chunk)

        # Coalescing: one row per fact, instead of one row per type that
        # mentions the fact.
        #
        # Here because this is the only point in the pipeline holding every
        # type's output at once. There used to be a per-type dedupe just above
        # this, since deleted: it grouped BY type, was gated on is_chunked,
        # and keyed on source_span, which only the ENTITY prompts ask for, so for
        # memories the key was always None and its first branch kept every row
        # unconditionally. A no-op that read as a safeguard. Knowledge update
        # cannot do this job either:
        # it finds rivals in Chroma, and these rows reach Chroma in stage 4,
        # after KU has already run.
        #
        # No flag. It does not need one to be safe: every group must pass the
        # verbatim detail guard and the event-window check, and anything that
        # fails — a bad model answer, a dropped number, two windows that cannot
        # be one event — leaves every member standing, which is today's
        # behaviour. See services/coalesce.
        if len(all_memories) > 1:
            result = await coalesce.coalesce_memories(
                all_memories,
                group_fn=self._coalesce_group_fn(request_scope, correlation_id),
                merge_fn=self._coalesce_merge_fn(request_scope, correlation_id),
                correlation_id=correlation_id,
            )
            if result.counts.get("groups_merged"):
                logger.info(
                    "stage3_coalesce in=%d out=%d groups=%d merged_away=%d "
                    "refused_detail=%d refused_window=%d instance=%s "
                    "correlation_id=%s",
                    result.counts["in"], result.counts["out"],
                    result.counts["groups_merged"], result.counts["merged_away"],
                    result.counts["refused_detail"],
                    result.counts["refused_window"],
                    getattr(request_scope, "instance_id", None),
                    correlation_id or "")
            all_memories = result.memories

        # Put back what the five type calls dropped. AFTER coalesce on purpose:
        # the repair should see one row per fact, not one row per type that
        # mentions it, or it spends its extends on rows that are about to be
        # merged away. See services/sweep_repair for the measurement.
        if sweep_repair.enabled_for(getattr(request_scope, "instance_id", None)):
            all_memories = await self._sweep_repair(
                content=content, memories=all_memories, llm=self._llm,
                request_scope=request_scope, correlation_id=correlation_id,
                document_id=document_id,
            )

        # Backstop for guardrails.compliance.prohibited_memory_content. The prompt
        # told the model (guidance); this only catches a literal listed phrase
        # (see services/guardrails). Before the graph step, so no entity link
        # or relationship is built from a memory that is about to be dropped.
        prohibited = PROHIBITED_CONTENT.get()
        if prohibited:
            all_memories, dropped = guardrails.drop_prohibited(all_memories, prohibited)
            if dropped:
                logger.info(
                    "stage3_prohibited_content_dropped dropped=%d kept=%d instance=%s "
                    "correlation_id=%s",
                    dropped, len(all_memories),
                    getattr(request_scope, "instance_id", None), correlation_id or "")

        # Resolve entities + relationships. In hybrid_graph mode, ONE merged
        # call produces both (nodes + edges) — its relationship endpoints are
        # guaranteed to be among the extracted entities. Otherwise the separate
        # entity task ran in parallel and relationships extract sequentially.
        _have_graph = False
        references = []
        from synap.cloud.agents.ingestion.services import writer_entity
        # "I" in this document, as one more entity (SYNAP_WRITER_ENTITY).
        writer = writer_entity.label_for(content, request_scope)
        # The v2 graph call reads entities and relationships together and does not use the separate
        # entity extractor, so it does not need one (Core has none).
        if graph_v2_on and graph_on:
            all_entities, all_relationships = await self._extract_graph_v2(
                content, request_scope=request_scope, correlation_id=correlation_id)
            references = await self._find_earlier_references(
                content, all_entities, request_scope, document_id)
            if references:
                all_entities, all_relationships = await self._reextract_with_references(
                    content, all_entities, all_relationships, references,
                    request_scope=request_scope, correlation_id=correlation_id, v2=True)
            _have_graph = True
        elif mode == "hybrid_graph" and self._entity_extractor and graph_on:
            all_entities, all_relationships = await self._extract_graph(
                content,
                request_scope=request_scope,
                correlation_id=correlation_id,
                writer=writer,
            )
            references = await self._find_earlier_references(
                content, all_entities, request_scope, document_id)
            if references:
                # The one call that read the document had no name for the
                # pronoun. Read it again, told who is meant.
                all_entities, all_relationships = await self._reextract_with_references(
                    content, all_entities, all_relationships, references,
                    request_scope=request_scope, correlation_id=correlation_id, writer=writer,
                )
            _have_graph = True
        elif entity_task is not None:
            all_entities = await entity_task
            logger.info(f"Stage 3: Parallel extraction complete, starting relationship extraction memories_count={len(all_memories)} entities_found={len(all_entities)} entity_names={[e.name for e in all_entities[:10]]} entity_types={list(set(e.entity_type for e in all_entities))}")
            references = await self._find_earlier_references(
                content, all_entities, request_scope, document_id)
            all_entities = self._with_referenced_entities(all_entities, references)
            # Relationship extraction depends on entities — must be sequential
            writer_node = writer_entity.entity(request_scope) if writer else None
            all_relationships = await self._extract_relationships(
                content,
                all_entities,
                request_scope=request_scope,
                correlation_id=correlation_id,
                references=references,
                writer=writer_node,
            )
            if writer_node is not None:
                all_entities = writer_entity.without_unused(
                    list(all_entities) + [writer_node], all_relationships)
            _have_graph = True

        if _have_graph and references:
            # The pronoun was resolved for the graph; name the entity in the
            # memories too, or nothing ties them to it (see named_in_text).
            from synap.cloud.agents.ingestion.services.earlier_reference import named_in_text
            for memory in all_memories:
                memory.content = named_in_text(memory.content, references, content)

        # After the graph step (it is what names the entities) and before the
        # link step, so no link is built from a memory about to be dropped.
        if REQUIRE_ENTITY_GROUNDING.get():
            all_memories = self._apply_entity_grounding(
                all_memories, all_entities, request_scope, correlation_id)

        if _have_graph:
            logger.info(f"Stage 3: graph complete, linking memories to entities entities_found={len(all_entities)} relationships_found={len(all_relationships)} relationship_types={list(set(r.relationship_type for r in all_relationships))}")
            # Link memories to entities (CPU-only)
            async with start_span("ingestion.stage3.link_entities", {
                "synap.memory_count": len(all_memories),
                "synap.entity_count": len(all_entities),
            }):
                t0 = time.monotonic()
                self._link_memories_to_entities(all_memories, all_entities)
                linked_count = sum(1 for m in all_memories if m.entities_mentioned)
                set_current_span_attributes({
                    "synap.links_created": sum(len(m.entities_mentioned) for m in all_memories),
                    "synap.memories_with_entities": linked_count,
                    "synap.link_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                })

            logger.info(f"Stage 3: Memory-entity linking complete memories_with_entities={linked_count} total_memories={len(all_memories)} total_links={sum(len(m.entities_mentioned) for m in all_memories)}")
        else:
            logger.info("Stage 3: No entity extractor available, skipping entity/relationship extraction")

        logger.info(f"Stage 3: Extraction complete input_id={stage2_output.input_id} memories_count={len(all_memories)} entities_count={len(all_entities)} relationships_count={len(all_relationships)}")

        return Stage3Output(
            input_id=stage2_output.input_id,
            memories=all_memories,
            entities=all_entities,
            relationships=all_relationships,
        )

    async def _extract_type(
        self,
        content: str,
        memory_type: MemoryType,
        categories: List,
        document_id: str,
        dct: Optional[datetime] = None,
        has_temporal_content: bool = False,
        request_scope=None,
        correlation_id: Optional[str] = None,
        preceding_summary: str = "",
        analysis_hints: Optional[Dict[str, Any]] = None,
        resolved_dates: Optional[Dict[str, Any]] = None,
        prior_block: str = "",
    ) -> List[ExtractedMemory]:
        """Extract memories of a specific type."""
        from synap.cloud.agents.ingestion.prompts.extraction import (
            EmotionsExtractionPrompt,
            EpisodesExtractionPrompt,
            FactsExtractionPrompt,
            PreferencesExtractionPrompt,
            TemporalExtractionPrompt,
        )
        from synap.cloud.agents.ingestion.prompts.prompt_resolver import PromptResolver

        # Get appropriate prompt builder
        prompt_builders = {
            MemoryType.FACTS: FactsExtractionPrompt,
            MemoryType.PREFERENCES: PreferencesExtractionPrompt,
            MemoryType.EPISODES: EpisodesExtractionPrompt,
            MemoryType.EMOTIONS: EmotionsExtractionPrompt,
            MemoryType.TEMPORAL_EVENTS: TemporalExtractionPrompt,
        }

        prompt_builder_class = prompt_builders.get(memory_type)
        if not prompt_builder_class:
            return []

        # Assistant-recall profile on the BASELINE path. The merged call swaps
        # its whole system prompt for the profile's; a per-type call cannot,
        # because that prompt ends by telling the model to choose between
        # facts, preferences and emotions and this call has already made that
        # choice. Handing it to all three would extract the assistant's content
        # three times over.
        #
        # So the rules are appended instead, and only to facts — the profile's
        # own typing rule stores assistant-supplied information as facts. Still
        # five calls, with one of their prompts changed.
        #
        # The avoid list comes out for the same reason the merged path drops
        # it: MACA's "do not extract" guidance is written about the user, and
        # it suppresses exactly the assistant content this profile is for.
        from synap.cloud.agents.ingestion.prompts.extraction import assistant_recall

        _recall_on = assistant_recall.enabled_for(
            getattr(request_scope, "instance_id", None)
        )
        _recall_addendum = (
            assistant_recall.system_addendum(memory_type) if _recall_on else ""
        )

        # Resolve to provider-specific variant if available
        prompt_builder_class = PromptResolver.resolve(prompt_builder_class)
        prompt_builder = prompt_builder_class(
            _without_avoid(self._cfg) if _recall_addendum else self._cfg
        )
        system_prompt, user_prompt = prompt_builder.build(
            content=content,
            categories=categories,
            dct=dct,
            has_temporal_content=has_temporal_content,
            instance_id=getattr(request_scope, "instance_id", None),
        )
        if _recall_addendum:
            # Appended, not substituted, so the instance guidance the builder
            # just added to this prompt survives — the merged path has to
            # re-append it precisely because it substitutes.
            system_prompt = system_prompt + _recall_addendum
            logger.info(
                "stage3_assistant_recall_baseline type=%s instance=%s correlation_id=%s",
                memory_type.value, getattr(request_scope, "instance_id", None),
                correlation_id or "",
            )
        system_prompt = _maybe_add_speaker_attribution(system_prompt, content)
        system_prompt = _maybe_add_customer_we(system_prompt, content, request_scope)

        # Inject preceding summary for chunked content (appears before "Content to analyze:")
        if preceding_summary:
            summary_block = (
                "\nPrevious context summary (from earlier parts of this document):\n"
                f"---\n{preceding_summary}\n---\n"
            )
            user_prompt = summary_block + user_prompt

        user_prompt = _maybe_add_analysis_hints(user_prompt, analysis_hints)
        # The document's dates, resolved once. Must reach BOTH paths: the
        # merged 'hybrid' call covers facts/preferences/emotions, and leaving
        # it out here would silently exempt three of the five types.
        user_prompt = _maybe_add_resolved_dates(user_prompt, resolved_dates)
        # Context LAST, so it sits at the top of the prompt above the
        # resolved dates and the content it is there to help read.
        user_prompt = prior_context.wrap(user_prompt, prior_block)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Select client: Tier 1 types use lighter model (gpt-4o) if available
        llm = self._llm_light if (self._llm_light and memory_type in self.LIGHT_MEMORY_TYPES) else self._llm

        # Call LLM
        async with start_span(f"ingestion.stage3.extract_{memory_type.value}", {
            "synap.memory_type": memory_type.value,
            "synap.prompt_len": len(system_prompt) + len(user_prompt),
            "synap.correlation_id": correlation_id or "",
        }):
            t0 = time.monotonic()
            try:
                response = await llm.complete(
                    messages=messages,
                    telemetry_scope=request_scope,
                    telemetry_operation=f"ingestion.stage3_extract_{memory_type.value}",
                    telemetry_correlation_id=correlation_id,
                    max_tokens=8000,
                    json_mode=True,
                )
            except TypeError:
                # Fallback for older LLMClient implementations without messages support
                prompt = system_prompt + "\n\n" + user_prompt
                response = await llm.complete(prompt)
            set_current_span_attributes({
                "synap.response_len": len(response),
                "synap.llm_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # Parse extracted memories
        raw_memories = self._parse_extraction(response, memory_type)

        # ⚠ Why this log exists. `memories.event_anchor` has been NULL on every row
        # ever written, and across four ingestions of the one scope built
        # to produce an anchor it stayed NULL, while the same text through the same
        # prompt and the same model produced one every time off the pipeline. Four
        # misses at the rate those runs suggested is about a 1-in-80 event, so the
        # off-pipeline reproduction is wrong somewhere and no amount of reasoning
        # about which block differs has found it.
        #
        # So this stops guessing and records the only thing that settles it: whether
        # the prompt asked, and what came back. Cheap (one line per type call) and it
        # prints the response head ONLY when the ask went unanswered, which is the
        # case nobody has been able to see.
        if '"event_anchor"' in user_prompt:
            _got = sum(1 for r in raw_memories
                       if isinstance(r, dict) and r.get("event_anchor"))
            logger.info(
                "event_anchor_ask type=%s asked=1 got=%d items=%d prompt_len=%d "
                "correlation_id=%s", memory_type.value, _got, len(raw_memories),
                len(user_prompt), correlation_id or "")
            if _got == 0:
                # The ONE distinction that is left, and the 700-char head was too
                # short to show it because event_anchor is the last field in the
                # enumeration. Does the key appear in the reply at all?
                #   absent  -> the model is not following the schema addition
                #   null     -> it read the field and declined, so the RULES are
                #               what need changing, not the schema
                _raw = str(response)
                _key = '"event_anchor"' in _raw
                _null = ('"event_anchor": null' in _raw
                         or '"event_anchor":null' in _raw)
                logger.info(
                    "event_anchor_unanswered type=%s key_in_reply=%s "
                    "null_in_reply=%s reply_len=%d prompt_len=%d correlation_id=%s",
                    memory_type.value, _key, _null,
                    len(_raw), len(user_prompt), correlation_id or "")
                # The 40 anchors the model returned with a value, and that were
                # then thrown away, are NOT visible from here: this block only
                # runs when no item carries one, which is the same condition the
                # old diagnostic here tested, so it could never print. Every
                # rejection now names itself in _clean_event_anchor instead.

        # The quote pass rides the facts call on this path, as it rides the
        # merged call on the other, so baseline runs the profile that was
        # measured rather than a near-copy of it. Conditional on the content
        # looking quoted, so a quote-free session pays nothing.
        if _recall_addendum and assistant_recall.wants_quote_pass(content):
            raw_memories = await self._quote_pass(
                content=content, raw_items=raw_memories, llm=llm,
                request_scope=request_scope, correlation_id=correlation_id,
            )

        # Process each memory
        processed = []
        for raw in raw_memories:
            memory = await self._process_memory(
                raw=raw,
                memory_type=memory_type,
                document_id=document_id,
                categories=categories,
                request_scope=request_scope,
            )
            if memory:
                processed.append(memory)

        set_current_span_attributes({
            "synap.raw_memories_parsed": len(raw_memories),
            "synap.memories_accepted": len(processed),
        })
        return processed

    async def _quote_pass(self, *, content, raw_items, llm, request_scope,
                          correlation_id):
        """One extra call whose only job is verbatim quotation capture.

        Separate because enumeration competes with the other extraction goals:
        on a 5.5k-char essay carrying six Borges quotes, gpt-4o at temperature
        0 captured four, duplicated one and skipped the one the question
        needed, three runs out of three. Given its own call it captured all
        six, three out of three.

        Shared by the merged and the per-type paths so the baseline profile is
        the same profile that was measured, not a near-copy of it. Gated on the
        content actually looking quoted, so a quote-free session pays nothing,
        and non-fatal: a failure here loses quotations, not the extraction.
        """
        from synap.cloud.agents.ingestion.prompts.extraction import assistant_recall

        try:
            async with start_span("ingestion.stage3.extract_quotes", {
                "synap.correlation_id": correlation_id or "",
            }):
                q_response = await llm.complete(
                    messages=[
                        {"role": "system",
                         "content": assistant_recall.QUOTE_PASS_SYSTEM_PROMPT},
                        {"role": "user",
                         "content": assistant_recall.build_quote_user_prompt(content)},
                    ],
                    telemetry_scope=request_scope,
                    telemetry_operation="ingestion.stage3_extract_quotes",
                    telemetry_correlation_id=correlation_id,
                    max_tokens=4000,
                    json_mode=True,
                )
            # Reuse the class's tolerant loader: same markdown-fence and
            # stray-prose handling the main extraction relies on.
            q_parsed = self._loads_merged(q_response)
            q_quotes = (
                q_parsed.get("quotes") if isinstance(q_parsed, dict) else None
            ) or []
            q_items = assistant_recall.merge_quote_entries(
                [str(r.get("content", "")) for r in raw_items],
                q_quotes,
            )
            logger.info(
                "stage3_quote_pass quotes_found=%d quotes_added=%d correlation_id=%s",
                len(q_quotes), len(q_items), correlation_id or "",
            )
            return list(raw_items) + q_items if q_items else raw_items
        except Exception as e:  # noqa: BLE001 - additive, must never fail ingest
            logger.warning("stage3_quote_pass_failed (non-fatal): %s", e)
            return raw_items

    async def _sweep_repair(self, *, content, memories, llm, request_scope,
                            correlation_id, document_id=""):
        """Two calls that put back the detail the five type calls dropped.

        Call 1 lists what the text says about this person and never sees our
        rows, so it cannot be talked into believing something is already
        covered. Call 2 places each statement: extend the row that already
        covers it, add it, or drop it.

        Measured through this method's own design on 9 scopes x 3 runs: the $60
        taxi fare, the cause of a missed run and a 7 AM departure all go from 0
        of 3 to 3 of 3, control scopes do not grow, and duplicate pairs fall
        from 52 to 50 because 101 rows are REWRITTEN rather than 101 added.
        See services/sweep_repair for the full table and for what it does NOT
        catch.

        Non-fatal, like the quote pass: anything that goes wrong here must
        leave extraction exactly as it is today.
        """
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            find_dropped_entities,
        )

        if not memories or not content:
            return memories
        try:
            async with start_span("ingestion.stage3.sweep", {
                "synap.correlation_id": correlation_id or "",
            }):
                raw = await llm.complete(
                    messages=[
                        {"role": "system", "content": sweep_repair.SWEEP_SYSTEM},
                        {"role": "user",
                         "content": sweep_repair.build_sweep_user_prompt(content)},
                    ],
                    telemetry_scope=request_scope,
                    telemetry_operation="ingestion.stage3_sweep",
                    telemetry_correlation_id=correlation_id,
                    max_tokens=4000, json_mode=True,
                )
            statements = sweep_repair.surviving_statements(
                sweep_repair.loads_json(raw).get("statements"), content)
            if not statements:
                logger.info("stage3_sweep statements=0 correlation_id=%s",
                            correlation_id or "")
                return memories

            rows = [m.content or "" for m in memories]
            async with start_span("ingestion.stage3.attach", {
                "synap.correlation_id": correlation_id or "",
            }):
                raw2 = await llm.complete(
                    messages=[
                        {"role": "system", "content": sweep_repair.ATTACH_SYSTEM},
                        {"role": "user",
                         "content": sweep_repair.build_attach_user_prompt(
                             rows, [s["content"] for s in statements])},
                    ],
                    telemetry_scope=request_scope,
                    telemetry_operation="ingestion.stage3_attach",
                    telemetry_correlation_id=correlation_id,
                    max_tokens=4000, json_mode=True,
                )
            extends, adds = sweep_repair.plan_from_decisions(
                sweep_repair.loads_json(raw2).get("decisions"),
                statements, rows)

            out = list(memories)
            applied = refused = 0
            done: set = set()
            for ri, text, statement in extends:
                # One row is only rewritten once: a second rewrite is written
                # against the ORIGINAL text, so applying it would discard the
                # first. Tracked HERE rather than when planning, so a refused
                # candidate does not block a valid later one for the same row.
                if ri in done:
                    continue
                # ⚠ The ONE guard coalesce cannot give us here. This runs after
                # coalesce, so the row being rewritten has already passed
                # find_dropped_entities; without this an extend could silently
                # drop a name, a number or a date that the row carried, trading
                # a recall bug for a data-loss bug, which is the worse trade.
                if find_dropped_entities([out[ri].content or ""], text):
                    refused += 1
                    continue
                # ⚠ And the same guard in REVERSE, which is the only thing that
                # stops invention on this path. An extend that keeps every
                # entity can still ADD one nobody said: "had a doctor's
                # appointment" became "had a doctor's appointment at the Mayo
                # Clinic" and every other guard allowed it. Every name, amount
                # and date in the rewrite must come from the row or from the
                # statement that justified it. Verified against the three real
                # extends from the measured run: all three pass.
                if find_dropped_entities(
                        [text], f"{out[ri].content or ''} {statement}"):
                    refused += 1
                    continue
                # content only: the row keeps its own dates, type, speaker and
                # confidence, because none of those was re-judged.
                out[ri] = out[ri].model_copy(update={"content": text})
                done.add(ri)
                applied += 1

            # The destination is DERIVED, never inherited from a sibling row.
            # Inheriting is exactly the defect fixed in e64a5062a: a row that
            # scored below the floor would send a new, well-formed memory to
            # `candidates`, which retrieval never reads.
            scope_level = getattr(request_scope, "scope_level", None) or "user"
            if scope_level == "customer":
                destination = MemoryDestination.CUSTOMER_KNOWLEDGE
            elif scope_level == "client":
                destination = MemoryDestination.CLIENT_KNOWLEDGE
            else:
                destination = MemoryDestination.MEMORIES
            _src = {"person": "user", "assistant": "assistant"}
            for a in adds:
                # Same reverse check for a brand-new row: it may carry nothing
                # its own statement did not.
                if find_dropped_entities([a["content"]], a.get("statement") or ""):
                    refused += 1
                    continue
                out.append(ExtractedMemory(
                    memory_id=uuid4(),
                    memory_type=MemoryType.FACTS,
                    content=a["content"],
                    confidence=0.9,
                    destination=destination,
                    source_document_id=document_id,
                    assertion_source=_src.get(a.get("said_by") or ""),
                ))

            logger.info(
                "stage3_sweep_repair statements=%d extended=%d refused_detail=%d "
                "added=%d rows_in=%d rows_out=%d instance=%s correlation_id=%s",
                len(statements), applied, refused, len(adds), len(memories),
                len(out), getattr(request_scope, "instance_id", None),
                correlation_id or "")
            return out
        except Exception as e:  # noqa: BLE001 - additive, must never fail ingest
            logger.warning("stage3_sweep_repair_failed (non-fatal): %s", e)
            return memories

    async def _extract_simple_group(
        self,
        content: str,
        memory_types: List[MemoryType],
        categories: List,
        document_id: str,
        dct: Optional[datetime] = None,
        has_temporal_content: bool = False,
        request_scope=None,
        correlation_id: Optional[str] = None,
        preceding_summary: str = "",
        analysis_hints: Optional[Dict[str, Any]] = None,
        resolved_dates: Optional[Dict[str, Any]] = None,
        prior_block: str = "",
    ) -> List[ExtractedMemory]:
        """Option-D hybrid: extract FACTS + PREFERENCES + EMOTIONS in ONE call.

        Assumes ``memory_types`` ⊆ {FACTS, PREFERENCES, EMOTIONS}. Each
        returned item is tagged with its type; post-processing (confidence
        threshold, destination) reuses the same ``_process_memory`` path
        as the per-type extractors, so quality handling is identical.
        """
        from synap.cloud.agents.ingestion.prompts.extraction.merged_simple import (
            MergedSimpleExtractionPrompt,
        )
        from synap.cloud.agents.ingestion.prompts.extraction import assistant_recall
        from synap.cloud.agents.ingestion.prompts.prompt_resolver import PromptResolver


        # Assistant-recall profile: allowlisted instances swap in a system prompt
        # that stores what the ASSISTANT said, not only facts about the user. The
        # user prompt is left alone — it carries the content, categories and
        # temporal addendum, none of which this profile changes. Off unless
        # SYNAP_ASSISTANT_RECALL_INSTANCES names this instance, so live clients
        # on the same fleet keep today's behaviour. See assistant_recall.py.
        #
        # Decided BEFORE the prompt is built, because it changes which MACA
        # guidance the user prompt may carry: see _without_avoid.
        _recall_on = assistant_recall.enabled_for(
            getattr(request_scope, "instance_id", None)
        )
        prompt_config = _without_avoid(self._cfg) if _recall_on else self._cfg

        prompt_builder_class = PromptResolver.resolve(MergedSimpleExtractionPrompt)
        prompt_builder = prompt_builder_class(prompt_config)
        system_prompt, user_prompt = prompt_builder.build(
            content=content,
            categories=categories,
            dct=dct,
            has_temporal_content=has_temporal_content,
            memory_types=memory_types,
            instance_id=getattr(request_scope, "instance_id", None),
        )

        if _recall_on:
            # The profile replaces the whole system prompt, which would drop the
            # instance guidance the builder just appended to it, so re-append it
            # here. Empty for an instance that has not opted in, so this is a
            # no-op for everyone else.
            from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
                MACAContextBuilder,
                with_instance_context,
            )
            try:
                instance_context = MACAContextBuilder.build_merged_extraction_context(
                    prompt_config, memory_types)
            except Exception:
                instance_context = ""
            system_prompt = with_instance_context(
                assistant_recall.SYSTEM_PROMPT, instance_context)
            logger.info(
                "stage3_assistant_recall_profile_active instance=%s correlation_id=%s",
                getattr(request_scope, "instance_id", None),
                correlation_id or "",
            )

        system_prompt = _maybe_add_speaker_attribution(system_prompt, content)
        system_prompt = _maybe_add_customer_we(system_prompt, content, request_scope)
        if preceding_summary:
            summary_block = (
                "\nPrevious context summary (from earlier parts of this document):\n"
                f"---\n{preceding_summary}\n---\n"
            )
            user_prompt = summary_block + user_prompt

        user_prompt = _maybe_add_analysis_hints(user_prompt, analysis_hints)
        # The document's dates, resolved once. Must reach BOTH paths: the
        # merged 'hybrid' call covers facts/preferences/emotions, and leaving
        # it out here would silently exempt three of the five types.
        user_prompt = _maybe_add_resolved_dates(user_prompt, resolved_dates)
        # Context LAST, so it sits at the top of the prompt above the
        # resolved dates and the content it is there to help read.
        user_prompt = prior_context.wrap(user_prompt, prior_block)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Merged call spans a light type (facts/emotions) and a tier-2 type
        # (preferences); use the capable client so preferences don't regress.
        llm = self._llm
        type_by_value = {
            "facts": MemoryType.FACTS,
            "preferences": MemoryType.PREFERENCES,
            "emotions": MemoryType.EMOTIONS,
        }
        allowed = {mt.value for mt in memory_types}
        # A custom name is accepted as a type and stored as a fact with the name as
        # its category. The names are already canonical and never a built-in's.
        custom_names = {c.name for c in custom_categories.ACTIVE.get()}

        async with start_span("ingestion.stage3.extract_merged_simple", {
            "synap.memory_types": ",".join(sorted(allowed)),
            "synap.prompt_len": len(system_prompt) + len(user_prompt),
            "synap.correlation_id": correlation_id or "",
        }):
            t0 = time.monotonic()
            try:
                response = await llm.complete(
                    messages=messages,
                    telemetry_scope=request_scope,
                    telemetry_operation="ingestion.stage3_extract_merged_simple",
                    telemetry_correlation_id=correlation_id,
                    max_tokens=8000,
                    json_mode=True,
                )
            except TypeError:
                prompt = system_prompt + "\n\n" + user_prompt
                response = await llm.complete(prompt)
            set_current_span_attributes({
                "synap.response_len": len(response),
                "synap.llm_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        raw_items = self._parse_merged_extraction(response)

        # Assistant-recall profile: a second call that only hunts quotations.
        #
        # One call cannot reliably do both jobs. On a 5.5k-char literary essay
        # carrying six Borges quotes, gpt-4o at temperature 0 captured four,
        # duplicated one, and skipped the one the question needed — three runs
        # out of three. Given its own call it captured all six, three out of
        # three. Gated on the content actually looking quoted, so a quote-free
        # session pays nothing. Non-fatal: a failure here loses quotations, not
        # the extraction.
        if _recall_on and assistant_recall.wants_quote_pass(content):
            raw_items = await self._quote_pass(
                content=content, raw_items=raw_items, llm=llm,
                request_scope=request_scope, correlation_id=correlation_id,
            )

        processed: List[ExtractedMemory] = []
        for raw in raw_items:
            t = str(raw.get("type", "")).strip().lower()
            custom_name = custom_categories.canonical_name(t) if t else ""
            if custom_name in custom_names:
                mt = MemoryType.FACTS
            else:
                custom_name = None
                mt = type_by_value.get(t)
                if mt is None or mt.value not in allowed:
                    continue  # ignore items for a type we didn't request
            memory = await self._process_memory(
                raw=raw,
                memory_type=mt,
                document_id=document_id,
                categories=categories,
                request_scope=request_scope,
                category=custom_name,
            )
            if memory:
                processed.append(memory)

        set_current_span_attributes({
            "synap.raw_memories_parsed": len(raw_items),
            "synap.memories_accepted": len(processed),
        })
        return processed

    def _loads_merged(self, response: str):
        """Robustly json-load a merged-call response (tolerates code fences)."""
        text = (response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
        return None

    def _parse_merged_extraction(self, response: str) -> List[Dict[str, Any]]:
        """Parse the merged simple-group response into typed raw memories.

        Accepts either ``{"reasoning": ..., "extracted": [...]}`` or a bare
        array, tolerating markdown fences. Each item must carry ``content``.
        """
        data = self._loads_merged(response)
        if isinstance(data, dict):
            items = data.get("extracted", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        return [it for it in items if isinstance(it, dict) and it.get("content")]

    @staticmethod
    def _or_default(raw: Dict[str, Any], key: str, default: Any) -> Any:
        """``raw[key]`` unless it is missing OR explicitly null, then *default*.

        ⚠ ``raw.get(key, default)`` is NOT this. It returns None when the key is
        PRESENT with a null value, and three ExtractedMemory fields are declared
        with a plain type and a default — temporal_category: str = "perpetual",
        temporal_confidence: float = 0.0, is_episode: bool = False. A default
        makes a key optional to SUPPLY; it does not make None a valid value. So
        an explicit null failed validation and pydantic rejected the whole
        memory.

        That became live when the extraction prompts were changed
        to require every key "with null where you do not know the value", which
        is right for a model and was previously expressed by omitting the key.
        A burst of rejected memories lost each one with its document, with the
        ingest still reporting success.

        Null and absent mean the same thing here — "the model had nothing to
        say" — so both now take the default, which is exactly what absent did
        before. ``or`` would be wrong: 0.0 and False are legitimate values.
        """
        value = raw.get(key)
        return default if value is None else value

    async def _process_memory(
        self,
        raw: Dict[str, Any],
        memory_type: MemoryType,
        document_id: str,
        categories: List,
        request_scope=None,
        category: Optional[str] = None,
    ) -> Optional[ExtractedMemory]:
        """Process a single extracted memory. *category* (a custom category's name) wins
        over the stage-2 category."""
        content = raw.get("content", "")
        if not content:
            return None

        # Score confidence. Same null-vs-absent trap as the fields below:
        # ExtractedMemory declares `confidence: float` with a range check, so an
        # explicit null rejects the whole memory.
        confidence = self._or_default(raw, "confidence", 0.5)

        # PII handling happens at the ingress gate (see
        # Synap_PII_Protection_Feature_Spec_v1.md). By the time text reaches
        # this stage it has already been through it, so there is nothing to do
        # here. These two fields stay false/empty until the gate stamps them.
        pii_obfuscated = False
        pii_fields: List[str] = []

        # Destination depends on the request's scope level (derived at the API
        # boundary from the IDs passed + user_context_isolation). Confidence
        # threshold only splits user-scope memories into MEMORIES vs CANDIDATES.
        scope_level = getattr(request_scope, "scope_level", None) or "user"
        threshold = self._get_threshold(memory_type)
        if scope_level == "customer":
            destination = MemoryDestination.CUSTOMER_KNOWLEDGE
        elif scope_level == "client":
            destination = MemoryDestination.CLIENT_KNOWLEDGE
        else:
            destination = (
                MemoryDestination.MEMORIES
                if confidence >= threshold
                else MemoryDestination.CANDIDATES
            )

        # Get category context
        category = category or (categories[0].category if categories else None)
        subcategory = categories[0].subcategory if categories else None

        # Generate memory ID
        memory_id = uuid4()

        # Log memory creation with ID and content for debugging duplicate key issues
        logger.info(f"memory_created memory_id={memory_id} memory_type={memory_type.value} content={truncate_string(content, 150)} confidence={confidence} destination={destination.value} category={category} subcategory={subcategory}")

        # Parse temporal fields
        from synap.cloud.agents.ingestion.utils.timestamp_parser import parse_timestamp
        from synap.cloud.agents.ingestion.services.event_window import parse_event_day

        return ExtractedMemory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            destination=destination,
            category=category,
            subcategory=subcategory,
            source_document_id=document_id,
            source_span=raw.get("source_span"),
            assertion_source=_clean_assertion_source(raw.get("assertion_source")),
            pii_obfuscated=pii_obfuscated,
            pii_fields_obfuscated=pii_fields,
            temporal_category=self._or_default(raw, "temporal_category", "perpetual"),
            valid_from=parse_timestamp(raw.get("valid_from")) if raw.get("valid_from") else None,
            valid_until=parse_timestamp(raw.get("valid_until")) if raw.get("valid_until") else None,
            # Parsed leniently here (a bad value becomes None, never a lost
            # memory). event_window.validate_event_windows checks them against
            # DCT afterwards, which this method does not have.
            event_start=parse_event_day(raw.get("event_start")),
            event_end=parse_event_day(raw.get("event_end")),
            event_precision=raw.get("event_precision") if isinstance(raw.get("event_precision"), str) else None,
            # A date that depends on another event, kept for the resolver instead
            # of being guessed from DCT. Shape-checked here, not trusted: a
            # malformed anchor becomes None so it can never reach the writers.
            event_anchor=_anchor_for(raw, content),
            temporal_confidence=self._or_default(raw, "temporal_confidence", 0.0),
            is_episode=self._or_default(raw, "is_episode", False),
            episode_metadata=raw.get("episode_metadata"),
            hypothetical_questions=self._or_default(raw, "hypothetical_questions", []),
        )

    def _get_threshold(self, memory_type: MemoryType) -> float:
        """Get confidence threshold for memory type from MACA."""
        # Get the category config for this memory type
        category_configs = {
            MemoryType.FACTS: self._cfg.memory_categories.facts,
            MemoryType.PREFERENCES: self._cfg.memory_categories.preferences,
            MemoryType.EPISODES: self._cfg.memory_categories.episodes,
            MemoryType.EMOTIONS: self._cfg.memory_categories.emotions,
            MemoryType.TEMPORAL_EVENTS: self._cfg.memory_categories.temporal_events,
        }
        
        category_config = category_configs.get(memory_type)
        if category_config and hasattr(category_config, 'confidence_threshold'):
            return category_config.confidence_threshold
        
        # Fall back to extraction filters min_confidence_threshold
        return self._cfg.memory_extraction.filters.min_confidence_threshold

    def _is_type_enabled(self, memory_type: MemoryType) -> bool:
        """Check if memory type is enabled in MACA."""
        category_configs = {
            MemoryType.FACTS: self._cfg.memory_categories.facts,
            MemoryType.PREFERENCES: self._cfg.memory_categories.preferences,
            MemoryType.EPISODES: self._cfg.memory_categories.episodes,
            MemoryType.EMOTIONS: self._cfg.memory_categories.emotions,
            MemoryType.TEMPORAL_EVENTS: self._cfg.memory_categories.temporal_events,
        }
        
        category_config = category_configs.get(memory_type)
        return category_config.enabled if category_config else True

    # --- Content Chunking (for large docs that bypass Stage 0) ---

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks if it exceeds STAGE3_CHUNK_BUDGET.

        Returns a single-element list if no splitting is needed.
        Uses paragraph-based greedy accumulation (same algorithm as Stage 0).
        """
        counter = get_token_counter()
        token_count = counter.count(content)

        if token_count <= STAGE3_CHUNK_BUDGET:
            return [content]

        # Split on paragraph boundaries
        paragraphs = re.split(r"\n\s*\n", content)
        chunks: List[str] = []
        current_paras: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = counter.count(para)

            if current_tokens + para_tokens > STAGE3_CHUNK_BUDGET and current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_tokens = 0

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            chunks.append("\n\n".join(current_paras))

        # Force-split any chunk that still exceeds budget (single giant paragraph)
        final_chunks: List[str] = []
        for chunk in chunks:
            if counter.count(chunk) > STAGE3_CHUNK_BUDGET:
                final_chunks.extend(self._force_split_lines(chunk, counter))
            else:
                final_chunks.append(chunk)

        return final_chunks if len(final_chunks) > 1 else [content]

    @staticmethod
    def _force_split_lines(content: str, counter) -> List[str]:
        """Force-split a single oversized chunk by lines."""
        lines = content.split("\n")
        chunks: List[str] = []
        current_lines: List[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = counter.count(line)
            if current_tokens + line_tokens > STAGE3_CHUNK_BUDGET and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    @staticmethod
    def _generate_preceding_summary(existing_summary: str, chunk_text: str) -> str:
        """Build extractive preceding summary from chunk text.

        Takes first 2 sentences + last sentence, capped at 300 chars.
        Concatenates with existing summary, capped at 500 chars total.
        Same algorithm as Stage0Preprocessor._generate_preceding_summary().
        """
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return existing_summary

        selected = sentences[:2]
        if len(sentences) > 2:
            selected.append(sentences[-1])

        new_summary = " ".join(selected)
        if len(new_summary) > 300:
            new_summary = new_summary[:297] + "..."

        if existing_summary:
            combined = f"{existing_summary} | {new_summary}"
            if len(combined) > 500:
                combined = combined[:497] + "..."
            return combined

        return new_summary

    # --- Coalescing (one row per fact) ---
    #
    # Two injected calls rather than one combined prompt, so the MERGE half is
    # the prompt that already exists and has already earned its rules: preserve
    # every number, date and named entity character-for-character, and output at
    # least as long as the longest contributor. Only the GROUPING question is new.

    def _coalesce_group_fn(self, request_scope, correlation_id):
        async def group(contents: List[str], types: List[str]):
            from synap.cloud.agents.ingestion.prompts.coalesce import (
                COALESCE_SYSTEM_PROMPT, build_coalesce_user_prompt,
            )
            return await self._json_llm_call(
                system=COALESCE_SYSTEM_PROMPT,
                user=build_coalesce_user_prompt(contents, types),
                operation="ingestion.stage3_coalesce_group",
                request_scope=request_scope,
                correlation_id=correlation_id,
                max_tokens=1200,
            )
        return group

    def _coalesce_merge_fn(self, request_scope, correlation_id):
        async def merge(contents: List[str], types: List[str]):
            from synap.cloud.agents.ingestion.prompts.merge import (
                MERGE_SYSTEM_PROMPT, build_merge_user_prompt,
            )
            return await self._json_llm_call(
                system=MERGE_SYSTEM_PROMPT,
                user=build_merge_user_prompt(contents, types),
                operation="ingestion.stage3_coalesce_merge",
                request_scope=request_scope,
                correlation_id=correlation_id,
                max_tokens=600,
            )
        return merge

    async def _json_llm_call(self, *, system: str, user: str, operation: str,
                             request_scope, correlation_id, max_tokens: int):
        """One JSON call. Returns the parsed dict, or None on any failure.

        None is always a safe answer here: coalescing treats it as "no groups"
        and returns its input unchanged.
        """
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        async with start_span(f"{operation}", {
            "synap.prompt_len": len(system) + len(user),
            "synap.correlation_id": correlation_id or "",
        }):
            try:
                response = await self._llm.complete(
                    messages=messages,
                    telemetry_scope=request_scope,
                    telemetry_operation=operation,
                    telemetry_correlation_id=correlation_id,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    json_mode=True,
                )
            except TypeError:
                # Older LLMClient implementations without messages support, the
                # same fallback _extract_type uses.
                response = await self._llm.complete(system + "\n\n" + user)
            except Exception:  # noqa: BLE001 — never fails an ingest
                logger.warning("%s_failed correlation_id=%s", operation,
                               correlation_id or "", exc_info=True)
                return None
        text = response if isinstance(response, str) else getattr(
            response, "content", str(response))
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            logger.warning("%s_not_json: %s", operation, exc)
            return None

    # --- Post-Chunk Deduplication ---

    def _parse_extraction(
        self,
        llm_response: str,
        memory_type: MemoryType,
    ) -> List[Dict[str, Any]]:
        """Parse LLM extraction response. JSON-first with legacy regex fallback."""
        parsed = self._try_parse_json_extraction(llm_response)
        if parsed is not None:
            return parsed
        # Fall back to legacy line-based parser for backward compatibility
        return self._parse_extraction_legacy(llm_response, memory_type)

    def _try_parse_json_extraction(
        self,
        llm_response: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Try to parse the LLM response as a JSON array of extracted memories.

        Returns None if parsing fails (triggering legacy fallback).
        """
        text = llm_response.strip()

        # Track parsing method for observability
        parse_method = "raw_json"
        has_reasoning = "<reasoning>" in text

        # Extract content from <output> tags if present (reasoning compartment)
        output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if output_match:
            text = output_match.group(1).strip()
            parse_method = "output_tags"
        elif has_reasoning and "<output>" in text:
            # <output> tag opened but never closed — likely truncated
            output_start = text.find("<output>")
            text = text[output_start + len("<output>"):].strip()
            parse_method = "output_tags_truncated"
            logger.warning(f"Stage 3: Response has <output> tag but no closing tag — likely truncated response_length={len(llm_response)}")

        logger.info(f"Stage 3: Parsing extraction response parse_method={parse_method} has_reasoning={has_reasoning} text_length={len(text)}")

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Stage 3: JSON parse failed parse_method={parse_method} text_preview={text[:200]}")
            return None

        if not isinstance(raw, list):
            # json_mode + reasoning-first format produces {"reasoning": "...", "extracted": [...]}
            if isinstance(raw, dict):
                for key in ("extracted", "output", "memories", "results", "data"):
                    if key in raw and isinstance(raw[key], list):
                        logger.info(f"Stage 3: Unwrapped memory array from JSON object key='{key}' count={len(raw[key])}")
                        raw = raw[key]
                        break
                else:
                    # Try first list value
                    for k, v in raw.items():
                        if isinstance(v, list):
                            logger.info(f"Stage 3: Unwrapped memory array from JSON object key='{k}' count={len(v)}")
                            raw = v
                            break
                    else:
                        # Last resort: single-memory dict (json_mode often returns
                        # one object for single-item arrays). If it has a "content"
                        # key, wrap it into a 1-element list rather than dropping.
                        if "content" in raw:
                            logger.info(f"Stage 3: Wrapped single-memory dict into 1-element list keys={list(raw.keys())[:5]}")
                            raw = [raw]
                        else:
                            logger.warning(f"Stage 3: JSON response is dict with no list values keys={list(raw.keys())[:5]}")
                            return None
            else:
                return None

        memories = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue

            memory = {
                "content": content,
                "confidence": item.get("confidence", 0.5),
                "source_span": item.get("source_span", ""),
            }

            # Temporal fields (present when temporal addendum is active)
            if "temporal_category" in item:
                memory["temporal_category"] = item["temporal_category"]
            if "valid_from" in item:
                memory["valid_from"] = item["valid_from"]
            if "valid_until" in item:
                memory["valid_until"] = item["valid_until"]
            for key in ("event_start", "event_end", "event_precision"):
                if key in item:
                    memory[key] = item[key]
            if "temporal_confidence" in item:
                memory["temporal_confidence"] = item["temporal_confidence"]
            if "is_episode" in item:
                memory["is_episode"] = item["is_episode"]
            if "episode_metadata" in item:
                memory["episode_metadata"] = item["episode_metadata"]

            # Speaker attribution (present when the attribution addendum is
            # active); normalized/validated again in _process_memory.
            if "assertion_source" in item:
                memory["assertion_source"] = item["assertion_source"]

            memories.append(memory)

        logger.info(f"Stage 3: JSON extraction parse succeeded memories_count={len(memories)}")
        return memories

    def _parse_extraction_legacy(
        self,
        llm_response: str,
        memory_type: MemoryType,
    ) -> List[Dict[str, Any]]:
        """Legacy line-based parser for MEMORY/CONFIDENCE/SOURCE_SPAN/--- format.

        Kept as fallback during migration to JSON output format.
        """
        memories = []
        current = {}

        for line in llm_response.strip().split("\n"):
            line = line.strip()

            # Accumulate multi-line EPISODE_METADATA JSON
            if "_episode_json_partial" in current:
                if line.startswith("MEMORY:") or line == "---":
                    joined = "\n".join(current["_episode_json_partial"])
                    del current["_episode_json_partial"]
                    try:
                        current["episode_metadata"] = json.loads(joined)
                    except json.JSONDecodeError:
                        try:
                            current["episode_metadata"] = json.loads(joined.replace("'", '"'))
                        except json.JSONDecodeError:
                            current["episode_metadata"] = None
                else:
                    current["_episode_json_partial"].append(line)
                    joined = "\n".join(current["_episode_json_partial"])
                    if joined.count("{") <= joined.count("}") or len(current["_episode_json_partial"]) > 10:
                        del current["_episode_json_partial"]
                        try:
                            current["episode_metadata"] = json.loads(joined)
                        except json.JSONDecodeError:
                            try:
                                current["episode_metadata"] = json.loads(joined.replace("'", '"'))
                            except json.JSONDecodeError:
                                current["episode_metadata"] = None
                    continue

            if line.startswith("MEMORY:"):
                if current:
                    memories.append(current)
                current = {"content": line.split(":", 1)[1].strip()}
            elif line.startswith("CONFIDENCE:"):
                try:
                    current["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    current["confidence"] = 0.5
            elif line.startswith("SOURCE_SPAN:"):
                current["source_span"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("TEMPORAL_CATEGORY:"):
                current["temporal_category"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("VALID_FROM:"):
                val = line.split(":", 1)[1].strip().strip('"\'`')
                current["valid_from"] = val if val.lower() not in ("null", "none", "n/a", "") else None
            elif line.startswith("VALID_UNTIL:"):
                val = line.split(":", 1)[1].strip().strip('"\'`')
                current["valid_until"] = val if val.lower() not in ("null", "none", "n/a", "") else None
            elif line.startswith(("EVENT_START:", "EVENT_END:", "EVENT_PRECISION:")):
                key, val = line.split(":", 1)
                val = val.strip().strip('"\'`')
                current[key.lower()] = val if val.lower() not in ("null", "none", "n/a", "") else None
            elif line.startswith("TEMPORAL_CONFIDENCE:"):
                try:
                    current["temporal_confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    current["temporal_confidence"] = 0.0
            elif line.startswith("IS_EPISODE:"):
                current["is_episode"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("EPISODE_METADATA:"):
                val = line.split(":", 1)[1].strip().strip('`')
                if val.lower() in ("null", "none", "n/a", ""):
                    current["episode_metadata"] = None
                elif val.startswith("{") and val.count("{") > val.count("}"):
                    current["_episode_json_partial"] = [val]
                else:
                    try:
                        current["episode_metadata"] = json.loads(val)
                    except json.JSONDecodeError:
                        try:
                            current["episode_metadata"] = json.loads(val.replace("'", '"'))
                        except json.JSONDecodeError:
                            current["episode_metadata"] = None
            elif line == "---":
                if current:
                    current.pop("_episode_json_partial", None)
                    memories.append(current)
                    current = {}

        if current:
            current.pop("_episode_json_partial", None)
            memories.append(current)

        if memories:
            logger.info(f"Stage 3: Legacy parser fallback used memories_count={len(memories)}")
        return memories

    async def _extract_entities(
        self,
        content: str,
        memories: List[ExtractedMemory],
        request_scope=None,
        correlation_id: Optional[str] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities from content."""
        if not self._entity_extractor:
            logger.info("Stage 3: Entity extractor not available, skipping")
            return []

        domain_hints = self._cfg.memory_extraction.domain_hints
        logger.info(f"Stage 3: Calling entity extractor content_length={len(content)} domain_hints_industry={domain_hints.industry} domain_hints_key_entities={domain_hints.key_entities}")

        async with start_span("ingestion.stage3.extract_entities", {
            "synap.content_len": len(content),
            "synap.domain_industry": domain_hints.industry if hasattr(domain_hints, "industry") else None,
        }):
            try:
                t0 = time.monotonic()
                try:
                    entities = await self._entity_extractor.extract(
                        content,
                        domain_hints,
                        telemetry_scope=request_scope,
                        telemetry_correlation_id=correlation_id,
                        # The extractor builds its prompt once at startup from
                        # the blank config; hand it this call's instance config
                        # so entity types come from the instance's graph schemas.
                        config=_ACTIVE_MACA_CONFIG.get(),
                    )
                except TypeError:
                    entities = await self._entity_extractor.extract(content, domain_hints)
                set_current_span_attributes({
                    "synap.entities_found": len(entities),
                    "synap.entity_types": ",".join(sorted({e.entity_type for e in entities}))[:256],
                    "synap.extract_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                })
                logger.info(f"Stage 3: Entity extractor returned entities_count={len(entities)} entities={[{'name': e.name, 'type': e.entity_type, 'id': e.entity_id} for e in entities[:10]]}")
                return entities
            except Exception:
                set_current_span_attributes({"synap.entity_extract_error": True})
                logger.warning("Stage 3: Entity extraction failed", exc_info=True)
                return []

    async def _find_earlier_references(self, content, entities, request_scope, document_id):
        """Pronouns this document cannot resolve, grounded in the writer's
        earlier notes. [] when the feature is off, not wired, or finds nothing."""
        from synap.cloud.agents.ingestion.services import earlier_reference
        if self._earlier_references is None or not earlier_reference.enabled(
                getattr(request_scope, "instance_id", None)):
            return []
        try:
            return await self._earlier_references.resolve(
                content, [e.entity_type for e in entities], request_scope, document_id)
        except Exception:  # noqa: BLE001 — an enhancement, never a gate
            logger.warning("stage3_earlier_references_failed", exc_info=True)
            return []

    async def _reextract_with_references(self, content, entities, relationships, references,
                                         request_scope=None, correlation_id=None, writer=None,
                                         v2=False):
        """Read the document again, told who each pronoun means. The first read
        stands unless the second is at least as complete: a failed or garbled
        second call returns nothing, or only the referenced entities, and must
        never replace a graph that was already extracted."""
        if v2:
            new_entities, new_relationships = await self._extract_graph_v2(
                content, request_scope=request_scope, correlation_id=correlation_id,
                references=references)
        else:
            new_entities, new_relationships = await self._extract_graph(
                content,
                request_scope=request_scope,
                correlation_id=correlation_id,
                references=references,
                writer=writer,
            )
        if len(new_entities) < len(entities):
            logger.warning(
                "stage3_earlier_references_reread_discarded first_entities=%d second_entities=%d",
                len(entities), len(new_entities))
            return entities, relationships
        return new_entities, new_relationships

    @staticmethod
    def _with_referenced_entities(entities, references):
        """The entities, plus one for each grounded pronoun not already there."""
        if not references:
            return entities
        have = {e.name.lower() for e in entities}
        added = [
            ExtractedEntity(
                entity_id=str(uuid4()), entity_type=r.entity_type, name=r.name,
                properties={"from_earlier_note": True, "earlier_reference_pronoun": r.pronoun,
                            "earlier_reference_score": r.score},
            )
            for r in references if r.name.lower() not in have
        ]
        return list(entities) + added

    async def _extract_graph(
        self,
        content: str,
        request_scope=None,
        correlation_id: Optional[str] = None,
        references=None,
        writer: Optional[str] = None,
    ):
        """hybrid_graph: extract entities (nodes) + relationships (edges) in ONE
        call. Relationships are resolved against the entities from the SAME
        response, so every edge endpoint is guaranteed to be an extracted
        entity (no orphan edges). Returns (entities, relationships)."""
        from synap.cloud.agents.ingestion.prompts.extraction.merged_graph import (
            MergedGraphExtractionPrompt,
        )
        from synap.cloud.agents.ingestion.prompts.prompt_resolver import PromptResolver

        resolved = PromptResolver.resolve(MergedGraphExtractionPrompt)
        prompt_builder = resolved(self._cfg)
        system_prompt, user_prompt = prompt_builder.build(
            content=content,
            writer=writer,
            references=[(r.pronoun, r.name, r.entity_type) for r in references or []] or None,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Entity extraction is Tier 1 (light client); relationships share the
        # same reading — use the light client to match the standalone entity call.
        llm = self._llm_graph or self._llm_light or self._llm

        async with start_span("ingestion.stage3.extract_graph", {
            "synap.content_len": len(content),
            "synap.correlation_id": correlation_id or "",
        }):
            t0 = time.monotonic()
            try:
                try:
                    response = await llm.complete(
                        messages=messages,
                        telemetry_scope=request_scope,
                        telemetry_operation="ingestion.stage3_extract_graph",
                        telemetry_correlation_id=correlation_id,
                        max_tokens=8000,
                        json_mode=True,
                    )
                except TypeError:
                    response = await llm.complete(system_prompt + "\n\n" + user_prompt)
            except Exception:
                set_current_span_attributes({"synap.graph_extract_error": True})
                logger.warning("Stage 3: Merged graph extraction failed", exc_info=True)
                return [], []

            entities = self._parse_graph_entities(response)
            if references:
                # The model was told to return these; make sure of it, so a
                # relationship that uses one is not dropped as unknown.
                named = {r.name.lower(): r for r in references}
                for e in entities:
                    r = named.get(e.name.lower())
                    if r is not None:
                        e.properties.update({"from_earlier_note": True,
                                             "earlier_reference_pronoun": r.pronoun,
                                             "earlier_reference_score": r.score})
                entities = self._with_referenced_entities(entities, references)
            if writer:
                from synap.cloud.agents.ingestion.services import writer_entity
                entities = writer_entity.stamped(entities, request_scope)
            relationships = (
                self._parse_relationships(response, entities) if len(entities) >= 2 else []
            )
            if writer:
                entities = writer_entity.without_unused(entities, relationships)
            set_current_span_attributes({
                "synap.entities_found": len(entities),
                "synap.relationships_found": len(relationships),
                "synap.llm_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })
            logger.info(
                f"Stage 3: Merged graph extraction complete entities={len(entities)} "
                f"relationships={len(relationships)} entity_names={[e.name for e in entities[:10]]}"
            )
            return entities, relationships

    async def _extract_graph_v2(
        self,
        content: str,
        request_scope=None,
        correlation_id: Optional[str] = None,
        references=None,
    ):
        """Entities and relationships in one call, with the second graph prompt
        (prompts/extraction/graph_v2, read by services/graph_v2). Returns
        (entities, relationships); ([], []) on any failure."""
        from synap.cloud.agents.ingestion.prompts.extraction.graph_v2 import GraphExtractionPromptV2
        from synap.cloud.agents.ingestion.services import graph_v2, writer_entity

        instance_id = getattr(request_scope, "instance_id", None)
        has_user = bool(getattr(request_scope, "user_id", None))
        customer = writer_entity.customer_label_for(content, request_scope)
        system_prompt, user_prompt = GraphExtractionPromptV2(self._cfg).build(
            content=content,
            writer=writer_entity.LABEL,
            customer=customer,
            references=[(r.pronoun, r.name, r.entity_type) for r in references or []] or None,
            events=graph_v2.events_enabled(instance_id),
            relationship_examples=graph_v2.type_examples(),
            statements=graph_v2.statements_enabled(instance_id),
            scope_kind=self._scope_kind(request_scope),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm = self._llm_graph or self._llm_light or self._llm

        async with start_span("ingestion.stage3.extract_graph", {
            "synap.content_len": len(content),
            "synap.correlation_id": correlation_id or "",
            "synap.graph_prompt": "v2",
            "synap.graph_maca_patterns": "<client_patterns>" in system_prompt,
        }):
            t0 = time.monotonic()
            try:
                try:
                    response = await llm.complete(
                        messages=messages,
                        telemetry_scope=request_scope,
                        telemetry_operation="ingestion.stage3_extract_graph",
                        telemetry_correlation_id=correlation_id,
                        max_tokens=int(os.environ.get("SYNAP_GRAPH_V2_MAX_TOKENS", "8000")),
                        json_mode=True,
                    )
                except TypeError:
                    response = await llm.complete(system_prompt + "\n\n" + user_prompt)
                data = self._loads_merged(response)
                if data is None and (response or "").strip():
                    # Most often the reply ran out of tokens part-way through the
                    # JSON. The whole note's graph is lost, so say so.
                    logger.warning(
                        "Stage 3: graph v2 reply is not valid JSON (truncated?) chars=%d; "
                        "raise SYNAP_GRAPH_V2_MAX_TOKENS if this repeats", len(response))
                parsed = graph_v2.parse(
                    data, content,
                    writer=writer_entity.LABEL if has_user else None, customer=customer)
                parsed.response_chars = len(response or "")

                # The writer and the writer's customer: one node each, keyed on
                # an id. A document nobody wrote (no user id) has neither.
                fixed = {writer_entity.LABEL, writer_entity.CUSTOMER_LABEL}
                gone = {
                    e.entity_id for e in parsed.entities
                    if e.name.strip().lower() in fixed and not (
                        has_user and (e.name.strip().lower() == writer_entity.LABEL or customer))
                }
                if gone:
                    parsed.entities = [e for e in parsed.entities if e.entity_id not in gone]
                    parsed.relationships = [
                        r for r in parsed.relationships
                        if r.source_entity_id not in gone and r.target_entity_id not in gone]
                if has_user:
                    for e in parsed.entities:
                        if e.name.strip().lower() == writer_entity.LABEL:
                            e.name = writer_entity.LABEL
                            writer_entity._mark(e, request_scope)
                    if customer:
                        parsed.entities = writer_entity.customer_stamped(parsed.entities, request_scope)

                if references:
                    named = {r.name.lower(): r for r in references}
                    for e in parsed.entities:
                        r = named.get(e.name.lower())
                        if r is not None:
                            e.properties.update({"from_earlier_note": True,
                                                 "earlier_reference_pronoun": r.pronoun,
                                                 "earlier_reference_score": r.score})
                    parsed.entities = self._with_referenced_entities(parsed.entities, references)

                parsed = await graph_v2.confident(
                    parsed, content, {e.entity_id: e.name for e in parsed.entities},
                    notes=" ".join(f'In this note "{r.pronoun}" means {r.name}.' for r in references or []),
                    customer_id=next((e.entity_id for e in parsed.entities if writer_entity.is_customer(e)), None))
                entities = writer_entity.without_unused(parsed.entities, parsed.relationships)
                if graph_v2.drop_lone_enabled(instance_id):
                    entities, lone = graph_v2.without_lone(entities, parsed.relationships)
                    parsed.lone = [e.name for e in lone]
            except Exception:
                set_current_span_attributes({"synap.graph_extract_error": True})
                logger.warning("Stage 3: graph extraction (v2) failed", exc_info=True)
                return [], []

            self._graph_v2_last = parsed          # read by the scenario harness
            set_current_span_attributes({
                "synap.entities_found": len(entities),
                "synap.relationships_found": len(parsed.relationships),
                "synap.relationships_dropped": len(parsed.dropped),
                "synap.entities_dropped_lone": len(parsed.lone),
                "synap.entities_from_relationship_ends": parsed.added_ends,
                "synap.llm_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })
            logger.info(
                f"Stage 3: graph extraction (v2) complete entities={len(entities)} "
                f"relationships={len(parsed.relationships)} dropped={parsed.dropped} lone_dropped={parsed.lone}"
            )
            return entities, parsed.relationships

    def _parse_graph_entities(self, response: str) -> List[ExtractedEntity]:
        """Parse the 'entities' array from the merged graph response into
        ExtractedEntity objects (same shape as the dedicated entity extractor)."""
        allowed = {"person", "organization", "location", "product", "technology", "event"}
        # The node types the instance configured are entities here too. The
        # prompt is already told to prioritise them (MergedGraphExtractionPrompt.build);
        # this list used to drop them again after the model returned them.
        try:
            from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
                _schema_or_hint_entity_types,
            )
            allowed |= {t.strip().lower() for t in _schema_or_hint_entity_types(self._cfg) if t}
        except Exception:  # noqa: BLE001 — a config problem must not stop extraction
            logger.warning("Stage 3: configured entity types unavailable", exc_info=True)
        data = self._loads_merged(response)
        raw = data.get("entities", []) if isinstance(data, dict) else []
        entities: List[ExtractedEntity] = []
        seen = set()
        for e in raw:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            etype = str(e.get("entity_type", "")).strip().lower()
            if not name or etype not in allowed:
                continue
            key = (etype, name.lower())
            if key in seen:
                continue
            seen.add(key)
            entities.append(ExtractedEntity(
                entity_id=str(uuid4()),
                entity_type=etype,
                name=name,
                source_span=e.get("source_span"),
                properties=_model_properties(e.get("properties")),
            ))
        return entities

    async def _extract_relationships(
        self,
        content: str,
        entities: List[ExtractedEntity],
        request_scope=None,
        correlation_id: Optional[str] = None,
        references=None,
        writer: Optional[ExtractedEntity] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships between entities using LLM.

        *writer*: the writer of the content as an entity (services/writer_entity).
        It is not in *entities*; the prompt adds it to the known list itself."""
        if len(entities or []) + (1 if writer is not None else 0) < 2:
            logger.info(f"Stage 3: Skipping relationship extraction (need >= 2 entities) entities_count={len(entities) if entities else 0}")
            return []

        if not self._entity_extractor:
            logger.info("Stage 3: Skipping relationship extraction (no entity extractor)")
            return []

        from synap.cloud.agents.ingestion.prompts.extraction.relationships import (
            RelationshipExtractionPrompt,
        )
        from synap.cloud.agents.ingestion.prompts.prompt_resolver import PromptResolver

        domain_hints = self._cfg.memory_extraction.domain_hints

        resolved_class = PromptResolver.resolve(RelationshipExtractionPrompt)
        prompt_builder = resolved_class(self._cfg)
        system_prompt, user_prompt = prompt_builder.build(
            content=content,
            entities=entities,
            domain_hints=domain_hints,
            writer=writer.name if writer is not None else None,
            references=[(r.pronoun, r.name) for r in references or []] or None,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"Stage 3: Calling LLM for relationship extraction entities_count={len(entities)} entity_names={[e.name for e in entities[:10]]} prompt_length={len(system_prompt) + len(user_prompt)} domain_hints_key_relationships={domain_hints.key_relationships}")

        async with start_span("ingestion.stage3.extract_relationships", {
            "synap.entity_count": len(entities),
            "synap.prompt_len": len(system_prompt) + len(user_prompt),
            "synap.correlation_id": correlation_id or "",
        }):
            try:
                t0 = time.monotonic()
                try:
                    response = await (self._llm_graph or self._llm).complete(
                        messages=messages,
                        telemetry_scope=request_scope,
                        telemetry_operation="ingestion.stage3_relationship_extract",
                        telemetry_correlation_id=correlation_id,
                        max_tokens=8000,
                        json_mode=True,
                    )
                except TypeError:
                    # Fallback for older LLMClient implementations without messages support
                    prompt = system_prompt + "\n\n" + user_prompt
                    response = await self._llm.complete(prompt)
                logger.info(f"Stage 3: LLM relationship response received, parsing response_length={len(response)} response_preview={response[:300]}")
                relationships = self._parse_relationships(
                    response, list(entities) + ([writer] if writer is not None else []))
                set_current_span_attributes({
                    "synap.relationships_found": len(relationships),
                    "synap.relationship_types": ",".join(sorted({r.relationship_type for r in relationships}))[:256],
                    "synap.llm_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                })
                logger.info(f"Stage 3: Relationship extraction complete relationships_count={len(relationships)} relationship_types={list(set(r.relationship_type for r in relationships))}")
                return relationships
            except Exception:
                set_current_span_attributes({"synap.relationship_extract_error": True})
                logger.warning("Stage 3: Relationship extraction failed", exc_info=True)
                return []

    def _parse_relationships(
        self,
        llm_response: str,
        entities: List[ExtractedEntity],
    ) -> List[ExtractedRelationship]:
        """Parse JSON relationship array from LLM response."""
        import json

        # Build name -> entity_id map for resolving names to IDs
        name_to_id: Dict[str, str] = {}
        for e in entities:
            name_to_id[e.name.lower()] = e.entity_id
            if e.canonical_name:
                name_to_id[e.canonical_name.lower()] = e.entity_id

        # Extract content from <output> tags if present (reasoning compartment)
        text = llm_response.strip()
        has_reasoning = "<reasoning>" in text
        parse_method = "raw_json"

        output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if output_match:
            text = output_match.group(1).strip()
            parse_method = "output_tags"
        elif has_reasoning and "<output>" in text:
            output_start = text.find("<output>")
            text = text[output_start + len("<output>"):].strip()
            parse_method = "output_tags_truncated"
            logger.warning(f"Stage 3: Relationship response has <output> tag but no closing tag — likely truncated response_length={len(llm_response)}")

        logger.info(f"Stage 3: Parsing relationship response parse_method={parse_method} has_reasoning={has_reasoning} text_length={len(text)}")

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            raw_rels = json.loads(text)
        except json.JSONDecodeError:
            # LLM may add preamble text around the JSON array — extract it
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start != -1 and bracket_end > bracket_start:
                json_slice = text[bracket_start:bracket_end + 1]
                try:
                    raw_rels = json.loads(json_slice)
                except json.JSONDecodeError:
                    try:
                        raw_rels = json.loads(json_slice.replace("'", '"'))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse relationship extraction JSON raw_response={text[:500]}")
                        return []
            else:
                logger.warning(f"Failed to parse relationship extraction JSON raw_response={text[:500]}")
                return []

        if not isinstance(raw_rels, list):
            # json_mode + reasoning-first format produces {"reasoning": "...", "extracted": [...]}
            if isinstance(raw_rels, dict):
                for key in ("extracted", "output", "relationships", "results", "data"):
                    if key in raw_rels and isinstance(raw_rels[key], list):
                        logger.info(f"Stage 3: Unwrapped relationship array from JSON object key='{key}' count={len(raw_rels[key])}")
                        raw_rels = raw_rels[key]
                        break
                else:
                    for k, v in raw_rels.items():
                        if isinstance(v, list):
                            logger.info(f"Stage 3: Unwrapped relationship array from JSON object key='{k}' count={len(v)}")
                            raw_rels = v
                            break
                    else:
                        logger.warning(f"Stage 3: Relationship JSON is dict with no list values keys={list(raw_rels.keys())[:5]}")
                        return []
            else:
                return []

        from synap.cloud.agents.ingestion.services import relationship_type

        relationships = []
        for raw in raw_rels:
            if not isinstance(raw, dict):
                continue

            # A model can return null for a field it is unsure of; that is a
            # relationship to skip, not a reason to lose the note's graph.
            source_name = str(raw.get("source_entity") or "").strip().lower()
            target_name = str(raw.get("target_entity") or "").strip().lower()
            rel_type = relationship_type.storable(raw.get("relationship_type"))

            source_id = name_to_id.get(source_name)
            target_id = name_to_id.get(target_name)

            if not source_id or not target_id:
                # Skip relationships referencing unknown entities
                continue

            if source_id == target_id:
                # Skip self-referencing relationships
                continue

            relationships.append(
                ExtractedRelationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel_type,
                    properties=raw.get("properties") if isinstance(raw.get("properties"), dict) else {},
                )
            )

        return relationships

    @staticmethod
    def _apply_entity_grounding(
        memories: List[ExtractedMemory],
        entities: List[ExtractedEntity],
        request_scope,
        correlation_id: Optional[str],
    ) -> List[ExtractedMemory]:
        """Drop memories that name no entity of this text (require_entity_grounding).

        The writer's label counts whenever the note has a writer: the graph step
        removes the writer's node when no relationship uses it, and a memory
        "The user likes tea" is about a named entity either way. When the graph
        produced no entities (it failed, was off, or found none) everything is
        kept: an extraction failure must not erase a conversation. Logs counts
        only, never memory text.
        """
        instance_id = getattr(request_scope, "instance_id", None)
        if not memories:
            return memories
        if not entities:
            logger.warning(
                "stage3_entity_grounding_skipped reason=no_entities memories=%d "
                "instance=%s correlation_id=%s", len(memories), instance_id,
                correlation_id or "")
            return memories
        from synap.cloud.agents.ingestion.services import writer_entity
        names = {e.name.lower() for e in entities if e.name and e.name.strip()}
        if getattr(request_scope, "user_id", None):
            names.add(writer_entity.LABEL)
        kept, dropped = guardrails.drop_ungrounded(memories, sorted(names), _names_entity)
        logger.info(
            "stage3_entity_grounding dropped=%d kept=%d instance=%s correlation_id=%s",
            dropped, len(kept), instance_id, correlation_id or "")
        return kept

    def _link_memories_to_entities(
        self,
        memories: List[ExtractedMemory],
        entities: List[ExtractedEntity],
    ) -> None:
        """Link memories to mentioned entities using word-boundary matching."""
        entity_names = {e.name.lower(): e.entity_id for e in entities}
        from synap.cloud.agents.ingestion.services import writer_entity
        writer_id = next((e.entity_id for e in entities if writer_entity.is_writer(e)), None)
        customer_id = next((e.entity_id for e in entities if writer_entity.is_customer(e)), None)

        # Sort by name length descending so longer names match first
        sorted_names = sorted(entity_names.keys(), key=len, reverse=True)

        for memory in memories:
            mentioned = []
            content_lower = memory.content.lower()
            for name in sorted_names:
                if _names_entity(name, content_lower):
                    mentioned.append(entity_names[name])
            # The memory prompts write "we were charged twice" as "The user was
            # charged twice", so the customer's own node is never named in a
            # memory. A memory that names the writer is linked to it too.
            if customer_id and writer_id in mentioned and customer_id not in mentioned:
                mentioned.append(customer_id)
            memory.entities_mentioned = mentioned


def _names_entity(name: str, content_lower: str) -> bool:
    """True when the lower-cased entity *name* appears in the lower-cased text as a
    whole word. The one rule for "this memory names that entity", used by the
    entity linker and by the grounding check."""
    from synap.cloud.agents.ingestion.services import writer_entity
    # Use word boundary matching to avoid substring false-positives
    # e.g. "maximem" should not match inside "maximem ai"
    pattern = r'\b' + re.escape(name) + r'\b'
    if name == writer_entity.LABEL:
        # "The user's company was charged twice" is about the customer.
        pattern += r"(?!['\u2019]s\s+company)"
    return re.search(pattern, content_lower) is not None


def _model_properties(value):
    from synap.cloud.agents.ingestion.services.writer_entity import model_properties
    return model_properties(value)
