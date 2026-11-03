"""Shared provider factory functions for RetrievalManager initialization.

Provides unified access to storage providers (Postgres, ChromaDB, Neo4j, Redis)
used by both ingestion and retrieval subsystems.
"""

import asyncio
import os
import logging
from typing import Optional
import redis.asyncio as redis

from synap.cloud.infrastructure.llm.models import LLMProvider
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)


# Detached usage-metering tasks kept alive until completion so CPython does not
# GC them mid-flight. Retrieval LLM usage tracking (2 telemetry DB writes) is
# scheduled here so it never sits on an LLM call's response critical path.
_LLM_USAGE_TASKS: set = set()


def _schedule_llm_usage(usage_tracker, **kwargs) -> None:
    """Record retrieval LLM usage off the caller's critical path.

    ``track_llm_usage`` issues two sequential telemetry INSERT+COMMIT round-trips.
    Awaiting it inline previously added that DB latency to every retrieval LLM
    call (fast-mode filter; accurate-mode subquery-gen + filter). This mirrors
    how ``LLMConfigManager`` already emits its own telemetry non-blocking. The
    metering still runs — just detached — and its own errors are swallowed here
    so a metering failure can never affect an LLM result.
    """
    async def _emit():
        try:
            await usage_tracker.track_llm_usage(**kwargs)
            logger.info(
                "cloud_usage_track_llm_usage operation=%s client_id=%s instance_id=%s "
                "input_tokens=%s output_tokens=%s model=%s correlation_id=%s",
                kwargs.get("operation"), kwargs.get("client_id"), kwargs.get("instance_id"),
                kwargs.get("input_tokens"), kwargs.get("output_tokens"),
                kwargs.get("model"), kwargs.get("correlation_id"),
            )
        except Exception as e:  # metering must never fail an LLM result
            logger.warning(
                "cloud_usage_track_llm_usage_failed operation=%s client_id=%s instance_id=%s error=%s",
                kwargs.get("operation"), kwargs.get("client_id"), kwargs.get("instance_id"), e,
            )

    try:
        task = asyncio.create_task(_emit())
    except RuntimeError:
        # No running loop (sync/test context) — run inline rather than drop it.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_emit())
        finally:
            loop.close()
        return
    _LLM_USAGE_TASKS.add(task)
    task.add_done_callback(_LLM_USAGE_TASKS.discard)


async def get_redis_provider() -> Optional[redis.Redis]:
    """Get Redis client for caching.

    Returns:
        Redis client or None if unavailable
    """
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        logger.info(f"Redis provider initialized: {redis_url}")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


def get_postgres_provider():
    """Get Postgres provider for short-term context queries.

    Note: Uses SQLAlchemy session factory, not a separate connection.
    RetrievalManager expects interface with fetch_all(query: SQLQuery).

    Returns:
        Postgres provider wrapper or None
    """
    try:
        from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper

        wrapper = SQLAlchemyRDBMSWrapper()
        logger.info("Postgres provider initialized (SQLAlchemy)")
        return wrapper
    except Exception as e:
        logger.warning(f"Postgres provider unavailable: {e}")
        return None


def get_vector_provider():
    """Get ChromaDB vector provider for semantic search.

    Returns:
        AsyncVectorWrapper or None if ChromaDB unavailable
    """
    chroma_host = os.getenv("CHROMA_HOST")
    if not chroma_host:
        logger.info("ChromaDB disabled (CHROMA_HOST not set)")
        return None

    try:
        from synap.providers.vector.adapters.chromadb import ChromaDBAdapter
        from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper

        adapter = ChromaDBAdapter({
            "host": chroma_host,
            "port": int(os.getenv("CHROMA_PORT", "8000")),
            "auth_token": os.getenv("CHROMA_AUTH_TOKEN"),
        })
        adapter.connect()
        wrapper = AsyncVectorWrapper(provider=adapter)
        logger.info(f"ChromaDB vector provider initialized: {chroma_host}")
        return wrapper
    except Exception as e:
        logger.warning(f"ChromaDB unavailable: {e}")
        return None


def get_embedding_provider():
    """Get embedding client for text → vector conversion.

    Returns:
        Embedding client with embed(text) method, or None
    """
    from synap.cloud.application.ingestion.embedding_client import UnpinnedEmbeddingModelError
    try:
        from synap.cloud.application.ingestion.startup import _get_embedding_client
        client = _get_embedding_client()
        logger.info("Embedding provider initialized")
        return client
    except UnpinnedEmbeddingModelError:
        # A model with no exact revision is refused, not skipped.
        raise
    except Exception as e:
        logger.warning(f"Embedding provider unavailable: {e}")
        return None


def get_compaction_storage():
    """Get CompactionStorage for compacted context access.

    Returns:
        CompactionStorage instance or None if unavailable
    """
    try:
        from synap.cloud.application.compaction.storage import CompactionStorage
        storage = CompactionStorage()
        logger.info("Compaction storage initialized")
        return storage
    except Exception as e:
        logger.warning(f"Compaction storage unavailable: {e}")
        return None


def get_compaction_manager_provider():
    """Get CompactionManager for compaction triggering from retrieval.

    Returns:
        CompactionManager instance or None if unavailable
    """
    try:
        from synap.cloud.application.compaction.manager import get_compaction_manager
        manager = get_compaction_manager()
        logger.info("Compaction manager initialized")
        return manager
    except Exception as e:
        logger.warning(f"Compaction manager unavailable: {e}")
        return None


def get_llm_provider():
    """Get LLM provider for retrieval subquery generation (ACCURATE mode).

    Returns a wrapper that bridges LLMConfigManager to the calling convention
    used by RetrievalManager._generate_retrieval_subqueries():
        response = await llm_client.complete(messages=[...], temperature=..., max_tokens=...)
        content = response.get("content", "")

    Returns:
        _RetrievalLLMClient or None if LLM unavailable. Always None in Core
        (``edition.retrieval_llm`` is False): retrieval makes no LLM call there.
    """
    if not edition.retrieval_llm:
        logger.info("Retrieval LLM client not built (this edition makes no retrieval LLM call)")
        return None
    try:
        from synap.cloud.infrastructure.llm import get_llm_manager
        from synap.cloud.infrastructure.llm.call_trace import trace_scope
        from synap.cloud.application.analytics.tracker import get_usage_tracker

        manager = get_llm_manager()

        class _RetrievalLLMClient:
            """Bridges LLMConfigManager → retrieval_manager's calling convention."""

            # Groq for fast retrieval LLM calls. Was llama-3.3-70b-versatile
            # until Groq decommissioned it on 2026-08-16 (announced 2026-06-17),
            # after which every call here 404'd and fell through manager.py's
            # ProviderError fallback. openai/gpt-oss-120b is Groq's own
            # recommended replacement and is already the model behind
            # anticipation, playground and context-decision.
            # https://console.groq.com/docs/deprecations
            _PROVIDER = LLMProvider.GROQ
            _MODEL = "openai/gpt-oss-120b"

            def __init__(self, llm_manager):
                self._manager = llm_manager
                self._usage_tracker = get_usage_tracker()

            async def complete(
                self,
                messages,
                temperature=0.3,
                max_tokens=2000,
                json_mode=True,
                telemetry_scope=None,
                telemetry_operation: Optional[str] = None,
                telemetry_correlation_id: Optional[str] = None,
                **kwargs,
            ):
                # Forward reasoning_effort only for models that support it
                reasoning_effort = kwargs.pop("reasoning_effort", None)
                extra_kwargs = {}
                if reasoning_effort is not None:
                    extra_kwargs["reasoning_effort"] = reasoning_effort

                # AI-2: bound the read time on retrieval Groq calls so a stalled
                # call fails fast (and is retried once in the provider) instead of
                # hanging to the fetch ceiling. Per-operation: subquery-gen uses a
                # larger max_tokens and legitimately runs longer than the filter.
                # Opt-in only — compaction's slower gpt-oss calls never set this.
                _op = telemetry_operation or ""
                if "subquery" in _op:
                    extra_kwargs["read_timeout_s"] = float(
                        os.environ.get("RETRIEVAL_LLM_SUBQUERY_TIMEOUT_S", "8.0")
                    )
                else:
                    extra_kwargs["read_timeout_s"] = float(
                        os.environ.get("RETRIEVAL_LLM_FILTER_TIMEOUT_S", "5.0")
                    )

                import time as _time
                _llm_start = _time.time()
                # Usage for this call is recorded below, not by the manager,
                # so the scope is handed to the trace alone: passing
                # client_id and instance_id into complete() would record the
                # usage twice.
                with trace_scope(
                    client_id=getattr(telemetry_scope, "client_id", None),
                    instance_id=getattr(telemetry_scope, "instance_id", None),
                    customer_id=getattr(telemetry_scope, "customer_id", None),
                    user_id=getattr(telemetry_scope, "user_id", None),
                    operation=telemetry_operation or "retrieval.llm",
                    correlation_id=telemetry_correlation_id,
                ):
                    response = await self._manager.complete(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        force_provider=self._PROVIDER,
                        force_model=self._MODEL,
                        **extra_kwargs,
                    )
                _llm_ms = round((_time.time() - _llm_start) * 1000, 1)
                logger.info(
                    "retrieval_llm_call provider=%s model=%s operation=%s round_trip_ms=%s in_toks=%s out_toks=%s json_mode=%s",
                    self._PROVIDER.value, self._MODEL,
                    telemetry_operation or "unknown", _llm_ms,
                    response.input_tokens, response.output_tokens, json_mode,
                )

                if telemetry_scope is not None:
                    client_id = str(getattr(telemetry_scope, "client_id", "") or "")
                    instance_id = str(getattr(telemetry_scope, "instance_id", "") or "")
                    if client_id and instance_id:
                        # Detach usage metering from the LLM critical path: the two
                        # telemetry DB writes inside track_llm_usage otherwise block
                        # every retrieval LLM round-trip's return.
                        _schedule_llm_usage(
                            self._usage_tracker,
                            client_id=client_id,
                            instance_id=instance_id,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            model=response.model,
                            operation=telemetry_operation or "retrieval.llm",
                            correlation_id=telemetry_correlation_id,
                            cached_input_tokens=getattr(
                                response, "cached_input_tokens", 0
                            ),
                        )
                    else:
                        logger.info(
                            "cloud_usage_track_llm_usage_skipped reason=missing_scope operation=%s correlation_id=%s",
                            telemetry_operation or "retrieval.llm",
                            telemetry_correlation_id,
                        )
                return {"content": response.content or ""}

        logger.info("LLM provider initialized for retrieval")
        return _RetrievalLLMClient(manager)
    except Exception as e:
        logger.warning(f"LLM provider unavailable: {e}")
        return None


def get_graph_provider():
    """Get Neo4j graph provider for entity relationship queries.

    Returns:
        AsyncGraphWrapper or None if Neo4j unavailable
    """
    neo4j_uri = os.getenv("NEO4J_URI")
    if not neo4j_uri:
        logger.info("Neo4j disabled (NEO4J_URI not set)")
        return None

    try:
        from synap.providers.graph.adapters.neo4j import Neo4jAdapter
        from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper

        adapter = Neo4jAdapter({
            "uri": neo4j_uri,
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", ""),
        })
        adapter.connect()
        wrapper = AsyncGraphWrapper(provider=adapter)
        logger.info(f"Neo4j graph provider initialized: {neo4j_uri}")
        return wrapper
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")
        return None


# Module-level singleton cache to avoid re-creating providers per request
_providers_cache = {
    "redis": None,
    "postgres": None,
    "vector": None,
    "graph": None,
    "embedding": None,
    "llm": None,
    "compaction_storage": None,
    "compaction_manager": None,
    "ingestion_manager": None,
    "id_mapper": None,
    "initialized": False,
}


def _get_ingestion_manager(redis_client):
    """Create a lightweight IngestionManager for long-term promotion.

    Only needs the Redis-backed queue, idempotency, rate limiter, and status
    tracker.  The existing ingestion workers (started separately in main.py)
    pick up whatever gets pushed to the queues.
    """
    if redis_client is None:
        return None
    try:
        from synap.cloud.application.ingestion.manager import IngestionManager
        from synap.cloud.application.ingestion.queue import IngestionQueue
        from synap.cloud.application.ingestion.idempotency import IdempotencyChecker
        from synap.cloud.application.ingestion.rate_limiter import RateLimiter
        from synap.cloud.application.ingestion.status import StatusTracker
        from synap.cloud.application.ingestion.validators import RequestValidator
        from synap.cloud.application.ingestion.storage_provider import FileStoreStorageProvider

        mgr = IngestionManager(
            queue=IngestionQueue(redis_client),
            idempotency=IdempotencyChecker(redis_client),
            rate_limiter=RateLimiter(redis_client),
            status_tracker=StatusTracker(redis_client),
            validator=RequestValidator(max_content_size_mb=10),
            storage_provider=FileStoreStorageProvider(),
        )
        logger.info("IngestionManager initialized for long-term promotion")
        return mgr
    except Exception as e:
        logger.warning(f"IngestionManager unavailable for promotion: {e}")
        return None


async def resolve_compaction_promotion_scope(event):
    """Resolve the ingestion Scope for a compaction-promotion event.

    The event's scope fields are populated verbatim by whichever caller
    triggered the compaction, and the retrieval-triggered path passes a MIXED
    pair — customer_id already RESOLVED (uuid5) + user_id still external.
    Resolving that mix re-hashes the customer (uuid5-of-uuid5) and fails the
    B2C user==customer check, minting a phantom scope no retrieval ever reads
    (promoted memories landed under the wrong customer and user ids). The conversation row
    stores the EXTERNAL ids — the same source the tracker/worker trigger
    paths pass — so it is authoritative here; event fields are only the
    fallback when the row is missing.

    Returns a Scope, or None when no customer can be determined (caller
    should skip promotion — the error is logged here).
    """
    from synap.cloud.agents.ingestion.models.inputs import Scope, ScopeRung
    from synap.cloud.shared.scope import parse_or_deterministic_uuid

    # Resolve client_id: the conversation table doesn't store it, so
    # event.client_id may be None.  Look it up from the instance table to
    # ensure memories land in the correct ChromaDB collection
    # (synap_{client_id}).  Without this, we'd fall back to instance_id
    # which produces a DIFFERENT collection than retrieval queries.
    raw_client = event.client_id
    if not raw_client and event.instance_id:
        try:
            from synap.cloud.database.connection import get_session_factory
            session_factory = get_session_factory()
            async with session_factory() as session:
                from sqlalchemy import text as _text
                result = await session.execute(
                    _text("SELECT client_id FROM instance WHERE instance_id = :iid"),
                    {"iid": event.instance_id},
                )
                row = result.first()
                if row:
                    raw_client = row[0]
        except Exception as e:
            logger.debug(f"Could not resolve client_id from instance: {e}")
    raw_client = raw_client or event.instance_id  # final fallback

    # Conversation row = source of truth for the external customer/user ids.
    raw_customer = event.customer_id
    raw_user = event.user_id
    try:
        from synap.cloud.shared.conversation_id_norm import coerce_conversation_id
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text as _text
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                _text(
                    "SELECT user_id, customer_id FROM conversation "
                    "WHERE conversation_id = :cid"
                ),
                {"cid": coerce_conversation_id(event.conversation_id)},
            )
            row = result.first()
        if row and row[1]:
            if raw_customer != row[1] or (raw_user or "") != (row[0] or ""):
                logger.warning(
                    "Compaction-to-ingestion: event scope ids diverge from "
                    "the conversation row (event customer=%s user=%s; row "
                    "customer=%s user=%s) — using the row's external ids",
                    raw_customer, raw_user, row[1], row[0],
                )
            raw_customer = row[1]
            raw_user = row[0] or raw_user
    except Exception as e:
        logger.debug(f"Could not resolve conversation row for scope: {e}")
    if not raw_customer:
        logger.error(
            "Compaction-to-ingestion: missing customer_id for conversation %s",
            event.conversation_id,
        )
        return None

    # Use IdMapper for persistent mapping when available
    try:
        from synap.cloud.shared.id_mapper import get_id_mapper
        mapper = get_id_mapper()
        client_uuid = await mapper.resolve_client(raw_client)
        instance_uuid = await mapper.resolve_instance(event.instance_id)
        customer_uuid = await mapper.resolve_customer(raw_client, raw_customer, client_uuid=client_uuid)
        _is_b2c = bool(raw_user and raw_customer and raw_user == raw_customer)
        user_uuid = (
            await mapper.resolve_user(
                raw_client, raw_user, raw_customer,
                customer_uuid=customer_uuid, is_b2c=_is_b2c,
            )
            if raw_user
            else None
        )
        scope = Scope(
            client_id=client_uuid,
            instance_id=instance_uuid,
            customer_id=customer_uuid,
            user_id=user_uuid,
        )
        # N-12 and S-03. The last point on this path that still holds the
        # client's own ids. Off unless SYNAP_NESTED_SCOPING is set, never raises.
        from synap.cloud.shared.scope import resolve_chain_rungs

        rungs = await resolve_chain_rungs(
            client_uuid,
            external_customer_id=raw_customer,
            external_user_id=raw_user,
            customer_uuid=customer_uuid,
            user_uuid=user_uuid,
            instance_id=str(instance_uuid),
        )
        if rungs:
            scope.chain = [ScopeRung(**r) for r in rungs]
        return scope
    except Exception:
        from synap.cloud.shared.scope import customer_deterministic_uuid, user_deterministic_uuid
        # No chain here, deliberately: IdMapper is unavailable so these ids are
        # *derived* rather than read, and N-14 measured that derivation
        # disagrees with the stored value for some clients. A node under a
        # derived id matches no vector record, which is worse than no node at
        # all: an unresolvable chain returns nothing, and a wrong one returns
        # nothing while looking correct.
        _client_uuid = parse_or_deterministic_uuid(raw_client)
        _cust_uuid = customer_deterministic_uuid(_client_uuid, raw_customer)
        return Scope(
            client_id=_client_uuid,
            instance_id=parse_or_deterministic_uuid(event.instance_id),
            customer_id=_cust_uuid,
            user_id=user_deterministic_uuid(_cust_uuid, raw_user, is_b2c=bool(raw_user and raw_customer and raw_user == raw_customer)) if raw_user else None,
        )


async def transcript_push_promotion_skip(event) -> bool:
    """Double-extraction guard for transcript-push conversations (spec §4.4).

    The one-shot transcript ingest endpoint already ingests the FULL
    transcript (higher fidelity than compaction output), so the post-
    compaction long-term promotion callback must not ingest it again.

    Authoritative signal: the conversation row's ``transcript_hash`` — set
    only by the transcript ingest. Checking the ROW (not just the event's
    trigger_type) matters because a failed/raced transcript-push compaction
    can be followed by a MANUAL_API or BACKGROUND_WORKER *full* compaction
    of the same conversation; a trigger-type-only skip would promote the
    entire transcript a second time (adversarial review, backend finding 1).

    Subordinate signal: ``event.trigger_type == TRANSCRIPT_PUSH`` — covers
    the window where the row cannot be read (DB hiccup).

    Returns True when promotion must be skipped.
    """
    trigger = getattr(event, "trigger_type", None)
    if trigger is not None and getattr(trigger, "value", trigger) == "transcript_push":
        return True
    try:
        from synap.cloud.shared.conversation_id_norm import coerce_conversation_id
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text as _text
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                _text(
                    "SELECT transcript_hash FROM conversation "
                    "WHERE conversation_id = :cid"
                ),
                {"cid": coerce_conversation_id(event.conversation_id)},
            )
            row = result.first()
        return bool(row and row[0])
    except Exception as e:
        logger.debug(f"Transcript-push promotion guard row check failed: {e}")
        return False


def build_compaction_update_bundle(event, compacted, all_messages) -> dict:
    """The ``compaction_update`` bundle pushed to the SDK after a compaction.

    ``all_messages`` is the conversation as it stands when the push fires, or
    None when it could not be read.

    v2 payload: includes the dual paragraphs from the compaction v2 pipeline
    (factual + conversational) AND the un-compacted "tail": messages that
    arrived AFTER the compaction boundary, so the SDK's local cache has the
    full picture of "what's compacted + what's recent and raw" without polling.

    Module-level so the bundle the SDK is tested against is the one the server
    builds.
    """
    ns = compacted.narrative_summary
    # Compaction v3: ``summary`` is the one framed context_block, the same
    # text REST and retrieval return, and the two v2 paragraphs go empty, so
    # an SDK released before v3 prints the compaction once.
    # Rows written before v3 keep their v2 shape until their next compaction.
    context_block = getattr(compacted, "context_block", "") or ""
    if context_block:
        factual_paragraph = conversational_paragraph = ""
        summary_text = context_block
    else:
        factual_paragraph = getattr(ns, "factual_paragraph", "") or ""
        conversational_paragraph = getattr(ns, "conversational_paragraph", "") or ""
        summary_text = (
            getattr(ns, "overview", "")
            or ((factual_paragraph + "\n\n" + conversational_paragraph).strip())
            or str(ns)
        )
    current_state = getattr(ns, "current_state", {}) or {}

    # Un-compacted tail: messages that arrived AFTER the
    # compaction boundary. The SDK's customer agent needs
    # both the compacted history AND the raw recent turns
    # for continuity.
    recent_turns: list = []
    smr = compacted.metadata.source_message_range or {}
    compacted_count = int(smr.get("message_count", 0) or 0)
    total_count = len(all_messages or [])
    if all_messages and compacted_count > 0 and len(all_messages) > compacted_count:
        tail = all_messages[compacted_count:]
    else:
        # Either no compaction-range info or no tail —
        # still include the absolute last few as recent_turns
        # so the SDK has SOMETHING raw to show.
        tail = (all_messages or [])[-6:]
    for m in tail:
        try:
            recent_turns.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", "") or "",
                "timestamp": (
                    m.timestamp.isoformat()
                    if getattr(m, "timestamp", None) and hasattr(m.timestamp, "isoformat")
                    else ""
                ),
            })
        except Exception:
            continue

    # Legacy structured key_extractions stay in the schema for
    # any SDK consumer that still reads them. v2 has empty
    # arrays — the prose paragraphs carry the data.
    legacy_ext = compacted.structured_extractions
    key_extractions = {
        "facts": [
            f.to_dict() for f in getattr(legacy_ext, "explicit_facts", []) or []
        ],
        "preferences": [
            p.to_dict() for p in getattr(legacy_ext, "user_preferences", []) or []
        ],
        "decisions": [
            d.to_dict() for d in getattr(legacy_ext, "decisions", []) or []
        ],
    }

    # The two counts let the SDK prune by count and not by clock: the last
    # (total - compacted) recent_turns are exactly the messages this summary
    # does not cover. Only a summary that says how many messages it covers,
    # over a conversation that still holds at least that many, gives the SDK a
    # tail it can trust. Otherwise both are 0, which the SDK reads as "the
    # server did not say".
    counts_known = 0 < compacted_count <= total_count
    return {
        "bundle_id": f"compaction_update_{compacted.metadata.compaction_id}",
        "_bundle_type": "compaction_update",
        "_anticipation_user_id": event.user_id,
        "_anticipation_customer_id": event.customer_id,
        "_anticipation_conversation_id": event.conversation_id,
        "search_queries": [],
        "search_keywords": [],
        "items_by_type": {},
        "conversation_context": {
            "summary": summary_text,
            "current_state": current_state,
            "key_extractions": key_extractions,
            # Empty for a v3 compaction (see above); kept for SDKs that read them.
            "factual_paragraph": factual_paragraph,
            "conversational_paragraph": conversational_paragraph,
            # v3: the block as received, the text without the
            # framing, the commitments as data, and which turns are covered.
            "context_block": context_block,
            "document": getattr(compacted, "document", "") or "",
            "agreed_actions": list(getattr(compacted, "agreed_actions", []) or []),
            "covers": {
                k: smr.get(k) for k in ("start_id", "end_id", "start_timestamp", "end_timestamp", "message_count")
            },
            "recent_turns": recent_turns,
            "compaction_id": compacted.metadata.compaction_id,
            "compacted_at": (
                compacted.metadata.created_at.isoformat()
                if compacted.metadata.created_at else None
            ),
            "conversation_id": event.conversation_id,
            "compacted_message_count": compacted_count if counts_known else 0,
            "total_message_count": total_count if counts_known else 0,
        },
    }


async def initialize_providers_once():
    """Initialize all providers once at app startup.

    Call this in app lifespan to pre-warm providers.
    """
    global _providers_cache

    if _providers_cache["initialized"]:
        return

    logger.info("Initializing shared providers...")
    _providers_cache["redis"] = await get_redis_provider()
    _providers_cache["postgres"] = get_postgres_provider()
    _providers_cache["vector"] = get_vector_provider()
    _providers_cache["graph"] = get_graph_provider()
    _providers_cache["embedding"] = get_embedding_provider()
    _providers_cache["llm"] = get_llm_provider()
    _providers_cache["compaction_storage"] = get_compaction_storage()
    _providers_cache["compaction_manager"] = get_compaction_manager_provider()

    # IdMapper for external → internal ID resolution
    try:
        from synap.cloud.shared.id_mapper import get_id_mapper
        _providers_cache["id_mapper"] = get_id_mapper()
    except Exception as e:
        logger.warning(f"IdMapper unavailable: {e}")
        _providers_cache["id_mapper"] = None

    # Wire post-compaction callbacks
    compaction_mgr = _providers_cache.get("compaction_manager")
    compaction_storage = _providers_cache.get("compaction_storage")
    redis_client = _providers_cache.get("redis")

    if compaction_mgr is not None and redis_client is not None:
        # Callback 1: Invalidate retrieval L2 cache
        async def _invalidate_retrieval_cache_on_compaction(event):
            try:
                if event.user_id:
                    from synap.cloud.application.client_api.retrieval_manager import CacheKeyGenerator
                    l2_key = CacheKeyGenerator.l2_user_context_key(
                        client_id=getattr(event, "client_id", ""),
                        customer_id=getattr(event, "customer_id", "") or "",
                        user_id=event.user_id,
                        conversation_id=event.conversation_id,
                    )
                    await redis_client.delete(l2_key)
                    logger.info(f"Invalidated L2 cache for {event.conversation_id}")
                else:
                    logger.warning(
                        f"No user_id on compaction event for {event.conversation_id}, "
                        "skipping L2 cache invalidation — compacted summary may be stale"
                    )
            except Exception as e:
                logger.warning(f"Failed to invalidate retrieval cache: {e}")

        compaction_mgr.on_compaction_complete(_invalidate_retrieval_cache_on_compaction)
        logger.info("Wired post-compaction → retrieval cache invalidation callback")

        # Callback 2: Trim Redis raw message buffer
        if compaction_storage is not None:
            async def _trim_redis_buffer_on_compaction(event):
                try:
                    compacted = await compaction_storage.get_latest(
                        event.conversation_id
                    )
                    if not compacted or not compacted.metadata.source_message_range:
                        return
                    end_ts = compacted.metadata.source_message_range.get("end_timestamp")
                    if not end_ts:
                        return
                    trimmed = await compaction_storage.trim_raw_messages(
                        event.conversation_id, end_ts
                    )
                    if trimmed > 0:
                        logger.info(
                            f"Trimmed {trimmed} compacted messages from Redis buffer "
                            f"for {event.conversation_id}"
                        )
                except Exception as e:
                    logger.warning(f"Redis buffer trim failed (non-fatal): {e}")

            compaction_mgr.on_compaction_complete(_trim_redis_buffer_on_compaction)
            logger.info("Wired post-compaction → Redis buffer trim callback")

        # Long-term promotion of raw turns runs on its own schedule (every 5
        # turns, idle, session end), no longer in a post-compaction callback:
        # compaction v3 runs far less often (COMPACTION_V3_PLAN.md §4.6).
        # See conversation/promotion.py; scope resolution and the
        # transcript-push guard are the helpers above, unchanged.
        ingestion_mgr = _get_ingestion_manager(redis_client)
        if ingestion_mgr is not None and compaction_storage is not None:
            from synap.cloud.application.conversation.promotion import LongTermPromoter, set_promoter

            set_promoter(LongTermPromoter(compaction_storage, ingestion_mgr, redis_client))
            logger.info("Wired long-term promotion (raw turns, every 5 turns + idle + session end)")

        # Callback 3b: keep one restorable backup object per conversation.
        # Independent of the long-term promotion above (which sends only the new
        # messages to extraction and skips transcript-push conversations): this
        # one holds the whole conversation, tail included.
        if ingestion_mgr is not None and compaction_storage is not None:
            async def _backup_conversation_on_compaction(event):
                try:
                    from synap.cloud.application.ingestion.conversation_backup import (
                        backup_conversation,
                    )
                    scope = await resolve_compaction_promotion_scope(event)
                    if scope is None:
                        return
                    messages = await compaction_storage.load_conversation_messages(
                        event.conversation_id
                    )
                    if messages:
                        await backup_conversation(
                            ingestion_mgr, event.conversation_id, scope, messages
                        )
                except Exception as e:
                    logger.error(f"Conversation backup failed (non-fatal): {e}", exc_info=True)

            compaction_mgr.on_compaction_complete(_backup_conversation_on_compaction)
            logger.info("Wired post-compaction → whole-conversation backup callback")

        # Callback 4: Push compacted context to SDK via gRPC stream
        if compaction_storage is not None:
            async def _push_compaction_to_grpc_stream(event):
                """Push a compaction_update bundle to the SDK's gRPC push queue.

                v2 payload: includes the dual paragraphs from the compaction
                v2 pipeline (factual + conversational) AND the un-compacted
                "tail" — messages that arrived AFTER the compaction
                boundary, so the SDK's local cache has the full picture of
                "what's compacted + what's recent and raw" without polling.

                The SDK handles ``_bundle_type=compaction_update`` by both:
                  - Invalidating stale compacted-context cache entries
                  - Storing the bundle in AnticipationCache (so the next
                    sdk.fetch picks up the new ConversationContext)
                Both behaviors are kept.
                """
                import asyncio as _asyncio

                try:
                    push_queues = _providers_cache.get("push_queues")
                    if not push_queues:
                        return

                    queue = push_queues.get(event.instance_id)
                    if queue is None:
                        return

                    compacted = await compaction_storage.get_latest(
                        event.conversation_id
                    )
                    if not compacted:
                        return

                    # None when the messages cannot be read. The bundle
                    # still carries the summary, without a tail or counts.
                    all_messages = None
                    try:
                        all_messages = await compaction_storage.load_conversation_messages(
                            event.conversation_id
                        )
                    except Exception as tail_err:
                        logger.debug(
                            f"compaction_update tail-fetch failed (non-fatal): {tail_err}"
                        )

                    bundle = build_compaction_update_bundle(event, compacted, all_messages)
                    cc = bundle["conversation_context"]
                    factual_paragraph = cc["factual_paragraph"]
                    conversational_paragraph = cc["conversational_paragraph"]
                    recent_turns = cc["recent_turns"]

                    try:
                        queue.put_nowait(bundle)
                        logger.info(
                            f"Pushed compaction_update to gRPC stream for instance "
                            f"{event.instance_id}: compaction_id={compacted.metadata.compaction_id} "
                            f"factual_chars={len(factual_paragraph)} "
                            f"conversational_chars={len(conversational_paragraph)} "
                            f"recent_turns={len(recent_turns)}"
                        )
                    except _asyncio.QueueFull:
                        logger.warning(
                            f"Push queue full for instance {event.instance_id}; "
                            f"compaction_update dropped"
                        )
                except Exception as e:
                    logger.warning(f"gRPC push callback failed (non-fatal): {e}")

            compaction_mgr.on_compaction_complete(_push_compaction_to_grpc_stream)
            logger.info("Wired post-compaction → gRPC push callback")

    _providers_cache["initialized"] = True

    logger.info(
        f"Providers initialized: "
        f"redis={bool(_providers_cache['redis'])}, "
        f"postgres={bool(_providers_cache['postgres'])}, "
        f"vector={bool(_providers_cache['vector'])}, "
        f"graph={bool(_providers_cache['graph'])}, "
        f"embedding={bool(_providers_cache['embedding'])}, "
        f"llm={bool(_providers_cache['llm'])}, "
        f"compaction_storage={bool(_providers_cache['compaction_storage'])}, "
        f"compaction_manager={bool(_providers_cache['compaction_manager'])}, "
        f"id_mapper={bool(_providers_cache['id_mapper'])}"
    )


def get_cached_providers():
    """Get cached providers for RetrievalManager.

    Returns:
        Dict with redis, postgres, vector, graph keys
    """
    return _providers_cache
