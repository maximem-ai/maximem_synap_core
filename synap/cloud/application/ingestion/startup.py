"""Ingestion worker startup and lifecycle management.

Initializes and starts the ingestion pipeline workers:
- PriorityQueueRouter: Routes priority queues to Stage1
- Stage1Worker: Type/Source/Nature qualification
- Stage23Worker: Memory extraction
- Stage4Worker: Organization and persistence

All workers run as background asyncio tasks.
Graceful degradation: if dependencies are missing, workers log warnings
but don't crash the application.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from redis.asyncio import Redis

from synap.cloud.application.ingestion.status import StatusTracker
from synap.cloud.shared import edition
from synap.cloud.memory_config.loader import check_config_at_startup, get_config_loader  # noqa: F401

logger = logging.getLogger(__name__)

# Module-level state for lifecycle management
_worker_tasks: List[asyncio.Task] = []
_redis_client: Optional[Redis] = None
_running = False
# Names of the workers that failed to start, for /health (Core reports "not ready" while any is here).
_start_failures: List[str] = []


def start_problems() -> List[str]:
    """What stops memories being processed: workers that failed to start, or none running at all."""
    problems = list(_start_failures)
    if not _running and not problems:
        problems.append("the ingestion workers are not running")
    return problems


class _NoOpReportingStore:
    """No-op reporting store for when TimescaleDB is not available."""

    def connect(self): pass
    def disconnect(self): pass
    def write_metric(self, metric_name, value, timestamp=None, tags=None): pass
    def write_metrics_batch(self, points): return 0
    def query(self, query): return None
    def query_latest(self, metric_name, tags=None): return None
    def delete_metrics(self, metric_name, before): return 0


async def start_ingestion_workers() -> None:
    """Start all ingestion pipeline workers.

    Creates Redis connections, initializes pipeline stages, and starts
    background tasks for the router and all stage workers.

    Safe to call multiple times (idempotent).
    Graceful degradation: logs warnings on failure, never crashes the app.
    """
    global _worker_tasks, _redis_client, _running

    if _running:
        logger.warning("Ingestion workers already running")
        return
    _start_failures.clear()

    # Core: a memory config file that fails validation stops the server here.
    # Cloud: does nothing (the config is in the table).
    check_config_at_startup()
    from synap.cloud.application.ingestion.multimodal import warn_if_no_vision_model
    warn_if_no_vision_model()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        _redis_client = Redis.from_url(redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info(f"Ingestion workers: Redis connected at {redis_url}")
    except Exception as e:
        logger.warning(
            f"Ingestion workers: Redis unavailable ({e}). "
            "Ingestion queue processing disabled."
        )
        _start_failures.append("the connection to Redis")
        return

    status_tracker = StatusTracker(_redis_client)
    reporting = _NoOpReportingStore()

    # Start PriorityQueueRouter (only needs Redis)
    try:
        from synap.cloud.agents.ingestion.workers.router import PriorityQueueRouter

        router = PriorityQueueRouter(redis=_redis_client)
        _worker_tasks.append(asyncio.create_task(router.run()))
        logger.info("Ingestion PriorityQueueRouter started")
    except Exception as e:
        logger.error(f"Failed to start PriorityQueueRouter: {e}", exc_info=True)
        _start_failures.append("PriorityQueueRouter")

    # Start Stage workers (require pipeline dependencies)
    try:
        await _start_stage_workers(
            redis_client=_redis_client,
            status_tracker=status_tracker,
            reporting=reporting,
        )
    except Exception as e:
        logger.warning(
            f"Ingestion stage workers failed to start: {e}. "
            "Items will be routed but not processed.",
            exc_info=True,
        )
        _start_failures.append("the stage workers")

    # The graph side of retired and deleted memories (graph_sync_pending).
    try:
        _start_graph_sync_worker()
    except Exception as e:
        logger.warning(f"Graph sync drain worker failed to start: {e}", exc_info=True)
        _start_failures.append("the graph sync worker")

    _running = True
    logger.info(f"Ingestion workers started: {len(_worker_tasks)} tasks")


def _start_graph_sync_worker() -> bool:
    """Start the outbox drain worker (storage/graph_sync_outbox.py) when the
    graph lifecycle is on and a graph store is configured. Returns whether it
    started. A database without the table is handled by the worker itself
    (one warning, then idle)."""
    from synap.cloud.agents.ingestion.storage.graph_lifecycle import lifecycle_enabled
    from synap.cloud.agents.ingestion.storage.graph_sync_outbox import run_drain_worker
    if not lifecycle_enabled() or not os.getenv("NEO4J_URI"):
        logger.info("Graph sync drain worker not started (SYNAP_GRAPH_LIFECYCLE off or no graph)")
        return False
    coordinator = _get_storage_coordinator()
    graph = getattr(coordinator, "_graph", None)
    if graph is None:
        logger.warning("Graph sync drain worker not started: the graph store is unavailable")
        return False
    try:
        interval = float(os.getenv("SYNAP_GRAPH_SYNC_INTERVAL_SECONDS", "10"))
    except ValueError:
        interval = 10.0
    _worker_tasks.append(asyncio.create_task(run_drain_worker(
        graph=graph, vector=getattr(coordinator, "_vector", None), interval=interval)))
    logger.info("Graph sync drain worker started (interval %.0fs)", interval)
    return True


async def _start_stage_workers(
    redis_client: Redis,
    status_tracker: StatusTracker,
    reporting,
) -> None:
    """Start individual stage workers."""
    global _worker_tasks

    # The edition picks the loader: the Postgres table (Cloud) or config/memory.yml (Core).
    config_loader = get_config_loader()

    # Build a default MACA config for pipeline stage init
    default_config = _get_default_maca_config()

    # --- Stage 1 Workers (configurable concurrency) ---
    stage1_count = int(os.environ.get("STAGE1_WORKER_COUNT", "1"))
    try:
        from synap.cloud.agents.ingestion.pipeline.stage1_qualification import Stage1Qualifier
        from synap.cloud.agents.ingestion.workers.stage1_worker import Stage1Worker

        for i in range(stage1_count):
            qualifier = Stage1Qualifier(config=default_config)
            worker = Stage1Worker(
                qualifier=qualifier,
                config_loader=config_loader,
                redis=redis_client,
                status_tracker=status_tracker,
                reporting=reporting,
                worker_id=f"stage1-worker-{i}",
            )
            _worker_tasks.append(asyncio.create_task(worker.run()))

        logger.info("Ingestion Stage1Workers started (count=%d)", stage1_count)
    except Exception as e:
        logger.warning(f"Stage1Worker failed to start: {e}", exc_info=True)
        _start_failures.append("Stage1Worker")

    # --- Stage 2+3 Workers (configurable concurrency) ---
    stage23_count = int(os.environ.get("STAGE23_WORKER_COUNT", "3"))
    try:
        from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
        from synap.cloud.agents.ingestion.workers.stage23_worker import Stage23Worker

        # Cloud registers the provider-specific prompt variants. Core has one prompt pack.
        edition.call("ingestion_prompt_variants", os.environ.get("EVAL_PROVIDER", "openai").lower())

        for i in range(stage23_count):
            llm_client = _get_llm_client("ingestion_agent")
            llm_client_light = _get_llm_client("ingestion_agent_light")

            # Entity extraction is Tier 1 — use the light client
            # The graph (entities, relationships) and entity resolution may run on their own
            # model: SYNAP_ER_MODEL. With it unset this is the same gpt-4o client as before.
            llm_client_graph = _get_llm_client("ingestion_agent_light", purpose="er")
            # The separate entity extractor is Cloud's: Core reads entities and relationships in the one
            # graph call and gets None here.
            entity_extractor = edition.call(
                "ingestion_entity_extractor", config=default_config, llm_client=llm_client_graph,
            )

            # Prior-memory context for resolving a reference the document cannot
            # resolve alone ("three months before the wedding"). Injected, not
            # reached for: the extractor keeps its config-and-LLM-only shape, and
            # a worker built without this simply never has the feature, whatever
            # SYNAP_STAGE3_PRIOR_CONTEXT says.
            from synap.cloud.agents.ingestion.services import prior_context
            from synap.cloud.database.connection import get_session_factory as _gsf
            prior_context_fn = prior_context.build_fetcher(_gsf())
            # Who "he" is when a document names no one: the same user's recent
            # memories, judged by Jev. Built always, used only when
            # SYNAP_STAGE3_EARLIER_REFERENCES is set (services/earlier_reference).
            from synap.cloud.agents.ingestion.services import earlier_reference
            earlier_reference_resolver = earlier_reference.EarlierReferenceResolver(
                earlier_reference.build_fetcher(_gsf()))

            extractor = Stage3Extractor(
                config=default_config,
                llm_client=llm_client,
                entity_extractor=entity_extractor,
                llm_client_light=llm_client_light,
                llm_client_graph=llm_client_graph,
                prior_context_fn=prior_context_fn,
                earlier_reference_resolver=earlier_reference_resolver,
            )

            # The question generator (HQE) is Cloud's; in Core this is None and the worker skips it.
            hqe_generator = edition.call(
                "ingestion_hqe_generator", llm_client=llm_client, llm_client_light=llm_client_light,
            )

            worker = Stage23Worker(
                extractor=extractor,
                config_loader=config_loader,
                hqe_generator=hqe_generator,
                redis=redis_client,
                status_tracker=status_tracker,
                reporting=reporting,
                worker_id=f"stage23-worker-{i}",
            )
            _worker_tasks.append(asyncio.create_task(worker.run()))

        logger.info("Ingestion Stage23Workers started (count=%d)", stage23_count)
    except Exception as e:
        logger.warning(f"Stage23Worker failed to start: {e}", exc_info=True)
        _start_failures.append("Stage23Worker")

    # --- Stage 4 Workers (configurable concurrency) ---
    stage4_count = int(os.environ.get("STAGE4_WORKER_COUNT", "2"))
    try:
        from synap.cloud.agents.ingestion.pipeline.stage4_organization import Stage4Organizer
        from synap.cloud.agents.ingestion.workers.stage4_worker import Stage4Worker

        # Get graph_config for StorageCoordinator
        graph_config_dict = None
        try:
            graph_config_dict = default_config.storage_architecture.graph_config.model_dump()
        except Exception:
            pass

        storage_coordinator = _get_storage_coordinator(graph_config=graph_config_dict)

        # Wire entity resolution service (graceful degradation)
        entity_resolver = None
        try:
            from synap.cloud.agents.ingestion.resolution.factory import create_entity_resolution_service
            # The judge steps that call a model are Cloud's: the hook builds the client
            # only when one of them is switched on, and Core gets None.
            second_look_llm = edition.call(
                "entity_judge_llm", lambda: _get_llm_client("ingestion_agent", purpose="er"),
            )
            entity_resolver = create_entity_resolution_service(
                default_config.storage_architecture.graph_config,
                llm_client=second_look_llm,
                # Entity profiles read an entity's links and shared memories from here.
                graph_wrapper=getattr(storage_coordinator, "_graph", None),
            )
            if entity_resolver:
                logger.info("Entity resolution service created for Stage 4")
        except Exception as e:
            logger.warning(f"Failed to create entity resolution service: {e}")

        from synap.cloud.database.connection import get_session_factory
        db_session_factory = get_session_factory()

        # Knowledge-update detector (on by default; override with ENABLE_KNOWLEDGE_UPDATE_DETECTION=false)
        knowledge_update_detector = None
        if os.environ.get("ENABLE_KNOWLEDGE_UPDATE_DETECTION", "true").lower() in ("true", "1", "yes"):
            try:
                from synap.cloud.agents.ingestion.pipeline.knowledge_update import KnowledgeUpdateDetector
                # Pair classification is a four-way label with a ~34-token
                # answer, and it is one of the largest LLM line items in the
                # pipeline: over a sample of real traffic, only about one call in ten
                # changed anything — the rest answered
                # "unrelated". That shape is what a small model is for.
                #
                # Default is "gpt-4o", i.e. exactly what ran before this knob
                # existed, so nothing moves until someone sets it. Set
                # SYNAP_KU_CLASSIFY_MODEL=gpt-4o-mini to switch the step, and
                # measure recall against the SUPERSEDE / ENRICH / NONE / DELETE
                # decisions already in memory_actions before leaving it on.
                ku_model = os.environ.get("SYNAP_KU_CLASSIFY_MODEL", "gpt-4o").strip() or "gpt-4o"
                ku_llm_client = _get_llm_client("ingestion_agent_light", model=ku_model,
                                                step="KNOWLEDGE_UPDATE")
                if ku_model != "gpt-4o":
                    logger.warning(
                        "knowledge_update_classify_model_override model=%s "
                        "(default gpt-4o) — verify decision quality against "
                        "memory_actions before treating this as settled",
                        ku_model,
                    )
                knowledge_update_detector = KnowledgeUpdateDetector(
                    vector=storage_coordinator._vector,
                    llm_client=ku_llm_client,
                )
                logger.info("Knowledge-update detector initialized for Stage 4")
            except Exception as e:
                logger.warning(f"Failed to create knowledge-update detector: {e}")

        # HQE generator for re-running hypothetical questions on
        # merged memories (merge synthesis creates them; this regenerates
        # HQE because the merged content is richer than the contributors').
        # Reuses the Stage5 HQE pipeline that's already wired for Stage 23.
        stage4_hqe_generator = edition.call(
            "ingestion_hqe_generator", llm_client=llm_client, llm_client_light=llm_client_light,
        )

        for i in range(stage4_count):
            organizer = Stage4Organizer(
                config=default_config,
                storage=storage_coordinator,
                entity_resolver=entity_resolver,
                session_factory=db_session_factory,
                knowledge_update_detector=knowledge_update_detector,
                hqe_generator=stage4_hqe_generator,
            )
            worker = Stage4Worker(
                organizer=organizer,
                # Same shared per-instance MACA loader the stage1/stage23
                # workers hold — the user-profile hook must not rely on the
                # organizer's startup-default config.
                config_loader=config_loader,
                redis=redis_client,
                status_tracker=status_tracker,
                reporting=reporting,
                worker_id=f"stage4-worker-{i}",
            )
            _worker_tasks.append(asyncio.create_task(worker.run()))

        logger.info("Ingestion Stage4Workers started (count=%d)", stage4_count)
    except Exception as e:
        logger.warning(f"Stage4Worker failed to start: {e}", exc_info=True)
        _start_failures.append("Stage4Worker")


# ---------------------------------------------------------------------------
# Helper classes and functions
# ---------------------------------------------------------------------------

def _get_default_maca_config():
    """Get a default MemoryArchitectureConfig for worker init."""
    try:
        from synap.cloud.memory_config.config_schema import (
            MemoryArchitectureConfig,
            ConfigMetadata,
        )

        return MemoryArchitectureConfig(
            metadata=ConfigMetadata(
                client_id="default",
                instance_id="default",
                config_version="1.0.0",
                generated_at=datetime.now(timezone.utc),
                generated_by="ingestion-worker-default",
            )
        )
    except Exception as e:
        logger.error(f"Failed to create default MACA config: {e}")
        raise


# --- Ingestion MACA Config Cache (mirrors retrieval_manager pattern) ---
_ingestion_maca_cache: Dict[str, tuple] = {}  # {instance_id: (MemoryArchitectureConfig, timestamp)}
_INGESTION_MACA_CACHE_TTL = 300  # 5 minutes, same as retrieval layer
# The DEFAULT config (no approved config found) is cached only briefly, for the
# same reason as in retrieval_manager: the real config can appear minutes after
# the instance is created, and a cached default would keep shaping ingestion
# until the 300s entry expires.
_INGESTION_DEFAULT_CACHE_TTL = 15


def _default_cache_age_offset(*instance_ids) -> float:
    """Back-dating for a cached DEFAULT config: short only for supervised-building instances.

    Every other instance (legacy, playground, review-gated, failed) keeps the 5
    minute TTL, so no extra database load for them. See retrieval_manager.
    """
    try:
        from synap.cloud.application.lifecycle.instance_readiness import is_supervised_building

        if is_supervised_building(*instance_ids):
            return _INGESTION_MACA_CACHE_TTL - _INGESTION_DEFAULT_CACHE_TTL
    except Exception:  # noqa: BLE001 - a hint only
        pass
    return 0


async def _resolve_external_instance_id(session, maybe_uuid: str):
    """Map an instance UUID (as carried by ingestion scope) to the external instance_id."""
    try:
        uuid.UUID(maybe_uuid)
    except (ValueError, AttributeError, TypeError):
        return None
    from sqlalchemy import text as _text

    row = await session.execute(
        _text("SELECT instance_id FROM instance WHERE uuid = CAST(:u AS uuid)"),
        {"u": maybe_uuid},
    )
    return row.scalar()


class _DBConfigLoader:
    """Lightweight config loader that reads MACA config from Postgres with 5-min TTL cache."""

    async def get_config(self, instance_id: str):
        """Load MACA config for an instance, using a 5-minute TTL cache."""
        now = time.time()
        cached = _ingestion_maca_cache.get(instance_id)
        if cached and (now - cached[1]) < _INGESTION_MACA_CACHE_TTL:
            return cached[0]

        try:
            from synap.cloud.database.connection import get_session_factory
            from synap.cloud.database.repositories.maca_config_repo import MACAConfigRepository
            from synap.cloud.memory_config.config_schema import (
                MemoryArchitectureConfig,
                ConfigMetadata,
            )

            session_factory = get_session_factory()
            external_id = None
            async with session_factory() as session:
                repo = MACAConfigRepository(session)
                config_orm = await repo.get_active_for_instance(instance_id)

                if config_orm is None:
                    # Ingestion scope carries the instance UUID (uuid5), but
                    # maca_config.instance_id stores the external "inst_..."
                    # string — resolve and retry, else every queue-path worker
                    # silently falls back to the default config.
                    external_id = await _resolve_external_instance_id(session, instance_id)
                    if external_id:
                        config_orm = await repo.get_active_for_instance(external_id)

                if config_orm and config_orm.config_content:
                    config = MemoryArchitectureConfig(**config_orm.config_content)
                    _ingestion_maca_cache[instance_id] = (config, now)
                    return config

            # Fallback to default
            default = _get_default_maca_config()
            _ingestion_maca_cache[instance_id] = (
                default, now - _default_cache_age_offset(instance_id, external_id),
            )
            return default
        except Exception as e:
            logger.warning(f"Failed to load MACA config for {instance_id}: {e}")
            # Return stale cache if available, otherwise default
            if cached:
                return cached[0]
            default = _get_default_maca_config()
            _ingestion_maca_cache[instance_id] = (
                default, now - _default_cache_age_offset(instance_id),
            )
            return default


def invalidate_ingestion_maca_cache(instance_id: str) -> None:
    """Drop this process's cached MACA config for an instance.

    Clean invalidation seam for the ingestion workers' ``_DBConfigLoader``
    cache, mirroring ``retrieval_manager.invalidate_maca_cache``. A dashboard
    edit to the user-profile-config (or any active-config write) calls this so
    the ingestion path picks up the new config immediately instead of after the
    300s TTL. In-process only; other worker processes converge within
    ``_INGESTION_MACA_CACHE_TTL``. Safe to call for an instance that isn't
    cached.
    """
    _ingestion_maca_cache.pop(instance_id, None)


def er_model_for(scope: Any = None) -> Optional[str]:
    """The model for entity-resolution and graph-extraction calls, or None for the
    default (gpt-4o, pinned below).

        SYNAP_ER_MODEL=gpt-6-luna
        SYNAP_ER_MODEL_INSTANCES=inst_a,inst_b      those instances ("*" = all)

    Only the instance list turns it on: the model name is not a truthy value, so
    ``SYNAP_ER_MODEL`` alone changes nothing. A call that carries no scope (the
    second look) is matched only by "*", because it cannot say which instance it
    is for.
    """
    model = os.environ.get("SYNAP_ER_MODEL", "").strip()
    if not model:
        return None
    from synap.cloud.shared.instance_flags import enabled_for
    return model if enabled_for("SYNAP_ER_MODEL", getattr(scope, "instance_id", None)) else None


def _get_llm_client(agent_type: str = "ingestion_agent", model: Optional[str] = None,
                    purpose: Optional[str] = None, step: Optional[str] = None):
    """Get an LLM client that implements the LLMClient protocol.

    ⚠ ``agent_type`` IS CURRENTLY IGNORED. Every client this returns pins
    gpt-4o (or the EVAL_PROVIDER override), whichever tier the caller asks
    for. The parameter reads like a model-tier switch and is not one.

    When ``SYNAP_LLM_PROVIDER`` and ``SYNAP_LLM_MODEL`` are set (Core always sets
    them), the pin above is ignored: the LLM manager sends every call to that
    provider and model, or to the step's model (``SYNAP_LLM_MODEL_<STEP>``), and
    never falls back to another provider. ``EVAL_PROVIDER`` is then
    ignored too. With neither set, Cloud behaves as described above.

    ``model`` IS honoured — it is the explicit, per-call-site escape hatch
    that ``agent_type`` is not. Pass it to run one step on a different model
    without changing any other step.

    That matters commercially. Callers do ask for the light tier —
    knowledge-update classification and the profile merge both pass
    "ingestion_agent_light" — and knowledge-update classification alone is
    ~41% of all LLM spend, running a four-way label with a ~34-token answer
    on the flagship model. AGENT_DEFAULTS["ingestion_agent_light"] would not
    help on its own either: its preferred_models is ["gpt-4o"] too.

    Making the tier real is a quality decision, not a cleanup: routing
    through the selector would stop pinning one model and change which model
    answers. Measure extraction quality first, then wire it deliberately.

    Args:
        agent_type: AGENT_DEFAULTS key, accepted for call-site intent and for
                    when the tier is made real. Use "ingestion_agent" for
                    Tier 2+3 types (preferences, episodes, relationships,
                    temporal) and "ingestion_agent_light" for Tier 1 types
                    (facts, emotions, entities, categorization).
    """
    from synap.cloud.infrastructure.llm import get_llm_manager

    manager = get_llm_manager()
    _forced_model = model
    # Which setting a developer's per-step model applies to (SYNAP_LLM_MODEL_<STEP>); only read
    # when SYNAP_LLM_PROVIDER and SYNAP_LLM_MODEL are set. Entity and relationship work is GRAPH.
    _step = step or ("GRAPH" if purpose == "er" else "INGESTION")

    class _LLMClientWrapper:
        def __init__(self, llm_manager):
            self._manager = llm_manager
            # Cloud counts usage; Core has no usage tracker, so this is None.
            self._usage_tracker = edition.call("ingestion_usage_tracking")

        async def complete(
            self,
            prompt: str = "",
            telemetry_scope=None,
            telemetry_operation: Optional[str] = None,
            telemetry_correlation_id: Optional[str] = None,
            max_tokens: int = 4000,
            messages: Optional[list] = None,
            temperature: float = 0.0,
            json_mode: bool = False,
            **kwargs,
        ) -> str:
            from synap.cloud.infrastructure.llm.models import LLMProvider
            eval_provider = os.environ.get("EVAL_PROVIDER", "openai").lower()
            if eval_provider == "gemini":
                force_provider = LLMProvider.GEMINI
                force_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            else:
                force_provider = LLMProvider.OPENAI
                force_model = _forced_model or (
                    er_model_for(telemetry_scope) if purpose == "er" else None) or "gpt-4o"
            llm_messages = messages if messages else [{"role": "user", "content": prompt}]
            response = await self._manager.complete(
                messages=llm_messages,
                force_provider=force_provider,
                force_model=force_model,
                step=_step,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                # Stay on the pinned provider when a call errors. Without this
                # a gpt-4o 5xx or timeout was answered by Groq's gpt-oss-120b,
                # a model none of the ingestion prompts were measured on.
                allow_cross_provider_fallback=False,
                # Provider-specific knobs the caller set, e.g. reasoning_effort.
                # The manager forwards **kwargs to the provider, which ignores
                # anything the model does not accept.
                **kwargs,
            )

            if telemetry_scope is not None and self._usage_tracker is not None:
                client_id = str(getattr(telemetry_scope, "client_id", "") or "")
                instance_id = str(getattr(telemetry_scope, "instance_id", "") or "")
                user_id = getattr(telemetry_scope, "user_id", None)
                try:
                    if client_id and instance_id:
                        await self._usage_tracker.track_llm_usage(
                            client_id=client_id,
                            instance_id=instance_id,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            model=response.model,
                            operation=telemetry_operation or "ingestion.llm",
                            correlation_id=telemetry_correlation_id,
                            cached_input_tokens=getattr(
                                response, "cached_input_tokens", 0
                            ),
                        )
                        logger.info(
                            "cloud_usage_track_llm_usage operation=%s client_id=%s instance_id=%s user_id=%s input_tokens=%s output_tokens=%s model=%s correlation_id=%s",
                            telemetry_operation or "ingestion.llm",
                            client_id,
                            instance_id,
                            user_id,
                            response.input_tokens,
                            response.output_tokens,
                            response.model,
                            telemetry_correlation_id,
                        )
                    else:
                        logger.info(
                            "cloud_usage_track_llm_usage_skipped reason=missing_scope operation=%s correlation_id=%s",
                            telemetry_operation or "ingestion.llm",
                            telemetry_correlation_id,
                        )
                except Exception as e:
                    logger.warning(
                        "cloud_usage_track_llm_usage_failed operation=%s client_id=%s instance_id=%s error=%s",
                        telemetry_operation or "ingestion.llm",
                        client_id,
                        instance_id,
                        e,
                    )
            return response.content

    return _LLMClientWrapper(manager)


from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import (
    SQLAlchemyRDBMSWrapper,
)


def _get_storage_coordinator(graph_config=None):
    """Get StorageCoordinator with configured storage backends."""
    from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
    from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper
    from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper
    from synap.providers.graph.adapters.neo4j import Neo4jAdapter
    from synap.providers.vector.adapters.chromadb import ChromaDBAdapter

    # RDBMS (Postgres) — reuse the app's existing SQLAlchemy connection
    # instead of creating a separate psycopg2 pool (avoids credential mismatches)
    rdbms = SQLAlchemyRDBMSWrapper()

    # Graph (Neo4j) — optional
    graph = None
    neo4j_uri = os.getenv("NEO4J_URI")
    if neo4j_uri:
        try:
            graph_provider = Neo4jAdapter({
                "uri": neo4j_uri,
                "user": os.getenv("NEO4J_USER", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", ""),
            })
            graph_provider.connect()
            graph = AsyncGraphWrapper(provider=graph_provider)
            logger.info("Ingestion: Neo4j graph store initialized")
        except Exception as e:
            logger.warning(f"Ingestion: Neo4j unavailable ({e}), graph storage disabled")

    # Vector (ChromaDB) — optional
    vector = None
    chroma_host = os.getenv("CHROMA_HOST")
    if chroma_host:
        try:
            vector_provider = ChromaDBAdapter({
                "host": chroma_host,
                "port": int(os.getenv("CHROMA_PORT", "8000")),
                "auth_token": os.getenv("CHROMA_AUTH_TOKEN"),
            })
            vector_provider.connect()
            vector = AsyncVectorWrapper(provider=vector_provider)
            logger.info("Ingestion: ChromaDB vector store initialized")
        except Exception as e:
            logger.warning(f"Ingestion: ChromaDB unavailable ({e}), vector storage disabled")

    # Embedding client
    embedding_client = _get_embedding_client()

    return StorageCoordinator(
        rdbms=rdbms,
        graph=graph,
        vector=vector,
        embedding_client=embedding_client,
        graph_config=graph_config,
    )


def _get_dimension_for_model(model_name: str) -> int:
    """
    Get embedding dimension for a given model.

    Args:
        model_name: The embedding model name

    Returns:
        The embedding dimension (defaults to 384 for sentence-transformers)
    """
    from synap.cloud.application.memory_architecture.configurators.storage import EMBEDDING_DIMENSIONS

    return EMBEDDING_DIMENSIONS.get(model_name, EMBEDDING_DIMENSIONS.get("default", 1024))


def _get_openai_client():
    """
    Get OpenAI embedding client (legacy).

    Uses OpenAI's text-embedding-3-small model directly, since
    LLMConfigManager only supports completion/streaming, not embeddings.

    Returns:
        OpenAI embedding client wrapper
    """
    import os
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    dimension = _get_dimension_for_model(model)

    client = AsyncOpenAI(api_key=api_key) if api_key else None

    class _OpenAIEmbeddingClientWrapper:
        def __init__(self, openai_client, model_name, dim):
            self._client = openai_client
            self._model = model_name
            self._dimension = dim

        async def embed(self, text: str):
            if not self._client:
                logger.warning("No OpenAI API key set; returning zero vector for embedding")
                return [0.0] * self._dimension
            resp = await self._client.embeddings.create(input=[text], model=self._model)
            return resp.data[0].embedding

        async def embed_batch(self, texts):
            if not texts:
                return []
            if not self._client:
                logger.warning("No OpenAI API key set; returning zero vectors for embeddings")
                return [[0.0] * self._dimension for _ in texts]
            resp = await self._client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in resp.data]

    return _OpenAIEmbeddingClientWrapper(client, model, dimension)


def _get_sentence_transformer_client():
    """
    Get sentence-transformers embedding client (default).

    Uses self-hosted sentence-transformers/all-MiniLM-L12-v2 model (384 dimensions).
    No external API calls required.

    Returns:
        SentenceTransformer embedding client
    """
    from synap.cloud.application.ingestion.embedding_client import get_sentence_transformer_client

    return get_sentence_transformer_client()


def _get_remote_client():
    """
    Get the client for a standalone embedding service, from the ``embedding_remote_client`` hook.

    Only Synap Cloud registers that hook. With nothing registered (Core) there is no such client, so the
    local sentence-transformers client is used.
    """
    client = edition.call("embedding_remote_client")
    if client is not None:
        return client
    logger.warning(
        "EMBEDDING_PROVIDER=remote has no embedding service to use in this edition; "
        "falling back to sentence-transformers"
    )
    return _get_sentence_transformer_client()


def _get_embedding_client():
    """
    Get an embedding client that implements the EmbeddingClient protocol.

    Supports multiple providers via EMBEDDING_PROVIDER environment variable:
    - "sentence-transformers" (default): Self-hosted local (in-process) embeddings
    - "openai": OpenAI API embeddings (requires OPENAI_API_KEY)
    - "remote": a standalone embedding service over HTTP (Synap Cloud only; see ``_get_remote_client``)

    Returns:
        Embedding client instance
    """
    import os

    provider = os.environ.get("EMBEDDING_PROVIDER", "sentence-transformers").lower()

    if provider == "sentence-transformers":
        logger.info("Using sentence-transformers embedding provider")
        return _get_sentence_transformer_client()
    elif provider == "openai":
        logger.info("Using OpenAI embedding provider")
        return _get_openai_client()
    elif provider == "remote":
        logger.info("Using remote embedding-service provider")
        return _get_remote_client()
    else:
        logger.warning(
            f"Unknown embedding provider '{provider}', falling back to sentence-transformers"
        )
        return _get_sentence_transformer_client()


async def stop_ingestion_workers() -> None:
    """Stop all ingestion workers gracefully."""
    global _worker_tasks, _redis_client, _running

    if not _running:
        return

    _running = False

    for task in _worker_tasks:
        task.cancel()

    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)

    _worker_tasks.clear()

    if _redis_client:
        await _redis_client.close()
        _redis_client = None

    # Close LLM provider HTTP clients
    try:
        from synap.cloud.infrastructure.llm import get_llm_manager
        llm_manager = get_llm_manager()
        await llm_manager.close()
    except Exception as e:
        logger.warning("Error closing LLM manager: %s", e)

    logger.info("Ingestion workers stopped")
