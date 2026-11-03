"""FastAPI application for dashboard/admin API.

Provides the main ASGI app used by integration tests and the dashboard API server.
"""

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.cors import CORSMiddleware

from synap.cloud.application.sdk_api.core_identity import install_edition_identity
from synap.cloud.application.sdk_api.core_seed import seed_on_startup
from synap.cloud.application.sdk_api.router import sdk_router
from synap.cloud.database.connection import close_db, get_database_url, get_session_factory, init_db
from synap.cloud.logging_config import configure_logging
from synap.cloud.memory_config.loader import check_config_at_startup
from synap.cloud.shared import core_startup, edition

logger = logging.getLogger(__name__)
if edition.quiet_absent_services:
    from synap.cloud.shared.absent_services import AbsentServiceFilter

    logger.addFilter(AbsentServiceFilter())

_compaction_worker_task: asyncio.Task | None = None
_credit_allowance_worker: "object | None" = None
_credit_allowance_worker_task: asyncio.Task | None = None
_trial_reset_worker: "object | None" = None
_trial_reset_worker_task: asyncio.Task | None = None
_overage_billing_worker: "object | None" = None
_overage_billing_worker_task: asyncio.Task | None = None
_annual_plan_worker: "object | None" = None
_annual_plan_worker_task: asyncio.Task | None = None
_notification_email_relay_worker: "object | None" = None
_notification_email_relay_worker_task: asyncio.Task | None = None
_lifecycle_email_worker: "object | None" = None
_lifecycle_email_worker_task: asyncio.Task | None = None
_maca_build_sweeper: "object | None" = None
_maca_build_sweeper_task: asyncio.Task | None = None
_credit_expiry_sweeper: "object | None" = None
_credit_expiry_sweeper_task: asyncio.Task | None = None
_request_payload_sweeper: "object | None" = None
_request_payload_sweeper_task: asyncio.Task | None = None
_scope_bleed_audit: "object | None" = None
_scope_bleed_audit_task: asyncio.Task | None = None
_forget_retry_worker: "object | None" = None
_forget_retry_worker_task: asyncio.Task | None = None
_retention_worker: "object | None" = None
_retention_worker_task: asyncio.Task | None = None
_ancestor_backfill_worker: "object | None" = None
_ancestor_backfill_worker_task: asyncio.Task | None = None
_anticipation_telemetry_sweeper: "object | None" = None
_anticipation_telemetry_sweeper_task: asyncio.Task | None = None
_pii_counters: "object | None" = None
_pii_counters_task: asyncio.Task | None = None
_enrichment_watcher: "object | None" = None
_enrichment_watcher_task: asyncio.Task | None = None
_aggregation_worker_task: asyncio.Task | None = None
_aggregation_worker = None
_instance_deletion_worker_task: asyncio.Task | None = None
_instance_deletion_worker = None
_drive_sync_worker_task: asyncio.Task | None = None
_drive_sync_worker = None
_notion_sync_worker_task: asyncio.Task | None = None
_notion_sync_worker = None
_sharepoint_sync_worker_task: asyncio.Task | None = None
_sharepoint_sync_worker = None
_slack_sync_worker_task: asyncio.Task | None = None
_slack_sync_worker = None
_linear_sync_worker_task: asyncio.Task | None = None
_linear_sync_worker = None
_mintlify_sync_worker_task: asyncio.Task | None = None
_mintlify_sync_worker = None
_zendesk_sync_worker_task: asyncio.Task | None = None
_zendesk_sync_worker = None
_grpc_server = None
_observability_phase0_state = None
_observability_phase1_trace_state = None
_observability_phase2_metrics_state = None
_observability_phase5_state = None


# What actually started, and what did not. The startup summary used to be a
# hardcoded list of twenty service names, printed unconditionally at the end of
# `lifespan()`. Every subsystem below it sits in a `try/except Exception` that
# degrades gracefully, so any of them could fail and the summary still announced
# it as started. A boot once printed `entity_resolution` in that
# list on the same boot where entity resolution had failed and logged itself
# disabled. The one line an operator reads to confirm a clean boot was a
# constant, which is what let a dead subsystem sit unnoticed.
_started_services: list[str] = []
_degraded_services: list[str] = []


def _ok(message: str, *args: object) -> None:
    """Log a subsystem as started AND record it, so the summary cannot drift."""
    logger.info(message, *args)
    try:
        _started_services.append(message % args if args else message)
    except Exception:  # noqa: BLE001 - a summary must never break startup
        _started_services.append(message)


def _degraded(name: str, exc: BaseException) -> None:
    """Record a subsystem that did not start. The caller still logs the detail."""
    _degraded_services.append(f"{name}: {type(exc).__name__}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    global _compaction_worker_task, _aggregation_worker_task, _aggregation_worker
    global _instance_deletion_worker_task, _instance_deletion_worker
    global _drive_sync_worker_task, _drive_sync_worker
    global _notion_sync_worker_task, _notion_sync_worker
    global _sharepoint_sync_worker_task, _sharepoint_sync_worker
    global _slack_sync_worker_task, _slack_sync_worker
    global _linear_sync_worker_task, _linear_sync_worker
    global _mintlify_sync_worker_task, _mintlify_sync_worker
    global _zendesk_sync_worker_task, _zendesk_sync_worker
    global _observability_phase0_state, _observability_phase1_trace_state
    global _observability_phase2_metrics_state, _observability_phase5_state

    # --- Startup ---
    # Configure logging FIRST (before any logging calls)
    configure_logging()
    logger.info(f"Application starting: environment={os.getenv('ENVIRONMENT', 'development')}, log_level={os.getenv('LOG_LEVEL', 'INFO')}")

    # Observability bootstrap (non-blocking, feature-flagged)
    try:
        from synap.cloud.observability.bootstrap import bootstrap_phase0_observability

        _observability_phase0_state = await bootstrap_phase0_observability()
    except Exception:
        logger.warning(
            "Observability phase-0 bootstrap failed; continuing without observability wiring",
            exc_info=True,
        )
        _observability_phase0_state = None

    # Tracing bootstrap (OTLP + FastAPI instrumentation)
    try:
        from synap.cloud.observability.tracing import bootstrap_phase1_tracing

        _observability_phase1_trace_state = await bootstrap_phase1_tracing(
            app,
            _observability_phase0_state,
        )
    except Exception:
        logger.warning(
            "Observability phase-1 tracing bootstrap failed; continuing without tracing instrumentation",
            exc_info=True,
        )
        _observability_phase1_trace_state = None

    # Metrics bootstrap (/metrics Prometheus exposition)
    try:
        from synap.cloud.observability.metrics import bootstrap_phase2_metrics

        _observability_phase2_metrics_state = await bootstrap_phase2_metrics(app)
    except Exception:
        logger.warning(
            "Observability phase-2 metrics bootstrap failed; continuing without /metrics endpoint",
            exc_info=True,
        )
        _observability_phase2_metrics_state = None

    # Observability pipeline runtime (queue/store/consumer/reporting)
    try:
        from synap.cloud.observability.phase5 import bootstrap_phase5_runtime

        _observability_phase5_state = await bootstrap_phase5_runtime()
    except Exception:
        logger.warning(
            "Observability phase-5 runtime bootstrap failed; continuing without phase-5 pipeline",
            exc_info=True,
        )
        _observability_phase5_state = None

    # Core: a missing prompt text, a missing or invalid LLM setting, or an embedding setting Core
    # does not support stops the server here with the reason. Cloud: does nothing.
    core_startup.check_at_startup()

    await init_db()

    # Core: a bad memory config file stops the server here. Deliberately outside
    # any try/except (start_ingestion_workers calls it again, but that call sits
    # inside a try that logs and carries on). Cloud: does nothing.
    check_config_at_startup()

    # Core: the folder for the copies of what is ingested must be writable, or the server stops here with
    # the reason. Cloud: does nothing.
    from synap.providers.file.registry import check_file_store_at_startup
    check_file_store_at_startup()

    # Core: create the one client and instance if missing, with the scoping mode
    # from the YAML. Cloud: does nothing. Before the ingestion workers start.
    await seed_on_startup()

    # Initialize synap-er entity resolution (graceful degradation if unavailable)
    try:
        from synap.cloud.agents.ingestion.resolution.startup import (
            init_entity_resolution,
        )

        session_factory = get_session_factory()
        db_url = get_database_url()
        await init_entity_resolution(session_factory, db_url)
    except Exception as exc:  # noqa: BLE001
        # The message used to assert a specific cause it had never checked,
        # blaming absent database tables. The real cause was an
        # UnboundLocalError a few lines up, and everyone who investigated went
        # and looked at the database instead. Report the exception, do not guess.
        _degraded("entity_resolution", exc)
        logger.warning(
            "synap-er initialization failed; entity resolution disabled: %s: %s",
            type(exc).__name__, exc, exc_info=True,
        )

    # D-007 (plan A4): cache-invalidation subscriber — applies worker-published
    # invalidations to this process's in-proc caches; flushes them on (re)subscribe.
    try:
        from synap.cloud.application.consolidation.invalidation import InvalidationSubscriber
        from synap.cloud.application.shared.providers import get_cached_providers
        app.state.invalidation_subscriber = InvalidationSubscriber(
            get_cached_providers().get("redis")
        )
        app.state.invalidation_subscriber.start()
    except Exception:  # noqa: BLE001 — TTL invariant bounds staleness without it
        logger.warning("invalidation subscriber failed to start (non-fatal)", exc_info=True)

    # Start compaction background worker (graceful degradation if unavailable)
    try:
        from synap.cloud.application.compaction import start_compaction_worker

        _compaction_worker_task = asyncio.create_task(start_compaction_worker())
        _ok("Compaction background worker started")
    except Exception:
        logger.warning(
            "Compaction worker failed to start; "
            "inline compaction via RetrievalManager still active",
            exc_info=True,
        )

    # The credit and email workers are Cloud's; Core does not try to start them.
    if edition.runs_cloud_services:
        # Start credit allowance worker (only active when credits are enabled)
        try:
            global _credit_allowance_worker, _credit_allowance_worker_task
            from synap.cloud.application.credits.allowance_worker import AllowanceWorker
            from synap.cloud.application.credits.config import get_config as _get_credit_config

            if _get_credit_config().enabled:
                _credit_allowance_worker = AllowanceWorker()
                _credit_allowance_worker_task = asyncio.create_task(
                    _credit_allowance_worker.run_forever()
                )
                _ok("Credit allowance worker started")
            else:
                logger.info("Credit allowance worker disabled (SYNAP_CREDITS_ENABLED=false)")
        except Exception:
            logger.warning(
                "Credit allowance worker failed to start", exc_info=True,
            )

        # Start credit expiry sweeper (daily-tick job that materialises
        # bucket expirations into EXPIRY ledger rows so cached_balance
        # stays accurate). Only runs when credits are enabled.
        try:
            global _credit_expiry_sweeper, _credit_expiry_sweeper_task
            from synap.cloud.application.credits.expiry_sweeper import ExpirySweeper
            from synap.cloud.application.credits.config import get_config as _get_credit_config2

            if _get_credit_config2().enabled:
                _credit_expiry_sweeper = ExpirySweeper()
                _credit_expiry_sweeper_task = asyncio.create_task(
                    _credit_expiry_sweeper.run_forever()
                )
                _ok("Credit expiry sweeper started")
            else:
                logger.info("Credit expiry sweeper disabled (SYNAP_CREDITS_ENABLED=false)")
        except Exception:
            logger.warning(
                "Credit expiry sweeper failed to start", exc_info=True,
            )

        # Start trial-cycle reset worker. Trial subscriptions have no payment webhook
        # to drive their monthly grant; this worker advances their synthetic
        # period and grants a fresh plan_monthly bucket.
        try:
            global _trial_reset_worker, _trial_reset_worker_task
            from synap.cloud.application.credits.trial_reset_worker import (
                TrialResetWorker,
            )
            from synap.cloud.application.credits.config import get_config as _get_credit_config3

            if _get_credit_config3().enabled:
                _trial_reset_worker = TrialResetWorker()
                _trial_reset_worker_task = asyncio.create_task(
                    _trial_reset_worker.run_forever()
                )
                _ok("Trial reset worker started")
            else:
                logger.info("Trial reset worker disabled (SYNAP_CREDITS_ENABLED=false)")
        except Exception:
            logger.warning("Trial reset worker failed to start", exc_info=True)

        # Start the overage billing worker and the annual-plan worker (contract
        # K: standalone overage invoices, retries, card-expiry sweep; monthly
        # grants and overage closes for annual plans). Only in the region that
        # receives payment webhooks, and only when credits are enabled.
        try:
            global _overage_billing_worker, _overage_billing_worker_task
            global _annual_plan_worker, _annual_plan_worker_task
            from synap.cloud.application.credits.annual_plan_worker import AnnualPlanWorker
            from synap.cloud.application.credits.config import get_config as _get_credit_config4
            from synap.cloud.application.credits.overage_billing_worker import (
                OverageBillingWorker,
                billing_workers_enabled,
            )

            if _get_credit_config4().enabled and billing_workers_enabled():
                _overage_billing_worker = OverageBillingWorker()
                _overage_billing_worker_task = asyncio.create_task(
                    _overage_billing_worker.run_forever()
                )
                _annual_plan_worker = AnnualPlanWorker()
                _annual_plan_worker_task = asyncio.create_task(
                    _annual_plan_worker.run_forever()
                )
                _ok("Overage billing and annual plan workers started")
            else:
                logger.info(
                    "Overage billing and annual plan workers disabled "
                    "(credits disabled or not the billing region)"
                )
        except Exception:
            logger.warning("Overage billing workers failed to start", exc_info=True)

        # Start the notification email relay (and the usage-alert sweep it runs).
        # Only the primary region emails: elsewhere notify() marks the in-app row
        # and this worker sends it once it replicates here (migration 141).
        try:
            global _notification_email_relay_worker, _notification_email_relay_worker_task
            from synap.cloud.application.notifications.email_relay import (
                NotificationEmailRelayWorker,
            )
            from synap.cloud.shared.region import is_primary_region as _is_primary_region

            if _is_primary_region():
                _notification_email_relay_worker = NotificationEmailRelayWorker()
                _notification_email_relay_worker_task = asyncio.create_task(
                    _notification_email_relay_worker.run_forever()
                )
                _ok("Notification email relay worker started")
            else:
                logger.info(
                    "Notification email relay worker disabled (not the primary region; "
                    "this region's notification emails are sent by the primary region)"
                )
        except Exception:
            logger.warning("Notification email relay worker failed to start", exc_info=True)

        # Start lifecycle onboarding-email worker. The D0 welcome already fires on
        # signup; this loop drives the D1/D3/D5/D7 next-best-action nudges, gated on
        # authoritative product data. Off by default (SYNAP_LIFECYCLE_EMAILS_ENABLED)
        # and dry-run by default so it can be enabled without dispatching real mail.
        try:
            global _lifecycle_email_worker, _lifecycle_email_worker_task
            from synap.cloud.application.lifecycle.config import (
                get_lifecycle_config as _get_lifecycle_config,
            )
            from synap.cloud.application.lifecycle.worker import LifecycleEmailWorker
            from synap.cloud.shared.region import is_primary_region as _is_primary_region2

            # Primary region only: its claims (lifecycle_email_sends) do not
            # replicate, so a loop in each region would send every nudge twice.
            if _get_lifecycle_config().enabled and not _is_primary_region2():
                logger.info("Lifecycle email worker disabled (not the primary region)")
            elif _get_lifecycle_config().enabled:
                _lifecycle_email_worker = LifecycleEmailWorker()
                _lifecycle_email_worker_task = asyncio.create_task(
                    _lifecycle_email_worker.run_forever()
                )
                _ok("Lifecycle email worker started")
            else:
                logger.info(
                    "Lifecycle email worker disabled (SYNAP_LIFECYCLE_EMAILS_ENABLED=false)"
                )
        except Exception:
            logger.warning("Lifecycle email worker failed to start", exc_info=True)

    # MACA build sweeper: restarts memory-architecture builds whose process died
    # (lease expired). Safe on every replica: a takeover is one atomic UPDATE, so
    # concurrent sweepers produce exactly one winner. Off with
    # SYNAP_MACA_BUILD_SWEEPER_ENABLED=0.
    try:
        global _maca_build_sweeper, _maca_build_sweeper_task
        _maca_started = await edition.acall("maca_build_supervisor", "start", ok=_ok)
        if _maca_started is not None:
            _maca_build_sweeper, _maca_build_sweeper_task = _maca_started
    except Exception:
        logger.warning("MACA build sweeper failed to start", exc_info=True)

    # LLM call traces: one row per attempt at a model call, written off the
    # request path. Registered before the workers below so their first calls
    # are recorded too.
    try:
        await edition.acall("llm_trace_sink", "start", ok=_ok)
    except Exception:
        logger.warning("LLM call traces failed to start", exc_info=True)

    # Start request-payload retention sweeper. Daily DELETE on
    # request_payloads.expires_at — independent of the credits stack.
    try:
        global _request_payload_sweeper, _request_payload_sweeper_task
        from synap.cloud.application.maintenance.request_payload_sweeper import (
            RequestPayloadRetentionSweeper,
        )

        _request_payload_sweeper = RequestPayloadRetentionSweeper()
        _request_payload_sweeper_task = asyncio.create_task(
            _request_payload_sweeper.run_forever()
        )
        _ok("Request payload retention sweeper started")
    except Exception:
        logger.warning(
            "Request payload retention sweeper failed to start", exc_info=True,
        )

    # Searches again, a few times over five hours, for a forget that matched nothing
    # because the note was still being saved. Needs migration 179 (logs and skips
    # without it).
    try:
        global _forget_retry_worker, _forget_retry_worker_task
        from synap.cloud.application.memory_forget.retry import ForgetRetryWorker

        _forget_retry_worker = ForgetRetryWorker()
        _forget_retry_worker_task = asyncio.create_task(_forget_retry_worker.run_forever())
        _ok("Forget retry worker started")
    except Exception:
        logger.warning("Forget retry worker failed to start", exc_info=True)

    # Expire memories past guardrails.retention.max_memory_age_days. Off unless
    # SYNAP_GUARDRAIL_RETENTION is on (Core sets it, Cloud leaves it off).
    try:
        global _retention_worker, _retention_worker_task
        from synap.cloud.application.retention import RetentionWorker, retention_wanted

        if retention_wanted():
            _retention_worker = RetentionWorker()
            _retention_worker_task = asyncio.create_task(_retention_worker.run_forever())
            _ok("Retention worker started")
    except Exception:
        logger.warning("Retention worker failed to start", exc_info=True)

    # Whether a read ever serves one scope's memories to another, asked daily.
    #
    # Nothing asked before. Cross-user serving was real on some
    # instances and was found by somebody replaying it by hand, not by us. The
    # answer is checkable because `request_payloads` holds the requester and the
    # response in the same row, so this is a query we simply were not running.
    try:
        global _scope_bleed_audit, _scope_bleed_audit_task
        _scope_started = edition.call("daily_scope_check", ok=_ok)
        if _scope_started is not None:
            _scope_bleed_audit, _scope_bleed_audit_task = _scope_started
    except Exception:
        logger.warning("Scope bleed audit failed to start", exc_info=True)

    # The vector-side ancestor backfill that inserting a level makes necessary.
    #
    # It was owed and nothing ran it: the record was a module-level dict that
    # died at the next deploy, and the whole mechanism was a warning line. A
    # client who added a rung got a rung that undercounted their own history,
    # silently and indefinitely.
    try:
        global _ancestor_backfill_worker, _ancestor_backfill_worker_task
        from synap.cloud.application.maintenance.ancestor_backfill_worker import (
            AncestorBackfillWorker,
        )

        _ancestor_backfill_worker = AncestorBackfillWorker()
        _ancestor_backfill_worker_task = asyncio.create_task(
            _ancestor_backfill_worker.run_forever()
        )
        _ok("Ancestor backfill worker started")
    except Exception:
        logger.warning("Ancestor backfill worker failed to start", exc_info=True)

    # Start the anticipation telemetry sweep. Same shape as the one above, and
    # started here for the reason that one is: the pool's retention was
    # shortened but the only thing that enforced it was a script meant to be
    # run by a cron nobody had written, so the window was a sentence in a
    # document while old turn text sat in the table.
    try:
        global _anticipation_telemetry_sweeper, _anticipation_telemetry_sweeper_task
        from synap.cloud.application.maintenance.anticipation_telemetry_sweeper import (
            AnticipationTelemetrySweeper,
        )

        _anticipation_telemetry_sweeper = AnticipationTelemetrySweeper()
        _anticipation_telemetry_sweeper_task = asyncio.create_task(
            _anticipation_telemetry_sweeper.run_forever()
        )
        _ok("Anticipation telemetry retention sweeper started")
    except Exception:
        logger.warning(
            "Anticipation telemetry retention sweeper failed to start", exc_info=True,
        )

    # PII watch-mode counter flush and detector heartbeat.
    #
    # Built here rather than lazily on first use, and loudly, because the
    # thing this whole feature replaces was a control that silently did
    # nothing. If the gate cannot be constructed the process says so at boot
    # instead of quietly scanning nothing forever (invariant I2).
    try:
        global _pii_counters, _pii_counters_task
        from synap.cloud.privacy.counters import get_counters
        from synap.cloud.privacy.gate import get_gate

        gate = get_gate()  # raises if it has no recognisers
        _pii_counters = get_counters()
        _pii_counters_task = asyncio.create_task(_pii_counters.run_forever())
        logger.info(
            "PII ingress gate ready detector_version=%s recognizers=%d",
            gate.detector_version,
            len(gate._recognizers),
        )
    except Exception:
        logger.error(
            "PII ingress gate FAILED to start — documents will be ingested "
            "without detection and the admin health panel will show this "
            "worker as not running",
            exc_info=True,
        )

    # Start context-audit-enrichment finalize-on-timeout watcher. Backfills
    # synthetic enrichment rows when the SDK fails to emit
    # ContextAssembledEvent within the timeout window
    # (see docs/internal/sdk_authoritative_short_term_context_plan.md).
    try:
        global _enrichment_watcher, _enrichment_watcher_task
        from synap.cloud.application.audit.enrichment_watcher import EnrichmentWatcher
        from synap.cloud.application.shared.providers import get_redis_provider
        # `get_session_factory` is imported at module level. Re-importing it here
        # made it local to the whole of `lifespan()`, so the use ~180 lines above
        # raised UnboundLocalError and entity resolution was off on every boot.

        redis_client = await get_redis_provider()
        session_factory = get_session_factory()
        _enrichment_watcher = EnrichmentWatcher(
            redis_client=redis_client,
            session_factory=session_factory,
            check_interval_seconds=float(
                os.getenv("ENRICHMENT_WATCHER_INTERVAL_SECONDS", "5")
            ),
            timeout_seconds=float(
                os.getenv("ENRICHMENT_WATCHER_TIMEOUT_SECONDS", "10")
            ),
        )
        _enrichment_watcher_task = asyncio.create_task(
            _enrichment_watcher.start()
        )
        _ok("Context-assembled enrichment watcher started")
    except Exception:
        logger.warning(
            "Enrichment watcher failed to start", exc_info=True,
        )

    # Start ingestion pipeline workers (graceful degradation if unavailable)
    try:
        from synap.cloud.application.ingestion.startup import start_ingestion_workers

        await start_ingestion_workers()
    except Exception:
        logger.warning(
            "Ingestion workers failed to start; "
            "memory ingestion will queue but not process",
            exc_info=True,
        )

    # Observability gauges the application must publish itself, because no exporter
    # can see inside the process: the commit this region is running, whether each
    # scheduled worker is still ticking, how deep the per-client ingestion queues are,
    # and whether this region's database is missing a migration artifact. Four signals
    # in the registry were blocked on these. Checklist p5-3 to p5-6.
    try:
        from synap.cloud.observability.runtime_gauges import publish_build_info

        publish_build_info()
        _ok("Build info gauge published")

        # Schema drift, asked once at boot. The parser derives what to look for from
        # the migration FILES rather than a list somebody maintains, so adding a
        # migration needs no change here.
        # `get_session_factory` is imported at module level and used ~350 lines
        # up by init_entity_resolution. Re-importing it here made it local to
        # the whole of `lifespan()`, so that earlier use raised
        # UnboundLocalError and entity resolution was off on every boot: the
        # twin of the bug fixed above, left behind because it reads as a
        # harmless local import.
        from synap.cloud.observability.runtime_gauges import publish_schema_drift

        # Held, not fired and forgotten. asyncio keeps only a WEAK reference to a task,
        # so one whose handle is discarded can be garbage-collected mid-flight. Every
        # other worker in this file assigns its task to a name for exactly this reason;
        # these two did not, and the ingestion sampler was collected on the first
        # boot. The gauges never appeared, nothing raised, and the failure log line is
        # at debug level so it would not have shown even if it had.
        _schema_drift_task = asyncio.create_task(publish_schema_drift(get_session_factory()))
        app.state.schema_drift_task = _schema_drift_task

        # Ingestion queue depth, sampled on a timer. No exporter can do this: the
        # queues are per-client Redis keys, so counting them means a bounded SCAN
        # from inside the process that already holds a connection.
        from synap.cloud.observability.runtime_gauges import (
            sample_ingestion_queues,
            sample_postgres_connections,
        )

        async def _sample_queues_forever() -> None:
            from synap.cloud.application.shared.providers import get_redis_provider
            while True:
                try:
                    await sample_ingestion_queues(await get_redis_provider())
                    # Rides the same tick rather than starting a second task: it is one
                    # cheap query a minute, and a second loop is a second thing that can
                    # be garbage-collected mid-flight, which is how the queue sampler
                    # silently stopped in the first place.
                    await sample_postgres_connections(get_session_factory())
                except Exception:
                    # warning, not debug: a sampler failing every tick at debug level is
                    # indistinguishable from one that is working, which is how this stayed
                    # invisible. Once a minute is not enough volume to be noise.
                    logger.warning("ingestion queue sampling tick failed", exc_info=True)
                await asyncio.sleep(60)

        _queue_sampler_task = asyncio.create_task(_sample_queues_forever())
        app.state.queue_sampler_task = _queue_sampler_task
        _ok("Ingestion queue sampler started")
    except Exception:
        logger.warning("Build info gauge failed; region version drift stays invisible",
                       exc_info=True)

    # Engine B (Cloud only). Evaluates the signals whose answer lives in our own databases,
    # and runs once per deployment.
    # A monitor watches its heartbeat, because an engine that has
    # stopped looks exactly like an engine with nothing to report.
    try:
        from synap.cloud.observability.alerts.runner import SignalRunner
        from synap.cloud.shared.region import is_primary_region

        if is_primary_region():
            _signal_runner = SignalRunner()
            _signal_runner_task = asyncio.create_task(_signal_runner.run())
            app.state.signal_runner = _signal_runner
            _ok("SignalRunner started (primary region)")
        else:
            logger.info("SignalRunner not started: this is not the primary region")
    except Exception:
        logger.warning("SignalRunner failed to start; SQL-sourced signals will not fire",
                       exc_info=True)

    # Start usage aggregation worker (graceful degradation if unavailable)
    try:
        _aggregation_started = edition.call("analytics_worker_start", ok=_ok)
        if _aggregation_started is not None:
            _aggregation_worker, _aggregation_worker_task = _aggregation_started
    except Exception:
        logger.warning(
            "Usage aggregation worker failed to start; usage_rollups will remain stale",
            exc_info=True,
        )

    # Initialize shared providers for RetrievalManager (graceful degradation if unavailable)
    try:
        from synap.cloud.application.shared.providers import initialize_providers_once

        await initialize_providers_once()
        _ok("Shared providers initialized for context retrieval")
    except Exception:
        logger.warning(
            "Shared providers initialization failed; "
            "context fetch may return empty results",
            exc_info=True,
        )

    # Instance deletion, migration and the connector workers are Cloud's; Core does not try to start them.
    if edition.runs_cloud_services:
        # Start instance deletion background worker (graceful degradation if unavailable)
        try:
            from synap.cloud.application.admin_api.instance_deletion_worker import (
                InstanceDeletionWorker,
            )

            _instance_deletion_worker = InstanceDeletionWorker(
                poll_interval_seconds=int(
                    os.getenv("INSTANCE_DELETION_POLL_INTERVAL_SECONDS", "3")
                )
            )
            _instance_deletion_worker_task = asyncio.create_task(
                _instance_deletion_worker.start()
            )
            _ok("Instance deletion background worker started")
        except Exception:
            logger.warning(
                "Instance deletion worker failed to start; async deletion jobs will remain queued",
                exc_info=True,
            )

        # Start migration background worker (graceful degradation if unavailable)
        try:
            from synap.cloud.application.migration.worker import MigrationWorker

            _migration_worker = MigrationWorker(
                poll_interval_seconds=int(
                    os.getenv("MIGRATION_POLL_INTERVAL_SECONDS", "5")
                )
            )
            _migration_worker_task = asyncio.create_task(_migration_worker.start())
            _ok("Migration background worker started")
        except Exception:
            logger.warning(
                "Migration worker failed to start; uploaded exports will remain queued",
                exc_info=True,
            )

        # Start Google Drive sync worker (graceful degradation if GCP not configured)
        try:
            from synap.cloud.agents.connectors.drive_sync_worker import DriveSyncWorker

            _drive_sync_worker = DriveSyncWorker(
                interval_seconds=int(os.getenv("GOOGLE_DRIVE_SYNC_INTERVAL_SECONDS", "3600"))
            )
            _drive_sync_worker_task = asyncio.create_task(_drive_sync_worker.start())
            _ok("Google Drive sync worker started")
        except Exception:
            logger.warning(
                "Google Drive sync worker failed to start; Drive connector unavailable",
                exc_info=True,
            )

        # Start Notion sync worker (graceful degradation if unavailable)
        try:
            from synap.cloud.agents.connectors.notion_sync_worker import NotionSyncWorker

            _notion_sync_worker = NotionSyncWorker(
                interval_seconds=int(os.getenv("NOTION_SYNC_INTERVAL_SECONDS", "3600"))
            )
            _notion_sync_worker_task = asyncio.create_task(_notion_sync_worker.start())
            _ok("Notion sync worker started")
        except Exception:
            logger.warning(
                "Notion sync worker failed to start; Notion connector unavailable",
                exc_info=True,
            )

        # Start SharePoint sync worker (graceful degradation if not configured)
        try:
            from synap.cloud.agents.connectors.sharepoint_sync_worker import SharePointSyncWorker

            _sharepoint_sync_worker = SharePointSyncWorker(
                interval_seconds=int(os.getenv("SHAREPOINT_SYNC_INTERVAL_SECONDS", "3600"))
            )
            _sharepoint_sync_worker_task = asyncio.create_task(_sharepoint_sync_worker.start())
            _ok("SharePoint sync worker started")
        except Exception:
            logger.warning(
                "SharePoint sync worker failed to start; SharePoint connector unavailable",
                exc_info=True,
            )

        # Start Slack sync worker (graceful degradation if not configured)
        try:
            from synap.cloud.agents.connectors.slack_sync_worker import SlackSyncWorker

            _slack_sync_worker = SlackSyncWorker(
                interval_seconds=int(os.getenv("SLACK_SYNC_INTERVAL_SECONDS", "21600"))
            )
            _slack_sync_worker_task = asyncio.create_task(_slack_sync_worker.start())
            _ok("Slack sync worker started")
        except Exception:
            logger.warning(
                "Slack sync worker failed to start; Slack connector unavailable",
                exc_info=True,
            )

        # Start Linear sync worker (graceful degradation if not configured)
        try:
            from synap.cloud.agents.connectors.linear_sync_worker import LinearSyncWorker

            _linear_sync_worker = LinearSyncWorker(
                interval_seconds=int(os.getenv("LINEAR_SYNC_INTERVAL_SECONDS", "3600"))
            )
            _linear_sync_worker_task = asyncio.create_task(_linear_sync_worker.start())
            _ok("Linear sync worker started")
        except Exception:
            logger.warning(
                "Linear sync worker failed to start; Linear connector unavailable",
                exc_info=True,
            )

        # Start Mintlify sync worker (graceful degradation if not configured)
        try:
            from synap.cloud.agents.connectors.mintlify_sync_worker import MintlifySyncWorker

            _mintlify_sync_worker = MintlifySyncWorker(
                interval_seconds=int(os.getenv("MINTLIFY_SYNC_INTERVAL_SECONDS", "604800"))
            )
            _mintlify_sync_worker_task = asyncio.create_task(_mintlify_sync_worker.start())
            _ok("Mintlify sync worker started")
        except Exception:
            logger.warning(
                "Mintlify sync worker failed to start; Mintlify connector unavailable",
                exc_info=True,
            )

        # Start Zendesk sync worker (graceful degradation if not configured)
        try:
            from synap.cloud.agents.connectors.zendesk_sync_worker import ZendeskSyncWorker

            _zendesk_sync_worker = ZendeskSyncWorker(
                interval_seconds=int(os.getenv("ZENDESK_SYNC_INTERVAL_SECONDS", "3600"))
            )
            _zendesk_sync_worker_task = asyncio.create_task(_zendesk_sync_worker.start())
            _ok("Zendesk sync worker started")
        except Exception:
            logger.warning(
                "Zendesk sync worker failed to start; Zendesk connector unavailable",
                exc_info=True,
            )

    # Start gRPC streaming server (graceful degradation if unavailable)
    global _grpc_server
    # Core does not serve gRPC: a plain yes/no value of the edition.
    if edition.serves_grpc:
        try:
            from synap.cloud.application.grpc.server import create_grpc_server
            from synap.cloud.gateway.routing.stream_manager import StreamManager
            from synap.cloud.agents.listening_retrieval.agent import ListeningRetrievalAgent
            from synap.cloud.application.conversation.tracker import ConversationTracker

            stream_manager = StreamManager()

            # Fetch shared providers (already initialized above)
            from synap.cloud.application.shared.providers import get_cached_providers
            providers = get_cached_providers()

            # Pass Redis to tracker for raw conversation turn buffering
            conversation_tracker = ConversationTracker(redis_client=providers.get("redis"))

            # Reuse the singleton RetrievalManager so gRPC and REST share
            # the same L2 in-memory adjacency cache.
            from synap.cloud.application.sdk_api.sdk_context import get_retrieval_manager
            retrieval_manager = get_retrieval_manager()

            from synap.cloud.infrastructure.llm import get_llm_manager

            agent = ListeningRetrievalAgent(
                llm_config=get_llm_manager(),
                retrieval_manager=retrieval_manager,
                conversation_tracker=conversation_tracker,
                redis_client=providers.get("redis"),
            )

            # Store push_queues reference so post-compaction callback can push
            # compaction_update bundles to the gRPC stream (Proposal 1).
            from synap.cloud.application.shared.providers import _providers_cache
            _providers_cache["push_queues"] = agent._push_queues
            # The HTTP event route dispatches into this same agent. Without it
            # there would be two listening paths doing slightly different things
            # with the same event, which is the drift the transport parity document
            # exists to prevent.
            _providers_cache["listening_agent"] = agent

            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            _grpc_server = await create_grpc_server(
                stream_manager=stream_manager,
                agent=agent,
                conversation_tracker=conversation_tracker,
                port=grpc_port,
            )
            await _grpc_server.start()
            _ok("gRPC streaming server started on port %d", grpc_port)

            # Construct the AnticipationRuntime — pool-of-workers + Redis-backed
            # state. Replaces the per-conversation AnticipationAgent + AgentSessionManager
            # that existed before. See
            # synap/docs/architecture/anticipation_runtime_migration_plan_2026_06_02.md
            # for the design and rollout history. The ANTICIPATION_RUNTIME_MODE
            # flag is gone; this is the only anticipation path now.
            try:
                from synap.cloud.agents.anticipation_runtime import (
                    AnticipationRuntime,
                    set_runtime,
                )
                anticipation_runtime = AnticipationRuntime(
                    redis_client=providers.get("redis"),
                    llm_manager=get_llm_manager(),
                    retrieval_manager=retrieval_manager,
                    stream_router=agent._streams,
                )
                await anticipation_runtime.start()
                set_runtime(anticipation_runtime)
                # Park on providers cache so listen_node can reach it without
                # importing the singleton helper directly.
                _providers_cache["anticipation_runtime"] = anticipation_runtime
                _ok(
                    "AnticipationRuntime started (pool_size=%d)",
                    anticipation_runtime.metrics()["pool_size"],
                )
            except Exception:
                logger.warning(
                    "AnticipationRuntime start failed — anticipation will "
                    "silently no-op until next deploy",
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "gRPC server failed to start; streaming unavailable",
                exc_info=True,
            )
            _grpc_server = None

    # The playground packages are Cloud's; Core does not try to start them.
    if edition.runs_cloud_services:
        # Initialize playground SDK instances and open per-agent gRPC streams
        # AFTER lifespan yields. Lifespan blocks uvicorn from binding the port
        # until it returns, but the SDK's HTTP transport calls back to
        # ``http://localhost:8000/api/v1/auth/whoami`` during its own
        # ``initialize()`` — a chicken-and-egg if we ``await`` here. Schedule
        # the init as a task instead; the event loop runs it after the
        # current coroutine yields, by which time the server is listening.
        try:
            from synap.cloud.application.admin_api.routes.playground_grpc import (
                initialize_playground_sdks,
            )

            async def _deferred_playground_init():
                try:
                    # Tiny yield so the ASGI server gets through its own startup
                    # handshake and binds the listener before we hit it. Anything
                    # past this point sees a live HTTP server on localhost:8000.
                    await asyncio.sleep(0.1)
                    await initialize_playground_sdks()
                    _ok("Playground gRPC streams initialized")
                except Exception:
                    logger.warning(
                        "Playground gRPC streams failed to initialize",
                        exc_info=True,
                    )

            asyncio.create_task(_deferred_playground_init())
        except Exception:
            logger.warning("Playground gRPC streams scheduling failed", exc_info=True)

        # Playground v2 session sweeper — deletes expired visitor sessions and
        # their cloned memory rows + Chroma docs. Runs in the background so the
        # demo DB doesn't grow infinitely as visitors mint fresh sessions.
        try:
            from synap.cloud.application.playground.v2.sweeper import (
                start_playground_v2_sweeper,
            )

            start_playground_v2_sweeper()
            _ok("Playground v2 session sweeper started")
        except Exception:
            logger.warning("Playground v2 sweeper failed to start", exc_info=True)

        # Playground v2 voice — pre-warm the worker pool (models loaded + SDK
        # initialised) so the first visitor connects in seconds, not a minute.
        # Scheduled as a background task: model loading must NOT block uvicorn
        # from binding the port.
        try:
            from synap.cloud.application.playground.v2.voice.spawner import (
                prewarm_voice_pool,
            )

            asyncio.create_task(prewarm_voice_pool())
            logger.info("Playground v2 voice pool pre-warm scheduled")
        except Exception:
            logger.warning("Playground v2 voice pool pre-warm scheduling failed", exc_info=True)

    logger.info(
        "Application startup complete: %d started, %d degraded%s",
        len(_started_services),
        len(_degraded_services),
        (" | degraded: " + ", ".join(_degraded_services)) if _degraded_services else "",
    )
    if _degraded_services:
        # Loud on purpose. A subsystem that degraded silently is the whole
        # reason this summary stopped being a constant.
        logger.warning(
            "Startup finished with %d degraded subsystem(s): %s",
            len(_degraded_services), ", ".join(_degraded_services),
        )

    yield

    # --- Shutdown ---
    logger.info("Application shutdown initiated")

    # The playground packages are Cloud's; Core does not try to stop them.
    if edition.runs_cloud_services:
        # Stop the playground v2 sweeper first so it doesn't try to run a
        # cleanup pass while we're tearing down the DB pool below.
        try:
            from synap.cloud.application.playground.v2.sweeper import (
                stop_playground_v2_sweeper,
            )

            await stop_playground_v2_sweeper()
        except Exception:
            pass

        # Shutdown the AnticipationRuntime next so any in-flight invocations
        # (LLM loops, pending bundle pushes) drain before we tear down the
        # gRPC streams they push to. Must run BEFORE ``shutdown_playground_sdks()``
        # immediately below — the runtime holds references to the SDK's gRPC
        # stream router and pushing on a torn-down stream would raise
        # mid-shutdown.
        try:
            from synap.cloud.agents.anticipation_runtime import get_runtime
            rt = get_runtime()
            if rt is not None:
                await rt.shutdown()
                logger.info("AnticipationRuntime shutdown complete")
        except Exception:
            logger.debug("AnticipationRuntime shutdown raised (non-fatal)", exc_info=True)

        # Shutdown playground SDK gRPC streams before other workers
        try:
            from synap.cloud.application.admin_api.routes.playground_grpc import (
                shutdown_playground_sdks,
            )

            await shutdown_playground_sdks()
        except Exception:
            pass

    # Stop ingestion workers
    try:
        from synap.cloud.application.ingestion.startup import stop_ingestion_workers

        await stop_ingestion_workers()
        logger.info("Ingestion workers stopped")
    except Exception:
        pass

    # Stop compaction worker
    # Stop credit allowance worker
    if _credit_allowance_worker is not None:
        try:
            _credit_allowance_worker.stop()
        except Exception:
            pass
    if _credit_allowance_worker_task is not None:
        try:
            await asyncio.wait_for(_credit_allowance_worker_task, timeout=5.0)
        except Exception:
            _credit_allowance_worker_task.cancel()

    # Stop credit expiry sweeper
    if _credit_expiry_sweeper is not None:
        try:
            _credit_expiry_sweeper.stop()
        except Exception:
            pass
    if _credit_expiry_sweeper_task is not None:
        try:
            await asyncio.wait_for(_credit_expiry_sweeper_task, timeout=5.0)
        except Exception:
            _credit_expiry_sweeper_task.cancel()

    # Stop trial-cycle reset worker
    if _trial_reset_worker is not None:
        try:
            _trial_reset_worker.stop()
        except Exception:
            pass
    if _trial_reset_worker_task is not None:
        try:
            await asyncio.wait_for(_trial_reset_worker_task, timeout=5.0)
        except Exception:
            _trial_reset_worker_task.cancel()

    # Stop overage billing and annual plan workers
    for _billing_worker, _billing_task in (
        (_overage_billing_worker, _overage_billing_worker_task),
        (_annual_plan_worker, _annual_plan_worker_task),
        (_notification_email_relay_worker, _notification_email_relay_worker_task),
    ):
        if _billing_worker is not None:
            try:
                _billing_worker.stop()
            except Exception:
                pass
        if _billing_task is not None:
            try:
                await asyncio.wait_for(_billing_task, timeout=5.0)
            except Exception:
                _billing_task.cancel()

    # Stop the MACA build sweeper, then hand any build still running in this
    # process back (lease expired now) so a surviving replica restarts it at once.
    if _maca_build_sweeper is not None:
        try:
            _maca_build_sweeper.stop()
        except Exception:
            pass
    if _maca_build_sweeper_task is not None:
        _maca_build_sweeper_task.cancel()
        try:
            await asyncio.wait_for(_maca_build_sweeper_task, timeout=5.0)
        except BaseException:
            pass
    try:
        await asyncio.wait_for(
            edition.acall("maca_build_supervisor", "cancel_running"), timeout=10.0
        )
    except Exception:
        pass

    # Stop lifecycle email worker
    if _lifecycle_email_worker is not None:
        try:
            _lifecycle_email_worker.stop()
        except Exception:
            pass
    if _lifecycle_email_worker_task is not None:
        try:
            await asyncio.wait_for(_lifecycle_email_worker_task, timeout=5.0)
        except Exception:
            _lifecycle_email_worker_task.cancel()

    # Stop request-payload retention sweeper
    # Write the LLM call traces still queued. Before the database engines
    # are disposed further down.
    try:
        await edition.acall("llm_trace_sink", "shutdown")
    except Exception:
        pass

    if _request_payload_sweeper is not None:
        try:
            _request_payload_sweeper.stop()
        except Exception:
            pass
    if _request_payload_sweeper_task is not None:
        try:
            await asyncio.wait_for(_request_payload_sweeper_task, timeout=5.0)
        except Exception:
            _request_payload_sweeper_task.cancel()

    # Stop the forget retry worker
    if _forget_retry_worker is not None:
        try:
            _forget_retry_worker.stop()
        except Exception:
            pass
    if _forget_retry_worker_task is not None:
        try:
            await asyncio.wait_for(_forget_retry_worker_task, timeout=5.0)
        except Exception:
            _forget_retry_worker_task.cancel()

    # Stop the retention worker
    if _retention_worker is not None:
        try:
            _retention_worker.stop()
        except Exception:
            pass
    if _retention_worker_task is not None:
        try:
            await asyncio.wait_for(_retention_worker_task, timeout=5.0)
        except Exception:
            _retention_worker_task.cancel()

    # Stop the scope bleed audit
    if _scope_bleed_audit is not None:
        try:
            _scope_bleed_audit.stop()
        except Exception:
            pass
    if _scope_bleed_audit_task is not None:
        try:
            await asyncio.wait_for(_scope_bleed_audit_task, timeout=5.0)
        except Exception:
            _scope_bleed_audit_task.cancel()

    # Stop the ancestor backfill worker
    if _ancestor_backfill_worker is not None:
        try:
            _ancestor_backfill_worker.stop()
        except Exception:
            pass
    if _ancestor_backfill_worker_task is not None:
        try:
            await asyncio.wait_for(_ancestor_backfill_worker_task, timeout=5.0)
        except Exception:
            _ancestor_backfill_worker_task.cancel()

    # Stop the anticipation telemetry sweeper
    if _anticipation_telemetry_sweeper is not None:
        try:
            _anticipation_telemetry_sweeper.stop()
        except Exception:
            pass
    if _anticipation_telemetry_sweeper_task is not None:
        try:
            await asyncio.wait_for(_anticipation_telemetry_sweeper_task, timeout=5.0)
        except Exception:
            _anticipation_telemetry_sweeper_task.cancel()

    # Stop the PII counter flusher. `run_forever` flushes once more on its way
    # out, so a clean shutdown loses no counts.
    if _pii_counters is not None:
        try:
            _pii_counters.stop()
        except Exception:
            pass
    if _pii_counters_task is not None:
        try:
            await asyncio.wait_for(_pii_counters_task, timeout=5.0)
        except Exception:
            _pii_counters_task.cancel()

    # Stop context-audit enrichment watcher
    if _enrichment_watcher is not None:
        try:
            await _enrichment_watcher.stop()
        except Exception:
            pass
    if _enrichment_watcher_task is not None:
        try:
            await asyncio.wait_for(_enrichment_watcher_task, timeout=5.0)
        except Exception:
            _enrichment_watcher_task.cancel()

    if _compaction_worker_task is not None:
        try:
            from synap.cloud.application.compaction import stop_compaction_worker

            await stop_compaction_worker()
            _compaction_worker_task.cancel()
            _compaction_worker_task = None
            logger.info("Compaction background worker stopped")
        except Exception:
            pass

    # Stop usage aggregation worker
    if _aggregation_worker_task is not None and _aggregation_worker is not None:
        try:
            await _aggregation_worker.stop()
            _aggregation_worker_task.cancel()
            _aggregation_worker_task = None
            _aggregation_worker = None
            logger.info("Usage aggregation worker stopped")
        except Exception:
            pass

    # Stop instance deletion worker
    if _instance_deletion_worker_task is not None and _instance_deletion_worker is not None:
        try:
            await _instance_deletion_worker.stop()
            _instance_deletion_worker_task.cancel()
            _instance_deletion_worker_task = None
            _instance_deletion_worker = None
            logger.info("Instance deletion background worker stopped")
        except Exception:
            pass

    # Stop Google Drive sync worker
    if _drive_sync_worker_task is not None and _drive_sync_worker is not None:
        try:
            await _drive_sync_worker.stop()
            _drive_sync_worker_task.cancel()
            _drive_sync_worker_task = None
            _drive_sync_worker = None
            logger.info("Google Drive sync worker stopped")
        except Exception:
            pass

    # Stop Notion sync worker
    if _notion_sync_worker_task is not None and _notion_sync_worker is not None:
        try:
            await _notion_sync_worker.stop()
            _notion_sync_worker_task.cancel()
            _notion_sync_worker_task = None
            _notion_sync_worker = None
            logger.info("Notion sync worker stopped")
        except Exception:
            pass

    # Stop SharePoint sync worker
    if _sharepoint_sync_worker_task is not None and _sharepoint_sync_worker is not None:
        try:
            await _sharepoint_sync_worker.stop()
            _sharepoint_sync_worker_task.cancel()
            _sharepoint_sync_worker_task = None
            _sharepoint_sync_worker = None
            logger.info("SharePoint sync worker stopped")
        except Exception:
            pass

    # Stop Slack sync worker
    if _slack_sync_worker_task is not None and _slack_sync_worker is not None:
        try:
            await _slack_sync_worker.stop()
            _slack_sync_worker_task.cancel()
            _slack_sync_worker_task = None
            _slack_sync_worker = None
            logger.info("Slack sync worker stopped")
        except Exception:
            pass

    # Stop Linear sync worker
    if _linear_sync_worker_task is not None and _linear_sync_worker is not None:
        try:
            await _linear_sync_worker.stop()
            _linear_sync_worker_task.cancel()
            _linear_sync_worker_task = None
            _linear_sync_worker = None
            logger.info("Linear sync worker stopped")
        except Exception:
            pass

    # Stop Mintlify sync worker
    if _mintlify_sync_worker_task is not None and _mintlify_sync_worker is not None:
        try:
            await _mintlify_sync_worker.stop()
            _mintlify_sync_worker_task.cancel()
            _mintlify_sync_worker_task = None
            _mintlify_sync_worker = None
            logger.info("Mintlify sync worker stopped")
        except Exception:
            pass

    # Stop Zendesk sync worker
    if _zendesk_sync_worker_task is not None and _zendesk_sync_worker is not None:
        try:
            await _zendesk_sync_worker.stop()
            _zendesk_sync_worker_task.cancel()
            _zendesk_sync_worker_task = None
            _zendesk_sync_worker = None
            logger.info("Zendesk sync worker stopped")
        except Exception:
            pass

    # Stop gRPC server
    if _grpc_server is not None:
        try:
            await _grpc_server.stop(grace=5.0)
            logger.info("gRPC streaming server stopped")
        except Exception:
            pass

    try:
        from synap.cloud.agents.ingestion.resolution.startup import (
            shutdown_entity_resolution,
        )

        await shutdown_entity_resolution()
    except Exception:
        pass

    try:
        from synap.cloud.observability.phase5 import shutdown_phase5_runtime

        await shutdown_phase5_runtime(_observability_phase5_state)
        _observability_phase5_state = None
    except Exception:
        pass

    try:
        from synap.cloud.observability.tracing import shutdown_phase1_tracing

        await shutdown_phase1_tracing(app, _observability_phase1_trace_state)
        _observability_phase1_trace_state = None
    except Exception:
        pass

    try:
        from synap.cloud.observability.metrics import shutdown_phase2_metrics

        await shutdown_phase2_metrics(_observability_phase2_metrics_state)
        _observability_phase2_metrics_state = None
    except Exception:
        pass

    try:
        from synap.cloud.observability.bootstrap import shutdown_phase0_observability

        await shutdown_phase0_observability(_observability_phase0_state)
        _observability_phase0_state = None
    except Exception:
        pass

    # Analytics shutdown goes LAST among the analytics-adjacent teardown:
    # drain in-flight detached retrieval/metering tasks (which may enqueue
    # activation events) FIRST so nothing captures after the delivery queue is
    # torn down, THEN flush and shut down the analytics client. Without the
    # drain the last pre-shutdown activation events can drop on a restart.
    try:
        from synap.cloud.application.client_api.retrieval_manager import (
            drain_detached_tasks,
        )

        drained = await drain_detached_tasks()
        await edition.acall("analytics_shutdown", drained)
    except Exception:
        logger.debug("Analytics shutdown flush raised (non-fatal)", exc_info=True)

    await close_db()

    # Shut down the dedicated Chroma thread pool. Without this, the
    # interpreter may hang waiting on the executor at exit.
    try:
        from synap.cloud.agents.ingestion.storage.vector_wrapper import (
            shutdown_chroma_executor,
        )
        shutdown_chroma_executor()
    except Exception:
        pass

    logger.info("Application shutdown complete")


# OpenAPI docs exposure depends on environment:
#   production → docs fully disabled (no /docs, /redoc, /openapi.json)
#   anything else → docs gated behind HTTP Basic (admin + DOCS_PASSWORD)
_ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
_DOCS_DISABLED = _ENVIRONMENT == "production"

app = FastAPI(
    title="Synap Dashboard API",
    description="Admin dashboard and client API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Core: swap in the fixed identity. Cloud: does nothing. Before routers are included.
install_edition_identity(app)


@app.get("/health", include_in_schema=False)
async def health():
    # Core: "not ready" (503) until the model downloads have finished.
    if edition.waits_for_first_start:
        from pathlib import Path

        from fastapi.responses import JSONResponse

        from synap.boot import first_start

        if not first_start.is_done(Path(os.environ.get("HF_HOME") or "/models")):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "edition": edition.name,
                    "reason": "The embedding model and the spaCy model are still downloading, or the "
                              "download failed. See the first-start container's log.",
                },
            )
    # Core: "not ready" (503) while an ingestion worker is down, so a server that cannot process a
    # memory does not look healthy. Cloud keeps answering ok.
    if edition.health_needs_ingestion:
        from synap.cloud.application.ingestion import startup as _ingestion_startup

        problems = _ingestion_startup.start_problems()
        if problems:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "edition": edition.name,
                    "reason": "Memories cannot be processed because these did not start: "
                              + ", ".join(problems) + ". The first error is in the synap log.",
                },
            )
    return {"status": "ok", "edition": edition.name}


if not _DOCS_DISABLED:
    _docs_security = HTTPBasic()
    _DOCS_USERNAME = "admin"
    _DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "").strip()

    if not _DOCS_PASSWORD:
        logger.warning(
            "DOCS_PASSWORD is not set in environment '%s'; /docs, /redoc, "
            "and /openapi.json will reject all requests until it is configured.",
            _ENVIRONMENT,
        )

    def _authenticate_docs(
        credentials: HTTPBasicCredentials = Depends(_docs_security),
    ) -> str:
        expected_password = _DOCS_PASSWORD
        unauthorized = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
        if not expected_password:
            raise unauthorized
        user_ok = secrets.compare_digest(credentials.username, _DOCS_USERNAME)
        pass_ok = secrets.compare_digest(credentials.password, expected_password)
        if not (user_ok and pass_ok):
            raise unauthorized
        return credentials.username

    @app.get("/openapi.json", include_in_schema=False)
    async def _protected_openapi(_: str = Depends(_authenticate_docs)) -> dict:
        return app.openapi()

    @app.get("/docs", include_in_schema=False)
    async def _protected_swagger(_: str = Depends(_authenticate_docs)):
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} — Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def _protected_redoc(_: str = Depends(_authenticate_docs)):
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{app.title} — ReDoc",
        )

# Cloud installs the full API (and the management plane) through a hook and
# returns True. With nothing registered (Core) only the SDK routes are served.
if not edition.call("cloud_routers", app):
    app.include_router(sdk_router)

# ------------------------------------------------------------------
# Credit system exception handler + response header middleware
# ------------------------------------------------------------------
edition.call("credit_enforcement", app)

# --- B2C identifier contract ---
# Deliberately NOT inside the try/except above. That block swallows any failure
# and continues, which is right for optional middleware and wrong here: if this
# handler fails to register, a rejection surfaces as a 500 instead of a 400 and
# the client is told nothing useful. A failure here should be loud at boot.
from fastapi import Request as _B2CRequest
from fastapi.responses import JSONResponse as _B2CJSONResponse
from synap.cloud.shared.b2c_contract import B2CCustomerIdRejected as _B2CRejected


def _b2c_customer_id_to_http(_request: _B2CRequest, exc: _B2CRejected) -> _B2CJSONResponse:
    """400 with a body that says what to send instead.

    The whole reason this contract is enforced rather than collapsed is that the
    old failure was an empty 200, which a client cannot debug. A rejection that
    does not explain itself would repeat that mistake in a different shape.
    """
    return _B2CJSONResponse(
        status_code=400,
        content={
            "error": "customer_id_not_accepted_on_b2c",
            "message": str(exc),
            "detail": {
                "code": "customer_id_not_accepted_on_b2c",
                "message": str(exc),
                "where": exc.where,
                "fix": (
                    "This instance is B2C (user_context_isolation=equals_customer). "
                    "Send user_id only and omit customer_id entirely; the user id is "
                    "the whole identity. Customer-scoped endpoints are not available "
                    "here at all: use POST /v1/context/user/fetch instead of "
                    "/v1/context/customer/fetch. GET /api/v1/auth/whoami reports the "
                    "mode as user_context_isolation."
                ),
            },
        },
    )


app.add_exception_handler(_B2CRejected, _b2c_customer_id_to_http)

# Readiness gate: a runtime write or fetch against an instance whose memory
# architecture is still being built is refused with a retryable 409, instead of
# being processed with the default config. See lifecycle/instance_readiness.py.
# The hook adds the build gate's two 409s: instance_not_ready (build running)
# and instance_build_failed (all attempts used). Core has no gate.
edition.call("instance_readiness_handlers", app)

# --- CORS Configuration ---
# Parse allowed origins from environment variable CORS_ORIGINS (comma-separated list).
# Deployments must set CORS_ORIGINS to the browser origins they serve (the dashboard and any
# admin site). The built-in default names only local development servers.
_DEFAULT_CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_origins_env:
    allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = _DEFAULT_CORS_ORIGINS
        logger.warning(
            "CORS_ORIGINS is set but parsed to empty list; using defaults: %s",
            allowed_origins,
        )
else:
    allowed_origins = _DEFAULT_CORS_ORIGINS
    logger.warning(
        "CORS_ORIGINS not set; using development defaults: %s. "
        "Production deployments must set CORS_ORIGINS environment variable.",
        allowed_origins,
    )

logger.info("CORS allowed origins: %s", allowed_origins)

# Add CORS middleware (runs first to handle preflight and add headers to all responses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Explicit origins required when credentials are used
    allow_credentials=True,  # Allow cookies and Authorization headers from frontend
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=["Authorization", "Content-Type", "Accept", "Cache-Control", "X-Requested-With", "X-Correlation-ID", "X-Client-ID", "X-Instance-ID"],  # Standard + SDK headers
    expose_headers=[
        "X-Correlation-ID",
        "X-Credits-Remaining",       # Credit system: post-response balance snapshot
        "X-Credits-Enforcement",     # Credit system: which enforcement mode applied
    ],  # Expose correlation ID for client-side tracking
    max_age=600,  # Cache preflight responses for 10 minutes
)

# Add logging middleware (after CORS to log all requests including CORS)
edition.call("request_logging_middleware", app)
