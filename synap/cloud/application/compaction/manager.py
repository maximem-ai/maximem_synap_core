"""
CompactionManager: runs compaction v3 for one conversation.

    stored record D + the turns after its boundary
      -> integrity check (sequence hash); a changed history is rebuilt
      -> the v3 writer (v3/engine.py): one bounded call per chunk, code
         checks, one repair, merge, roll-up at the ceiling
      -> one framed context_block for every caller
      -> store, then the post-compaction callbacks

Triggers and size limits are model-adaptive (budgets.py). The design and the
issues each part answers are in COMPACTION_V3_PLAN.md at the repo root.

The manager is stateless and can be called by:
- the conversation tracker (inline trigger)
- the background worker (idle)
- the SDK API (explicit compaction request)
"""
import asyncio
import hashlib
import sys
import time
import logging
from typing import Awaitable, Callable, Dict, Any, Optional, List, Sequence
from datetime import datetime, timedelta, timezone

from synap.cloud.infrastructure.llm import LLMConfigManager
from synap.cloud.shared.tokens import count_tokens

from .budgets import (
    DEFAULT_AGENT_WINDOW,
    WINDOW_ENV,
    configured_window,
    AgentBudget,
    CompactionModelLimits,
    agent_budget,
    derive_model_limits,
    resolve_agent_window,
)
from .models import (
    CompactionConfig,
    CompactedContext,
    CompactionEvent,
    CompactionMetadata,
    ClassificationResult,
    ConversationMessage,
    ExtractionResult,
    NarrativeSummary,
    QualityAssurance,
    CompactionStrategy,
)
from .templates import get_template_loader
from .passes.base import resolve_compaction_profile
from synap.cloud.shared import edition
from .storage import CompactionStorage
from .interfaces import (
    ICompactionManager,
    CompactionTriggerType
)
from .v3 import (
    CompactionRecord,
    CompactionWriter,
    EngineSettings,
    Turn,
    UpdateResult,
    build_guidance,
    label_turns,
    render_context_block,
    render_document,
)


logger = logging.getLogger(__name__)

# New turns below this many tokens are not compacted (should_compact).
MIN_SPAN_TOKENS = 2_000


def manager_min_span() -> int:
    return MIN_SPAN_TOKENS

# Message metadata keys an SDK or integration sets to say which model the
# agent runs on. The latest turn that carries one wins, so a
# model switch mid-conversation takes effect on the next trigger check.
AGENT_MODEL_KEY = "agent_model"
AGENT_CONTEXT_WINDOW_KEY = "agent_context_window"


class CompactionError(Exception):
    """Error during compaction."""
    pass


def order_messages(messages: Sequence[ConversationMessage]) -> List[ConversationMessage]:
    """Messages in conversation order: timestamp, then message_id.

    Ordering by timestamp alone lets two turns with the same timestamp swap
    between loads, which breaks the sequence hash and the turn numbering
    (I-24).
    """
    def key(m: ConversationMessage):
        ts = m.timestamp
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (ts if isinstance(ts, datetime) else datetime.min.replace(tzinfo=timezone.utc), m.message_id)

    return sorted(messages, key=key)


def agent_model_from(messages: Sequence[ConversationMessage]) -> tuple[Optional[str], Optional[int]]:
    """(model, context_window) from the latest turn that reports either."""
    for message in reversed(messages):
        meta = message.metadata or {}
        model, window = meta.get(AGENT_MODEL_KEY), meta.get(AGENT_CONTEXT_WINDOW_KEY)
        if model or window:
            try:
                window = int(window) if window else None
            except (TypeError, ValueError):
                window = None
            return (str(model) if model else None), window
    return None, None


class CompactionManager(ICompactionManager):
    """
    Orchestrates compaction.

    Usage:
        manager = CompactionManager(llm_manager)
        result = await manager.compact(
            conversation_id="conv_123",
            instance_id="inst_456",
            messages=messages
        )
    """

    def __init__(
        self,
        llm_manager: LLMConfigManager,
        storage: Optional[CompactionStorage] = None
    ):
        self._llm_manager = llm_manager
        self.storage = storage or CompactionStorage()
        self.template_loader = get_template_loader()
        self._post_compaction_callbacks: List[Callable[[CompactionEvent], Awaitable[None]]] = []
        self._usage_tracker = None  # lazy-loaded
        self._client_id_cache: Dict[str, Optional[str]] = {}  # instance_id → client_id

        # One profile for every call: which model writes, and whether the Jev
        # completeness check feeds the repair. Groq (architecture decision D9)
        # unless this process has no Groq key or SYNAP_COMPACTION_PROFILE says
        # otherwise. Its model is also the metering label for compaction
        # tokens: as a second literal that label once drifted to a
        # decommissioned model and priced compaction at the wrong rate.
        self._profile = resolve_compaction_profile(self._llm_manager)
        logger.info(
            "Compaction profile: %s (%s:%s, jev_check=%s)",
            self._profile.name, self._profile.provider.value,
            self._profile.model, self._profile.jev_check,
        )

        # K (chunk size) and G (global record ceiling) come from OUR model's
        # window, so every writer call stays inside 25% of it.
        # Logged so that replacing the model shows up; pinned by a test so it
        # shows up in review too.
        try:
            written = configured_window()
        except ValueError as exc:
            logger.warning("%s Ignored; the model table decides.", exc)
            written = None
        try:
            self._limits: CompactionModelLimits = derive_model_limits(self._profile.model, window=written)
        except KeyError:
            self._limits = derive_model_limits(self._profile.model, window=DEFAULT_AGENT_WINDOW)
            logger.warning(
                "Compaction model %s is not in the model window table, so its calls are sized for %d tokens. "
                "If your model's window is different, set %s to its size in tokens.",
                self._profile.model, DEFAULT_AGENT_WINDOW, WINDOW_ENV,
            )
        logger.info(self._limits.describe())
        self._writer = CompactionWriter(self._llm_manager, self._profile)

    @property
    def limits(self) -> CompactionModelLimits:
        return self._limits

    def budget_for(self, messages: Sequence[ConversationMessage],
                   context_window: Optional[int] = None, model: Optional[str] = None) -> AgentBudget:
        """T and C for this conversation's agent model."""
        reported_model, reported_window = agent_model_from(messages)
        window = resolve_agent_window(
            context_window=context_window or reported_window,
            model=model or reported_model,
        )
        return agent_budget(window, self._limits.record_ceiling)

    def on_compaction_complete(
        self,
        callback: Callable[[CompactionEvent], Awaitable[None]]
    ):
        """Register a post-compaction callback.

        Callbacks are invoked after successful compaction and storage.
        Failures are logged but never fail the compaction itself.
        """
        self._post_compaction_callbacks.append(callback)

    async def compact(
        self,
        conversation_id: str,
        instance_id: str,
        messages: List[ConversationMessage],
        config: Optional[CompactionConfig] = None,
        trigger_type: CompactionTriggerType = CompactionTriggerType.MANUAL_API,
        force: bool = False,
        strategy: Optional[str] = None,
        target_tokens: Optional[int] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope_path: Optional[Dict[str, str]] = None,
        context_window: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Optional[CompactedContext]:
        """
        Compact the conversation: update its stored record with the turns
        after the record's boundary.

        Args:
            conversation_id: ID of the conversation to compact
            instance_id: Instance ID for config lookup
            messages: The conversation's turns at the moment compaction was
                requested. Every turn after the record's boundary is folded
                in; nothing is held back as a "recent" carve-out.
            config: Optional config override (uses MACA config if not provided)
            trigger_type: What triggered this compaction
            force: Compact even if under threshold
            strategy: Recorded on the result; does not change the writer
                (MACA ``strategy``, COMPACTION_V3_PLAN.md §15).
            target_tokens: Ignored. Retired in v3: size follows
                ``target_ratio`` (MACA) and the model-adaptive ceiling.
            context_window / model: The agent's model, when the caller knows
                it. Otherwise the latest turn's metadata, then 128K.

        Returns:
            CompactedContext. With no new turns since the last compaction,
            the stored one, unchanged. None when the compaction is not due
            (and not forced) and nothing is stored yet.
        """
        start_time = time.time()

        if config is None:
            config = await self._load_config(instance_id)

        if strategy is not None:
            try:
                config.strategy = CompactionStrategy(strategy)
            except ValueError:
                logger.warning(f"Invalid strategy '{strategy}', keeping MACA default: {config.strategy.value}")
        if target_tokens is not None:
            logger.debug("target_tokens=%s ignored: retired in compaction v3", target_tokens)

        messages = order_messages(messages)
        budget = self.budget_for(messages, context_window=context_window, model=model)

        existing = await self.storage.get_latest(conversation_id)
        if not force:
            should, detected_trigger = self.should_compact(messages, config, last_compaction=existing, budget=budget)
            if not should:
                if existing:
                    logger.info(f"Compaction not needed for {conversation_id}, returning existing")
                    return existing
                # Not due, and nothing stored yet. This used to fall through and
                # compact anyway, at any size: the first read of a conversation
                # compacted the user's first message alone (B1). A first
                # compaction that has to happen is forced (session end,
                # transcript push, an explicit force=True).
                logger.info(f"Compaction not needed for {conversation_id}, and none stored yet")
                return None
            else:
                trigger_type = detected_trigger

        # Integrity: the record carries on only if the turns it covers are
        # exactly the turns it covered. An edit, a deletion or a backfill
        # rebuilds from the full history, so deleted content cannot survive
        # in the record.
        record, covered = self._continuable_record(conversation_id, existing, messages)
        new_messages = messages[covered:]
        if record is not None and not new_messages:
            logger.info(f"No new turns for {conversation_id} since its last compaction")
            return existing  # type: ignore[return-value]
        if not new_messages:
            raise CompactionError(f"Nothing to compact for {conversation_id}: no messages")
        if force and record is not None and sum(m.token_count for m in new_messages) < manager_min_span():
            # A developer-triggered compaction follows the same rule: too few
            # new tokens to be worth folding, so the stored record stands.
            logger.info(f"Forced compaction of {conversation_id} skipped: under {MIN_SPAN_TOKENS} new tokens")
            return existing  # type: ignore[return-value]

        logger.info(
            f"Starting compaction for {conversation_id} (trigger: {trigger_type.value}, "
            f"{'update' if record else 'rebuild'}, {len(new_messages)} new turns, "
            f"window {budget.window}, T={budget.trigger_tokens}, C={budget.record_ceiling})"
        )

        settings = EngineSettings(
            target_ratio=config.target_ratio,
            record_ceiling=budget.record_ceiling,
            chunk_tokens=self._limits.chunk_tokens,
            guidance=build_guidance(config),
            density_rates=dict(config.template.compression_rates) if config.template else
            EngineSettings().density_rates,
            steering_enabled=config.steering_enabled,
        )
        known_turns = {t.label: t.content for t in label_turns(messages[:covered])}
        new_turns = label_turns(new_messages, first_position=covered + 1)

        try:
            result = await self._writer.update(
                record, new_turns, settings, known_turns=known_turns,
                find_omissions=self._omission_finder() if self._profile.jev_check else None,
            )
        except Exception as e:
            # Never store a broken compaction: the previous record stays.
            logger.error(f"Compaction failed for {conversation_id}: {e}", exc_info=True)
            raise CompactionError(f"Compaction failed: {str(e)}") from e

        compacted_context = self._build_context(
            result=result,
            conversation_id=conversation_id,
            instance_id=instance_id,
            client_id=client_id or "",
            messages=messages,
            previous_count=covered,
            config=config,
            trigger_type=trigger_type,
            start_time=start_time,
        )

        await self.storage.save(compacted_context)

        if self._post_compaction_callbacks:
            event = CompactionEvent(
                conversation_id=conversation_id,
                instance_id=instance_id,
                compaction_id=compacted_context.metadata.compaction_id,
                compression_ratio=compacted_context.metadata.compression_ratio,
                validation_score=compacted_context.quality_assurance.validation_score,
                is_incremental=compacted_context.metadata.source_message_range.get("incremental", False),
                message_count=len(messages),
                user_id=user_id,
                customer_id=customer_id,
                client_id=client_id,
                scope_path=scope_path,
                trigger_type=trigger_type,
            )
            for cb in self._post_compaction_callbacks:
                try:
                    await cb(event)
                except Exception as cb_err:
                    logger.warning(f"Post-compaction callback failed: {cb_err}")

        total_latency = (time.time() - start_time) * 1000
        logger.info(
            f"Compaction completed for {conversation_id}: {result.calls} calls "
            f"({result.chunks} chunks, {result.repairs} repairs, roll-up={result.rollup}), "
            f"{compacted_context.metadata.compacted_token_count} tokens for "
            f"{compacted_context.metadata.original_token_count} covered, {total_latency:.0f}ms"
        )
        if result.dropped:
            logger.info(f"Compaction for {conversation_id} dropped after repair: {result.dropped}")

        # Metering identity. Auto-compaction (conversation.tracker) and the
        # SDK compaction route call in without a client_id — the
        # conversation table has no such column — so fall back to the
        # instance's owner. Without this those compactions run untracked,
        # which is most of them.
        metering_client_id = client_id or await self._resolve_client_id(instance_id)
        if metering_client_id and instance_id and (result.input_tokens or result.output_tokens):
            self._track_compaction_usage(
                client_id=metering_client_id,
                instance_id=instance_id,
                user_id=user_id,
                conversation_id=conversation_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        # Count the compaction itself. Deliberately not gated on token
        # count: a compaction that consumed no LLM tokens still happened,
        # and "how often are conversations being compacted" is the question
        # this metric answers.
        if metering_client_id and instance_id:
            self._track_compaction_event(
                client_id=metering_client_id,
                instance_id=instance_id,
                user_id=user_id,
                conversation_id=conversation_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=int(total_latency),
            )

        return compacted_context

    def _continuable_record(
        self,
        conversation_id: str,
        existing: Optional[CompactedContext],
        messages: List[ConversationMessage],
    ) -> tuple[Optional[CompactionRecord], int]:
        """(record, turns it covers), or (None, 0) to rebuild from scratch."""
        if existing is None:
            return None, 0
        record = CompactionRecord.from_dict(existing.record)
        if record is None:
            logger.info(f"Compaction for {conversation_id} predates v3; rebuilding from the full history")
            return None, 0
        smr = existing.metadata.source_message_range or {}
        count = int(smr.get("message_count") or 0)
        if not (0 < count <= len(messages)) or messages[count - 1].message_id != smr.get("end_id"):
            logger.warning(
                f"Conversation continuity broken for {conversation_id}; rebuilding from the full history"
            )
            return None, 0
        stored_hash = smr.get("sequence_hash", "")
        if stored_hash and self._compute_sequence_hash(messages[:count]) != stored_hash:
            logger.warning(
                f"Sequence hash mismatch for {conversation_id}: the covered turns changed; "
                f"rebuilding from the full history"
            )
            return None, 0
        return record, count

    def _omission_finder(self):
        """The Jev completeness check, for profiles that already run it (openai_jev)."""
        async def find(turns: Sequence[Turn], document: str) -> List[str]:
            messages = [
                ConversationMessage(message_id=t.label, role=t.role, content=t.content,
                                    timestamp=datetime.now(timezone.utc))
                for t in turns
            ]
            # Looked up on the module (see __getattr__ below), so a test can replace it.
            report = await sys.modules[__name__].find_omissions(messages, document)
            return list(report.hints)
        return find

    def _build_context(
        self,
        result: UpdateResult,
        conversation_id: str,
        instance_id: str,
        client_id: str,
        messages: List[ConversationMessage],
        previous_count: int,
        config: CompactionConfig,
        trigger_type: CompactionTriggerType,
        start_time: float,
    ) -> CompactedContext:
        """The stored envelope around a v3 record.

        Transition: ``narrative_summary.overview`` carries the
        framed block, so an SDK released before v3 prints it once, and the
        two paragraphs are sent empty. The v2 extraction lists, the stub
        current_state and the "last N turns" copy are gone.
        """
        record = result.record
        document = render_document(record)
        block = render_context_block(record, document)
        covered = messages[: int(record.covers.get("message_count") or len(messages))]
        original = sum(m.token_count for m in covered)
        compacted = count_tokens(block)

        metadata = CompactionMetadata(
            conversation_id=conversation_id,
            instance_id=instance_id,
            client_id=client_id,
            source_message_range={
                "start_id": covered[0].message_id if covered else "",
                "end_id": covered[-1].message_id if covered else "",
                "start_timestamp": _iso(covered[0].timestamp) if covered else "",
                "end_timestamp": _iso(covered[-1].timestamp) if covered else "",
                "message_count": len(covered),
                "incremental": previous_count > 0,
                "new_messages_processed": len(covered) - previous_count,
                "sequence_hash": self._compute_sequence_hash(covered),
            },
            original_token_count=original,
            compacted_token_count=compacted,
            compression_ratio=compacted / original if original else 0.0,
            messages_preserved_verbatim=0,
            messages_compressed=len(covered),
            trigger_type=trigger_type,
            strategy_used=config.strategy,
            total_latency_ms=(time.time() - start_time) * 1000,
            pass_latencies={"writer": result.latency_ms},
            groq_tokens_consumed=result.input_tokens + result.output_tokens,
            groq_api_calls=result.calls,
        )
        classification = ClassificationResult(
            primary_category=record.use_case,
            subcategory="general",
            objective="",
            characteristics={},
            recommended_templates=[],
            special_handling=[],
            compression_strategy=config.strategy.value,
            section_plan={"factual_sections": [s.header for s in record.sections], "conversational_sections": []},
        )
        narrative = NarrativeSummary(
            overview=block,
            chronological_narrative=[],
            current_state={},
            factual_paragraph="",
            conversational_paragraph="",
        )
        clean = not result.problems_after_repair
        warnings = [{"category": "dropped_after_repair", "item": d} for d in result.dropped]
        warnings += [{"category": "added_by_code", "item": v} for v in result.auto_literals]
        quality = QualityAssurance(
            validation_passed=clean,
            validation_score=1.0 if clean else 0.5,
            information_completeness=1.0 if clean else 0.5,
            quality_warning=bool(result.dropped),
            warning_message=(f"{len(result.dropped)} item(s) failed the quote check twice and were dropped"
                             if result.dropped else None),
            flags=[f for f, on in (("repaired", result.repairs > 0), ("rolled_up", result.rollup),
                                   (f"chunks_{result.chunks}", result.chunks > 1)) if on],
            warnings=warnings,
        )
        return CompactedContext(
            metadata=metadata,
            conversation_classification=classification,
            narrative_summary=narrative,
            structured_extractions=_empty_extractions(),
            reference_index={},
            verbatim_recent_messages=[],
            quality_assurance=quality,
            density_scores=None,
            record=record.to_dict(),
            context_block=block,
            document=document,
        )

    def _track_compaction_usage(
        self,
        client_id: str,
        instance_id: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Fire-and-forget compaction token attribution to TimescaleDB."""
        if self._usage_tracker is None:
            # Cloud's hook returns get_usage_tracker(); Core has none and returns None.
            self._usage_tracker = edition.call("usage_tracking", "tracker")
        if self._usage_tracker is None:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self._usage_tracker.track_llm_usage(
                    client_id=client_id,
                    instance_id=instance_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._profile.model,
                    operation="compaction",
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
        except Exception as e:
            logger.debug("Failed to schedule compaction usage tracking: %s", e)

    async def _resolve_client_id(self, instance_id: str) -> Optional[str]:
        """Best-effort ``instance_id`` → owning ``client_id``, for metering only.

        Memoised for the process lifetime: an instance never changes owner, and
        this sits on the tail of every compaction. Returns ``None`` rather than
        raising — failing to attribute a compaction must not fail it.
        """
        if not instance_id:
            return None
        if instance_id in self._client_id_cache:
            return self._client_id_cache[instance_id]
        try:
            from sqlalchemy import text as _sql_text

            from synap.cloud.database.connection import get_session_factory

            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    _sql_text(
                        "SELECT client_id FROM instance WHERE instance_id = :iid LIMIT 1"
                    ),
                    {"iid": instance_id},
                )
                row = result.fetchone()
        except Exception as exc:
            logger.debug("Could not resolve client_id for %s: %s", instance_id, exc)
            return None

        client_id = row.client_id if row is not None else None
        self._client_id_cache[instance_id] = client_id
        return client_id

    def _track_compaction_event(
        self,
        client_id: str,
        instance_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Fire-and-forget ``context_compact`` count event.

        Cost stays off this row and credits are not burned: the sibling
        :meth:`_track_compaction_usage` call already attributes this pipeline's
        tokens (and their cost), and compaction is not a billed operation today.
        See ``UsageTracker.track_compaction`` for why both are opt-in.
        """
        if self._usage_tracker is None:
            # Cloud's hook returns get_usage_tracker(); Core has none and returns None.
            self._usage_tracker = edition.call("usage_tracking", "tracker")
        if self._usage_tracker is None:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self._usage_tracker.track_compaction(
                    client_id=client_id,
                    instance_id=instance_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
        except Exception as e:
            logger.debug("Failed to schedule compaction event tracking: %s", e)

    async def get_or_create(
        self,
        conversation_id: str,
        instance_id: str,
        messages: List[ConversationMessage],
        client_id: str = "",
        user_id: str = "",
        customer_id: str = "",
    ) -> Optional[CompactedContext]:
        """
        Get existing compacted context or create new one if stale/missing.

        Reads no longer call this (retrieval_manager._get_compacted_context
        never starts a compaction). None when the compaction is not due and
        nothing is stored yet.
        """
        # Check Redis cache first
        cached = await self.storage.get_from_cache(conversation_id)

        if cached and not self._is_stale(cached, messages, await self._load_config(instance_id)):
            logger.debug(f"Returning cached compacted context for {conversation_id}")
            return cached

        # Atomically acquire compaction lock
        lock_acquired = await self.storage.set_compaction_in_progress(conversation_id)
        if not lock_acquired:
            # Compaction already in progress — return stale data if available
            if cached:
                logger.info(f"Compaction in progress for {conversation_id}, returning stale data")
                return cached

            logger.warning(f"Compaction in progress with no cached data for {conversation_id}")
            raise CompactionError("Compaction in progress, no cached data available")

        try:
            # Trigger compaction
            result = await self.compact(
                conversation_id=conversation_id,
                instance_id=instance_id,
                messages=messages,
                trigger_type=CompactionTriggerType.BACKGROUND_WORKER,
                client_id=client_id,
                user_id=user_id,
                customer_id=customer_id,
            )
            return result
        finally:
            await self.storage.clear_compaction_in_progress(conversation_id)

    def should_compact(
        self,
        messages: List[ConversationMessage],
        config: CompactionConfig,
        last_compaction: Optional[CompactedContext] = None,
        budget: Optional[AgentBudget] = None,
    ) -> tuple[bool, CompactionTriggerType]:
        """Is it time to compact?

        Trigger tokens = the stored compaction's tokens + the raw turns after
        its boundary: exactly what is injected into the agent's prompt. The
        token trigger T and the message trigger are whichever comes first;
        the idle trigger is per agent through MACA. Every count uses the one
        tiktoken counter.

        Returns:
            Tuple of (should_compact, trigger_type)
        """
        budget = budget or self.budget_for(messages)
        message_trigger = config.message_trigger or budget.message_trigger

        if last_compaction is None:
            tail = messages
            injected = 0
        else:
            covered = int((last_compaction.metadata.source_message_range or {}).get("message_count") or 0)
            tail = messages[covered:] if covered <= len(messages) else messages
            if not tail:
                return False, CompactionTriggerType.MANUAL_API
            injected = count_tokens(last_compaction.context_block or last_compaction.narrative_summary.overview or "")

        trigger_tokens = injected + sum(m.token_count for m in tail)
        logger.info(
            "compaction_trigger_check",
            extra={
                "event": "compaction_trigger_check",
                "trigger_tokens": trigger_tokens,
                "token_threshold": budget.trigger_tokens,
                "raw_turns": len(tail),
                "message_threshold": message_trigger,
                "window": budget.window,
            },
        )
        if trigger_tokens >= budget.trigger_tokens:
            return True, CompactionTriggerType.TOKEN_THRESHOLD
        # Below MIN_SPAN_TOKENS of new turns a compaction saves nothing: the
        # record of a span that small is as large as the span (measured 19 Oct:
        # 973 tokens for ~1,000-token conversations), so the raw turns stay.
        if sum(m.token_count for m in tail) < manager_min_span():
            return False, CompactionTriggerType.MANUAL_API
        if len(tail) >= message_trigger:
            return True, CompactionTriggerType.MESSAGE_COUNT

        if last_compaction is not None:
            # Idle long enough since the last compaction to fold in what
            # arrived. Per-agent via MACA: a voice agent turns this down so
            # its memory is not left unwritten for most of a short call.
            last_time = last_compaction.metadata.created_at
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_time > timedelta(minutes=config.recompaction_idle_minutes):
                return True, CompactionTriggerType.IDLE_PERIOD

        return False, CompactionTriggerType.MANUAL_API

    async def _load_config(self, instance_id: str) -> CompactionConfig:
        """Load MACA config for instance and merge with template."""
        # Load MACA config from config store
        maca_config = await self._fetch_maca_config(instance_id)

        # Determine template based on MACA
        # Only configs generated before templates were removed carry a
        # ``metadata.template_base``; anything newer falls through to "base".
        instance_category = (
            maca_config.get("metadata", {}).get("template_base") or "base"
        )
        template = self.template_loader.load_template(instance_category)

        # Merge template with MACA customizations
        return self.template_loader.merge_with_maca(template, maca_config)

    async def _fetch_maca_config(self, instance_id: str) -> Dict[str, Any]:
        """
        Fetch approved MACA config from Postgres for this instance.

        Falls back to sensible defaults if no approved config exists,
        so compaction still works for instances that haven't run MACA yet.
        """
        # Core: the one mounted file, read as the developer wrote it.
        from .instance_config import file_config

        file_content = await file_config(instance_id)
        if file_content is not None:
            file_content.setdefault("metadata", {})["instance_id"] = instance_id
            return file_content

        try:
            from synap.cloud.database.connection import get_session_factory
            from synap.cloud.database.repositories.maca_config_repo import MACAConfigRepository

            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = MACAConfigRepository(session)
                config_orm = await repo.get_active_for_instance(instance_id)

                if config_orm and config_orm.config_content:
                    logger.info(
                        f"Loaded approved MACA config for instance {instance_id} "
                        f"(version={config_orm.version})"
                    )
                    maca_config = config_orm.config_content

                    # Ensure metadata section exists with instance_id
                    if "metadata" not in maca_config:
                        maca_config["metadata"] = {}
                    maca_config["metadata"]["instance_id"] = instance_id

                    return maca_config

                logger.warning(
                    f"No approved MACA config for instance {instance_id}, "
                    f"using defaults"
                )
        except Exception as e:
            logger.error(
                f"Failed to fetch MACA config for {instance_id}: {e}. "
                f"Falling back to defaults.",
                exc_info=True,
            )

        # Fallback defaults — keeps compaction working for unconfigured instances
        return {
            "metadata": {"instance_id": instance_id},
            "compaction_config": {
                "strategy": "adaptive",
                "adaptive_thresholds": {"target_ratio": 0.30},
                "preservation_priorities": [
                    "user_stated_facts",
                    "explicit_preferences",
                    "current_objective"
                ]
            }
        }

    def _is_stale(
        self,
        cached: CompactedContext,
        messages: List[ConversationMessage],
        config: CompactionConfig,
    ) -> bool:
        """Is the cached compaction due for an update? The same rule as the
        trigger, with the instance's own config, so retrieval's inline fallback
        and the tracker agree (§4.5)."""
        if not cached.record:
            return True  # predates v3: rebuild
        due, _ = self.should_compact(order_messages(messages), config, last_compaction=cached)
        return due

    @staticmethod
    def _compute_sequence_hash(messages: List[ConversationMessage]) -> str:
        """Compute MD5 hash of message ID sequence for integrity checking."""
        sequence = "|".join(m.message_id for m in messages)
        return hashlib.md5(sequence.encode()).hexdigest()



def _iso(ts) -> str:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts or "")


def _empty_extractions() -> ExtractionResult:
    """The v2 extraction shell. v3 fills none of it; readers get empty lists."""
    return ExtractionResult(
        extraction_metadata={"compaction_version": 3},
        explicit_facts=[], decisions=[], constraints=[], attempted_solutions=[],
        user_preferences=[], code_artifacts=[], open_questions=[], dependencies=[],
        conflicts_detected=[], reference_pointers=[], emotional_states=[], custom_extractions={},
    )


# Singleton instance
_manager: Optional[CompactionManager] = None
_llm_manager: Optional[LLMConfigManager] = None


def get_compaction_manager(
    llm_manager: Optional[LLMConfigManager] = None
) -> CompactionManager:
    """
    Get or create singleton CompactionManager.
    
    Args:
        llm_manager: Optional LLM manager. If not provided, creates one.
        
    Returns:
        Singleton CompactionManager instance
    """
    global _manager, _llm_manager
    if _manager is None:
        if llm_manager is None:
            from synap.cloud.infrastructure.llm import get_llm_manager
            llm_manager = get_llm_manager()
        _llm_manager = llm_manager
        _manager = CompactionManager(llm_manager=_llm_manager)
    return _manager


def __getattr__(name: str):
    """``find_omissions`` is the Jev completeness check, which Core does not ship.

    Cloud gets the same function under the same module attribute; the import
    happens on first use instead of when this file loads.
    """
    if name == "find_omissions":
        from .completeness_check import find_omissions

        globals()["find_omissions"] = find_omissions
        return find_omissions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
