"""
Compaction Worker: Background job for proactive compaction.

Runs periodically to keep compacted context fresh for active conversations.
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from synap.cloud.infrastructure.llm import LLMConfigManager

from .manager import get_compaction_manager, CompactionManager
from .storage import CompactionStorage
from .interfaces import CompactionTriggerType


logger = logging.getLogger(__name__)


class CompactionWorker:
    """
    Background worker for idle-sweep compaction.

    Token and message-count triggers are handled inline by
    ConversationTracker._maybe_trigger_compaction().  This worker only
    catches conversations that went idle with un-compacted messages.
    """

    _THRESHOLD_CACHE_TTL_SECONDS = 300
    # The regular sweep looks at conversations touched inside this window.
    _ACTIVE_WINDOW = timedelta(minutes=30)

    def __init__(
        self,
        llm_manager: Optional[LLMConfigManager] = None,
        check_interval_seconds: int = 60,
        idle_threshold_minutes: int = 5,
        max_concurrent_compactions: int = 50,
        backlog_page_size: int = 50,
        backlog_pause_seconds: float = 2.0,
    ):
        """
        Initialize worker.

        Args:
            llm_manager: Optional LLM manager for compaction. If not provided, uses singleton.
            check_interval_seconds: How often to check for compaction needs
            idle_threshold_minutes: How long until a conversation is considered idle
            max_concurrent_compactions: Maximum number of concurrent compaction tasks
            backlog_page_size: Conversations handled per page by the start-up backlog sweep
            backlog_pause_seconds: Pause between two pages of the backlog sweep
        """
        self._llm_manager = llm_manager
        self.check_interval = check_interval_seconds
        self.idle_threshold = timedelta(minutes=idle_threshold_minutes)
        self.manager: Optional[CompactionManager] = None
        self.storage: Optional[CompactionStorage] = None
        self._running = False
        # instance_id -> (cached_at_monotonic, (token, message, incremental_delta, idle_minutes))
        # TTL'd for the same reason as ConversationTracker's: a regenerated MACA
        # config must reach a long-running worker without a deploy.
        self._threshold_cache: Dict[str, tuple[float, tuple[int, int, int, int]]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_compactions)
        self._active_tasks: set = set()
        self._backlog_page_size = max(1, backlog_page_size)
        self._backlog_pause = backlog_pause_seconds
        self._backlog_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background worker."""
        self._running = True
        try:
            self.manager = get_compaction_manager(llm_manager=self._llm_manager)
        except Exception:
            # Compaction without a usable model (Core with no configured LLM) must be
            # seen, not left as an unread failure in a background task.
            logger.critical("Compaction worker cannot start: no usable compaction model", exc_info=True)
            self._running = False
            raise
        self.storage = self.manager.storage  # Reuse singleton's storage

        logger.info("Compaction worker started")

        # Conversations that went idle before this process started are outside the
        # regular sweep's window. Flush them in the background, a page at a time.
        self._backlog_task = asyncio.create_task(self._sweep_backlog())

        while self._running:
            try:
                await self._check_and_compact()
            except Exception as e:
                logger.error(f"Compaction worker error: {e}", exc_info=True)

            await asyncio.sleep(self.check_interval)

    async def stop(self):
        """Stop the background worker gracefully."""
        self._running = False
        if self._backlog_task is not None:
            self._backlog_task.cancel()
            await asyncio.gather(self._backlog_task, return_exceptions=True)
            self._backlog_task = None
        # Cancel active tasks on shutdown
        for task in self._active_tasks:
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
        logger.info("Compaction worker stopped")

    async def _sweep_backlog(self):
        """Once, at start: flush idle conversations older than the regular sweep's window.

        Each page of ``backlog_page_size`` conversations goes through the same check as
        the regular sweep (promote what is unsent, compact what is newer than its last
        compaction) and is finished before the next page is read, with a pause between
        pages. Never raises.
        """
        before = datetime.now(timezone.utc) - self._ACTIVE_WINDOW
        after = None
        handled = 0
        try:
            while self._running:
                page = await self.storage.get_unflushed_idle_conversations(
                    before, after, self._backlog_page_size
                )
                if not page:
                    break
                tasks = await self._check_and_compact(page)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                handled += len(page)
                last = page[-1]
                after = (last["last_message_at"], last["conversation_id"])
                if len(page) < self._backlog_page_size:
                    break
                await asyncio.sleep(self._backlog_pause)
            logger.info(f"Compaction backlog sweep finished: {handled} idle conversations checked")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Compaction backlog sweep stopped after {handled} conversations: {e}", exc_info=True)

    async def _check_and_compact(self, conversations: Optional[List[Dict[str, Any]]] = None):
        """Check active conversations and trigger compaction if needed.

        ``conversations`` defaults to the recently active ones. Returns the compaction
        tasks it started.
        """
        active_conversations = (
            conversations if conversations is not None else await self._get_active_conversations()
        )
        started: List[asyncio.Task] = []

        logger.debug(f"Checking {len(active_conversations)} active conversations for compaction")

        for conv in active_conversations:
            try:
                should_compact, trigger = await self._should_compact_conversation(conv)

                if should_compact:
                    # Atomically acquire lock (SET NX)
                    if not await self.storage.set_compaction_in_progress(conv["conversation_id"]):
                        logger.debug(
                            f"Compaction lock not acquired for {conv['conversation_id']}, skipping"
                        )
                        continue

                    logger.info(
                        f"Triggering compaction for {conv['conversation_id']} "
                        f"(trigger: {trigger.value})"
                    )

                    # Load messages
                    messages = await self._load_messages(conv["conversation_id"])

                    if not messages:
                        await self.storage.clear_compaction_in_progress(conv["conversation_id"])
                        logger.warning(f"No messages found for {conv['conversation_id']}, skipping")
                        continue

                    # Trigger bounded async compaction
                    task = asyncio.create_task(
                        self._bounded_compact(
                            conversation_id=conv["conversation_id"],
                            instance_id=conv["instance_id"],
                            client_id=conv.get("client_id", ""),
                            messages=messages,
                            trigger=trigger,
                            user_id=conv.get("user_id"),
                            customer_id=conv.get("customer_id"),
                            scope_path=conv.get("scope_path"),
                        )
                    )
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)
                    started.append(task)

            except Exception as e:
                logger.error(
                    f"Error checking conversation {conv.get('conversation_id')}: {e}",
                    exc_info=True
                )
        return started

    async def _get_active_conversations(self) -> List[Dict[str, Any]]:
        """
        Get list of active conversations that might need compaction.

        An active conversation is one with activity in the last 30 minutes.
        """
        try:
            return await self.storage.get_active_conversations(
                since=datetime.now(timezone.utc) - self._ACTIVE_WINDOW
            )
        except Exception as e:
            logger.error(f"Failed to get active conversations: {e}")
            return []

    async def _should_compact_conversation(
        self,
        conv: Dict[str, Any]
    ) -> tuple[bool, CompactionTriggerType]:
        """Determine if an idle conversation should be compacted.

        Token and message-count triggers are handled inline by
        ConversationTracker._maybe_trigger_compaction().  This method
        only checks for the IDLE trigger: conversation inactive for
        5+ minutes with un-compacted messages.
        """
        last_activity = conv.get("last_message_at")
        if not last_activity:
            return False, CompactionTriggerType.MANUAL_API

        if isinstance(last_activity, str):
            try:
                last_activity = datetime.fromisoformat(last_activity)
            except (ValueError, TypeError):
                return False, CompactionTriggerType.MANUAL_API

        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)

        # Per-agent, not one number for the whole deployment. The worker's own
        # idle_threshold stays as the fallback for a conversation with no MACA
        # config; anything with one gets the cadence that config asks for.
        #
        # This is also what finally calls _get_instance_thresholds. It has been
        # here since the MACA thresholds were added, loading config that nothing
        # in production ever asked it for — only its tests did.
        _, _, _, idle_minutes = await self._get_instance_thresholds(
            conv.get("instance_id") or ""
        )
        idle_threshold = timedelta(minutes=idle_minutes)

        idle_time = datetime.now(timezone.utc) - last_activity
        if idle_time < idle_threshold:
            return False, CompactionTriggerType.MANUAL_API

        # Idle: send long-term memory whatever turns it has not had yet. Its
        # own schedule, independent of compaction.
        await self._flush_promotion(conv)

        # The config can turn automatic compaction off. The promotion above has run: the
        # switch stops the summary, never the move of turns into long-term memory.
        from .instance_config import automatic_compaction_enabled

        if not await automatic_compaction_enabled(conv.get("instance_id")):
            return False, CompactionTriggerType.MANUAL_API

        # Idle long enough — check if there's new activity since last compaction
        try:
            last_compaction = await self.storage.get_latest(
                conv["conversation_id"],
            )
        except Exception as e:
            logger.warning(f"Failed to get last compaction: {e}")
            last_compaction = None

        if last_compaction:
            created_at = last_compaction.metadata.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at and last_activity <= created_at:
                return False, CompactionTriggerType.MANUAL_API  # Already compacted

        return True, CompactionTriggerType.IDLE_PERIOD

    async def _flush_promotion(self, conv: Dict[str, Any]) -> None:
        from synap.cloud.application.conversation.promotion import get_promoter

        promoter = get_promoter()
        if promoter is None or not conv.get("instance_id"):
            return
        await promoter.maybe_promote(
            conv["conversation_id"], conv["instance_id"], user_id=conv.get("user_id"),
            customer_id=conv.get("customer_id"), client_id=conv.get("client_id"),
            scope_path=conv.get("scope_path"), flush=True,
        )

    def _has_new_activity_since_compaction(
        self,
        conv: Dict[str, Any],
        last_compaction: Any
    ) -> bool:
        """Check if there has been new activity since the last compaction.

        Returns True if compaction is needed (no previous compaction exists
        or new messages arrived after the last compaction).
        """
        if not last_compaction:
            return True

        created_at = last_compaction.metadata.created_at
        if not created_at:
            return True

        last_activity = conv.get("last_message_at")
        if not last_activity:
            return True

        if isinstance(last_activity, str):
            try:
                last_activity = datetime.fromisoformat(last_activity)
            except (ValueError, TypeError):
                return True

        # Ensure both datetimes are timezone-aware for comparison
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=timezone.utc)

        return last_activity > created_at

    async def _get_instance_thresholds(
        self, instance_id: str
    ) -> tuple[int, int, int, int]:
        """Load compaction thresholds from MACA config for an instance.

        Returns:
            (token_threshold, message_threshold, incremental_delta, idle_minutes)
            — defaults to (15000, 30, 30, 5) if no MACA config is found.
        """
        # Only idle_minutes is used (the sweep is the idle trigger). The
        # other three are reported for compatibility; the token and message
        # triggers are model-adaptive and decided by
        # CompactionManager.should_compact (budgets.py).
        default_token_threshold = 15000
        default_message_threshold = 30
        default_incremental_delta = 30
        default_idle_minutes = int(self.idle_threshold.total_seconds() // 60) or 5

        if not instance_id:
            return (
                default_token_threshold,
                default_message_threshold,
                default_incremental_delta,
                default_idle_minutes,
            )

        # Check in-memory cache first (avoid DB hit on every 30s poll)
        cached = self._threshold_cache.get(instance_id)
        if cached is not None:
            cached_at, thresholds = cached
            if time.monotonic() - cached_at < self._THRESHOLD_CACHE_TTL_SECONDS:
                return thresholds

        try:
            config_content = await self._read_config_content(instance_id)
            if config_content:
                compaction_section = config_content.get("compaction_config", {})
                thresholds = compaction_section.get("adaptive_thresholds", {})

                token_threshold = thresholds.get(
                    "trigger_at_tokens", default_token_threshold
                )
                message_threshold = thresholds.get(
                    "trigger_at_messages", default_message_threshold
                )
                incremental_delta = thresholds.get(
                    "incremental_message_delta", default_incremental_delta
                )
                idle_minutes = thresholds.get(
                    "idle_minutes", default_idle_minutes
                )

                self._threshold_cache[instance_id] = (
                    time.monotonic(),
                    (token_threshold, message_threshold, incremental_delta, idle_minutes),
                )
                return token_threshold, message_threshold, incremental_delta, idle_minutes

        except Exception as e:
            logger.warning(f"Failed to load MACA thresholds for {instance_id}: {e}")

        result = (
            default_token_threshold,
            default_message_threshold,
            default_incremental_delta,
            default_idle_minutes,
        )
        self._threshold_cache[instance_id] = (time.monotonic(), result)
        return result

    async def _read_config_content(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """The instance's config as a dict: the mounted file (Core) or the active Postgres row (Cloud)."""
        from .instance_config import file_config

        content = await file_config(instance_id)
        if content is not None:
            return content

        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.database.repositories.maca_config_repo import MACAConfigRepository

        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = MACAConfigRepository(session)
            config_orm = await repo.get_active_for_instance(instance_id)
            return config_orm.config_content if config_orm else None

    async def _load_messages(self, conversation_id: str) -> List:
        """Load messages for a conversation."""
        try:
            return await self.storage.load_conversation_messages(conversation_id)
        except Exception as e:
            logger.error(f"Failed to load messages for {conversation_id}: {e}")
            return []

    async def _bounded_compact(self, **kwargs):
        """Execute compaction with concurrency limit."""
        async with self._semaphore:
            await self._compact_conversation(**kwargs)

    async def _compact_conversation(
        self,
        conversation_id: str,
        instance_id: str,
        messages: List,
        trigger: CompactionTriggerType,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope_path: Optional[Dict[str, str]] = None,
    ):
        """Execute compaction for a conversation."""
        try:
            await self.manager.compact(
                conversation_id=conversation_id,
                instance_id=instance_id,
                client_id=client_id or "",
                messages=messages,
                trigger_type=trigger,
                user_id=user_id,
                customer_id=customer_id,
                scope_path=scope_path,
            )

            logger.info(f"Compaction completed successfully for {conversation_id}")

        except Exception as e:
            logger.error(
                f"Compaction failed for {conversation_id}: {e}",
                exc_info=True
            )
        finally:
            await self.storage.clear_compaction_in_progress(conversation_id)


# Worker management
_worker: Optional[CompactionWorker] = None


def get_compaction_worker(
    llm_manager: Optional[LLMConfigManager] = None
) -> CompactionWorker:
    """
    Get or create singleton worker.
    
    Args:
        llm_manager: Optional LLM manager. If not provided, creates default.
        
    Returns:
        Singleton CompactionWorker instance
    """
    global _worker
    if _worker is None:
        _worker = CompactionWorker(llm_manager=llm_manager)
    return _worker


async def start_compaction_worker():
    """Start the compaction worker."""
    worker = get_compaction_worker()
    await worker.start()


async def stop_compaction_worker():
    """Stop the compaction worker."""
    worker = get_compaction_worker()
    await worker.stop()
