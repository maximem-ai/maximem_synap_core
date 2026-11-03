"""Long-term promotion: raw conversation turns to ingestion, on their own schedule.

Long-term memories are built from the raw turns, never from a compaction.
This sends the turns not yet sent to the ingestion pipeline, whose Stage 2/3
extracts facts, entities and relationships from the full conversational
text, the same as ``sdk.memories.create()``.

Until compaction v3 this ran in a post-compaction callback, so memory was
written whenever compaction ran, every 5 messages or so. v3 compacts at 15K
tokens or 30 messages, which would leave memory unwritten for most of a
conversation, so promotion is decoupled (COMPACTION_V3_PLAN.md §4.6):

  * every PROMOTE_EVERY new turns   (conversation tracker, per turn)
  * when the conversation goes idle (compaction worker sweep, flush)
  * at session end                  (tracker, flush)

The cursor is ``conversation.promoted_message_count`` (migration 163), how
many turns in (timestamp, message_id) order have been sent. Each batch has a
deterministic document id built from its first and last message id, so a
retried batch is the same document. Transcript-push conversations are
skipped: the transcript ingest already ingested the full transcript.
"""
from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace
from typing import Dict, Optional

from sqlalchemy import text

from synap.cloud.database.connection import get_session_factory
from synap.cloud.shared import edition
from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

logger = logging.getLogger(__name__)

PROMOTE_EVERY = 5
_LOCK_TTL_SECONDS = 120


class LongTermPromoter:
    def __init__(self, storage, ingestion_manager, redis=None):
        self._storage = storage
        self._ingestion = ingestion_manager
        self._redis = redis
        # The ingestion id of the batch each conversation last sent, for the
        # end-of-conversation profile trigger (Core). Read and cleared by maybe_promote.
        self._last_sent: Dict[str, object] = {}

    async def _counts(self, conversation_id: str) -> Optional[tuple[int, int]]:
        session_factory = get_session_factory()
        async with session_factory() as session:
            row = (await session.execute(
                text("SELECT promoted_message_count, message_count FROM conversation WHERE conversation_id = :cid"),
                {"cid": conversation_id},
            )).first()
        return (int(row[0] or 0), int(row[1] or 0)) if row else None

    async def _advance(self, conversation_id: str, old: int, new: int) -> bool:
        """Move the cursor from ``old`` to ``new``; False if someone else moved it."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("""
                    UPDATE conversation SET promoted_message_count = :new
                     WHERE conversation_id = :cid AND promoted_message_count = :old
                """),
                {"cid": conversation_id, "old": old, "new": new},
            )
            await session.commit()
        return (result.rowcount or 0) > 0

    async def _lock(self, conversation_id: str) -> bool:
        if self._redis is None:
            return True
        try:
            return bool(await self._redis.set(f"promotion_lock:{conversation_id}", "1", nx=True,
                                              ex=_LOCK_TTL_SECONDS))
        except Exception:
            return True  # the conditional cursor update still guards a race

    async def _unlock(self, conversation_id: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(f"promotion_lock:{conversation_id}")
            except Exception:
                pass

    async def maybe_promote(
        self,
        conversation_id: str,
        instance_id: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope_path: Optional[Dict[str, str]] = None,
        flush: bool = False,
        trigger_type=None,
    ) -> int:
        """Send unsent turns when PROMOTE_EVERY have accumulated, or any
        unsent turns when ``flush``. Returns how many were sent. Never raises."""
        try:
            sent = await self._promote(conversation_id, instance_id, user_id, customer_id, client_id,
                                       scope_path, flush, trigger_type)
        except Exception as e:
            logger.warning(f"Long-term promotion failed for {conversation_id} (non-fatal): {e}")
            sent = 0
        # A flush is a conversation ending or going idle. Where the edition asks for it
        # (Core), update the user's profile now; Cloud does it in the consolidation cycle.
        # The batch id is kept across mid-conversation sends so the final flush can wait for it.
        ingestion_id = None
        try:
            key = coerce_conversation_id(conversation_id)
            if flush or not edition.profile_at_conversation_end:
                ingestion_id = self._last_sent.pop(key, None)
        except Exception:
            ingestion_id = None
        if flush and edition.profile_at_conversation_end:
            from synap.cloud.application.profile.conversation_end import spawn_on_conversation_end
            spawn_on_conversation_end(
                conversation_id=conversation_id, instance_id=instance_id, user_id=user_id,
                customer_id=customer_id, client_id=client_id, scope_path=scope_path,
                ingestion=self._ingestion, ingestion_id=ingestion_id, sent=sent,
            )
        return sent

    async def _promote(self, conversation_id, instance_id, user_id, customer_id, client_id, scope_path,
                       flush, trigger_type) -> int:
        from synap.cloud.application.compaction.manager import order_messages
        from synap.cloud.application.shared.providers import (
            resolve_compaction_promotion_scope,
            transcript_push_promotion_skip,
        )

        conversation_id = coerce_conversation_id(conversation_id)
        counts = await self._counts(conversation_id)
        if counts is None:
            return 0
        promoted, total = counts
        pending = total - promoted
        if pending <= 0 or (not flush and pending < PROMOTE_EVERY):
            return 0
        if not await self._lock(conversation_id):
            return 0
        try:
            event = SimpleNamespace(
                conversation_id=conversation_id, instance_id=instance_id, user_id=user_id,
                customer_id=customer_id, client_id=client_id, scope_path=scope_path,
                trigger_type=trigger_type,
            )
            messages = order_messages(await self._storage.load_conversation_messages(conversation_id))
            if await transcript_push_promotion_skip(event):
                await self._advance(conversation_id, promoted, len(messages))
                return 0
            batch = [m for m in messages[promoted:] if m.content and m.content.strip()]
            if not batch:
                if len(messages) > promoted:
                    await self._advance(conversation_id, promoted, len(messages))
                return 0
            scope = await resolve_compaction_promotion_scope(event)
            if scope is None:
                return 0

            from synap.cloud.agents.ingestion.models.enums import DocumentSource, DocumentType, IngestPriority
            from synap.cloud.agents.ingestion.models.inputs import IngestDocumentRequest

            first, last = batch[0], batch[-1]
            doc_key = f"promotion:{conversation_id}:{first.message_id}:{last.message_id}"
            doc_id = "promotion_" + hashlib.sha256(doc_key.encode()).hexdigest()[:24]
            transcript = "\n".join(f"{m.role}: {m.content}" for m in batch)
            request = IngestDocumentRequest(
                content=transcript,
                document_id=doc_id,
                document_type=DocumentType.CONVERSATION,
                source=DocumentSource.LISTENING_AGENT,
                priority=IngestPriority.REALTIME,
                scope=scope,
                metadata={
                    # Stage 4 writes the memory_actions changelog only when a
                    # correlation id is present; one batch, one set of actions.
                    "_trace": {"correlation_id": doc_id},
                    "source": "conversation_promotion",
                    "conversation_id": conversation_id,
                    "message_count": len(batch),
                    "source_message_range": {
                        "start_id": first.message_id,
                        "end_id": last.message_id,
                        "from_position": promoted + 1,
                        "to_position": len(messages),
                    },
                },
            )
            response = await self._ingestion.ingest_document(request)
            await self._advance(conversation_id, promoted, len(messages))
            self._last_sent[conversation_id] = getattr(response, "ingestion_id", None)
            # A forget that found nothing may be waiting for exactly this batch; its timed
            # attempts count from now. Never raises (memory_forget/retry.py).
            # Wrapped here too: a missing Jev key or any forget failure must never break
            # promotion, whose batch is already sent and its cursor advanced.
            try:
                from synap.cloud.application.memory_forget import retry as forget_retry
                await forget_retry.on_batch_sent(scope=scope, conversation_id=conversation_id,
                                                 from_position=promoted + 1)
            except Exception:
                logger.warning("forget_on_batch_sent_failed conversation=%s", conversation_id,
                               exc_info=True)
            logger.info(
                f"Promoted {len(batch)} raw turns of {conversation_id} to ingestion "
                f"(doc_id={doc_id}, id={getattr(response, 'ingestion_id', '')}, flush={flush})"
            )
            return len(batch)
        finally:
            await self._unlock(conversation_id)


_promoter: Optional[LongTermPromoter] = None


def set_promoter(promoter: Optional[LongTermPromoter]) -> None:
    """Installed at startup by shared/providers.py, once ingestion is wired."""
    global _promoter
    _promoter = promoter


def get_promoter() -> Optional[LongTermPromoter]:
    return _promoter
