"""
ConversationTracker: Persists conversation events to Postgres.

Writes to the conversation, conversation_session, and message tables
defined in migration 003_core_entities.sql. The migration 011 trigger
auto-updates conversation stats (last_message_at, message_count,
approximate_token_count) on every message INSERT.

Also triggers automatic compaction when token/message thresholds are exceeded.
"""

import asyncio
import json
import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy import text

from synap.cloud.database.connection import get_session_factory
from synap.cloud.shared import edition
from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

logger = logging.getLogger(__name__)

NAMESPACE_SESSION = _uuid.NAMESPACE_URL


def _coerce_uuid(value: str, namespace: _uuid.UUID = _uuid.NAMESPACE_URL) -> str:
    """Ensure *value* is a valid UUID string.

    If it already is, return it unchanged.  Otherwise derive a
    deterministic UUIDv5 from *value* so that the same input always
    maps to the same UUID.
    """
    try:
        _uuid.UUID(value)
        return value
    except (ValueError, AttributeError):
        return str(_uuid.uuid5(namespace, value))

# A cheap pre-filter in front of CompactionManager.should_compact, which makes
# the real decision (record + raw tail >= T, or 30 new messages; budgets.py).
# Both are necessary conditions, so the gate never holds back a compaction
# that is due: the record is at most T/2, so a conversation reaches T only
# once its turns alone pass T/2, here 7.5K at the 128K default. The
# conversation row's count is chars/4 (the DB trigger), only a coarse
# estimate, which is why the bound is the loosest one rather than T itself.
DEFAULT_COMPACTION_TOKEN_THRESHOLD = 7500
DEFAULT_COMPACTION_MESSAGE_THRESHOLD = 30


def _session_end_compaction_enabled() -> bool:
    """Deployment-wide off switch for compacting a conversation when it ends.

    On unless ``SYNAP_SESSION_END_COMPACTION`` is set to 0/false/off/no. One
    agent turns it off for itself with ``memory_extraction.triggers.
    on_session_end`` in its MACA config.

    It stops the compaction only. The last turns still go to long-term memory:
    callers check it after they have started the promotion, never before.
    """
    value = os.getenv("SYNAP_SESSION_END_COMPACTION", "1").strip().lower()
    return value not in ("0", "false", "off", "no")

def _as_scope_path(value: Any) -> Optional[str]:
    """Encode a scope path for the jsonb column, refusing anything else.

    ⚠ THIS EXISTS BECAUSE OF A REAL OUTAGE. The transcript route
    was once changed to store the rung and was handed the resolved `Scope` model
    instead of the client's path dict. The two are one underscore apart at the
    call site (`scope` vs `request.scope`) and the parameter is typed
    `Optional[Dict[str, str]]`, but nothing enforced that at runtime. The first
    thing that noticed was `json.dumps`, which raised
    `TypeError: Object of type Scope is not JSON serializable` on the FIRST
    write of the request, so every transcript ingest 500'd for every client
    and stored nothing.

    A TypeError from a serializer names the serializer, not the mistake. This
    names the mistake, and it fails on the first call in any test that goes
    through the real tracker rather than a mock.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            "scope_path must be the client's path dict, e.g. "
            "{'tenant': 'acme', 'practice': 'high_street'}, not "
            f"{type(value).__name__}. If you are in a route, you almost "
            "certainly want `request.scope` (what the client sent) rather "
            "than `scope` (the resolved Scope model built for the ingestion "
            "pipeline)."
        )
    if not value:
        return None
    return json.dumps(value)


class ConversationTracker:
    """Persists conversation events to Postgres conversation/message tables.

    All write methods use INSERT ... ON CONFLICT DO NOTHING for idempotency.
    Errors are logged but never raised — tracking must never block the
    listening agent's retrieval pipeline.

    Optionally buffers raw conversation turns in Redis for fast short-term
    retrieval between compactions (key: ``conv_messages:{conversation_id}``).
    """

    # A regenerated or hand-edited MACA config has to reach a running process.
    # Without this the cache below is process-lifetime, so changing an agent's
    # compaction cadence did nothing until the next deploy.
    _THRESHOLD_CACHE_TTL_SECONDS = 300

    def __init__(self, redis_client=None):
        self._redis = redis_client
        # instance_id -> (cached_at_monotonic, (token_threshold, message_threshold))
        self._threshold_cache: Dict[str, tuple] = {}
        # instance_id -> (cached_at_monotonic, on_session_end)
        self._session_end_cache: Dict[str, tuple] = {}

    async def _get_compaction_thresholds(self, instance_id: str) -> tuple:
        """(token_threshold, message_threshold) for the pre-filter gate.

        No longer read from MACA: its stored trigger_at_tokens /
        trigger_at_messages are v2's every-few-turns cadence, and the real
        trigger is model-adaptive (compaction/budgets.py). Kept as a method so
        a per-instance gate can return when MACA assesses these.
        """
        return DEFAULT_COMPACTION_TOKEN_THRESHOLD, DEFAULT_COMPACTION_MESSAGE_THRESHOLD

    # ------------------------------------------------------------------
    # Low-level operations
    # ------------------------------------------------------------------

    async def ensure_conversation(
        self,
        conversation_id: str,
        instance_id: str,
        user_id: str,
        customer_id: str,
        scope_path: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create conversation row if it doesn't exist.

        Uses ON CONFLICT so concurrent calls are safe. Returns conversation_id.

        `scope_path` is the rung this conversation was opened at, and it is the
        only thing on the row that can name one: `customer_id` and `user_id`
        are the two-level model and cannot say "the high-street practice". A
        memory extracted from this conversation is filed wherever this says,
        so a conversation that carries no path behaves exactly as it always
        has.

        ⚠ The conflict clause FILLS IN a missing path rather than doing
        nothing. Two reasons, and the second is the one that matters.

        A conversation row is created by whichever turn arrives first, and on
        the streaming path that is often a turn recorded before the caller has
        anything to say about scope, or by a code path that does not carry it
        at all. `DO NOTHING` would mean the first turn permanently decides that
        this conversation has no rung, and every later turn that DOES name one
        would be silently ignored. That is the "healthy, versioned, doing
        nothing" shape.

        It fills only when the stored value IS NULL, so a path already recorded
        is never overwritten by a later turn that omits it or sends a different
        one. A conversation that changes rung mid-flight is not a thing we
        support, and quietly moving it would move the memories with it.

        `customer_id` gets the same self-heal, for the same reason: the
        gRPC-streamed path (playground) used to write it as '' for every B2C
        conversation because nothing collapsed it to user_id before the first
        INSERT, and that blank value made every later compaction-triggered
        promotion for the conversation fail permanently at
        providers.resolve_compaction_promotion_scope. Filling it in only when
        the stored value is blank means a conversation created before that
        collapse existed repairs itself on its next message, without
        clobbering a customer_id that was already correct.
        """
        conversation_id = _coerce_uuid(conversation_id)

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO conversation (
                        conversation_id, instance_id, user_id,
                        customer_id, start_timestamp, status, scope_path
                    ) VALUES (
                        :conversation_id, :instance_id, :user_id,
                        :customer_id, :start_timestamp, 'active',
                        CAST(:scope_path AS jsonb)
                    )
                    ON CONFLICT (conversation_id) DO UPDATE
                        SET scope_path = COALESCE(conversation.scope_path, EXCLUDED.scope_path),
                            customer_id = CASE
                                WHEN COALESCE(conversation.customer_id, '') = ''
                                     AND COALESCE(EXCLUDED.customer_id, '') != ''
                                THEN EXCLUDED.customer_id
                                ELSE conversation.customer_id
                            END
                        WHERE conversation.scope_path IS NULL
                           OR COALESCE(conversation.customer_id, '') = ''
                    RETURNING (xmax = 0) AS created
                """),
                {
                    "conversation_id": conversation_id,
                    "instance_id": instance_id,
                    "user_id": user_id,
                    "customer_id": customer_id,
                    "start_timestamp": datetime.now(timezone.utc),
                    "scope_path": _as_scope_path(scope_path),
                },
            )
            row = result.first()
            await session.commit()

        # A user opening a new conversation is the only sign a REST caller ever
        # gives that their previous one is over: `record_message` has no "last
        # turn" and nothing on that path sends a session end. `xmax = 0` is true
        # for the insert and false for the conflict update, so this runs once
        # per conversation, on the turn that created it.
        if row is not None and row[0] is True and user_id:
            asyncio.create_task(
                self._compact_previous_conversation(
                    conversation_id=conversation_id,
                    instance_id=instance_id,
                    user_id=user_id,
                    customer_id=customer_id,
                )
            )
        return conversation_id

    async def ensure_session(
        self,
        conversation_id: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a conversation_session row if it doesn't exist.

        If *session_id* is None a deterministic UUID5 is derived from
        the conversation_id so repeated calls produce the same session.
        Returns session_id.
        """
        conversation_id = _coerce_uuid(conversation_id)

        if session_id is None:
            session_id = str(
                _uuid.uuid5(NAMESPACE_SESSION, f"default-session:{conversation_id}")
            )
        else:
            session_id = _coerce_uuid(session_id, namespace=NAMESPACE_SESSION)

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO conversation_session (
                        session_id, conversation_id, user_id, start_timestamp
                    ) VALUES (
                        CAST(:session_id AS uuid), :conversation_id,
                        :user_id, :start_timestamp
                    )
                    ON CONFLICT (session_id) DO NOTHING
                """),
                {
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "start_timestamp": datetime.now(timezone.utc),
                },
            )
            await session.commit()
        return session_id

    async def _protect_turn(
        self,
        content: str,
        *,
        client_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """The gate, applied to one conversation turn.

        ``client_id`` comes from the caller rather than being looked up here.
        An instance-to-client query was tried and removed: it added a statement
        to every transcript write, and the callers all know the client already.
        Without it the floor still runs, because the floor needs no client, so
        a caller that omits it gets the universal promise and not the
        client-specific one.

        ⚠️ Every caller omitted it. Both SDK routes call this with no client
        at all, so the branch below was never entered and stored turns got the
        floor and nothing else. A client on Regulated had their Aadhaar, PAN,
        bank account, IFSC, UPI, email and phone correctly aliased in the
        memory and the vector store, and sitting in full in `message.content`
        and the Redis turn buffer. Found by reading the stores after pushing a
        transcript at each preset, not by any test: the parameter exists, the
        call passes it on, and nothing supplies it.

        The two forms of the id matter here. `pii_policies.client_id` is a
        uuid; the SDK auth context carries the public `cli_...` form, and
        handing that to `load_effective` raises rather than returning an
        unconfigured policy. So they are resolved first.

        Never raises, and a failure still leaves the floor applied. The
        earlier version returned the text as it arrived from the except block,
        which meant a blip in the policy store put a card number into the turn
        row: worse than the no-policy path, from the branch meant to be safe.

        Returns the text to store. Callers must use the return value for both
        the row and the Redis buffer: writing the protected text to one and the
        original to the other would leave the value in the faster of the two.
        """
        if not content or not isinstance(content, str):
            return content

        # Core has no PII step and registers nothing: the turn is stored as it
        # arrived, with no import of a folder Core does not ship and no error per message.
        if not edition.is_registered("conversation_turn_protect"):
            return content
        return await edition.acall(
            "conversation_turn_protect",
            content,
            client_id=client_id,
            instance_id=instance_id,
            customer_id=customer_id,
            user_id=user_id,
        )

    async def record_message(
        self,
        conversation_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        client_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        # Scope for the alias key. Without them a value aliased in the memory
        # gets a different label in the turn, and one value reads as two.
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Insert a message row and return the generated message_id.

        Content is stored as a JSON string in the JSONB column so that
        ``json.loads(content::text)`` in ``load_conversation_messages``
        returns a Python ``str``.

        *timestamp* is the client-supplied message time; when omitted the
        server clock (UTC now) is used — exactly the previous behavior.

        The migration-011 trigger automatically updates the parent
        conversation's ``last_message_at``, ``message_count``, and
        ``approximate_token_count``.

        If token count exceeds threshold, triggers async compaction.
        """
        conversation_id = _coerce_uuid(conversation_id)
        session_id = _coerce_uuid(session_id, namespace=NAMESPACE_SESSION)
        message_id = str(_uuid.uuid4())
        message_ts = timestamp if timestamp is not None else datetime.now(timezone.utc)

        # Map role to DB enum: 'user' | 'assistant'
        db_role = "user" if role in ("user", "human") else "assistant"

        # ⚠️ A conversation turn is stored text, and until now it was the one
        # inbound path with no privacy gate on it at all.
        #
        # Feature spec 2.3 names "the stored conversation turns" among the
        # places the old design could not reach, and spec 12 promises a
        # never-store field type does not come to rest in Postgres. A client's
        # Aadhaar typed into a chat turn sat in `message` in plain text
        # indefinitely: `memories` was clean, this table was not, and nothing
        # shortens these rows the way migration 084 shortened request bodies.
        #
        # The floor runs whatever we know, because it needs no client. The
        # client's own policy runs when the caller told us who they are.
        content = await self._protect_turn(
            content,
            client_id=client_id,
            instance_id=instance_id,
            customer_id=customer_id,
            user_id=user_id,
        )

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO message (
                        message_id, conversation_id, session_id,
                        message_type, content, timestamp, metadata
                    ) VALUES (
                        CAST(:message_id AS uuid), :conversation_id,
                        CAST(:session_id AS uuid), :message_type,
                        CAST(:content AS jsonb), :timestamp,
                        CAST(:metadata AS jsonb)
                    )
                """),
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "message_type": db_role,
                    "content": json.dumps(content),
                    "timestamp": message_ts,
                    "metadata": json.dumps(metadata) if metadata else None,
                },
            )
            await session.commit()

            # Buffer raw turn in Redis for fast short-term retrieval
            if self._redis is not None:
                try:
                    redis_key = f"conv_messages:{conversation_id}"
                    entry = json.dumps({
                        "message_id": message_id,
                        "role": db_role,
                        "content": content,
                        "timestamp": message_ts.isoformat(),
                    })
                    await self._redis.rpush(redis_key, entry)
                    await self._redis.expire(redis_key, 14400)  # 4h TTL
                except Exception as e:
                    logger.warning(f"Redis raw buffer write failed (non-fatal): {e}")

            # Check if compaction needed (non-blocking)
            try:
                await self._maybe_trigger_compaction(conversation_id, session)
            except Exception as e:
                # Never fail message insertion due to compaction check
                logger.warning(f"Compaction trigger check failed (non-fatal): {e}")

        return message_id

    async def count_messages(self, conversation_id: str) -> Optional[int]:
        """How many user and assistant messages the conversation holds.

        Returned to the SDK after a write so it can tell whether its local
        short-term cache holds the whole conversation. Counts the same rows
        ``CompactionStorage.load_conversation_messages`` reads, so the number
        matches what ``/context-for-prompt`` would assemble.

        None on failure: the caller reports "not known" and the SDK falls back
        to asking the server, which is the safe direction.
        """
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM message
                        WHERE conversation_id = :conversation_id
                          AND message_type IN ('user', 'assistant')
                    """),
                    {"conversation_id": _coerce_uuid(conversation_id)},
                )
                return int(result.scalar() or 0)
        except Exception as e:
            logger.warning("Message count failed (non-fatal): %s", e)
            return None

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    async def record_conversation_event(
        self,
        instance_id: str,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        customer_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scope_path: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """End-to-end: ensure conversation + session, then record message.

        `timestamp` is when the turn arrived, for a caller that records it
        later than that (a turn whose images are read first); None is now.

        Returns message_id on success, None on failure.
        Failures are logged but never raised.

        `scope_path` is the streamed path's half of the rung. This is the entry
        point `Listen` uses, so without it a gRPC client could send a scope on
        the proto and have it stop here, which is the "field nothing reads"
        failure the parity document was written to prevent.
        """
        try:
            await self.ensure_conversation(
                conversation_id=conversation_id,
                instance_id=instance_id,
                user_id=user_id,
                customer_id=customer_id,
                scope_path=scope_path,
            )

            sid = await self.ensure_session(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
            )

            message_id = await self.record_message(
                conversation_id=conversation_id,
                session_id=sid,
                role=role,
                content=content,
                metadata=metadata,
                timestamp=timestamp,
            )

            logger.debug(
                "Tracked message %s for conversation %s",
                message_id,
                conversation_id,
            )
            return message_id

        except Exception as e:
            logger.warning(
                "Conversation tracking failed (non-fatal): %s", e, exc_info=True
            )
            return None

    async def record_transcript(
        self,
        conversation_id: str,
        instance_id: str,
        user_id: str,
        customer_id: Optional[str],
        turns: List[Dict[str, Any]],
        conversation_type: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        conversation_metadata: Optional[Dict[str, Any]] = None,
        transcript_hash: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        session_id: Optional[str] = None,
        external_conversation_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope_path: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Bulk-record a full conversation transcript in one shot.

        Storage primitive for the one-shot transcript ingest (spec §4.4
        step 3). Ensures the conversation and session rows exist, inserts
        every turn with its client-supplied timestamp (the migration-011
        trigger keeps ``message_count`` / ``approximate_token_count`` /
        ``last_message_at`` correct automatically), buffers the last 20
        turns in Redis for ``conversation_context.recent_turns``, and
        finalizes the conversation row (type/analysis/metadata/hash/
        timestamps, ``status='completed'``).

        Each *turn* is a dict: ``role`` (required), ``content`` (required),
        and optional ``timestamp`` (datetime; server now(UTC) if absent),
        ``speaker`` (display label) and ``metadata`` — speaker and metadata
        are merged into the message ``metadata`` JSONB.

        *external_conversation_id* is the RAW client-supplied conversation
        id (pre UUID5-coercion), persisted so the id round-trips back to
        the client (the PK stores only the coerced form).

        Turns are inserted in timestamp-ascending order regardless of input
        order: the migration-011 stats trigger copies the LAST inserted
        row's timestamp into ``conversation.last_message_at`` (not the max),
        so out-of-order inserts would corrupt recency ordering.

        Deliberately does NOT call ``_maybe_trigger_compaction``: the
        tracker's >=10-message auto-trigger would fire mid-push and collide
        with the ingest endpoint's explicit force-compaction (spec §4.4
        step 3 rationale). Errors propagate to the caller — unlike the
        fire-and-forget tracking paths, the ingest endpoint must know
        whether the write succeeded.

        Returns ``{"conversation_id": <coerced>, "session_id": ...,
        "turns_recorded": <int>}``.
        """
        conversation_id = await self.ensure_conversation(
            conversation_id=conversation_id,
            instance_id=instance_id,
            user_id=user_id,
            customer_id=customer_id,
            # ⚠ This was missing, and the miss was invisible because the route
            # DOES resolve a scope and DOES hand it to extraction, so memories
            # landed on the right rung. Only the conversation ROW was left
            # without a path. Everything that later reads the rung back off
            # the conversation, `_resolve_scope_from_conversation` on the
            # fetch and the compaction event, then fell back to the two ids
            # and lost the rung, which is the exact failure this feature
            # exists to remove. One-shot transcript ingest is the main route
            # for voice traffic, so the gap covered most real calls.
            scope_path=scope_path,
        )
        session_id = await self.ensure_session(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
        )

        # Pre-build every message row so the DB transaction below is pure I/O.
        rows: List[Dict[str, Any]] = []
        for turn in turns:
            role = turn.get("role")
            db_role = "user" if role in ("user", "human") else "assistant"

            # Same gate as `record_message`, per turn. See `_protect_turn`.
            #
            # `record_transcript` knows the instance but not the client, so the
            # client id is resolved once outside the loop below and passed in;
            # when it cannot be resolved the floor still applies, because the
            # floor needs nothing from the policy store.
            content = await self._protect_turn(
                turn.get("content") or "",
                client_id=client_id,
                instance_id=instance_id,
                # The alias key is scoped by these. Omitting them minted a
                # second label for a value the memory had already aliased, so
                # one asset tag read as two.
                customer_id=customer_id,
                user_id=user_id,
            )
            ts = turn.get("timestamp")
            if ts is None:
                ts = datetime.now(timezone.utc)
            elif ts.tzinfo is None:
                # Naive client timestamps are taken as UTC; mixing naive and
                # aware datetimes would also make the sort below blow up.
                ts = ts.replace(tzinfo=timezone.utc)
            meta: Dict[str, Any] = dict(turn.get("metadata") or {})
            if turn.get("speaker") is not None:
                meta["speaker"] = turn["speaker"]
            rows.append({
                "message_id": str(_uuid.uuid4()),
                "conversation_id": conversation_id,
                "session_id": session_id,
                "message_type": db_role,
                "content": json.dumps(content),
                "timestamp": ts,
                "metadata": json.dumps(meta) if meta else None,
            })

        # Timestamp-ASC insert order (stable: equal/absent timestamps keep
        # their relative input order) — see docstring re the 011 trigger.
        rows.sort(key=lambda r: r["timestamp"])

        end_ts = ended_at if ended_at is not None else datetime.now(timezone.utc)
        update_sets = [
            "conversation_type = :conversation_type",
            "analysis = CAST(:analysis AS jsonb)",
            "conversation_metadata = CAST(:conversation_metadata AS jsonb)",
            "transcript_hash = :transcript_hash",
            "external_conversation_id = :external_conversation_id",
            "end_timestamp = :end_timestamp",
            "status = 'completed'",
        ]
        update_params: Dict[str, Any] = {
            "cid": conversation_id,
            "conversation_type": conversation_type,
            "analysis": json.dumps(analysis) if analysis is not None else None,
            "conversation_metadata": (
                json.dumps(conversation_metadata)
                if conversation_metadata is not None
                else None
            ),
            "transcript_hash": transcript_hash,
            "external_conversation_id": external_conversation_id,
            "end_timestamp": end_ts,
        }
        if started_at is not None:
            update_sets.append("start_timestamp = :start_timestamp")
            update_params["start_timestamp"] = started_at

        # One transaction: all turn INSERTs + the conversation finalize UPDATE.
        session_factory = get_session_factory()
        async with session_factory() as session:
            for row in rows:
                await session.execute(
                    text("""
                        INSERT INTO message (
                            message_id, conversation_id, session_id,
                            message_type, content, timestamp, metadata
                        ) VALUES (
                            CAST(:message_id AS uuid), :conversation_id,
                            CAST(:session_id AS uuid), :message_type,
                            CAST(:content AS jsonb), :timestamp,
                            CAST(:metadata AS jsonb)
                        )
                    """),
                    row,
                )
            await session.execute(
                text(
                    "UPDATE conversation SET "
                    + ", ".join(update_sets)
                    + " WHERE conversation_id = :cid"
                ),
                update_params,
            )
            await session.commit()
            # NOTE: no _maybe_trigger_compaction here — see docstring.

        # Buffer the tail of the transcript in Redis so
        # conversation_context.recent_turns works if the client ever fetches
        # in-conversation context (entries are TTL'd 4h and cheap).
        if self._redis is not None and rows:
            try:
                redis_key = f"conv_messages:{conversation_id}"
                for row in rows[-20:]:
                    entry = json.dumps({
                        "message_id": row["message_id"],
                        "role": row["message_type"],
                        "content": json.loads(row["content"]),
                        "timestamp": row["timestamp"].isoformat(),
                    })
                    await self._redis.rpush(redis_key, entry)
                await self._redis.expire(redis_key, 14400)  # 4h TTL
            except Exception as e:
                logger.warning(f"Redis raw buffer write failed (non-fatal): {e}")

        return {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "turns_recorded": len(rows),
        }

    async def finalize_transcript(
        self,
        conversation_id: str,
        *,
        conversation_type: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        conversation_metadata: Optional[Dict[str, Any]] = None,
        transcript_hash: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> bool:
        """Finalize a conversation from a transcript, without inserting turns.

        ``record_transcript`` does two things: it writes the turns and it
        finalizes the row. This is the second half on its own, for the
        conversation that was streamed live and then had its corrected
        transcript pushed at the end of the call.

        Those turns are already in the table. Inserting them a second time
        would double the history, corrupt the recency ordering the
        migration-011 trigger derives from the last inserted row, and feed the
        same words to compaction twice. What the client is actually adding is
        the authoritative version of what was said, plus the fields the
        streaming path has nowhere to put: ``analysis``,
        ``conversation_type``, and real start and end times.

        Setting ``transcript_hash`` here also stops the later compaction from
        promoting the same content again: `transcript_push_promotion_skip`
        keys on that column being set.

        Returns whether a row was updated.
        """
        conversation_id = coerce_conversation_id(conversation_id)
        update_sets = [
            "conversation_type = COALESCE(:conversation_type, conversation_type)",
            "analysis = COALESCE(CAST(:analysis AS jsonb), analysis)",
            "conversation_metadata = COALESCE("
            "CAST(:conversation_metadata AS jsonb), conversation_metadata)",
            "transcript_hash = :transcript_hash",
            "status = 'completed'",
        ]
        params: Dict[str, Any] = {
            "cid": conversation_id,
            "conversation_type": conversation_type,
            "analysis": json.dumps(analysis) if analysis is not None else None,
            "conversation_metadata": (
                json.dumps(conversation_metadata)
                if conversation_metadata is not None else None
            ),
            "transcript_hash": transcript_hash,
        }
        # COALESCE on the timestamps too: a transcript that names no window
        # must not blank the one the streamed turns established.
        if started_at is not None:
            update_sets.append("start_timestamp = :start_timestamp")
            params["start_timestamp"] = started_at
        if ended_at is not None:
            update_sets.append("end_timestamp = :end_timestamp")
            params["end_timestamp"] = ended_at

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    text(
                        "UPDATE conversation SET "
                        + ", ".join(update_sets)
                        + " WHERE conversation_id = :cid"
                    ),
                    params,
                )
                await session.commit()
                return bool(getattr(result, "rowcount", 0))
        except Exception as e:  # noqa: BLE001
            logger.warning("finalize_transcript failed: %s", e, exc_info=True)
            return False

    async def set_last_ingestion_id(
        self,
        conversation_id: str,
        ingestion_id: str,
    ) -> None:
        """Persist the extraction-enqueue handle on the conversation row.

        Written AFTER the ingestion enqueue succeeds (spec §4.4 step 4,
        write order: commit transcript txn → enqueue → this UPDATE). The
        transcript-ingest endpoint's duplicate branch re-enqueues whenever
        this is NULL, closing the crash-between-record-and-enqueue retry
        hole. Errors propagate — the endpoint must know the handle was not
        persisted.
        """
        conversation_id = _coerce_uuid(conversation_id)
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text("""
                    UPDATE conversation
                    SET last_ingestion_id = CAST(:ingestion_id AS uuid)
                    WHERE conversation_id = :cid
                """),
                {"ingestion_id": str(ingestion_id), "cid": conversation_id},
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Automatic Compaction Triggering
    # ------------------------------------------------------------------

    async def _maybe_trigger_compaction(
        self,
        conversation_id: str,
        session,
    ) -> None:
        """Trigger async compaction if thresholds exceeded.

        Checks conversation token_count and message_count against thresholds.
        If exceeded, sets a Redis lock and fires an async compaction task
        (non-blocking). The lock prevents duplicate compactions from
        concurrent record_message() calls.
        """
        try:
            # Get current stats from conversation table
            result = await session.execute(
                text("""
                    SELECT
                        c.approximate_token_count,
                        c.message_count,
                        c.instance_id,
                        c.user_id,
                        c.customer_id,
                        i.client_id,
                        c.scope_path
                    FROM conversation c
                    LEFT JOIN instance i ON i.instance_id = c.instance_id
                    WHERE c.conversation_id = :cid
                """),
                {"cid": conversation_id},
            )
            row = result.first()
            if not row:
                return

            # Long-term promotion runs on its own schedule, every few turns,
            # whatever compaction does. Background, never blocking.
            self._spawn_promotion(
                conversation_id, row[2], row[3], row[4],
                row[5] if len(row) > 5 else None, row[6] if len(row) > 6 else None,
            )

            token_count, message_count, instance_id = row[0], row[1], row[2]
            conv_user_id, conv_customer_id = row[3], row[4]

            # The config can turn automatic compaction off. Promotion was started above
            # and is not affected.
            from synap.cloud.application.compaction.instance_config import automatic_compaction_enabled

            if not await automatic_compaction_enabled(instance_id):
                return
            # The client the conversation belongs to. The post-compaction cache
            # invalidation builds the conversation-context cache key from it, and
            # this call used to leave it out: every compaction triggered here
            # (message count or token count, which is nearly all of them) cleared
            # a key that did not exist and left the real entry serving the FIRST
            # compaction's summary for its whole 30 minute life. The worker and the
            # retrieval path already pass it.
            conv_client_id = row[5] if len(row) > 5 else None

            # Load per-instance thresholds from MACA config (cached)
            token_threshold, message_threshold = await self._get_compaction_thresholds(instance_id)
            if token_count < token_threshold and message_count < message_threshold:
                return  # Not yet time to compact

            # Lazy import to avoid circular dependencies
            from synap.cloud.application.compaction import (
                get_compaction_manager,
                CompactionTriggerType,
            )

            manager = get_compaction_manager()
            storage = manager.storage  # Reuse singleton's storage to avoid connection leak

            # Atomically acquire compaction lock (SET NX) — single atomic operation
            if not await storage.set_compaction_in_progress(conversation_id):
                logger.debug(f"Compaction lock not acquired for {conversation_id}, skipping")
                return

            # Load messages
            messages = await storage.load_conversation_messages(conversation_id)
            if not messages:
                await storage.clear_compaction_in_progress(conversation_id)
                logger.warning(f"No messages found for {conversation_id}, skipping compaction")
                return

            # Determine trigger type using per-instance thresholds
            if token_count >= token_threshold:
                trigger = CompactionTriggerType.TOKEN_THRESHOLD
            else:
                trigger = CompactionTriggerType.MESSAGE_COUNT

            # Fire async task (non-blocking, fire-and-forget)
            self._spawn_compaction(
                manager,
                conversation_id=conversation_id,
                instance_id=instance_id,
                messages=messages,
                trigger=trigger,
                user_id=conv_user_id,
                customer_id=conv_customer_id,
                client_id=conv_client_id or "",
            )

            logger.info(
                f"Triggered automatic compaction for {conversation_id} "
                f"(trigger: {trigger.value}, {token_count} tokens, {message_count} messages)"
            )

        except Exception as e:
            # Never fail message insertion due to compaction check
            logger.warning(f"Compaction trigger check failed (non-fatal): {e}", exc_info=True)

    def _spawn_promotion(self, conversation_id, instance_id, user_id, customer_id, client_id,
                         scope_path, flush: bool = False) -> None:
        from synap.cloud.application.conversation.promotion import get_promoter

        promoter = get_promoter()
        if promoter is None or not instance_id:
            return
        try:
            asyncio.get_running_loop().create_task(promoter.maybe_promote(
                conversation_id, instance_id, user_id=user_id, customer_id=customer_id,
                client_id=client_id, scope_path=scope_path, flush=flush,
            ))
        except RuntimeError:
            pass

    def _spawn_compaction(
        self,
        manager,
        *,
        conversation_id: str,
        instance_id: str,
        messages: list,
        trigger,
        user_id: Optional[str],
        customer_id: Optional[str],
        client_id: str,
        force: bool = False,
        scope_path: Optional[Dict[str, str]] = None,
    ) -> None:
        """Run one compaction in the background and release its lock after.

        The caller has already taken the lock and loaded the messages. A
        failure is written to Redis for the status endpoint, never raised.
        """
        storage = manager.storage

        async def _compact_async():
            try:
                await manager.compact(
                    conversation_id=conversation_id,
                    instance_id=instance_id,
                    messages=messages,
                    trigger_type=trigger,
                    force=force,
                    user_id=user_id,
                    customer_id=customer_id,
                    client_id=client_id,
                    scope_path=scope_path,
                )
                logger.info(
                    f"Compaction completed for {conversation_id} "
                    f"(trigger: {trigger.value}, {len(messages)} messages)"
                )
            except Exception as e:
                logger.error(
                    f"Compaction failed for {conversation_id} "
                    f"(trigger: {trigger.value}): {e}",
                    exc_info=True,
                )
                # Store failure in Redis for status reporting
                try:
                    r = await storage._get_redis()
                    failure_key = f"compaction_failed:{conversation_id}"
                    await r.setex(failure_key, 3600, json.dumps({
                        "error_message": str(e),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    }))
                except Exception:
                    pass  # Best-effort failure tracking
            finally:
                await storage.clear_compaction_in_progress(conversation_id)

        asyncio.create_task(_compact_async())

    # ------------------------------------------------------------------
    # Compaction when a conversation ends
    # ------------------------------------------------------------------

    async def _session_end_trigger_enabled(self, instance_id: str) -> bool:
        """Whether this agent's config asks for extraction when a session ends.

        ``memory_extraction.triggers.on_session_end`` has been in every MACA
        config since the field was added, and until now nothing read it. True
        when the config does not say, which is the schema default.
        """
        cached = self._session_end_cache.get(instance_id)
        if cached is not None:
            cached_at, enabled = cached
            if time.monotonic() - cached_at < self._THRESHOLD_CACHE_TTL_SECONDS:
                return enabled

        enabled = True
        try:
            from synap.cloud.application.compaction.instance_config import file_config

            # Core reads the mounted file; Cloud (None) reads its Postgres row.
            file_content = await file_config(instance_id)
            if file_content is not None:
                enabled = (
                    (file_content.get("memory_extraction") or {})
                    .get("triggers", {})
                    .get("on_session_end", True)
                ) is not False
            else:
                from synap.cloud.database.repositories.maca_config_repo import MACAConfigRepository
                session_factory = get_session_factory()
                async with session_factory() as session:
                    repo = MACAConfigRepository(session)
                    config_orm = await repo.get_active_for_instance(instance_id)
                    if config_orm and config_orm.config_content:
                        enabled = (
                            config_orm.config_content
                            .get("memory_extraction", {})
                            .get("triggers", {})
                            .get("on_session_end", True)
                        ) is not False
        except Exception as e:
            logger.debug("Failed to load on_session_end for %s: %s", instance_id, e)

        self._session_end_cache[instance_id] = (time.monotonic(), enabled)
        return enabled

    async def compact_on_session_end(
        self,
        conversation_id: str,
        *,
        instance_id: Optional[str] = None,
        reason: str = "session_end",
    ) -> bool:
        """Compact a conversation that has ended, so its last turns are promoted.

        A turn reaches long-term memory when its conversation is compacted,
        and compaction otherwise waits for a threshold or for the idle sweep.
        A conversation that ends under the threshold therefore kept its last
        turns out of memory for the whole idle period, which is exactly when a
        user who starts a new conversation asks about them.

        This flushes long-term promotion (conversation/promotion.py), which
        sends the turns not yet sent, and forces the compaction.

        *instance_id*, when given, must own the conversation: the gRPC caller
        names the conversation itself, and one client must not be able to
        start work on another's.

        Returns True when a compaction was started. Never raises.
        """
        try:
            conversation_id = _coerce_uuid(conversation_id)
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT
                            c.message_count,
                            c.instance_id,
                            c.user_id,
                            c.customer_id,
                            i.client_id,
                            c.scope_path,
                            c.transcript_hash
                        FROM conversation c
                        LEFT JOIN instance i ON i.instance_id = c.instance_id
                        WHERE c.conversation_id = :cid
                    """),
                    {"cid": conversation_id},
                )
                row = result.first()
            if not row:
                return False

            message_count = row[0] or 0
            conv_instance_id = row[1]
            if instance_id is not None and conv_instance_id != instance_id:
                logger.warning(
                    "Session-end compaction refused: conversation %s is not "
                    "owned by instance %s", conversation_id, instance_id,
                )
                return False
            if message_count <= 0:
                return False
            # Send the turns long-term memory has not seen yet. Promotion has
            # its own cursor, so this does not wait on whether a
            # compaction runs below.
            self._spawn_promotion(conversation_id, conv_instance_id, row[2], row[3], row[4], row[5],
                                  flush=True)
            # A pushed transcript was ingested whole by its own endpoint; the
            # promotion would skip it, so the compaction buys nothing.
            if row[6]:
                return False
            # From here on only the compaction is at stake; the promotion has been started.
            if not _session_end_compaction_enabled():
                return False
            from synap.cloud.application.compaction.instance_config import automatic_compaction_enabled

            if not await automatic_compaction_enabled(conv_instance_id):
                return False
            if not await self._session_end_trigger_enabled(conv_instance_id):
                return False

            # Lazy import to avoid circular dependencies
            from synap.cloud.application.compaction import (
                get_compaction_manager,
                CompactionTriggerType,
            )

            manager = get_compaction_manager()
            storage = manager.storage

            # Nothing said since the last compaction: forcing one would run
            # the whole pipeline to promote no turns.
            existing = await storage.get_latest(conversation_id)
            if existing is not None:
                compacted_count = existing.metadata.source_message_range.get(
                    "message_count", 0
                )
                if message_count <= compacted_count:
                    return False

            if not await storage.set_compaction_in_progress(conversation_id):
                logger.debug(f"Compaction lock not acquired for {conversation_id}, skipping")
                return False

            messages = await storage.load_conversation_messages(conversation_id)
            if not messages:
                await storage.clear_compaction_in_progress(conversation_id)
                return False

            self._spawn_compaction(
                manager,
                conversation_id=conversation_id,
                instance_id=conv_instance_id,
                messages=messages,
                trigger=CompactionTriggerType.SESSION_END,
                user_id=row[2],
                customer_id=row[3],
                client_id=row[4] or "",
                force=True,
                scope_path=row[5],
            )
            logger.info(
                f"Triggered session-end compaction for {conversation_id} "
                f"(reason: {reason}, {message_count} messages)"
            )
            return True

        except Exception as e:
            logger.warning(f"Session-end compaction failed (non-fatal): {e}", exc_info=True)
            return False

    async def _compact_previous_conversation(
        self,
        *,
        conversation_id: str,
        instance_id: str,
        user_id: str,
        customer_id: Optional[str],
    ) -> None:
        """Compact the conversation this user had before the one just created.

        Only the most recent one: anything older has been idle long enough for
        the sweep to have reached it, and ``compact_on_session_end`` returns
        early for a conversation with nothing new in it.
        """
        try:
            # No switch is checked here: compact_on_session_end promotes the previous
            # conversation's last turns first, then checks the switches that stop only
            # the compaction.
            session_factory = get_session_factory()
            async with session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT conversation_id
                        FROM conversation
                        WHERE instance_id = :instance_id
                          AND user_id = :user_id
                          AND COALESCE(customer_id, '') = :customer_id
                          AND conversation_id != :conversation_id
                          AND status = 'active'
                          AND message_count > 0
                        ORDER BY last_message_at DESC NULLS LAST
                        LIMIT 1
                    """),
                    {
                        "instance_id": instance_id,
                        "user_id": user_id,
                        "customer_id": customer_id or "",
                        "conversation_id": conversation_id,
                    },
                )
                row = result.first()
            if not row:
                return
            await self.compact_on_session_end(
                str(row[0]), instance_id=instance_id, reason="next_conversation",
            )
        except Exception as e:
            logger.warning(
                f"Previous-conversation compaction failed (non-fatal): {e}", exc_info=True
            )
