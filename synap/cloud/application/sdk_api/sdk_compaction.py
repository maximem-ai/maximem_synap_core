"""SDK compaction routes.

Provides endpoints for triggering and retrieving compacted conversation context:
- POST /v1/conversations/compact - Trigger compaction
- GET /v1/conversations/{conversation_id}/compacted - Get compacted context
- GET /v1/conversations/{conversation_id}/compaction/status - Check status
"""

import asyncio
import json as _json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.application.compaction import (
    get_compaction_manager,
    CompactionStorage,
    CompactionTriggerType,
)
from synap.cloud.database.connection import get_async_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/conversations", tags=["SDK Compaction"])

# Singleton storage instance
_storage = CompactionStorage()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CompactConversationRequest(BaseModel):
    conversation_id: str
    force: bool = False  # Compact even if under threshold
    strategy: Optional[str] = None  # "aggressive", "balanced", "conservative", "adaptive"
    target_tokens: Optional[int] = None  # Override MACA default

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id_is_uuid(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError(
                f"conversation_id must be a valid UUID, got '{v}'"
            )
        return v


class CompactConversationResponse(BaseModel):
    compaction_id: str
    conversation_id: str
    status: str  # "in_progress" | "completed" | "failed"
    trigger_type: str  # "manual_api"
    initiated_at: datetime
    estimated_completion_seconds: Optional[int] = 60

    # Stale-while-revalidate: existing compacted context (if any)
    previous_context: Optional[Dict[str, Any]] = None
    previous_context_age_seconds: Optional[int] = None
    previous_compaction_id: Optional[str] = None


class CompactedContextResponse(BaseModel):
    compaction_id: str
    conversation_id: str
    version: int
    created_at: datetime

    # Metadata
    compression_ratio: float
    original_token_count: int
    compacted_token_count: int

    # Formatted content (ready for LLM injection)
    formatted_context: str

    # Structured extractions
    facts: List[Dict[str, Any]] = []
    preferences: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    current_state: Dict[str, Any] = {}

    # Strategy
    strategy_used: Optional[str] = None

    # Quality
    validation_score: float
    validation_passed: Optional[bool] = None
    quality_warning: bool
    warnings: List[str] = []


class CompactionStatusResponse(BaseModel):
    conversation_id: str
    status: str  # "not_started" | "in_progress" | "completed" | "failed"

    # If completed
    compaction_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    compression_ratio: Optional[float] = None
    validation_score: Optional[float] = None

    # If in progress
    estimated_completion_seconds: Optional[int] = None

    # If failed
    error_message: Optional[str] = None

    # Latest version info
    latest_version: Optional[int] = None
    latest_created_at: Optional[datetime] = None


class RecentMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime
    message_id: str


class ContextForPromptCloudResponse(BaseModel):
    """Combined compacted context + recent un-compacted messages."""

    # Combined formatted output (compacted + recent messages)
    formatted_context: str = ""
    available: bool = False

    # Recent un-compacted messages (raw, for custom formatting)
    recent_messages: List[RecentMessage] = []
    recent_message_count: int = 0

    # Compaction metadata
    compacted_message_count: int = 0
    total_message_count: int = 0
    compression_ratio: Optional[float] = None
    validation_score: Optional[float] = None
    compaction_age_seconds: Optional[int] = None
    is_stale: bool = False
    quality_warning: bool = False

    # Compaction identity
    compaction_id: Optional[str] = None
    compaction_version: Optional[int] = None

    # The framed compaction block alone (also inside formatted_context), so a
    # caller can put it in a system message ahead of recent_messages without
    # parsing formatted_context; and the record's mode: "steering" leaves
    # detail to a search of the earlier turns.
    context_block: str = ""
    record_mode: str = ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _verify_conversation_ownership(
    session: AsyncSession,
    conversation_id: str,
    instance_id: str,
) -> bool:
    """Verify that the instance owns the conversation.

    Post-migration 063: conversation_id is TEXT, so any opaque string the
    SDK provided is accepted. No UUID shape check required.
    """
    from sqlalchemy import text
    from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

    if not conversation_id:
        return False

    # Match writer-side coercion (tracker.record_message hashes non-UUID
    # conversation_ids before insert), or the lookup misses for every
    # non-UUID id like ``pgv2_xxx``.
    conversation_id = coerce_conversation_id(conversation_id)

    result = await session.execute(
        text("""
            SELECT COUNT(*) FROM conversation
            WHERE conversation_id = :cid
              AND instance_id = :iid
        """),
        {"cid": conversation_id, "iid": instance_id},
    )
    count = result.scalar()
    return count > 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/compact", response_model=CompactConversationResponse)
async def compact_conversation(
    request: CompactConversationRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Trigger conversation compaction.

    Creates a compacted summary of the conversation history, extracting
    facts, preferences, decisions, and current state. The compaction runs
    asynchronously - use GET /compaction/status to poll for completion.

    Args:
        request: Compaction request with conversation_id and options
        auth: SDK authentication context
        session: Database session

    Returns:
        Compaction response with compaction_id and status

    Raises:
        400: Conversation not found
        403: Access denied (conversation not owned by instance)
        409: Compaction already in progress
        500: Unexpected error
    """
    try:
        # 1. Verify conversation ownership
        owns_conversation = await _verify_conversation_ownership(
            session,
            request.conversation_id,
            auth.instance_id,
        )

        if not owns_conversation:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Conversation not found or access denied",
            )

        # 2. Check if compaction already in progress
        if await _storage.is_compaction_in_progress(request.conversation_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Compaction already in progress for this conversation",
            )

        # 3. Load messages
        messages = await _storage.load_conversation_messages(request.conversation_id)

        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages found in conversation",
            )

        # 4. Trigger compaction (async, non-blocking)
        manager = get_compaction_manager()

        # An explicit request on a conversation with nothing stored yet
        # compacts, as it always has: compact() no longer makes a first
        # summary on its own when the thresholds say it is not due, and a
        # caller polling the status would otherwise wait on "not_started".
        first = await _storage.get_latest(conversation_id=request.conversation_id) is None

        # Set compaction in progress flag
        await _storage.set_compaction_in_progress(request.conversation_id)

        # Fire async task
        async def _compact_async():
            try:
                result = await manager.compact(
                    conversation_id=request.conversation_id,
                    instance_id=auth.instance_id,
                    messages=messages,
                    trigger_type=CompactionTriggerType.MANUAL_API,
                    force=request.force or first,
                    strategy=request.strategy,
                    target_tokens=request.target_tokens,
                )
                logger.info(
                    f"Compaction completed for {request.conversation_id}: "
                    f"compression_ratio={result.metadata.compression_ratio:.2f}"
                )
            except Exception as e:
                logger.error(
                    f"Compaction failed for {request.conversation_id}: {e}",
                    exc_info=True,
                )
                # Store failure in Redis so get_compaction_status() can report it
                try:
                    r = await _storage._get_redis()
                    failure_key = f"compaction_failed:{request.conversation_id}"
                    await r.setex(failure_key, 3600, _json.dumps({
                        "error_message": str(e),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    }))
                except Exception:
                    pass  # Best-effort failure tracking
            finally:
                # Clear in-progress flag
                await _storage.clear_compaction_in_progress(request.conversation_id)

        asyncio.create_task(_compact_async())

        # 5. Fetch existing compacted context (stale-while-revalidate)
        previous_context = None
        previous_context_age_seconds = None
        previous_compaction_id = None
        try:
            existing = await _storage.get_latest(
                conversation_id=request.conversation_id,
            )
            if existing:
                age = (datetime.now(timezone.utc) - existing.metadata.created_at).total_seconds()
                previous_compaction_id = existing.metadata.compaction_id
                previous_context_age_seconds = int(age)
                previous_context = {
                    "compacted_context": existing.get_injection_format(style="structured"),
                    "original_token_count": existing.metadata.original_token_count,
                    "compacted_token_count": existing.metadata.compacted_token_count,
                    "compression_ratio": existing.metadata.compression_ratio,
                    "compaction_id": existing.metadata.compaction_id,
                    "strategy_used": existing.metadata.strategy_used.value if existing.metadata.strategy_used else None,
                    "validation_score": existing.quality_assurance.validation_score,
                    "validation_passed": existing.quality_assurance.validation_passed,
                    "facts": [f.to_dict() for f in existing.structured_extractions.explicit_facts],
                    "preferences": [p.to_dict() for p in existing.structured_extractions.user_preferences],
                    "decisions": [d.to_dict() for d in existing.structured_extractions.decisions],
                    "summary": existing.context_block or existing.narrative_summary.overview,
                    "current_state": existing.narrative_summary.current_state,
                    "quality_warning": existing.quality_assurance.quality_warning,
                }
        except Exception as e:
            logger.debug(f"Could not fetch previous compacted context (non-fatal): {e}")

        # 6. Return response immediately
        return CompactConversationResponse(
            compaction_id="pending",  # Will be set when compaction completes
            conversation_id=request.conversation_id,
            status="in_progress",
            trigger_type="manual_api",
            initiated_at=datetime.now(timezone.utc),
            estimated_completion_seconds=60,
            previous_context=previous_context,
            previous_context_age_seconds=previous_context_age_seconds,
            previous_compaction_id=previous_compaction_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error triggering compaction for {request.conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger compaction: {str(e)}",
        )


@router.get("/{conversation_id}/compacted", response_model=CompactedContextResponse)
async def get_compacted_context(
    conversation_id: str,
    version: Optional[int] = None,
    format: str = "structured",
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Get compacted context for a conversation.

    Retrieves the compacted summary of a conversation, including extracted
    facts, preferences, decisions, and formatted text ready for LLM injection.

    Args:
        conversation_id: The conversation to retrieve compacted context for
        version: Specific version (optional, defaults to latest)
        format: Output format - "structured" | "narrative" | "bullet_points"
        auth: SDK authentication context
        session: Database session

    Returns:
        Compacted context with formatted text and structured extractions

    Raises:
        403: Access denied (conversation not owned by instance)
        404: No compacted context found
        500: Unexpected error
    """
    try:
        # 1. Verify conversation ownership
        owns_conversation = await _verify_conversation_ownership(
            session,
            conversation_id,
            auth.instance_id,
        )

        if not owns_conversation:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Conversation not found or access denied",
            )

        # 2. Get compacted context from storage
        if version is not None:
            compacted = await _storage.get_version(
                conversation_id=conversation_id,
                version=version,
            )
        else:
            compacted = await _storage.get_latest(
                conversation_id=conversation_id,
            )

        if not compacted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No compacted context found for this conversation",
            )

        # 3. Format the context
        formatted_context = compacted.get_injection_format(style=format)

        # 4. Build response
        return CompactedContextResponse(
            compaction_id=compacted.metadata.compaction_id,
            conversation_id=compacted.metadata.conversation_id,
            version=compacted.metadata.version,
            created_at=compacted.metadata.created_at,
            compression_ratio=compacted.metadata.compression_ratio,
            original_token_count=compacted.metadata.original_token_count,
            compacted_token_count=compacted.metadata.compacted_token_count,
            formatted_context=formatted_context,
            facts=[f.to_dict() for f in compacted.structured_extractions.explicit_facts],
            preferences=[p.to_dict() for p in compacted.structured_extractions.user_preferences],
            decisions=[d.to_dict() for d in compacted.structured_extractions.decisions],
            current_state=compacted.narrative_summary.current_state,
            strategy_used=compacted.metadata.strategy_used.value if compacted.metadata.strategy_used else None,
            validation_score=compacted.quality_assurance.validation_score,
            validation_passed=compacted.quality_assurance.validation_passed,
            quality_warning=compacted.quality_assurance.quality_warning,
            warnings=[
                w.get("issue", str(w)) if isinstance(w, dict) else str(w)
                for w in compacted.quality_assurance.warnings
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving compacted context for {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve compacted context: {str(e)}",
        )


# The styles `CompactedContext.get_injection_format` renders.
PROMPT_STYLES = ("structured", "narrative", "bullet_points")


@router.get("/{conversation_id}/context-for-prompt", response_model=ContextForPromptCloudResponse)
async def get_context_for_prompt_endpoint(
    response: Response,  # injected by FastAPI for setting Deprecation headers
    conversation_id: str = "",
    style: str = "structured",
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Get compacted context + recent un-compacted messages for LLM prompt injection.

    **Soft-deprecation**: clients with a warm
    ``ShortTermContextStore`` should render this locally via
    ``LocalContextFormatter`` — no round-trip required. This endpoint is
    kept as a fallback for cold-cache callers and any non-SDK caller that
    still wants a server-rendered string. The response carries a
    ``Deprecation`` header so HTTP middleware can detect the soft-deprecation.

    Returns a single combined formatted_context string ready for prompt injection,
    plus raw recent_messages for custom formatting. If no compaction exists yet,
    returns all conversation messages as recent_messages.

    Args:
        conversation_id: The conversation to retrieve context for
        style: Formatting style - "structured" | "narrative" | "bullet_points"
        auth: SDK authentication context
        session: Database session
    """
    # Soft-deprecation signaling per RFC 8594. Clients should switch to
    # local rendering once their SDK supports SDK-authoritative ST.
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "TBD"
    response.headers["Link"] = (
        '<https://docs.maximem.ai/sdk/short-term-context>; '
        'rel="successor-version"'
    )
    # A style `get_injection_format` cannot render raised ValueError inside
    # the handler below and came back as a 500, or was silently ignored when
    # the compaction has a pre-rendered block. It is the caller's input: 400.
    if style not in PROMPT_STYLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid style '{style}'. Must be one of: {', '.join(PROMPT_STYLES)}",
        )
    try:
        # 1. Verify ownership
        owns_conversation = await _verify_conversation_ownership(
            session, conversation_id, auth.instance_id,
        )
        if not owns_conversation:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Conversation not found or access denied",
            )

        # 2. Get compacted context
        compacted = await _storage.get_latest(
            conversation_id=conversation_id,
        )

        # 3. Load recent messages
        recent_msgs = []
        compacted_msg_count = 0
        formatted_parts = []
        compaction_id = None
        compaction_version = None
        compression_ratio = None
        validation_score = None
        compaction_age_seconds = None
        is_stale = False
        quality_warning = False

        if compacted:
            # Format the compacted context
            formatted_compacted = compacted.get_injection_format(style=style)
            if formatted_compacted:
                formatted_parts.append(formatted_compacted)

            # Extract cutoff from metadata
            source_range = compacted.metadata.source_message_range
            end_timestamp = source_range.get("end_timestamp", "")
            compacted_msg_count = source_range.get("message_count", 0)

            # Load messages since compaction cutoff
            if end_timestamp:
                recent_msgs = await _storage.load_messages_since_compaction(
                    conversation_id=conversation_id,
                    cutoff_timestamp=end_timestamp,
                )

            # Compute staleness
            if compacted.metadata.created_at:
                age = datetime.now(timezone.utc) - compacted.metadata.created_at
                compaction_age_seconds = int(age.total_seconds())
                is_stale = compaction_age_seconds > 900  # 15 minutes

            compaction_id = compacted.metadata.compaction_id
            compaction_version = compacted.metadata.version
            compression_ratio = compacted.metadata.compression_ratio
            validation_score = compacted.quality_assurance.validation_score
            quality_warning = compacted.quality_assurance.quality_warning
        else:
            # No compaction yet — load all messages as recent
            recent_msgs = await _storage.load_conversation_messages(conversation_id)

        # 4. Format recent messages and append to output
        if recent_msgs:
            lines = []
            for m in recent_msgs:
                lines.append(f"[{m.role}]: {m.content}")
            recent_section = "\n".join(lines)

            if formatted_parts:
                formatted_parts.append(
                    f"\n--- Recent Messages (not yet compacted) ---\n{recent_section}"
                )
            else:
                formatted_parts.append(recent_section)

        formatted_context = "\n".join(formatted_parts)
        total_msg_count = compacted_msg_count + len(recent_msgs)

        # 5. Build response
        recent_message_dicts = [
            RecentMessage(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                message_id=m.message_id,
            )
            for m in recent_msgs
        ]

        return ContextForPromptCloudResponse(
            formatted_context=formatted_context,
            available=bool(formatted_context),
            recent_messages=recent_message_dicts,
            recent_message_count=len(recent_msgs),
            compacted_message_count=compacted_msg_count,
            total_message_count=total_msg_count,
            compression_ratio=compression_ratio,
            validation_score=validation_score,
            compaction_age_seconds=compaction_age_seconds,
            is_stale=is_stale,
            quality_warning=quality_warning,
            compaction_id=compaction_id,
            compaction_version=compaction_version,
            context_block=(compacted.context_block if compacted else "") or "",
            record_mode=compacted.record_mode if compacted else "",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error building context-for-prompt for {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build context for prompt: {str(e)}",
        )


class ConversationTurn(BaseModel):
    position: int  # 1-based, the turn number the compaction record uses ("m12" is 12)
    role: str
    content: str
    timestamp: datetime
    message_id: str


class ConversationTurnsResponse(BaseModel):
    turns: List[ConversationTurn] = []
    total_message_count: int = 0


MAX_TURNS_PER_PAGE = 500


@router.get("/{conversation_id}/messages", response_model=ConversationTurnsResponse)
async def get_conversation_turns(
    conversation_id: str,
    start: int = 1,
    limit: int = 200,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Raw user and assistant turns by position, in the order the compaction
    numbers them, so an SDK can fill its local store with turns another
    process wrote. The SDK searches its store locally; this route only
    copies turns, it does not search.
    """
    if not await _verify_conversation_ownership(session, conversation_id, auth.instance_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Conversation not found or access denied")
    from synap.cloud.application.compaction.manager import order_messages

    messages = order_messages(await _storage.load_conversation_messages(conversation_id))
    start = max(1, start)
    limit = max(1, min(limit, MAX_TURNS_PER_PAGE))
    page = messages[start - 1:start - 1 + limit]
    return ConversationTurnsResponse(
        turns=[ConversationTurn(position=start + i, role=m.role, content=m.content, timestamp=m.timestamp,
                                message_id=m.message_id) for i, m in enumerate(page)],
        total_message_count=len(messages),
    )


@router.get("/{conversation_id}/compaction/status", response_model=CompactionStatusResponse)
async def get_compaction_status(
    conversation_id: str,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Get compaction status for a conversation.

    Checks if compaction is in progress, completed, or not started.
    Use this endpoint to poll for completion after triggering compaction.

    Args:
        conversation_id: The conversation to check status for
        auth: SDK authentication context
        session: Database session

    Returns:
        Compaction status with completion details if available

    Raises:
        403: Access denied (conversation not owned by instance)
        500: Unexpected error
    """
    try:
        # 1. Verify conversation ownership
        owns_conversation = await _verify_conversation_ownership(
            session,
            conversation_id,
            auth.instance_id,
        )

        if not owns_conversation:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Conversation not found or access denied",
            )

        # 2. Check if compaction in progress
        in_progress = await _storage.is_compaction_in_progress(conversation_id)

        if in_progress:
            return CompactionStatusResponse(
                conversation_id=conversation_id,
                status="in_progress",
                estimated_completion_seconds=60,
            )

        # 3. Get latest compaction
        compacted = await _storage.get_latest(
            conversation_id=conversation_id,
        )

        if compacted:
            return CompactionStatusResponse(
                conversation_id=conversation_id,
                status="completed",
                compaction_id=compacted.metadata.compaction_id,
                completed_at=compacted.metadata.created_at,
                compression_ratio=compacted.metadata.compression_ratio,
                validation_score=compacted.quality_assurance.validation_score,
                latest_version=compacted.metadata.version,
                latest_created_at=compacted.metadata.created_at,
            )

        # 4. Check if last compaction failed
        try:
            r = await _storage._get_redis()
            failure_key = f"compaction_failed:{conversation_id}"
            failure_data = await r.get(failure_key)
            if failure_data:
                failure = _json.loads(failure_data)
                return CompactionStatusResponse(
                    conversation_id=conversation_id,
                    status="failed",
                    error_message=failure.get("error_message"),
                )
        except Exception:
            pass  # Fall through to not_started

        # 5. No compaction found
        return CompactionStatusResponse(
            conversation_id=conversation_id,
            status="not_started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error checking compaction status for {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check compaction status: {str(e)}",
        )
