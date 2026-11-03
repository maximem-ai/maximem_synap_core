"""Real LLM E2E compaction test — hits Groq API with openai/gpt-oss-120b.

NO mocked LLM responses. Uses real Groq API calls to verify that the
compaction pipeline produces accurate, meaningful output from a realistic
B2B customer support conversation.

Requires: GROQ_API_KEY environment variable (loaded from .env.staging)

Run:
    pytest synap/tests/integration/compaction/test_compaction_real_llm.py -v -s
"""

import os
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from synap.cloud.application.compaction.manager import CompactionManager
from synap.cloud.application.compaction.models import (
    CompactedContext,
    CompactionConfig,
    ConversationMessage,
)
from synap.cloud.application.compaction.interfaces import (
    CompactionStrategy,
    CompactionTriggerType,
)
from synap.cloud.infrastructure.llm import LLMConfigManager, LLMProvider


# ---------------------------------------------------------------------------
# Load GROQ_API_KEY from .env.staging
# ---------------------------------------------------------------------------

def _load_env_staging():
    """Load env vars from .env.staging if not already set."""
    env_file = Path(__file__).resolve().parents[4] / ".env.staging"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env_staging()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SKIP_REASON = "GROQ_API_KEY not set — cannot run real LLM tests"


# ---------------------------------------------------------------------------
# MACA config (same as mocked test)
# ---------------------------------------------------------------------------

REALISTIC_MACA_CONFIG = {
    "metadata": {
        "client_id": "cli_acmecorp001",
        "instance_id": "inst_support_bot_01",
        "config_version": "1.2.0",
        "generated_at": "2025-06-15T10:00:00Z",
        "generated_by": "maca-v1.0",
        "template_base": "customer_support",
    },
    "memory_categories": {
        "facts": {"enabled": True, "priority": "high"},
        "preferences": {"enabled": True, "priority": "high"},
        "episodes": {"enabled": True, "priority": "medium", "retention_days": 90},
        "emotions": {"enabled": False},
        "temporal_events": {"enabled": True, "priority": "medium"},
        "custom_categories": [
            {
                "name": "ticket_context",
                "enabled": True,
                "priority": "high",
                "description": "Support ticket IDs, statuses, and escalation history",
            },
            {
                "name": "account_details",
                "enabled": True,
                "priority": "high",
                "description": "Customer account tier, plan, and billing info",
            },
        ],
    },
    "compaction_config": {
        "strategy": "adaptive",
        "adaptive_thresholds": {
            "trigger_at_tokens": 2500,
            "target_tokens": 1200,
            "preserve_last_n_turns": 4,
        },
        "preservation_priorities": [
            "current_objective",
            "unresolved_issues",
            "key_decisions",
            "ticket_context",
            "account_details",
        ],
        "summarization_candidates": [
            "chitchat",
            "resolved_issues",
            "repeated_clarifications",
        ],
    },
    "retrieval_config": {
        "default_mode": "fast",
        "ranking": {
            "primary_signals": {"recency": 0.3, "relevance": 0.5, "importance": 0.2},
        },
    },
    "agent_hints": {
        "compaction_agent": {
            "preserve_signals": [
                "error_codes",
                "api_endpoints_mentioned",
                "escalation_status",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Realistic conversation (same as mocked test)
# ---------------------------------------------------------------------------

def build_realistic_conversation() -> list[ConversationMessage]:
    """Build a realistic B2B support conversation about API errors."""
    base_time = datetime(2025, 7, 10, 14, 0, 0, tzinfo=timezone.utc)
    messages = []

    conversation_turns = [
        ("user", "Hi, I'm Sarah from TechFlow Inc. We're on the Enterprise plan "
         "(account ID: TF-2847). Our API integration has been returning 500 errors "
         "intermittently for the past 2 hours. This is blocking our production deployment."),

        ("assistant", "Hello Sarah! I can see your Enterprise account TF-2847. "
         "I'm sorry about the 500 errors — let me look into this right away. "
         "Could you share the specific endpoint and any error response bodies you're seeing?"),

        ("user", "The endpoint is POST /api/v2/webhooks/register. The error body says: "
         '{"error": "internal_server_error", "message": "Connection pool exhausted", '
         '"request_id": "req_8f3a2b"}. It happens about 30% of the time.'),

        ("assistant", "Thank you for that detail. The 'Connection pool exhausted' error "
         "on POST /api/v2/webhooks/register suggests a backend resource constraint. "
         "I can see from our monitoring that there's elevated load on the webhook service. "
         "Let me check if this is a known issue. "
         "\n\nIn the meantime, could you try these workarounds:\n"
         "1. Add retry logic with exponential backoff (start at 1s, max 30s)\n"
         "2. Reduce concurrent webhook registrations to max 5 parallel requests"),

        ("user", "We already have retry logic but with a fixed 500ms delay. "
         "The issue is that even with retries, about 10% still fail permanently. "
         "Our deployment pipeline requires all webhook registrations to succeed. "
         "Is this a platform-wide issue or specific to our account?"),

        ("assistant", "I've confirmed this is a platform-wide issue affecting the webhook "
         "registration service. Our engineering team is aware — ticket INFRA-4521 was "
         "opened 90 minutes ago. Current ETA for fix: approximately 3 hours.\n\n"
         "For your immediate needs:\n"
         "- Switch to exponential backoff (your fixed 500ms may be too aggressive)\n"
         "- We can temporarily increase your connection pool allocation\n"
         "- I'm escalating your case to Priority 1 given the production impact\n\n"
         "Decision: I'm raising your support priority to P1 and assigning engineer "
         "Mike Chen to monitor your account specifically."),

        ("user", "Thank you for the escalation. A couple more things:\n"
         "1. Can you increase our rate limit temporarily? We normally do 100 req/min "
         "but need 200 req/min during deployments.\n"
         "2. We also noticed that GET /api/v2/webhooks/list is slower than usual — "
         "taking 3-5 seconds instead of the normal <500ms."),

        ("assistant", "Absolutely:\n\n"
         "1. **Rate limit increase**: I've temporarily raised your rate limit from "
         "100 to 200 req/min for the next 24 hours. This will auto-revert. "
         "Confirmation: RATE-2847-TMP.\n\n"
         "2. **Slow GET /api/v2/webhooks/list**: This is likely a cascading effect "
         "from the same connection pool issue. The webhook list endpoint shares the "
         "same database pool. It should normalize once INFRA-4521 is resolved.\n\n"
         "I've noted these additional impacts on your P1 ticket."),

        ("user", "Got it. One more question — we're evaluating whether to implement "
         "a circuit breaker pattern on our side. Do you have any recommendations "
         "for timeout values and failure thresholds for the webhook endpoints?"),

        ("assistant", "Great question! Here are our recommended circuit breaker settings "
         "for the webhook API:\n\n"
         "- **Timeout**: 10 seconds for POST (registration), 5 seconds for GET (list)\n"
         "- **Failure threshold**: Open circuit after 5 consecutive failures\n"
         "- **Half-open retry**: Attempt 1 request after 30 seconds\n"
         "- **Success threshold**: Close circuit after 3 consecutive successes\n\n"
         "These values work well with our SLA guarantees. I'd also recommend:\n"
         "- Log all circuit breaker state changes for debugging\n"
         "- Use a separate circuit breaker per endpoint\n"
         "- Consider adding a fallback queue for failed registrations"),

        ("user", "Perfect, we'll implement that. Can you also send me the postmortem "
         "for INFRA-4521 once it's resolved? Our CTO wants to review it."),

        ("assistant", "Of course! I'll make sure the postmortem for INFRA-4521 is shared "
         "with your team once available (usually within 48 hours of resolution).\n\n"
         "Summary of actions taken:\n"
         "- P1 escalation with engineer Mike Chen assigned\n"
         "- Temporary rate limit increase to 200 req/min (RATE-2847-TMP)\n"
         "- Circuit breaker recommendations provided\n"
         "- Postmortem will be shared post-resolution\n\n"
         "Is there anything else I can help with while we wait for the fix?"),

        ("user", "No, that covers everything. Thanks for the quick response!"),

        ("assistant", "You're welcome, Sarah! I'll keep monitoring INFRA-4521 and "
         "notify you as soon as it's resolved. If the 500 errors persist beyond "
         "the 3-hour ETA, please don't hesitate to reach back out. "
         "Have a great rest of your day!"),
    ]

    for i, (role, content) in enumerate(conversation_turns):
        messages.append(ConversationMessage(
            message_id=f"msg_{i:04d}",
            role=role,
            content=content,
            timestamp=base_time + timedelta(minutes=i * 2),
        ))

    return messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_real_llm_manager() -> LLMConfigManager:
    """Build a real LLMConfigManager with mocked Redis (only Groq is needed)."""
    # Mock Redis — it's only used for usage tracking, not for LLM calls
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()
    mock_redis.incrby = AsyncMock(return_value=1)
    mock_redis.ttl = AsyncMock(return_value=-2)
    mock_redis.pipeline = MagicMock(return_value=mock_redis)
    mock_redis.execute = AsyncMock(return_value=[1, True])
    mock_redis.__aenter__ = AsyncMock(return_value=mock_redis)
    mock_redis.__aexit__ = AsyncMock(return_value=False)

    manager = LLMConfigManager(redis_client=mock_redis)

    # Verify Groq provider initialized
    provider = manager.get_provider(LLMProvider.GROQ)
    assert provider is not None, "Groq provider failed to initialize — is GROQ_API_KEY set?"

    return manager


def _build_mock_storage():
    """Mock storage (Redis + Postgres) — we only care about LLM accuracy."""
    storage = AsyncMock()
    storage.save = AsyncMock()
    storage.get_latest = AsyncMock(return_value=None)
    storage.get_from_cache = AsyncMock(return_value=None)
    storage.set_in_cache = AsyncMock()
    storage.is_compaction_in_progress = AsyncMock(return_value=False)
    storage.set_compaction_in_progress = AsyncMock()
    storage.clear_compaction_in_progress = AsyncMock()
    return storage


# ---------------------------------------------------------------------------
# Tests — REAL LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GROQ_API_KEY, reason=SKIP_REASON)
class TestRealLLMCompaction:
    """Full pipeline with REAL Groq LLM calls — no mocking of LLM responses."""

    @pytest.mark.asyncio
    async def test_full_pipeline_real_llm(self):
        """Run 6-pass compaction with real LLM and verify output accuracy."""
        llm_manager = _build_real_llm_manager()
        storage = _build_mock_storage()
        messages = build_realistic_conversation()

        manager = CompactionManager(
            llm_manager=llm_manager,
            storage=storage,
        )

        from unittest.mock import patch
        with patch.object(manager, "_fetch_maca_config", return_value=REALISTIC_MACA_CONFIG):
            result = await manager.compact(
                conversation_id="conv_real_llm_001",
                instance_id="inst_support_bot_01",
                messages=messages,
                trigger_type=CompactionTriggerType.MANUAL_API,
                force=True,
            )

        # ===================================================================
        # 1. STRUCTURAL CHECKS — pipeline produced valid output
        # ===================================================================
        assert isinstance(result, CompactedContext), f"Expected CompactedContext, got {type(result)}"

        print("\n" + "=" * 70)
        print("REAL LLM COMPACTION RESULTS")
        print("=" * 70)

        # --- Metadata ---
        meta = result.metadata
        print(f"\n--- Metadata ---")
        print(f"  Compaction ID:       {meta.compaction_id}")
        print(f"  Original tokens:     {meta.original_token_count}")
        print(f"  Compacted tokens:    {meta.compacted_token_count}")
        print(f"  Compression ratio:   {meta.compression_ratio:.2f}")
        print(f"  Strategy used:       {meta.strategy_used}")
        print(f"  Total latency:       {meta.total_latency_ms:.0f}ms")

        assert meta.original_token_count > 0
        assert meta.compacted_token_count > 0
        assert meta.compression_ratio > 0

        # ===================================================================
        # 2. CLASSIFICATION — did the LLM understand the conversation domain?
        # ===================================================================
        classification = result.conversation_classification
        print(f"\n--- Classification ---")
        print(f"  Category:    {classification.primary_category}")
        if hasattr(classification, 'subcategory'):
            print(f"  Subcategory: {classification.subcategory}")
        print(f"  Objective:   {classification.objective}")

        assert classification.primary_category != "", "Classification should have a category"
        assert classification.objective != "", "Classification should have an objective"

        # ===================================================================
        # 3. NARRATIVE SUMMARY — is it coherent?
        # ===================================================================
        narrative = result.narrative_summary
        print(f"\n--- Narrative Summary ---")
        print(f"  Overview: {narrative.overview[:200]}...")
        print(f"  Segments: {len(narrative.chronological_narrative)}")
        for i, seg in enumerate(narrative.chronological_narrative):
            if isinstance(seg, dict):
                summary_text = seg.get("summary", "")[:120]
            else:
                summary_text = str(seg)[:120]
            print(f"    Segment {i}: {summary_text}...")

        assert narrative.overview != "", "Should have a narrative overview"
        assert len(narrative.chronological_narrative) > 0, "Should have narrative segments"

        # ===================================================================
        # 4. STRUCTURED EXTRACTIONS — key facts preserved?
        # ===================================================================
        extractions = result.structured_extractions
        if isinstance(extractions, dict):
            facts = extractions.get("explicit_facts", [])
            decisions = extractions.get("decisions", [])
            constraints = extractions.get("constraints", [])
            open_questions = extractions.get("open_questions", [])
        else:
            facts = getattr(extractions, "explicit_facts", [])
            decisions = getattr(extractions, "decisions", [])
            constraints = getattr(extractions, "constraints", [])
            open_questions = getattr(extractions, "open_questions", [])

        print(f"\n--- Structured Extractions ---")
        print(f"  Facts:          {len(facts)}")
        for f in facts:
            content = f.get("content", "") if isinstance(f, dict) else f.content
            print(f"    - {content[:100]}")
        print(f"  Decisions:      {len(decisions)}")
        for d in decisions:
            content = d.get("content", "") if isinstance(d, dict) else d.content
            print(f"    - {content[:100]}")
        print(f"  Constraints:    {len(constraints)}")
        print(f"  Open questions: {len(open_questions)}")

        # --- ACCURACY CHECKS: These specific details MUST be extracted ---
        all_facts_text = " ".join(
            f.get("content", "") if isinstance(f, dict) else f.content
            for f in facts
        ).lower()

        all_decisions_text = " ".join(
            d.get("content", "") if isinstance(d, dict) else d.content
            for d in decisions
        ).lower()

        # Combine all extracted text for broader checks
        all_extracted = (all_facts_text + " " + all_decisions_text).lower()

        print(f"\n--- Accuracy Checks ---")

        # Account ID
        has_account = "tf-2847" in all_extracted
        print(f"  Account TF-2847:        {'PASS' if has_account else 'FAIL'}")
        assert has_account, f"Account ID TF-2847 must be extracted. Got: {all_extracted[:300]}"

        # Ticket number
        has_ticket = "infra-4521" in all_extracted.replace("infra ", "infra-").replace("infra_", "infra-")
        print(f"  Ticket INFRA-4521:      {'PASS' if has_ticket else 'FAIL'}")
        assert has_ticket, f"Ticket INFRA-4521 must be extracted. Got: {all_extracted[:300]}"

        # Error type
        has_error = "500" in all_extracted or "connection pool" in all_extracted
        print(f"  500/pool error:         {'PASS' if has_error else 'FAIL'}")
        assert has_error, "500 error or connection pool issue must be mentioned"

        # API endpoint
        has_endpoint = "webhook" in all_extracted or "/api/v2" in all_extracted
        print(f"  Webhook endpoint:       {'PASS' if has_endpoint else 'FAIL'}")
        assert has_endpoint, "Webhook endpoint must be mentioned"

        # P1 escalation
        has_p1 = "p1" in all_extracted or "priority 1" in all_extracted or "priority one" in all_extracted
        print(f"  P1 escalation:          {'PASS' if has_p1 else 'FAIL'}")
        assert has_p1, "P1 escalation must be captured"

        # Rate limit change
        has_rate = "200" in all_extracted and ("rate" in all_extracted or "req" in all_extracted or "limit" in all_extracted)
        print(f"  Rate limit 200:         {'PASS' if has_rate else 'FAIL'}")
        assert has_rate, "Rate limit increase to 200 must be captured"

        # Mike Chen
        has_engineer = "mike" in all_extracted or "chen" in all_extracted
        print(f"  Engineer Mike Chen:     {'PASS' if has_engineer else 'FAIL'}")
        assert has_engineer, "Engineer Mike Chen must be mentioned"

        # ===================================================================
        # 5. QUALITY ASSURANCE — validation pass results
        # ===================================================================
        qa = result.quality_assurance
        print(f"\n--- Quality Assurance ---")
        print(f"  Validation score:  {qa.validation_score:.2f}")
        print(f"  Validation passed: {qa.validation_passed}")
        if hasattr(qa, 'questions_asked'):
            print(f"  Questions asked:   {qa.questions_asked}")
        if hasattr(qa, 'questions_answered'):
            print(f"  Questions answered: {qa.questions_answered}")

        assert qa.validation_score >= 0.5, (
            f"Validation score {qa.validation_score} too low — "
            "compaction lost too much information"
        )

        # ===================================================================
        # 6. VERBATIM MESSAGES — recent messages preserved
        # ===================================================================
        print(f"\n--- Verbatim Recent Messages ---")
        print(f"  Count: {len(result.verbatim_recent_messages)}")
        assert len(result.verbatim_recent_messages) > 0, "Should preserve recent messages"

        # ===================================================================
        # 7. INJECTION FORMATS — can this be fed back to an LLM?
        # ===================================================================
        print(f"\n--- Injection Formats ---")
        for style in ["structured", "narrative", "bullet_points"]:
            formatted = result.get_injection_format(style=style)
            print(f"  {style}: {len(formatted)} chars")
            assert len(formatted) > 50, f"'{style}' format should have substantial content"

        # ===================================================================
        # 8. SERIALIZATION — survives roundtrip?
        # ===================================================================
        data = result.to_dict()
        restored = CompactedContext.from_dict(data)
        assert restored.metadata.compaction_id == result.metadata.compaction_id
        print(f"\n  Serialization roundtrip: PASS")

        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print(f"\n{'=' * 70}")
        print(f"ALL CHECKS PASSED — Real LLM compaction is accurate")
        print(f"  Model: openai/gpt-oss-120b via Groq")
        print(f"  Compression: {meta.original_token_count} → {meta.compacted_token_count} tokens "
              f"({meta.compression_ratio:.1%} ratio)")
        print(f"  Validation score: {qa.validation_score:.2f}")
        print(f"  LLM calls: ~13 across 6 passes")
        print(f"  Total latency: {meta.total_latency_ms:.0f}ms")
        print(f"{'=' * 70}\n")
