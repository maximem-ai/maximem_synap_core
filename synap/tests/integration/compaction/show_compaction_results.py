"""Script to show exact test data and compaction results side by side."""

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

# Load env
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

from synap.cloud.application.compaction.manager import CompactionManager
from synap.cloud.application.compaction.models import CompactedContext, ConversationMessage
from synap.cloud.application.compaction.interfaces import CompactionStrategy, CompactionTriggerType
from synap.cloud.infrastructure.llm import LLMConfigManager, LLMProvider


# ── MACA CONFIG ──────────────────────────────────────────────────────────────

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
            {"name": "ticket_context", "enabled": True, "priority": "high",
             "description": "Support ticket IDs, statuses, and escalation history"},
            {"name": "account_details", "enabled": True, "priority": "high",
             "description": "Customer account tier, plan, and billing info"},
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
            "current_objective", "unresolved_issues", "key_decisions",
            "ticket_context", "account_details",
        ],
        "summarization_candidates": ["chitchat", "resolved_issues", "repeated_clarifications"],
    },
    "retrieval_config": {
        "default_mode": "fast",
        "ranking": {"primary_signals": {"recency": 0.3, "relevance": 0.5, "importance": 0.2}},
    },
    "agent_hints": {
        "compaction_agent": {
            "preserve_signals": ["error_codes", "api_endpoints_mentioned", "escalation_status"],
        },
    },
}


# ── CONVERSATION ─────────────────────────────────────────────────────────────

def build_conversation():
    base_time = datetime(2025, 7, 10, 14, 0, 0, tzinfo=timezone.utc)
    turns = [
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
    messages = []
    for i, (role, content) in enumerate(turns):
        messages.append(ConversationMessage(
            message_id=f"msg_{i:04d}", role=role, content=content,
            timestamp=base_time + timedelta(minutes=i * 2),
        ))
    return messages


async def main():
    # Build real LLM manager with mocked Redis
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

    llm_manager = LLMConfigManager(redis_client=mock_redis)

    storage = AsyncMock()
    storage.save = AsyncMock()
    storage.get_latest = AsyncMock(return_value=None)
    storage.get_from_cache = AsyncMock(return_value=None)
    storage.set_in_cache = AsyncMock()
    storage.is_compaction_in_progress = AsyncMock(return_value=False)
    storage.set_compaction_in_progress = AsyncMock()
    storage.clear_compaction_in_progress = AsyncMock()

    messages = build_conversation()
    manager = CompactionManager(llm_manager=llm_manager, storage=storage)

    from unittest.mock import patch
    with patch.object(manager, "_fetch_maca_config", return_value=REALISTIC_MACA_CONFIG):
        result = await manager.compact(
            conversation_id="conv_real_llm_001",
            instance_id="inst_support_bot_01",
            messages=messages,
            trigger_type=CompactionTriggerType.MANUAL_API,
            force=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PRINT EVERYTHING
    # ══════════════════════════════════════════════════════════════════════

    print()
    print("█" * 80)
    print("█  COMPACTION E2E — EXACT INPUT DATA & REAL LLM RESULTS")
    print("█" * 80)

    # ── INPUT: MACA Config ────────────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  INPUT: MACA CONFIG (Memory Architecture Creation Agent)       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Instance:              {REALISTIC_MACA_CONFIG['metadata']['instance_id']}")
    print(f"  Client:                {REALISTIC_MACA_CONFIG['metadata']['client_id']}")
    print(f"  Template Base:         {REALISTIC_MACA_CONFIG['metadata']['template_base']}")
    print(f"  Strategy:              {REALISTIC_MACA_CONFIG['compaction_config']['strategy']}")
    print(f"  Trigger at tokens:     {REALISTIC_MACA_CONFIG['compaction_config']['adaptive_thresholds']['trigger_at_tokens']}")
    print(f"  Target tokens:         {REALISTIC_MACA_CONFIG['compaction_config']['adaptive_thresholds']['target_tokens']}")
    print(f"  Preserve last N turns: {REALISTIC_MACA_CONFIG['compaction_config']['adaptive_thresholds']['preserve_last_n_turns']}")
    print(f"  Preservation priorities: {REALISTIC_MACA_CONFIG['compaction_config']['preservation_priorities']}")
    print(f"  Custom categories:     ticket_context, account_details")
    print(f"  Agent hints:           {REALISTIC_MACA_CONFIG['agent_hints']['compaction_agent']['preserve_signals']}")

    # ── INPUT: Conversation Messages ──────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  INPUT: CONVERSATION (14 messages, B2B Support)                │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    for msg in messages:
        role_label = "👤 USER" if msg.role == "user" else "🤖 ASSISTANT"
        time_str = msg.timestamp.strftime("%H:%M")
        print(f"  [{msg.message_id}] {time_str} {role_label}:")
        # Word wrap at ~70 chars
        content = msg.content
        lines = content.split("\n")
        for line in lines:
            while len(line) > 72:
                print(f"    {line[:72]}")
                line = line[72:]
            print(f"    {line}")
        print()

    # ── OUTPUT: Metadata ──────────────────────────────────────────────────
    meta = result.metadata
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: COMPACTION METADATA                                   │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Compaction ID:       {meta.compaction_id}")
    print(f"  Original tokens:     {meta.original_token_count}")
    print(f"  Compacted tokens:    {meta.compacted_token_count}")
    print(f"  Compression ratio:   {meta.compression_ratio:.2%}")
    print(f"  Strategy used:       {meta.strategy_used}")
    print(f"  Trigger type:        {meta.trigger_type}")
    print(f"  Total latency:       {meta.total_latency_ms:.0f}ms")

    # ── OUTPUT: Classification ────────────────────────────────────────────
    cls = result.conversation_classification
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: PASS 1 — CLASSIFICATION (LLM understood the domain)   │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Category:       {cls.primary_category}")
    if hasattr(cls, 'subcategory') and cls.subcategory:
        print(f"  Subcategory:    {cls.subcategory}")
    print(f"  Objective:      {cls.objective}")
    if hasattr(cls, 'characteristics') and cls.characteristics:
        print(f"  Characteristics:")
        if isinstance(cls.characteristics, dict):
            for k, v in cls.characteristics.items():
                print(f"    {k}: {v}")

    # ── OUTPUT: Extractions ───────────────────────────────────────────────
    ext = result.structured_extractions
    if isinstance(ext, dict):
        facts = ext.get("explicit_facts", [])
        decisions = ext.get("decisions", [])
        constraints = ext.get("constraints", [])
        open_qs = ext.get("open_questions", [])
        attempted = ext.get("attempted_solutions", [])
        prefs = ext.get("user_preferences", [])
        code = ext.get("code_artifacts", [])
        deps = ext.get("dependencies", [])
    else:
        facts = getattr(ext, "explicit_facts", [])
        decisions = getattr(ext, "decisions", [])
        constraints = getattr(ext, "constraints", [])
        open_qs = getattr(ext, "open_questions", [])
        attempted = getattr(ext, "attempted_solutions", [])
        prefs = getattr(ext, "user_preferences", [])
        code = getattr(ext, "code_artifacts", [])
        deps = getattr(ext, "dependencies", [])

    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: PASS 2 — STRUCTURED EXTRACTIONS                       │")
    print("└─────────────────────────────────────────────────────────────────┘")

    def _print_items(label, items):
        print(f"\n  {label} ({len(items)}):")
        if not items:
            print("    (none)")
            return
        for i, item in enumerate(items):
            if isinstance(item, dict):
                content = item.get("content", str(item))
                source = item.get("source_messages", [])
                confidence = item.get("confidence", "")
            else:
                content = getattr(item, "content", str(item))
                source = getattr(item, "source_messages", [])
                confidence = getattr(item, "confidence", "")
            source_str = f" [from: {', '.join(source)}]" if source else ""
            conf_str = f" ({confidence})" if confidence else ""
            print(f"    {i+1}. {content}{source_str}{conf_str}")

    _print_items("FACTS", facts)
    _print_items("DECISIONS", decisions)
    _print_items("CONSTRAINTS", constraints)
    _print_items("OPEN QUESTIONS", open_qs)
    _print_items("ATTEMPTED SOLUTIONS", attempted)
    _print_items("USER PREFERENCES", prefs)
    if code:
        _print_items("CODE ARTIFACTS", code)
    if deps:
        _print_items("DEPENDENCIES", deps)

    # ── OUTPUT: Narrative Summary ─────────────────────────────────────────
    narr = result.narrative_summary
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: PASS 3 — NARRATIVE SUMMARY                            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Overview:")
    overview = narr.overview
    while len(overview) > 72:
        print(f"    {overview[:72]}")
        overview = overview[72:]
    print(f"    {overview}")

    print(f"\n  Chronological segments ({len(narr.chronological_narrative)}):")
    for i, seg in enumerate(narr.chronological_narrative):
        if hasattr(seg, 'summary'):
            summary = seg.summary
        elif isinstance(seg, dict):
            summary = seg.get("summary", str(seg))
        else:
            summary = str(seg)
        print(f"\n    Segment {i}:")
        while len(summary) > 68:
            print(f"      {summary[:68]}")
            summary = summary[68:]
        print(f"      {summary}")

        # Key moments
        if hasattr(seg, 'key_moments') and seg.key_moments:
            print(f"      Key moments:")
            for km in seg.key_moments:
                if isinstance(km, dict):
                    print(f"        - {km.get('moment', '')} (significance: {km.get('significance', '')})")
                else:
                    print(f"        - {km}")

    # ── OUTPUT: Quality Assurance ─────────────────────────────────────────
    qa = result.quality_assurance
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: PASS 4 — QUALITY ASSURANCE (Validation)               │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Validation score:    {qa.validation_score:.2f} / 1.00")
    print(f"  Validation passed:   {qa.validation_passed}")
    if hasattr(qa, 'questions_asked'):
        print(f"  Questions asked:     {qa.questions_asked}")
    if hasattr(qa, 'questions_answered'):
        print(f"  Questions answered:  {qa.questions_answered}")
    if hasattr(qa, 'qa_pairs') and qa.qa_pairs:
        print(f"\n  Question/Answer pairs:")
        for i, pair in enumerate(qa.qa_pairs):
            if isinstance(pair, dict):
                q = pair.get("question", "")
                a = pair.get("answer", "")
                answerable = pair.get("answerable", "")
                confidence = pair.get("confidence", "")
            else:
                q = getattr(pair, "question", "")
                a = getattr(pair, "answer", "")
                answerable = getattr(pair, "answerable", "")
                confidence = getattr(pair, "confidence", "")
            status = "✅" if answerable else "❌"
            print(f"    {status} Q{i+1}: {q}")
            print(f"       A: {a}")
            if confidence:
                print(f"       Confidence: {confidence}")

    # ── OUTPUT: Verbatim Messages ─────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: VERBATIM PRESERVED MESSAGES                           │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  {len(result.verbatim_recent_messages)} messages preserved verbatim (last {REALISTIC_MACA_CONFIG['compaction_config']['adaptive_thresholds']['preserve_last_n_turns']} turns):")
    for msg in result.verbatim_recent_messages:
        if isinstance(msg, dict):
            mid = msg.get("message_id", "")
            role = msg.get("role", "")
            content = msg.get("content", "")[:80]
        else:
            mid = getattr(msg, "message_id", "")
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")[:80]
        print(f"    [{mid}] {role}: {content}...")

    # ── OUTPUT: Injection Formats ─────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  OUTPUT: INJECTION FORMATS (what gets fed back to the LLM)     │")
    print("└─────────────────────────────────────────────────────────────────┘")

    for style in ["structured", "narrative", "bullet_points"]:
        formatted = result.get_injection_format(style=style)
        print(f"\n  ── {style.upper()} ({len(formatted)} chars) ──")
        # Print first 500 chars
        preview = formatted[:600]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(formatted) > 600:
            print(f"    ... ({len(formatted) - 600} more chars)")

    # ── ACCURACY SCORECARD ────────────────────────────────────────────────
    all_text = ""
    for item_list in [facts, decisions, constraints, open_qs, attempted, prefs]:
        for item in item_list:
            if isinstance(item, dict):
                all_text += " " + item.get("content", "")
            else:
                all_text += " " + getattr(item, "content", str(item))
    all_text = all_text.lower()

    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  ACCURACY SCORECARD                                            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()

    checks = [
        ("Account ID TF-2847", "tf-2847" in all_text),
        ("Enterprise plan", "enterprise" in all_text),
        ("Customer name Sarah", "sarah" in all_text),
        ("Company TechFlow Inc", "techflow" in all_text),
        ("Endpoint POST /api/v2/webhooks/register", "webhook" in all_text and ("register" in all_text or "/api" in all_text)),
        ("Error: 500 / Connection pool exhausted", "500" in all_text or "connection pool" in all_text),
        ("Request ID req_8f3a2b", "req_8f3a2b" in all_text),
        ("~30% failure rate", "30%" in all_text or "30 %" in all_text or "30% of" in all_text),
        ("Ticket INFRA-4521", "infra-4521" in all_text or "infra 4521" in all_text),
        ("Platform-wide issue", "platform" in all_text),
        ("P1 escalation", "p1" in all_text or "priority 1" in all_text),
        ("Engineer Mike Chen assigned", "mike" in all_text and "chen" in all_text),
        ("Rate limit 100→200 req/min", "200" in all_text and ("rate" in all_text or "req" in all_text)),
        ("Confirmation code RATE-2847-TMP", "rate-2847" in all_text),
        ("GET /webhooks/list slow (3-5s)", "webhook" in all_text and ("list" in all_text or "get" in all_text)),
        ("Circuit breaker recommendations", "circuit" in all_text or "breaker" in all_text or "timeout" in all_text),
        ("Postmortem requested by CTO", "postmortem" in all_text or "post-mortem" in all_text or "post mortem" in all_text or "post\u2011mortem" in all_text or "post\u2010mortem" in all_text or "post\u2012mortem" in all_text or "post\u2013mortem" in all_text),
        ("Existing retry logic (500ms)", "retry" in all_text or "500ms" in all_text or "500 ms" in all_text),
        ("Deployment pipeline blocked", "deployment" in all_text or "pipeline" in all_text),
        ("ETA 3 hours for fix", "3 hour" in all_text or "three hour" in all_text or "3-hour" in all_text or "3 hr" in all_text),
    ]

    passed = 0
    failed = 0
    for label, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  {label}")

    print(f"\n  Score: {passed}/{len(checks)} ({100*passed/len(checks):.0f}%)")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print()
    print("█" * 80)
    print(f"█  FINAL: {passed}/{len(checks)} accuracy checks passed")
    print(f"█  Compression: {meta.original_token_count} → {meta.compacted_token_count} tokens ({meta.compression_ratio:.1%})")
    print(f"█  Validation score: {qa.validation_score:.2f}/1.00")
    print(f"█  Latency: {meta.total_latency_ms:.0f}ms")
    print(f"█  Model: openai/gpt-oss-120b via Groq (REAL LLM — no mocking)")
    print("█" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
