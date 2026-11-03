"""Regression tests for _apply_token_budget's conv_ctx allocation cap.

A long compaction summary used to deduct its full token count from the
items budget, leaving the items budget negative and silently truncating
every memory to zero. The fix caps the deduction at the conv_ctx
allocation (30% / 600 tokens) and truncates the summary text when over
allocation, so items always receive at least `budget - cap` tokens.
"""

from uuid import uuid4

import pytest

from synap.cloud.agents.models.bundles import (
    ContextBundle,
    ContextItem,
    ConversationContext,
)
from synap.cloud.agents.models.decisions import ContextDecision, TriggerType
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


def _bundle_with_items_and_long_summary() -> ContextBundle:
    """Bundle with 6 short facts and a 30 KB compaction summary."""
    facts = [
        ContextItem(item_id=f"f{i}", content=f"Fact {i}: short content here.", context_type="facts", source="vector")
        for i in range(6)
    ]
    # ~30 KB ≈ 7.5k tokens — easily exceeds the 2000-token budget.
    huge_summary = "## Employee Profile\n" + ("Alex is an engineer. " * 1500)
    return ContextBundle(
        bundle_id=str(uuid4()),
        decision_id=str(uuid4()),
        items_by_type={"facts": facts},
        conversation_context=ConversationContext(
            summary=huge_summary,
            recent_turns=[],
            conversation_id="conv-1",
        ),
    )


def _decision(token_budget: int = 2000) -> ContextDecision:
    return ContextDecision(
        decision_id=str(uuid4()),
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        conversation_id="conv-1",
        user_id="user-1",
        token_budget=token_budget,
        budget_mode="hard",
    )


def _manager() -> RetrievalManager:
    # Minimal manager — _apply_token_budget only needs self.token_counter.
    return RetrievalManager(
        cache_provider=None,
        vector_provider=None,
        graph_provider=None,
        postgres_provider=None,
    )


def test_long_conv_ctx_does_not_wipe_items():
    """The smoking-gun: a long summary used to truncate every memory."""
    bundle = _bundle_with_items_and_long_summary()
    decision = _decision(token_budget=2000)
    manager = _manager()

    result = manager._apply_token_budget(bundle, decision)

    # At least one fact must survive — the bug returned zero.
    fact_count = len(result.items_by_type.get("facts", []))
    assert fact_count > 0, (
        "items wiped to zero because conv_ctx deduction exceeded total budget"
    )


def test_long_conv_ctx_summary_is_truncated_in_place():
    """Summary text must be shortened so it fits the conv_ctx allocation."""
    bundle = _bundle_with_items_and_long_summary()
    original_summary_len = len(bundle.conversation_context.summary)
    decision = _decision(token_budget=2000)

    _manager()._apply_token_budget(bundle, decision)

    truncated = bundle.conversation_context.summary
    assert len(truncated) < original_summary_len, "summary should be truncated"
    assert truncated.endswith("…[truncated]"), (
        "truncation must use the documented sentinel so consumers can detect it"
    )


def test_short_conv_ctx_left_alone():
    """No truncation should happen when conv_ctx fits its allocation."""
    short_summary = "Short summary, well under 600 tokens."
    bundle = ContextBundle(
        bundle_id=str(uuid4()),
        decision_id=str(uuid4()),
        items_by_type={
            "facts": [ContextItem(item_id="f1", content="A short fact.", context_type="facts", source="vector")]
        },
        conversation_context=ConversationContext(
            summary=short_summary, conversation_id="conv-1"
        ),
    )
    _manager()._apply_token_budget(bundle, _decision(token_budget=2000))

    assert bundle.conversation_context.summary == short_summary
    assert len(bundle.items_by_type.get("facts", [])) == 1


def test_no_conv_ctx_does_not_steal_budget():
    """Items must keep the full budget when no conv_ctx is present."""
    bundle = ContextBundle(
        bundle_id=str(uuid4()),
        decision_id=str(uuid4()),
        items_by_type={
            "facts": [ContextItem(item_id=f"f{i}", content="x" * 100, context_type="facts", source="vector") for i in range(20)]
        },
        conversation_context=None,
    )
    decision = _decision(token_budget=2000)
    result = _manager()._apply_token_budget(bundle, decision)
    # All 20 items should fit comfortably; at minimum nothing gets wiped.
    assert len(result.items_by_type.get("facts", [])) > 0
