"""A memory's custom category on fetch: making the name visible.

The name is stored in ``category``. A fetch shows it as ``metadata.custom_category``,
only while SYNAP_CUSTOM_CATEGORIES is on for the instance and only when the stored
value is one of the instance's enabled custom categories. Rows written by the retired
categoriser hold other non-empty values; those must never appear as a custom name.
"""
from __future__ import annotations

import json
from uuid import UUID

import pytest

import synap.cloud.application.client_api.retrieval_manager as rm
from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.application.client_api.retrieval_manager import MACARetrievalSettings, RetrievalManager
from synap.cloud.application.sdk_api.sdk_context import _bundle_item_to_context_item
from synap.cloud.shared.scope import ScopeHierarchy
from unittest.mock import AsyncMock

FLAG = "SYNAP_CUSTOM_CATEGORIES"
CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
USER = str(UUID("33333333-3333-5333-8333-333333333333"))
INSTANCE = "inst_custom"

RAW_CONFIG = {"memory_categories": {"custom_categories": [
    {"name": "Check In", "description": "d"},
    {"name": "Warranty", "description": "d", "enabled": False},
]}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(f"{FLAG}_INSTANCES", raising=False)


class FakeCache:
    def __init__(self):
        self.writes = {}

    async def set(self, key, value, ex=None):
        self.writes[key] = value

    async def get(self, key):
        return self.writes.get(key)


# --- the item ------------------------------------------------------------------
def test_the_field_survives_the_cache_round_trip():
    item = ContextItem(item_id="m", content="c", context_type="facts", source="vector",
                       custom_category="check_in")
    again = ContextItem.from_dict(json.loads(json.dumps(item.to_dict())))
    assert again.custom_category == "check_in"
    assert ContextItem.from_dict(ContextItem(
        item_id="n", content="c", context_type="facts", source="vector").to_dict()).custom_category is None


def test_the_response_metadata_carries_the_name_only_when_known():
    named = _bundle_item_to_context_item(ContextItem(
        item_id="m", content="c", context_type="facts", source="vector", custom_category="check_in"))
    assert named.metadata["custom_category"] == "check_in"
    plain = _bundle_item_to_context_item(ContextItem(
        item_id="n", content="c", context_type="facts", source="vector"))
    assert "custom_category" not in plain.metadata


# --- reading it off a stored row -----------------------------------------------
def _in_request(names):
    return rm._REQUEST_MACA.set(MACARetrievalSettings(custom_category_names=names))


def test_a_stored_category_is_served_only_when_it_names_a_configured_one():
    token = _in_request(("check_in",))
    try:
        assert rm._event_fields({"category": "check_in"}.get)["custom_category"] == "check_in"
        # A value the retired categoriser wrote
        assert rm._event_fields({"category": "personal_info"}.get)["custom_category"] is None
        assert rm._event_fields({"category": "check_in", "event_start": "2024-01-02"}.get)["custom_category"] == "check_in"
        assert rm._event_fields({}.get)["custom_category"] is None
    finally:
        rm._REQUEST_MACA.reset(token)


def test_nothing_is_served_with_no_configured_names_or_no_request():
    assert rm._event_fields({"category": "check_in"}.get)["custom_category"] is None
    token = _in_request(())
    try:
        assert rm._event_fields({"category": "check_in"}.get)["custom_category"] is None
    finally:
        rm._REQUEST_MACA.reset(token)


def test_the_settings_list_the_enabled_names_in_canonical_form():
    got = MACARetrievalSettings.from_maca_config(RAW_CONFIG)
    assert got.custom_category_names == ("check_in",)
    assert MACARetrievalSettings.from_maca_config({}).custom_category_names == ()


# --- a whole fetch, through the cache --------------------------------------------
def _decision():
    return ContextDecision(
        decision_id="d", trigger_type=TriggerType.EXPLICIT_REQUEST, client_id=CLIENT,
        instance_id=INSTANCE, user_id=USER, customer_id=CUST, retrieval_mode=RetrievalMode.FAST,
        context_types=["facts"], search_queries=["q"], token_budget=1000)


def _scope():
    return ScopeHierarchy(instance_id=INSTANCE, client_id=CLIENT, customer_id=CUST, user_id=USER)


def _manager(cache):
    manager = RetrievalManager(cache_provider=cache)
    manager._get_maca_settings = AsyncMock(
        return_value=MACARetrievalSettings.from_maca_config(RAW_CONFIG))
    manager._get_visibility_allowed_set = AsyncMock(return_value=rm.VISIBILITY_ALL)

    async def fake_fetch(decision, scope, bundle):
        # Items built the way the Chroma readers build them: from stored metadata.
        stored = {"a": "check_in", "b": "personal_info", "c": "warranty", "d": None}
        bundle.items_by_type = {"facts": [
            ContextItem(item_id=k, content=f"fact {k}", context_type="facts", source="vector",
                        confidence=0.9, similarity_score=0.9, relevance_score=0.9,
                        **rm._event_fields({"category": v}.get))
            for k, v in stored.items()]}
        return bundle

    manager._fetch_fast = fake_fetch
    return manager


def _served(bundle):
    return {it.item_id: _bundle_item_to_context_item(it).metadata.get("custom_category")
            for it in bundle.items_by_type["facts"]}


@pytest.mark.asyncio
async def test_switch_on_a_fetch_and_the_cache_hit_after_it_both_show_the_configured_name(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cache = FakeCache()
    manager = _manager(cache)
    live = await manager.fetch_context(_decision(), scope=_scope())
    assert cache.writes, "the live bundle was not cached"
    assert not live.cache_hit
    # a legacy value and a disabled category's name are not served as custom names
    assert _served(live) == {"a": "check_in", "b": None, "c": None, "d": None}

    hit = await manager.fetch_context(_decision(), scope=_scope())
    assert hit.cache_hit
    assert _served(hit) == _served(live)


@pytest.mark.asyncio
async def test_switch_off_nothing_is_served_and_no_settings_leak_past_the_fetch():
    manager = _manager(FakeCache())
    bundle = await manager.fetch_context(_decision(), scope=_scope())
    assert set(_served(bundle).values()) == {None}
    assert rm._REQUEST_MACA.get() is None


@pytest.mark.asyncio
async def test_the_switch_can_be_named_per_instance(monkeypatch):
    monkeypatch.setenv(f"{FLAG}_INSTANCES", INSTANCE)
    bundle = await _manager(FakeCache()).fetch_context(_decision(), scope=_scope())
    assert _served(bundle)["a"] == "check_in"
    monkeypatch.setenv(f"{FLAG}_INSTANCES", "inst_other")
    bundle = await _manager(FakeCache()).fetch_context(_decision(), scope=_scope())
    assert _served(bundle)["a"] is None
