"""Compaction and the conversation tracker on the Core path.

Fakes only: no database, no model. Shows (1) the profile built from the configured LLM,
(2) the YAML switch stops compaction and never promotion, (3) ``SYNAP_SESSION_END_COMPACTION=0``
likewise, (4) Core reads the mounted file, (5) a Core turn is stored without the privacy import.
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.application.compaction import instance_config
from synap.cloud.application.compaction.interfaces import CompactionTriggerType
from synap.cloud.application.compaction.passes import base
from synap.cloud.application.compaction.worker import CompactionWorker
from synap.cloud.application.conversation.tracker import ConversationTracker
from synap.cloud.infrastructure.llm import settings as llm_settings
from synap.cloud.infrastructure.llm.models import LLMProvider
from synap.cloud.shared import edition


class FakeLoader:
    def __init__(self, raw):
        self.raw = raw

    async def get_raw_config(self, instance_id=None):
        import copy
        return copy.deepcopy(self.raw)


@pytest.fixture
def core_file(monkeypatch):
    """Core reads this dict as the mounted file."""
    holder = {"raw": {"metadata": {}}}
    monkeypatch.setattr(edition, "config_source", "file")
    monkeypatch.setattr(
        "synap.cloud.memory_config.loader.get_config_loader", lambda: FakeLoader(holder["raw"])
    )
    return holder


# ---- 1. the profile -------------------------------------------------------------------


class TestConfiguredProfile:
    def _settings(self, monkeypatch, provider="ollama", model="m1", **extra):
        env = {"SYNAP_LLM_PROVIDER": provider, "SYNAP_LLM_MODEL": model, **extra}
        monkeypatch.setattr(llm_settings, "active", lambda: llm_settings.load(env))

    def test_core_uses_the_configured_provider_and_model(self, monkeypatch):
        monkeypatch.setattr(edition, "compaction_profile", "configured")
        self._settings(monkeypatch)
        p = base.resolve_compaction_profile(MagicMock())
        assert (p.name, p.provider, p.model) == ("configured", LLMProvider.OLLAMA, "m1")
        assert p.jev_check is False and p.cross_provider_fallback is False
        assert p.reasoning_token_overhead == 0

    def test_the_compaction_step_model_wins(self, monkeypatch):
        monkeypatch.setattr(edition, "compaction_profile", "configured")
        self._settings(monkeypatch, SYNAP_LLM_MODEL_COMPACTION="small")
        assert base.resolve_compaction_profile(MagicMock()).model == "small"

    def test_the_env_profile_is_ignored_in_core(self, monkeypatch, caplog):
        monkeypatch.setattr(edition, "compaction_profile", "configured")
        monkeypatch.setenv(base.COMPACTION_PROFILE_ENV, "groq")
        self._settings(monkeypatch)
        with caplog.at_level(logging.WARNING):
            assert base.resolve_compaction_profile(MagicMock()).name == "configured"
        assert "ignored" in caplog.text

    def test_no_configured_llm_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(edition, "compaction_profile", "configured")
        monkeypatch.setattr(llm_settings, "active", lambda: None)
        with pytest.raises(base.CompactionProfileError):
            base.resolve_compaction_profile(MagicMock())

    def test_cloud_is_untouched(self, monkeypatch):
        monkeypatch.setattr(edition, "compaction_profile", None)
        monkeypatch.delenv(base.COMPACTION_PROFILE_ENV, raising=False)
        assert base.resolve_compaction_profile(MagicMock()) is base.OPENAI_PROFILE


# ---- 2. the file and the switch -------------------------------------------------------


class TestInstanceConfig:
    @pytest.mark.asyncio
    async def test_cloud_reads_no_file_and_automatic_is_on(self, monkeypatch):
        monkeypatch.setattr(edition, "config_source", "table")
        assert await instance_config.file_config("i") is None
        assert await instance_config.automatic_compaction_enabled("i") is True

    @pytest.mark.asyncio
    async def test_core_switch_off_and_default_on(self, core_file):
        assert await instance_config.automatic_compaction_enabled() is True
        core_file["raw"] = {"metadata": {}, "compaction_config": {"automatic": False}}
        assert await instance_config.automatic_compaction_enabled() is False

    def test_the_schema_accepts_and_defaults_the_switch(self):
        from synap.cloud.memory_config.config_schema import CompactionConfig

        assert CompactionConfig().automatic is True
        assert CompactionConfig(automatic=False).automatic is False

    @pytest.mark.asyncio
    async def test_the_manager_takes_its_config_from_the_file(self, core_file):
        from synap.cloud.application.compaction.manager import CompactionManager

        core_file["raw"] = {"metadata": {}, "compaction_config": {
            "adaptive_thresholds": {"idle_minutes": 9}}}
        from synap.cloud.application.compaction.templates import get_template_loader

        mgr = CompactionManager.__new__(CompactionManager)
        mgr.template_loader = get_template_loader()
        got = await mgr._fetch_maca_config("inst-1")
        assert got["compaction_config"]["adaptive_thresholds"]["idle_minutes"] == 9
        assert got["metadata"]["instance_id"] == "inst-1"
        assert (await mgr._load_config("inst-1")).idle_minutes == 9

    @pytest.mark.asyncio
    async def test_the_worker_reads_idle_minutes_from_the_file(self, core_file):
        core_file["raw"] = {"metadata": {}, "compaction_config": {
            "adaptive_thresholds": {"idle_minutes": 2}}}
        w = CompactionWorker()
        assert (await w._get_instance_thresholds("inst-1"))[3] == 2


# ---- worker: the sweep ----------------------------------------------------------------


def _idle_conv():
    return {"conversation_id": "c1", "instance_id": "inst-1", "user_id": "u",
            "last_message_at": datetime(2020, 1, 1, tzinfo=timezone.utc)}


class TestWorkerSweep:
    @pytest.mark.asyncio
    async def test_switch_off_still_promotes_but_does_not_compact(self, core_file):
        core_file["raw"] = {"metadata": {}, "compaction_config": {"automatic": False}}
        w = CompactionWorker()
        w.storage = MagicMock()
        w.storage.get_latest = AsyncMock(return_value=None)
        w._flush_promotion = AsyncMock()
        should, _ = await w._should_compact_conversation(_idle_conv())
        assert should is False
        w._flush_promotion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_switch_on_compacts_an_idle_conversation(self, core_file):
        w = CompactionWorker()
        w.storage = MagicMock()
        w.storage.get_latest = AsyncMock(return_value=None)
        w._flush_promotion = AsyncMock()
        should, trigger = await w._should_compact_conversation(_idle_conv())
        assert should is True and trigger is CompactionTriggerType.IDLE_PERIOD

    @pytest.mark.asyncio
    async def test_the_worker_start_fails_loudly_without_a_model(self, monkeypatch):
        w = CompactionWorker()
        with patch("synap.cloud.application.compaction.worker.get_compaction_manager",
                   side_effect=base.CompactionProfileError("no model")):
            with pytest.raises(base.CompactionProfileError):
                await w.start()


# ---- tracker: promotion is never stopped ----------------------------------------------

ROW = (6, "inst-1", "user-1", "cust-1", "cli-1", None, None)


def _factory(first):
    result = MagicMock()
    result.first.return_value = first
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def _tracker():
    t = ConversationTracker()
    t._spawn_promotion = MagicMock()
    t._spawn_compaction = MagicMock()
    t._session_end_trigger_enabled = AsyncMock(return_value=True)
    return t


CONV = "caadeafd-fdfc-4cc0-85e4-3f4de2aac07f"


class TestTrackerKeepsPromotion:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    async def test_the_env_switch_stops_compaction_not_promotion(self, monkeypatch, value):
        monkeypatch.setenv("SYNAP_SESSION_END_COMPACTION", value)
        t = _tracker()
        with patch("synap.cloud.application.conversation.tracker.get_session_factory",
                   return_value=_factory(ROW)):
            assert await t.compact_on_session_end(CONV) is False
        t._spawn_promotion.assert_called_once()
        assert t._spawn_promotion.call_args.kwargs["flush"] is True
        t._spawn_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_yaml_switch_stops_compaction_not_promotion(self, core_file):
        core_file["raw"] = {"metadata": {}, "compaction_config": {"automatic": False}}
        t = _tracker()
        with patch("synap.cloud.application.conversation.tracker.get_session_factory",
                   return_value=_factory(ROW)):
            assert await t.compact_on_session_end(CONV) is False
        t._spawn_promotion.assert_called_once()
        t._spawn_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_new_conversation_still_promotes_the_previous_one_with_the_env_off(self, monkeypatch):
        monkeypatch.setenv("SYNAP_SESSION_END_COMPACTION", "0")
        t = _tracker()
        t.compact_on_session_end = AsyncMock(return_value=False)
        with patch("synap.cloud.application.conversation.tracker.get_session_factory",
                   return_value=_factory(("prev",))):
            await t._compact_previous_conversation(
                conversation_id=CONV, instance_id="inst-1", user_id="u", customer_id=None)
        t.compact_on_session_end.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_message_trigger_promotes_then_stops_when_off(self, core_file):
        core_file["raw"] = {"metadata": {}, "compaction_config": {"automatic": False}}
        t = _tracker()
        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = (99999, 99, "inst-1", "u", "c", "cli", None)
        session.execute = AsyncMock(return_value=result)
        with patch("synap.cloud.application.compaction.get_compaction_manager") as gm:
            await t._maybe_trigger_compaction(CONV, session)
        t._spawn_promotion.assert_called_once()
        gm.assert_not_called()
        t._spawn_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_end_flag_comes_from_the_file_in_core(self, core_file):
        core_file["raw"] = {"metadata": {}, "memory_extraction": {"triggers": {"on_session_end": False}}}
        assert await ConversationTracker()._session_end_trigger_enabled("inst-1") is False


# ---- 5. the turn gate -----------------------------------------------------------------


class TestProtectTurn:
    @pytest.mark.asyncio
    async def test_core_stores_the_turn_as_it_arrived_and_logs_nothing(self, monkeypatch, caplog):
        monkeypatch.setattr(edition, "is_registered", lambda hook: False)
        t = ConversationTracker.__new__(ConversationTracker)
        with caplog.at_level(logging.DEBUG):
            out = await t._protect_turn("card 4111 1111 1111 1111", client_id="cli_x")
        assert out == "card 4111 1111 1111 1111"
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_cloud_still_removes_a_card(self):
        t = ConversationTracker.__new__(ConversationTracker)
        out = await t._protect_turn("The card on file is 4111 1111 1111 1111.")
        assert "4111111111111111" not in out.replace(" ", "")

    def test_the_tracker_names_no_privacy_module(self):
        import inspect
        from synap.cloud.application.conversation import tracker

        assert "cloud.privacy" not in inspect.getsource(tracker)
