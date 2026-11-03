"""Engine bugs (Gaps 5 & 6) — deterministic guards, no LLM/DB.

Gap 5: the candidate-pair similarity gate must be env-tunable, clamped to a
sane band. The default moved 0.82 -> 0.55; see
SIMILARITY_THRESHOLD for the measurement, and the Rachel test below for the
case it exists to serve.

Gap 6: natural-language DELETE must be OFF by default (the "system never
deletes" invariant), and the DELETE outcome must not leak into the classifier
prompt unless the flag is on.
"""
from __future__ import annotations

from synap.cloud.agents.ingestion.pipeline import knowledge_update as ku


class TestSimilarityThreshold:
    def test_default_is_055(self, monkeypatch):
        monkeypatch.delenv("SYNAP_KU_SIMILARITY_THRESHOLD", raising=False)
        assert ku._similarity_threshold() == ku.SIMILARITY_THRESHOLD == 0.55

    def test_the_floor_admits_the_rachel_pair_that_0_82_hid(self, monkeypatch):
        """The measured case, pinned as a number rather than a comment.

        Every live Chicago row in lme3_830ce83f scores between 0.595 and 0.762
        against the gold suburbs row, and all four are ranks 1-4 of 20, so the
        top-5 budget already held them. The floor was the only thing hiding
        them. If someone raises this default again, this test says what breaks.
        """
        monkeypatch.delenv("SYNAP_KU_SIMILARITY_THRESHOLD", raising=False)
        rachel_cosines = [0.7618, 0.6427, 0.6386, 0.5951]
        floor = ku._similarity_threshold()
        assert all(c >= floor for c in rachel_cosines), (
            f"floor {floor} hides {[c for c in rachel_cosines if c < floor]}")
        assert not any(c >= 0.82 for c in rachel_cosines)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_SIMILARITY_THRESHOLD", "0.75")
        assert ku._similarity_threshold() == 0.75

    def test_clamped_low(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_SIMILARITY_THRESHOLD", "0.1")
        assert ku._similarity_threshold() == 0.5  # floor

    def test_clamped_high(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_SIMILARITY_THRESHOLD", "1.5")
        assert ku._similarity_threshold() == 0.99  # ceiling

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_SIMILARITY_THRESHOLD", "not-a-number")
        assert ku._similarity_threshold() == ku.SIMILARITY_THRESHOLD


class TestDeleteFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("ENABLE_KU_DELETE", raising=False)
        assert ku._delete_enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        assert ku._delete_enabled() is True


class TestDeletePromptGating:
    def test_base_prompt_has_no_delete(self):
        # The default classifier must not even mention DELETE, so its output
        # distribution is unchanged when the flag is off.
        assert "DELETE" not in ku.CLASSIFICATION_SYSTEM_PROMPT

    def test_addendum_defines_delete_and_format(self):
        assert "DELETE" in ku._DELETE_ADDENDUM
        assert '"DELETE"' in ku._DELETE_ADDENDUM  # in the response-format enum
        assert "retraction" in ku._DELETE_ADDENDUM.lower()


class TestResultShape:
    def test_deletions_defaults_empty(self):
        r = ku.KnowledgeUpdateResult(supersessions=[], skips=[], enrichments=[])
        assert r.deletions == []

    def test_deletion_action_fields(self):
        a = ku.DeletionAction(
            old_memory_id="old", old_memory_content="home address is 12 Oak St",
            new_memory_id="new", new_memory_content="delete my home address",
            reason="explicit retraction",
        )
        assert a.old_memory_id == "old"
        assert a.memory_type is None  # optional, defaults None
