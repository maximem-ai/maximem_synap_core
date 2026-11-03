"""Workstream B lever 2 — per-type reserve pass in assemble_context_bundle.

Deterministic guard that the floor (a) is default-0 (byte-for-byte today's
single global cut for every instance without the MACA key) and (b) rescues a
lone low-frequency type from being crowded out of the top-N.
"""
from __future__ import annotations

from synap.cloud.application.client_api import retrieval_manager as rm


class _FakeItem:
    """Stand-in with just the attributes _reserve_per_type reads."""

    def __init__(self, context_type: str, score: float):
        self.context_type = context_type
        self._final_score = score


class TestReservePerType:
    def test_floor_rescues_rare_type(self):
        # 3 high-scoring facts + 1 low-scoring preference; budget 3.
        # A plain global cut (top-3) drops the preference entirely.
        items = [
            _FakeItem("facts", 0.90),
            _FakeItem("facts", 0.85),
            _FakeItem("facts", 0.80),
            _FakeItem("preferences", 0.40),
        ]
        out = rm._reserve_per_type(items, budget=3, floor=1)
        types = [i.context_type for i in out]
        assert len(out) == 3
        assert "preferences" in types  # the lone preference survived
        assert types.count("facts") == 2  # one fact yielded its slot

    def test_result_is_score_sorted(self):
        items = [
            _FakeItem("facts", 0.90),
            _FakeItem("facts", 0.85),
            _FakeItem("preferences", 0.40),
        ]
        out = rm._reserve_per_type(items, budget=3, floor=1)
        scores = [i._final_score for i in out]
        assert scores == sorted(scores, reverse=True)

    def test_budget_zero_returns_empty(self):
        assert rm._reserve_per_type([_FakeItem("facts", 0.9)], budget=0, floor=1) == []

    def test_floor_capped_by_budget(self):
        items = [_FakeItem("facts", 0.9), _FakeItem("preferences", 0.4), _FakeItem("emotions", 0.3)]
        out = rm._reserve_per_type(items, budget=2, floor=5)
        assert len(out) == 2  # never exceeds budget even with a high floor


class TestMacaFloorParse:
    def test_default_is_zero(self):
        s = rm.MACARetrievalSettings.from_maca_config({})
        assert s.per_type_floor == 0

    def test_parsed_from_retrieval_config(self):
        s = rm.MACARetrievalSettings.from_maca_config(
            {"retrieval_config": {"per_type_floor": 2}}
        )
        assert s.per_type_floor == 2

    def test_garbage_falls_back_to_zero(self):
        s = rm.MACARetrievalSettings.from_maca_config(
            {"retrieval_config": {"per_type_floor": None}}
        )
        assert s.per_type_floor == 0


class TestPlaygroundTemplatesCarryFloor:
    """Every playground v2 MACA template must set per_type_floor >= 1.

    Pins the BZQR6J fix: a single-topic episode cluster crowded the stored
    contact preference out of the top-8 (candidate pool held 8-15 preference
    items; final cut was 8/8 episodes). The floor guarantees each requested
    type keeps its best item. If a new agent template is added without the
    key, this fails at CI time instead of regressing silently.
    """

    def test_all_templates(self):
        import json
        from pathlib import Path

        maca_dir = (
            Path(rm.__file__).resolve().parents[1]
            / "playground" / "v2" / "maca"
        )
        templates = sorted(maca_dir.glob("*.json"))
        assert templates, f"no MACA templates found under {maca_dir}"
        for path in templates:
            cfg = json.loads(path.read_text())
            s = rm.MACARetrievalSettings.from_maca_config(cfg)
            assert s.per_type_floor >= 1, f"{path.name} missing per_type_floor"
