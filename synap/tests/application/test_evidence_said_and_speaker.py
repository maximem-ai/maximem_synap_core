"""Every excerpt says when it was said, and who was speaking.

Two gaps in the raw passage we attach under each fact, both measured on the
4,250 excerpts served for the 293-question cohort:

1. 1,890 of them (44.5%) went out with NO date. A reader takes an undated
   passage as current. In one case the sentence "User: She moved to Chicago."
   was attached, undated, to a weather fact and the answer was Chicago, five
   days after the user said the friend had moved back to the suburbs.

2. 3,152 of them (74.2%) begin in the middle of somebody's turn, because a
   chunk is a 500-token window and not a turn, and half the characters of
   those excerpts (50.3%) sit in that unattributed opening turn. 481 carry no
   marker at all. The assistant's words then read as the user's.

   This is not the fix for c4a1ceb8 ("How many different types of citrus fruits
   have I used in my cocktail recipes?", gold 3, answered 4), and
   `test_who_owns_the_grapefruit` says so against the eight served excerpts
   verbatim: grapefruit is only ever the assistant's suggestion, seven of the
   eight already show "Assistant:" before the mention, and the reader counted
   it anyway. One of the eight gains the missing speaker.

Both are off unless the deployment turns them on, and with both off the text is
byte-for-byte what shipped before.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from synap.cloud.application.client_api import retrieval_manager as rm

FIXTURE = Path(__file__).parent / "fixtures" / "real_excerpts_lme23.json"
REAL = json.loads(FIXTURE.read_text())


class TestFlags:
    def test_both_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", raising=False)
        monkeypatch.delenv("SYNAP_EVIDENCE_SPEAKER_LABEL", raising=False)
        assert rm.evidence_said_from_document_enabled() is False
        assert rm.evidence_speaker_label_enabled() is False

    @pytest.mark.parametrize("raw", ["on", "1", "true", "yes", "ON", " True "])
    def test_the_same_spellings_as_every_other_retrieval_flag(self, monkeypatch, raw):
        monkeypatch.setenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", raw)
        monkeypatch.setenv("SYNAP_EVIDENCE_SPEAKER_LABEL", raw)
        assert rm.evidence_said_from_document_enabled() is True
        assert rm.evidence_speaker_label_enabled() is True

    def test_anything_else_is_off(self, monkeypatch):
        monkeypatch.setenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", "soon")
        monkeypatch.setenv("SYNAP_EVIDENCE_SPEAKER_LABEL", "maybe")
        assert rm.evidence_said_from_document_enabled() is False
        assert rm.evidence_speaker_label_enabled() is False


class TestWhoIsSpeaking:
    """The rule is alternation, measured against ground truth on real sessions.

    575 windows cut from 60 real LongMemEval sessions by the real ingest
    windowing: 433 of them start mid-turn. Alternation from the first marker in
    the passage got 379 right and **0 wrong**, and said nothing on 54. The
    previous window's last marker, on its own, was WRONG on 68 of 433 (15.7%),
    because windows overlap by 75 tokens and that marker often sits inside the
    overlap -- after this window has already begun. Falling back to it only for
    the 54 answers all 433 with nothing wrong.
    """

    def test_who_owns_the_grapefruit(self):
        """What the label does and does not buy on c4a1ceb8, on the real eight.

        Five of the eight are labelled `[Assistant]`; three need nothing (two
        already open on "Assistant:", one opens on "User:"). Not one is
        labelled `[User]`. And only ONE of the eight has its grapefruit
        mention inside the unlabelled opening turn -- the other seven already
        show "Assistant:" before it, and the answer was still 4. The label
        helps a reader see who is talking; it does not make a reader count only
        the user.
        """
        import re
        excerpts = REAL["c4a1ceb8_grapefruit"]
        assert len(excerpts) == 8
        labels = [rm._opening_speaker([e], 0) for e in excerpts]
        assert labels.count("Assistant") == 5
        assert labels.count(None) == 3
        assert "User" not in labels

        hidden = [
            e for e in excerpts
            if not rm._SPEAKER_MARKER.search(e[:e.index("rapefruit")])
        ]
        assert len(hidden) == 1

    def test_a_passage_that_already_opens_on_a_marker_is_left_alone(self):
        chicago = REAL["830ce83f_chicago"][0]
        assert rm._opening_speaker(["User: " + chicago], 0) is None
        assert rm._opening_speaker(["Assistant: " + chicago], 0) is None

    def test_the_first_marker_names_the_other_speaker(self):
        # Taken from a real excerpt: an assistant list continuing into the
        # user's next turn, so the top of the window is the assistant.
        mid = REAL["c4a1ceb8_grapefruit"][0]
        assert "User:" in mid and not mid.lstrip().startswith("User:")
        assert rm._opening_speaker([mid], 0) == "Assistant"

    def test_a_window_with_no_marker_keeps_the_speaker_of_the_one_before(self):
        a = "User: I made a cocktail. Assistant: Here are some ideas."
        b = REAL["830ce83f_chicago"][0].split("User:")[0]
        assert "User:" not in b and "Assistant:" not in b
        assert rm._opening_speaker([a, b], 1) == "Assistant"

    def test_nothing_before_it_and_no_marker_means_no_label(self):
        plain = "Simmer for twenty minutes, then strain."
        assert rm._opening_speaker([plain], 0) is None

    @pytest.mark.parametrize("line", ["Pros: cheaper", "Ingredients: 1 cup honey",
                                      "Instructions: stir", "Cons: slower"])
    def test_a_colon_is_not_a_speaker(self, line):
        """93 "Pros:", 51 "Ingredients:" and 47 "Instructions:" in the excerpts
        served on the 293. None of them is somebody talking."""
        assert rm._opening_speaker([line + " and more text here."], 0) is None

    def test_out_of_range_index_is_not_an_error(self):
        assert rm._opening_speaker([], 0) is None
        assert rm._opening_speaker(["x"], 5) is None


class TestTheLabel:
    DOC = "longterm:answer_830ce83f_1"
    SAID = datetime(2023, 5, 21, 9, 14, tzinfo=timezone.utc)
    OTHER = datetime(2023, 3, 4, tzinfo=timezone.utc)

    def test_the_date_comes_from_the_document_the_chunk_came_from(self):
        got = rm._excerpt_label(
            "User: She moved to Chicago.", (self.DOC, 0), {self.DOC: ["c"]},
            {self.DOC: self.SAID}, None, True, False,
        )
        assert got == "(said May 21, 2023) User: She moved to Chicago."

    def test_the_facts_own_date_is_only_the_fallback(self):
        """What shipped before used the fact's date. It stays as the fallback for
        an excerpt whose document we could not date, so this can only add."""
        got = rm._excerpt_label("text", ("unknown-doc", 0), {}, {}, self.OTHER, True, False)
        assert got == "(said Mar 4, 2023) text"

    def test_no_date_anywhere_leaves_the_text_alone(self):
        assert rm._excerpt_label("text", None, {}, {}, None, True, True) == "text"

    def test_date_then_speaker(self):
        mid = REAL["c4a1ceb8_grapefruit"][0]
        got = rm._excerpt_label(
            mid, (self.DOC, 2), {self.DOC: ["a", "b", mid]},
            {self.DOC: self.SAID}, None, True, True,
        )
        assert got.startswith("(said May 21, 2023) [Assistant] ")
        assert got.endswith(mid)

    def test_speaker_alone(self):
        mid = REAL["c4a1ceb8_grapefruit"][0]
        got = rm._excerpt_label(mid, (self.DOC, 0), {self.DOC: [mid]}, {}, None, False, True)
        assert got == "[Assistant] " + mid

    def test_both_off_changes_nothing(self):
        got = rm._excerpt_label("text", (self.DOC, 0), {self.DOC: ["text"]},
                                {self.DOC: self.SAID}, self.OTHER, False, False)
        assert got == "text"


class TestTheDocumentDateQuery:
    class _PG:
        def __init__(self, rows=None, boom=False):
            self.rows, self.boom, self.sql, self.params = rows or [], boom, None, None

        async def fetch_all(self, sql, params):
            self.sql, self.params = sql, params
            if self.boom:
                raise RuntimeError("no connection")
            return self.rows

    def _manager(self, pg):
        mgr = rm.RetrievalManager.__new__(rm.RetrievalManager)
        mgr.postgres = pg
        return mgr

    @pytest.mark.asyncio
    async def test_it_groups_documents_to_dates(self):
        said = datetime(2023, 5, 21, 9, 14, tzinfo=timezone.utc)
        pg = self._PG([{"source_document_id": "longterm:x", "said": said}])
        got = await self._manager(pg)._document_said_dates(["longterm:x"])
        assert got == {"longterm:x": said}
        assert "MIN(document_created_at)" in pg.sql
        assert "CAST(:ids AS text[])" in pg.sql
        assert pg.params == {"ids": ["longterm:x"]}

    @pytest.mark.asyncio
    async def test_nothing_to_ask_asks_nothing(self):
        pg = self._PG()
        assert await self._manager(pg)._document_said_dates([]) == {}
        assert pg.sql is None

    @pytest.mark.asyncio
    async def test_no_database_is_not_an_error(self):
        assert await self._manager(None)._document_said_dates(["longterm:x"]) == {}

    @pytest.mark.asyncio
    async def test_a_failed_lookup_costs_the_label_and_nothing_else(self):
        pg = self._PG(boom=True)
        assert await self._manager(pg)._document_said_dates(["longterm:x"]) == {}

    @pytest.mark.asyncio
    async def test_a_null_date_is_dropped_not_stored_as_none(self):
        pg = self._PG([{"source_document_id": "longterm:x", "said": None}])
        assert await self._manager(pg)._document_said_dates(["longterm:x"]) == {}


def test_the_free_half_of_the_map_comes_from_the_items_themselves():
    """A single-document item's `said_at` IS its document's date, so the query
    runs only for documents that reach the bundle solely as a contributor to a
    merged memory. If that stops being true the query gets asked for every
    document on every fetch."""
    import inspect
    src = inspect.getsource(rm.RetrievalManager._enrich_with_chunk_evidence)
    i = src.index("doc_said: Dict[str, datetime] = {}")
    block = src[i:i + 900]
    assert 'getattr(_item, "said_at", None)' in block
    assert "len(_docs) == 1" in block
    assert "_missing = [d for d in unique_doc_ids if d not in doc_said]" in block
    assert "if _missing:" in block


class TestEndToEndThroughTheEnrichment:
    """Drive the real method, with real chunk text, and read the served string.

    The helpers above are unit-tested; this is the wiring: the chunk-to-document
    mapping that the union used to throw away, the free half of the date map
    taken from the items, and the order of the two prefixes.
    """

    @staticmethod
    def _manager(doc_to_chunks):
        from unittest.mock import AsyncMock
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

        vector = AsyncMock()

        async def _batched(scope, source_document_ids, limit_per_document=50):
            return {
                d: [{"content": c, "metadata": {"chunk_index": i}}
                    for i, c in enumerate(doc_to_chunks.get(d, []))]
                for d in source_document_ids
            }

        vector.get_chunks_by_document_ids = AsyncMock(side_effect=_batched)
        pg = AsyncMock()
        pg.fetch_all = AsyncMock(return_value=[])
        return RetrievalManager(vector_provider=vector, postgres_provider=pg)

    @staticmethod
    def _bundle(item):
        from synap.cloud.agents.models.bundles import ContextBundle
        return ContextBundle(bundle_id="b", decision_id="d", items_by_type={"facts": [item]})

    @staticmethod
    def _scope():
        from uuid import uuid4
        from synap.cloud.shared.scope import ScopeHierarchy
        return ScopeHierarchy(
            instance_id=str(uuid4()), client_id=str(uuid4()),
            customer_id=str(uuid4()), user_id=str(uuid4()),
            internal_user_id="internal-user-1",
        )

    @pytest.mark.asyncio
    async def test_the_served_excerpt_says_when_and_who(self, monkeypatch):
        from uuid import uuid4
        from synap.cloud.agents.models.bundles import ContextItem

        monkeypatch.setenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", "on")
        monkeypatch.setenv("SYNAP_EVIDENCE_SPEAKER_LABEL", "on")
        monkeypatch.delenv("SYNAP_EVIDENCE_MIN_SUPPORT", raising=False)

        doc = "longterm:answer_c4a1ceb8_1"
        real = REAL["c4a1ceb8_grapefruit"][0]
        item = ContextItem(
            item_id=str(uuid4()),
            content="The user infuses cucumber into cocktails.",
            context_type="facts",
            source="vector",
            source_document_id=doc,
            # What _event_fields sets on every item: the document's own date.
            said_at=datetime(2023, 5, 21, 9, 14, tzinfo=timezone.utc),
        )
        bundle = self._bundle(item)
        await self._manager({doc: [real]})._enrich_with_chunk_evidence(
            bundle=bundle,
            search_queries=["cucumber infusion"],
            scope=self._scope(),
            max_items_to_enrich=3,
        )
        served = item.source_evidence
        assert served and served[0].startswith("(said May 21, 2023) [Assistant] ")
        assert served[0].endswith(real)

    @pytest.mark.asyncio
    async def test_both_flags_off_serves_the_bare_chunk(self, monkeypatch):
        from uuid import uuid4
        from synap.cloud.agents.models.bundles import ContextItem

        monkeypatch.delenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", raising=False)
        monkeypatch.delenv("SYNAP_EVIDENCE_SPEAKER_LABEL", raising=False)
        monkeypatch.delenv("SYNAP_RETRIEVAL_DATED_EVIDENCE", raising=False)
        monkeypatch.delenv("SYNAP_EVIDENCE_MIN_SUPPORT", raising=False)

        doc = "longterm:answer_c4a1ceb8_1"
        real = REAL["c4a1ceb8_grapefruit"][0]
        item = ContextItem(
            item_id=str(uuid4()),
            content="The user infuses cucumber into cocktails.",
            context_type="facts",
            source="vector",
            source_document_id=doc,
            said_at=datetime(2023, 5, 21, 9, 14, tzinfo=timezone.utc),
        )
        bundle = self._bundle(item)
        await self._manager({doc: [real]})._enrich_with_chunk_evidence(
            bundle=bundle,
            search_queries=["cucumber infusion"],
            scope=self._scope(),
            max_items_to_enrich=3,
        )
        assert item.source_evidence == [real]

    @pytest.mark.asyncio
    async def test_a_merged_memory_dates_each_excerpt_from_its_own_document(self, monkeypatch):
        """The case the old rule refused to label at all: two contributor
        documents, so the fact's own date could not stand for both. Each
        excerpt now carries the date of the document it came from."""
        from uuid import uuid4
        from unittest.mock import AsyncMock
        from synap.cloud.agents.models.bundles import ContextItem

        monkeypatch.setenv("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", "on")
        monkeypatch.delenv("SYNAP_EVIDENCE_SPEAKER_LABEL", raising=False)
        monkeypatch.delenv("SYNAP_EVIDENCE_MIN_SUPPORT", raising=False)

        merged_id = str(uuid4())
        item = ContextItem(
            item_id=merged_id,
            content="The user does yoga three times a week.",
            context_type="facts",
            source="vector",
            source_document_id=None,
        )
        manager = self._manager({
            "doc-a": ["User: I started yoga twice a week last spring."],
            "doc-b": ["User: Now I'm doing yoga three times a week, it's been great."],
        })
        # The join table says two contributors; the date query covers both,
        # because no single-document item in the bundle can supply them.
        manager.postgres.fetch_all = AsyncMock(side_effect=[
            [{"memory_id": merged_id, "source_document_id": "doc-a"},
             {"memory_id": merged_id, "source_document_id": "doc-b"}],
            [{"source_document_id": "doc-a", "said": datetime(2023, 3, 4, tzinfo=timezone.utc)},
             {"source_document_id": "doc-b", "said": datetime(2023, 9, 1, tzinfo=timezone.utc)}],
        ])
        bundle = self._bundle(item)
        await manager._enrich_with_chunk_evidence(
            bundle=bundle,
            search_queries=["how often yoga"],
            scope=self._scope(),
            max_items_to_enrich=3,
            max_chunks_per_item=2,
        )
        served = item.source_evidence or []
        assert served
        for e in served:
            if "twice a week" in e:
                assert e.startswith("(said Mar 4, 2023) ")
            if "three times a week" in e:
                assert e.startswith("(said Sep 1, 2023) ")
