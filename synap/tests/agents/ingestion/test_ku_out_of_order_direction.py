"""SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE: which of two conflicting memories wins.

The direction of a SUPERSEDE was set by arrival order. The memories of the
document being ingested are the "new" side and whatever the vector search
returns is the "existing" side, so a conversation ingested out of order retired
a statement the user made LATER.

Everything in this file is the real pair from the LongMemEval question
830ce83f, with the real contents and the real session dates, as stored:

    said 2023-05-21  "The user has a friend named Rachel who moved to Chicago."
    said 2023-05-26  "The user's friend Rachel recently moved back to the suburbs."

The 26 May session was ingested first, so Chicago arrived as the "new" value
and retired the suburbs. The changelog reason recorded reads "the new
memory updating the location from the suburbs to Chicago", and retrieval hides
superseded rows, so "Where did Rachel move to after her recent relocation?"
(gold: the suburbs) could only be answered Chicago, in all six scored cells.

Measured on the benchmark instance: 54 of 507 supersessions
(10.7%) kept the earlier statement. Among them "leads a team of five
engineers" lost to "a team of 4 engineers", "a to-watch list with 25 titles"
lost to "20 titles" five times, and "is a senior UX designer" lost to "is a
sales manager". Asked again in the right time order, gpt-4o retires the
earlier statement on 21 of the 54 and declines on 33; in none of them does the
later statement stay hidden.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
    _parse_document_date,
)

SAID_21_MAY = datetime(2023, 5, 21, 9, 14, tzinfo=timezone.utc)
SAID_26_MAY = datetime(2023, 5, 26, 3, 55, tzinfo=timezone.utc)
CHICAGO = "The user has a friend named Rachel who moved to Chicago."
SUBURBS = "The user's friend Rachel recently moved back to the suburbs."


def _scope() -> Scope:
    return Scope(client_id=uuid4(), instance_id=uuid4(),
                 customer_id=uuid4(), user_id=uuid4())


def _arriving(content: str, dct, assertion_source="user") -> ExtractedMemory:
    """A memory from the document being ingested now."""
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=MemoryType.FACTS,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-arriving",
        assertion_source=assertion_source,
        document_created_at=dct,
    )


def _stored(content: str, dct, assertion_source="user", score: float = 0.9):
    """The vector hit, with the metadata ChromaDB actually carries: the date
    is an ISO string there, not a datetime."""
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[{
        "id": str(uuid4()),
        "content": content,
        "score": score,
        "metadata": {
            "memory_type": "facts",
            "assertion_source": assertion_source,
            "document_created_at": dct.isoformat() if dct else None,
        },
    }]])
    return vec


def _llm(*verdicts: str):
    """One response per classification call, in order: the first is the pair as
    it arrives, the second is the same pair with the sides swapped."""
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=[
        json.dumps({"classification": v, "reason": f"stub {v}", "confidence": 0.95})
        for v in verdicts
    ])
    return llm


async def _run(arriving, vector, llm):
    detector = KnowledgeUpdateDetector(vector=vector, llm_client=llm)
    return await detector.detect_updates(
        memories=[arriving], memory_embeddings=[[0.1] * 8], scope=_scope())


class TestParseDocumentDate:
    def test_reads_the_iso_string_chroma_stores(self):
        assert _parse_document_date("2023-05-26T03:55:00+00:00") == SAID_26_MAY

    def test_reads_a_datetime_unchanged(self):
        assert _parse_document_date(SAID_26_MAY) == SAID_26_MAY

    def test_a_naive_value_is_utc(self):
        assert _parse_document_date("2023-05-26T03:55:00") == SAID_26_MAY

    def test_z_suffix(self):
        assert _parse_document_date("2023-05-26T03:55:00Z") == SAID_26_MAY

    def test_nothing_unparseable_becomes_a_date(self):
        assert _parse_document_date(None) is None
        assert _parse_document_date("") is None
        assert _parse_document_date("last Tuesday") is None


@pytest.mark.asyncio
class TestOutOfOrderDirection:
    async def test_flag_off_keeps_todays_behaviour_bug_included(self, monkeypatch):
        """With the flag unset this is exactly the earlier behavior:
        the arriving 21 May statement retires the stored 26 May one."""
        monkeypatch.delenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", raising=False)
        result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                            _stored(SUBURBS, SAID_26_MAY), _llm("SUPERSEDE"))
        assert len(result.supersessions) == 1
        assert result.supersessions[0].old_memory_content == SUBURBS
        assert result.skips == []

    async def test_arriving_older_statement_is_the_one_retired(self, monkeypatch):
        """The fix. Re-asked in the right order the model still says SUPERSEDE,
        so the 21 May statement is retired and the 26 May one keeps serving."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                            _stored(SUBURBS, SAID_26_MAY),
                            _llm("SUPERSEDE", "SUPERSEDE"))
        assert result.supersessions == []
        assert len(result.skips) == 1
        skip = result.skips[0]
        assert skip.new_memory_content == CHICAGO          # retired
        assert skip.existing_memory_content == SUBURBS     # survives
        assert skip.action_label == "SUPERSEDE"            # not a duplicate drop
        assert "out-of-order ingest" in skip.reason
        assert "2023-05-21" in skip.reason and "2023-05-26" in skip.reason

    async def test_second_opinion_declining_retires_nothing(self, monkeypatch):
        """Asked in the right order the model may refuse, because SUPERSEDE
        needs the later statement to preserve every detail the earlier one
        had. Then both memories keep serving: never the old direction."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        for second in ("ENRICH", "UNRELATED", "NONE"):
            result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                                _stored(SUBURBS, SAID_26_MAY),
                                _llm("SUPERSEDE", second))
            assert result.supersessions == [], second
            assert result.skips == [], second

    async def test_a_dead_second_call_retires_nothing(self, monkeypatch):
        """_classify_pair returns None when the call fails or the answer is not
        a known verdict. Nothing is retired in either direction."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=[
            json.dumps({"classification": "SUPERSEDE", "reason": "stub"}),
            "not json at all",
        ])
        result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                            _stored(SUBURBS, SAID_26_MAY), llm)
        assert result.supersessions == []
        assert result.skips == []

    async def test_arriving_later_statement_supersedes_as_before(self, monkeypatch):
        """The normal case, and the common one: the document being ingested is
        the later conversation. One classification call, no second opinion."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        llm = _llm("SUPERSEDE")
        result = await _run(_arriving(SUBURBS, SAID_26_MAY),
                            _stored(CHICAGO, SAID_21_MAY), llm)
        assert len(result.supersessions) == 1
        assert result.supersessions[0].old_memory_content == CHICAGO
        assert llm.complete.await_count == 1

    async def test_same_conversation_is_left_alone(self, monkeypatch):
        """Two memories from one document: 3,561 of 4,479 pairs on the
        benchmark instance. There is no order to read from the dates, and the
        sibling path upstream already orders those by event_start."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        llm = _llm("SUPERSEDE")
        result = await _run(_arriving(CHICAGO, SAID_26_MAY),
                            _stored(SUBURBS, SAID_26_MAY), llm)
        assert len(result.supersessions) == 1
        assert llm.complete.await_count == 1

    async def test_a_missing_date_never_invents_an_order(self, monkeypatch):
        """Legacy rows carry no document_created_at. Unknown is unknown, so the
        guard does not fire and behaviour is today's."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                            _stored(SUBURBS, None), _llm("SUPERSEDE"))
        assert len(result.supersessions) == 1
        result = await _run(_arriving(CHICAGO, None),
                            _stored(SUBURBS, SAID_26_MAY), _llm("SUPERSEDE"))
        assert len(result.supersessions) == 1

    async def test_a_verdict_that_is_not_supersede_is_untouched(self, monkeypatch):
        """ENRICH and UNRELATED keep both memories already, so the guard has
        no work and costs no second call."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        for verdict in ("ENRICH", "UNRELATED"):
            llm = _llm(verdict)
            result = await _run(_arriving(CHICAGO, SAID_21_MAY),
                                _stored(SUBURBS, SAID_26_MAY), llm)
            assert result.supersessions == []
            assert result.skips == []
            assert llm.complete.await_count == 1, verdict

    async def test_echo_guard_still_protects_the_user_when_swapped(self, monkeypatch):
        """The stored later statement is the assistant's and the arriving
        earlier one is the user's: the assistant's restatement may not retire
        what the user said, in this direction either."""
        monkeypatch.setenv("SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE", "true")
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _arriving(CHICAGO, SAID_21_MAY, assertion_source="user"),
            _stored(SUBURBS, SAID_26_MAY, assertion_source="assistant"),
            _llm("SUPERSEDE", "SUPERSEDE"))
        assert result.supersessions == []
        assert result.skips == []
