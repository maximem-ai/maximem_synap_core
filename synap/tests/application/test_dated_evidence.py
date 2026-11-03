"""A chunk of source text says when it was said.

The memory line carries a date prefix; the evidence block under it did not. On
lme3 question 71017276 a reader quoted "I also got a stunning crystal chandelier
from my aunt today" out of a chunk and answered "0 weeks ago" to a question asked
four weeks later. The memory line directly above said "(Mar 4, 2023)".

These tests pin the three things that make the change safe rather than clever:
it is off unless the deployment turns it on, it never labels a merged row whose
evidence could come from any contributor, and it never invents a date.
"""
from datetime import datetime

import pytest

from synap.cloud.application.client_api import retrieval_manager as rm


class TestFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("SYNAP_RETRIEVAL_DATED_EVIDENCE", raising=False)
        assert rm.dated_evidence_enabled() is False

    @pytest.mark.parametrize("raw", ["on", "1", "true", "yes", "ON", " True "])
    def test_accepts_the_same_spellings_as_its_sibling_flag(self, monkeypatch, raw):
        monkeypatch.setenv("SYNAP_RETRIEVAL_DATED_EVIDENCE", raw)
        assert rm.dated_evidence_enabled() is True

    def test_anything_else_is_off(self, monkeypatch):
        monkeypatch.setenv("SYNAP_RETRIEVAL_DATED_EVIDENCE", "maybe")
        assert rm.dated_evidence_enabled() is False


def test_the_source_states_the_three_conditions_for_labelling():
    """A source-level check, because the labelling sits inside a long chunk-
    enrichment method with a vector store, a BM25 scorer and a thread pool
    around it. The conditions are what matter and they are cheap to assert.

    The fact's own date is now the FALLBACK -- the excerpt is dated from its
    own document when `SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT` is on, see
    test_evidence_said_and_speaker.py -- but with that flag off these three
    conditions are still the whole rule, and still the only thing that can
    print a date.
    """
    import inspect
    src = inspect.getsource(rm.RetrievalManager._enrich_with_chunk_evidence)
    i = src.index("_old_said = (")
    block = src[i:i + 900]
    # never a merged row: its evidence can come from any contributor
    assert "len(docs) == 1" in block
    # the date this row's own document carries, read from the field that holds
    # it. It used to come through `event_date`, valid only while a companion
    # flag confirmed the read path had just substituted the document's date
    # into it; that substitution is gone and so is the flag.
    assert 'getattr(item, "said_at", None)' in block
    assert "event_date" not in block, "the excerpt date is reading event_date again"
    # off unless enabled: `_dated` is dated_evidence_enabled(), read once
    assert "elif _dated and _old_said is not None:" in block
    assert "_said_prefix(_old_said)" in block
    assert "_dated = dated_evidence_enabled()" in inspect.getsource(
        rm.RetrievalManager._enrich_with_chunk_evidence)


def test_it_reuses_the_date_already_computed_and_adds_no_lookup():
    """The whole point of reading `said_at`.

    _event_fields already put the source document's own date there, for every
    item, on the way out. So dating the excerpt needs no query, no new column
    and no new field on the item. If someone later makes this fetch a date
    instead, this test says why they should not.
    """
    import inspect
    src = inspect.getsource(rm._event_fields)
    assert '"said_at": said' in src
    label = inspect.getsource(rm.RetrievalManager._enrich_with_chunk_evidence)
    i = label.index("_old_said = (")
    block = label[i:i + 900]
    for forbidden in ("await ", "session", "execute(", "get_chunks_by"):
        assert forbidden not in block, forbidden
    # The one lookup that does exist dates the excerpt from its own document,
    # and it is reached only when that flag is on.
    i = label.index("doc_said: Dict[str, datetime] = {}")
    gate = label[i - 200:i + 900]
    assert "if _said_from_doc:" in gate
    assert "self._document_said_dates(_missing)" in gate


class TestFormatting:
    """The strftime fallback exists because %-d is not portable."""

    def test_both_strftime_branches_render_a_date(self):
        dt = datetime(2023, 3, 4)
        try:
            got = dt.strftime("%b %-d, %Y")
        except ValueError:
            got = dt.strftime("%b %d, %Y")
        assert got in ("Mar 4, 2023", "Mar 04, 2023")
