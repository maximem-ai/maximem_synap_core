"""The day a conversation happened is the client's to state, or it is empty.

`resolve_dct` used to answer two questions with one value: when did this
conversation happen, and what should extraction count "three months ago" back
from. The second question always needs an answer, so when the client said
nothing the function returned our own ingest clock, labelled as ours. Two fixes
were built on top of that label, one to stop storing the clock and one to stop
serving it, and both left extraction still DATING EVENTS off it.

Measured by ingesting one real benchmark session
(really said on 2023-03-27 05:51) twice through the transcript route:

    turn timestamps, no ended_at  ->  stored 2023-03-27T06:02   (right)
    no dates anywhere             ->  stored 2026-10-10T08:16   (our clock)
                                      on 42 of 42 memories

and, on the same ingest, 8 of 10 served items carried an event date in 2026 for
a conversation from 2023, because "three months ago" had been counted back from
our clock and stored as an ordinary event date.

So there is one question here now. This file is the contract: the client's date
or nothing, and every reader handles nothing.
"""
from datetime import datetime, timezone

from uuid import uuid4

from synap.cloud.agents.ingestion.models.inputs import (
    IngestDocumentRequest,
    Scope,
)
from synap.cloud.agents.ingestion.services import temporal_tagging_service
from synap.cloud.agents.ingestion.services.temporal_tagging_service import (
    resolve_dct,
)

# The real session's own date, and a turn that states nothing about time.
SAID_ON = datetime(2023, 3, 27, 5, 51)
TIMELESS = "I've been dealing with a lingering cough for six weeks now."


def _request(**kw):
    base = dict(
        content=TIMELESS, document_id="doc", request_id="req",
        scope=Scope(client_id=uuid4(), instance_id=uuid4(),
                    customer_id=uuid4(), user_id=uuid4(), scope_level="user"),
    )
    base.update(kw)
    return IngestDocumentRequest(**base)


class TestWhatResolveDctReports:
    def test_a_client_date_is_reported_with_its_source(self):
        dct, source = resolve_dct(_request(document_created_at=SAID_ON))
        assert dct == SAID_ON
        assert source == "client"

    def test_nothing_supplied_means_NOTHING(self):
        """The whole fix. There is no fourth priority to reach."""
        dct, source = resolve_dct(_request())
        assert dct is None
        assert source is None

    def test_a_date_in_the_metadata_is_the_client_s(self):
        ms = int(SAID_ON.replace(tzinfo=timezone.utc).timestamp() * 1000)
        dct, source = resolve_dct(_request(metadata={"timestamp_ms": ms}))
        assert source == "grpc"
        assert dct.year == 2023

    def test_our_ingest_clock_is_never_returned(self):
        """`ingested_at` is set on the request and must not be reachable.

        It was priority 4. The request still carries it, because rows are
        written with it, so the only thing stopping it coming back out of here
        is that the branch is gone.
        """
        request = _request()
        assert request.ingested_at is not None, "the clock is still on the request"
        dct, _ = resolve_dct(request)
        assert dct is None

    def test_the_module_cannot_reach_a_clock_for_this_answer(self):
        """A guard against the fallback growing back in another shape.

        `resolve_dct` must not consult the current time at all. Asserted on the
        source rather than by patching, because the next version of this bug
        will not be spelled `request.ingested_at`.
        """
        import inspect
        # The code only. The docstring and one comment both name
        # `request.ingested_at` on purpose, to say why it is not used.
        src = inspect.getsource(resolve_dct)
        body = "\n".join(
            line for line in src.split('"""', 2)[2].splitlines()
            if not line.strip().startswith("#"))
        for forbidden in ("now(", "utcnow", "ingested_at", "today("):
            assert forbidden not in body, (
                "resolve_dct is reaching for a clock again: %r" % forbidden)

    def test_the_source_label_no_longer_gates_anything(self):
        """`DCT_SOURCE_SERVER` is deleted, not renamed.

        It existed so a caller could tell our clock from the client's date. With
        no clock to tell apart, a caller that still consults it would be reading
        a distinction that no longer exists.
        """
        assert not hasattr(temporal_tagging_service, "DCT_SOURCE_SERVER")


class TestWhatTheWorkerStamps:
    """The worker stamps what the resolver returned, including nothing.

    Asserted on the source because the alternative is standing up stages 1-3
    with a live extractor; the expression is one line and the condition is the
    whole fix.
    """

    def _source(self):
        import inspect

        from synap.cloud.agents.ingestion.workers import stage23_worker
        return inspect.getsource(stage23_worker)

    def test_it_stamps_the_resolved_date_as_it_comes(self):
        src = self._source()
        assert "memory.document_created_at = dct" in src

    def test_there_is_no_source_check_left_to_drift(self):
        src = self._source()
        assert "DCT_SOURCE_SERVER" not in src
        assert '== "server"' not in src, "re-spelled the resolver's old value"
        assert "_said_on" not in src, "the filtered copy is gone"
