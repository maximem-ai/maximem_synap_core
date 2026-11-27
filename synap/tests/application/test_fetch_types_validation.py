"""Issue #23: an unknown type in ``ContextFetchRequest.types`` must be refused.

A typo'd type used to be dropped on the floor and the fetch silently widened to
every enabled category, so a caller asking for ``["facts"]`` could not tell the
filter was ignored. The request model now refuses anything outside
``'all'`` + the five default categories, the same strict contract ``scope_path``
follows.
"""
import pytest
from pydantic import ValidationError

from synap.cloud.application.sdk_api.sdk_context import (
    DEFAULT_CONTEXT_TYPES,
    ContextFetchRequest,
)


def test_default_types_pass():
    req = ContextFetchRequest()
    assert req.types == ["all"]


@pytest.mark.parametrize("types", [
    ["all"],
    ["facts"],
    ["facts", "preferences", "episodes", "emotions", "temporal_events"],
])
def test_known_types_pass(types):
    assert ContextFetchRequest(types=types).types == types


def test_unknown_type_is_refused():
    with pytest.raises(ValidationError) as exc:
        ContextFetchRequest(types=["fact_typo"])
    msg = str(exc.value)
    assert "fact_typo" in msg
    assert "facts" in msg  # the message names the valid values


def test_mixed_known_and_unknown_is_refused():
    with pytest.raises(ValidationError):
        ContextFetchRequest(types=["facts", "fact_typo"])


def test_empty_types_is_refused():
    with pytest.raises(ValidationError) as exc:
        ContextFetchRequest(types=[])
    assert "at least one" in str(exc.value)


def test_default_list_unchanged():
    assert DEFAULT_CONTEXT_TYPES == [
        "facts", "preferences", "episodes", "emotions", "temporal_events",
    ]
