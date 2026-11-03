"""A rung with nothing written to it was logged as a failure on every request.

Each rung is searched in its own collection, and a collection is created on the
first write to that rung. A client who has never written client-level knowledge
has no client-level collection, so that arm of the search raises NotFoundError,
finds nothing, and the read is correct without it.

It was logged at warning on every such read, which buries
the real failures the line exists to surface.

Verified before downgrading rather than assumed, because "this error is
harmless" is exactly the belief that hides a naming bug:

    client WITH 25 client-level memories -> collection exists, count = 25
    client WITH none                     -> NotFoundError

The count matches Postgres exactly, so the name is right and the absence is
data.

⚠ This fix was written once, lost to a `git reset --hard` before it was
committed, and a commit message claiming it shipped reached main. It was caught
by checking the running container rather than the branch. That is the reason
this file exists: so the next loss fails a test instead of a log review.
"""

import ast
import inspect

from synap.cloud.application.client_api import retrieval_manager as R


def _handler_source() -> str:
    src = inspect.getsource(R)
    i = src.find("Vector scope chain sub-task failed")
    assert i != -1, "the sub-task handler is gone"
    return src[max(0, i - 1500) : i + 400]


def test_a_missing_collection_is_not_logged_as_a_failure():
    block = _handler_source()
    assert "NotFoundError" in block, (
        "every rung with nothing written to it still logs a warning, which "
        "buries the real failures this line exists to surface"
    )
    assert "logger.debug" in block


def test_a_real_failure_is_still_a_warning():
    """The point is to separate the two, not to silence both."""
    block = _handler_source()
    assert "logger.warning" in block, (
        "genuine sub-task failures are now silent, which is worse than noisy"
    )


def test_both_branches_continue():
    """Neither case may abort the whole search: the other rungs still have
    results worth returning."""
    tree = ast.parse(inspect.getsource(R))
    fn = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "Vector scope chain sub-task failed" in ast.unparse(n):
                fn = n
                break
    assert fn is not None
    body = ast.unparse(fn)
    i = body.find("Vector scope chain sub-task failed")
    assert "continue" in body[i : i + 400], "a failed sub-task aborts the search"
