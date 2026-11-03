"""A client's own field type has to be looked for everywhere the text lands.

Found by minting three field types through the dashboard route and pushing a
transcript through the published SDK. The asset tag and the policy number were
correctly gone from `memories` and from `message`, and sat in full in
`request_payloads`, which renders on the client's own Requests page.

That was the third place in one day with the same omission: the ingest path
had it, the turn path had it, and the request log I wrote that afternoon had
it too. A feature that must be threaded through N call sites will be missed at
one of them, so these are structural and read the call sites.

The second fix here is quieter and worse to debug. The alias key is scoped by
customer and user; the turn path passed neither, so the same asset tag was
given one label in the memory and a different one in the turn it came from,
and a support person correlating them would read one value as two.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
CAPTURE = ROOT / "cloud/application/middleware/request_payload_capture.py"
TRACKER = ROOT / "cloud/application/conversation/tracker.py"
# The gate itself moved out of the tracker into Cloud's hook, so Core stores a
# turn without importing the privacy folder. The tracker still takes and passes the scope.
TURN_HOOK = ROOT / "cloud/cloud_hooks/application.py"


def _calls(path, attr_prefix):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "").startswith(attr_prefix):
            yield n


class TestTheRequestLogLooksForThem:
    def test_the_gate_call_is_given_recognisers(self):
        calls = list(_calls(CAPTURE, "inspect"))
        assert calls, "the request log no longer runs the gate"
        assert any(
            len(c.args) > 1 and not isinstance(c.args[1], ast.Constant)
            for c in calls
        ), (
            "the request log passes None for the client's own field types, so "
            "an asset tag is removed from the memory and the turn and left in "
            "full on their Requests page"
        )

    def test_it_compiles_them_from_the_policy_it_already_loaded(self):
        src = CAPTURE.read_text()
        assert "recognizers_for" in src
        assert "custom_field_types" in src

    def test_a_bad_custom_pattern_does_not_lose_the_body(self):
        """The shipped types must still run. Degrading to "your own field type
        was not looked for" is acceptable; dropping the body is not."""
        src = CAPTURE.read_text()
        start = src.index("extra = None")
        end = src.index("removed = 0", start)
        assert "except Exception:" in src[start:end]


class TestOneValueGetsOneLabel:
    """The alias key is scoped by customer and user. Omitting them on one
    write path mints a second label for a value the other path already
    aliased."""

    def test_the_turn_path_passes_the_full_scope(self):
        src = TURN_HOOK.read_text()
        start = src.index("doc = await protect(")
        end = src.index(")", src.index("user_id=", start))
        block = src[start:end]
        for field in ("client_id=", "instance_id=", "customer_id=", "user_id="):
            assert field in block, f"protect() called without {field}"

    def test_protect_turn_hands_the_full_scope_to_the_hook(self):
        """The tracker no longer calls `protect` itself; it must give the hook all four ids."""
        tree = ast.parse(TRACKER.read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_protect_turn"
        )
        calls = [
            c for c in ast.walk(fn)
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "acall"
        ]
        assert calls, "_protect_turn no longer calls the conversation_turn_protect hook"
        passed = {kw.arg for kw in calls[0].keywords}
        assert {"client_id", "instance_id", "customer_id", "user_id"} <= passed

    def test_protect_turn_accepts_the_scope(self):
        tree = ast.parse(TRACKER.read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_protect_turn"
        )
        names = {a.arg for a in fn.args.kwonlyargs}
        assert {"client_id", "instance_id", "customer_id", "user_id"} <= names

    @pytest.mark.parametrize("caller", ["record_message", "record_transcript"])
    def test_every_caller_supplies_it(self, caller):
        tree = ast.parse(TRACKER.read_text())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == caller
        )
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "_protect_turn":
                kwargs = {k.arg for k in call.keywords}
                assert "customer_id" in kwargs and "user_id" in kwargs, (
                    f"{caller} protects the turn without the alias scope, so a "
                    "value gets a different label here than in the memory"
                )
                return
        pytest.fail(f"{caller} does not protect its turn at all")

    def test_the_route_supplies_them_to_record_message(self):
        routes = ROOT / "cloud/application/sdk_api/sdk_conversations.py"
        tree = ast.parse(routes.read_text())
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "record_message":
                kwargs = {k.arg for k in call.keywords}
                if "client_id" in kwargs:
                    assert {"customer_id", "user_id"} <= kwargs
                    return
        pytest.fail("no tracker.record_message call found in the route")
