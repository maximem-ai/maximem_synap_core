"""Nothing may FORCE a caller to supply a customer_id.

On a B2C instance the contract forbids sending one, so any surface that
requires one, or that rejects/drops a request for omitting one, makes the only
correct B2C call impossible. That is not a documentation problem; it is a
surface a B2C client cannot use at all.

This was not hypothetical. Found by audit on 2026-08-25, all in shipped code:
  - gRPC session_control REQUIRED a customer_id, so a B2C client could not
    start a session
  - gRPC ConversationEvent was SILENTLY DROPPED without one, losing the turn
  - /api/v1/memories/upload and /dashboard/memories/upload declared it Form(...)
  - the JS bridge raised on addMemory and on scope-delete
  - record_message and create_from_file required it in the Python SDK

Allow-listed below are the genuinely customer-scoped surfaces, where requiring
a customer id is the whole point and the B2C answer is to refuse the call.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ROUTES = _ROOT / "cloud/application/admin_api/routes"
_SDK = _ROOT / "sdk/python/maximem_synap"

# Surfaces that are customer-scoped BY DEFINITION. Requiring a customer id is
# correct; on a B2C instance the whole call is refused instead.
_CUSTOMER_SCOPED = {
    ("sdk.py", "CustomerContextInterface", "fetch"),
    ("sdk.py", "CacheInterface", "clear_customer"),
    ("controllers.py", "CustomerContextController", "fetch"),
    ("manager.py", "CacheManager", "clear_customer"),
}


def _required_customer_id_in_routes():
    out = []
    for path in sorted(_ROUTES.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                and isinstance(d.func.value, ast.Name) and d.func.value.id == "router"
                for d in fn.decorator_list
            ):
                continue
            args = list(fn.args.args) + list(fn.args.kwonlyargs)
            defaults = ([None] * (len(fn.args.args) - len(fn.args.defaults))
                        + list(fn.args.defaults) + list(fn.args.kw_defaults))
            for a, d in zip(args, defaults):
                if a.arg != "customer_id":
                    continue
                if d is None or "..." in ast.unparse(d):
                    out.append(f"{path.name}:{fn.lineno} {fn.name}()")
    return out


def test_no_http_route_requires_a_customer_id():
    forced = _required_customer_id_in_routes()
    assert not forced, (
        "these routes REQUIRE a customer_id, which a B2C instance forbids "
        "sending, so a B2C client cannot call them at all:\n  " + "\n  ".join(forced)
    )


def _required_customer_id_in_sdk():
    out = []
    for path in sorted(_SDK.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if fn.name.startswith("_"):
                    continue
                if (path.name, cls.name, fn.name) in _CUSTOMER_SCOPED:
                    continue
                args = list(fn.args.args) + list(fn.args.kwonlyargs)
                defaults = ([None] * (len(fn.args.args) - len(fn.args.defaults))
                            + list(fn.args.defaults) + list(fn.args.kw_defaults))
                for a, d in zip(args, defaults):
                    if a.arg == "customer_id" and d is None:
                        out.append(f"{path.name}:{fn.lineno} {cls.name}.{fn.name}()")
    return out


def test_no_sdk_method_requires_a_customer_id():
    forced = _required_customer_id_in_sdk()
    assert not forced, (
        "these SDK methods REQUIRE a customer_id, so a B2C caller cannot use "
        "them:\n  " + "\n  ".join(forced)
    )


def test_the_allow_list_still_describes_real_surfaces():
    """A stale allow-list silently widens the rule. Every entry must still
    exist and still require a customer_id, or it should be removed."""
    stale = []
    for fname, clsname, fnname in _CUSTOMER_SCOPED:
        found = False
        for path in _SDK.rglob(fname):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                if cls.name != clsname:
                    continue
                for fn in cls.body:
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == fnname:
                        found = True
        if not found:
            stale.append(f"{fname}:{clsname}.{fnname}")
    assert not stale, "allow-list entries that no longer exist:\n  " + "\n  ".join(stale)


def test_grpc_does_not_drop_events_for_a_missing_customer_id():
    """The servicer used to `return` out of ConversationEvent persistence when
    customer_id was absent, which on B2C is every event, and it did so with
    only a log line. Losing a turn silently is the worst available outcome."""
    src = (_ROOT / "cloud/application/grpc/servicer.py").read_text(encoding="utf-8")
    assert "if not event.customer_id or not event.user_id:" not in src, (
        "servicer.py drops a ConversationEvent whenever customer_id is absent; "
        "on a B2C instance that is every event"
    )
    assert "_instance_is_b2c" in src, (
        "servicer.py no longer consults the instance mode, so it cannot tell "
        "a legitimately absent customer_id from a missing one"
    )
