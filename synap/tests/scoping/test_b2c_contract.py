"""The B2C single-identifier contract: send user_id, nothing else.

Two halves, and the second is the one that will still be here in a year.

Behaviour: the policy switch does what it says, and `warn` genuinely changes
nothing, because a rollout state that quietly alters behaviour is not a rollout
state.

Call-site guard: every SDK-reachable route that accepts a `customer_id` calls
the check. The original defect was six functions each deciding the same
question for themselves, so a test that pins three routes and misses the fourth
would repeat exactly the mistake it is meant to prevent.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from synap.cloud.shared import b2c_contract as C

INSTANCE = "inst_test"
B2C_MODE = "equals_customer"


@pytest.fixture
def as_b2c(monkeypatch):
    async def _iso(_instance_id):
        return B2C_MODE
    monkeypatch.setattr(C, "load_isolation", _iso)


@pytest.fixture
def as_b2b(monkeypatch):
    async def _iso(_instance_id):
        return "strict"
    monkeypatch.setattr(C, "load_isolation", _iso)


def _policy(monkeypatch, value):
    monkeypatch.setenv("SYNAP_B2C_ID_POLICY", value)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_default_policy_is_warn_not_enforce(monkeypatch):
    """An unset variable must not reject live traffic. Shipping the code and
    turning it on are two decisions, and only one of them can break a client."""
    monkeypatch.delenv("SYNAP_B2C_ID_POLICY", raising=False)
    assert C.policy() == C.POLICY_WARN


def test_a_typo_falls_back_to_warn_never_enforce(monkeypatch, caplog):
    """`enfroce` must not silently mean `off`, and must never mean `enforce`."""
    _policy(monkeypatch, "enfroce")
    with caplog.at_level(logging.WARNING):
        assert C.policy() == C.POLICY_WARN
    assert any("is not one of" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_warn_logs_but_does_not_raise(monkeypatch, as_b2c, caplog):
    _policy(monkeypatch, C.POLICY_WARN)
    with caplog.at_level(logging.WARNING):
        await C.check_customer_id(INSTANCE, "acme", where="probe", user_id="alice")
    assert any("b2c_customer_id_sent" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_enforce_raises_with_a_message_that_says_what_to_do(monkeypatch, as_b2c):
    _policy(monkeypatch, C.POLICY_ENFORCE)
    with pytest.raises(C.B2CCustomerIdRejected) as exc:
        await C.check_customer_id(INSTANCE, "acme", where="probe", user_id="alice")
    msg = str(exc.value)
    assert "Send user_id only" in msg
    assert "acme" in msg, "the rejected value must be echoed or it cannot be traced"


@pytest.mark.asyncio
async def test_off_is_a_true_no_op(monkeypatch, as_b2c, caplog):
    _policy(monkeypatch, C.POLICY_OFF)
    with caplog.at_level(logging.WARNING):
        await C.check_customer_id(INSTANCE, "acme", where="probe")
    assert not caplog.records


# ---------------------------------------------------------------------------
# Who it applies to
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [C.POLICY_WARN, C.POLICY_ENFORCE])
async def test_b2b_is_never_touched(monkeypatch, as_b2b, caplog, mode):
    """customer_id is REQUIRED on B2B. Rejecting it there would break every
    B2B client on the platform, which is the opposite failure."""
    _policy(monkeypatch, mode)
    with caplog.at_level(logging.WARNING):
        await C.check_customer_id(INSTANCE, "acme", where="probe", user_id="alice")
    assert not caplog.records


@pytest.mark.asyncio
async def test_no_customer_id_is_the_correct_shape_and_stays_silent(monkeypatch, as_b2c, caplog):
    _policy(monkeypatch, C.POLICY_ENFORCE)
    with caplog.at_level(logging.WARNING):
        await C.check_customer_id(INSTANCE, None, where="probe", user_id="alice")
    assert not caplog.records


@pytest.mark.asyncio
async def test_an_unreadable_maca_mode_does_nothing(monkeypatch, caplog):
    """`load_isolation` returns None when it cannot read the config. Guessing
    B2C there would reject a B2B client's mandatory field on a transient DB
    blip, so not knowing has to mean not acting."""
    async def _none(_instance_id):
        return None
    monkeypatch.setattr(C, "load_isolation", _none)
    _policy(monkeypatch, C.POLICY_ENFORCE)
    with caplog.at_level(logging.WARNING):
        await C.check_customer_id(INSTANCE, "acme", where="probe")
    assert not caplog.records


@pytest.mark.asyncio
async def test_customer_scoped_route_is_refused_on_b2c(monkeypatch, as_b2c):
    """Refused on B2C. Note this is NOT free: a client measurably got real facts
    back from this route while their user fetch returned empty. It
    is refused because what it serves is filed under a key their user reads can
    never reach, so serving it hides the split. See `reject_customer_route`."""
    _policy(monkeypatch, C.POLICY_ENFORCE)
    with pytest.raises(C.B2CCustomerIdRejected):
        await C.reject_customer_route(INSTANCE, where="POST /v1/context/customer/fetch")


@pytest.mark.asyncio
async def test_customer_scoped_route_is_fine_on_b2b(monkeypatch, as_b2b):
    _policy(monkeypatch, C.POLICY_ENFORCE)
    await C.reject_customer_route(INSTANCE, where="POST /v1/context/customer/fetch")


# ---------------------------------------------------------------------------
# Call-site guard
# ---------------------------------------------------------------------------

_ROUTES = Path(__file__).resolve().parents[2] / "cloud/application/sdk_api"

# Files whose routes are reachable with an SDK key. Dashboard-only routes are
# out of scope: they are driven by our own UI, which hides the field instead.
_SDK_ROUTE_FILES = [
    "sdk_memories.py",
    "sdk_context.py",
    "sdk_conversations.py",
    "sdk_users.py",
]

_CHECKERS = {"check_customer_id", "_b2c_check", "reject_customer_route",
             "_b2c_reject_customer_route"}


def _models_declaring_customer_id(tree):
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for b in node.body:
            if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                if b.target.id == "customer_id":
                    out.add(node.name)
    return out


def _route_functions(tree, models):
    """(name, decorator_path) for handlers that can receive a customer_id."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        path = None
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "router"):
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    path = f"{dec.func.attr.upper()} {dec.args[0].value}"
        if path is None:
            continue
        takes = False
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            if a.arg == "customer_id":
                takes = True
            ann = ast.unparse(a.annotation) if a.annotation else ""
            if any(m == ann or m in ann for m in models):
                takes = True
        if takes:
            found.append((node, path))
    return found


def _calls_a_checker(fn) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name in _CHECKERS:
                return True
    return False


def test_every_sdk_route_taking_a_customer_id_applies_the_contract():
    """A route added next year that accepts a customer_id and forgets the check
    fails here, rather than quietly exempting itself on every B2C instance."""
    missing = []
    for fn_name in _SDK_ROUTE_FILES:
        path = _ROUTES / fn_name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        models = _models_declaring_customer_id(tree)
        for fn, route in _route_functions(tree, models):
            if not _calls_a_checker(fn):
                missing.append(f"{fn_name}:{fn.lineno} {route} -> {fn.name}()")
    assert not missing, (
        "these SDK-reachable routes accept a customer_id but never apply the "
        "B2C contract:\n  " + "\n  ".join(sorted(missing))
    )


@pytest.mark.asyncio
async def test_a_refused_route_does_not_quote_a_placeholder_as_the_value(monkeypatch, as_b2c):
    """The route rejection used to interpolate the literal string
    "<customer-scoped route>" into "Received customer_id=...", putting a
    placeholder in front of a client as if it were the value they sent. The
    message a client reads is the deliverable here, not an implementation
    detail."""
    _policy(monkeypatch, C.POLICY_ENFORCE)
    with pytest.raises(C.B2CCustomerIdRejected) as exc:
        await C.reject_customer_route(INSTANCE, where="POST /v1/context/customer/fetch")
    msg = str(exc.value)
    assert "<customer-scoped route>" not in msg
    assert "Received customer_id" not in msg
    assert "not available on a B2C instance" in msg
    assert "/v1/context/user/fetch" in msg, "must name what to call instead"
