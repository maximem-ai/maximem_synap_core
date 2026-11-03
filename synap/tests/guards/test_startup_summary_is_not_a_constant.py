"""The startup summary must report what happened, not a fixed list.

`lifespan()` used to end with a hardcoded string naming twenty services as
started. Every subsystem it named sits in a `try/except Exception` that degrades
gracefully, so any of them could fail and the summary still announced it.

A boot once printed `entity_resolution` in that list on the same
boot where entity resolution had failed with an UnboundLocalError and logged
itself disabled. The one line an operator reads to confirm a clean boot was a
constant, which is exactly what let a dead subsystem sit unnoticed.
"""
from __future__ import annotations

import ast
import pathlib

MAIN = (
    pathlib.Path(__file__).resolve().parents[3]
    / "synap" / "cloud" / "application" / "main.py"
)


def test_the_startup_summary_names_no_services_in_a_literal():
    src = MAIN.read_text(encoding="utf-8")
    assert "Application startup complete: services=[" not in src, (
        "the startup summary is a hardcoded list of service names again. It "
        "announces subsystems as started without checking, which is how a dead "
        "entity-resolution service went unnoticed in production."
    )
    assert "Application startup complete" in src, "the summary should still exist"


def test_success_is_recorded_as_well_as_logged():
    """`_ok` has to do both. If it only logged, the summary would count zero and
    the guard above would pass while telling the operator nothing."""
    from synap.cloud.application import main as m

    before = len(m._started_services)
    m._ok("Test worker started")
    assert m._started_services[-1] == "Test worker started"
    m._ok("gRPC streaming server started on port %d", 50051)
    assert m._started_services[-1] == "gRPC streaming server started on port 50051"
    assert len(m._started_services) == before + 2
    del m._started_services[before:]


def test_a_degraded_subsystem_is_recorded_with_its_exception_type():
    from synap.cloud.application import main as m

    before = len(m._degraded_services)
    m._degraded("entity_resolution", UnboundLocalError("boom"))
    assert m._degraded_services[-1] == "entity_resolution: UnboundLocalError"
    del m._degraded_services[before:]


def test_entity_resolution_records_itself_degraded_and_stops_guessing_the_cause():
    """The handler asserted a cause it had never checked. Two regressions to
    lock: it must record the degradation, and it must not claim to know why."""
    src = MAIN.read_text(encoding="utf-8")
    assert '_degraded("entity_resolution"' in src
    assert "tables may not exist yet" not in src, (
        "the handler is guessing at a cause again. It sent everyone who "
        "investigated to look at the database for over a year."
    )


def test_every_ok_call_is_a_real_started_or_initialized_line():
    """`_ok` must not be used for a 'disabled' line, which would record a
    subsystem that was deliberately switched off as having started."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_ok"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and "disabled" in node.args[0].value.lower()
        ):
            bad.append(node.args[0].value)
    assert not bad, f"_ok used for a 'disabled' message: {bad}"


def test_the_scope_mapping_fallback_is_not_silent():
    """`sdk_context._build_scope` falls back from the IdMapper-backed scope
    builder to one that registers nothing. Both derive the same uuids, so the
    response looks identical and the data stays readable, but only the mapped
    path creates `customer` and `user` rows.

    In practice that leaves scopes owning memory rows with no
    control-plane row (SHIP-22). The fallback logged nothing, so the rate could
    not be measured at all. This locks the log line in place: without it, the
    next person measuring SHIP-22 has no signal to count.
    """
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "synap" / "cloud" / "application" / "sdk_api" / "sdk_context.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_build_scope":
            target = node
    assert target is not None, "_build_scope has been renamed; update this guard"

    for handler in [n for n in ast.walk(target) if isinstance(n, ast.ExceptHandler)]:
        calls_fallback = any(
            isinstance(c, ast.Call)
            and getattr(c.func, "id", None) == "build_scope_with_maca"
            for c in ast.walk(handler)
        )
        if not calls_fallback:
            continue
        logs = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and getattr(c.func.value, "id", None) == "logger"
            and c.func.attr in ("warning", "error")
            for c in ast.walk(handler)
        )
        assert logs, (
            "the unmapped-scope fallback is silent again. It stops customer and "
            "user rows being created while the response looks normal, and with "
            "no log line the rate cannot be measured."
        )
