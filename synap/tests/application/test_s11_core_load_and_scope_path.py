"""Three things Core relies on, shown with fakes (no database, no model).

1. The ingestion worker base class loads without ``synap.providers.reporting``,
   which Core does not ship.
2. The fetch by conversation id does not read the scope path stored on the
   conversation row when the edition has no scope ladder.
3. Cloud still reads it.
"""
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_the_worker_base_class_loads_without_the_reporting_package():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.modules['synap.providers.reporting'] = None   # an import of it now fails
        import synap.cloud.agents.ingestion.workers.base as b
        assert hasattr(b, 'BaseIngestionWorker')
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)}, cwd=str(ROOT), timeout=120)
    assert done.returncode == 0, done.stderr[-800:]


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _Session:
    def __init__(self, sql_seen, row):
        self._sql_seen, self._row = sql_seen, row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, statement, params):
        self._sql_seen.append(str(statement))
        return _Result(self._row)


async def _resolve(monkeypatch, scope_levels, row):
    from synap.cloud.application.sdk_api import sdk_context as m

    seen = []
    monkeypatch.setattr("synap.cloud.database.connection.get_session_factory", lambda: (lambda: _Session(seen, row)))
    monkeypatch.setattr(m, "edition", SimpleNamespace(scope_levels=scope_levels))
    out = await m._resolve_scope_from_conversation("11111111-1111-1111-1111-111111111111", "inst")
    return out, seen


@pytest.mark.asyncio
async def test_core_does_not_select_the_stored_scope_path(monkeypatch):
    out, seen = await _resolve(monkeypatch, ("user", "customer", "install"), ("u1", "c1", None))
    assert out == ("u1", "c1", None)
    assert "scope_path" not in seen[0]


@pytest.mark.asyncio
async def test_cloud_still_reads_the_stored_scope_path(monkeypatch):
    out, seen = await _resolve(monkeypatch, None, ("u1", "c1", {"tenant": "acme"}))
    assert out == ("u1", "c1", {"tenant": "acme"})
    assert "scope_path" in seen[0]
