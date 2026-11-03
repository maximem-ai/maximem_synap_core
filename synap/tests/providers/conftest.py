"""Shared config for provider tests.

Provider tests come in two flavours:

* Unit / contract tests — verify model shapes, interface abstractness, the
  registry wiring, and adapter behaviour against in-memory or mocked backends.
  These always run.

* Integration tests (``@pytest.mark.integration``) — exercise an adapter
  against a *real* datastore (MinIO/S3, Neo4j, TimescaleDB/Postgres). They are
  only meaningful when that backend is actually reachable. In CI/dev
  environments without the infra they are skipped (env-block), not failed —
  there is no in-memory substitute for these adapters and the test charter
  explicitly excludes spinning up real datastore connections.

This file registers the ``integration`` marker (silencing the
PytestUnknownMarkWarning) and skips integration tests whose backing service is
not listening, so the suite stays green without faking the datastore.
"""

import socket

import pytest


# Default endpoints used by the integration fixtures in this package.
# (host, port) probed with a short TCP connect; unreachable -> skip.
_BACKEND_ENDPOINTS = {
    "test_file.py": ("localhost", 9000),       # MinIO / S3-compatible
    "test_graph.py": ("localhost", 7687),      # Neo4j bolt
    "test_graph_wrapper_traverse.py": ("localhost", 7687),  # Neo4j bolt
    "test_reporting.py": ("localhost", 5432),  # TimescaleDB / Postgres
}

# ⚠ A TCP connect answers "is something listening", which is not the question.
# The question is "is the backend these tests need there".
#
# Found 2026-08-23, the hard way. CI gained a Postgres service container on
# 5432 for the scoping suite. That made `localhost:5432` reachable, so these
# four tests stopped skipping, ran against a database that is not TimescaleDB
# and has different credentials, and errored. Four skips became four errors and
# the whole gate went red, without a line of the code under test changing.
#
# So a backend may declare a real check. Where one exists it decides, and the
# TCP probe is only the cheap first pass that avoids a slow connect attempt
# against a port with nothing on it at all.
def _postgres_reporting_ready(host: str, port: int) -> bool:
    """Can we open the exact connection `test_reporting.py`'s fixture opens?"""
    try:
        import psycopg2
    except Exception:
        return False
    try:
        conn = psycopg2.connect(host=host, port=port, dbname="synap_test",
                                user="test", password="test", connect_timeout=2)
    except Exception:
        return False
    conn.close()
    return True


_BACKEND_VERIFIERS = {
    ("localhost", 5432): _postgres_reporting_ready,
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: test requires a live datastore backend (skipped when unreachable)",
    )


def _is_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def pytest_collection_modifyitems(config, items):
    # Cache reachability per (host, port) so we probe each backend once.
    reachable_cache: dict = {}
    for item in items:
        if "integration" not in item.keywords:
            continue
        filename = item.path.name
        endpoint = _BACKEND_ENDPOINTS.get(filename)
        if endpoint is None:
            continue
        if endpoint not in reachable_cache:
            ok = _is_reachable(*endpoint)
            verifier = _BACKEND_VERIFIERS.get(endpoint)
            if ok and verifier is not None:
                # Listening is necessary and not sufficient. Something else on
                # this port must not be mistaken for the backend under test.
                ok = verifier(*endpoint)
            reachable_cache[endpoint] = ok
        if not reachable_cache[endpoint]:
            host, port = endpoint
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"integration backend {host}:{port} not usable "
                        f"(env-block: nothing listening, or something else is)"
                    )
                )
            )
