"""Shared fixtures / import guards for the agents test package.

grpc-shadow guard
-----------------
pytest's default ``prepend`` import mode puts ``synap/tests`` on
``sys.path`` (the first parent directory without an ``__init__.py``). That
makes the empty ``synap/tests/grpc`` package resolvable as the top-level
``grpc`` module, shadowing the real grpcio package. Several agents tests
import the Synap SDK, whose transport layer does ``from grpc import aio``
at import time — which then fails with
``ImportError: cannot import name 'aio'``. This is a test-infrastructure
collision, not a product bug.

We pre-import the real grpcio package here (with the test dirs filtered off
the search path) so it is cached in ``sys.modules`` before any test module
imports the SDK. Subsequent ``import grpc`` / ``from grpc import aio`` hit
the correct cached module instead of resolving the shadow. Idempotent.
"""

from __future__ import annotations

import importlib
import os
import sys


def _ensure_real_grpc_package() -> None:
    existing = sys.modules.get("grpc")
    if existing is not None and hasattr(existing, "aio"):
        return  # real grpc already loaded

    for name in [m for m in sys.modules if m == "grpc" or m.startswith("grpc.")]:
        sys.modules.pop(name, None)

    tests_dir_targets = {
        os.path.normpath(os.path.join(os.getcwd(), "synap", "tests")),
        os.path.normpath("synap/tests"),
    }
    saved_path = list(sys.path)
    sys.path[:] = [
        p for p in sys.path
        if os.path.normpath(p).rstrip(os.sep) not in tests_dir_targets
    ]
    try:
        importlib.import_module("grpc")
        importlib.import_module("grpc.aio")
    except Exception:
        # If grpcio is genuinely unavailable, leave sys.modules as-is; the
        # tests that need it will surface a clear error.
        pass
    finally:
        sys.path[:] = saved_path


_ensure_real_grpc_package()
