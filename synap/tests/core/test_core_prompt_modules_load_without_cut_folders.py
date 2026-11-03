"""Every Core prompt module loads when the folders Core does not ship are missing.

The S9 wiring tests ran in the full monorepo, where ``agents/memory_architecture``
exists, so four Core prompt files that imported it passed here and would have
failed in a Core tree the first time ingestion started. This test blocks the
folders that stay behind and imports every ``prompts/core`` module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

BLOCKED = [
    "synap.cloud.agents.memory_architecture",
    "synap.cloud.agents.registry",
    "synap.cloud.privacy",
    "synap.cloud.auth",
    "synap.cloud.observability.metrics",
    "synap.cloud.application.analytics",
    "synap.cloud.application.credits",
    "synap.cloud.application.admin_api",
    "synap.cloud.application.lifecycle",
    "synap.cloud.application.consolidation",
]


def _core_prompt_modules():
    names = []
    for path in sorted((ROOT / "synap/cloud").glob("**/prompts/**/core/*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def test_there_are_core_prompt_modules_to_check():
    assert len(_core_prompt_modules()) >= 18


def test_every_core_prompt_module_loads_with_the_cut_folders_blocked():
    code = textwrap.dedent(
        f"""
        import importlib, sys
        BLOCKED = {BLOCKED!r}
        class Block:
            def find_spec(self, name, path=None, target=None):
                for b in BLOCKED:
                    if name == b or name.startswith(b + "."):
                        raise ImportError("blocked: " + name)
                return None
        sys.meta_path.insert(0, Block())
        failed = []
        for name in {_core_prompt_modules()!r}:
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001
                failed.append(name + ": " + type(exc).__name__ + ": " + str(exc)[:120])
        print("\\n".join(failed))
        sys.exit(1 if failed else 0)
        """
    )
    env = dict(os.environ, SYNAP_EDITION="core", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
