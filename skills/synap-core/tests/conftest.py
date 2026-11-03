"""Helpers for the skill's script tests. No server, no Docker and no model is needed.

Every script is run as a subprocess, the way an agent runs it, and judged by its exit code and its output.
The tests never write a key-shaped string into this repository: where a test needs a fake secret it builds the
text at run time.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL / "scripts"
EXAMPLES = SKILL / "examples"
REPO = SKILL.parent.parent


def run_script(name: str, *args: str, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run scripts/<name> with a clean environment: nothing of the caller's SYNAP_* or provider settings leaks in."""
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("SYNAP_", "OPENAI_", "ANTHROPIC_", "JEV_", "OLLAMA_"))}
    clean.update(env or {})
    return subprocess.run([sys.executable, "-I", str(SCRIPTS / name), *args],
                          capture_output=True, text=True, cwd=cwd, env=clean, timeout=60)


@pytest.fixture
def run():
    return run_script


@pytest.fixture
def env_example(tmp_path) -> Path:
    """A copy of the repository's .env.example, so set_env.py can create a .env from it."""
    (tmp_path / ".env.example").write_text((REPO / ".env.example").read_text())
    return tmp_path / ".env"
