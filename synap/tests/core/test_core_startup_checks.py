"""Core stops at start with the reason when a prompt text, the LLM setting or an embedding setting is wrong.

The edition is read once at import, so the Core cases run in a subprocess with ``SYNAP_EDITION=core``.
They start the server's own ``lifespan``, which makes the check before it touches the database.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

GOOD = {
    "SYNAP_LLM_PROVIDER": "openai",
    "SYNAP_LLM_MODEL": "some-model",
    "OPENAI_API_KEY": "sk-placeholder-not-real",
    "EMBEDDING_PROVIDER": "sentence-transformers",
}

BOOT = """
import asyncio, json, sys
from synap.cloud.application import main
from synap.cloud.shared import core_startup
{prelude}
async def go():
    try:
        async with main.lifespan(main.app):
            return "started"
    except core_startup.CoreStartupError as exc:
        return str(exc)
    except BaseException as exc:
        return "OTHER " + type(exc).__name__
print(json.dumps({{"result": asyncio.run(go())}}))
"""

SEED = """
import types
from synap.cloud.shared import prompt_switch as ps
class U(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith('__'):
            raise AttributeError(n)
        return 'x'
for i, (status, _p) in ps.INTERFACES.items():
    if status == ps.CORE_STATUS and not i.endswith('prompt_resolver'):
        sys.modules[ps.core_name(i)] = U(ps.core_name(i))
"""


def _run(env_extra, prelude="", edition="core", drop=()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"SYNAP_EDITION": edition, "PYTHONPATH": str(ROOT)})
    env.update(env_extra)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(BOOT.format(prelude=prelude))],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=240,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])["result"]


_NO_LLM = ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "EMBEDDING_PROVIDER", "OPENAI_API_KEY")


def test_core_refuses_to_start_when_a_prompt_text_is_missing():
    kn = "synap.cloud.agents.ingestion.prompts.knowledge_update"
    prelude = SEED + f"sys.modules[ps.core_name({kn!r})] = None\n"
    result = _run(GOOD, prelude, drop=_NO_LLM)
    assert kn in result and "has no text for the prompt" in result, result


def test_core_refuses_to_start_with_no_llm_provider_and_model():
    result = _run({}, SEED, drop=_NO_LLM)
    assert "SYNAP_LLM_PROVIDER and SYNAP_LLM_MODEL are not set" in result, result


def test_core_refuses_to_start_with_only_one_of_provider_and_model():
    env = {k: v for k, v in GOOD.items() if k != "SYNAP_LLM_MODEL"}
    result = _run(env, SEED, drop=_NO_LLM)
    assert "SYNAP_LLM_MODEL is not set" in result, result


def test_core_refuses_the_remote_embedding_provider():
    result = _run({**GOOD, "EMBEDDING_PROVIDER": "remote"}, SEED, drop=_NO_LLM)
    assert "EMBEDDING_PROVIDER=remote is not available in Core" in result, result


def test_core_refuses_openai_embeddings_without_a_key():
    env = {**GOOD, "EMBEDDING_PROVIDER": "openai"}
    del env["OPENAI_API_KEY"]
    result = _run(env, SEED, drop=_NO_LLM)
    assert "EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY" in result, result


def test_core_passes_the_check_with_everything_set():
    result = _run(GOOD, SEED, drop=_NO_LLM)
    # The check passed: the boot went on to the database, which is not here.
    assert not result.startswith("Synap Core cannot start"), result


def test_core_refuses_the_eval_provider_setting():
    for name in ("EVAL_PROVIDER",):
        env = {**os.environ, "SYNAP_EDITION": "core", "PYTHONPATH": str(ROOT), name: "x"}
        done = subprocess.run([sys.executable, "-c", "import synap.cloud.shared.edition"],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert done.returncode != 0 and f"{name} is set but Core does not read it" in done.stderr, name


def test_only_core_checks_its_settings_at_start():
    from synap.cloud.shared import edition

    assert edition.resolve({}).checks_settings_at_start is False
    assert edition.resolve({"SYNAP_EDITION": "core"}).checks_settings_at_start is True
