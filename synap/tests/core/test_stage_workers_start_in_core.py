"""The ingestion stage workers must start in Core (found at the first Docker boot).

The server reported healthy while the stage 2/3 worker had failed to start, because building the entity
extractor asked for a prompt Core does not have. Every memory then sat in the queue. This starts the
workers the way the server does, in Core mode, with a fake Redis and a placeholder model key.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("fakeredis")

SCRIPT = """
import asyncio, logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
import fakeredis
from synap.cloud.application.ingestion import startup as st

async def main():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    await st._start_stage_workers(redis_client=redis, status_tracker=st.StatusTracker(redis),
                                  reporting=st._NoOpReportingStore())
    await asyncio.sleep(1)
    print("TASKS", len(st._worker_tasks), "DIED", sum(t.done() for t in st._worker_tasks))
    import sys
    print("EXTRACTOR_LOADED", "synap.cloud.agents.ingestion.extractors.llm_entity_extractor" in sys.modules)
    for t in st._worker_tasks:
        t.cancel()

asyncio.run(main())
"""


def test_every_stage_worker_starts_in_core_mode():
    env = {
        **os.environ,
        "SYNAP_EDITION": "core",
        "SYNAP_LLM_PROVIDER": "openai",
        "SYNAP_LLM_MODEL": "placeholder-model",
        "OPENAI_API_KEY": "sk-placeholder-not-real",
        "EMBEDDING_PROVIDER": "sentence-transformers",
    }
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=240)
    text = out.stdout + out.stderr
    assert "failed to start" not in text, text[-3000:]
    assert "CorePromptMissing" not in text, text[-3000:]
    # one stage 1 worker, three stage 2/3 workers, two stage 4 workers, none already dead
    assert "TASKS 6 DIED 0" in out.stdout, text[-3000:]
    # Core reads entities and relationships in one graph call: the separate entity extractor is not built
    assert "EXTRACTOR_LOADED False" in out.stdout, text[-3000:]
