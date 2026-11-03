"""Graph extraction (v2) runs to the end in Core (found at the first Docker boot).

After a memory was stored, the log showed "graph extraction (v2) failed" with CorePromptMissing: the support
check read the Jev judge's wording, a prompt Core does not have, and the whole graph step was dropped. Core has
no judge: the relationships keep the model's own confidence. This runs the real stage 3 extractor in
Core mode with a model that answers the graph prompt with a fixed graph.
"""
import os
import subprocess
import sys

SCRIPT = r'''
import asyncio, json, logging
from datetime import datetime, timezone
from types import SimpleNamespace

records = []

class Keep(logging.Handler):
    def emit(self, record):
        records.append(record)

logging.getLogger().addHandler(Keep())
logging.getLogger().setLevel(logging.WARNING)

from synap.cloud.agents.ingestion.models import Stage2Output
from synap.cloud.agents.ingestion.models.enums import MemoryType
from synap.cloud.agents.ingestion.pipeline import stage3_extraction as s3
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig

GRAPH = {"entities": [{"name": "Maria", "entity_type": "person"}, {"name": "Acme", "entity_type": "organization"}],
         "relationships": [{"source_entity": "Maria", "source_type": "person", "target_entity": "Acme",
                            "target_type": "organization", "relationship_type": "WORKS_AT", "inverse_type": "EMPLOYS",
                            "basis": "stated", "confidence": 0.9, "evidence": "Maria works at Acme", "properties": {}}]}

class Llm:
    async def complete(self, messages=None, **kwargs):
        system = (messages or [{"content": ""}])[0]["content"].lower()
        return json.dumps(GRAPH if "relationship" in system and "entit" in system else {"reasoning": "", "extracted": []})

async def main():
    cfg = MemoryArchitectureConfig(metadata=ConfigMetadata(
        client_id="c", instance_id="i", config_version="1.0.0", generated_at=datetime.now(timezone.utc),
        generated_by="test"))
    ex = s3.Stage3Extractor(config=cfg, llm_client=Llm(), llm_client_light=Llm(), llm_client_graph=Llm())
    scope = SimpleNamespace(instance_id="44eff2d7-2a3d-5075-8c39-addf2a0188d0", client_id=None, customer_id=None,
                            user_id=None, scope_level="user")
    out = await ex.extract(
        content="user: Maria works at Acme.",
        stage2_output=Stage2Output(input_id="t", categories=[], extraction_priorities=[MemoryType.FACTS]),
        document_id="d", dct=None, has_temporal_content=False, request_scope=scope, correlation_id="c", config=cfg)
    print("ENTITIES", sorted(e.name for e in out.entities), "RELATIONSHIPS", len(out.relationships))
    bad = [r.getMessage() for r in records if "graph extraction" in r.getMessage() or "CorePromptMissing" in r.getMessage()]
    print("PROBLEMS", bad)

asyncio.run(main())
'''


def test_the_graph_step_keeps_its_relationships_in_core():
    env = {**os.environ, "SYNAP_EDITION": "core"}
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=240)
    text = out.stdout + out.stderr
    assert "PROBLEMS []" in out.stdout, text[-3000:]
    assert "ENTITIES ['Acme', 'Maria'] RELATIONSHIPS 1" in out.stdout, text[-3000:]
