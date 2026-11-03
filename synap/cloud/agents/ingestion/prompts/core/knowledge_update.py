"""Core's text for the knowledge-update classifier.

The prompts are read from the Core prompt pack (``core_prompts/knowledge_update``). The three
blocks the Cloud classifier adds for delete, provenance and batching belong to switches that are
off in Core (``edition.CORE_REFUSED_SWITCHES``), so they have no Core text; using one stops with a
clear message.
"""

from synap.cloud.shared.prompt_switch import core_gap, pack_text

_ME = "synap.cloud.agents.ingestion.prompts.knowledge_update"

CLASSIFICATION_SYSTEM_PROMPT = pack_text("knowledge_update", "system.txt")
CLASSIFICATION_USER_PROMPT = pack_text("knowledge_update", "user.txt")  # {existing_content}, {new_content}

BATCH_CLASSIFICATION_ADDENDUM = core_gap(_ME, "BATCH_CLASSIFICATION_ADDENDUM")
DELETE_ADDENDUM = core_gap(_ME, "DELETE_ADDENDUM")
PROVENANCE_ADDENDUM = core_gap(_ME, "PROVENANCE_ADDENDUM")


# Functions of the same switches: asking for one stops with the same message when it is called.
provenance_note = core_gap(_ME, "provenance_note")
build_batch_user_prompt = core_gap(_ME, "build_batch_user_prompt")
