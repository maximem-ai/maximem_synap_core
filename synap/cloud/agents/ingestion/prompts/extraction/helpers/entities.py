"""Entity-prompt flag that is logic, not prompt text: SYNAP_ENTITY_EXTRACTION_PROMPT=v2.

Moved byte for byte out of cloud/entities.py so Core can ship it; that module imports it back.
"""
import os

def entity_prompt_v2_enabled() -> bool:
    """True when SYNAP_ENTITY_EXTRACTION_PROMPT=v2. Off by default.

    v2 returns every name a text uses for an entity (the most complete as
    ``name``, the rest as ``aliases``), adds the names it is widely known by
    (``known_as``), keeps named teams, and carries no built-in limit: a limit
    comes from DomainHints.max_entities.
    """
    return os.environ.get("SYNAP_ENTITY_EXTRACTION_PROMPT", "").strip().lower() == "v2"
