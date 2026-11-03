"""Core's prompt for temporal events, with the "keep the person" block.

The temporal text is read from ``core_prompts/episodes_and_temporal``; the block that keeps the
person a dated event belongs to is ``core_prompts/person_and_company/keep_person.txt``.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.shared.prompt_switch import pack_text

PERSON_SECTION = pack_text("person_and_company", "keep_person.txt").rstrip("\n")


def keep_person_enabled(instance_id: object = None) -> bool:
    """On when SYNAP_TEMPORAL_KEEP_PERSON is set (Core sets it)."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for("SYNAP_TEMPORAL_KEEP_PERSON", instance_id)


class TemporalExtractionPrompt:
    """Builds prompts for TEMPORAL_EVENTS extraction."""

    SYSTEM_PROMPT = pack_text("episodes_and_temporal", "temporal.txt")

    def __init__(self, config: MemoryArchitectureConfig):
        self._config = config

    def build(self, content: str, categories: List, dct: Optional[datetime] = None,
              has_temporal_content: bool = False, *, instance_id: object) -> Tuple[str, str]:
        """(system_prompt, user_prompt). ``categories`` is accepted for the same call and not used."""
        from synap.cloud.agents.ingestion.models.enums import MemoryType
        from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
            MACAContextBuilder,
            with_instance_context,
        )

        maca_context = MACAContextBuilder.build_extraction_context(self._config, MemoryType.TEMPORAL_EVENTS)
        user_prompt = f"Content to analyze:\n---\n{content}\n---\n\nExtract all TEMPORAL EVENTS from this content:"
        if has_temporal_content:
            from synap.cloud.agents.ingestion.prompts.extraction.temporal_addendum import (
                render_temporal_addendum,
            )
            user_prompt += "\n\n" + render_temporal_addendum(dct, instance_id=instance_id)
        system_prompt = self.SYSTEM_PROMPT
        if keep_person_enabled(instance_id):
            system_prompt += "\n\n" + PERSON_SECTION
        return (with_instance_context(system_prompt, maca_context), user_prompt)
