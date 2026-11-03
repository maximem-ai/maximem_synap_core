"""Core's prompt for episodes.

The text is read from the Core prompt pack (``core_prompts/episodes_and_temporal``).
"""

from datetime import datetime
from typing import List, Optional, Tuple

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.shared.prompt_switch import pack_text


class EpisodesExtractionPrompt:
    """Builds prompts for EPISODES extraction."""

    SYSTEM_PROMPT = pack_text("episodes_and_temporal", "episodes.txt")

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

        maca_context = MACAContextBuilder.build_extraction_context(self._config, MemoryType.EPISODES)
        user_prompt = f"Content to analyze:\n---\n{content}\n---\n\nExtract all EPISODES from this content:"
        if has_temporal_content:
            from synap.cloud.agents.ingestion.prompts.extraction.temporal_addendum import (
                render_temporal_addendum,
            )
            user_prompt += "\n\n" + render_temporal_addendum(dct, instance_id=instance_id)
        return (with_instance_context(self.SYSTEM_PROMPT, maca_context), user_prompt)
