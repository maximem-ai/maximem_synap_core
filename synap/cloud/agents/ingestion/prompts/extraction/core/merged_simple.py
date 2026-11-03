"""Core's prompt for facts, preferences and emotions in one call.

The texts are read from the Core prompt pack (``core_prompts/merged_extraction``). The YAML guidance
is added to the end of the system prompt by the shared builder, as in Cloud.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from synap.cloud.agents.ingestion.services import custom_categories
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.shared.prompt_switch import fill, pack_text


class MergedSimpleExtractionPrompt:
    """Builds one prompt that extracts FACTS + PREFERENCES + EMOTIONS."""

    #: memory types this prompt covers (values match MemoryType.value)
    COVERS = ("facts", "preferences", "emotions")

    # The text still holds the custom-types blocks; SYSTEM_PROMPT is what it reads with none.
    _SYSTEM_TEXT = pack_text("merged_extraction", "system.txt")
    SYSTEM_PROMPT = fill(_SYSTEM_TEXT, builtin_only=True, custom_types="")

    def __init__(self, config: MemoryArchitectureConfig):
        self._config = config

    def build(
        self,
        content: str,
        categories: List,
        dct: Optional[datetime] = None,
        has_temporal_content: bool = False,
        memory_types: Optional[List] = None,
        *,
        instance_id: object,
    ) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt). ``categories`` is accepted for the same call and not used."""
        from synap.cloud.agents.ingestion.models.enums import MemoryType
        from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
            MACAContextBuilder,
            with_instance_context,
        )

        types = list(memory_types) if memory_types else [
            MemoryType.FACTS, MemoryType.PREFERENCES, MemoryType.EMOTIONS]
        try:
            maca_context = MACAContextBuilder.build_merged_extraction_context(self._config, types)
        except Exception:
            maca_context = ""

        user_prompt = fill(pack_text("merged_extraction", "user.txt"), content=content)
        if has_temporal_content:
            from synap.cloud.agents.ingestion.prompts.extraction.temporal_addendum import (
                render_temporal_addendum,
            )
            user_prompt += "\n\n" + render_temporal_addendum(dct, instance_id=instance_id)
        # Set by Stage 3 only when SYNAP_CUSTOM_CATEGORIES is on for the instance; with it off the
        # fields below drop their blocks and the text is the file as it was.
        custom = custom_categories.ACTIVE.get()
        system_prompt = fill(
            self._SYSTEM_TEXT,
            builtin_only=not custom,
            custom_types=custom_categories.format_for_prompt(custom),
        )
        return (with_instance_context(system_prompt, maca_context), user_prompt)
