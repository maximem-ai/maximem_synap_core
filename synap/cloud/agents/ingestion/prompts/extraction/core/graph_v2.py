"""Core's prompt for graph extraction v2.

The texts are read from the Core prompt pack (``core_prompts/graph_extraction``).

The writer ("I") and the writer's organization ("we") are told in the prompt itself:
``graph_extraction/system.txt`` has a block for the organization that is kept only when a
``customer`` name is given. ``person_and_company/customer_graph.txt`` says the same thing in its
own paragraph and adds three rules the block lacks (use the exact name in every "we" relationship,
create one entity, and create none when the note has no "we"). Core stated the rule once: the
paragraph from ``customer_graph.txt`` takes the place of the block's body, and nothing is added
beside it. The ``core_prompts`` files are not edited.

Core's graph prompt has no statements, event-node, example or per-scope pattern layer: those
belong to switches that are off in Core (``edition.CORE_REFUSED_SWITCHES``). The arguments are
accepted so the call is the same as in Cloud.
"""

from datetime import datetime
from typing import Optional, Tuple

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
import re

from synap.cloud.shared.prompt_switch import CorePromptMissing, fill, pack_text

_PRONOUNS = {"he": "he / him / his", "she": "she / her", "they": "they / them / their"}


class GraphExtractionPromptV2:
    """Builds the one prompt that extracts entities and relationships together."""

    def __init__(self, config: MemoryArchitectureConfig):
        self._config = config

    def build(
        self,
        content: str,
        dct: Optional[datetime] = None,
        writer: str = "the user",
        customer: Optional[str] = None,
        references=None,
        events: bool = False,
        relationship_examples: bool = True,
        statements: bool = False,
        scope_kind: Optional[str] = None,
    ) -> Tuple[str, str]:
        """(system_prompt, user_prompt).

        *writer*: the name the writer of the content goes by.
        *customer*: the name of the writer's organization when "we" means it, else None.
        *references*: (pronoun, name, entity_type) for pronouns whose entity is known from earlier notes.
        *relationship_examples*: show the instance's configured relationship types as wording examples.
        """
        system_prompt = fill(
            self._template(),
            writer_name=writer,
            customer_name=customer,
            domain_block=self._domain(relationship_examples) or None,
            earlier_references=self._references(references) or None,
        )
        user_prompt = fill(pack_text("graph_extraction", "user.txt"), content=content)
        return (system_prompt, user_prompt)

    @staticmethod
    def _template() -> str:
        """``system.txt`` with the body of its first company block replaced by ``customer_graph.txt``."""
        text = pack_text("graph_extraction", "system.txt")
        block = re.compile(r"(\[\[IF CUSTOMER_NAME\]\]\n\nTHE WRITER'S ORGANIZATION\n).*?(\n\[\[END IF\]\])", re.S)
        rule = pack_text("person_and_company", "customer_graph.txt").strip()
        replaced, count = block.subn(lambda m: m.group(1) + rule + m.group(2), text, count=1)
        if count != 1:
            raise CorePromptMissing(
                "core_prompts/graph_extraction/system.txt no longer has its company block, so the "
                "company rule of person_and_company/customer_graph.txt cannot be placed. Core stops "
                "here instead of running without it.")
        return replaced

    @staticmethod
    def _references(references) -> str:
        return "\n".join(
            f"{_PRONOUNS.get(pronoun, pronoun)}: {name} ({entity_type})"
            for pronoun, name, entity_type in (references or [])
        )

    def _domain(self, relationship_examples: bool) -> str:
        """What the instance's configuration says about its graph, as lines. "" when nothing is
        configured, or on any fault reading it (a config fault must not stop extraction)."""
        try:
            from synap.cloud.agents.ingestion.prompts.maca_context_builder import (
                _resolve_relationship_types,
                _schema_or_hint_entity_types,
            )
            lines = []
            dh = self._config.memory_extraction.domain_hints
            if dh.industry:
                lines.append(f"This content comes from: {dh.industry}.")
            entity_types = [t.lower() for t in _schema_or_hint_entity_types(self._config)]
            if entity_types:
                lines.append(
                    "Kinds of entity that matter here: " + ", ".join(entity_types) + ". Each of these is an "
                    "entity whenever the content states a fact about one. Other kinds are entities too.")
            rel_types = _resolve_relationship_types(self._config) if relationship_examples else []
            if rel_types:
                lines.append(
                    "Examples of how relationships are worded here: " + ", ".join(rel_types) + ". Use one "
                    "only when it says exactly what the content says. Otherwise write your own type.")
            ing = self._config.agent_hints.ingestion_agent
            if ing.domain_context:
                lines.append(ing.domain_context)
            if ing.focus_areas:
                lines.append("Focus on: " + "; ".join(ing.focus_areas) + ".")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""
