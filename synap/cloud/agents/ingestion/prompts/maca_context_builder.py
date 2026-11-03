"""Builds MACA context blocks for injection into prompt user messages.

The MACA (Memory Architecture Configuration Agent) config contains instance-specific
fields that should influence extraction behavior. These fields go in the user message
(not the system prompt) to keep the system prompt static and cacheable.

All methods return XML-structured strings or empty string when no MACA context exists.
"""

from contextvars import ContextVar
from typing import List, Optional, Tuple

from synap.cloud.agents.ingestion.models.enums import MemoryType
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.memory_config.config_schema import (
    DomainHints,
    MemoryCategoryConfig,
)


# Default entity types when MACA provides none
DEFAULT_ENTITY_TYPES = [
    "person",
    "organization",
    "product",
    "technology",
    "location",
    "concept",
]

# Budget for the MACA context block. It was 2,000 chars (~500 tokens), which
# was never exercised: every extractor used to be built with a blank default
# config, so this block rendered empty for every instance. Once real configs
# reached it, the LongMemEval instance's global part
# (focus areas + avoid + industry) alone is 1,800 chars, and each type adds
# 821-1,051 chars of its own rationale and examples. At 2,000 the type-specific
# half lost 621-851 chars in every call. 6,000 (~1,500 tokens) fits a real
# config for both the per-type prompt and the merged three-type prompt; the
# ordering below still protects type-specific guidance if a config overflows.
_MAX_CONTEXT_CHARS = 6000

# What this call must never turn into a memory (guardrails.compliance.prohibited_memory_content).
# Set by Stage 3 for the length of one extraction, and only when the instance's
# SYNAP_GUARDRAIL_PROHIBITED_CONTENT switch is on, so the renderer needs no
# instance id and an instance with the switch off renders exactly what it did.
PROHIBITED_CONTENT: ContextVar[Tuple[str, ...]] = ContextVar(
    "extraction_prohibited_content", default=())

# True while an extraction enforces memory_extraction.filters.require_entity_grounding
# (set by Stage 3 under the SYNAP_GUARDRAIL_ENTITY_GROUNDING switch, like the list above).
REQUIRE_ENTITY_GROUNDING: ContextVar[bool] = ContextVar(
    "extraction_require_entity_grounding", default=False)


class MACAContextBuilder:
    """Builds XML-structured MACA context for LLM prompt user messages."""

    @staticmethod
    def build_extraction_context(
        config: MemoryArchitectureConfig,
        memory_type: MemoryType,
    ) -> str:
        """Build MACA context for one memory type's extraction prompt.

        Returns an empty string when the config carries nothing to say, which is
        what a blank default config produces, so an instance that has not opted
        in to MACA-steered extraction gets a byte-identical prompt.

        Order is deliberate: this type's own rationale, examples, priority and
        aggressiveness come FIRST, then the instance-wide guidance. Truncation
        keeps the start, so if a config overflows the budget it is the generic
        part that is cut, never the part specific to this call.
        """
        sections = _type_extraction_sections(config, memory_type)
        sections += _global_extraction_sections(config, [memory_type])
        return _extraction_context(sections)

    @staticmethod
    def build_merged_extraction_context(
        config: MemoryArchitectureConfig,
        memory_types: List[MemoryType],
    ) -> str:
        """Build MACA context for a merged call covering several memory types.

        Each type's own guidance appears once and the instance-wide guidance
        appears once. Building a full block per type and joining them, as the
        merged prompt used to, repeats the instance-wide part for every type and
        lets it crowd each type's guidance out of the budget. Only the types
        actually extracted in this call are included, so a disabled type's
        rationale and examples never leak into it.
        """
        sections: List[str] = []
        for memory_type in memory_types:
            sections += _type_extraction_sections(config, memory_type)
        sections += _global_extraction_sections(config, memory_types)
        return _extraction_context(sections)

    @staticmethod
    def build_entity_context(
        config: MemoryArchitectureConfig,
        domain_hints: Optional[DomainHints] = None,
    ) -> str:
        """The entity types to look for, for the user message.

        Consolidates entity types from multiple sources into a single list:
        1. graph_schemas[].node_types (highest priority — schema-defined)
        2. domain_hints.key_entities (MACA domain hints)
        3. DEFAULT_ENTITY_TYPES (fallback)

        This block is never empty, because the fallback list always applies, so
        it stays in the user message where it has always been. The instance's
        own standing guidance moved to ``build_entity_instance_context``.
        """
        entity_types = _resolve_entity_types(config, domain_hints)
        content = f"<entity_types>{', '.join(entity_types)}</entity_types>"
        effective_hints = domain_hints or config.memory_extraction.domain_hints
        if effective_hints.max_entities:
            # Read by the v2 entity prompt, which has no limit of its own.
            keep = effective_hints.entity_priority or "the entities the content is mainly about"
            content += (
                f"\n<entity_limit>At most {effective_hints.max_entities} entities. "
                f"When there are more, keep {keep}.</entity_limit>"
            )
        return _truncate(f"<context>\n{content}\n</context>")

    @staticmethod
    def build_entity_instance_context(
        config: MemoryArchitectureConfig,
        domain_hints: Optional[DomainHints] = None,
    ) -> str:
        """Standing guidance for entity extraction, for the system prompt.

        Empty for a config that sets none of it, which is every instance that
        has not opted in to MACA-steered extraction.
        """
        sections: List[str] = []
        effective_hints = domain_hints or config.memory_extraction.domain_hints
        if effective_hints.industry:
            sections.append(f"<domain>Industry: {effective_hints.industry}</domain>")
        hints = config.agent_hints.ingestion_agent
        if hints.domain_context:
            sections.append(f"<domain_context>{hints.domain_context}</domain_context>")
        if hints.focus_areas:
            sections.append(
                "<focus_areas>\nFocus on:\n" + _bullets(hints.focus_areas) + "\n</focus_areas>")
        if not sections:
            return ""
        return _truncate("<context>\n" + "\n".join(sections) + "\n</context>")

    @staticmethod
    def build_relationship_context(
        config: MemoryArchitectureConfig,
        domain_hints: Optional[DomainHints] = None,
    ) -> str:
        """Build MACA context for relationship extraction.

        Consolidates relationship types from multiple sources:
        1. graph_schemas[].edge_types (highest priority — schema-defined)
        2. domain_hints.key_relationships (MACA domain hints)

        If both are empty, returns empty string (prompt uses its own defaults).
        """
        sections: List[str] = []

        # Resolve relationship types
        rel_types = _resolve_relationship_types(config, domain_hints)
        if rel_types:
            sections.append(
                f"<relationship_types>Domain-specific relationship types "
                f"(prioritize these): {', '.join(rel_types)}</relationship_types>"
            )

        # Industry context
        effective_hints = domain_hints or config.memory_extraction.domain_hints
        if effective_hints.industry:
            sections.append(f"<domain>Industry: {effective_hints.industry}</domain>")

        # Agent hints
        hints = config.agent_hints.ingestion_agent
        if hints.domain_context:
            sections.append(f"<domain_context>{hints.domain_context}</domain_context>")

        if not sections:
            return ""

        content = "\n".join(sections)
        return _truncate(f"<context>\n{content}\n</context>")


def with_instance_context(system_prompt: str, maca_context: str) -> str:
    """Append this instance's guidance to the END of the system prompt.

    It used to go at the start of the user message, to keep the system prompt
    identical for every instance and therefore cacheable. That reasoning was
    half right. The user message carries the document, so it is different on
    every single call and nothing in it is ever cached: the instance block was
    paying full price on every call, about 800 tokens for the LongMemEval
    instance. At the end of the system prompt the shared part in front of it
    still matches across instances, and the instance's own tail becomes a
    cached prefix after its first call, at half price on OpenAI.

    It also puts standing instructions where the model treats them as
    instructions, next to the ones they qualify, instead of in front of the
    content they are meant to be applied to.

    An empty block returns the system prompt untouched, so an instance that has
    not opted in sends exactly what it sent before.
    """
    if not maca_context:
        return system_prompt
    return system_prompt + "\n\n" + maca_context


def _types_without_focus(
    config: MemoryArchitectureConfig,
    memory_types: Optional[List[MemoryType]],
) -> bool:
    """True when any type in this call has no focus areas of its own."""
    if not memory_types:
        return True
    for memory_type in memory_types:
        cat_config = _get_category_config(config, memory_type)
        if not cat_config or not getattr(cat_config, "focus_areas", None):
            return True
    return False


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _extraction_context(sections: List[str]) -> str:
    """The <context> block for an extraction call, or "" when there is nothing to say.

    The prohibited-content and grounding sections are added AFTER truncation:
    they are rules the instance must be able to rely on, and a long config
    would otherwise cut them first, since truncation keeps the start. Empty list (the switch off, or
    nothing listed) returns exactly what this returned before.
    """
    text = _truncate("<context>\n" + "\n".join(sections) + "\n</context>") if sections else ""
    blocks = []
    items = PROHIBITED_CONTENT.get()
    if items:
        blocks.append(
            "<prohibited>\nNever extract a memory that contains any of the following, "
            "even when the content states it plainly and even when it looks useful:\n"
            + _bullets(list(items)) + "\n</prohibited>")
    if REQUIRE_ENTITY_GROUNDING.get():
        blocks.append(
            "<grounding>\nEvery memory must name, in its own text, the person, "
            "organization or thing it is about. Do not write a bare takeaway "
            "with no named subject.\n</grounding>")
    if not blocks:
        return text
    block = "\n".join(blocks)
    if not text:
        return "<context>\n" + block + "\n</context>"
    return text[: -len("\n</context>")] + "\n" + block + "\n</context>"


def _global_extraction_sections(
    config: MemoryArchitectureConfig,
    memory_types: Optional[List[MemoryType]] = None,
) -> List[str]:
    """Instance-wide extraction guidance, the same for every memory type.

    *memory_types* names what THIS call extracts. MACA writes one focus list for
    the whole instance and its items name several memory types at once, so the
    facts call was being handed items about preferences and dates. The list is
    still sent whole, because an item can be relevant to a type it does not
    name, but the call says which types it is actually extracting.
    """
    sections: List[str] = []

    # Who "the user" is. Every extraction prompt resolves "I", "me", "my" to the
    # user in its coreference section; role semantics says who that actually
    # is for this instance (a patient, an employee of a customer company, the
    # same person as the customer in B2C).
    roles = getattr(config, "role_semantics", None)
    if roles is not None:
        user = (getattr(roles, "user_description", None) or "").strip()
        customer = (getattr(roles, "customer_description", None) or "").strip()
        client = (getattr(roles, "client_description", None) or "").strip()
        lines = []
        # B2C: the user IS the customer and MACA writes the same text for both.
        if user and user == customer:
            lines.append(f"User (also the customer): {user}")
        else:
            if user:
                lines.append(f"User: {user}")
            if customer:
                lines.append(f"Customer: {customer}")
        if client:
            lines.append(f"Client operating this agent: {client}")
        if lines:
            sections.append("<roles>\n" + "\n".join(lines) + "\n</roles>")

    industry = config.memory_extraction.domain_hints.industry
    if industry:
        sections.append(f"<domain>Industry: {industry}</domain>")

    hints = config.agent_hints.ingestion_agent
    if hints.domain_context:
        sections.append(f"<domain_context>{hints.domain_context}</domain_context>")
    # One item per line: MACA writes items that contain commas and colons
    # themselves, so joining them with ", " ran them into one unreadable sentence.
    # The instance-wide list is a FALLBACK. A category that carries its own
    # focus areas gets those in its own call (see _type_extraction_sections),
    # and then this list is not sent at all: it is one list for the whole
    # instance whose items name several memory types, which is what put
    # preference and date guidance into the facts call.
    needs_fallback = _types_without_focus(config, memory_types)
    if hints.focus_areas and needs_fallback:
        running = ", ".join(mt.value for mt in memory_types) if memory_types else ""
        scope = (f"This instance cares about the items below across ALL its memory types. "
                 f"THIS call extracts only {running}; an item that belongs to another memory "
                 f"type is handled by that type's own call.\n" if running else
                 "This instance cares about the items below.\n")
        sections.append(
            "<focus_areas>\n" + scope + _bullets(hints.focus_areas) + "\n</focus_areas>")
    if hints.avoid:
        sections.append(
            "<avoid>\nDo not extract memories about:\n" + _bullets(hints.avoid) + "\n</avoid>")
    if hints.extraction_guidance:
        sections.append(
            f"<extraction_guidance>{hints.extraction_guidance}</extraction_guidance>")
    return sections


def _type_extraction_sections(
    config: MemoryArchitectureConfig,
    memory_type: MemoryType,
) -> List[str]:
    """Guidance specific to one memory type."""
    sections: List[str] = []
    cat_config = _get_category_config(config, memory_type)
    if not cat_config:
        return sections
    if cat_config.rationale:
        sections.append(
            f"<rationale>Why {memory_type.value} matters for this instance: "
            f"{cat_config.rationale}</rationale>")
    if cat_config.examples:
        sections.append(
            f"<instance_examples>Instance-specific {memory_type.value} examples: "
            f"{'; '.join(cat_config.examples[:5])}</instance_examples>")
    if getattr(cat_config, "focus_areas", None):
        sections.append(
            f"<focus_areas>\nFor {memory_type.value}, look for:\n"
            + _bullets(cat_config.focus_areas) + "\n</focus_areas>")
    if cat_config.priority:
        sections.append(
            f"<priority>Extraction priority for {memory_type.value}: "
            f"{cat_config.priority.value}</priority>")
    if cat_config.extraction_aggressiveness:
        sections.append(
            f"<aggressiveness>Extraction level for {memory_type.value}: "
            f"{cat_config.extraction_aggressiveness.value}</aggressiveness>")
    return sections


def _get_category_config(
    config: MemoryArchitectureConfig,
    memory_type: MemoryType,
) -> Optional[MemoryCategoryConfig]:
    """Get the memory category config for a given type."""
    mapping = {
        MemoryType.FACTS: config.memory_categories.facts,
        MemoryType.PREFERENCES: config.memory_categories.preferences,
        MemoryType.EPISODES: config.memory_categories.episodes,
        MemoryType.EMOTIONS: config.memory_categories.emotions,
        MemoryType.TEMPORAL_EVENTS: config.memory_categories.temporal_events,
    }
    return mapping.get(memory_type)


def _resolve_entity_types(
    config: MemoryArchitectureConfig,
    domain_hints: Optional[DomainHints] = None,
) -> List[str]:
    """Resolve entity types from MACA sources into a single consolidated list.

    Priority order:
    1. graph_schemas[].node_types (schema-defined types)
    2. domain_hints.key_entities (MACA-configured types)
    3. DEFAULT_ENTITY_TYPES (hardcoded fallback)
    """
    # Check graph schemas for node types
    schema_types: List[str] = []
    for schema in config.storage_architecture.graph_schemas:
        if schema.enabled and schema.node_types:
            schema_types.extend(schema.node_types)

    if schema_types:
        # Deduplicate while preserving order
        seen = set()
        result = []
        for t in schema_types:
            lower = t.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(lower)
        return result

    # Fall back to domain hints
    effective_hints = domain_hints or config.memory_extraction.domain_hints
    if effective_hints.key_entities:
        return [t.lower() for t in effective_hints.key_entities]

    # Ultimate fallback
    return list(DEFAULT_ENTITY_TYPES)


def _schema_or_hint_entity_types(
    config: MemoryArchitectureConfig,
    domain_hints: Optional[DomainHints] = None,
) -> List[str]:
    """Entity types the instance configured, or [] when it configured none.

    Same sources and order as ``_resolve_entity_types`` (graph schema node types,
    then domain_hints.key_entities) but without the built-in default list, for
    prompts that already carry their own defaults and must not change when the
    config is blank.
    """
    has_schema_types = any(
        s.enabled and s.node_types for s in config.storage_architecture.graph_schemas)
    effective_hints = domain_hints or config.memory_extraction.domain_hints
    if not has_schema_types and not effective_hints.key_entities:
        return []
    return _resolve_entity_types(config, domain_hints)


def _resolve_relationship_types(
    config: MemoryArchitectureConfig,
    domain_hints: Optional[DomainHints] = None,
) -> List[str]:
    """Resolve relationship types from MACA sources.

    Priority order:
    1. graph_schemas[].edge_types (schema-defined types)
    2. domain_hints.key_relationships (MACA-configured types)
    3. Empty list (prompt uses its own built-in taxonomy)
    """
    # Check graph schemas for edge types
    schema_types: List[str] = []
    for schema in config.storage_architecture.graph_schemas:
        if schema.enabled and schema.edge_types:
            schema_types.extend(schema.edge_types)

    if schema_types:
        seen = set()
        result = []
        for t in schema_types:
            upper = t.upper()
            if upper not in seen:
                seen.add(upper)
                result.append(upper)
        return result

    # Fall back to domain hints
    effective_hints = domain_hints or config.memory_extraction.domain_hints
    if effective_hints.key_relationships:
        return [r.upper() for r in effective_hints.key_relationships]

    return []


def _truncate(text: str) -> str:
    """Truncate MACA context to prevent prompt bloat."""
    if len(text) <= _MAX_CONTEXT_CHARS:
        return text
    # Truncate to last complete XML tag before limit
    truncated = text[:_MAX_CONTEXT_CHARS]
    last_close = truncated.rfind("</")
    if last_close > 0:
        end = truncated.find(">", last_close)
        if end > 0:
            return truncated[: end + 1] + "\n</context>"
    return truncated + "\n</context>"
