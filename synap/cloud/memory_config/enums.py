"""MACA enums."""

from enum import Enum
from typing import Optional


class AgentType(str, Enum):
    """Type of agent being configured."""
    INTRA_ORG = "intra_org"
    EXTERNAL_FACING = "external_facing"
    OFFERING = "offering"
    SPECIAL_PURPOSE = "special_purpose"
    CUSTOM = "custom"


class UserRelationship(str, Enum):
    """Relationship between agent and users."""
    B2B = "b2b"
    B2C = "b2c"
    INTERNAL = "internal"
    AGENT_TO_AGENT = "agent_to_agent"


class MemoryCategoryName(str, Enum):
    """Base memory category names."""
    FACTS = "facts"
    PREFERENCES = "preferences"
    EPISODES = "episodes"
    EMOTIONS = "emotions"
    TEMPORAL_EVENTS = "temporal_events"


class Priority(str, Enum):
    """Priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractionAggressiveness(str, Enum):
    """How aggressively to extract memories."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class PIIMode(str, Enum):
    """PII handling mode."""
    STRICT = "strict"
    RELAXED = "relaxed"
    OFF = "off"


class StorageMode(str, Enum):
    """Primary storage architecture."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class RetrievalStrategy(str, Enum):
    """Retrieval execution strategy."""
    CACHE_FIRST = "cache_first"
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_RERANK = "hybrid_rerank"
    FULL_SEARCH = "full_search"
    MULTI_PASS = "multi_pass"


class CompactionStrategy(str, Enum):
    """Context compaction strategy."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ChunkingStrategy(str, Enum):
    """Document chunking strategy."""
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    TURN_BASED = "turn_based"
    PAGE_AWARE = "page_aware"
    RECORD_BASED = "record_based"
    TOPIC_BASED = "topic_based"


class ConfigStatus(str, Enum):
    """Status of a memory architecture config."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MACAStep(str, Enum):
    """MACA reasoning steps."""
    UNDERSTAND = "understand"
    CLASSIFY = "classify"
    ANTICIPATE = "anticipate"
    DESIGN = "design"
    VALIDATE = "validate"
    REFINE = "refine"
    OUTPUT = "output"


class GraphSchemaType(str, Enum):
    """Type of graph schema purpose."""
    KNOWLEDGE = "knowledge"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


class DecayFunction(str, Enum):
    """Recency decay shape. The ranker (retrieval_manager) acts on these two."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class RetrievalModeName(str, Enum):
    """Values of ``retrieval_config.default_mode``.

    ``fast`` and ``accurate`` are what a fetch accepts. ``comprehensive`` is not
    a fetch mode, but the MACA design prompt offers it and validation_tools and
    conflict_detector special-case it, so stored Cloud configs can hold it.
    """
    FAST = "fast"
    ACCURATE = "accurate"
    COMPREHENSIVE = "comprehensive"


class ContextInjectionStyle(str, Enum):
    """How retrieved context is laid out for the agent (nothing reads it yet)."""
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    MINIMAL = "minimal"


class UserContextIsolation(str, Enum):
    """How a user id resolves against a customer id.

    ``maca_suggests`` is a Cloud value that Core rejects by its own rule
    (``core_rules.REMOVED_VALUES``), so it stays a member here.
    """
    STRICT = "strict"
    EQUALS_CUSTOMER = "equals_customer"
    MACA_SUGGESTS = "maca_suggests"


class TriggerSource(str, Enum):
    """What triggered MACA execution."""
    DASHBOARD = "dashboard"
    API = "api"
    AUTO = "auto"


class ConflictType(str, Enum):
    """Types of config conflicts."""
    SCHEMA_BREAKING = "schema_breaking"
    DATA_LOSS = "data_loss"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    CATEGORY_REMOVED = "category_removed"
    STORAGE_MODE_CHANGE = "storage_mode_change"


class Severity(str, Enum):
    """Conflict severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Retention ceilings ---
#
# The one pair of numbers anything with an opinion about retention reads. They
# live here, in the module that already holds MACA's shared non-schema values
# (DEFAULT_INSTANCE_TYPE_CONFIG and the scope defaults under it), because this
# is the only module in the package that every layer can import: it imports
# nothing of its own, so the validator in ``tools/`` and the prompt strings in
# ``workflow/`` can both read it, and ``tools/`` importing ``workflow/`` would
# be a cycle (``workflow/__init__`` pulls in ``nodes``, which imports
# ``tools``).
#
# Before this, retention was four opinions that did not agree:
#
#   * ``tools/validation_tools.py:147`` rejected anything above the one-year
#     ceiling with no carve-out of any kind,
#   * ``workflow/prompts.py:19`` told the model most agents get a year and a
#     knowledge base gets five,
#   * ``workflow/design_prompts.py:510`` said the same thing again, in its own
#     words, next to the field,
#   * ``templates/base/internal_knowledge.yaml`` shipped the five-year value
#     outright (that file is gone with the rest of the base templates, so three
#     sources have to agree rather than four).
#
# So a knowledge-base config that did exactly what two prompts asked of it
# failed the validator that ran immediately afterwards, and the failure read as
# the model's mistake rather than ours. The owner's decision is the one encoded
# here: **at most a year for everyone, five years for a knowledge agent.**
#
# Nothing else may restate either number. A literal in a prompt is the same bug
# in a different coat: the prompt is where the last disagreement started, and
# a string is not something a grep for a constant finds.

#: Retention ceiling for every archetype but one. Days.
RETENTION_MAX_DAYS = 365

#: Retention ceiling for a knowledge agent. A knowledge base is a corpus, not a
#: conversation: the documents it holds stay true for years and expiring them
#: on the ordinary clock empties the thing the agent exists to read.
RETENTION_MAX_DAYS_KNOWLEDGE = 1825

#: The archetype the higher ceiling belongs to, out of the five in
#: ``workflow/prompts.py:AGENT_ARCHETYPES``. Spelled once so the carve-out and
#: the classifier cannot drift on the spelling.
KNOWLEDGE_ARCHETYPE = "knowledge"


def retention_ceiling_days(agent_archetype: Optional[str]) -> int:
    """How many days this archetype may keep a memory for.

    ``None`` is the common case rather than an error: the ceiling is checked
    against a ``MemoryArchitectureConfig``, and the 183 configs already stored
    predate ``metadata.agent_archetype`` entirely. An unknown archetype gets
    the ordinary ceiling, which is the safe direction to be wrong in: a
    knowledge config whose archetype was never recorded is flagged for a human
    instead of being waved through at five years.

    Matched on a stripped, lower-cased value, the same normalisation
    ``_validate_agent_archetype`` applies in CLASSIFY, so ``"Knowledge"`` is
    this model's capitalisation rather than a different archetype.
    """
    archetype = (agent_archetype or "").strip().lower()
    if archetype == KNOWLEDGE_ARCHETYPE:
        return RETENTION_MAX_DAYS_KNOWLEDGE
    return RETENTION_MAX_DAYS


# --- Instance Type Configuration: Role Semantics + Scope Defaults ---
#
# Maps UserRelationship values to default role descriptions, scope defaults,
# and a derived default agent_type. user_relationship is the single source
# of truth for scoping decisions (user_context_isolation in particular).
# agent_type remains in the instance row for template selection and analytics
# but no longer drives scoping.

# ⚠ `maca_should_evaluate_cross_user` is False for every shape, and that is the
# point of this block.
#
# The safe value was always set here: `allow_cross_user_within_customer: False`.
# Then b2b and internal set the flag above to True, which puts
# "MACA should evaluate cross-user sharing: True" into the design prompt, whose
# instruction is "you MUST analyze the use-case ... If you recommend enabling,
# set allow_cross_user_retrieval=true".
#
# So the boundary defaulted closed and a language model was invited to reopen it
# on every B2B and internal instance. It said yes on every instance it was
# run for, and a console replay on one of them showed two users under one
# customer reading each other's records, including a medical allergy. The
# instances were labelled `user_context_isolation: strict`, which decides ID
# RESOLUTION and nothing about retrieval, so the config read as isolated and
# behaved as shared. MACA itself warned that the label was misleading and then
# emitted the combination anyway.
#
# Cross-user sharing is now a decision a person makes explicitly, never one a
# model recommends at instance creation. Setting this back to True re-opens a
# privacy boundary by prompt.
DEFAULT_INSTANCE_TYPE_CONFIG: dict[str, dict] = {
    # B2C: Customer = User = End consumer
    "b2c": {
        "default_agent_type": "external_facing",
        "role_semantics": {
            "client_description": "Company making the B2C agent",
            "customer_description": "End consumer of the agent",
            "user_description": "End consumer of the agent",
        },
        "scope_defaults": {
            # Not the authority: _assemble_config reads this off DESIGN's own
            # output, not from here, so nothing behavioural reads this key.
            # Kept aligned anyway, because a default that says the opposite of
            # the schema and the prompt is what the next person wires into a
            # prompt by accident.
            "allow_cross_instance_for_subagents": False,
            "customer_context_isolation": "strict",
            "user_context_isolation": "equals_customer",
            "allow_cross_user_within_customer": False,
            "allow_cross_customer": False,
            "maca_should_evaluate_cross_user": False,
            "maca_should_evaluate_cross_customer": False,
        },
    },
    # B2B: Customer = Business customer, User = Employees of the business customer
    "b2b": {
        "default_agent_type": "external_facing",
        "role_semantics": {
            "client_description": "Company making agent to serve business customers",
            "customer_description": "Business customers of client",
            "user_description": "Employees of business customer",
        },
        "scope_defaults": {
            # Not the authority: _assemble_config reads this off DESIGN's own
            # output, not from here, so nothing behavioural reads this key.
            # Kept aligned anyway, because a default that says the opposite of
            # the schema and the prompt is what the next person wires into a
            # prompt by accident.
            "allow_cross_instance_for_subagents": False,
            "customer_context_isolation": "strict",
            "user_context_isolation": "strict",
            "allow_cross_user_within_customer": False,
            "allow_cross_customer": False,
            "maca_should_evaluate_cross_user": False,
            "maca_should_evaluate_cross_customer": False,
        },
    },
    # Internal: Customer = User = Employees of the client (client using agent internally)
    "internal": {
        "default_agent_type": "intra_org",
        "role_semantics": {
            "client_description": "Company making agent for internal use by its employees",
            "customer_description": "Employees of client",
            "user_description": "Employees of client",
        },
        "scope_defaults": {
            # Not the authority: _assemble_config reads this off DESIGN's own
            # output, not from here, so nothing behavioural reads this key.
            # Kept aligned anyway, because a default that says the opposite of
            # the schema and the prompt is what the next person wires into a
            # prompt by accident.
            "allow_cross_instance_for_subagents": False,
            "customer_context_isolation": "strict",
            # Internal: customer == user == employee → B2C-shaped ID resolution.
            "user_context_isolation": "equals_customer",
            "allow_cross_user_within_customer": False,
            "allow_cross_customer": False,
            "maca_should_evaluate_cross_user": False,
            "maca_should_evaluate_cross_customer": False,
        },
    },
    # Agent-to-Agent: Technical integration, no human user
    "agent_to_agent": {
        "default_agent_type": "special_purpose",
        "role_semantics": {
            "client_description": "Company operating the orchestrator agent",
            "customer_description": "Orchestrator or parent agent",
            "user_description": "Sub-agent or peer agent caller",
        },
        "scope_defaults": {
            # Not the authority: _assemble_config reads this off DESIGN's own
            # output, not from here, so nothing behavioural reads this key.
            # Kept aligned anyway, because a default that says the opposite of
            # the schema and the prompt is what the next person wires into a
            # prompt by accident.
            "allow_cross_instance_for_subagents": False,
            "customer_context_isolation": "strict",
            "user_context_isolation": "strict",
            "allow_cross_user_within_customer": False,
            "allow_cross_customer": False,
            "maca_should_evaluate_cross_user": False,
            "maca_should_evaluate_cross_customer": False,
        },
    },
}


def resolve_instance_type_config(user_relationship: str) -> dict:
    """Resolve role semantics + scope defaults from user_relationship.

    Returns a dict with 'role_semantics', 'scope_defaults', and
    'default_agent_type' keys. Raises ValueError on unknown values —
    there is no silent fallback. Instance creation
    validates user_relationship as a required enum so this never fires
    in normal operation; it does catch misuse in tests and tooling.
    """
    if not user_relationship:
        raise ValueError(
            "user_relationship is required — must be one of: "
            f"{sorted(DEFAULT_INSTANCE_TYPE_CONFIG.keys())}"
        )
    key = user_relationship.lower()
    if key not in DEFAULT_INSTANCE_TYPE_CONFIG:
        raise ValueError(
            f"Unknown user_relationship: {user_relationship!r}. "
            f"Must be one of: {sorted(DEFAULT_INSTANCE_TYPE_CONFIG.keys())}"
        )
    return DEFAULT_INSTANCE_TYPE_CONFIG[key]


def derive_agent_type(user_relationship) -> str:
    """Derive the default agent_type from a user_relationship.

    Used at instance creation when agent_type isn't provided by the client,
    so the NOT NULL instance.agent_type column is always filled with a
    value that matches the scoping model.
    """
    value = (
        user_relationship.value
        if hasattr(user_relationship, "value")
        else str(user_relationship).lower()
    )
    config = DEFAULT_INSTANCE_TYPE_CONFIG.get(value)
    if not config:
        raise ValueError(
            f"Unknown user_relationship: {user_relationship!r}. "
            f"Must be one of: {sorted(DEFAULT_INSTANCE_TYPE_CONFIG.keys())}"
        )
    return config["default_agent_type"]
