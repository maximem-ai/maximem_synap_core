"""Canonical relationship types used across ingestion and retrieval.

Convention: ALL relationship types MUST be UPPERCASE.
Neo4j relationship types are case-sensitive, so this ensures consistency
between what we extract, store, and query.

NOTE: This list is NOT exhaustive. The extraction prompt allows the LLM
to create domain-specific types (e.g., DIAGNOSED_WITH, REVIEWED) when
common types don't fit. This list serves as:
1. The common vocabulary shared across all instances
2. The fallback set for retrieval when no instance-specific registry exists
3. Documentation of well-known types with their directionality
"""

# Common relationship types — the LLM may create additional domain-specific
# types beyond this list. All types MUST be UPPERCASE.
CANONICAL_RELATIONSHIP_TYPES = [
    # Professional & Organizational
    "WORKS_AT",        # source=employee, target=org
    "MANAGES",         # source=manager, target=subordinate
    "REPORTS_TO",      # source=subordinate, target=manager
    "BELONGS_TO",      # source=member, target=group
    "PART_OF",         # source=part, target=whole
    "CREATED_BY",      # source=product, target=creator
    # Interpersonal
    "KNOWS",           # source=person, target=person (general acquaintance)
    "FAMILY_OF",       # source=person, target=relative (include relationship in properties)
    "CONTACTED",       # source=initiator, target=recipient
    # Usage & Preference
    "USES",            # source=user, target=tool/product
    "PREFERENCE",      # source=person, target=entity (preference signal)
    "INTERESTED_IN",   # source=person, target=concept/technology
    "PURCHASED",       # source=buyer, target=product/service
    "OWNS",            # source=owner, target=asset
    # Situational
    "HAS_CONDITION",   # source=person, target=condition
    "HAS_GOAL",        # source=person/org, target=goal
    "HAS_CONCERN",     # source=person/org, target=concern
    "SCHEDULED",       # source=person, target=event
    "DISCUSSED",       # source=entity, target=entity (co-discussed)
    "LOCATED_IN",      # source=entity, target=location
    # Fallback
    "RELATED_TO",      # catch-all when no specific type fits
]
