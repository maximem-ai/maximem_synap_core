"""Shared utilities for context retrieval."""

from .scope import (
    ScopeHierarchy,
    ScopeResolver,
    build_scope_from_ids_mapped,
    build_scope_with_maca_mapped,
)
from .cache_keys import CacheKeyGenerator
from .confidence import ConfidenceConfig, meets_confidence_threshold, calculate_confidence_score
from .tokens import TokenCounter, get_token_counter, count_tokens

__all__ = [
    # Scope resolution
    "ScopeHierarchy",
    "ScopeResolver",
    "build_scope_from_ids_mapped",
    "build_scope_with_maca_mapped",
    # Cache keys
    "CacheKeyGenerator",
    # Confidence thresholds
    "ConfidenceConfig",
    "meets_confidence_threshold",
    "calculate_confidence_score",
    # Token counting
    "TokenCounter",
    "get_token_counter",
    "count_tokens",
]
