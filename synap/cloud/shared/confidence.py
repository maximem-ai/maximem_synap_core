"""
Confidence threshold utilities for early termination.

Based on a proven pattern:
- Stop when avg similarity >= 0.7 with >= 5 results
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ConfidenceConfig:
    """Configuration for confidence thresholds."""
    similarity_threshold: float = 0.7   # Minimum average similarity
    min_results: int = 5                 # Minimum results before checking
    top_k_for_avg: int = 10              # Top K results to average

    # Differential thresholds for graph traversal
    edge_safety_floor: float = 0.25     # Minimum score for entity→entity edges (traversal gate)
    memory_threshold: float = 0.35       # Threshold for memory edges (only output from graph)
    memory_boost: float = 1.0            # No boost needed — memories are the only graph output


def meets_confidence_threshold(
    results: List[Dict[str, Any]],
    config: ConfidenceConfig = None,
) -> bool:
    """
    Check if results meet confidence threshold for early termination.

    Args:
        results: List of result dicts with 'similarity' key
        config: Threshold configuration

    Returns:
        True if threshold met, False otherwise
    """
    if config is None:
        config = ConfidenceConfig()

    if not results or len(results) < config.min_results:
        return False

    # Get top K results by similarity
    sorted_results = sorted(
        results,
        key=lambda x: x.get("similarity", 0),
        reverse=True
    )[:config.top_k_for_avg]

    if len(sorted_results) < config.min_results:
        return False

    # Calculate average similarity
    avg_similarity = sum(
        r.get("similarity", 0) for r in sorted_results
    ) / len(sorted_results)

    return avg_similarity >= config.similarity_threshold


def calculate_confidence_score(results: List[Dict[str, Any]]) -> float:
    """
    Calculate overall confidence score for a result set.

    Returns value between 0 and 1.
    """
    if not results:
        return 0.0

    # Weight by similarity and confidence
    total_score = 0.0
    for r in results:
        similarity = r.get("similarity", 0)
        confidence = r.get("confidence", 0.5)
        total_score += similarity * confidence

    return min(1.0, total_score / len(results))
