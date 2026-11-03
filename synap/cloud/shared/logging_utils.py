"""Logging utilities for consistent, readable log formatting."""

import json
from typing import Any, Dict, List, Optional


def truncate_string(s: str, max_length: int = 500) -> str:
    """Truncate a string for logging, adding ellipsis if truncated."""
    if len(s) <= max_length:
        return s
    return s[:max_length] + f"... (truncated, {len(s)} total chars)"


def truncate_dict(d: Dict[str, Any], max_depth: int = 3, max_items: int = 10, max_str_len: int = 200) -> Dict[str, Any]:
    """
    Truncate a dictionary for logging.

    Args:
        d: Dictionary to truncate (None returns empty dict)
        max_depth: Maximum nesting depth
        max_items: Maximum number of items per dict/list
        max_str_len: Maximum string length

    Returns:
        Truncated dictionary suitable for logging
    """
    # Handle None case
    if d is None:
        return {"result": None}

    if max_depth == 0:
        return {"...": "max depth reached"}

    result = {}
    items_count = 0

    for k, v in d.items():
        if items_count >= max_items:
            result[f"... +{len(d) - items_count} more"] = "truncated"
            break

        if isinstance(v, dict):
            result[k] = truncate_dict(v, max_depth - 1, max_items, max_str_len)
        elif isinstance(v, list):
            result[k] = truncate_list(v, max_depth - 1, max_items, max_str_len)
        elif isinstance(v, str):
            result[k] = truncate_string(v, max_str_len)
        else:
            result[k] = v

        items_count += 1

    return result


def truncate_list(lst: List[Any], max_depth: int = 3, max_items: int = 10, max_str_len: int = 200) -> List[Any]:
    """
    Truncate a list for logging.

    Args:
        lst: List to truncate
        max_depth: Maximum nesting depth
        max_items: Maximum number of items
        max_str_len: Maximum string length

    Returns:
        Truncated list suitable for logging
    """
    if max_depth == 0:
        return ["... max depth reached"]

    result = []

    for i, item in enumerate(lst[:max_items]):
        if isinstance(item, dict):
            result.append(truncate_dict(item, max_depth - 1, max_items, max_str_len))
        elif isinstance(item, list):
            result.append(truncate_list(item, max_depth - 1, max_items, max_str_len))
        elif isinstance(item, str):
            result.append(truncate_string(item, max_str_len))
        else:
            result.append(item)

    if len(lst) > max_items:
        result.append(f"... +{len(lst) - max_items} more items")

    return result


def safe_json_dumps(obj: Any, max_length: int = 1000) -> str:
    """
    Safely convert an object to JSON string for logging.

    Args:
        obj: Object to convert
        max_length: Maximum length of resulting string

    Returns:
        JSON string, truncated if necessary
    """
    try:
        json_str = json.dumps(obj, default=str, indent=None)
        return truncate_string(json_str, max_length)
    except Exception as e:
        return f"<unable to serialize: {str(e)}>"


def format_sql_query(sql: str, params: Optional[tuple] = None) -> str:
    """
    Format SQL query for logging.

    Args:
        sql: SQL query string
        params: Query parameters

    Returns:
        Formatted query string
    """
    # Clean up whitespace
    formatted = " ".join(sql.split())

    # Truncate if too long
    if len(formatted) > 500:
        formatted = formatted[:500] + "..."

    if params:
        try:
            # Show first few params
            param_str = str(params) if len(str(params)) < 200 else str(params)[:200] + "..."
            return f"{formatted} [params: {param_str}]"
        except:
            return formatted

    return formatted


def log_data_sample(data: List[Any], sample_size: int = 3, label: str = "items") -> str:
    """
    Create a log-friendly sample of data items.

    Args:
        data: List of data items
        sample_size: Number of items to include in sample
        label: Label for the data type

    Returns:
        Formatted string showing sample and count
    """
    if not data:
        return f"0 {label}"

    count = len(data)
    sample = data[:sample_size]

    # Truncate each sample item
    truncated_sample = []
    for item in sample:
        if isinstance(item, dict):
            truncated_sample.append(truncate_dict(item, max_depth=2, max_items=5, max_str_len=100))
        elif isinstance(item, str):
            truncated_sample.append(truncate_string(item, 100))
        else:
            truncated_sample.append(item)

    sample_str = safe_json_dumps(truncated_sample, max_length=500)

    if count > sample_size:
        return f"{count} {label} (showing first {sample_size}): {sample_str}"
    else:
        return f"{count} {label}: {sample_str}"
