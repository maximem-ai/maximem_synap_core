"""
Timestamp Parsing Utility.

Handles parsing of various timestamp formats and normalizes them to datetime objects.
Supports formats like "15 January", "January 15, 2024", "Monday, 15", and ISO timestamps.

Ported from maximem_memoryinfra with adaptations for synap pipeline.
"""

from typing import Optional
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)

# Month name mappings (full and abbreviated)
MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}

# Day name mappings
DAY_NAMES = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1, 'tues': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6,
}


def parse_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """
    Parse timestamp string in various formats to datetime object.

    Supported formats:
    - "15 January" or "January 15" (assumes current year)
    - "15 January 2024" or "January 15, 2024"
    - "15 Jan" or "Jan 15" (assumes current year)
    - "Monday, 15" or "Monday 15" (assumes current month/year)
    - ISO timestamp strings (e.g., "2024-01-15T10:30:00Z")
    - US date formats: MM/DD/YYYY, YYYY-MM-DD

    Args:
        timestamp_str: Timestamp string in various formats, or None/empty

    Returns:
        datetime object (naive, UTC assumed), or None if invalid
        Partial dates are normalized to start of day (00:00:00)
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        return None

    timestamp_str = timestamp_str.strip().strip('"\'`')
    if not timestamp_str:
        return None

    # Handle null-like values from LLM output
    if timestamp_str.lower() in ("null", "none", "n/a", "undefined"):
        return None

    # Strip trailing LLM commentary like "(approximate, recently)" or "(estimated)"
    # Extract the leading ISO-like portion before any parenthetical or text suffix
    iso_match = re.match(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)', timestamp_str)
    if iso_match:
        timestamp_str = iso_match.group(1)

    # Try ISO format first (exact timestamp)
    try:
        parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return parsed.replace(tzinfo=None)
    except (ValueError, AttributeError):
        pass

    # Try using dateutil parser (handles many formats)
    try:
        from dateutil import parser as date_parser
        parsed = date_parser.parse(timestamp_str, default=datetime.utcnow())
        return datetime(parsed.year, parsed.month, parsed.day, 0, 0, 0)
    except (ImportError, ValueError, TypeError):
        pass

    # Manual parsing for specific formats
    timestamp_lower = timestamp_str.lower()

    # Try to match day + month patterns
    day_month_patterns = [
        r'(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?',  # "15 January" or "15 January 2024"
        r'([a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?',  # "January 15" or "January 15, 2024"
    ]

    for pattern in day_month_patterns:
        match = re.match(pattern, timestamp_lower)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[0].isdigit():
                        day = int(groups[0])
                        month_name = groups[1]
                        year_str = groups[2]
                    else:
                        month_name = groups[0]
                        day = int(groups[1])
                        year_str = groups[2]

                    month = MONTH_NAMES.get(month_name)
                    if month:
                        year = int(year_str) if year_str else datetime.utcnow().year
                        return datetime(year, month, day, 0, 0, 0)
            except (ValueError, TypeError):
                continue

    # Try day name pattern: "Monday, 15" or "Monday 15"
    day_name_pattern = r'([a-z]+)[,\s]+(\d{1,2})'
    match = re.match(day_name_pattern, timestamp_lower)
    if match:
        try:
            day_name = match.group(1)
            day = int(match.group(2))

            if day_name in DAY_NAMES:
                now = datetime.utcnow()
                return datetime(now.year, now.month, day, 0, 0, 0)
        except (ValueError, TypeError):
            pass

    logger.warning(f"Failed to parse timestamp: {timestamp_str}")
    return None
