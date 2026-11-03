"""Standard models for file store operations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class FileMetadata:
    """File metadata."""
    key: str
    size_bytes: int
    content_type: str
    last_modified: datetime
    custom_metadata: Optional[Dict[str, str]] = None


@dataclass
class FileQuery:
    """File listing query."""
    prefix: Optional[str] = None
    limit: int = 1000
    continuation_token: Optional[str] = None


@dataclass
class FileListResult:
    """File listing result."""
    files: List[FileMetadata]
    continuation_token: Optional[str] = None
    is_truncated: bool = False
