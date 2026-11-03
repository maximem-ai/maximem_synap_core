"""File store provider abstraction."""

from .interface import FileStoreInterface
from .models import FileMetadata, FileQuery, FileListResult
from .registry import get_file_store, reset_file_store
from .adapters import LocalDiskAdapter

__all__ = [
    "FileStoreInterface",
    "FileMetadata",
    "FileQuery",
    "FileListResult",
    "get_file_store",
    "reset_file_store",
    "LocalDiskAdapter",
]
