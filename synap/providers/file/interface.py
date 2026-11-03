"""Base interface for file/object storage."""

from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from .models import FileMetadata, FileQuery, FileListResult


class FileStoreInterface(ABC):
    """
    Base interface for file/object storage.
    Multi-tenancy: prefix per instance_id.
    """

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def upload(self, key: str, data: BinaryIO, content_type: str = None, metadata: dict = None) -> FileMetadata:
        """Upload file."""
        pass

    async def write(self, path: str, content: bytes, metadata: dict = None) -> FileMetadata:
        """Write bytes content to storage."""
        pass

    @abstractmethod
    def download(self, key: str) -> BinaryIO:
        """Download file."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete file."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[FileMetadata]:
        """Get file metadata."""
        pass

    @abstractmethod
    def list_files(self, query: FileQuery) -> FileListResult:
        """List files with optional prefix filter."""
        pass

    @abstractmethod
    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for direct access."""
        pass
