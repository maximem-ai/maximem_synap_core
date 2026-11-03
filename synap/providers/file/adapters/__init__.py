"""File store adapters. The S3 adapter is imported by the registry only when it is chosen."""

from .local import LocalDiskAdapter

__all__ = [
    "LocalDiskAdapter",
]
