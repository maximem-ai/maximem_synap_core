"""Request validation for ingestion."""

from dataclasses import dataclass
from typing import List, Optional, Union

from synap.cloud.agents.ingestion.models import IngestDocumentRequest, Scope


@dataclass
class ValidationResult:
    """Result of request validation."""
    is_valid: bool
    errors: List[str]


class RequestValidator:
    """
    Validates ingestion requests beyond Pydantic validation.
    
    Checks:
    - Content size limits
    - Encoding validity for bytes content
    - Scope completeness
    """

    def __init__(
        self,
        max_content_size_mb: int = 100,
        max_content_size_bytes: Optional[int] = None,
    ):
        """Initialize validator.
        
        Args:
            max_content_size_mb: Maximum content size in MB
            max_content_size_bytes: Maximum content size in bytes (overrides MB if set)
        """
        self._max_size = (
            max_content_size_bytes 
            if max_content_size_bytes is not None
            else max_content_size_mb * 1024 * 1024
        )

    async def validate(self, request: IngestDocumentRequest) -> ValidationResult:
        """Validate ingestion request.
        
        Args:
            request: Request to validate
            
        Returns:
            ValidationResult with is_valid flag and errors list
        """
        errors: List[str] = []

        # Content size validation
        content_size = self._get_content_size(request.content)
        if content_size > self._max_size:
            errors.append(
                f"Content size ({content_size} bytes) exceeds maximum "
                f"allowed size ({self._max_size} bytes)"
            )

        # Empty content validation
        if content_size == 0:
            errors.append("Content cannot be empty")

        # Encoding validation for bytes content
        if isinstance(request.content, bytes):
            if request.content_encoding and request.content_encoding == "base64":
                try:
                    import base64
                    base64.b64decode(request.content, validate=True)
                except Exception as e:
                    errors.append(f"Invalid base64 encoding: {str(e)}")

        # Scope validation
        scope_errors = self._validate_scope(request.scope)
        errors.extend(scope_errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _get_content_size(self, content: Union[str, bytes]) -> int:
        """Get content size in bytes."""
        if isinstance(content, bytes):
            return len(content)
        return len(content.encode('utf-8'))

    def _validate_scope(self, scope: Scope) -> List[str]:
        """Validate scope completeness.
        
        Args:
            scope: Scope to validate
            
        Returns:
            List of validation errors
        """
        errors: List[str] = []

        # All required UUIDs must be present
        if not scope.client_id:
            errors.append("Scope missing client_id")
        if not scope.instance_id:
            errors.append("Scope missing instance_id")
        if not scope.customer_id and getattr(scope, "scope_level", None) != "client":
            errors.append("Scope missing customer_id")

        return errors
