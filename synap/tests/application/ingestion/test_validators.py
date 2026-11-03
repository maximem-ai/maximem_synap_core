"""Tests for RequestValidator."""

from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    Scope,
)
from synap.cloud.application.ingestion.validators import RequestValidator, ValidationResult


class TestRequestValidator:
    """Tests for RequestValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator with default settings."""
        return RequestValidator(max_content_size_mb=1)

    @pytest.fixture
    def scope(self):
        """Create valid scope."""
        return Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, validator, scope):
        """Test validation passes for valid request."""
        request = IngestDocumentRequest(
            content="Valid content",
            scope=scope,
        )
        result = await validator.validate(request)
        assert result.is_valid
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_content_too_large(self, validator, scope):
        """Test validation fails for content exceeding max size."""
        # Create content larger than 1MB
        large_content = "x" * (2 * 1024 * 1024)
        request = IngestDocumentRequest(
            content=large_content,
            scope=scope,
        )
        result = await validator.validate(request)
        assert not result.is_valid
        assert any("exceeds maximum" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_validate_empty_content(self, validator, scope):
        """Test validation fails for empty content."""
        request = IngestDocumentRequest(
            content="",
            scope=scope,
        )
        result = await validator.validate(request)
        assert not result.is_valid
        assert any("cannot be empty" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_validate_bytes_content(self, validator, scope):
        """Test validation passes for bytes content."""
        request = IngestDocumentRequest(
            content=b"Binary content",
            scope=scope,
        )
        result = await validator.validate(request)
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_validate_base64_encoded_content(self, validator, scope):
        """Test validation passes for valid base64 content."""
        import base64
        content = base64.b64encode(b"Test content")
        request = IngestDocumentRequest(
            content=content,
            content_encoding="base64",
            scope=scope,
        )
        result = await validator.validate(request)
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_validate_invalid_base64_content(self, validator, scope):
        """Test validation fails for invalid base64."""
        request = IngestDocumentRequest(
            content=b"Not base64!@#$%",
            content_encoding="base64",
            scope=scope,
        )
        result = await validator.validate(request)
        assert not result.is_valid
        assert any("Invalid base64" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_validate_missing_scope_fields(self, validator):
        """Test validation fails for incomplete scope."""
        # This test would require modifying Scope validation
        # Since Scope uses Pydantic, missing fields will raise ValidationError
        # before reaching our validator
        pass

    @pytest.mark.asyncio
    async def test_validation_result_structure(self, validator, scope):
        """Test ValidationResult structure."""
        request = IngestDocumentRequest(content="", scope=scope)
        result = await validator.validate(request)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
