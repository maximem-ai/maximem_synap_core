"""Tests for Ingestion models."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from synap.cloud.agents.ingestion.models import (
    # Enums
    ContentNature,
    DocumentSource,
    DocumentType,
    IngestPriority,
    IngestStatus,
    MemoryDestination,
    MemoryType,
    PipelineStage,
    # Input models
    ConvertSessionRequest,
    IngestBatchRequest,
    IngestDocumentRequest,
    IngestFromListeningAgentRequest,
    Scope,
    # Output models
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelationship,
    IngestBatchResponse,
    IngestResponse,
    IngestionTrace,
    # Pipeline models
    CategoryAssignment,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    # State models
    IngestionPipelineState,
)


class TestEnums:
    """Tests for enum definitions."""

    def test_document_type_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.CONVERSATION == "conversation"
        assert DocumentType.DOCUMENT == "document"
        assert DocumentType.EMAIL == "email"
        assert DocumentType.PDF == "pdf"
        assert DocumentType.IMAGE == "image"
        assert DocumentType.AUDIO == "audio"
        assert DocumentType.MEETING_TRANSCRIPT == "meeting_transcript"
        assert DocumentType.INTERNAL_COMM == "internal_comm"

    def test_document_source_values(self):
        """Test DocumentSource enum values."""
        assert DocumentSource.API == "api"
        assert DocumentSource.GDRIVE == "gdrive"
        assert DocumentSource.UPLOADED == "uploaded"
        assert DocumentSource.GMAIL == "gmail"
        assert DocumentSource.SLACK == "slack"
        assert DocumentSource.ZENDESK == "zendesk"
        assert DocumentSource.SALESFORCE == "salesforce"
        assert DocumentSource.LISTENING_AGENT == "listening_agent"

    def test_ingest_priority_values(self):
        """Test IngestPriority enum values."""
        assert IngestPriority.REALTIME == "realtime"
        assert IngestPriority.EXPLICIT == "explicit"
        assert IngestPriority.BOOTSTRAP == "bootstrap"

    def test_ingest_status_values(self):
        """Test IngestStatus enum values."""
        assert IngestStatus.QUEUED == "queued"
        assert IngestStatus.PROCESSING == "processing"
        assert IngestStatus.COMPLETED == "completed"
        assert IngestStatus.FAILED == "failed"
        assert IngestStatus.PARTIAL_SUCCESS == "partial_success"

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.FACTS == "facts"
        assert MemoryType.PREFERENCES == "preferences"
        assert MemoryType.EPISODES == "episodes"
        assert MemoryType.EMOTIONS == "emotions"
        assert MemoryType.TEMPORAL_EVENTS == "temporal_events"

    def test_memory_destination_values(self):
        """Test MemoryDestination enum values."""
        assert MemoryDestination.MEMORIES == "memories"
        assert MemoryDestination.CANDIDATES == "candidates"
        assert MemoryDestination.CLIENT_KNOWLEDGE == "client_knowledge"
        assert MemoryDestination.CUSTOMER_KNOWLEDGE == "customer_knowledge"
        assert MemoryDestination.SHORT_TERM == "short_term"

    def test_content_nature_values(self):
        """Test ContentNature enum values."""
        assert ContentNature.SUBSTANTIVE == "substantive"
        assert ContentNature.TRIVIAL == "trivial"
        assert ContentNature.SPAM == "spam"
        assert ContentNature.DUPLICATE == "duplicate"

    def test_pipeline_stage_values(self):
        """Test PipelineStage enum values."""
        assert PipelineStage.STAGE_1_QUALIFICATION == "stage_1_qualification"
        assert PipelineStage.STAGE_2_CATEGORIZATION == "stage_2_categorization"
        assert PipelineStage.STAGE_3_EXTRACTION == "stage_3_extraction"
        assert PipelineStage.STAGE_4_ORGANIZATION == "stage_4_organization"


class TestScope:
    """Tests for Scope model."""

    def test_scope_creation(self):
        """Test Scope model creation."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        assert isinstance(scope.client_id, UUID)
        assert isinstance(scope.instance_id, UUID)
        assert isinstance(scope.customer_id, UUID)
        assert scope.user_id is None

    def test_scope_with_user_id(self):
        """Test Scope with optional user_id."""
        user_id = uuid4()
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
            user_id=user_id,
        )
        assert scope.user_id == user_id

    def test_scope_serialization(self):
        """Test Scope serialization."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        data = scope.model_dump()
        assert "client_id" in data
        assert "instance_id" in data
        assert "customer_id" in data


class TestIngestDocumentRequest:
    """Tests for IngestDocumentRequest model."""

    def test_minimal_request(self):
        """Test minimal request with required fields only."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = IngestDocumentRequest(
            content="Test content",
            scope=scope,
        )
        assert request.content == "Test content"
        assert request.scope == scope
        assert isinstance(request.request_id, str)
        assert request.priority == IngestPriority.EXPLICIT
        assert request.mode == "long-range"

    def test_request_with_all_fields(self):
        """Test request with all fields populated."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        doc_created = datetime(2025, 1, 1, 12, 0, 0)
        request = IngestDocumentRequest(
            request_id="custom-request-id",
            document_id="doc-123",
            content="Test content",
            content_encoding="utf-8",
            document_type=DocumentType.EMAIL,
            source=DocumentSource.GMAIL,
            document_created_at=doc_created,
            scope=scope,
            priority=IngestPriority.REALTIME,
            mode="fast",
            metadata={"key": "value"},
            security_info={"classification": "internal"},
        )
        assert request.request_id == "custom-request-id"
        assert request.document_id == "doc-123"
        assert request.document_type == DocumentType.EMAIL
        assert request.source == DocumentSource.GMAIL
        assert request.document_created_at == doc_created
        assert request.priority == IngestPriority.REALTIME
        assert request.metadata == {"key": "value"}

    def test_request_auto_generates_id(self):
        """Test that request_id is auto-generated."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request1 = IngestDocumentRequest(content="Test", scope=scope)
        request2 = IngestDocumentRequest(content="Test", scope=scope)
        assert request1.request_id != request2.request_id

    def test_request_with_bytes_content(self):
        """Test request with bytes content."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = IngestDocumentRequest(
            content=b"binary content",
            scope=scope,
        )
        assert isinstance(request.content, bytes)


class TestIngestBatchRequest:
    """Tests for IngestBatchRequest model."""

    def test_batch_request_creation(self):
        """Test batch request creation."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        doc1 = IngestDocumentRequest(content="Doc 1", scope=scope)
        doc2 = IngestDocumentRequest(content="Doc 2", scope=scope)
        batch = IngestBatchRequest(
            documents=[doc1, doc2],
            scope=scope,
        )
        assert len(batch.documents) == 2
        assert batch.priority == IngestPriority.BOOTSTRAP
        assert batch.fail_fast is False

    def test_batch_request_empty_documents(self):
        """Test batch request with empty documents list."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        batch = IngestBatchRequest(documents=[], scope=scope)
        assert len(batch.documents) == 0


class TestIngestFromListeningAgentRequest:
    """Tests for IngestFromListeningAgentRequest model."""

    def test_listening_agent_request(self):
        """Test listening agent request creation."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = IngestFromListeningAgentRequest(
            session_id="session-123",
            conversation_id="conv-456",
            content="User said something important",
            content_type="turn",
            scope=scope,
            destination="short_term",
        )
        assert request.session_id == "session-123"
        assert request.conversation_id == "conv-456"
        assert request.content_type == "turn"
        assert request.priority == IngestPriority.REALTIME
        assert request.pre_qualified is False

    def test_listening_agent_request_pre_qualified(self):
        """Test pre-qualified listening agent request."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = IngestFromListeningAgentRequest(
            session_id="session-123",
            conversation_id="conv-456",
            content="Important fact",
            content_type="compacted_context",
            scope=scope,
            destination="long_term",
            pre_qualified=True,
            qualification_result={"nature": "substantive"},
        )
        assert request.pre_qualified is True
        assert request.qualification_result == {"nature": "substantive"}


class TestConvertSessionRequest:
    """Tests for ConvertSessionRequest model."""

    def test_convert_session_request(self):
        """Test session conversion request."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = ConvertSessionRequest(
            session_id="session-789",
            scope=scope,
        )
        assert request.session_id == "session-789"
        assert request.priority == IngestPriority.REALTIME
        assert isinstance(request.request_id, str)


class TestExtractedMemory:
    """Tests for ExtractedMemory model."""

    def test_extracted_memory_creation(self):
        """Test extracted memory creation."""
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="User prefers Python over JavaScript",
            confidence=0.85,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
        )
        assert memory.memory_type == MemoryType.FACTS
        assert memory.confidence == 0.85
        assert memory.destination == MemoryDestination.MEMORIES
        assert memory.pii_obfuscated is False

    def test_extracted_memory_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.PREFERENCES,
            content="Test",
            confidence=0.5,
            destination=MemoryDestination.CANDIDATES,
            source_document_id="doc-1",
        )
        assert memory.confidence == 0.5

        # Invalid confidence - below 0
        with pytest.raises(ValidationError):
            ExtractedMemory(
                memory_id=uuid4(),
                memory_type=MemoryType.FACTS,
                content="Test",
                confidence=-0.1,
                destination=MemoryDestination.MEMORIES,
                source_document_id="doc-1",
            )

        # Invalid confidence - above 1
        with pytest.raises(ValidationError):
            ExtractedMemory(
                memory_id=uuid4(),
                memory_type=MemoryType.FACTS,
                content="Test",
                confidence=1.5,
                destination=MemoryDestination.MEMORIES,
                source_document_id="doc-1",
            )

    def test_extracted_memory_with_pii(self):
        """Test extracted memory with PII obfuscation."""
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="User's email is [REDACTED]",
            confidence=0.9,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            pii_obfuscated=True,
            pii_fields_obfuscated=["email"],
        )
        assert memory.pii_obfuscated is True
        assert "email" in memory.pii_fields_obfuscated


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_entity_creation(self):
        """Test entity creation."""
        entity = ExtractedEntity(
            entity_id="entity-123",
            entity_type="person",
            name="John Doe",
        )
        assert entity.entity_id == "entity-123"
        assert entity.entity_type == "person"
        assert entity.name == "John Doe"
        assert entity.canonical_name is None

    def test_entity_with_canonicalization(self):
        """Test entity with canonical name."""
        entity = ExtractedEntity(
            entity_id="entity-456",
            entity_type="company",
            name="Acme Corp",
            canonical_name="Acme Corporation",
            properties={"industry": "technology"},
        )
        assert entity.canonical_name == "Acme Corporation"
        assert entity.properties["industry"] == "technology"


class TestExtractedRelationship:
    """Tests for ExtractedRelationship model."""

    def test_relationship_creation(self):
        """Test relationship creation."""
        rel = ExtractedRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="works_at",
        )
        assert rel.source_entity_id == "entity-1"
        assert rel.target_entity_id == "entity-2"
        assert rel.relationship_type == "works_at"


class TestIngestResponse:
    """Tests for IngestResponse model."""

    def test_response_creation(self):
        """Test ingest response creation."""
        queued = datetime.utcnow()
        response = IngestResponse(
            request_id="req-123",
            ingestion_id=uuid4(),
            status=IngestStatus.QUEUED,
            document_id="doc-456",
            queued_at=queued,
        )
        assert response.request_id == "req-123"
        assert response.status == IngestStatus.QUEUED
        assert response.memories_created == 0
        assert response.error_message is None

    def test_response_completed(self):
        """Test completed response."""
        response = IngestResponse(
            request_id="req-123",
            ingestion_id=uuid4(),
            status=IngestStatus.COMPLETED,
            document_id="doc-456",
            queued_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            memories_created=5,
            candidates_created=2,
            entities_created=3,
            relationships_created=1,
        )
        assert response.status == IngestStatus.COMPLETED
        assert response.memories_created == 5
        assert response.candidates_created == 2


class TestIngestBatchResponse:
    """Tests for IngestBatchResponse model."""

    def test_batch_response_creation(self):
        """Test batch response creation."""
        response = IngestBatchResponse(
            request_id="batch-123",
            batch_id=uuid4(),
            status=IngestStatus.PROCESSING,
            total_documents=10,
            queued_at=datetime.utcnow(),
        )
        assert response.total_documents == 10
        assert response.succeeded == 0
        assert response.failed == 0
        assert len(response.document_results) == 0


class TestIngestionTrace:
    """Tests for IngestionTrace model."""

    def test_trace_creation(self):
        """Test trace creation."""
        trace = IngestionTrace(
            trace_id=uuid4(),
            ingestion_id=uuid4(),
            stage="stage_1_qualification",
            input_hash="abc123",
            started_at=datetime.utcnow(),
        )
        assert trace.stage == "stage_1_qualification"
        assert trace.pii_detected is False
        assert trace.processing_time_ms == 0

    def test_trace_with_decisions(self):
        """Test trace with decision information."""
        trace = IngestionTrace(
            trace_id=uuid4(),
            ingestion_id=uuid4(),
            stage="stage_3_extraction",
            input_hash="def456",
            output_hash="ghi789",
            qualification_decision="process",
            content_nature=ContentNature.SUBSTANTIVE,
            categories_assigned=["technical", "billing"],
            memory_types_extracted=["facts", "preferences"],
            entities_extracted=5,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        assert trace.qualification_decision == "process"
        assert trace.content_nature == ContentNature.SUBSTANTIVE
        assert len(trace.categories_assigned) == 2
        assert trace.entities_extracted == 5


class TestPipelineModels:
    """Tests for pipeline stage models."""

    def test_stage1_output(self):
        """Test Stage1Output creation."""
        output = Stage1Output(
            input_id="input-123",
            document_type=DocumentType.EMAIL,
            source=DocumentSource.GMAIL,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
        )
        assert output.document_type == DocumentType.EMAIL
        assert output.should_process is True
        assert output.is_chunked is False

    def test_stage1_output_chunked(self):
        """Test Stage1Output with chunking info."""
        output = Stage1Output(
            input_id="input-456",
            document_type=DocumentType.PDF,
            source=DocumentSource.UPLOADED,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
            is_chunked=True,
            chunk_index=2,
            total_chunks=5,
            preceding_summary="Summary of previous chunks",
        )
        assert output.is_chunked is True
        assert output.chunk_index == 2
        assert output.total_chunks == 5

    def test_category_assignment(self):
        """Test CategoryAssignment creation."""
        category = CategoryAssignment(
            category="Technical Issue",
            subcategory="API Error",
            confidence=0.92,
        )
        assert category.category == "Technical Issue"
        assert category.confidence == 0.92

    def test_category_assignment_confidence_validation(self):
        """Test category confidence validation."""
        with pytest.raises(ValidationError):
            CategoryAssignment(
                category="Test",
                subcategory="Test",
                confidence=1.5,
            )

    def test_stage2_output(self):
        """Test Stage2Output creation."""
        output = Stage2Output(
            input_id="input-789",
            categories=[
                CategoryAssignment(
                    category="Technical",
                    subcategory="Bug",
                    confidence=0.85,
                )
            ],
            extraction_priorities=[MemoryType.FACTS, MemoryType.TEMPORAL_EVENTS],
        )
        assert len(output.categories) == 1
        assert len(output.extraction_priorities) == 2

    def test_stage3_output(self):
        """Test Stage3Output creation."""
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Test fact",
            confidence=0.8,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-1",
        )
        entity = ExtractedEntity(
            entity_id="entity-1",
            entity_type="person",
            name="Alice",
        )
        relationship = ExtractedRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="knows",
        )
        output = Stage3Output(
            input_id="input-999",
            memories=[memory],
            entities=[entity],
            relationships=[relationship],
        )
        assert len(output.memories) == 1
        assert len(output.entities) == 1
        assert len(output.relationships) == 1

    def test_stage4_output(self):
        """Test Stage4Output creation."""
        output = Stage4Output(
            input_id="input-111",
            rdbms_memory_ids=["mem-1", "mem-2"],
            graph_entity_ids=["entity-1"],
            vector_ids=["vec-1", "vec-2"],
            s3_original_path="s3://bucket/doc.pdf",
        )
        assert len(output.rdbms_memory_ids) == 2
        assert len(output.graph_entity_ids) == 1
        assert output.s3_original_path == "s3://bucket/doc.pdf"


class TestIngestionPipelineState:
    """Tests for IngestionPipelineState model."""

    def test_pipeline_state_creation(self):
        """Test pipeline state creation."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        state = IngestionPipelineState(
            ingestion_id=uuid4(),
            request_id="req-123",
            document_id="doc-456",
            content="Test content",
            content_hash="hash123",
            scope=scope,
        )
        assert state.request_id == "req-123"
        assert state.document_id == "doc-456"
        assert state.current_stage == "pending"
        assert state.error is None
        assert len(state.traces) == 0

    def test_pipeline_state_with_stages(self):
        """Test pipeline state with stage outputs."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        stage1 = Stage1Output(
            input_id="input-1",
            document_type=DocumentType.DOCUMENT,
            source=DocumentSource.API,
            nature=ContentNature.SUBSTANTIVE,
            should_process=True,
        )
        state = IngestionPipelineState(
            ingestion_id=uuid4(),
            request_id="req-789",
            document_id="doc-789",
            content="Content",
            content_hash="hash789",
            scope=scope,
            stage1_output=stage1,
            current_stage="stage_2_categorization",
        )
        assert state.stage1_output is not None
        assert state.stage1_output.document_type == DocumentType.DOCUMENT
        assert state.current_stage == "stage_2_categorization"


class TestModelSerialization:
    """Tests for model serialization."""

    def test_scope_serialization_roundtrip(self):
        """Test Scope serialization and deserialization."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        data = scope.model_dump()
        restored = Scope(**data)
        assert restored.client_id == scope.client_id
        assert restored.instance_id == scope.instance_id

    def test_extracted_memory_serialization(self):
        """Test ExtractedMemory serialization."""
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.PREFERENCES,
            content="User likes dark mode",
            confidence=0.95,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-1",
        )
        data = memory.model_dump()
        assert "memory_type" in data
        assert "confidence" in data
        assert data["memory_type"] == "preferences"

    def test_ingest_request_serialization(self):
        """Test IngestDocumentRequest serialization."""
        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        request = IngestDocumentRequest(
            content="Test content",
            scope=scope,
            priority=IngestPriority.REALTIME,
        )
        data = request.model_dump()
        assert "content" in data
        assert "scope" in data
        assert data["priority"] == "realtime"
