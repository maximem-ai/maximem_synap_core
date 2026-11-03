"""Tests for Storage Coordinator."""

from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    ExtractedEntity,
    ExtractedRelationship,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.storage.coordinator import (
    StorageCoordinator,
    RDBMSPersistResult,
    GraphPersistResult,
    VectorPersistResult,
)


class MockRDBMSWrapper:
    """Mock RDBMS wrapper."""

    def __init__(self):
        self.memories = []
        self.candidates = []
        self.entity_links = []
        self.source_documents = []

    async def insert_memory(self, memory, scope, document_id):
        self.memories.append((memory, scope, document_id))
        return memory.memory_id

    async def insert_candidate(self, memory, scope, document_id):
        self.candidates.append((memory, scope, document_id))
        return memory.memory_id

    async def insert_entity_reference(self, memory_id, entity_id, scope):
        self.entity_links.append((memory_id, entity_id, scope))

    async def insert_memory_source_documents(self, memory_id, source_document_ids):
        # KU v2 M:N source-document attribution written during ingestion.
        self.source_documents.append((memory_id, source_document_ids))

    async def insert_client_knowledge(self, memory, scope, document_id):
        return memory.memory_id

    async def insert_customer_knowledge(self, memory, scope, document_id):
        return memory.memory_id


class MockGraphWrapper:
    """Mock Graph wrapper."""

    def __init__(self):
        self.namespaces = []
        self.entities = []
        self.relationships = []
        self.attached_memories = []

    async def ensure_namespace(self, scope):
        self.namespaces.append(scope)

    async def upsert_entity_node(self, entity, scope, merge_similar=False, embedding=None):
        self.entities.append((entity, scope))
        return entity.entity_id

    async def create_relationship(self, relationship, scope, embedding=None):
        self.relationships.append((relationship, scope))
        return f"{relationship.source_entity_id}_{relationship.relationship_type}_{relationship.target_entity_id}"

    async def attach_memories_to_relationships(
        self,
        memories,
        relationships,
        relationship_edge_ids,
        scope,
        memory_embeddings=None,
    ):
        self.attached_memories.append((memories, relationships, relationship_edge_ids, scope))
        return len(relationship_edge_ids)


class MockVectorWrapper:
    """Mock Vector wrapper."""

    def __init__(self):
        self.namespaces = []
        self.batches = []

    async def ensure_namespace(self, scope, dimension=384):
        self.namespaces.append((scope, dimension))

    async def upsert_memories_batch(self, memories, embeddings, scope, ere_contexts=None, remapped_entities=None):
        self.batches.append((memories, embeddings, scope, ere_contexts))
        return [str(m.memory_id) for m in memories]


class MockEmbeddingClient:
    """Mock Embedding client."""

    # Coordinator reads embedding_client.dimension when ensuring the vector
    # namespace; keep it consistent with the 384-dim vectors below.
    dimension = 384

    def __init__(self):
        self.embed_calls = []
        self.batch_calls = []

    async def embed(self, text):
        self.embed_calls.append(text)
        return [0.1] * 384

    async def embed_batch(self, texts):
        self.batch_calls.append(texts)
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def scope():
    """Create a test scope."""
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


@pytest.fixture
def coordinator():
    """Create a coordinator with mock providers."""
    rdbms = MockRDBMSWrapper()
    graph = MockGraphWrapper()
    vector = MockVectorWrapper()
    embedding = MockEmbeddingClient()
    
    return StorageCoordinator(rdbms, graph, vector, embedding), rdbms, graph, vector, embedding


class TestRDBMSPersistence:
    """Tests for RDBMS persistence."""

    @pytest.mark.asyncio
    async def test_persist_memories(self, scope, coordinator):
        """Test persisting memories to RDBMS."""
        coord, rdbms, _, _, _ = coordinator

        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Test memory",
            confidence=0.85,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1"],
        )

        result = await coord.persist_rdbms(
            memories=[memory],
            candidates=[],
            entities=[],
            scope=scope,
            document_id="doc-123",
        )

        assert len(result.memory_ids) == 1
        assert result.memory_ids[0] == memory.memory_id
        assert len(rdbms.memories) == 1
        assert len(rdbms.entity_links) == 1

    @pytest.mark.asyncio
    async def test_persist_candidates(self, scope, coordinator):
        """Test persisting candidates to RDBMS."""
        coord, rdbms, _, _, _ = coordinator

        candidate = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.PREFERENCES,
            content="Low confidence memory",
            confidence=0.50,
            destination=MemoryDestination.CANDIDATES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1"],
        )

        result = await coord.persist_rdbms(
            memories=[],
            candidates=[candidate],
            entities=[],
            scope=scope,
            document_id="doc-123",
        )

        assert len(result.candidate_ids) == 1
        assert result.candidate_ids[0] == candidate.memory_id
        assert len(rdbms.candidates) == 1
        assert len(rdbms.entity_links) == 1  # Candidates also get entity links

    @pytest.mark.asyncio
    async def test_persist_multiple_entities(self, scope, coordinator):
        """Test persisting memory with multiple entity references."""
        coord, rdbms, _, _, _ = coordinator

        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Memory with multiple entities",
            confidence=0.90,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1", "entity-2", "entity-3"],
        )

        result = await coord.persist_rdbms(
            memories=[memory],
            candidates=[],
            entities=[],
            scope=scope,
            document_id="doc-123",
        )

        assert len(rdbms.entity_links) == 3  # Three entity references


class TestKnowledgeIdsAreReported:
    """Customer- and client-scope facts live in their own tables. Their ids used to be dropped,
    so the status endpoint said memories_created=0 and memory_ids=[] for an ingest that stored
    thousands (2026-10-06). They are returned separately so metering still counts memories."""

    @pytest.mark.asyncio
    async def test_customer_and_client_knowledge_ids_are_returned(self, scope, coordinator):
        coord, rdbms, _, _, _ = coordinator

        def _mem(destination):
            return ExtractedMemory(
                memory_id=uuid4(), memory_type=MemoryType.FACTS, content="x", confidence=0.9,
                destination=destination, source_document_id="doc-1", entities_mentioned=[],
            )

        customer = _mem(MemoryDestination.CUSTOMER_KNOWLEDGE)
        client = _mem(MemoryDestination.CLIENT_KNOWLEDGE)

        result = await coord.persist_rdbms(
            memories=[], candidates=[], entities=[], scope=scope, document_id="doc-1",
            client_knowledge=[client], customer_knowledge=[customer],
        )

        assert result.memory_ids == []
        assert set(result.knowledge_ids) == {customer.memory_id, client.memory_id}


class TestGraphPersistence:
    """Tests for Graph persistence."""

    @pytest.mark.asyncio
    async def test_persist_entities(self, scope, coordinator):
        """Test persisting entities to graph."""
        coord, _, graph, _, _ = coordinator

        entity = ExtractedEntity(
            entity_id="entity-1",
            entity_type="skill",
            name="Python",
            canonical_name="Python Programming",
        )

        result = await coord.persist_graph(
            entities=[entity],
            relationships=[],
            memories=[],
            scope=scope,
        )

        assert len(result.entity_ids) == 1
        assert result.entity_ids[0] == "entity-1"
        assert len(graph.entities) == 1
        assert len(graph.namespaces) == 1

    @pytest.mark.asyncio
    async def test_persist_relationships(self, scope, coordinator):
        """Test persisting relationships to graph."""
        coord, _, graph, _, _ = coordinator

        # Both endpoints must be among the persisted entities — the coordinator
        # remaps relationship endpoints to graph node IDs and skips any edge
        # with an unmapped endpoint (prevents phantom edges).
        source_entity = ExtractedEntity(
            entity_id="entity-1",
            entity_type="skill",
            name="Python",
        )
        target_entity = ExtractedEntity(
            entity_id="entity-2",
            entity_type="skill",
            name="Django",
        )

        relationship = ExtractedRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="related_to",
            properties={"strength": 0.8},
        )

        result = await coord.persist_graph(
            entities=[source_entity, target_entity],
            relationships=[relationship],
            memories=[],
            scope=scope,
        )

        assert len(result.relationship_ids) == 1
        assert len(graph.relationships) == 1

    @pytest.mark.asyncio
    async def test_persist_memory_content_attached_to_graph(self, scope, coordinator):
        """Test that memory content is attached to graph relationship edges.

        Memory-entity linkage moved out of persist_graph: per-entity references
        are now written to RDBMS (see persist_rdbms), and memories are attached
        to graph edges via attach_memories_to_relationships. This verifies the
        graph-side attachment fires when memories + relationships are present.
        """
        coord, _, graph, _, _ = coordinator

        source_entity = ExtractedEntity(
            entity_id="entity-1", entity_type="skill", name="Python",
        )
        target_entity = ExtractedEntity(
            entity_id="entity-2", entity_type="skill", name="Django",
        )
        relationship = ExtractedRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="related_to",
        )
        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Python relates to Django",
            confidence=0.9,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1", "entity-2"],
        )

        await coord.persist_graph(
            entities=[source_entity, target_entity],
            relationships=[relationship],
            memories=[memory],
            scope=scope,
        )

        # attach_memories_to_relationships was invoked with the memories/edges.
        assert len(graph.attached_memories) == 1
        attached_memories, _, edge_ids, _ = graph.attached_memories[0]
        assert len(attached_memories) == 1
        assert len(edge_ids) == 1


class TestVectorPersistence:
    """Tests for Vector persistence."""

    @pytest.mark.asyncio
    async def test_persist_single_memory(self, scope, coordinator):
        """Test persisting a single memory to vector store."""
        coord, _, _, vector, embedding = coordinator

        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Test memory",
            confidence=0.85,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=[],
        )

        result = await coord.persist_vector(
            memories=[memory],
            entities=[],
            scope=scope,
        )

        assert len(result.vector_ids) == 1
        assert len(embedding.batch_calls) == 1
        assert len(vector.batches) == 1
        assert len(vector.namespaces) == 1

    @pytest.mark.asyncio
    async def test_persist_with_ere_context(self, scope, coordinator):
        """Test persisting memory with ERE context."""
        coord, _, _, vector, embedding = coordinator

        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="User likes Python",
            confidence=0.90,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1"],
        )

        entity = ExtractedEntity(
            entity_id="entity-1",
            entity_type="skill",
            name="Python",
            canonical_name="Python Programming",
        )

        result = await coord.persist_vector(
            memories=[memory],
            entities=[entity],
            scope=scope,
        )

        # Check that ERE context was built
        assert len(vector.batches) == 1
        batch_memories, _, _, ere_contexts = vector.batches[0]
        assert ere_contexts[0] is not None
        assert "skill:Python Programming" in ere_contexts[0]

    @pytest.mark.asyncio
    async def test_persist_empty_memories(self, scope, coordinator):
        """Test handling empty memories list."""
        coord, _, _, vector, embedding = coordinator

        result = await coord.persist_vector(
            memories=[],
            entities=[],
            scope=scope,
        )

        assert len(result.vector_ids) == 0
        assert len(embedding.batch_calls) == 0
        assert len(vector.batches) == 0


class TestCoordinatorIntegration:
    """Integration tests for coordinator."""

    @pytest.mark.asyncio
    async def test_full_persistence_flow(self, scope, coordinator):
        """Test complete persistence flow across all stores."""
        coord, rdbms, graph, vector, embedding = coordinator

        memory = ExtractedMemory(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="Test memory",
            confidence=0.85,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc-123",
            entities_mentioned=["entity-1"],
        )

        entity = ExtractedEntity(
            entity_id="entity-1",
            entity_type="skill",
            name="Python",
        )

        # Second entity so the relationship endpoints both map to graph nodes
        # (the coordinator skips edges with unmapped endpoints).
        entity2 = ExtractedEntity(
            entity_id="entity-2",
            entity_type="skill",
            name="Django",
        )

        relationship = ExtractedRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="related_to",
        )

        # RDBMS
        rdbms_result = await coord.persist_rdbms(
            memories=[memory],
            candidates=[],
            entities=[entity],
            scope=scope,
            document_id="doc-123",
        )

        # Graph
        graph_result = await coord.persist_graph(
            entities=[entity, entity2],
            relationships=[relationship],
            memories=[memory],
            scope=scope,
        )

        # Vector
        vector_result = await coord.persist_vector(
            memories=[memory],
            entities=[entity],
            scope=scope,
        )

        # Verify all results
        assert len(rdbms_result.memory_ids) == 1
        assert len(graph_result.entity_ids) == 2
        assert len(graph_result.relationship_ids) == 1
        assert len(vector_result.vector_ids) == 1
