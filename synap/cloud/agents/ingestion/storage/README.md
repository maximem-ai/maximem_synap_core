# Ingestion Storage Layer

Implementation of Stage 4 (Organization & Persistence) for the ingestion pipeline.

## Overview

The storage layer coordinates persistence of extracted memories, entities, and relationships across three storage systems:
- **RDBMS** (PostgreSQL): Structured memory data with ACID guarantees
- **Graph** (Neo4j): Entity relationships and graph traversal
- **Vector** (ChromaDB): Embeddings for semantic search

## Architecture

```
Stage 4 Organizer
    ↓
Storage Coordinator
    ↓
┌─────────────┬──────────────┬──────────────┐
│             │              │              │
RDBMS       Graph          Vector
(sync)      (async)        (async)
```

### Persistence Order

1. **RDBMS first (synchronous)** - generates authoritative UUIDs
2. **Graph and Vector in parallel (asynchronous)** - use RDBMS-generated IDs

## Components

### 1. Storage Coordinator (`coordinator.py`)

Orchestrates persistence across all storage systems.

**Key Methods:**
- `persist_rdbms()` - Persist memories, candidates, and entity references
- `persist_graph()` - Persist entities, relationships, and memory-entity links
- `persist_vector()` - Generate embeddings and persist to vector store

**Result Types:**
- `RDBMSPersistResult` - Memory/candidate IDs
- `GraphPersistResult` - Entity/relationship IDs
- `VectorPersistResult` - Vector IDs

### 2. Async Wrappers

Wrap synchronous provider interfaces with `asyncio.to_thread()` for non-blocking I/O.

#### AsyncRDBMSWrapper (`rdbms_wrapper.py`)

Maps ingestion domain models to SQL queries.

**Methods:**
- `insert_memory()` - Insert to memories table
- `insert_candidate()` - Insert to candidates table
- `insert_entity_reference()` - Create memory-entity link
- `insert_client_knowledge()` - Insert client-level knowledge
- `insert_customer_knowledge()` - Insert customer-level knowledge

#### AsyncGraphWrapper (`graph_wrapper.py`)

Adapts GraphStoreInterface for ingestion use cases.

**Methods:**
- `upsert_entity_node()` - Create/update entity node
- `create_relationship()` - Create relationship edge
- `create_memory_entity_link()` - Link memory to entity
- `ensure_namespace()` - Initialize namespace for tenant

**Namespace:** Uses `instance_id` for tenant isolation

#### AsyncVectorWrapper (`vector_wrapper.py`)

Converts domain models to VectorDocument format.

**Methods:**
- `upsert_memory()` - Upsert single memory with embedding
- `upsert_memories_batch()` - Batch upsert for efficiency
- `ensure_namespace()` - Initialize collection
- `search()` - Semantic search by embedding

**Namespace:** Uses `synap_{client_id}` for collections

### 3. Short-term Storage (`short_term.py`)

Handles session-scoped context with TTL expiration.

**Storage:**
- RDBMS: `short_term_context` table
- Vector: `synap_{client_id}_shortterm` collection
- No Graph (too transient)

**Methods:**
- `persist()` - Store session context with TTL
- `get_by_session()` - Retrieve all context for session
- `cleanup_expired()` - Delete expired records
- `extend_ttl()` - Extend expiration time
- `delete_session()` - Immediate cleanup

**Default TTL:** 24 hours from creation

### 4. Stage 4 Organizer (`../pipeline/stage4_organization.py`)

Orchestrates the full persistence flow.

**Method:** `organize_and_persist()`

**Flow:**
1. Separate memories by destination (MEMORIES, CANDIDATES, CLIENT_KNOWLEDGE, CUSTOMER_KNOWLEDGE)
2. Persist to RDBMS (sync)
3. Persist to Graph and Vector in parallel (async)
4. Handle exceptions gracefully with partial results

## Memory Destinations

| Destination | RDBMS Table | Graph | Vector | Scope |
|-------------|-------------|-------|--------|-------|
| MEMORIES | `memories` | ✓ | ✓ | User/Customer |
| CANDIDATES | `candidates` | ✗ | ✗ | User/Customer |
| CLIENT_KNOWLEDGE | `client_knowledge` | ✓ | ✓ | Client-wide |
| CUSTOMER_KNOWLEDGE | `customer_knowledge` | ✓ | ✓ | Customer-wide |
| SHORT_TERM | `short_term_context` | ✗ | ✓ | Session (TTL) |

## Database Schema

See `synap/docs/architecture/05_Ingestion_Storage_Schema.md` for full schema documentation.

**Key Tables:**
- `memories` - Qualified memories (above threshold)
- `candidates` - Below threshold (pending review)
- `short_term_context` - Session context with TTL
- `client_knowledge` - Client-level knowledge
- `customer_knowledge` - Customer-level knowledge
- `memory_entity_links` - Audit trail for entity references

## Usage Example

```python
from synap.cloud.agents.ingestion.storage import (
    StorageCoordinator,
    AsyncRDBMSWrapper,
    AsyncGraphWrapper,
    AsyncVectorWrapper,
)
from synap.cloud.agents.ingestion.pipeline import Stage4Organizer
from synap.cloud.agents.ingestion.interfaces import EmbeddingClient

# Initialize providers (existing sync interfaces)
rdbms_provider = PostgresAdapter(...)
graph_provider = Neo4jAdapter(...)
vector_provider = ChromaDBAdapter(...)
embedding_client = OpenAIEmbedding(...)

# Wrap with async adapters
rdbms = AsyncRDBMSWrapper(rdbms_provider)
graph = AsyncGraphWrapper(graph_provider)
vector = AsyncVectorWrapper(vector_provider)

# Create coordinator
coordinator = StorageCoordinator(rdbms, graph, vector, embedding_client)

# Create Stage 4 organizer
organizer = Stage4Organizer(maca_config, coordinator)

# Persist Stage 3 output
stage4_output = await organizer.organize_and_persist(
    stage3_output=stage3_output,
    scope=scope,
    document_id="doc-123",
    s3_path="s3://bucket/key",
)

print(f"Persisted {len(stage4_output.rdbms_memory_ids)} memories")
print(f"Created {len(stage4_output.graph_entity_ids)} entity nodes")
print(f"Stored {len(stage4_output.vector_ids)} embeddings")
```

## Testing

Tests are located in `synap/tests/agents/ingestion/`:

- `pipeline/test_stage4.py` - Stage 4 Organizer tests
- `storage/test_coordinator.py` - Storage Coordinator tests
- `storage/test_short_term.py` - Short-term Storage tests

All tests use mock providers to avoid external dependencies.

Run tests:
```bash
pytest synap/tests/agents/ingestion/storage/
pytest synap/tests/agents/ingestion/pipeline/test_stage4.py
```

## Error Handling

The implementation follows a graceful degradation model:

1. **RDBMS failures**: Fail entire operation (RDBMS is authoritative)
2. **Graph failures**: Log error, return partial results with empty graph IDs
3. **Vector failures**: Log error, return partial results with empty vector IDs

This ensures that RDBMS persistence always succeeds or fails atomically, while Graph/Vector failures don't block the pipeline.

## Future Enhancements

1. **Batch optimization**: Chunk large entity/memory batches for better performance
2. **Retry logic**: Add exponential backoff for transient failures
3. **Metrics**: Track persistence latency and success rates
4. **Rollback**: Implement compensating transactions for partial failures
5. **Vector cleanup**: Automated cleanup of expired short-term vectors
