# Listening & Retrieval Agent Integration Tests

## Overview

Integration tests for the Listening & Retrieval flow, verifying the complete path from `ListeningRetrievalAgent` through `RetrievalManager` to storage providers.

## Test Structure

```
synap/tests/integration/agents/listening_retrieval/
├── __init__.py
├── conftest.py                    # Shared fixtures and mock providers
├── test_agent_graph.py            # Agent graph flow tests (13 tests)
├── test_retrieval_manager.py      # Manager integration tests (15 tests)
└── test_end_to_end.py             # Full flow tests (12 tests)
```

## Test Files

### conftest.py

Shared test fixtures including:
- **Mock Providers**: MockVectorStore, MockGraphStore, MockCacheProvider, MockPostgresProvider, MockLLMAdapter
- **Sample Data**: Context items, vector data, graph nodes/edges
- **Fixtures**: sample_scope, sample_decision, accurate_decision, sample_auth_context

### test_agent_graph.py (13 tests)

Tests for LangGraph nodes and routing:

**TestListenNode** (3 tests):
- SDK request parsing
- gRPC stream event handling
- Signal emission for ingestion

**TestDecideNode** (4 tests):
- ContextDecision creation
- FAST mode selection for simple queries
- ACCURATE mode selection for complex queries
- Token budget enforcement

**TestSubqueryNode** (2 tests):
- LLM-based subquery generation
- Fallback to standard B2B queries (D5)

**TestRetrieveNode** (2 tests):
- RetrievalManager integration
- Error handling and graceful degradation

**TestEarlyTermination** (2 tests):
- High confidence threshold termination (D22)
- Low confidence continuation

### test_retrieval_manager.py (15 tests)

Tests for storage provider coordination:

**TestShortTermRetrieval** (3 tests):
- L2 cache hit path
- Cache miss → Postgres query
- Time window filtering

**TestLongTermRetrieval** (4 tests):
- Vector-first retrieval (D7)
- Graph augmentation (D7)
- Hybrid edge scoring (0.7*edge + 0.3*target)
- Memory edge boost (1.2x)

**TestFilteringPipeline** (4 tests):
- Utility check (D21 Check 1)
- Echo check deduplication (D21 Check 2)
- State vs Action classification (D21 Check 3)
- Hallucination flagging (D21 Check 4)

**TestRetrievalManagerIntegration** (4 tests):
- FAST mode complete flow
- ACCURATE mode complete flow
- Graceful degradation on store failure
- Hard token budget enforcement

### test_end_to_end.py (12 tests)

End-to-end flow tests:

**TestSDKRequestFlow** (2 tests):
- Simple fetch_context request
- Context type filtering

**TestStreamEventFlow** (2 tests):
- Stream event triggering retrieval
- Ingestion signal emission

**TestMultiTenantFlow** (2 tests):
- User-scoped retrieval
- Customer-level context inclusion

**TestCachingBehavior** (2 tests):
- L1 query cache hit
- Cache invalidation on update

**TestDegradationScenarios** (2 tests):
- Vector store failure fallback
- Complete failure handling

**TestPerformanceConstraints** (2 tests):
- Latency target validation
- Parallel store queries

## Running Tests

### All Tests
```bash
pytest synap/tests/integration/agents/listening_retrieval/ -v
```

### Specific Test File
```bash
pytest synap/tests/integration/agents/listening_retrieval/test_agent_graph.py -v
```

### Specific Test Class
```bash
pytest synap/tests/integration/agents/listening_retrieval/test_retrieval_manager.py::TestFilteringPipeline -v
```

### With Coverage (Note: May have torch import conflicts)
```bash
pytest synap/tests/integration/agents/listening_retrieval/ --cov=synap.cloud.agents.listening_retrieval --cov=synap.cloud.application.client_api.retrieval_manager -v
```

## Test Results

**Total Tests**: 40  
**Status**: ✅ All Passing  
**Execution Time**: ~0.2s

## Key Decisions Tested

| Decision | Description | Test Coverage |
|----------|-------------|---------------|
| D3 | Minimal LangGraph nodes | Graph routing tests |
| D4 | FAST vs ACCURATE mode | Mode selection tests |
| D5 | Fallback hierarchy: LLM → MACA → Standard B2B | Subquery fallback tests |
| D7 | Vector-first with graph augmentation | Long-term retrieval tests |
| D21 | 4-check filtering algorithm | Filtering pipeline tests |
| D22 | Early termination on confidence | Early termination tests |

## Mock Providers

### MockVectorStore
- Simulates vector search with configurable results
- Tracks queries received for verification
- Supports both `search()` and `query()` methods

### MockGraphStore
- Simulates graph traversal with adjacency data
- Tracks traversal operations
- Provides nodes and edges for testing

### MockCacheProvider
- In-memory cache simulation
- Tracks gets and sets for verification
- Supports TTL and key-based operations

### MockPostgresProvider
- Short-term storage simulation
- Supports time-based filtering
- Tracks queries for verification

### MockLLMAdapter
- Configurable LLM responses
- Tracks completion calls
- Default subquery generation response

## Notes

- Tests use `pytest-asyncio` for async test support
- State mutation tests verify LangGraph node behavior
- Mock providers follow existing test patterns from ingestion tests
- Degradation tests verify graceful failure handling
