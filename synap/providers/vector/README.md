# Vector Providers

## What belongs here
- Vector database client integrations
- Provider-specific query translation
- Connection management
- Embedding storage and retrieval

## What must NEVER belong here
- Core retrieval algorithms
- Business logic
- HTTP endpoints
- Authentication middleware
- Embedding generation (use LLM providers)

## Dependencies
- May depend on: core (algorithms), contracts
- Must NOT depend on: sdk, cloud, other providers
