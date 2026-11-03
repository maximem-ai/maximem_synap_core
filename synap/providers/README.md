# Providers

## Purpose

The `providers/` folder contains adapters and integrations for third-party services. This layer is the sole boundary between Synap and external systems including vector databases, graph databases, LLM APIs, and storage services. Providers apply `core/` algorithms to provider-specific data and translate between Synap semantics and external service APIs.

## Allowed Responsibilities

- Vector database client integrations (Pinecone, Weaviate, etc.)
- Graph database client integrations (Neo4j, TigerGraph, etc.)
- LLM provider integrations (OpenAI, Anthropic, etc.)
- Object storage integrations (S3, GCS, etc.)
- Connection management, pooling, and retry logic
- Provider-specific query translation
- Embedding generation via LLM providers
- Application of `core/` algorithms to provider data
- Provider-specific error handling and mapping
- Schema and data format translation

## Explicit Non-Goals

- MUST NOT implement core algorithms (use `core/`)
- MUST NOT contain business logic or workflows
- MUST NOT include API endpoints or routing
- MUST NOT implement authentication or authorization
- MUST NOT contain SDK client code
- MUST NOT orchestrate agents or pipelines (belongs in `cloud/`)
- MUST NOT serve HTTP requests
- MUST NOT implement multi-tenancy logic
- MUST NOT depend on other providers (cross-provider logic belongs in `cloud/`)
- MUST NOT bypass `core/` to implement algorithms locally

## Dependency Rules

**What `providers/` may depend on:**
- `core/` for algorithmic implementations
- `contracts/` for interface definitions
- Third-party service SDKs and clients

**What MUST depend on `providers/`:**
- `cloud/` MUST depend on `providers/` for all external integrations

**What MUST NOT depend on `providers/`:**
- `core/` MUST NOT depend on `providers/`
- `sdk/` MUST NOT depend on `providers/`
- `contracts/` MUST NOT depend on `providers/`
- Other `providers/` subdirectories MUST NOT depend on each other

## Stability Guarantee

**Fast-evolving.**

Provider integrations change as third-party services evolve, new providers are added, and existing integrations are optimized. Provider implementations are isolated from each other. Only the interfaces exposed to `cloud/` are stable (defined in `contracts/`).
