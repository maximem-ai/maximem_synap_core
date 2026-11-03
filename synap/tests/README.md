# Tests

## Purpose

The `tests/` folder contains all test code for the Synap platform. This layer verifies correctness, performance, and integration behavior across all other layers. Tests are the sole exception to dependency rules and may depend on any layer to validate behavior.

## Allowed Responsibilities

- Unit tests for isolated components
- Integration tests for cross-layer behavior
- End-to-end tests for complete workflows
- Performance benchmarks and profiling
- Test fixtures and mock data
- Test utilities and helpers
- Contract validation tests
- Regression tests
- Load and stress tests
- Test-specific configuration

## Explicit Non-Goals

- MUST NOT contain production code
- MUST NOT be imported by production code
- MUST NOT include business logic
- MUST NOT contain documentation (belongs in `docs/`)
- MUST NOT include production configuration
- MUST NOT contain build artifacts
- MUST NOT include deployment scripts
- MUST NOT be published as part of SDK
- MUST NOT define production interfaces (belongs in `contracts/`)
- MUST NOT implement production algorithms (belongs in `core/`)

## Dependency Rules

**What `tests/` may depend on:**
- `core/` for testing algorithms
- `sdk/` for testing client libraries
- `cloud/` for testing server-side logic
- `providers/` for testing integrations
- `contracts/` for validating contracts
- `context_cloud/` for validating documentation accuracy
- Test-specific dependencies and frameworks

**What MUST depend on `tests/`:**
- Nothing
- `tests/` is a terminal node

**What MUST NOT depend on `tests/`:**
- `core/` MUST NOT depend on `tests/`
- `sdk/` MUST NOT depend on `tests/`
- `cloud/` MUST NOT depend on `tests/`
- `providers/` MUST NOT depend on `tests/`
- `contracts/` MUST NOT depend on `tests/`

## Stability Guarantee

**Fast-evolving.**

Tests change frequently as features are added, bugs are fixed, and edge cases are discovered. Test structure mirrors the layer being tested. Breaking production code breaks tests; breaking tests does not break production code.
