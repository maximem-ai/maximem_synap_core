# Contracts

## Purpose

The `contracts/` folder defines interface boundaries, schemas, and protocols for communication between Synap layers. Contracts are language-agnostic, implementation-independent specifications that freeze vocabulary and structure. This layer exists to prevent coupling between implementation layers and to provide stable integration points.

## Allowed Responsibilities

- Interface definitions for cross-layer communication
- Request and response schemas
- Data structure specifications
- Protocol definitions
- Type definitions for shared concepts
- Enum and constant definitions
- Semantic contracts (what data means, not how it is processed)
- Validation rules for data interchange
- Version specifications for contracts

## Explicit Non-Goals

- MUST NOT contain executable code or implementations
- MUST NOT include business logic
- MUST NOT contain provider-specific details
- MUST NOT include test data or fixtures
- MUST NOT contain configuration values
- MUST NOT implement validation logic (define rules only)
- MUST NOT include algorithmic implementations
- MUST NOT contain client or server code
- MUST NOT include database schemas (provider concern)
- MUST NOT reference specific technologies or vendors

## Dependency Rules

**What `contracts/` may depend on:**
- Nothing
- `contracts/` is a leaf node with zero dependencies

**What MUST depend on `contracts/`:**
- `sdk/` MUST depend on `contracts/` for interface definitions
- `providers/` MUST depend on `contracts/` for interface definitions
- `cloud/` MUST depend on `contracts/` for interface definitions

**What MUST NOT depend on `contracts/`:**
- `core/` MUST NOT depend on `contracts/`

## Stability Guarantee

**Contract-level.**

Contracts are stable by definition. Breaking changes to contracts require major versioning and coordinated updates across multiple layers. Additions are permitted; modifications require deprecation cycles. Contracts evolve slowly and deliberately.
