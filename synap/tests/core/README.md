# Core Tests

## What belongs here
- Unit tests for core algorithms
- Algorithm correctness tests
- Edge case validation
- Performance benchmarks for pure functions

## What must NEVER belong here
- Integration tests
- Provider-specific tests
- API endpoint tests
- Authentication tests
- Network-dependent tests

## Dependencies
- May depend on: core
- Must NOT depend on: sdk, cloud, providers (except for test utilities)
