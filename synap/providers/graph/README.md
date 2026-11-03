# Graph Providers

## What belongs here
- Graph database client integrations
- Provider-specific query translation
- Connection management
- Graph persistence and traversal

## What must NEVER belong here
- Core graph algorithms
- Business workflows
- HTTP endpoints
- Authentication
- Vector operations

## Dependencies
- May depend on: core (algorithms), contracts
- Must NOT depend on: sdk, cloud, other providers
