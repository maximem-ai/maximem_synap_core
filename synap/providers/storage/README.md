# Storage Providers

## What belongs here
- Object storage integrations
- File persistence
- Blob storage operations
- Provider-specific upload/download logic

## What must NEVER belong here
- Vector or graph storage
- Business logic
- HTTP endpoints
- Authentication middleware
- Data processing algorithms

## Dependencies
- May depend on: contracts
- Must NOT depend on: sdk, cloud, core, other providers
