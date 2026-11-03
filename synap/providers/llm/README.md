# LLM Providers

## What belongs here
- LLM API client integrations
- Embedding generation
- Completion and chat interfaces
- Model-specific parameter handling

## What must NEVER belong here
- Memory extraction logic
- Prompt templates (unless provider-specific)
- Business workflows
- Vector or graph storage
- Authentication middleware

## Dependencies
- May depend on: contracts
- Must NOT depend on: sdk, cloud, core, other providers
