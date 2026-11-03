# Agent Tests

## What belongs here
- Agent behavior tests
- Workflow orchestration tests
- Agent integration tests
- Message handling validation

## What must NEVER belong here
- Core algorithm tests
- Provider implementation tests
- Gateway routing tests
- SDK client tests
- Pure unit tests (use focused test directories)

## Dependencies
- May depend on: cloud/agents, contracts, providers (mocked)
- Must NOT depend on: sdk
