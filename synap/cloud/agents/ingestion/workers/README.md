# Ingestion Worker Runtime Requirements

The SDK ingestion endpoints now enqueue work into Redis via `IngestionManager`.
To complete jobs end-to-end, the following long-running processes must be active.

## Required Processes

1. `PriorityQueueRouter`
   - Reads from priority queues:
     - `synap:ingestion:realtime`
     - `synap:ingestion:explicit`
     - `synap:ingestion:bootstrap`
   - Pushes into:
     - `synap:ingestion:stage1`

2. `Stage1Worker`
   - Reads from `synap:ingestion:stage1`
   - Writes to `synap:ingestion:stage23`

3. `Stage23Worker`
   - Reads from `synap:ingestion:stage23`
   - Writes to `synap:ingestion:stage4`

4. `Stage4Worker`
   - Reads from `synap:ingestion:stage4`
   - Persists to RDBMS, Graph, and Vector stores
   - Marks ingestion status terminal (`COMPLETED` or `FAILED`)

## Operational Checklist

- Ensure Redis is reachable via `REDIS_URL`.
- Ensure storage and provider dependencies are configured:
  - RDBMS
  - Vector store
  - Graph store
  - File store (for original payload persistence)
- Ensure workers are supervised (systemd, container orchestrator, or process manager).
- Ensure graceful shutdown signals call worker `stop()` before process exit.

## Symptoms of Missing Worker Runtime

- `POST /api/v1/memories/create` returns `queued`, but status never progresses.
- Priority queues (`realtime`/`explicit`/`bootstrap`) grow continuously.
- Stage queues remain empty because router is not running.

## Log-First Troubleshooting Flow

To trace one job end-to-end, filter logs by `ingestion_id` and/or `request_id`.

Expected event sequence:

1. API acceptance
   - `sdk_ingestion_request_received`
   - `sdk_ingestion_request_mapped`
   - `sdk_ingestion_queued`
2. Manager + queue/status
   - `ingestion_validate_passed`
   - `ingestion_rate_limit_passed`
   - `ingestion_idempotency_miss` (or `ingestion_idempotency_hit`)
   - `ingestion_queue_enqueued`
   - `ingestion_status_create`
3. Router + workers
   - `ingestion_router_routed`
   - `ingestion_worker_dequeued`
   - `ingestion_stage1_*`
   - `ingestion_stage2_completed`
   - `ingestion_stage3_completed`
   - `ingestion_stage4_persisted`
   - `ingestion_stage4_completed`
4. Failures and retries
   - `ingestion_worker_attempt_failed`
   - `ingestion_worker_retry_scheduled`
   - `ingestion_worker_sent_dlq`
   - `ingestion_worker_terminal_failed`

