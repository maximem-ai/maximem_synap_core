"""Re-drives ingestion dead letters. Until now there was no way back in.

    python -m synap.cloud.agents.ingestion.replay                    # dry-run, everything
    python -m synap.cloud.agents.ingestion.replay --commit           # actually replay
    python -m synap.cloud.agents.ingestion.replay --commit --stage stage_4_organization
    python -m synap.cloud.agents.ingestion.replay --commit --id 00cd4fe2-...
    python -m synap.cloud.agents.ingestion.replay --commit --terminal   # include non-transient

Dry-run is the default and ``--commit`` is required to act, following
``synap/cloud/replication/replay.py``: every item in this queue is there because
something already went wrong once.

WHY THIS EXISTS
---------------
``_send_to_dlq`` wrote to ``synap:ingestion:dlq:{stage}:{client_id}`` and NOTHING
ever read it. Not a worker, not a job, not an endpoint. A dead letter was final,
and the only evidence was a metric counting how many there were.

That can cost a real ingestion. A conversation transcript is accepted, clears
qualification, and stage 3 extracts memories from it. Stage 4 then tries to
persist them while the database is restarting and fails three times in seconds,
so the item is dead-lettered and the customer keeps their HTTP 200.

The retry budget is the whole story. MAX_RETRIES is 3 and BASE_DELAY_SECONDS is 1,
so the delays are 1s and 2s: a THREE SECOND budget. A database outage can last
minutes, so the retry is short by orders of magnitude and cannot succeed. (MAX_DELAY_SECONDS
is set to 60
and is unreachable: with three attempts the largest delay computable is 2 seconds.)

Simply raising MAX_RETRIES does not fix it and makes something else worse. The
worker loop is serial — ``_process_with_retry`` is awaited before the next
``_dequeue`` — so every second of in-line backoff is a second that worker cannot
process anything else. A budget long enough to ride out a database restart would
take a stage offline for the duration, turning one lost item into a stalled
pipeline. The backoff is for blips. Outages need a way back in afterwards, which
is this.

WHAT IS SAFE TO REPLAY
----------------------
Only errors that were about the world being temporarily unavailable, never about
the payload itself. A poison message replayed is a poison message re-failed, and
re-queuing it forever would hide it behind its own noise. Anything unrecognised is
treated as terminal and left alone, so the failure mode is "a human looks at it"
rather than "a loop runs all night". ``--terminal`` overrides that deliberately.

Replayed items carry ``replay_count`` and ``replayed_at``. An item that has already
been replayed ``MAX_REPLAYS`` times is left in place whatever its error says: if it
has failed that many times across separate outages, the error string is no longer
telling the truth about why.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Which stage queue an item goes back onto. Taken from each worker's `input_queue`
# rather than rebuilt, because a mismatch here re-runs the wrong stage: sending a
# stage-4 item to stage 1 would re-extract and bill the customer a second time.
STAGE_INPUT_QUEUE: Dict[str, str] = {
    "stage_1_qualification": "synap:ingestion:stage1",
    "stage_2_enrichment": "synap:ingestion:stage23",
    "stage_3_extraction": "synap:ingestion:stage23",
    "stage_4_organization": "synap:ingestion:stage4",
}

# Transient: the world was briefly unavailable. The payload was never the problem.
TRANSIENT_PATTERNS = [
    r"the database system is (shutting down|starting up|in recovery)",
    r"connection (refused|reset|closed|timed out)",
    r"could not connect",
    r"server closed the connection",
    r"ConnectionDoesNotExistError",
    r"InterfaceError",
    r"CannotConnectNowError",
    r"TooManyConnectionsError",
    r"pool timeout|QueuePool limit",
    r"Temporary failure in name resolution",
    r"(read|write) timeout",
    r"\b(502|503|504)\b",
    r"ServiceUnavailable",
    r"TimeoutError",
]
_TRANSIENT = re.compile("|".join(TRANSIENT_PATTERNS), re.I)

MAX_REPLAYS = 3
SCAN_LIMIT = 5000  # bounded, like the queue sampler: a partial pass beats a blocked Redis


def classify(error: str) -> str:
    """``transient`` if this looks worth replaying, else ``terminal``.

    Unrecognised errors are terminal ON PURPOSE. The cost of wrongly calling
    something terminal is that a person reads it; the cost of wrongly calling
    something transient is an item that re-fails forever and buries the queue.
    """
    if not error:
        return "terminal"
    return "transient" if _TRANSIENT.search(error) else "terminal"


async def _dlq_keys(redis: Any, stage: Optional[str], client: Optional[str]) -> List[str]:
    pattern = "synap:ingestion:dlq:%s:%s" % (stage or "*", client or "*")
    keys: List[str] = []
    cursor, scanned = 0, 0
    while scanned < SCAN_LIMIT:
        cursor, batch = await redis.scan(cursor=cursor, match=pattern, count=500)
        keys.extend(k.decode() if isinstance(k, bytes) else k for k in batch)
        scanned += len(batch)
        if cursor == 0:
            break
    return keys


async def collect(redis: Any, stage: Optional[str] = None, client: Optional[str] = None,
                  ingestion_id: Optional[str] = None) -> List[Tuple[str, int, Dict[str, Any]]]:
    """Every dead letter matching the filters, as (key, index, payload)."""
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for key in await _dlq_keys(redis, stage, client):
        depth = int(await redis.llen(key) or 0)
        for idx in range(depth):
            raw = await redis.lindex(key, idx)
            if not raw:
                continue
            try:
                item = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                # Unparseable is reported, never silently skipped: a DLQ entry nothing
                # can read is itself a finding.
                item = {"_unparseable": True, "error": "payload is not JSON"}
            if ingestion_id and str(item.get("ingestion_id")) != ingestion_id:
                continue
            out.append((key, idx, item))
    return out


def describe(key: str, item: Dict[str, Any]) -> str:
    stage = key.split(":")[3] if len(key.split(":")) > 3 else "?"
    err = str(item.get("error", ""))[:90]
    return (
        "  %-22s %-38s replays=%s  %s\n        failed_at=%s  error=%s"
        % (
            stage,
            item.get("ingestion_id", "?"),
            item.get("replay_count", 0),
            classify(str(item.get("error", ""))).upper(),
            item.get("failed_at", "?"),
            err,
        )
    )


async def replay(redis: Any, *, commit: bool = False, stage: Optional[str] = None,
                 client: Optional[str] = None, ingestion_id: Optional[str] = None,
                 include_terminal: bool = False) -> Dict[str, int]:
    """Re-enqueue replayable dead letters. Reports without acting unless ``commit``."""
    found = await collect(redis, stage, client, ingestion_id)
    counts = {"found": len(found), "replayed": 0, "skipped_terminal": 0,
              "skipped_exhausted": 0, "skipped_unknown_stage": 0}

    for key, _idx, item in found:
        print(describe(key, item))
        kind = classify(str(item.get("error", "")))
        replays = int(item.get("replay_count", 0) or 0)
        item_stage = key.split(":")[3] if len(key.split(":")) > 3 else ""
        queue = STAGE_INPUT_QUEUE.get(item_stage)

        if queue is None:
            counts["skipped_unknown_stage"] += 1
            print("        -> SKIP: no input queue known for stage %r" % item_stage)
            continue
        if replays >= MAX_REPLAYS:
            counts["skipped_exhausted"] += 1
            print("        -> SKIP: already replayed %d times" % replays)
            continue
        if kind == "terminal" and not include_terminal:
            counts["skipped_terminal"] += 1
            print("        -> SKIP: terminal error, use --terminal to force")
            continue

        if not commit:
            print("        -> would re-enqueue to %s" % queue)
            continue

        payload = {k: v for k, v in item.items()
                   if k not in ("error", "failed_at", "worker_id")}
        payload["replay_count"] = replays + 1
        payload["replayed_at"] = datetime.now(timezone.utc).isoformat()

        # Enqueue FIRST, remove second. The queues are LPUSH/RPOP, so a crash between
        # the two leaves a duplicate rather than a loss, and the pipeline is
        # idempotent on ingestion_id. Removing first would risk losing the only copy.
        await redis.lpush(queue, json.dumps(payload))
        # LREM by exact value rather than by index: indices shift as the list is
        # modified, and another replay running concurrently would delete the wrong row.
        await redis.lrem(key, 1, json.dumps(item) if not isinstance(item, str) else item)
        counts["replayed"] += 1
        print("        -> RE-ENQUEUED to %s (replay %d)" % (queue, replays + 1))
        logger.warning(
            "ingestion_dlq_replayed ingestion_id=%s stage=%s queue=%s replay_count=%s",
            item.get("ingestion_id"), item_stage, queue, replays + 1,
        )
    return counts


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", action="store_true", help="actually replay; default is a dry run")
    ap.add_argument("--stage", help="only this stage, e.g. stage_4_organization")
    ap.add_argument("--client", help="only this client_id")
    ap.add_argument("--id", dest="ingestion_id", help="only this ingestion_id")
    ap.add_argument("--terminal", action="store_true",
                    help="include errors classified terminal (they will probably fail again)")
    args = ap.parse_args()

    from synap.cloud.application.shared.providers import get_redis_provider
    redis = await get_redis_provider()
    if redis is None:
        print("no redis provider available")
        return 1

    counts = await replay(redis, commit=args.commit, stage=args.stage, client=args.client,
                          ingestion_id=args.ingestion_id, include_terminal=args.terminal)
    print("\n%s" % ("committed" if args.commit else "DRY RUN, nothing changed"))
    for k, v in counts.items():
        print("  %-22s %d" % (k, v))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(asyncio.run(_main()))
