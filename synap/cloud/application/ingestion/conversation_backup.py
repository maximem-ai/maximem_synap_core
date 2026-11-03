"""One backup object per conversation, kept current as the conversation compacts.

The per-ingest copies (`{ingestion_id}.bin`) are one chunk per compaction, with
no conversation id and no message ids. This is the copy you can restore from:

    conversations/{client}/{instance}/{customer}/{conversation_id}.json

Every message is passed through the same PII gate an ingest uses before it is
stored. A message whose treatment fails is stored as received and flagged, with
where it failed, then retried. Each run adds only what the object does not hold
(or holds untreated), so a message is treated once, a lost race repairs itself on
the next compaction, and nothing is duplicated.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, List

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def conversation_key(scope: Any, conversation_id: str) -> str:
    return (f"conversations/{scope.client_id}/{scope.instance_id}/"
            f"{scope.customer_id}/{conversation_id}.json")


def _iso(ts: Any) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts or "")


async def backup_conversation(manager: Any, conversation_id: str, scope: Any,
                              messages: List[Any]) -> None:
    """Bring the stored conversation up to date with `messages`. Never raises.

    A message whose PII treatment failed is stored anyway, flagged
    `pii.status = "failed"` with where it failed, and treated again on the next
    compaction. The object-level metadata carries the count so a bucket scan can
    find conversations that hold untreated text without opening them.
    """
    try:
        storage = manager._storage
        key = conversation_key(scope, conversation_id)

        existing = await storage.read_object(key)
        doc = json.loads(existing) if existing else {
            "version": FORMAT_VERSION,
            "conversation_id": conversation_id,
            "client_id": str(scope.client_id),
            "instance_id": str(scope.instance_id),
            "customer_id": str(scope.customer_id),
            "messages": [],
        }
        by_id = {m["message_id"]: m for m in doc["messages"]}

        changed = False
        for msg in messages:
            held = by_id.get(msg.message_id)
            if (held and held["pii"]["status"] in ("treated", "not_applied")) or not (msg.content or "").strip():
                continue  # already held and treated, or nothing to store
            t = await manager.treat_for_storage(
                msg.content, scope,
                request_id=f"convbackup:{conversation_id}:{msg.message_id}",
            )
            if held and t.failed_at:
                continue  # failed again: keep what is there, flag unchanged
            by_id[msg.message_id] = {
                "message_id": msg.message_id,
                "role": msg.role,
                "timestamp": _iso(msg.timestamp),
                "content": t.text,
                "pii": _pii(t),
            }
            changed = True

        if not changed:
            return

        doc["messages"] = sorted(by_id.values(), key=lambda m: m["timestamp"])
        failed = [m["message_id"] for m in doc["messages"] if m["pii"]["status"] == "failed"]
        doc["pii_failed_message_ids"] = failed
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        await storage.write_object(
            key, json.dumps(doc, ensure_ascii=False).encode("utf-8"),
            metadata={"conversation_id": conversation_id,
                      "pii_failed_messages": str(len(failed))},
        )
        logger.info("conversation_backup_saved conversation=%s total=%d pii_failed=%d",
                    conversation_id, len(doc["messages"]), len(failed))
        if failed:
            logger.error("conversation_backup_holds_untreated_text conversation=%s "
                         "messages=%d", conversation_id, len(failed))
    except Exception:
        logger.error("conversation_backup_failed conversation=%s", conversation_id,
                     exc_info=True)


def _pii(t: Any) -> dict:
    if not t.applied:
        return {"status": "not_applied"}
    if t.failed_at:
        return {"status": "failed", "failed_at": t.failed_at, "reason": t.reason}
    return {"status": "treated", "detected": t.detected, "floor_blocked": t.floor_blocked,
            "aliased": t.aliased, "policy_version": t.policy_version}
