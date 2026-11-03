"""Post-ingestion user-profile merge (spec §5.3).

Invoked from the tail of ``Stage4Worker.process`` after a successful
persist. One light LLM call merges the just-created memories (plus the
caller-supplied ``metadata["analysis"]`` hints) into the owner's profile
document; the result is written through :class:`ProfileStore` under a
single-row lock.

Concurrency (review finding 9): the LLM call runs OUTSIDE any transaction.
The merge callback executed under the row lock receives the FRESH document;
if another writer bumped the version while we were at the LLM, the LLM
output is applied ATTRIBUTE-WISE onto the fresh document (per-attribute
``updated_at`` newest-wins, fresh overview kept) — never a whole-document
overwrite of state the LLM never saw.

Billing: the LLM call is stamped with the plain ``request_id`` and
``ingestion.profile_merge`` is in ``BILLABLE_OPERATION_PREFIXES``, so the
merge is charged as part of the per-document ingestion charge.

It used to carry a derived ``f"{request_id}:profile"`` id instead (review
finding 10). That was a workaround for a charge query with no operation
filter: any row sharing the request id was billed, so the only way to exclude
one call was to break its id. The query now names the endpoints it bills, so
inclusion is a list entry rather than a naming trick, and Stage 4 was
reordered to run this merge before the flush so its rows exist in time.

Failure posture: ``on_ingestion_complete`` never raises; every failure is
logged and reported as a ``skipped:<reason>`` / ``error:<type>`` result
string so the Stage-4 hook can emit its counter-ish log line.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from synap.cloud.agents.ingestion.prompts.user_profile import (
    build_merge_prompt_text,
    render_overview_rule,
)

logger = logging.getLogger(__name__)

PROFILE_SCHEMA_VERSION = 1
_MAX_MEMORIES_IN_PROMPT = 50


def ingestion_profile_merge_enabled() -> bool:
    """Whether the Stage-4 hook may run the profile merge at ingestion time.

    Off unless ``ENABLE_INGESTION_PROFILE_MERGE`` is set by hand. The profile
    is maintained by the Nap cycle's profile-merge step instead; this flag only
    exists so the old path can be switched back on manually.
    """
    return os.environ.get("ENABLE_INGESTION_PROFILE_MERGE", "false").lower() in (
        "true", "1", "yes",
    )


class _AlreadyApplied(Exception):
    """Raised inside the merge callback when the row already carries this
    ingestion_id (a Stage-4 retry lost the race after the first attempt's
    write landed). Aborts the write transaction cleanly."""


# Memory types that describe a moment, not the person: what happened (episodes),
# how they felt (emotions) and when something is due (temporal events). A profile
# holds stable attributes and a summary, so a batch made only of these cannot
# change it. Facts and preferences can, and so can any type this list does not
# know. The check is a plain rule, not a model call.
_MOMENT_ONLY_TYPES = frozenset({
    "episode", "episodes", "emotion", "emotions", "temporal_event", "temporal_events",
})
_LINE_TYPE = re.compile(r"^\[([A-Za-z_]+)\]\s")


def memory_lines_can_change_profile(memory_lines: List[str], analysis: Optional[Any] = None) -> bool:
    """False when the merge call would be wasted: no memories at all, or only
    memories of a moment-only type, and no caller analysis to go on.

    *memory_lines* are ``"[type] content"`` lines, as built by this module and by
    the end-of-conversation trigger. A line with no recognisable type counts as
    able to change the profile.
    """
    if analysis:
        return True
    for line in memory_lines:
        found = _LINE_TYPE.match(line or "")
        if not found or found.group(1).lower() not in _MOMENT_ONLY_TYPES:
            return True
    return False


def _as_datetime(value: Any) -> Optional[datetime]:
    """An ISO string or datetime as a datetime; naive values are read as UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(value: Any) -> float:
    """Parse an ISO timestamp defensively; unparsable -> epoch 0 (oldest)."""
    if not value or not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def attribute_config_hash(critical_attributes: List[Any]) -> str:
    """sha256[:12] over the canonical attribute config (spec §5.2 _meta)."""
    canonical = json.dumps(
        [
            a.model_dump(mode="json") if hasattr(a, "model_dump") else a
            for a in critical_attributes
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# LLM prompt / response handling
# ---------------------------------------------------------------------------

def _build_merge_prompt(
    *,
    user_profile_cfg,
    current_profile: Optional[dict],
    memory_lines: List[str],
    analysis: Optional[Any],
    conversation_id: Optional[str],
) -> str:
    attrs_spec = []
    for a in user_profile_cfg.critical_attributes:
        spec: Dict[str, Any] = {
            "name": a.name,
            "type": a.type,
            "description": a.description,
            "required": a.required,
        }
        if a.allowed_values:
            spec["allowed_values"] = a.allowed_values
        if a.examples:
            spec["examples"] = a.examples
        attrs_spec.append(spec)

    current_view = {
        "attributes": (current_profile or {}).get("attributes") or {},
        "overview": (current_profile or {}).get("overview") or "",
        "extras": (current_profile or {}).get("extras") or {},
    }

    overview_rule = render_overview_rule(user_profile_cfg)

    return build_merge_prompt_text(
        attrs_spec=attrs_spec,
        current_view=current_view,
        memory_lines=memory_lines,
        analysis=analysis,
        conversation_id=conversation_id,
        overview_rule=overview_rule,
    )


def _parse_llm_json(raw: Any) -> Optional[dict]:
    """Parse the LLM response defensively. None on anything malformed."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    # Strip markdown fences some providers emit even in json mode.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def _normalize_llm_doc(
    doc: dict,
    *,
    user_profile_cfg,
    current_profile: Optional[dict],
    conversation_id: Optional[str],
    now_iso: str,
) -> Optional[dict]:
    """Validate/normalize the LLM document into the spec §5.2 shape.

    Returns None when the document is structurally unusable. Enforces
    allowed_values for enum attributes (invalid -> previous value kept or
    attribute dropped), restricts attributes to the configured set, stamps
    per-attribute ``updated_at`` (changed/new values get ``now``; unchanged
    values keep their previous stamp), and defaults
    ``source_conversation_id``.
    """
    raw_attrs = doc.get("attributes")
    if raw_attrs is None:
        raw_attrs = {}
    if not isinstance(raw_attrs, dict):
        return None

    configured = {a.name: a for a in user_profile_cfg.critical_attributes}
    old_attrs = (current_profile or {}).get("attributes") or {}

    attributes: Dict[str, dict] = {}
    for name, raw in raw_attrs.items():
        if name not in configured:
            continue  # undeclared keys never enter attributes (extras is their home)
        if not isinstance(raw, dict) or "value" not in raw:
            continue
        cfg = configured[name]
        value = raw.get("value")
        old = old_attrs.get(name) if isinstance(old_attrs.get(name), dict) else None
        if cfg.type == "enum" and cfg.allowed_values and value not in cfg.allowed_values:
            if old:
                attributes[name] = dict(old)  # keep previous on rule violation
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        changed = old is None or old.get("value") != value
        source = raw.get("source_conversation_id")
        if not isinstance(source, str) or not source:
            source = (
                conversation_id
                if changed
                else (old or {}).get("source_conversation_id") or conversation_id
            )
        attributes[name] = {
            "value": value,
            "confidence": confidence,
            "updated_at": now_iso if changed else (old or {}).get("updated_at", now_iso),
            "source_conversation_id": source,
        }

    # Unchanged attributes the LLM omitted survive (full-document contract is
    # belt; this is suspenders — never lose a known value to LLM truncation).
    for name, old in old_attrs.items():
        if name not in attributes and name in configured and isinstance(old, dict):
            attributes[name] = dict(old)

    overview = doc.get("overview")
    if not isinstance(overview, str) or not user_profile_cfg.overview_enabled:
        overview = (current_profile or {}).get("overview") or ""

    extras = doc.get("extras")
    if not isinstance(extras, dict):
        extras = (current_profile or {}).get("extras") or {}

    return {"attributes": attributes, "overview": overview, "extras": extras}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def on_ingestion_complete(
    request,
    stage4_output,
    ingestion_id,
    config_loader,
    llm_client,
    store,
    *,
    rdbms=None,
) -> str:
    """Run the post-ingestion profile merge. NEVER raises.

    Returns a result string for the Stage-4 hook's
    ``profile_update result=...`` log line:
    ``"updated"`` | ``"skipped:<reason>"`` | ``"error:<ExceptionType>"``.
    """
    if not ingestion_profile_merge_enabled():
        return "skipped:flag_off"
    try:
        return await _run(
            request, stage4_output, ingestion_id, config_loader, llm_client,
            store, rdbms=rdbms,
        )
    except Exception as exc:  # noqa: BLE001 — profile merge never fails a persist
        logger.warning(
            "profile_update_error ingestion_id=%s request_id=%s error=%s",
            ingestion_id,
            getattr(request, "request_id", None),
            exc,
            exc_info=True,
        )
        return f"error:{type(exc).__name__}"


async def _run(
    request,
    stage4_output,
    ingestion_id,
    config_loader,
    llm_client,
    store,
    *,
    rdbms=None,
) -> str:
    scope = request.scope
    instance_id = str(scope.instance_id)

    # 1. Per-instance MACA config — stage workers hold only the startup
    #    DEFAULT config, so this must go through the shared 300s-TTL loader.
    config = await config_loader.get_config(instance_id=instance_id)
    up_cfg = getattr(config, "user_profile", None)
    if up_cfg is None or not up_cfg.enabled:
        return "skipped:disabled"

    # 2. Scope must identify a caller (user or B2C customer).
    if not scope.user_id and not scope.customer_id:
        return "skipped:no_user_scope"

    # 3. The ingest must have produced memories.
    memory_ids = [str(m) for m in (getattr(stage4_output, "rdbms_memory_ids", None) or [])]
    if not memory_ids:
        return "skipped:no_memories"

    # 4. Owning row (client-scoped uuid5 lookup; B2C falls back to customer).
    owner = await store.resolve_owner_by_scope_uuids(
        client_uuid=scope.client_id,
        user_uuid=scope.user_id,
        customer_uuid=scope.customer_id,
    )
    if owner is None:
        warn_owner_not_found(scope, log_ref=f"ingestion_id={ingestion_id} request_id={request.request_id}")
        return "skipped:owner_not_found"

    # 5. Idempotency pre-check (Stage-4 retries re-run this hook up to 3x).
    current = await store.read(owner)
    pre_meta = (current or {}).get("_meta") or {}
    if pre_meta.get("last_ingestion_id") == str(ingestion_id):
        return "skipped:already_applied"
    pre_version = int(pre_meta.get("version") or 0)

    # 6. Gather inputs: the just-created memories' content + caller analysis.
    memory_lines = await _fetch_memory_lines(memory_ids, rdbms=rdbms)
    if not memory_lines:
        return "skipped:no_memory_content"
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    analysis = metadata.get("analysis")
    conversation_id = metadata.get("conversation_id")
    if not memory_lines_can_change_profile(memory_lines, analysis):
        return "skipped:not_profile_relevant"

    # 7. ONE LLM merge call — outside any transaction/lock (finding 9). It
    #    carries the plain request_id: the merge IS billed now, and Stage 4
    #    runs it before the charge flush so these tokens are on disk in time.
    async def _complete(*, prompt: str, max_tokens: int) -> Any:
        return await llm_client.complete(
            prompt=prompt,
            json_mode=True,
            telemetry_scope=scope,
            telemetry_operation="ingestion.profile_merge",
            telemetry_correlation_id=request.request_id,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    result, _ = await merge_profile(
        owner=owner,
        up_cfg=up_cfg,
        current=current,
        memory_lines=memory_lines,
        analysis=analysis,
        conversation_id=conversation_id,
        idempotency_key=str(ingestion_id),
        written_by="ingestion",
        complete=_complete,
        store=store,
        log_ref=f"ingestion_id={ingestion_id} request_id={request.request_id}",
    )
    return result


def warn_owner_not_found(scope, *, log_ref: str) -> None:
    """Say, at warning level, that a user has no row to hold a profile.

    The row is created the first time an id is seen (``IdMapper.resolve_user`` and
    ``resolve_customer``), so a missing one means the id never went through them.
    Without this line the merge returned a result string nobody read.
    """
    logger.warning(
        "profile_owner_not_found %s client_uuid=%s instance_id=%s user_uuid=%s customer_uuid=%s"
        " (no user or customer row for this scope; the profile was not written)",
        log_ref,
        getattr(scope, "client_id", None),
        getattr(scope, "instance_id", None),
        getattr(scope, "user_id", None),
        getattr(scope, "customer_id", None),
    )


async def merge_profile(
    *,
    owner,
    up_cfg,
    current: Optional[dict],
    memory_lines: List[str],
    analysis: Optional[Any],
    conversation_id: Optional[str],
    idempotency_key: str,
    written_by: str,
    complete: Callable[..., Awaitable[Any]],
    store,
    log_ref: str,
    shadow: bool = False,
    watermark: Optional[str] = None,
) -> Tuple[str, Optional[dict]]:
    """The merge core shared by the ingestion hook and the Nap step.

    One LLM merge of ``memory_lines`` into ``current`` under the instance's
    ``user_profile`` config, then a locked write through ``store``. The caller
    resolves the owner, reads ``current``, gathers the memories and runs its
    own idempotency pre-check; ``idempotency_key`` (an ingestion id or a Nap
    run id) is stamped in ``_meta.last_ingestion_id`` so a retry is a no-op.

    ``shadow=True`` stops after the LLM call and writes nothing.

    ``watermark`` (Core's end-of-conversation trigger) is the ``created_at`` of the
    newest memory merged. It is stored in ``_meta.last_memory_at``, and a write
    whose watermark is not newer than the stored one is dropped as already
    applied. Left as None (Cloud) nothing about the stored document changes.

    Returns ``(result, normalized_doc)``: result is ``"updated"``,
    ``"shadow:would_update"`` or ``"skipped:<reason>"``.
    """
    pre_meta = (current or {}).get("_meta") or {}
    pre_version = int(pre_meta.get("version") or 0)

    now_iso = _now_iso()
    prompt = _build_merge_prompt(
        user_profile_cfg=up_cfg,
        current_profile=current,
        memory_lines=memory_lines,
        analysis=analysis,
        conversation_id=conversation_id,
    )
    raw = await complete(
        prompt=prompt,
        max_tokens=min(4000, max(1000, up_cfg.max_profile_tokens * 2)),
    )

    doc = _parse_llm_json(raw)
    normalized = (
        _normalize_llm_doc(
            doc,
            user_profile_cfg=up_cfg,
            current_profile=current,
            conversation_id=conversation_id,
            now_iso=now_iso,
        )
        if doc is not None
        else None
    )
    if normalized is None:
        logger.warning(
            "profile_update_bad_llm_json %s preview=%r", log_ref, str(raw)[:200],
        )
        return "skipped:bad_llm_json", None

    if shadow:
        return "shadow:would_update", normalized

    attr_hash = attribute_config_hash(up_cfg.critical_attributes)
    ingestion_key = str(idempotency_key)

    def merge(fresh: Optional[dict]) -> dict:
        """Executed under the row lock, on the FRESH document.

        If nobody wrote since our pre-read, take the normalized LLM output
        wholesale (it was computed against exactly this version). If a
        concurrent writer won the race, reconcile attribute-wise: newest
        per-attribute ``updated_at`` wins, the fresh overview is kept, and
        extras merge with ours on top. Whole-document overwrite of unseen
        state is forbidden (finding 9).
        """
        fresh = fresh if isinstance(fresh, dict) else None
        fresh_meta = (fresh or {}).get("_meta") or {}
        if fresh_meta.get("last_ingestion_id") == ingestion_key:
            raise _AlreadyApplied()
        if watermark is not None:
            ours, stored = _as_datetime(watermark), _as_datetime(fresh_meta.get("last_memory_at"))
            if ours is not None and stored is not None and ours <= stored:
                raise _AlreadyApplied()
        fresh_version = int(fresh_meta.get("version") or 0)

        if fresh_version == pre_version:
            merged = {
                "attributes": normalized["attributes"],
                "overview": normalized["overview"],
                "extras": normalized["extras"],
            }
        else:
            fresh_attrs = dict((fresh or {}).get("attributes") or {})
            for name, ours in normalized["attributes"].items():
                theirs = fresh_attrs.get(name)
                if not isinstance(theirs, dict) or _ts(ours.get("updated_at")) >= _ts(
                    theirs.get("updated_at")
                ):
                    fresh_attrs[name] = ours
            merged = {
                "attributes": fresh_attrs,
                "overview": (fresh or {}).get("overview") or "",
                "extras": {
                    **((fresh or {}).get("extras") or {}),
                    **normalized["extras"],
                },
            }

        merged["_meta"] = {
            "schema": PROFILE_SCHEMA_VERSION,
            "version": fresh_version + 1,
            "updated_at": now_iso,
            "last_ingestion_id": ingestion_key,
            "written_by": written_by,
            "attribute_config_hash": attr_hash,
        }
        if watermark is not None:
            merged["_meta"]["last_memory_at"] = watermark
        return merged

    try:
        written = await store.write(owner, merge)
    except _AlreadyApplied:
        return "skipped:already_applied", normalized
    if written is None:
        logger.warning(
            "profile_owner_row_missing %s owner=%s (the row vanished before the write)",
            log_ref, owner,
        )
        return "skipped:owner_row_missing", normalized
    return "updated", normalized

async def _fetch_memory_lines(memory_ids: List[str], *, rdbms=None) -> List[str]:
    """Fetch the just-created memories' content rows.

    Uses the same storage wrapper the Stage-4 worker already persists
    through (SQLAlchemyRDBMSWrapper over the app session factory); ``rdbms``
    is injectable for tests.
    """
    if rdbms is None:
        from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import (
            SQLAlchemyRDBMSWrapper,
        )

        rdbms = SQLAlchemyRDBMSWrapper()

    ids = memory_ids[:_MAX_MEMORIES_IN_PROMPT]
    placeholders = ", ".join(f"CAST(:id{i} AS uuid)" for i in range(len(ids)))
    params = {f"id{i}": mid for i, mid in enumerate(ids)}
    rows = await rdbms.fetch_all(
        f"SELECT id::text AS id, memory_type, content FROM memories "  # noqa: S608 — numbered bind params only
        f"WHERE id IN ({placeholders}) AND deleted_at IS NULL",
        params,
    )
    lines = []
    for row in rows:
        content = (row.get("content") or "").strip()
        if content:
            lines.append(f"[{row.get('memory_type', 'memory')}] {content}")
    return lines
