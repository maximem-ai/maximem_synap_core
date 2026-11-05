"""Which edition of the server is running, and the hooks that differ between them.

``SYNAP_EDITION`` is read here and nowhere else (a guard test enforces it).
Values: ``cloud`` (the default when unset or empty) and ``core``. Anything else
raises ``ValueError`` at import, so a typo stops the server instead of silently
running as Cloud.

Three things live here:

* Plain values the engine asks for: ``name``, ``is_core``, the prompt pack, the
  compaction profile, whether retrieval uses an LLM, where the config comes
  from, the file store adapter, the scope levels, the request-shape policy.
  A value of ``None`` means "no edition choice; the existing setting decides",
  which is how Cloud behaves today.
* The Core switches. Core writes its values into ``os.environ`` at import, so
  the files that read those variables stay unchanged. Core FORCES them: a
  different value already in the environment stops the server with
  ``EditionConflict``. Cloud writes nothing.
* The hooks registry, for features Core does not have.

Hooks
-----
All hook names are declared once, in ``HOOK_NAMES``. Registering or calling a
name that is not declared raises ``UndeclaredHook``. A declared hook with
nothing registered does nothing and returns ``None``. Cloud registers the real
functions the first time a hook is used: ``call``, ``acall`` and ``is_registered``
import ``synap.cloud.cloud_hooks`` and run its ``register_all`` once, in every Cloud process (the consolidation
runner and the MCP service never import main.py). If that import fails, the call
raises ``RuntimeError`` and the next call tries again. Core never imports it and
registers nothing.

Two ways to call, because the engine has both kinds of call site:

* ``call(name, *a, **kw)``: calls the function and returns what it returns. If
  the function is ``async def``, the caller gets the coroutine and must await it.
* ``acall(name, *a, **kw)``: always awaitable. Awaits the function's result if
  it is awaitable. With nothing registered it returns ``None``.

An async call site that uses ``call`` on an empty hook gets ``None`` and cannot
await it, so async code uses ``acall``.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

EDITION_ENV = "SYNAP_EDITION"
CLOUD = "cloud"
CORE = "core"


class EditionConflict(RuntimeError):
    """The environment holds a different value for something Core owns."""


class UndeclaredHook(KeyError):
    """A hook name that is not in ``HOOK_NAMES``."""


@dataclass(frozen=True)
class Edition:
    name: str
    is_core: bool
    prompt_pack: str
    # None: pick automatically, as today (SYNAP_COMPACTION_PROFILE or the keys).
    compaction_profile: Optional[str]
    retrieval_llm: bool
    config_source: str
    file_store: str
    # None: the configured scope ladder, any depth.
    scope_levels: Optional[Tuple[str, ...]]
    # None: SYNAP_B2C_ID_POLICY decides (default warn).
    request_shape_policy: Optional[str]
    # Whether this server starts the gRPC streaming server. A yes/no value.
    serves_grpc: bool
    # Whether the user profile is updated when a conversation ends or goes idle
    # (Core). Cloud updates it inside the consolidation cycle instead. A yes/no value.
    profile_at_conversation_end: bool
    # Whether nested scoping (the scope ladder and its rollout gate) exists. A
    # yes/no value. Core has neither file, so it never asks them anything.
    nested_scoping: bool
    # Whether /health answers "not ready" until the first-start downloads are done.
    # A yes/no value. Cloud ships its models in the image.
    waits_for_first_start: bool
    # Whether a start-up service whose package is not in the build is reported as one INFO line,
    # not a warning with a traceback (Core). A yes/no value.
    quiet_absent_services: bool
    # Whether /health answers "not ready" while an ingestion worker has failed to start (Core). Without it a
    # server whose workers are down looks healthy and every memory waits in the queue. A yes/no value.
    health_needs_ingestion: bool
    # Whether the engine's own Jev steps (graph support check, resolution judges, temporal gate, compaction
    # completeness check) may call Jev when a key is set. Core runs none of them; its key, if the
    # developer sets one, is for forget by description only. A yes/no value.
    engine_uses_jev: bool
    # The mode forget by description uses when the request does not name one: "preview" (file a request for the
    # dashboard to approve) or "direct" (find and delete now). Core has no dashboard to approve a preview.
    forget_default_mode: str
    # Whether the server checks its required settings and prompt texts before it starts and stops with
    # the reason when one is wrong (Core). A yes/no value; see ``shared/core_startup.py``.
    checks_settings_at_start: bool
    # Whether this server starts the Cloud-only background services at boot (credit and email workers,
    # instance deletion, migration, connector sync, the playground). Core does not ship their packages
    # and does not try them. A yes/no value.
    runs_cloud_services: bool


_CLOUD = Edition(
    name=CLOUD,
    is_core=False,
    prompt_pack="cloud",
    compaction_profile=None,
    retrieval_llm=True,
    config_source="table",
    file_store="s3",
    scope_levels=None,
    request_shape_policy=None,
    serves_grpc=True,
    profile_at_conversation_end=False,
    nested_scoping=True,
    waits_for_first_start=False,
    quiet_absent_services=False,
    health_needs_ingestion=False,
    engine_uses_jev=True,
    forget_default_mode="preview",
    checks_settings_at_start=False,
    runs_cloud_services=True,
)

_CORE = Edition(
    name=CORE,
    is_core=True,
    prompt_pack="core",
    compaction_profile="configured",
    retrieval_llm=False,
    config_source="file",
    file_store="local",
    scope_levels=("user", "customer", "install"),
    request_shape_policy="enforce",
    serves_grpc=False,
    profile_at_conversation_end=True,
    nested_scoping=False,
    waits_for_first_start=True,
    quiet_absent_services=True,
    health_needs_ingestion=True,
    engine_uses_jev=False,
    forget_default_mode="direct",
    checks_settings_at_start=True,
    runs_cloud_services=False,
)

# What Core puts in the environment. The readers of these
# variables are not edition-aware and are not changed.
CORE_ENV: Dict[str, str] = {
    "STAGE3_EXTRACTION_MODE": "hybrid_graph",
    "SYNAP_GRAPH_EXTRACTION_V2": "true",
    "SYNAP_GRAPH_PER_SCOPE": "true",
    "SYNAP_GRAPH_NODE_MEMORIES": "true",
    "SYNAP_GRAPH_EDGE_MEMORY_IDS": "true",
    "SYNAP_GRAPH_SERVE_RELATIONSHIPS": "true",
    "SYNAP_GRAPH_DROP_LONE_ENTITIES": "true",
    # Retiring or deleting a memory releases its graph support (edges and
    # nodes close or go), and accurate mode serves a graph statement only
    # while a live memory supports it. See agents/ingestion/storage/graph_lifecycle.py.
    "SYNAP_GRAPH_LIFECYCLE": "true",
    "SYNAP_GRAPH_SERVE_CHECKS_LIVENESS": "true",
    "SYNAP_WRITER_ENTITY": "true",
    "SYNAP_CUSTOMER_ENTITY": "true",
    "SYNAP_TEMPORAL_KEEP_PERSON": "true",
    "SYNAP_ER_RESOLUTION_MODE": "adjudicated",
    "SYNAP_ER_HOLD_UNSURE": "false",
    "SYNAP_ER_SCOPED_REGISTRY": "true",
    "SYNAP_ER_IDENTIFIERS": "true",
    "SYNAP_ER_NODE_KEYS": "true",
    "SYNAP_ER_GLOBAL_DICTIONARY": "true",
    "SYNAP_ER_UPGRADE_NAMES": "true",
    # The three exceptions.
    "SYNAP_MACA_EXTRACTION_INSTANCES": "*",
    "SYNAP_B2C_ID_POLICY": "enforce",
    "SYNAP_MACA_CHUNKING_INSTANCES": "*",
    # The temporal gate uses its regex path; the Jev path is Cloud's.
    "SYNAP_TEMPORAL_GATE": "regex",
    # Per-user dedup stays on: a knowledge update never compares one user's
    # memories with another's. This is the code default, pinned.
    "ENABLE_KU_USER_SCOPED_DEDUP": "true",
    # The guardrail fixes. They change behaviour for both editions, so each is
    # off on Cloud until the regression check and on for Core. A setting left out of the YAML limits nothing.
    "SYNAP_GUARDRAIL_INGEST_CAP": "true",
    "SYNAP_GUARDRAIL_PROHIBITED_CONTENT": "true",
    "SYNAP_GUARDRAIL_RETRIEVAL_RATE": "true",
    "SYNAP_GUARDRAIL_RETENTION": "true",
    # The last guardrail fixes: entity grounding and custom categories.
    # Same rule: off on Cloud, on for Core.
    "SYNAP_GUARDRAIL_ENTITY_GROUNDING": "true",
    "SYNAP_CUSTOM_CATEGORIES": "true",
}

# Switches that are off by default and belong to something Core does not have,
# and the flags of things already cut. Core refuses to
# start when one is turned on, so a developer who sets it hears that it does
# nothing here, instead of getting a Core that quietly ignores it.
CORE_REFUSED_SWITCHES: Tuple[str, ...] = (
    # Date pre-pass, assistant recall, speaker attribution, graph relink.
    "SYNAP_TEMPORAL_V2",
    "SYNAP_ASSISTANT_RECALL_IN_BASELINE",
    "ENABLE_SPEAKER_ATTRIBUTION",
    "ENABLE_ECHO_GUARD",
    "ENABLE_GRAPH_RELINK",
    # Knowledge-update guards, delete and batching; the profile merge after stage 4.
    "ENABLE_KU_DELETE",
    "ENABLE_KU_DELETE_SUBJECT_GUARD",
    "ENABLE_KU_DEPRECATE_WITH_LINK",
    "ENABLE_KU_MERGE_SYNTHESIS",
    "ENABLE_KU_TYPE_AWARE_DISPATCH",
    "ENABLE_KU_VALUE_AWARE_RECALL",
    "ENABLE_KU_BATCH_CLASSIFY",
    "SYNAP_KU_DIRECTION_BY_DOCUMENT_DATE",
    "SYNAP_KU_JEV_SCREEN",
    "SYNAP_KU_RETIRED_VALUE_GUARD",
    "SYNAP_KU_RETRACTION_SWEEP_V2",
    "SYNAP_KU_SHARED_BANDS",
    "SYNAP_KU_VALUE_SIBLING_SWEEP",
    "ENABLE_INGESTION_PROFILE_MERGE",
    # The entity-resolution extras (judge steps).
    "SYNAP_ER_SECOND_LOOK",
    "SYNAP_ER_WORLD_LOOKUP",
    "SYNAP_ER_DESCRIPTIONS",
    "SYNAP_ER_PROFILES",
    "SYNAP_ER_JEV_ADJUDICATION",
    "SYNAP_ER_LOCAL_JUDGE",
    # The graph Jev steps and the measured no-gain extras.
    "SYNAP_STAGE3_EARLIER_REFERENCES",
    "SYNAP_GRAPH_EVENT_NODES",
    "SYNAP_GRAPH_STATEMENTS",
    "SYNAP_ENTITY_EXTRACTION_PROMPT",
    # The sweep repair (a second extraction call that adds rows the first call missed).
    # Its prompt is a Cloud prompt that Core does not have, so turning it on would do nothing.
    "SYNAP_INGEST_SWEEP_REPAIR",
)

# Settings that Core does not read at all. Any value is refused. ``JEV_API_KEY`` is not here: Core reads it for
# forget by description and for nothing else. The engine's own Jev steps are shut off by the edition
# value ``engine_uses_jev`` (``shared/jev_client.py``), not by refusing the key.
CORE_REFUSED_SETTINGS: Tuple[str, ...] = (
    "SYNAP_ER_MODEL",
    # Picks Cloud's per-provider prompt variants and a Gemini client; Core has one provider setting.
    "EVAL_PROVIDER",
)

# Per-instance lists name the instances a feature is on for. Core has one instance and runs the code defaults,
# so a list a developer writes is refused. The edition's own pins of this shape are in CORE_ENV.
_PER_INSTANCE_LIST = re.compile(
    r"^(SYNAP_|STAGE|GRAPH_|ENABLE_)\w*(_BY_INSTANCE|_INSTANCES|_INSTANCE_ALLOWLIST)$"
)

_OFF_WORDS = ("", "0", "false", "no", "off")


def resolve(env: Mapping[str, str]) -> Edition:
    """The edition named by *env*. Unset or empty means Cloud."""
    raw = (env.get(EDITION_ENV) or "").strip().lower()
    if raw in ("", CLOUD):
        return _CLOUD
    if raw == CORE:
        return _CORE
    raise ValueError(
        f"{EDITION_ENV}={raw!r} is not valid. Use {CLOUD!r} (the default) or {CORE!r}."
    )


def apply_environment(edition: Edition, env: MutableMapping[str, str]) -> None:
    """Write the edition's environment into *env*. Cloud writes nothing.

    Checks every variable before writing any, so a refusal leaves *env* as it
    was. Equal values are accepted; a different one raises ``EditionConflict``.
    """
    if not edition.is_core:
        return
    clashes = []
    for key, wanted in CORE_ENV.items():
        found = env.get(key)
        if found is not None and found.strip().lower() != wanted.lower():
            clashes.append(
                f"{key} is {found!r} but Core needs {wanted!r}"
            )
    for key in CORE_REFUSED_SWITCHES:
        found = env.get(key)
        if found is not None and found.strip().lower() not in _OFF_WORDS:
            clashes.append(f"{key} is {found!r} but it is a Cloud-only switch and Core has none of what it turns on")
    for key in CORE_REFUSED_SETTINGS:
        found = env.get(key)
        if found is not None and found.strip():
            clashes.append(f"{key} is set but Core does not read it")
    for key, found in env.items():
        if key not in CORE_ENV and _PER_INSTANCE_LIST.match(key) and found.strip():
            clashes.append(f"{key} is a per-instance list, and Core has one instance")
    if clashes:
        raise EditionConflict(
            "Synap Core sets these itself and will not run with a different value. "
            "Remove them from the environment: " + "; ".join(clashes) + "."
        )
    for key, wanted in CORE_ENV.items():
        env[key] = wanted


_edition: Edition = resolve(os.environ)
apply_environment(_edition, os.environ)

name: str = _edition.name
is_core: bool = _edition.is_core
current: Edition = _edition
prompt_pack: str = _edition.prompt_pack
compaction_profile: Optional[str] = _edition.compaction_profile
retrieval_llm: bool = _edition.retrieval_llm
config_source: str = _edition.config_source
file_store: str = _edition.file_store
scope_levels: Optional[Tuple[str, ...]] = _edition.scope_levels
request_shape_policy: Optional[str] = _edition.request_shape_policy
serves_grpc: bool = _edition.serves_grpc
profile_at_conversation_end: bool = _edition.profile_at_conversation_end
nested_scoping: bool = _edition.nested_scoping
waits_for_first_start: bool = _edition.waits_for_first_start
quiet_absent_services: bool = _edition.quiet_absent_services
health_needs_ingestion: bool = _edition.health_needs_ingestion
engine_uses_jev: bool = _edition.engine_uses_jev
forget_default_mode: str = _edition.forget_default_mode
checks_settings_at_start: bool = _edition.checks_settings_at_start
runs_cloud_services: bool = _edition.runs_cloud_services


# Every hook the engine may call. Add a name here before registering or calling it.
HOOK_NAMES: Tuple[str, ...] = (
    # Ingestion. Several of these serve more than one call site and take
    # the site name as their first argument (see cloud_hooks/ingestion.py).
    "pii_ingestion",
    "billing_charge",
    "analytics",
    "consolidation_links",
    "replication_log_write",
    "backup_pii_step",
    # The alert log line for a failed vector persist (stage 4).
    "vector_failure_log",
    # Ingestion's own names for usage tracking and analytics (the shared names
    # "usage_tracking" and "analytics_*" are separate hooks).
    "ingestion_usage_tracking",
    "ingestion_analytics",
    # Retrieval
    "retrieval_pii_classify",
    "retrieval_pii_reveal",
    "consolidation_tripwire",
    "anticipation_miss_record",
    "visibility_and_grants",
    # The HQE question search, called from client_api/retrieval_manager.py.
    "retrieval_question_search",
    # "billing_charge" above belongs to ingestion, and a hook name holds one
    # function, so retrieval's billing block has its own name. "usage_tracking"
    # below is also retrieval's (retrieval usage events).
    "retrieval_billing_charge",
    # Application
    "instance_readiness_check",
    "maca_build_supervisor",
    "llm_trace_sink",
    "daily_scope_check",
    "forget_dashboard_queue_write",
    # Added 30 Oct 2026: the Cloud route assembly, the readiness
    # exception handlers, and the analytics worker start and flush.
    "cloud_routers",
    "instance_readiness_handlers",
    "analytics_worker_start",
    "analytics_shutdown",
    "request_logging_middleware",
    "credit_enforcement",
    "grpc_push",
    "review_queue_write",
    "usage_tracking",
    # Added 30 Oct 2026: the privacy boundary around every provider call, called
    # from infrastructure/llm/manager.py. Core has no privacy folder.
    "provider_guard_messages",
    "provider_guard_response",
    # The Jev re-rank of retrieval candidates ("enabled" / "items"), called
    # from client_api/retrieval_manager.py. jev_rerank.py stays behind.
    "jev_rerank",
    # Graph and entity resolution (cloud_hooks/graph.py). The judge steps and
    # what they need, the question generator (HQE), the per-provider prompt variants.
    "entity_judge_llm",
    "entity_judges",
    "ingestion_hqe_generator",
    "ingestion_prompt_variants",
    # The separate entity extractor (one LLM call for entities, then another for relationships). Cloud's
    # older graph path uses it; Core extracts entities and relationships in one call (graph extraction
    # v2) and has no entity prompt, so the hook returns None there and the file stays behind.
    "ingestion_entity_extractor",
    # Metrics counters and security log lines (observability/metrics.py and
    # request_fields.py stay behind). Reached through shared/hook_signals.py.
    "metrics_counter",
    "security_log",
    # The PII gate on one stored conversation turn (conversation/tracker.py). Core has no
    # PII step and registers nothing, so the turn is stored as it arrived.
    "conversation_turn_protect",
    # Core audit (30 Oct 2026): three steps of delete and forget that read Cloud-only
    # folders (consolidation, the LLM call traces). Core has none of those, so it
    # registers nothing and each step does nothing.
    "memory_deletion_propagation",
    "forget_cache_invalidate",
    "forget_llm_trace_scrub",
    # The client for a standalone embedding service over HTTP (Cloud). The engine's "remote" embedding
    # provider asks this hook. Core registers nothing, ships no such client, and uses the local model.
    "embedding_remote_client",
)

# The Cloud-only package that registers every real hook function. Named by this
# constant, never written as an import, so the Core import check does not see it.
CLOUD_HOOKS_PACKAGE = "synap.cloud.cloud_hooks"

_hooks: Dict[str, Callable[..., Any]] = {}
_hooks_loaded = False
_hooks_loading = False
_hooks_lock = threading.RLock()


def _declared(hook: str) -> None:
    if hook not in HOOK_NAMES:
        raise UndeclaredHook(
            f"{hook!r} is not a declared hook. Add it to HOOK_NAMES in "
            "synap/cloud/shared/edition.py first."
        )


def _load_cloud_hooks() -> None:
    """Import the Cloud hooks package once. Core never does. Raises if it fails."""
    global _hooks_loaded, _hooks_loading
    if _hooks_loaded or is_core:
        return
    with _hooks_lock:
        # The lock is re-entrant: a hook module that calls a hook while the
        # package is importing gets here on the same thread and must not recurse.
        if _hooks_loaded or _hooks_loading:
            return
        _hooks_loading = True
        try:
            importlib.import_module(CLOUD_HOOKS_PACKAGE).register_all()
            _hooks_loaded = True
        except Exception as exc:
            raise RuntimeError(
                f"Cloud could not load its hooks package {CLOUD_HOOKS_PACKAGE!r}: "
                f"{exc!r}. Billing, PII and the other Cloud-only features would be "
                "silently off, so the server stops instead."
            ) from exc
        finally:
            _hooks_loading = False


def register(hook: str, fn: Callable[..., Any]) -> None:
    """Attach *fn* to a declared hook. A second registration replaces the first."""
    _declared(hook)
    _hooks[hook] = fn


def is_registered(hook: str) -> bool:
    _declared(hook)
    _load_cloud_hooks()
    return hook in _hooks


def reset_hooks() -> None:
    """Remove every registered hook and stop the lazy load from refilling them. For tests."""
    global _hooks_loaded
    with _hooks_lock:
        _hooks.clear()
        _hooks_loaded = True


def load_cloud_hooks_for_test() -> None:
    """Undo ``reset_hooks``: the next hook use loads the Cloud package again. For tests."""
    global _hooks_loaded
    with _hooks_lock:
        _hooks_loaded = False


def call(hook: str, *args: Any, **kwargs: Any) -> Any:
    """Call the hook's function and return its result; ``None`` if none is registered.

    An ``async def`` function returns a coroutine here, which the caller awaits.
    """
    _declared(hook)
    _load_cloud_hooks()
    fn = _hooks.get(hook)
    if fn is None:
        return None
    return fn(*args, **kwargs)


async def acall(hook: str, *args: Any, **kwargs: Any) -> Any:
    """Like ``call``, for async code: awaits the result when it is awaitable."""
    result = call(hook, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# Who the caller is. "key": each request proves who it is with an
# API key (Cloud). "fixed": there is one client and one instance, no key lookup
# (Core). ``sdk_api/core_identity.install_edition_identity`` reads this and swaps
# the auth dependency; no engine file looks at the edition to decide it.
identity: str = "fixed" if _edition.is_core else "key"
