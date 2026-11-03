"""Prompt text lives in a prompts package, never in an engine file.

Synap Core is cut from these folders by copying files, never by editing them.
A prompt written inline in an engine file would be copied to the open-source
repo with it, and the Core prompt pack could never replace it. On 7 Oct 2026 a
search found seven such prompts. By 23 Oct, when they were moved, this test
counted 67 in 25 files, Jev questions included, which count as prompts for this
rule. Most had arrived in the two weeks between, which is why the rule is a test.

A literal is treated as prompt text when it reads as instructions: at least
fifteen words with sentence punctuation. SQL, Cypher, regular expressions, log
lines and exception messages are not prompts and are skipped. Section labels
such as "THE NEW MEMORY:" are too short to count and may stay in code.

To fix a failure, move the text into the prompts package next to the engine
file (``agents/ingestion/prompts``, ``application/client_api/prompts``,
``application/compaction/passes/prompts``, ``application/compaction/v3/prompts``,
``application/memory_forget/prompts``, ``application/ingestion/prompts``)
and import it under the name the engine already uses. If the text really is not
model-facing, add it to ``NOT_PROMPTS`` with the reason.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# The engine folders that ship in Synap Core.
SCANNED = [
    "cloud/agents/ingestion",
    "cloud/agents/retrieval",
    "cloud/application/client_api",
    "cloud/application/ingestion",
    "cloud/application/conversation",
    "cloud/application/compaction",
    "cloud/application/memory_architecture",
    "cloud/application/profile",
    "cloud/application/memory_forget",
    "cloud/application/shared",
    "cloud/application/sdk_api",
    "cloud/infrastructure/llm",
    # The rest of what ships under synap/cloud (checked 30 Oct 2026).
    # Each is scanned so a prompt written there is caught before the cut.
    "cloud/agents/models",
    "cloud/application/models",
    "cloud/application/main.py",
    "cloud/database",
    "cloud/memory_config",
    "cloud/observability/__init__.py",
    "cloud/observability/tracing.py",
    "cloud/observability/bootstrap.py",
    "cloud/shared",
    "cloud/logging_config.py",
]

# Files inside a scanned folder that do not ship in Core. Their texts are API replies and
# dashboard wording, and the files never reach Core.
NOT_SHIPPED = {
    "cloud/shared/ladder_api.py",
    "cloud/shared/ladder_proposal.py",
    "cloud/shared/ladder_validation.py",
    "cloud/shared/usecase_ladder.py",
    "cloud/shared/scope_ladder.py",
    "cloud/shared/scope_rollout.py",
    "cloud/shared/scope_grants.py",
    "cloud/shared/scope_health.py",
    "cloud/shared/sdk_readiness.py",
    "cloud/shared/event_bus.py",
    "cloud/shared/progress_tracker.py",
    "cloud/shared/region.py",
}

# Prompt text known to sit in an engine file, and why it is still there.
# Each entry is a path relative to synap/; every literal in that file is skipped.
KNOWN_INLINE = {
    # Empty since 30 Oct 2026: the five compaction v3 files moved their texts
    # into cloud/application/compaction/v3/prompts.
}

# Long text that is not sent to any model: (path relative to synap/, first words).
NOT_PROMPTS = {
    ("cloud/application/sdk_api/core_seed.py",
     "config/memory.yml sets user_context_isolation"): "error shown to the developer at server start",
    ("cloud/agents/ingestion/pipeline/knowledge_update.py",
     "out-of-order ingest:"): "reason stored on the skip record",
    ("cloud/application/memory_architecture/build_supervisor.py",
     "This instance is built in region"): "API error shown to the caller",
    ("cloud/application/memory_architecture/build_supervisor.py",
     "This instance has already been rebuilt"): "API error shown to the caller",
    ("cloud/application/memory_architecture/conflict_detector.py",
     "Changing user context isolation"): "config warning shown in the dashboard",
    ("cloud/application/memory_architecture/conflict_detector.py",
     "Confirm the instance's user_relationship"): "config warning shown in the dashboard",
    ("cloud/application/memory_architecture/setup_manager.py",
     "A new memory architecture configuration"): "notification text",
    # Answers shown to the caller: why a text stored nothing, and why a user has no profile (Core).
    ("cloud/agents/ingestion/pipeline/stage1_qualification.py",
     "Nothing was stored"): "reason shown to the caller in the ingestion status",
    ("cloud/application/sdk_api/sdk_users.py",
     "No profile yet for this user"): "API error shown to the caller",
    # The forget route's answer when the server has no Jev key.
    ("cloud/application/sdk_api/sdk_forget.py",
     "Forget by description needs a Jev API key"): "API error shown to the caller",
    # Core's "not available" answer and the not-ready health answer.
    ("cloud/application/sdk_api/core_routes.py",
     "Delete by scope is not available"): "API error shown to the caller",
    ("cloud/application/main.py",
     "The embedding model and the spaCy model are still"): "readiness answer shown to the operator",
    # The MACA plugin: messages the route returns to the developer.
    ("cloud/application/sdk_api/maca_plugin.py",
     "Could not reach Synap Cloud"): "failed-job message returned to the developer",
    ("cloud/application/sdk_api/maca_plugin.py",
     "Synap Cloud returned a config that Synap Core cannot load"): "invalid-job message returned to the developer",
    ("cloud/application/sdk_api/maca_plugin.py",
     "Send the use-case file"): "422 detail returned to the caller",
    ("cloud/application/sdk_api/maca_plugin.py",
     "Say who your users are"): "422 detail returned to the caller",
    ("cloud/application/sdk_api/maca_plugin.py",
     "This install already holds memories"): "422 detail returned to the caller",
    ("cloud/application/sdk_api/maca_plugin.py",
     "The config sets user_context_isolation"): "invalid-job problem returned to the developer",
    # sdk_api was added to SCANNED when the SDK routes moved out of admin_api
    # These are text returned to the API caller,
    # never sent to a model: error details, response notes, warnings, field docs.
    ("cloud/application/sdk_api/sdk_context.py",
     "The scope path could not be resolved"): "503 detail returned to the caller",
    ("cloud/application/sdk_api/sdk_context.py",
     "This instance has user_context_isolation=equals_customer"): "scope note in the response",
    ("cloud/application/sdk_api/sdk_context.py",
     "The scope path names the identity rung"): "scope note in the response",
    ("cloud/application/sdk_api/sdk_context.py",
     "This customer-scope fetch found no tenant-wide"): "scope note in the response",
    ("cloud/application/sdk_api/sdk_conversations.py",
     "no event time supplied"): "warning in the response",
    ("cloud/application/sdk_api/sdk_forget.py",
     "Forget by description is scoped to one user"): "400 detail returned to the caller",
    ("cloud/application/sdk_api/sdk_memories.py",
     "The scope path could not be resolved"): "error message raised to the caller",
    ("cloud/application/sdk_api/sdk_memories.py",
     "The manifest's file parts"): "400 detail returned to the caller",
    ("cloud/application/sdk_api/sdk_users.py",
     "External customer id."): "OpenAPI field description",
    # Added when the check began to scan every folder that ships under
    # synap/cloud (30 Oct 2026). Text for the API caller or the operator.
    ("cloud/application/ingestion/multimodal.py",
     "Only image attachments"): "400 detail returned to the caller",
    ("cloud/application/sdk_api/document_extractor.py",
     "The configured model cannot read images"): "422 detail returned to the caller",
    ("cloud/application/main.py",
     "This instance is B2C"): "400 detail returned to the caller",
    ("cloud/database/schema_compat.py",
     "supervised MACA builds are unavailable"): "startup notice for the operator",
    ("cloud/shared/b2c_contract.py",
     "is a customer-scoped endpoint"): "400 detail returned to the caller",
    ("cloud/shared/b2c_contract.py",
     "customer_id is not accepted"): "400 detail returned to the caller",
    ("cloud/shared/edition.py",
     "Synap Core sets these itself"): "startup error shown to the operator",
    ("cloud/shared/instance_limits.py",
     "This workspace already has"): "429 detail returned to the caller",
    ("cloud/shared/instance_limits.py",
     "This workspace has already created"): "429 detail returned to the caller",
    ("cloud/shared/instance_limits.py",
     "This setup session has already created"): "429 detail returned to the caller",
    ("cloud/shared/prompt_switch.py",
     "Synap Core has no text for the prompt"): "startup error shown to the operator",
    ("cloud/shared/prompt_switch.py",
     "The prompt"): "startup error shown to the operator",
    ("cloud/shared/prompt_switch.py",
     "The Core prompt file"): "startup error shown to the operator",
    ("cloud/shared/scope.py",
     "nested scoping is not enabled"): "400 detail returned to the caller",
    ("cloud/shared/scope.py",
     "this client's scope ladder has not been accepted"): "400 detail returned to the caller",
    # Added when core_identity and memory_config merged into the folders
    # the guard scans. All text shown to a caller or a
    # developer, never sent to a model.
    ("cloud/application/sdk_api/core_identity.py",
     "scope_path is not supported by the open-source server"): "400 detail returned to the caller",
    ("cloud/memory_config/config_schema.py",
     "Which signal types should the agent consider firing anticipation on"): "pydantic field description",
    ("cloud/memory_config/config_schema.py",
     "Lowest pattern priority that participates in candidate generation"): "pydantic field description",
    ("cloud/memory_config/config_schema.py",
     "Explicit scope tier for the prefetched bundle"): "pydantic field description",
    ("cloud/memory_config/config_schema.py",
     "Activate when the user shows one of these affective states"): "pydantic field description",
    ("cloud/memory_config/core_rules.py",
     ": a partial set of ranking weights is not accepted"): "validation message shown to the developer",
}

_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
_SQL = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|MATCH|MERGE|CREATE|UNWIND|OPTIONAL MATCH|CALL)\b")
_REGEX_HINTS = ("\\b", "(?:", "(?P<", "\\s", "\\d", "[^")


def _is_prompts_path(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return "prompts" in parts[:-1] or name == "prompts.py" or name.endswith("_prompts.py")


def _text(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant))
    return None


def _reads_as_instructions(text: str) -> bool:
    if len(text) < 90 or _SQL.match(text) or any(h in text for h in _REGEX_HINTS):
        return False
    words = text.split()
    if len(words) < 15:
        return False
    return bool(re.search(r"[.:?!](\s|$)", text))


def _skipped_parents(tree: ast.AST) -> set:
    """Ids of literals that are docstrings, log arguments or exception messages."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
        if isinstance(node, ast.Call):
            func = node.func
            logging_call = isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS
            exception = (isinstance(func, ast.Name) and func.id.endswith(("Error", "Exception"))) or (
                isinstance(func, ast.Attribute) and func.attr.endswith(("Error", "Exception")))
            if logging_call or exception:
                for arg in node.args:
                    for sub in ast.walk(arg):
                        out.add(id(sub))
    return out


def _violations() -> list:
    found = []
    for folder in SCANNED:
        root = _ROOT / folder
        for path in sorted([root] if root.is_file() else root.rglob("*.py")):
            rel = path.relative_to(_ROOT).as_posix()
            if _is_prompts_path(rel) or rel in KNOWN_INLINE or rel in NOT_SHIPPED:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            skipped = _skipped_parents(tree)
            inner = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values}
            for node in ast.walk(tree):
                if id(node) in skipped or id(node) in inner:
                    continue
                text = _text(node)
                if text is None or not _reads_as_instructions(text):
                    continue
                if any(rel == p and text.lstrip().startswith(w) for p, w in NOT_PROMPTS):
                    continue
                found.append(f"{rel}:{node.lineno}  {' '.join(text.split())[:80]}")
    return found


def test_no_prompt_text_in_engine_files():
    found = _violations()
    assert not found, (
        "Prompt text in an engine file. Move it into the prompts package next to "
        "it (see this file's docstring):\n  " + "\n  ".join(found)
    )


def test_known_inline_entries_still_exist():
    """An exemption for a file that is gone, or moved, exempts nothing."""
    missing = [p for p in KNOWN_INLINE if not (_ROOT / p).exists()]
    assert not missing, f"KNOWN_INLINE names files that do not exist: {missing}"


def test_the_detector_still_sees_a_prompt():
    """The rule itself must keep catching the shape it was written for."""
    sample = ("You are a memory-update classifier for a personal memory system. Decide "
              "what should happen when a NEW memory is similar to an EXISTING one.")
    assert _reads_as_instructions(sample)
    assert not _reads_as_instructions("SELECT content FROM memories WHERE id = :id " * 3)
    assert not _reads_as_instructions("the a an and or of to in on at for with " * 3)
