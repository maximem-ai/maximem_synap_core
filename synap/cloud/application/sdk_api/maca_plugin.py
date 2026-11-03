"""The MACA plugin, Core side.

A developer on a paid Synap Cloud plan can have MACA write their memory config. Core sends the
developer's use-case file to Cloud with their Cloud key and gets a YAML back. No MACA logic is in
Core: this module is the outbound call, the job around it, and the checks on what comes back.

    POST /v1/maca/generate      start a job (a MACA run takes 15 to 20 minutes); returns a job id
    GET  /v1/maca/jobs/{id}     status, and the YAML once it is done

The result is saved as ``memory.generated.yml`` beside the active config. It never overwrites
``memory.yml``: the developer reads it and renames it. A result Core's loader refuses is saved as
``memory.generated.invalid.yml`` for inspection and is not offered as usable.

The Cloud key (``SYNAP_CLOUD_API_KEY``) is outbound only. It goes to one fixed address, in one
header, with redirects off. It is never returned by a route, written to a job file, or left in a
log line or an error message: everything that came from Cloud or from an exception passes through
``scrub`` first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.memory_config import file_loader
from synap.cloud.memory_config.core_convert import to_core_config

logger = logging.getLogger("synap.maca_plugin")

KEY_ENV = "SYNAP_CLOUD_API_KEY"
#: The one address the key is ever sent to. Not a setting.
CLOUD_URL = "https://synap-cloud-prod.maximem.ai"
BUILDS_PATH = "/api/v1/management/maca/builds"
POLL_SECONDS = 10.0
MAX_WAIT_SECONDS = 45 * 60
MAX_POLL_FAILURES = 6
REQUEST_TIMEOUT = 30.0
MAX_MARKDOWN_BYTES = 256 * 1024
MAX_YAML_BYTES = 1024 * 1024
DOCS_HINT = "See the MACA plugin page in the Synap Core docs: https://docs.maximem.ai/core/maca-plugin"
FOLDER_HINT = " ".join((
    "Synap Core cannot write to the config folder.",
    "The plugin has nowhere to save its result.",
    "On Linux, start the stack with SYNAP_USER=$(id -u):$(id -g) in front of docker compose,",
    "so the server runs as the owner of the folder.",
    "Also check that it is not mounted read-only.",
))

GENERATED_NAME = "memory.generated.yml"
INVALID_NAME = "memory.generated.invalid.yml"

RUNNING = ("queued", "running")


# --------------------------------------------------------------------------- the key

def _key() -> Optional[str]:
    value = (os.environ.get(KEY_ENV) or "").strip()
    return value or None


_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}")
_CLOUD_KEY_SHAPE = re.compile(r"synap_[A-Za-z0-9]{8,}")


def scrub(text: Any) -> str:
    """Text with the Cloud key, and anything shaped like one, taken out."""
    out = str(text)
    key = _key()
    if key:
        out = out.replace(key, "[key removed]")
    out = _BEARER.sub("Bearer [key removed]", out)
    return _CLOUD_KEY_SHAPE.sub("[key removed]", out)


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub(record.getMessage())
        record.args = ()
        record.exc_info = None  # a traceback is text we have not read
        return True


logger.addFilter(_ScrubFilter())


# --------------------------------------------------------------------------- jobs

@dataclass
class Job:
    job_id: str
    status: str = "queued"          # queued, running, succeeded, invalid, failed
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    build_id: Optional[str] = None  # Cloud's id for the run; never the key
    problems: List[str] = field(default_factory=list)
    removed_fields: List[str] = field(default_factory=list)
    config_file: Optional[str] = None
    user_relationship: str = ""     # b2c or b2b: what the developer built the config for

    def touch(self) -> None:
        self.updated_at = _now()


_jobs: Dict[str, Job] = {}
_tasks: Dict[str, "asyncio.Task[None]"] = {}
_loaded = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def active_config_path() -> Path:
    return Path(os.environ.get(file_loader.CONFIG_PATH_ENV) or file_loader.DEFAULT_CONFIG_PATH)


def _config_dir() -> Path:
    return active_config_path().parent


def _jobs_dir() -> Path:
    return _config_dir() / ".maca" / "jobs"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _folder_problem() -> Optional[str]:
    """None when the plugin can write in the config folder, else what to do about it."""
    probe = _jobs_dir() / f".probe-{secrets.token_hex(4)}"
    try:
        _write_atomic(probe, "")
    except OSError:
        return FOLDER_HINT
    probe.unlink(missing_ok=True)
    return None


def _save(job: Job) -> None:
    job.touch()
    try:
        _write_atomic(_jobs_dir() / f"{job.job_id}.json", json.dumps(asdict(job), indent=2))
    except OSError as exc:
        logger.warning("maca job %s could not be saved: %s", job.job_id, type(exc).__name__)


def _load_jobs() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        for p in sorted(_jobs_dir().glob("maca_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                job = Job(**data)
                if job.status in RUNNING and job.build_id is None:
                    # cut off before Cloud had the file; nothing can resume it
                    job.status = "failed"
                    job.message = "The job was interrupted before it reached Synap Cloud. Start it again."
                    _save(job)
                _jobs.setdefault(job.job_id, job)
            except (OSError, ValueError, TypeError, KeyError):
                continue
    except OSError:
        pass


def reset_for_tests() -> None:
    global _loaded
    for t in _tasks.values():
        t.cancel()
    _jobs.clear()
    _tasks.clear()
    _loaded = False


def _public(job: Job) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "removed_fields": job.removed_fields,
        "problems": job.problems,
        "config_file": job.config_file,
    }
    if job.config_file and job.status in ("succeeded", "invalid"):
        try:
            out["yaml"] = (_config_dir() / job.config_file).read_text(encoding="utf-8")
        except OSError:
            out["yaml"] = None
    return out


def _fail(job: Job, message: str) -> None:
    job.status, job.message = "failed", scrub(message)[:600]
    _save(job)


# --------------------------------------------------------------------------- talking to Cloud

def _reason(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return scrub(resp.text)[:300]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or json.dumps(detail)
    return scrub(detail)[:300]


def _cloud_error(resp: httpx.Response) -> str:
    reason = _reason(resp)
    code = resp.status_code
    if code in (401, 403):
        return f"Synap Cloud did not accept the key: {reason}"
    if code == 402:
        return f"Synap Cloud could not start the run, usually because the account has too few credits: {reason}"
    if code == 429:
        return f"Synap Cloud is not accepting a run for this key right now. Try again later. {reason}".strip()
    if 300 <= code < 400:
        return "Synap Cloud answered with a redirect, which Synap Core does not follow."
    return f"Synap Cloud answered {code}: {reason}".strip()


NETWORK_MESSAGE = (
    "Could not reach Synap Cloud (a network error). Synap Core keeps working. "
    "Check the connection and try again."
)


class _CloudError(Exception):
    """A failure whose message is already safe to show."""


#: Tests put a fake Cloud here. It is never set in a running server.
_transport: Optional[httpx.AsyncBaseTransport] = None


def _client() -> httpx.AsyncClient:
    # Redirects off: the key must go to the one address above and nowhere else.
    return httpx.AsyncClient(
        base_url=CLOUD_URL, timeout=REQUEST_TIMEOUT, follow_redirects=False, transport=_transport,
        headers={"Authorization": f"Bearer {_key()}", "Accept": "application/json"},
    )


async def _submit(client: httpx.AsyncClient, markdown: str, relationship: str) -> str:
    try:
        resp = await client.post(BUILDS_PATH, json={
            "usecase_markdown": markdown,
            "config_schema_version": file_loader.CONFIG_SCHEMA_VERSION,
            "target": "core",
            "user_relationship": relationship,
        })
    except httpx.RequestError:
        raise _CloudError(NETWORK_MESSAGE) from None
    if resp.status_code not in (200, 201, 202):
        raise _CloudError(_cloud_error(resp))
    try:
        build_id = str(resp.json()["build_id"])
    except (ValueError, KeyError, TypeError):
        raise _CloudError("Synap Cloud accepted the file but did not return a build id.") from None
    return build_id


async def _poll(client: httpx.AsyncClient, job: Job) -> None:
    waited, failures = 0.0, 0
    while True:
        try:
            resp = await client.get(f"{BUILDS_PATH}/{job.build_id}")
            failures = 0
        except httpx.RequestError:
            failures += 1
            if failures >= MAX_POLL_FAILURES:
                raise _CloudError(NETWORK_MESSAGE) from None
            resp = None
        if resp is not None:
            if resp.status_code != 200:
                raise _CloudError(_cloud_error(resp))
            try:
                body = resp.json()
            except ValueError:
                raise _CloudError("Synap Cloud sent a status Synap Core could not read.") from None
            state = str(body.get("status", ""))
            if state == "succeeded":
                return
            if state == "failed":
                raise _CloudError(f"The MACA run failed: {scrub(body.get('error') or 'no reason given')[:300]}")
            if state in RUNNING and job.status != "running":
                job.status = "running"
                _save(job)
        if waited >= MAX_WAIT_SECONDS:
            raise _CloudError("The MACA run is taking longer than expected. It may still finish on Synap Cloud; run the job again later.")
        await asyncio.sleep(POLL_SECONDS)
        waited += POLL_SECONDS


async def _fetch_config(client: httpx.AsyncClient, build_id: str) -> str:
    try:
        resp = await client.get(f"{BUILDS_PATH}/{build_id}/config", headers={"Accept": "application/yaml"})
    except httpx.RequestError:
        raise _CloudError(NETWORK_MESSAGE) from None
    if resp.status_code != 200:
        raise _CloudError(_cloud_error(resp))
    if len(resp.content) > MAX_YAML_BYTES:
        raise _CloudError("Synap Cloud returned a config larger than Synap Core accepts.")
    return resp.text


# --------------------------------------------------------------------------- what comes back

def _header(removed: List[str]) -> str:
    lines = [
        f"# Generated by the MACA plugin on {_now()} for Synap Core.",
        f"# Review it, then rename it to {active_config_path().name} to use it. "
        "Synap never overwrites that file.",
    ]
    if removed:
        shown = ", ".join(removed[:25]) + (f", and {len(removed) - 25} more" if len(removed) > 25 else "")
        lines.append(f"# Left out, because Synap Core has nothing that reads them ({len(removed)}): {shown}")
    return "\n".join(lines) + "\n"


def _accept(job: Job, returned: str) -> None:
    """Convert, validate and save what Cloud returned."""
    try:
        raw = yaml.safe_load(returned)
    except yaml.YAMLError:
        return _store_invalid(job, returned, ["Synap Cloud returned text that is not valid YAML."])
    converted, removed = to_core_config(raw)
    job.removed_fields = removed
    wanted = RELATIONSHIPS.get(job.user_relationship)
    got = file_loader._scoping_mode_of(converted if isinstance(converted, dict) else {})
    if wanted and got and got != wanted:
        return _store_invalid(job, _header(removed) + yaml.safe_dump(converted, sort_keys=False, allow_unicode=True), [
            f"The config sets user_context_isolation to {got!r}, but you asked for {job.user_relationship!r} "
            f"({wanted!r}). Set it to {wanted!r} by hand, or run the plugin again."])
    text = _header(removed) + yaml.safe_dump(converted, sort_keys=False, allow_unicode=True)
    try:
        file_loader.parse_and_validate(text, GENERATED_NAME)
    except file_loader.MemoryConfigError as exc:
        return _store_invalid(job, text, [scrub(line) for line in str(exc).splitlines() if line.strip()])
    target = _config_dir() / GENERATED_NAME
    if target.resolve() == active_config_path().resolve():
        return _fail(job, f"{GENERATED_NAME} is the active config file; the plugin never overwrites it.")
    try:
        _write_atomic(target, text)
    except OSError:
        return _fail(job, FOLDER_HINT)
    job.status, job.config_file = "succeeded", GENERATED_NAME
    job.message = (
        f"Saved as {GENERATED_NAME}. Review it and rename it to {active_config_path().name} to use it."
        + (f" {len(removed)} fields Synap Core does not use were left out." if removed else "")
    )
    _save(job)


def _store_invalid(job: Job, text: str, problems: List[str]) -> None:
    job.problems = problems[:40]
    try:
        _write_atomic(_config_dir() / INVALID_NAME, text)
        job.config_file = INVALID_NAME
    except OSError:
        job.config_file = None
    job.status = "invalid"
    job.message = (
        f"Synap Cloud returned a config that Synap Core cannot load. It is saved as {INVALID_NAME} "
        "so you can read it; it is not usable as it is."
    )
    _save(job)


# --------------------------------------------------------------------------- running a job

async def _run(job_id: str, markdown: Optional[str], relationship: str = "b2b") -> None:
    job = _jobs[job_id]
    try:
        async with _client() as client:
            if job.build_id is None:
                if markdown is None:  # only a job loaded from disk has none, and loading fails those
                    return _fail(job, "The job was interrupted before it reached Synap Cloud. Start it again.")
                job.build_id = await _submit(client, markdown, relationship)
                job.status, job.message = "running", "The MACA run has started on Synap Cloud. It takes 15 to 20 minutes."
                _save(job)
            await _poll(client, job)
            _accept(job, await _fetch_config(client, job.build_id))
    except _CloudError as exc:
        _fail(job, str(exc))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a job must always end in a state the developer can read
        logger.error("maca job %s failed: %s", job_id, type(exc).__name__)
        _fail(job, "The MACA job failed unexpectedly. See the server log.")
    finally:
        _tasks.pop(job_id, None)


def _start(job: Job, markdown: Optional[str], relationship: str = "b2b") -> None:
    _tasks[job.job_id] = asyncio.create_task(_run(job.job_id, markdown, relationship))


def _running_job() -> Optional[Job]:
    for job in _jobs.values():
        if job.status in RUNNING:
            return job
    return None


# --------------------------------------------------------------------------- who the users are

#: What the developer can say, and the scoping mode each one means. Core has no other kind of app.
RELATIONSHIPS = {"b2c": "equals_customer", "b2b": "strict"}
_RELATIONSHIP_OF_MODE = {mode: rel for rel, mode in RELATIONSHIPS.items()}


def _current_relationship() -> Optional[str]:
    """The relationship of the install as it stands: the mode memories are stored under, else the active config's."""
    mode = file_loader.pinned_scoping_mode()
    if mode is None:
        try:
            raw = yaml.safe_load(active_config_path().read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            raw = None
        mode = file_loader._scoping_mode_of(raw if isinstance(raw, dict) else {})
    return _RELATIONSHIP_OF_MODE.get(mode or "")


def _choose_relationship(asked: Any) -> "str | JSONResponse":
    """The relationship to build for. Asked for wins; otherwise the install's own; a mismatch with stored memories is refused."""
    current = _current_relationship()
    if asked in (None, ""):
        if current is None:
            return _error(422, "user_relationship_missing",
                          "Say who your users are: user_relationship 'b2c' (each user is their own customer) "
                          "or 'b2b' (users sit under a customer).")
        return current
    relationship = str(asked).strip().lower()
    if relationship not in RELATIONSHIPS:
        return _error(422, "user_relationship_invalid", "user_relationship must be 'b2c' or 'b2b'.")
    held = file_loader.pinned_scoping_mode()
    if held is not None and RELATIONSHIPS[relationship] != held:
        return _error(422, "user_relationship_conflict",
                      f"This install already holds memories stored for '{_RELATIONSHIP_OF_MODE.get(held, held)}'. "
                      "The scoping mode cannot change once memories exist, so a config for another kind of app would not load.")
    return relationship


# --------------------------------------------------------------------------- routes

router = APIRouter(prefix="/v1/maca", tags=["MACA plugin"])


def _error(code: int, error: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": {"error": error, "message": message, **extra}})


async def _read_body(request: Request) -> "tuple[Optional[str], Any]":
    """The use-case text and the ``user_relationship`` the developer sent with it (None when absent)."""
    ctype = request.headers.get("content-type", "")
    asked: Any = None
    try:
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            asked = form.get("user_relationship")
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                return None, asked
            data = await upload.read()
        else:
            body = await request.json()
            asked = body.get("user_relationship") if isinstance(body, dict) else None
            value = body.get("usecase_markdown") if isinstance(body, dict) else None
            data = value.encode("utf-8") if isinstance(value, str) else None
    except (ValueError, UnicodeError):
        return None, None
    if data is None:
        return None, asked
    if len(data) > MAX_MARKDOWN_BYTES:
        raise OverflowError
    try:
        return data.decode("utf-8"), asked
    except UnicodeDecodeError:
        return None, asked


# 200, not 202: the Python SDK transport takes only 200 and 204, and it is shared with Cloud, which must not change.
@router.post("/generate")
async def generate(request: Request, auth: SDKAuthContext = Depends(get_sdk_auth)):
    """Send a use-case file to Synap Cloud and have MACA write a memory config from it."""
    if _key() is None:
        return _error(400, "maca_not_configured",
                      f"The MACA plugin is not configured: set {KEY_ENV} to a Synap Cloud management key. {DOCS_HINT}")
    try:
        markdown, asked = await _read_body(request)
    except OverflowError:
        return _error(413, "usecase_too_large", f"The use-case file is larger than {MAX_MARKDOWN_BYTES // 1024} KB.")
    if not markdown or not markdown.strip():
        return _error(422, "usecase_missing",
                      "Send the use-case file as a UTF-8 markdown 'file' upload, or as JSON {\"usecase_markdown\": \"...\"}.")
    relationship = _choose_relationship(asked)
    if isinstance(relationship, JSONResponse):
        return relationship
    problem = _folder_problem()  # before the run starts: it takes 15 to 20 minutes and costs credits
    if problem:
        return _error(500, "maca_config_folder_not_writable", problem)
    _load_jobs()
    running = _running_job()
    if running is not None:
        if running.job_id not in _tasks and running.build_id:
            _start(running, None)  # left over from before a restart: resume it
        return _error(409, "maca_job_running", "A MACA job is already running. Wait for it to finish.", job_id=running.job_id)
    job = Job(job_id="maca_" + secrets.token_hex(8), created_at=_now(), updated_at=_now(),
              message="Sending the use-case file to Synap Cloud.", user_relationship=relationship)
    _jobs[job.job_id] = job
    _save(job)
    _start(job, markdown, relationship)
    return {"job_id": job.job_id, "status": job.status, "message": job.message}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, auth: SDKAuthContext = Depends(get_sdk_auth)):
    _load_jobs()
    job = _jobs.get(job_id)
    if job is None:
        return _error(404, "maca_job_not_found", "No MACA job has that id.")
    if job.status in RUNNING and job.build_id and job_id not in _tasks and _key():
        _start(job, None)  # the server restarted while Cloud was still building: pick it up again
    return _public(job)
