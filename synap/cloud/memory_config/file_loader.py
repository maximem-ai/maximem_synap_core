"""The memory config read from a file.

``FileConfigLoader`` has the same interface as ``_DBConfigLoader`` in
``application/ingestion/startup.py`` (``await loader.get_config(instance_id)``),
so every reader that takes a loader takes this one. It reads one YAML file for
the whole install; the instance id is ignored.

* The path is ``SYNAP_MEMORY_CONFIG``, default ``config/memory.yml`` (the
  mounted folder is ``/app/config`` in the image, and ``/app`` is the working
  directory).
* The file is checked on every call with one ``stat`` and read again only when
  its size or modification time changed.
* A file that fails validation when the server starts stops it: call
  ``check_memory_config_at_startup`` and let ``MemoryConfigError`` through. A
  file that fails on a later reload keeps the last good config and logs the
  error once.
* With no file the loader hands back the built-in defaults and says so in the
  log, once.

Two views of the same file:

* ``get_config`` returns the validated ``MemoryArchitectureConfig``. The schema
  fills in its defaults there.
* ``get_raw_config`` returns what the developer wrote (plus an empty
  ``metadata`` when missing), for readers that apply their own defaults.
  Retrieval is one: with no ``ranking`` block it falls back to relevance 0.70,
  importance 0.20, recency 0.10. Handing it the validated copy would give the
  schema's 0.5, 0.3, 0.2.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import core_rules
from .config_schema import ConfigMetadata, EvolutionConfig, MemoryArchitectureConfig, ScopingConfig

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV = "SYNAP_MEMORY_CONFIG"
DEFAULT_CONFIG_PATH = "config/memory.yml"
#: The version of the config schema this loader accepts: the schema's own version. The MACA plugin sends it to Cloud.
CONFIG_SCHEMA_VERSION: str = EvolutionConfig.model_fields["schema_version"].default


class MemoryConfigError(RuntimeError):
    """The config file is not acceptable. The message names the file, the lines and the reasons."""


def _default_config() -> MemoryArchitectureConfig:
    return MemoryArchitectureConfig(
        metadata=ConfigMetadata(
            client_id="default",
            instance_id="default",
            config_version=CONFIG_SCHEMA_VERSION,
            generated_at=datetime.now(timezone.utc),
            generated_by="core-built-in-defaults",
        )
    )


def _line_index(text: str) -> Dict[Tuple[Any, ...], int]:
    """Path (keys and list positions) -> 1-based line of that key, from the YAML node tree."""
    index: Dict[Tuple[Any, ...], int] = {}

    def walk(node: Any, path: Tuple[Any, ...]) -> None:
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                key = getattr(key_node, "value", None)
                child = path + (key,)
                index[child] = key_node.start_mark.line + 1
                walk(value_node, child)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                child = path + (i,)
                index[child] = item.start_mark.line + 1
                walk(item, child)

    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return index
    if root is not None:
        walk(root, ())
    return index


def _line_for(index: Dict[Tuple[Any, ...], int], path: Tuple[Any, ...]) -> Optional[int]:
    for end in range(len(path), 0, -1):
        if path[:end] in index:
            return index[path[:end]]
    return None


def parse_and_validate(
    text: str, source: str = DEFAULT_CONFIG_PATH
) -> Tuple[Dict[str, Any], MemoryArchitectureConfig]:
    """Parse and check a config. Returns (what the developer wrote, the validated config).

    Raises ``MemoryConfigError`` listing every problem found, each with its line.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" line {mark.line + 1}" if mark is not None else ""
        reason = getattr(exc, "problem", None) or str(exc)
        raise MemoryConfigError(f"{source},{where}: not valid YAML: {reason}") from None
    if raw is None:
        raw = {}

    index = _line_index(text)
    problems: List[str] = []

    def add(path: Tuple[Any, ...], message: str) -> None:
        line = _line_for(index, path)
        problems.append(f"{source}, line {line}: {message}" if line else f"{source}: {message}")

    for path, message in core_rules.find_problems(raw):
        add(path, message)
    if problems:
        raise MemoryConfigError("\n".join(problems))

    if raw.get("metadata") is None:
        # A hand-written file has no metadata; the schema requires the field.
        raw["metadata"] = {}
    try:
        config = MemoryArchitectureConfig(**copy.deepcopy(raw))
    except Exception as exc:  # pydantic ValidationError
        errors = getattr(exc, "errors", None)
        if errors is None:
            raise MemoryConfigError(f"{source}: {exc}") from None
        for err in errors():
            loc = tuple(err.get("loc", ()))
            add(loc, f"{'.'.join(str(p) for p in loc) or '(top level)'}: {err.get('msg')}")
        raise MemoryConfigError("\n".join(problems)) from None
    return raw, config


# The scoping mode this install was seeded with (Core). Set once at startup by ``core_seed`` after it has
# compared the YAML with the instance. A reload that names another mode is refused, because the mode cannot
# change once memories exist; editing it live used to flip the whole install half way.
_pinned_scoping_mode: Optional[str] = None


def pin_scoping_mode(mode: Optional[str]) -> None:
    global _pinned_scoping_mode
    _pinned_scoping_mode = mode


def pinned_scoping_mode() -> Optional[str]:
    """The scoping mode this install holds memories under, once seeding has pinned it."""
    return _pinned_scoping_mode


def _scoping_mode_of(raw: Dict[str, Any]) -> Optional[str]:
    value = (((raw or {}).get("storage_architecture") or {}).get("scoping") or {}).get("user_context_isolation")
    return value if isinstance(value, str) and value else None


class FileConfigLoader:
    """Reads the memory config from one YAML file; same interface as ``_DBConfigLoader``."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()
        self._signature: Optional[Tuple[int, int]] = None   # of the file behind _raw/_config
        self._raw: Dict[str, Any] = {"metadata": {}}
        self._config: Optional[MemoryArchitectureConfig] = None
        self._loaded_once = False
        self._failed_signature: Optional[Tuple[int, int]] = None
        self._said_no_file = False

    @property
    def path(self) -> str:
        return self._path

    def _stat(self) -> Optional[Tuple[int, int]]:
        try:
            st = os.stat(self._path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def refresh(self, *, strict: bool = False) -> None:
        """Read the file again if it changed.

        ``strict`` (startup): a bad file raises ``MemoryConfigError``. Otherwise
        (a reload) a bad file keeps the last good config and logs the error once.
        """
        with self._lock:
            sig = self._stat()
            if sig is None:
                self._file_missing()
                return
            if sig == self._signature or (not strict and sig == self._failed_signature):
                self._loaded_once = True
                return
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                raw, config = parse_and_validate(text, self._path)
            except (MemoryConfigError, OSError) as exc:
                if strict or self._signature is None:
                    # Startup, or no good config to fall back on: stop.
                    raise MemoryConfigError(str(exc)) from None
                self._failed_signature = sig
                logger.error(
                    "Memory config file %s failed validation on reload; keeping the last good config:\n%s",
                    self._path, exc,
                )
                return
            # A file that leaves the mode out runs under the schema's default, so it counts as naming it.
            wanted = _scoping_mode_of(raw) or ScopingConfig.model_fields["user_context_isolation"].default
            if not strict and _pinned_scoping_mode and wanted and wanted != _pinned_scoping_mode:
                self._failed_signature = sig
                logger.error(
                    "Memory config file %s runs under user_context_isolation %s, but this install was set up with %s. "
                    "Changing the scoping mode after memories exist is not supported, so the change is ignored and "
                    "the last good config is kept. Set it back to %s, or start from nothing.",
                    self._path, wanted, _pinned_scoping_mode, _pinned_scoping_mode,
                )
                return
            self._raw, self._config, self._signature = raw, config, sig
            self._failed_signature = None
            self._loaded_once = True
            logger.info("Memory config loaded from %s", self._path)

    def _file_missing(self) -> None:
        if self._signature is not None:
            # It was there and good, and now it is gone: keep what we had.
            if self._failed_signature != (0, 0):
                self._failed_signature = (0, 0)
                logger.error("Memory config file %s is gone; keeping the last good config.", self._path)
        elif not self._said_no_file:
            self._said_no_file = True
            logger.warning(
                "No memory config file at %s: starting on the built-in defaults "
                "(ranking weights relevance 0.70, importance 0.20, recency 0.10).",
                self._path,
            )
        if self._config is None:
            self._config = _default_config()
        self._loaded_once = True

    async def get_config(self, instance_id: Optional[str] = None) -> MemoryArchitectureConfig:
        """The validated config (schema defaults filled in). The instance id is ignored."""
        self.refresh()
        assert self._config is not None
        return self._config

    async def get_raw_config(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        """The config as the developer wrote it (a copy), with an empty ``metadata`` when missing.

        With no file this is ``{"metadata": {}}``: not empty, so a reader that
        treats an empty config as "no config" still gets a config and applies
        its own defaults.
        """
        self.refresh()
        return copy.deepcopy(self._raw)


_shared: Optional[FileConfigLoader] = None
_shared_lock = threading.Lock()


def shared_loader() -> FileConfigLoader:
    """One loader per process, so a reload is seen by every reader."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = FileConfigLoader()
        return _shared


def check_memory_config_at_startup(loader: Optional[FileConfigLoader] = None) -> None:
    """Read the file once, strictly. Raises ``MemoryConfigError`` so the server does not start on a bad file."""
    (loader or shared_loader()).refresh(strict=True)
