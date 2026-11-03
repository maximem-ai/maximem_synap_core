"""The prompt switch: one neutral interface per prompt, a Cloud file and a Core file behind it.

Every prompt module the engine imports (for example
``synap.cloud.agents.ingestion.prompts.knowledge_update``) is a two-line
*interface* file that calls ``bind(__name__)``. The text lives in two sibling
folders next to it::

    prompts/knowledge_update.py          the interface (ships)
    prompts/cloud/knowledge_update.py    the Cloud text (never ships)
    prompts/core/knowledge_update.py     the Core text (ships)

``bind`` asks the edition module (``edition.prompt_pack``) which pack to use and
puts the chosen module in ``sys.modules`` under the interface's name. So
``from ...prompts.knowledge_update import CLASSIFICATION_SYSTEM_PROMPT`` gets the
very same object whichever edition runs, and no call site chooses.

* **Cloud** imports the Cloud file. Nothing else changes: same text, same classes.
* **Core** imports the Core file and never the Cloud one. If the Core file does
  not exist, the interface becomes a *missing-prompt module*. For a prompt Core
  owes, asking it for any name raises ``CorePromptMissing`` with a clear message,
  so the server stops instead of running with an empty prompt.
  ``check_core_pack`` lists every missing Core prompt at once, for a startup
  check. For a Cloud-only prompt (a feature Core does not have), importing a name
  works, so shipped files that import it at load time still load, and any use of
  the value raises the same error.

Which prompts Core needs, and which are Cloud only, is the table ``INTERFACES``
below. The test ``synap/tests/core_prompts/test_switch_core.py`` keeps it equal to the interface
files on disk.

The Core prompt pack (``synap/cloud/core_prompts/<folder>``)
holds plain text files with ``[[NAME]]`` fields. A Core module reads them with
``read_pack``; the table names the folder each interface expects.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from synap.cloud.shared import edition

CLOUD = "cloud"
CORE = "core"

# "core": Core owes a text for this prompt.   "cloud_only": Core never has it.
CORE_STATUS = "core"
CLOUD_ONLY = "cloud_only"

_ING = "synap.cloud.agents.ingestion.prompts"
_EXT = _ING + ".extraction"
_RES = _ING + ".resolution"
_CAPI = "synap.cloud.application.client_api.prompts"
_FORGET = "synap.cloud.application.memory_forget.prompts"
_APP_ING = "synap.cloud.application.ingestion.prompts"
_V3 = "synap.cloud.application.compaction.v3.prompts"

# interface module -> (status, folder of the Core prompt pack or None)
INTERFACES: Dict[str, Tuple[str, Optional[str]]] = {
    # --- agents/ingestion/prompts ---
    f"{_ING}.prompt_resolver": (CORE_STATUS, None),
    f"{_ING}.coalesce": (CORE_STATUS, "coalesce_merge"),
    f"{_ING}.merge": (CORE_STATUS, "coalesce_merge"),
    f"{_ING}.knowledge_update": (CORE_STATUS, "knowledge_update"),
    f"{_ING}.user_profile": (CORE_STATUS, "user_profile"),
    # INFERRED: the next three are marked Core-owed by reasoning.
    # The owner to confirm: temporal_addendum (the date block added to extraction),
    # stage3_extraction (blocks stage 3 appends to the extraction prompts), prior_context
    # (the header above earlier memories).
    f"{_ING}.stage3_extraction": (CORE_STATUS, "extraction_blocks"),  # INFERRED
    f"{_ING}.prior_context": (CORE_STATUS, "extraction_blocks"),  # INFERRED
    f"{_ING}.date_prepass": (CLOUD_ONLY, None),
    f"{_ING}.earlier_reference": (CLOUD_ONLY, None),
    f"{_ING}.edge_identity": (CLOUD_ONLY, None),
    f"{_ING}.graph_v2_judge": (CLOUD_ONLY, None),
    f"{_ING}.hqe": (CLOUD_ONLY, None),
    f"{_ING}.relatedness": (CLOUD_ONLY, None),
    f"{_ING}.sweep_repair": (CLOUD_ONLY, None),
    f"{_ING}.temporal_gate": (CLOUD_ONLY, None),
    # --- agents/ingestion/prompts/extraction ---
    f"{_EXT}.merged_simple": (CORE_STATUS, "merged_extraction"),
    f"{_EXT}.graph_v2": (CORE_STATUS, "graph_extraction"),
    f"{_EXT}.temporal": (CORE_STATUS, "person_and_company"),
    f"{_EXT}.temporal_addendum": (CORE_STATUS, "date_guidance"),  # INFERRED, see the note above
    f"{_EXT}.episodes": (CORE_STATUS, "episodes_and_temporal"),
    f"{_EXT}.assistant_recall": (CLOUD_ONLY, None),
    f"{_EXT}.emotions": (CLOUD_ONLY, None),
    f"{_EXT}.entities": (CLOUD_ONLY, None),
    f"{_EXT}.facts": (CLOUD_ONLY, None),
    f"{_EXT}.merged_graph": (CLOUD_ONLY, None),
    f"{_EXT}.preferences": (CLOUD_ONLY, None),
    f"{_EXT}.relationships": (CLOUD_ONLY, None),
    # --- agents/ingestion/prompts/resolution (all Cloud only) ---
    f"{_RES}.conflict_check": (CLOUD_ONLY, None),
    f"{_RES}.jev_adjudicator": (CLOUD_ONLY, None),
    f"{_RES}.local_judge": (CLOUD_ONLY, None),
    f"{_RES}.profiles": (CLOUD_ONLY, None),
    f"{_RES}.second_look": (CLOUD_ONLY, None),
    f"{_RES}.world_lookup": (CLOUD_ONLY, None),
    # --- application/client_api/prompts (Core retrieval has no LLM) ---
    f"{_CAPI}.jev_rerank": (CLOUD_ONLY, None),
    f"{_CAPI}.relevance_filter": (CLOUD_ONLY, None),
    f"{_CAPI}.subqueries": (CLOUD_ONLY, None),
    f"{_CAPI}.temporal_question": (CLOUD_ONLY, None),
    # --- application/memory_forget/prompts ---
    f"{_FORGET}.judge": (CORE_STATUS, "forget_judge"),
    # --- application/ingestion/prompts ---
    f"{_APP_ING}.image_description": (CORE_STATUS, "image_description"),
    # --- application/compaction/v3/prompts ---
    f"{_V3}.writer": (CORE_STATUS, "compaction_v3"),
    f"{_V3}.checks": (CORE_STATUS, "compaction_v3"),
    f"{_V3}.framing": (CORE_STATUS, "compaction_v3"),
    f"{_V3}.retry": (CORE_STATUS, "compaction_v3"),
    f"{_V3}.statuses": (CLOUD_ONLY, None),
    f"{_V3}.steering": (CLOUD_ONLY, None),
}


class CorePromptMissing(RuntimeError):
    """Core was asked for a prompt that has no Core text."""


def core_pack_dir() -> Path:
    """The folder of Core's prompt pack."""
    return Path(__file__).resolve().parents[1] / "core_prompts"


def cloud_name(interface: str) -> str:
    parent, _, leaf = interface.rpartition(".")
    return f"{parent}.cloud.{leaf}"


def core_name(interface: str) -> str:
    parent, _, leaf = interface.rpartition(".")
    return f"{parent}.core.{leaf}"


def _missing_message(interface: str) -> str:
    status, pack = INTERFACES.get(interface, (CORE_STATUS, None))
    if status == CLOUD_ONLY:
        return (
            f"The prompt {interface} belongs to a feature Synap Core does not have "
            "Switch that feature off, or run the Cloud edition."
        )
    where = f" and no usable text in synap/cloud/core_prompts/{pack}" if pack else ""
    return (
        f"Synap Core has no text for the prompt {interface}: there is no file "
        f"{core_name(interface).replace('.', '/')}.py{where}. Core stops here "
        "instead of running with an empty prompt."
    )


class _Unavailable:
    """What a name from a Cloud-only prompt is in Core.

    A shipped file may import a name from a prompt Core does not have
    (``from ...date_prepass import SYSTEM``) at load time, for a feature that is
    off in Core. The import works, and every use of the value stops with
    ``CorePromptMissing`` and a clear message, so the text can never reach a model
    as an empty prompt. That covers calling it, formatting it, joining it to a
    string (either side), comparing it, hashing it, iterating it and asking for
    an attribute. ``json.dumps`` raises ``TypeError`` because the value is not a
    string; its text names the prompt, because each name gets a class of its own
    called ``CorePromptMissing_<module>__<name>``.
    """

    __slots__ = ("_what",)

    def __init__(self, what: str):
        object.__setattr__(self, "_what", what)

    def _stop(self, *_a, **_k):
        raise CorePromptMissing(self._what)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._stop()

    __call__ = __str__ = __format__ = __bool__ = __len__ = __iter__ = __next__ = _stop
    __getitem__ = __contains__ = __add__ = __radd__ = __mod__ = __rmod__ = _stop
    __mul__ = __rmul__ = __fspath__ = __bytes__ = __int__ = __float__ = __index__ = _stop
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __hash__ = _stop
    __reversed__ = __enter__ = __setitem__ = __delitem__ = _stop

    def __repr__(self) -> str:
        return f"<Cloud-only prompt text, not available in Core: {self._what}>"


def _unavailable(interface: str, name: str, message: str) -> "_Unavailable":
    leaf = re.sub(r"\W", "_", interface.rsplit(".", 1)[-1])
    safe = re.sub(r"\W", "_", name)
    cls = type(f"CorePromptMissing_{leaf}__{safe}", (_Unavailable,), {"__slots__": ()})
    return cls(message)


class _MissingPrompt(types.ModuleType):
    """Stands in for a prompt module that Core does not have.

    Names that are logic, not prompt text, come from the shipped helper module
    ``<package>/helpers/<module>.py`` when there is one (flags, predicates, merges).
    Any other name: for a prompt Core owes (status ``core``) it raises
    ``CorePromptMissing`` at once, so the server stops instead of running with an
    empty prompt. For a Cloud-only prompt it gives an ``_Unavailable`` value that
    raises when used, so a shipped file can still import it for a feature that is off.
    """

    def __init__(self, interface: str, helpers: Optional[types.ModuleType] = None):
        super().__init__(interface)
        self.__dict__["_interface"] = interface
        self.__dict__["_helpers"] = helpers

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        helpers = self.__dict__["_helpers"]
        if helpers is not None and hasattr(helpers, name):
            return getattr(helpers, name)
        interface = self.__dict__["_interface"]
        message = _missing_message(interface)
        if INTERFACES.get(interface, (CORE_STATUS, None))[0] == CLOUD_ONLY:
            return _unavailable(interface, name, message)
        raise CorePromptMissing(message)


def is_missing(module: object) -> bool:
    return isinstance(module, _MissingPrompt)


def helpers_name(interface: str) -> str:
    parent, _, leaf = interface.rpartition(".")
    return f"{parent}.helpers.{leaf}"


def _helpers_for(interface: str) -> Optional[types.ModuleType]:
    """The shipped module with the logic that used to sit in this prompt module, if any."""
    wanted = helpers_name(interface)
    try:
        return importlib.import_module(wanted)
    except ModuleNotFoundError as exc:
        if exc.name and (wanted == exc.name or wanted.startswith(exc.name + ".")):
            return None
        raise


def bind(interface: str) -> None:
    """Called by an interface module with its own ``__name__``.

    Replaces the interface module in ``sys.modules`` with the module of the
    pack the edition module names. Cloud imports the Cloud file; Core imports
    the Core file only.
    """
    if interface not in INTERFACES:
        raise KeyError(
            f"{interface} is not in prompt_switch.INTERFACES; add it there with its Core status."
        )
    if edition.prompt_pack == CLOUD:
        chosen = importlib.import_module(cloud_name(interface))
    elif edition.prompt_pack == CORE:
        wanted = core_name(interface)
        try:
            chosen = importlib.import_module(wanted)
        except ModuleNotFoundError as exc:
            # Only "the Core file is not there" is a missing prompt. A Core file
            # that is there but fails to import must show its own error.
            if not (exc.name and (wanted == exc.name or wanted.startswith(exc.name + "."))):
                raise
            chosen = _MissingPrompt(interface, _helpers_for(interface))
    else:  # edition.py already refuses other values; this is a second lock
        raise ValueError(f"unknown prompt pack {edition.prompt_pack!r}")
    sys.modules[interface] = chosen


def check_core_pack() -> List[str]:
    """Messages for every prompt Core owes and does not have. Empty means whole.

    Meant for a startup check in Core. Imports each interface
    marked ``core`` and reports the ones that are missing-prompt modules. A
    missing text can also stop the import of a package that uses it; that
    message is reported too, once.
    """
    problems: List[str] = []

    def add(message: str) -> None:
        if message not in problems:
            problems.append(message)

    for interface, (status, _pack) in INTERFACES.items():
        if status != CORE_STATUS:
            continue
        try:
            module = importlib.import_module(interface)
        except CorePromptMissing as exc:
            add(str(exc))
            continue
        if is_missing(module):
            add(_missing_message(interface))
    return problems


# --- reading the Core prompt pack (layout in synap/cloud/core_prompts/README.md) ---

_FIELD = re.compile(r"\[\[([A-Z_]+)\]\]")
_BLOCK = re.compile(r"\[\[IF ([A-Z_]+)\]\](.*?)\[\[END IF\]\]", re.DOTALL)


def render(text: str, **fields: Optional[str]) -> str:
    """Fill ``[[NAME]]`` fields; keep ``[[IF X]]...[[END IF]]`` only when X is given.

    A field with no value, or None, counts as not given. A ``[[NAME]]`` left
    without a value outside a dropped block raises, so a prompt never goes out
    with a hole in it.
    """
    given = {k.upper(): v for k, v in fields.items() if v not in (None, "")}

    def _block(match: "re.Match[str]") -> str:
        return match.group(2) if match.group(1) in given else ""

    text = _BLOCK.sub(_block, text)

    def _field(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in given:
            raise CorePromptMissing(f"The Core prompt needs the field [[{name}]] and was not given it.")
        return str(given[name])

    return _FIELD.sub(_field, text)


def read_pack(folder: str, filename: str, **fields: Optional[str]) -> str:
    """Read ``core_prompts/<folder>/<filename>`` and fill its fields.

    Raises ``CorePromptMissing`` when the file is not there, so a Core module
    built on it fails with a clear message.
    """
    path = core_pack_dir() / folder / filename
    if not path.is_file():
        raise CorePromptMissing(
            f"The Core prompt file synap/cloud/core_prompts/{folder}/{filename} does not exist. "
            "Core stops here instead of running with an empty prompt."
        )
    return render(path.read_text(encoding="utf-8"), **fields)


def pack_meta(folder: str) -> dict:
    return json.loads((core_pack_dir() / folder / "meta.json").read_text(encoding="utf-8"))


def pack_text(folder: str, filename: str) -> str:
    """The text of ``core_prompts/<folder>/<filename>`` exactly as it is on disk.

    For a text whose ``[[NAME]]`` fields are filled later, once the call has its values
    (see ``fill`` and ``PackTemplate``). Raises ``CorePromptMissing`` when the file is not there.
    """
    path = core_pack_dir() / folder / filename
    if not path.is_file():
        raise CorePromptMissing(
            f"The Core prompt file synap/cloud/core_prompts/{folder}/{filename} does not exist. "
            "Core stops here instead of running with an empty prompt."
        )
    return path.read_text(encoding="utf-8")


_TAG = re.compile(r"\[\[(IF [A-Z_]+|END IF)\]\]")


def fill(text: str, **fields: object) -> str:
    """Fill ``[[NAME]]`` fields. Unlike ``render``, an empty string is a value.

    ``[[IF X]]...[[END IF]]`` is kept when X is given and not empty (or not False). X must be a
    name the caller passed, even as None, so a misspelt tag is an error and not a silently dropped
    block. Everything wrong raises ``CorePromptMissing``, so a prompt never goes out with a hole,
    a stray tag or a dropped block it should have had: an unknown tag, an unclosed or stray
    ``[[END IF]]``, a block inside a block, and a field with no value.
    """
    given = {k.upper(): v for k, v in fields.items()}
    out, depth_tag, pos, keep_from = [], None, 0, 0
    for match in _TAG.finditer(text):
        token = match.group(1)
        if token.startswith("IF "):
            if depth_tag is not None:
                raise CorePromptMissing(f"The Core prompt has a block [[{token}]] inside [[IF {depth_tag}]]; blocks do not nest.")
            name = token[3:]
            if name not in given:
                raise CorePromptMissing(f"The Core prompt has [[IF {name}]] and the caller gave no {name}, not even None.")
            out.append(text[pos:match.start()])
            depth_tag, pos, keep_from = name, match.end(), match.end()
        else:
            if depth_tag is None:
                raise CorePromptMissing("The Core prompt has an [[END IF]] with no [[IF ...]] before it.")
            if given[depth_tag] not in (None, "", False):
                out.append(text[keep_from:match.start()])
            depth_tag, pos = None, match.end()
    if depth_tag is not None:
        raise CorePromptMissing(f"The Core prompt has [[IF {depth_tag}]] with no [[END IF]].")
    out.append(text[pos:])
    text = "".join(out)

    def _field(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in given or given[name] is None:
            raise CorePromptMissing(f"The Core prompt needs the field [[{name}]] and was not given it.")
        return str(given[name])

    return _FIELD.sub(_field, text)


class PackTemplate(str):
    """A text from the pack that answers ``.format(...)`` the way a Cloud template does.

    The engine calls ``WRITER_USER.format(record=..., turns=...)``. Here the value of
    ``record`` goes into ``[[RECORD]]``. A float is written with one decimal. A name the
    text does not use is ignored; a field the call does not give raises.
    """

    __slots__ = ()

    def format(self, *args: object, **fields: object) -> str:  # type: ignore[override]
        shown = {k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in fields.items()}
        return fill(str(self), **shown)


def core_gap(interface: str, name: str) -> "_Unavailable":
    """The value of a name whose Core text does not exist because its feature is off in Core.

    Importing it works, so a shipped file that imports it at load time still loads. Every
    use of the value raises ``CorePromptMissing`` and names the prompt.
    """
    return _unavailable(
        interface,
        name,
        f"{interface}.{name} is not in Synap Core; its feature is off in Core.",
    )

