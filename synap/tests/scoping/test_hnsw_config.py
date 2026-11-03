"""R-13 / SHIP-09. The one HNSW parameter we set on purpose, and its evidence.

A constant with a measured value behind it is only useful while the measurement
is still attached to it. These pin both: that the value is actually sent to
Chroma at creation, and that the curve justifying it is still written down next
to it. A number nobody can trace back to a measurement is a number the next
person will change on a hunch.
"""
from __future__ import annotations

import ast
from pathlib import Path

from synap.providers.vector.adapters.chromadb import _HNSW_SEARCH_EF

_ADAPTER = (
    Path(__file__).resolve().parents[2] / "providers" / "vector" / "adapters" / "chromadb.py"
)


def _create_namespace_metadata() -> dict[str, object]:
    """The metadata dict handed to get_or_create_collection, keys AND values.

    ⚠ This used to return keys only, so one test asserted the key exists and
    another asserted the constant is 200, with nothing joining them. Changing
    the literal to `"hnsw:search_ef": 16` while leaving the constant at 200
    passed all four tests, and every new collection would have been created at
    recall 0.775 by this file's own measured curve. Found by audit 2026-08-23.

    An `ast.Name` value is resolved against the module, which is the whole
    point: the test has to follow the reference rather than stop at it.
    """
    import synap.providers.vector.adapters.chromadb as adapter

    tree = ast.parse(_ADAPTER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "create_namespace"
    )
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "get_or_create_collection"):
            for kw in node.keywords:
                if kw.arg == "metadata" and isinstance(kw.value, ast.Dict):
                    out: dict[str, object] = {}
                    for k, v in zip(kw.value.keys, kw.value.values):
                        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                            continue
                        if isinstance(v, ast.Constant):
                            out[k.value] = v.value
                        elif isinstance(v, ast.Name):
                            out[k.value] = getattr(adapter, v.id, f"<unresolved {v.id}>")
                        else:
                            out[k.value] = "<expression>"
                    return out
    raise AssertionError("create_namespace no longer passes a metadata dict literal")


def _create_namespace_metadata_keys() -> set[str]:
    return set(_create_namespace_metadata())


def test_new_collections_are_created_with_an_explicit_search_ef():
    """The whole of R-13: every collection this system has ever created set
    hnsw:space and nothing else, so search_ef sat at whatever the default was
    and nobody had measured what that meant."""
    keys = _create_namespace_metadata_keys()
    assert "hnsw:search_ef" in keys, (
        f"create_namespace sends {sorted(keys)}; search_ef is unset again and "
        "new collections are back on an untested default"
    )


def test_the_distance_function_is_still_set_alongside_it():
    """Adding search_ef must not displace hnsw:space. modify() erases it, so
    creation is the only place it reliably gets recorded."""
    assert "hnsw:space" in _create_namespace_metadata_keys()


def test_the_value_actually_sent_is_the_one_the_curve_supports():
    """The join the old tests were missing: what reaches Chroma, not what the
    constant says. These were two separate assertions with nothing between
    them, so the literal could be changed while the constant stayed."""
    sent = _create_namespace_metadata().get("hnsw:search_ef")
    assert sent == _HNSW_SEARCH_EF, (
        f"create_namespace sends {sent!r} while the constant is {_HNSW_SEARCH_EF}; "
        "the value in the dict is what collections are actually built with"
    )
    assert sent == 200, (
        f"search_ef is {sent!r}. Anything under 100 measured badly (0.936 at 50, "
        "0.775 at 10) and the curve is in create_namespace's comment."
    )


def test_the_value_is_the_one_the_curve_supports():
    """200 is where measured recall reached 1.000. Below 100 it falls off a
    cliff: 0.936 at 50 and 0.775 at 10."""
    assert _HNSW_SEARCH_EF == 200, (
        f"search_ef is {_HNSW_SEARCH_EF}. Anything under 100 measured badly "
        "(0.936 at 50, 0.775 at 10) and the comment in create_namespace has the "
        "curve. If this changed on purpose, update the curve with it."
    )


def test_the_measurement_is_recorded_where_the_value_is():
    """A constant whose evidence lives only in a commit message is a constant
    the next person changes on a hunch."""
    src = _ADAPTER.read_text(encoding="utf-8")
    for marker in ("recall 0.775", "recall 0.993", "recall 1.000", "modify()"):
        assert marker in src, f"the evidence for search_ef no longer mentions {marker!r}"
