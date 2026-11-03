"""Name the rung a prefetched bundle belongs to, in the caller's vocabulary.

WHY THIS EXISTS
---------------
Everything else in the scoping feature identifies a rung by its registry node
id, and that is right for the store: the filter is `node_id IN chain` and the
ids are what records carry. A pushed bundle cannot use it. The bundle crosses
the gRPC stream into an SDK running in the client's process, and that SDK has
never seen a node id in its life. What it has is the `scope` path the caller
handed to `fetch`, which is level keys and the client's own external ids.

So a bundle is stamped with the rung written the way the caller writes one, and
the SDK compares that stamp against the path its own request names. Two strings,
compared for equality, in three languages. Nothing needs to resolve anything.

WHY EQUALITY AND NOT A PREFIX OR A SUBSET
-----------------------------------------
A subset rule reads as the generous one and is the dangerous one. If a bundle
prefetched at `customer=acme|team=payments|user=dana` were servable to a request
that named only `customer=acme|user=dana`, the SDK would be deciding that the
unnamed rung does not matter. It is precisely the rung that distinguishes two
people who share an external id, which is the case the whole ladder exists for.
Equality costs a cold fetch when it is wrong. A subset costs one person's
context handed to another.

WHAT AN ABSENT RUNG MEANS
-------------------------
`None`, on either side, means "this request or bundle named no rung". That is
every request on the default configuration: clients are on the default three
rungs, the deployment flag is off, and only a few instances have it pinned on. So the
comparison rule has to leave that case exactly as it is, and it does:

    bundle None  + request None  -> serve      (today, unchanged)
    bundle None  + request named -> refuse     (the caller narrowed, the bundle did not)
    bundle named + request None  -> refuse     (the bundle is narrower than the caller)
    bundle named + request named -> serve iff the two strings are equal

The first line is the whole fleet. The other three cannot fire until a client
turns nested scoping on AND uses the streamed prefetch, and no client does both
today (the instances with nested
scoping pinned on had no anticipation turns between them when this was checked).

THE ENCODING
------------
`key=value`, sorted by key, joined with `|`. Sorted because the SDK receives a
dict and a dict has whatever order the caller wrote it in; ladder order is not
available to the SDK at all. Level keys are `^[a-z][a-z0-9_]{0,63}$`, enforced
by `ladder_validation`, so a key can hold neither separator and only the value
needs escaping. Values are the client's own external ids and can hold anything.

Kept deliberately dull, because it is written three times: here, in
`maximem_synap.cache.anticipation_cache`, and in the JS SDK's
`anticipation-cache.ts`. The golden vectors in
`synap/sdk/CONTRACT/behavior/anticipation.json` are what stops the three drifting.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

# The root rung is never named by a caller: `scope` paths start below the
# client, because the client is who you are, not somewhere you are.
ROLE_ROOT = "root"

_SEPARATOR = "|"
_ASSIGN = "="


def escape_value(value: str) -> str:
    """Make an external id safe to sit between the two separators.

    Backslash first, or the two replacements below would rewrite each other's
    output and `a|b` and `a\\pb` would encode to the same thing.
    """
    return (
        value.replace("\\", "\\\\")
        .replace(_SEPARATOR, "\\p")
        .replace(_ASSIGN, "\\e")
    )


def rung_key_from_pairs(pairs: Iterable[Tuple[str, str]]) -> Optional[str]:
    """The canonical rung key for `(level_key, external_id)` pairs.

    None when there is nothing to name, which the comparison rule treats as
    "this named no rung" rather than as an error.
    """
    cleaned = [
        (str(k), str(v))
        for k, v in pairs
        if k is not None and v is not None and str(k) != "" and str(v) != ""
    ]
    if not cleaned:
        return None
    cleaned.sort(key=lambda kv: kv[0])
    return _SEPARATOR.join(f"{k}{_ASSIGN}{escape_value(v)}" for k, v in cleaned)


def rung_key_from_path(scope_path: Optional[Mapping[str, str]]) -> Optional[str]:
    """The rung a caller named with a `scope` path.

    The read side of the comparison. The SDKs compute this; it lives here too
    so the server's tests can assert both sides against one implementation
    rather than against a second copy written to agree with the first.
    """
    if not scope_path:
        return None
    return rung_key_from_pairs(scope_path.items())


def rung_key_from_rungs(rungs: Optional[Sequence[Mapping[str, Any]]]) -> Optional[str]:
    """The rung a resolved chain sits at, or None.

    None whenever the chain cannot be written in the caller's vocabulary:
    no chain at all, or a rung with no external id. That second case is a
    fallback and not a failure. An unstamped bundle is treated as "named no
    rung", which is the state every bundle is in today, so the worst outcome is
    that a path-naming caller refuses it and pays for a fetch. Stamping a rung
    we could not actually name would be the other kind of mistake.
    """
    if not rungs:
        return None
    pairs: list[Tuple[str, str]] = []
    for rung in rungs:
        try:
            roles = rung.get("roles") or ([rung.get("role")] if rung.get("role") else [])
            if ROLE_ROOT in roles:
                # The caller's path starts below the root, so including it here
                # would make every server stamp differ from every SDK path by
                # one segment and nothing would ever match.
                continue
            key = rung.get("level_key")
            ext = rung.get("external_id")
        except AttributeError:
            return None
        if not key or not ext:
            return None
        pairs.append((str(key), str(ext)))
    return rung_key_from_pairs(pairs)


def serves(bundle_rung: Optional[str], request_rung: Optional[str]) -> bool:
    """May a bundle stamped `bundle_rung` answer a request at `request_rung`?

    The one place the rule lives, so the three call sites in this repo cannot
    each decide it slightly differently. Empty string is normalised to None:
    a proto string field has no null, so an unstamped bundle arrives as "".
    """
    return (bundle_rung or None) == (request_rung or None)
