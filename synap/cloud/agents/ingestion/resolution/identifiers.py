"""Names that carry an identifier, settled before anyone is asked.

"INV-7802", "workspace bexley-main", "Visa ending 4421": the identifier says
which thing is meant, so two such names are the same thing exactly when they
share it. Matching by spelling or meaning cannot see that: "bexley-main" reads
close to "Bexley Studios", and the judge, unsure, held it for review in every
run of the billing scenario.

``settle`` rewrites a resolution only when an identifier decides it:
  - the mention and exactly one candidate share an identifier   MATCH that one
  - every candidate is a different thing by identifier          NEW
      (both carry one and none is shared; or only one side carries one and
      the two are different kinds of thing, a workspace and an organization)
Every other case is left as it came: a name with no identifier, or a generic
word against an identifier ("the workspace" and "workspace corliss-ops"), goes
the way it went before. So the rule can only replace a hold or a guess with a
decision the identifiers make; it changes nothing it cannot decide.

``part_of_other_kind`` covers the other way a related thing is joined to
the wrong entry: "Chordline refund policy" (a policy) matched to "Chordline"
(an organization), then renamed to the fuller name, so every relationship of
the organization read as one of the policy. A name that is another name plus
more words, of a different kind, is a different thing. Kinds alone are not
enough: one company is "company" in one note and "organization" in the next.

With ``SYNAP_ER_IDENTIFIERS`` set (per instance, as the other flags).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional, Set

_FLAG_ENV = "SYNAP_ER_IDENTIFIERS"

# A token with a digit (INV-7802, 4421, TCK-5521) or a lower-case slug with a
# hyphen or underscore (bexley-main, corliss_ops).
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./]*")


def enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id)


def identifiers(names: Iterable[str]) -> Set[str]:
    out = set()
    for name in names:
        for token in _TOKEN.findall(name or ""):
            token = token.strip(".-_/")
            if any(c.isdigit() for c in token) or (
                    token.islower() and ("-" in token or "_" in token)):
                out.add(token.lower())
    return out


def _kind(value: Any) -> str:
    return str(value or "").strip().lower()


def settle(resolution: Any, names: Iterable[str], kind: str,
           entry_of: Callable[[Any], Any]) -> Optional[str]:
    """Rewrite *resolution* in place when identifiers decide it. *names*: the
    mention and its aliases; *kind*: its entity type; *entry_of*: a candidate's
    registry entry by canonical id. Returns "match" or "new", or None."""
    from synap_er.pipeline.adjudication import Decision

    if resolution.decision is Decision.MATCH and resolution.path in ("exact", "normalized"):
        return None                       # the spelling already settled it
    candidates = list(resolution.candidates or [])
    if not candidates:
        return None
    mine = identifiers(names)
    same, different = [], 0
    for candidate in candidates:
        entry = entry_of(candidate.canonical_id)
        theirs = identifiers([candidate.canonical_name, *((getattr(entry, "aliases", None) or []))])
        their_kind = _kind(((getattr(entry, "metadata_", None) or {}).get("entity_subtype")) if entry else "")
        if mine and theirs:
            if mine & theirs:
                same.append(candidate)
            else:
                different += 1
        elif (mine or theirs) and kind and their_kind and their_kind != _kind(kind):
            different += 1
    if len(same) == 1:
        resolution.decision = Decision.MATCH
        resolution.canonical_id = same[0].canonical_id
        resolution.canonical_name = same[0].canonical_name
        resolution.same_as_index = None
        resolution.path = "identifier"
        return "match"
    if not same and different == len(candidates):
        resolution.decision = Decision.NEW
        resolution.canonical_id = resolution.canonical_name = resolution.same_as_index = None
        resolution.path = "identifier"
        return "new"
    return None


def _words(name: str):
    return re.findall(r"[a-z0-9]+", (name or "").lower())


def extends(longer: str, shorter: str) -> bool:
    """True when *longer* is *shorter* with more words around it."""
    a, b = _words(longer), _words(shorter)
    if not b or len(a) <= len(b):
        return False
    return any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


def part_of_other_kind(names: Iterable[str], kind: str, entry_name: str, entry: Any) -> bool:
    """A mention that is an entry's name plus more words, of another kind
    ("Chordline refund policy", a policy, and "Chordline", an organization)."""
    their_kind = _kind(((getattr(entry, "metadata_", None) or {}).get("entity_subtype")) if entry else "")
    if not kind or not their_kind or _kind(kind) == their_kind:
        return False
    theirs = [entry_name, *(getattr(entry, "aliases", None) or [])]
    return any(extends(n, t) or extends(t, n) for n in names for t in theirs if n and t)


def keep_apart(resolution: Any, names: Iterable[str], kind: str,
               entry_of: Callable[[Any], Any]) -> Optional[str]:
    """Rewrite *resolution* when what it matched, or every candidate it was
    unsure between, is a related thing of another kind. Returns "new" or None."""
    from synap_er.pipeline.adjudication import Decision

    names = list(names)
    if resolution.decision is Decision.MATCH and resolution.canonical_id is not None:
        if not part_of_other_kind(names, kind, resolution.canonical_name, entry_of(resolution.canonical_id)):
            return None
    elif resolution.decision is Decision.AMBIGUOUS and resolution.candidates:
        if not all(part_of_other_kind(names, kind, c.canonical_name, entry_of(c.canonical_id))
                   for c in resolution.candidates):
            return None
    else:
        return None
    resolution.decision = Decision.NEW
    resolution.canonical_id = resolution.canonical_name = resolution.same_as_index = None
    resolution.path = "other_kind"
    return "new"
