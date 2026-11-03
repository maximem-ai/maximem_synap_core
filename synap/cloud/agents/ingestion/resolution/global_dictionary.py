"""The global dictionary: names generally accepted to mean the same.

"Bob" / "Robert", "Infy" / "Infosys", "BLR" / "Bengaluru". One dictionary for
every customer, so it holds general knowledge only:

* For people it stores single given-name words and nothing else. "Bob Smith"
  matched to "Robert Smith" teaches it bob = robert; no person's full name is
  ever written.
* For everything else it stores a pair only when the second name came from
  general knowledge (the extraction model's ``known_as``, a seed file, a
  reviewer) and does not appear in the note it was learned from. A name the
  note itself supplied is that customer's alias, and stays in their registry.

No row carries a client, customer or user id.

An equivalent is a reason to OFFER a candidate, never to match one: synap-er
searches the registry under the other name and the adjudicator decides.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

PERSON = "person"
OTHER = "other"

# Where a pair came from. Kept per row so a bad source can be pulled.
SOURCE_EXTRACTION = "extraction"   # the extraction model's known_as
SOURCE_JUDGE = "judge"             # a match the adjudicator confirmed
SOURCE_SECOND_LOOK = "second_look"
SOURCE_SEED = "seed"
SOURCE_HUMAN = "human"

MAX_EQUIVALENTS = 6
_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


# A form of address is not a name: "Herr Müller" is not a nickname of "Jürgen Müller".
_HONORIFICS = frozenset({
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "madam", "dame", "lord",
    "lady", "herr", "frau", "monsieur", "madame", "mme", "mlle", "señor", "senor", "señora",
    "senora", "sr", "sra", "don", "doña", "shri", "sri", "smt", "kumari", "san", "sama", "sensei",
})


def global_dictionary_enabled() -> bool:
    """SYNAP_ER_GLOBAL_DICTIONARY, on by default inside adjudicated mode."""
    return os.environ.get("SYNAP_ER_GLOBAL_DICTIONARY", "true").strip().lower() in ("true", "1", "yes")


def _key(name: str) -> str:
    return " ".join(name.replace(".", "").casefold().split())


# "Developer Experience team" and "Developer Experience" are one name to look up.
_UNIT_WORDS = frozenset({
    "the", "team", "teams", "department", "dept", "group", "squad", "guild", "function",
    "division", "unit", "org", "organisation", "organization", "office",
})


def _lookup_key(kind: str, name: str) -> str:
    """The key a name is stored and found under."""
    key = _key(name)
    if kind == PERSON:
        return key
    words = [w for w in key.split() if w not in _UNIT_WORDS]
    return " ".join(words) or key


def _kind(entity_type: Any) -> str:
    value = str(getattr(entity_type, "value", entity_type) or "").lower()
    return PERSON if value == "person" else OTHER


def given_name_pair(a: str, b: str) -> Optional[Tuple[str, str]]:
    """The one word that differs between two forms of a person's name.

    ("Bob Smith", "Robert Smith") -> ("Bob", "Robert"). None when the names
    differ in more than one word each, share nothing, differ somewhere other
    than the first word, or differ by a spelling: none of that is a nickname,
    and nothing about it is general knowledge.
    """
    words_a = [w for w in _WORD.findall(a) if w.casefold() not in _HONORIFICS]
    words_b = [w for w in _WORD.findall(b) if w.casefold() not in _HONORIFICS]
    if not words_a or not words_b:
        return None
    keys_a, keys_b = [w.casefold() for w in words_a], [w.casefold() for w in words_b]
    only_a = [w for w, k in zip(words_a, keys_a) if k not in keys_b]
    only_b = [w for w, k in zip(words_b, keys_b) if k not in keys_a]
    shared = set(keys_a) & set(keys_b)
    single = len(words_a) == 1 and len(words_b) == 1
    if len(only_a) != 1 or len(only_b) != 1 or not (shared or single):
        return None
    one, two = only_a[0], only_b[0]
    if min(len(one), len(two)) < 2:
        return None   # an initial is not a name
    if keys_a[0] != one.casefold() or keys_b[0] != two.casefold():
        return None   # not in the given-name position: a surname is not ours to keep
    if one.casefold() in two.casefold() or two.casefold() in one.casefold():
        return None   # "Rob" / "Robert": code already relates these
    if _edits(one.casefold(), two.casefold()) <= 1:
        return None   # a typo, or two different names a letter apart ("Maria" / "Mario")
    return one, two


def _edits(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


class GlobalNameDictionary:
    """Implements synap_er.pipeline.adjudication.NameEquivalents.

    Lookups are in memory. A learned pair is added here at once and handed to
    *store* (when given) to persist; other workers pick it up when they load.
    """

    def __init__(self, store: Optional[Any] = None) -> None:
        self._store = store
        self._pairs: Dict[Tuple[str, str], Dict[str, str]] = {}   # (kind, key) -> {other key: other name}
        self.learned: List[dict] = []    # this process, for logs and the harness

    def __len__(self) -> int:
        return sum(len(v) for v in self._pairs.values()) // 2

    def load(self, rows: Iterable[Tuple[str, str, str]]) -> None:
        """Rows of (kind, name, equivalent)."""
        for kind, name, equivalent in rows:
            self._put(kind, name, equivalent)

    def equivalents(self, name: str, entity_type: Any) -> List[str]:
        kind = _kind(entity_type)
        found = self._pairs.get((kind, _lookup_key(kind, name)), {})
        return list(found.values())[:MAX_EQUIVALENTS]

    async def learn(
        self,
        name: str,
        other: str,
        entity_type: Any,
        source: str,
        note: Optional[str] = None,
    ) -> bool:
        """Record that *name* and *other* mean the same in general.

        Returns True when a new pair was stored. *note* is the text the pair
        was learned from: for a non-person, an *other* found in it is refused,
        since the note and not general knowledge supplied it.
        """
        kind = _kind(entity_type)
        if kind == PERSON:
            pair = given_name_pair(name, other)
            if pair is None:
                return False
            name, other = pair
        else:
            if source in (SOURCE_JUDGE, SOURCE_SECOND_LOOK):
                # A confirmed match between two names a customer wrote. Theirs.
                return False
            if note and _key(other) in _key(note):
                return False
        if not _key(name) or _key(name) == _key(other):
            return False
        if not self._put(kind, name, other):
            return False
        self.learned.append({"kind": kind, "name": name, "equivalent": other, "source": source})
        if self._store is not None:
            try:
                await self._store.add(kind, name, other, source)
            except Exception:
                logger.warning("Could not persist name equivalent %r / %r", name, other, exc_info=True)
        return True

    def _put(self, kind: str, name: str, other: str) -> bool:
        a, b = _lookup_key(kind, name), _lookup_key(kind, other)
        if not a or not b or a == b:
            return False
        fresh = b not in self._pairs.get((kind, a), {})
        self._pairs.setdefault((kind, a), {})[b] = other
        self._pairs.setdefault((kind, b), {})[a] = name
        return fresh


class PostgresNameEquivalentStore:
    """``er_name_equivalents`` (migration 163) through the shared session factory."""

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def load_all(self) -> List[Tuple[str, str, str]]:
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(text(
                "SELECT kind, name, equivalent FROM er_name_equivalents WHERE retired_at IS NULL"
            ))
            return [tuple(row) for row in result.all()]

    async def add(self, kind: str, name: str, equivalent: str, source: str) -> None:
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "INSERT INTO er_name_equivalents "
                    "(kind, name_key, equivalent_key, name, equivalent, source) "
                    "VALUES (:kind, :name_key, :equivalent_key, :name, :equivalent, :source) "
                    "ON CONFLICT (kind, name_key, equivalent_key) DO UPDATE "
                    "SET sightings = er_name_equivalents.sightings + 1, updated_at = now()"
                ),
                {
                    "kind": kind, "name_key": _key(name), "equivalent_key": _key(equivalent),
                    "name": name, "equivalent": equivalent, "source": source,
                },
            )
            await session.commit()
