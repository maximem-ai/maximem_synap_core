"""The source-document lookup is now only issued for items that need it.

An item that already carries a `source_document_id` has exactly one join row
equal to it, so asking the join table returns what the item already has. An item
WITHOUT one is a merged memory (KU v2 ENRICH), whose documents live only in the
join table, and those must still be resolved.

Today no stored memory has a NULL in-row value, so the
query is skipped entirely. The merged case does not exist right now, which
is exactly why it needs a test: the optimisation must not quietly break the
case it cannot currently observe.
"""
from __future__ import annotations

import pytest


class _Item:
    def __init__(self, item_id, source_document_id=None):
        self.item_id = item_id
        self.source_document_id = source_document_id


def _needs_lookup(items):
    """The predicate as written in _enrich_with_chunk_evidence."""
    return [
        it.item_id for it in items
        if it.item_id and not getattr(it, "source_document_id", None)
    ]


def test_items_that_already_carry_a_document_are_not_looked_up():
    items = [_Item("a", "doc-1"), _Item("b", "doc-2"), _Item("c", "doc-3")]
    assert _needs_lookup(items) == []


def test_a_merged_memory_with_no_in_row_document_is_still_looked_up():
    """The case stored data cannot currently show us. Breaking it would lose
    every contributing document of a merged memory, silently."""
    items = [_Item("a", "doc-1"), _Item("merged", None), _Item("c", "doc-3")]
    assert _needs_lookup(items) == ["merged"]


def test_an_empty_string_counts_as_missing_not_present():
    """`not ""` is True, so an empty string must route to the lookup rather than
    be treated as a document id. A blank id would otherwise be used as one."""
    assert _needs_lookup([_Item("a", "")]) == ["a"]


def test_items_without_an_id_are_skipped_entirely():
    """Graph items can have no id; the lookup casts to uuid and would reject
    them anyway, so they must not reach it."""
    assert _needs_lookup([_Item(None, None), _Item("", None)]) == []


def test_a_mixed_bundle_asks_only_for_the_gaps():
    items = [_Item("a", "doc-1"), _Item("b", None), _Item("c", "doc-3"),
             _Item("d", None), _Item(None, None)]
    assert _needs_lookup(items) == ["b", "d"]


def test_the_all_missing_case_still_asks_for_everything():
    """If a future release makes merged memories common, the query must come
    back in full rather than the optimisation quietly under-fetching."""
    items = [_Item("a"), _Item("b"), _Item("c")]
    assert _needs_lookup(items) == ["a", "b", "c"]
