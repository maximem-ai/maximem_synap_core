"""The enums a registry entry is described with.

They are synap-er's, because its resolver switches on them. synap-er 0.4
keeps them in ``synap_er.registry``; older releases, which also owned the
table, kept them beside its model.
"""

try:
    from synap_er.registry import EntityType, Scope, Source
except ImportError:      # synap-er < 0.4
    from synap_er.models.registry import EntityType, Scope, Source

__all__ = ["EntityType", "Scope", "Source"]
