"""Entity resolution integration with synap-er.

Imports are lazy to avoid pulling in synap-er at module load time
(synap-er may not be installed in all environments).
"""

__all__ = [
    "EntityResolutionService",
    "create_entity_resolution_service",
    "get_er_components",
]


def __getattr__(name: str):
    if name == "EntityResolutionService":
        from .entity_resolver import EntityResolutionService
        return EntityResolutionService
    if name == "create_entity_resolution_service":
        from .factory import create_entity_resolution_service
        return create_entity_resolution_service
    if name == "get_er_components":
        from .startup import get_er_components
        return get_er_components
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
