"""Factory for creating EntityResolutionService from initialized synap-er components."""

import logging
from typing import Any, Optional

from synap.cloud.memory_config.config_schema import GraphConfig

logger = logging.getLogger(__name__)


def create_entity_resolution_service(
    graph_config: GraphConfig,
    llm_client: Optional[Any] = None,
    graph_wrapper: Optional[Any] = None,
) -> Optional["EntityResolutionService"]:
    """Create an EntityResolutionService from the global synap-er singleton.

    Returns None (with a debug log) if synap-er was not initialized at startup,
    allowing callers to proceed without entity resolution.
    """
    from .entity_resolver import EntityResolutionService
    from .startup import get_er_components

    components = get_er_components()
    if not components.initialized:
        logger.debug("synap-er not initialized; entity resolution disabled")
        return None

    from .mode import adjudicated_mode

    if not adjudicated_mode():
        return EntityResolutionService.from_config(
            graph_config=graph_config,
            resolver=components.resolver,
            write_through=components.write_through,
        )

    from synap.cloud.shared import edition

    # The judge steps (Jev, the local judge, the second look, the world lookup,
    # entity profiles), the held-mention store, the review page's sink and the
    # undo log are Cloud's. The hook returns the keyword arguments for
    # ``from_config``; in Core nothing is registered, so the resolver runs with no
    # judge, holds nothing, and queues nothing for review.
    judges = edition.call(
        "entity_judges", components=components, llm_client=llm_client, graph_wrapper=graph_wrapper,
    )
    if judges is None:
        logger.info(
            "Entity resolution in adjudicated mode: adjudicator=none dictionary=%s",
            "on" if getattr(components, "dictionary", None) is not None else "off",
        )
        judges = {}
    return EntityResolutionService.from_config(
        graph_config=graph_config,
        resolver=components.resolver,
        write_through=components.write_through,
        dictionary=getattr(components, "dictionary", None),
        index_manager=getattr(components, "index_manager", None),
        **judges,
    )
