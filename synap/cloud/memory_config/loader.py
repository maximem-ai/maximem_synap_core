"""Which config loader this edition uses.

``get_config_loader()`` returns the Postgres loader when the edition's
``config_source`` is ``"table"`` (Cloud, the default) and the file loader when
it is ``"file"`` (Core). Readers call this instead of building a loader, so
none of them checks the edition.
"""

from __future__ import annotations

from typing import Any

from synap.cloud.shared import edition as _edition_module


def get_config_loader() -> Any:
    source = _edition_module.config_source
    if source == "file":
        from .file_loader import shared_loader

        return shared_loader()
    from synap.cloud.application.ingestion.startup import _DBConfigLoader

    return _DBConfigLoader()


def check_config_at_startup() -> None:
    """Core: read the memory config file once; a bad file raises ``MemoryConfigError`` and stops the server.

    Cloud (``config_source`` is ``"table"``): does nothing.
    """
    if _edition_module.config_source == "file":
        from .file_loader import check_memory_config_at_startup

        check_memory_config_at_startup()
