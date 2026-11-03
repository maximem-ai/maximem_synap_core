"""Core's date guidance added to extraction requests.

The text is read from the Core prompt pack (``core_prompts/date_guidance``). Core states the date
the content was created and the fields to fill; it has no event-window or event-anchor rules.
"""

from datetime import datetime
from typing import Optional

from synap.cloud.shared.prompt_switch import fill, pack_text


def render_temporal_addendum(dct: Optional[datetime], *, instance_id: object = None) -> str:
    """The guidance for one document. ``dct`` is None when nothing says when it was written.

    ``instance_id`` is accepted so the call is the same as in Cloud; Core has one instance.
    """
    return fill(
        pack_text("date_guidance", "addendum.txt"),
        dct=dct.strftime("%Y-%m-%d") if dct else None,
        weekday=dct.strftime("%A") if dct else None,
        undated="yes" if dct is None else None,
    )
