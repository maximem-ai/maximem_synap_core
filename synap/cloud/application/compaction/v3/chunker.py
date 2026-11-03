"""Turn labelling and the chunker.

Every turn the writer sees is one line, ``[m12 · 2026-09-20 09:12 UTC · user]``
followed by its content. m12 is the turn's position in the conversation; the
timestamp is there so relative dates ("tomorrow", "next Friday") can be
resolved to absolute ones (I-11).

The chunker splits a span of turns into chunks of at most K tokens, only at
turn boundaries. A single turn larger than K is cut at paragraph breaks and
each piece labelled "part 2/3 of m57". Chunks are folded one at a time by the
engine, never in parallel: each update has to see what the previous chunk
established, or replaced values come out wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence

from synap.cloud.shared.tokens import count_tokens

from ..models import ConversationMessage


@dataclass
class Turn:
    label: str  # "m12"
    role: str
    content: str
    timestamp: str  # ISO 8601
    part: str = ""  # "part 2/3 of m57" when one turn was split

    def render(self) -> str:
        head = f"[{self.label} · {_display_time(self.timestamp)} · {self.role}"
        if self.part:
            head += f" · {self.part}"
        return f"{head}]\n{self.content}"

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())


def _iso(ts) -> str:
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts or "")


def _display_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC (%A)")


def label_turns(messages: Sequence[ConversationMessage], first_position: int = 1) -> List[Turn]:
    """Label ``messages`` m<first_position>, m<first_position+1>, ..."""
    return [
        Turn(label=f"m{first_position + i}", role=(m.role or "").lower(),
             content=m.content or "", timestamp=_iso(m.timestamp))
        for i, m in enumerate(messages)
    ]


def render_turns(turns: Sequence[Turn]) -> str:
    return "\n\n".join(t.render() for t in turns)


def _split_turn(turn: Turn, limit: int) -> List[Turn]:
    paragraphs = [p for p in turn.content.split("\n\n")] or [turn.content]
    pieces: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and count_tokens(candidate) > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    # A paragraph that is itself over the limit is cut by words.
    final: List[str] = []
    for piece in pieces:
        while count_tokens(piece) > limit:
            words = piece.split(" ")
            cut = max(1, int(len(words) * limit / max(1, count_tokens(piece))) - 1)
            final.append(" ".join(words[:cut]))
            piece = " ".join(words[cut:])
        final.append(piece)
    total = len(final)
    return [
        Turn(label=turn.label, role=turn.role, content=text, timestamp=turn.timestamp,
             part=f"part {i}/{total} of {turn.label}")
        for i, text in enumerate(final, 1)
    ]


def chunk_turns(turns: Sequence[Turn], limit: int) -> List[List[Turn]]:
    """Split into chunks of at most ``limit`` tokens, at turn boundaries."""
    chunks: List[List[Turn]] = []
    current: List[Turn] = []
    used = 0
    for turn in turns:
        pieces = [turn] if turn.tokens <= limit else _split_turn(turn, int(limit * 0.9))
        for piece in pieces:
            size = piece.tokens + 2
            if current and used + size > limit:
                chunks.append(current)
                current, used = [], 0
            current.append(piece)
            used += size
    if current:
        chunks.append(current)
    return chunks
