"""Temporal Tagging Service.

Lightweight temporal detection service using regex patterns.
Designed for <50ms latency per message.

Provides temporal flags for conditional processing in ingestion pipeline.
Ported from maximem_memoryinfra, simplified (regex-only, no TimeNorm dependency).
"""

import re
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass

from synap.cloud.agents.ingestion.models.inputs import IngestDocumentRequest

logger = logging.getLogger(__name__)


@dataclass
class TemporalSignal:
    """Represents a detected temporal expression."""
    text: str
    normalized_value: Optional[datetime]
    signal_type: str  # 'absolute', 'relative', 'duration', 'range'
    start_char: int
    end_char: int


@dataclass
class TemporalFlags:
    """Lightweight flags for conditional processing."""
    has_explicit_date: bool = False
    has_relative_time: bool = False
    has_duration: bool = False
    has_temporal_range: bool = False
    has_narrative_markers: bool = False

    @property
    def has_temporal_content(self) -> bool:
        """Check if any temporal content was detected."""
        return any([
            self.has_explicit_date,
            self.has_relative_time,
            self.has_duration,
            self.has_temporal_range,
            self.has_narrative_markers,
        ])

    def to_dict(self) -> dict:
        return {
            'has_explicit_date': self.has_explicit_date,
            'has_relative_time': self.has_relative_time,
            'has_duration': self.has_duration,
            'has_temporal_range': self.has_temporal_range,
            'has_narrative_markers': self.has_narrative_markers,
            'has_temporal_content': self.has_temporal_content,
        }


# Regex patterns for narrative detection
NARRATIVE_PATTERNS = [
    # Past tense action verbs suggesting episodes
    r'\b(went|traveled|visited|attended|met|started|finished|completed|moved|graduated|married|divorced)\b',
    # Event nouns
    r'\b(trip|journey|vacation|meeting|event|conference|wedding|birthday|funeral|ceremony|party|interview|flight)\b',
    # Temporal range indicators
    r'\b(from\s+\w+\s+to\s+\w+)',
    # Narrative connectors
    r'\b(during|throughout|while\s+\w+ing)\b',
]

COMPILED_NARRATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NARRATIVE_PATTERNS]


class TemporalTaggingService:
    """Lightweight temporal tagging service (<50ms per message)."""

    def extract_temporal_flags(
        self,
        text: str,
        dct: Optional[datetime] = None,
    ) -> TemporalFlags:
        """Extract temporal flags from text using regex patterns."""
        if not text or len(text.strip()) == 0:
            return TemporalFlags()

        signals = self._extract_with_regex(text, dct or datetime.now(timezone.utc))

        flags = TemporalFlags()
        for signal in signals:
            if signal.signal_type == 'absolute':
                flags.has_explicit_date = True
            elif signal.signal_type == 'relative':
                flags.has_relative_time = True
            elif signal.signal_type == 'duration':
                flags.has_duration = True
            elif signal.signal_type == 'range':
                flags.has_temporal_range = True

        flags.has_narrative_markers = self._detect_narrative_markers(text)
        return flags

    def _extract_with_regex(
        self,
        text: str,
        dct: datetime,
    ) -> List[TemporalSignal]:
        """Regex-based temporal extraction."""
        signals = []

        # Absolute date patterns
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\w+ \d{1,2},? \d{4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                signals.append(TemporalSignal(
                    text=match.group(),
                    normalized_value=None,
                    signal_type='absolute',
                    start_char=match.start(),
                    end_char=match.end(),
                ))

        # Relative time patterns
        relative_patterns = [
            r'\b(yesterday|tomorrow|today)\b',
            r'\b(next|last)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d+\s+)?(days?|weeks?|months?|years?)\s+ago\b',
            r'\b(in|after)\s+\d+\s+(days?|weeks?|months?|years?)\b',
            r'\b(recently|lately|the other day)\b',
            r'\b(in|next|this)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ]

        for pattern in relative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                signals.append(TemporalSignal(
                    text=match.group(),
                    normalized_value=None,
                    signal_type='relative',
                    start_char=match.start(),
                    end_char=match.end(),
                ))

        # Duration patterns
        duration_pattern = r'\b(for|over|during)\s+\d+\s+(days?|weeks?|months?|years?|hours?|minutes?)\b'
        for match in re.finditer(duration_pattern, text, re.IGNORECASE):
            signals.append(TemporalSignal(
                text=match.group(),
                normalized_value=None,
                signal_type='duration',
                start_char=match.start(),
                end_char=match.end(),
            ))

        # Range patterns
        range_pattern = r'\b(from|between)\s+.*?\s+(to|until|through)\s+.*?\b'
        for match in re.finditer(range_pattern, text, re.IGNORECASE):
            signals.append(TemporalSignal(
                text=match.group(),
                normalized_value=None,
                signal_type='range',
                start_char=match.start(),
                end_char=match.end(),
            ))

        return signals

    def _detect_narrative_markers(self, text: str) -> bool:
        """Detect narrative markers using compiled regex patterns."""
        for pattern in COMPILED_NARRATIVE_PATTERNS:
            if pattern.search(text):
                return True
        return False


def _extract_dct_from_content(content: str) -> Optional[datetime]:
    """Extract latest timestamp from content text for DCT resolution.

    Scans content for embedded timestamps (e.g. chat log timestamps like
    [2024-01-15T14:30:00]) and returns the latest one found, as it represents
    the most recent point of the conversation — the correct anchor for
    resolving relative references like "today", "tomorrow", "yesterday".

    Reuses Stage 0's timestamp patterns to avoid duplication.
    """
    from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import (
        TIMESTAMP_PATTERNS, TIMESTAMP_FORMATS,
    )

    timestamps: List[datetime] = []
    for line in content.split("\n"):
        for pattern in TIMESTAMP_PATTERNS:
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                for fmt in TIMESTAMP_FORMATS:
                    try:
                        timestamps.append(datetime.strptime(ts_str, fmt))
                        break
                    except ValueError:
                        continue

    return max(timestamps) if timestamps else None




def resolve_dct(request: IngestDocumentRequest) -> Tuple[Optional[datetime], Optional[str]]:
    """When the conversation happened, as the client told us, or nothing.

    Priority:
      1. request.document_created_at (client-provided)
      2. request.metadata["timestamp_ms"] (the gRPC event's own send time)
      3. a timestamp embedded in the conversation text itself

    Returns (date, source), or (None, None) when nothing in the request says
    when the conversation happened.

    ⚠ There used to be a fourth priority, `request.ingested_at`, our own clock.
    It was defended as an ANCHOR rather than a fact: extraction has to count
    "three months ago" back from something, and when the client says nothing
    our clock is the only date available. The defence does not survive contact
    with the result. One value cannot answer two questions, and the moment it
    tried, our clock became the day a 2023 conversation happened: a transcript
    ingested with no dates in it dated "three months ago" to a date in our own
    present and stored it as an ordinary event date, indistinguishable from one
    somebody stated. A large share of event dates belonging to clients other
    than the benchmark landed within a day of the minute we wrote the row; on the
    benchmark, which always sends a conversation date, almost none did.

    So there is one question here now, and when the answer is unknown this says
    so. A phrase counted from a day that is not the conversation's day resolves
    to a wrong date, and a wrong date is worse than no date, which leaves
    nothing worth falling back to. Callers must handle None: a phrase counted
    from today cannot be dated without one, while a phrase that states its own
    date, or that counts from another event this conversation dates, still can.
    """
    # Priority 1: Client-provided DCT
    if request.document_created_at:
        return request.document_created_at, "client"

    # Priority 2: gRPC timestamp from metadata
    timestamp_ms = request.metadata.get("timestamp_ms") if request.metadata else None
    if timestamp_ms:
        try:
            dct = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc)
            return dct.replace(tzinfo=None), "grpc"
        except (ValueError, TypeError, OSError):
            pass

    # Priority 3: Extract from content timestamps
    if isinstance(request.content, str) and request.content:
        content_dct = _extract_dct_from_content(request.content)
        if content_dct:
            return content_dct, "content"

    # No fourth priority. `request.ingested_at` is when WE processed the
    # document and it is not an answer to "when did this conversation happen".
    return None, None
