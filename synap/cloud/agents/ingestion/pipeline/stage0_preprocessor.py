"""Stage 0: Pre-processor for smart document chunking.

Runs before the ingestion pipeline queues. Splits large or multi-session
documents into smaller chunks that preserve context and temporal reasoning.

All three ingestion paths converge at IngestionManager.ingest_document(),
so this preprocessor automatically applies to:
1. memories.create() (priority=EXPLICIT)
2. memories.batch_create() (priority=BOOTSTRAP)
3. Post-compaction promotion (priority=REALTIME)
"""

import logging
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, List, Optional, Tuple

from synap.cloud.agents.ingestion.models import IngestDocumentRequest
from synap.cloud.shared.tokens import get_token_counter

logger = logging.getLogger(__name__)

# Chunk size budget — fits comfortably within LLM context windows
# while producing semantically coherent units
MAX_TOKENS_PER_CHUNK = 3000

# The smallest chunk a config may ask for. A MACA config is LLM-generated, and a
# 50-token chunk is not a unit anything can extract a memory from: it would turn
# one document into dozens of extraction calls for no gain.
MIN_TOKENS_PER_CHUNK = 256

# Chunking strategies Stage 0 can actually carry out. The MACA schema accepts
# seven names and keeps accepting them on load, but only these change what Stage
# 0 does; every other name (hierarchical, page_aware, record_based, topic_based)
# is read as ``semantic``, the default rules, and says so in the log. The design
# call is only offered the supported ones.
#
#   turn_based  split chat at turn boundaries (after session boundaries)
#   semantic    the default rules: sessions then turns for chat, paragraphs for text
#   fixed       cut to the token budget at line boundaries and nothing smarter
SUPPORTED_CHUNKING_STRATEGIES = frozenset({"turn_based", "semantic", "fixed"})


def maca_chunking_enabled_for(instance_id: Optional[object]) -> bool:
    """Whether this instance's own chunking config steers Stage 0.

    ``SYNAP_MACA_CHUNKING_INSTANCES`` is a comma list of instance UUIDs or
    ``inst_...`` ids, or ``*`` for all. Unset means off everywhere, and that is
    the default on purpose: the configs MACA has written ask for 512-token
    conversation chunks against a fixed 3000 here, which is roughly six times as
    many extraction calls for the same conversation. Moving an instance onto its
    own chunk sizes is a cost decision, not a side effect of deploying this.
    """
    raw = os.environ.get("SYNAP_MACA_CHUNKING_INSTANCES", "") or ""
    allow = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not allow or instance_id is None:
        return False
    if "*" in allow:
        return True
    wanted = str(instance_id).strip().lower()
    if wanted in allow:
        return True
    try:
        from synap.cloud.shared.scope import parse_or_deterministic_uuid
        return any(wanted == str(parse_or_deterministic_uuid(a)).lower() for a in allow)
    except Exception:  # noqa: BLE001 - an id form we cannot read is simply not on the list
        return False


# Time gap between messages that suggests a new session
SESSION_GAP_SECONDS = 1800  # 30 minutes

# Patterns for explicit session boundary markers
SESSION_BOUNDARY_PATTERNS = [
    re.compile(r"^-{3,}\s*session\s*-{3,}", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^={3,}\s*Session\s+Start\s*={3,}", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\[session\s+started\]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^-{3,}\s*new\s+session\s*-{3,}", re.IGNORECASE | re.MULTILINE),
]

# Timestamp patterns for gap detection
TIMESTAMP_PATTERNS = [
    # ISO 8601: [2024-01-15T14:30:00] or (2024-01-15T14:30:00)
    re.compile(r"[\[(](\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)[\])]"),
    # US date: 1/15/2024 2:30 PM
    re.compile(r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)"),
    # Compact: 2024-01-15 14:30
    re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})"),
]

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %H:%M",
]

# Turn boundary pattern for conversation splitting
TURN_PATTERN = re.compile(
    r"^(user|assistant|human|ai|system|bot)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


class Stage0Preprocessor:
    """Pre-processes documents before pipeline queue entry.

    Splits large or multi-session documents into smaller chunks that:
    - Respect session boundaries in conversations
    - Preserve temporal reasoning via preceding summaries
    - Stay within LLM context window limits
    - Maintain traceability via chunk metadata
    """

    def __init__(self, max_tokens: int = MAX_TOKENS_PER_CHUNK):
        self._max_tokens = max_tokens
        self._counter = get_token_counter()

    def resolve_chunking(
        self, request: IngestDocumentRequest, chunking: Optional[Any]
    ) -> Tuple[str, int]:
        """The (strategy, token budget) that applies to this request.

        ``chunking`` is the instance's ``MemoryArchitectureConfig.chunking_config``
        or None. None, or an entry with no size, gives the fixed budget this class
        was built with. A configured size can only LOWER the budget: the fixed
        figure is what fits a model's context comfortably, so it is the ceiling,
        and :data:`MIN_TOKENS_PER_CHUNK` is the floor.

        Only the request's document type is matched, by its enum value
        (``conversation``, ``email``, ...). Names a config invents that are not a
        document type (``message``, ``policy``, ``code``) never match anything.
        """
        if chunking is None:
            return "semantic", self._max_tokens

        doc_type = getattr(request.document_type, "value", str(request.document_type))
        entry = (getattr(chunking, "by_document_type", None) or {}).get(doc_type)
        raw_strategy = getattr(entry, "strategy", None) or getattr(
            chunking, "default_strategy", None)
        strategy = getattr(raw_strategy, "value", raw_strategy) or "semantic"
        if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
            logger.info(
                "stage0_strategy_unsupported request_id=%s strategy=%s "
                "treated_as=semantic", request.request_id, strategy)
            strategy = "semantic"

        budget = self._max_tokens
        configured = getattr(entry, "max_tokens", None)
        if isinstance(configured, int) and configured > 0:
            floor = min(MIN_TOKENS_PER_CHUNK, self._max_tokens)
            budget = max(floor, min(configured, self._max_tokens))
        return strategy, budget

    def preprocess(
        self, request: IngestDocumentRequest, chunking: Optional[Any] = None
    ) -> List[IngestDocumentRequest]:
        """Split a request into chunks if needed.

        Args:
            request: Original ingestion request
            chunking: the instance's ``chunking_config`` when it is allowed to
                steer Stage 0 (see :func:`maca_chunking_enabled_for`), else None
                for the fixed rules.

        Returns:
            List of requests — single-element if no chunking needed,
            multiple elements if the document was split.
        """
        strategy, max_tokens = self.resolve_chunking(request, chunking)
        # Skip binary content
        if isinstance(request.content, bytes):
            return [request]

        content = request.content
        if not content or not content.strip():
            return [request]

        token_count = self._counter.count(content)

        # Small enough — no chunking needed
        if token_count <= max_tokens:
            return [request]

        logger.info(
            "stage0_chunking_needed request_id=%s tokens=%d max=%d strategy=%s",
            request.request_id,
            token_count,
            max_tokens,
            strategy,
        )

        # Try session splitting for conversation-like content
        from synap.cloud.agents.ingestion.models import DocumentType
        is_conversation = (
            request.document_type == DocumentType.CONVERSATION
            or self._looks_like_conversation(content)
        )

        chunks: List[str] = []
        if strategy == "fixed":
            # Budget-only: no session or turn logic, cut at line boundaries.
            chunks = self._force_split(content, max_tokens)
        else:
            if is_conversation:
                chunks = self._split_by_sessions(content)

            # If session splitting didn't produce useful results, logical chunk
            if len(chunks) <= 1:
                if is_conversation:
                    chunks = self._chunk_conversation_by_turns(content, max_tokens)
                else:
                    chunks = self._chunk_text_by_paragraphs(content, max_tokens)

        # Re-check: split any oversized chunks further
        final_chunks: List[str] = []
        for chunk in chunks:
            if self._counter.count(chunk) > max_tokens:
                # Force split by lines
                final_chunks.extend(self._force_split(chunk, max_tokens))
            else:
                final_chunks.append(chunk)

        if len(final_chunks) <= 1:
            return [request]

        logger.info(
            "stage0_split_complete request_id=%s chunks=%d",
            request.request_id,
            len(final_chunks),
        )

        # Build sub-requests with chunk metadata
        sub_requests = []
        preceding_summary = ""
        for i, chunk_text in enumerate(final_chunks):
            sub = self._clone_request_with_chunk(
                original=request,
                chunk_text=chunk_text,
                chunk_index=i,
                total_chunks=len(final_chunks),
                preceding_summary=preceding_summary if i > 0 else "",
            )
            sub_requests.append(sub)

            # Build preceding summary for next chunk (extractive)
            preceding_summary = self._generate_preceding_summary(
                preceding_summary, chunk_text,
            )

        return sub_requests

    def _looks_like_conversation(self, content: str) -> bool:
        """Heuristic: does this content look like a conversation transcript?"""
        matches = TURN_PATTERN.findall(content[:2000])
        return len(matches) >= 2

    # --- Session Splitting ---

    def _split_by_sessions(self, content: str) -> List[str]:
        """Try to split content at session boundaries.

        Tries explicit markers first, then timestamp gaps.
        """
        # Try explicit markers
        for pattern in SESSION_BOUNDARY_PATTERNS:
            parts = pattern.split(content)
            meaningful = [p.strip() for p in parts if p.strip()]
            if len(meaningful) >= 2:
                logger.info(
                    "stage0_session_split_by_marker sessions=%d", len(meaningful)
                )
                return meaningful

        # Try timestamp gap detection
        return self._split_by_timestamp_gaps(content)

    def _split_by_timestamp_gaps(self, content: str) -> List[str]:
        """Split content where timestamp gaps exceed SESSION_GAP_SECONDS."""
        lines = content.split("\n")
        timestamps: List[Tuple[int, datetime]] = []

        for line_idx, line in enumerate(lines):
            ts = self._parse_timestamp(line)
            if ts is not None:
                timestamps.append((line_idx, ts))

        if len(timestamps) < 2:
            return [content]  # Not enough timestamps to detect gaps

        # Find gap points
        split_indices: List[int] = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i][1] - timestamps[i - 1][1]).total_seconds()
            if gap >= SESSION_GAP_SECONDS:
                split_indices.append(timestamps[i][0])

        if not split_indices:
            return [content]

        # Split at gap points
        chunks = []
        prev = 0
        for split_line in split_indices:
            chunk = "\n".join(lines[prev:split_line]).strip()
            if chunk:
                chunks.append(chunk)
            prev = split_line
        # Last chunk
        remaining = "\n".join(lines[prev:]).strip()
        if remaining:
            chunks.append(remaining)

        if len(chunks) >= 2:
            logger.info(
                "stage0_session_split_by_timestamp sessions=%d gaps=%d",
                len(chunks),
                len(split_indices),
            )

        return chunks if len(chunks) >= 2 else [content]

    def _parse_timestamp(self, line: str) -> Optional[datetime]:
        """Try to extract a timestamp from a line of text."""
        for pattern in TIMESTAMP_PATTERNS:
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                for fmt in TIMESTAMP_FORMATS:
                    try:
                        return datetime.strptime(ts_str, fmt)
                    except ValueError:
                        continue
        return None

    # --- Logical Chunking ---

    def _chunk_conversation_by_turns(
        self, content: str, max_tokens: Optional[int] = None
    ) -> List[str]:
        """Split conversation at turn boundaries, respecting token budget."""
        max_tokens = max_tokens or self._max_tokens
        # Find all turn starts
        turn_starts = [m.start() for m in TURN_PATTERN.finditer(content)]

        if len(turn_starts) < 2:
            # Fallback to paragraph-based
            return self._chunk_text_by_paragraphs(content, max_tokens)

        # Extract turns
        turns: List[str] = []
        for i, start in enumerate(turn_starts):
            end = turn_starts[i + 1] if i + 1 < len(turn_starts) else len(content)
            turns.append(content[start:end].rstrip())

        # Greedily accumulate turns into chunks
        chunks: List[str] = []
        current_turns: List[str] = []
        current_tokens = 0

        for turn in turns:
            turn_tokens = self._counter.count(turn)
            if current_tokens + turn_tokens > max_tokens and current_turns:
                chunks.append("\n".join(current_turns))
                current_turns = []
                current_tokens = 0
            current_turns.append(turn)
            current_tokens += turn_tokens

        if current_turns:
            chunks.append("\n".join(current_turns))

        return chunks

    def _chunk_text_by_paragraphs(
        self, content: str, max_tokens: Optional[int] = None
    ) -> List[str]:
        """Split text at paragraph boundaries, respecting token budget."""
        max_tokens = max_tokens or self._max_tokens
        paragraphs = re.split(r"\n\s*\n", content)

        chunks: List[str] = []
        current_paras: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self._counter.count(para)

            if current_tokens + para_tokens > max_tokens and current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_tokens = 0

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            chunks.append("\n\n".join(current_paras))

        return chunks

    def _force_split(self, content: str, max_tokens: Optional[int] = None) -> List[str]:
        """Force-split content that exceeds token budget by lines."""
        max_tokens = max_tokens or self._max_tokens
        lines = content.split("\n")
        chunks: List[str] = []
        current_lines: List[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._counter.count(line)
            if current_tokens + line_tokens > max_tokens and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    # --- Preceding Summary ---

    def _generate_preceding_summary(
        self, existing_summary: str, chunk_text: str
    ) -> str:
        """Build extractive preceding summary from chunk text.

        Takes first 2 sentences + last sentence, capped at 500 chars.
        Concatenates with any existing summary from prior chunks.
        """
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return existing_summary

        # First 2 + last 1 (may overlap if <=3 sentences)
        selected = sentences[:2]
        if len(sentences) > 2:
            selected.append(sentences[-1])

        new_summary = " ".join(selected)
        if len(new_summary) > 300:
            new_summary = new_summary[:297] + "..."

        # Combine with existing (keep it compact)
        if existing_summary:
            combined = f"{existing_summary} | {new_summary}"
            if len(combined) > 500:
                combined = combined[:497] + "..."
            return combined

        return new_summary

    # --- Request Cloning ---

    def _clone_request_with_chunk(
        self,
        original: IngestDocumentRequest,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        preceding_summary: str,
    ) -> IngestDocumentRequest:
        """Clone a request with chunk content and metadata."""
        data = original.model_dump(mode="json")
        data["content"] = chunk_text
        data["request_id"] = f"{original.request_id}__chunk{chunk_index}"

        # Store chunk metadata for downstream stages
        metadata = dict(data.get("metadata") or {})
        metadata["_chunk"] = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "preceding_summary": preceding_summary,
            "original_request_id": original.request_id,
            "token_count": self._counter.count(chunk_text),
        }
        data["metadata"] = metadata

        return IngestDocumentRequest.model_validate(data)
