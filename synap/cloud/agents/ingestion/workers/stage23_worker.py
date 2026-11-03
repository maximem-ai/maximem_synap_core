"""Worker for Stage 2+3: extraction priorities from the config + Memory Extraction."""

import logging
import time
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    MemoryType,
    Stage1Output,
    Stage2Output,
    PipelineStage,
)
from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.shared import edition
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from .base import BaseIngestionWorker

if TYPE_CHECKING:  # HQE stays in Cloud; the worker only needs its name for the type
    from synap.cloud.agents.ingestion.pipeline.stage5_hqe import Stage5HQEGenerator

if TYPE_CHECKING:  # Cloud's table-backed loader; Core passes the file loader, so nothing is imported at run time
    from synap.cloud.application.memory_architecture.repository import MemoryArchitectureRepository

logger = logging.getLogger(__name__)


def _build_stage2_from_config(
    config: MemoryArchitectureConfig,
    input_id: str,
) -> Stage2Output:
    """Build Stage2Output directly from MACA config, bypassing LLM categorization.

    Extraction priorities = all enabled memory types from memory_categories.
    """
    type_map = {
        "facts": MemoryType.FACTS,
        "preferences": MemoryType.PREFERENCES,
        "episodes": MemoryType.EPISODES,
        "emotions": MemoryType.EMOTIONS,
        "temporal_events": MemoryType.TEMPORAL_EVENTS,
    }
    config_map = {
        "facts": config.memory_categories.facts,
        "preferences": config.memory_categories.preferences,
        "episodes": config.memory_categories.episodes,
        "emotions": config.memory_categories.emotions,
        "temporal_events": config.memory_categories.temporal_events,
    }

    priorities = [
        type_map[name]
        for name, cat_config in config_map.items()
        if cat_config.enabled
    ]

    return Stage2Output(
        input_id=input_id,
        categories=[],
        extraction_priorities=priorities,
    )


class Stage23Worker(BaseIngestionWorker):
    """
    Worker for Stage 2 + 3 (grouped for efficiency).

    These stages are LLM-heavy and benefit from being in same process
    to reduce queue overhead.
    """

    @property
    def tracing_span_name(self) -> str:
        return "ingestion.stage23"

    @property
    def input_queue(self) -> str:
        return self.QUEUE_STAGE23

    @property
    def output_queue(self) -> str:
        return self.QUEUE_STAGE4

    @property
    def stage(self) -> PipelineStage:
        # Report as Stage 3 (the final stage in this worker)
        return PipelineStage.STAGE_3_EXTRACTION

    def __init__(
        self,
        extractor: Stage3Extractor,
        config_loader: "MemoryArchitectureRepository",
        hqe_generator: Optional["Stage5HQEGenerator"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._extractor = extractor
        self._config_loader = config_loader
        self._hqe_generator = hqe_generator

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process Stage 2 + 3: Categorize and Extract."""
        request = IngestDocumentRequest.model_validate(payload["request"])
        stage1_output = Stage1Output.model_validate(payload["stage1_output"])
        ingestion_id = payload.get("ingestion_id")
        instance_id = str(request.scope.instance_id)
        logger.info(
            "ingestion_stage23_start ingestion_id=%s request_id=%s stage1_should_process=%s",
            ingestion_id,
            request.request_id,
            stage1_output.should_process,
        )

        # Stage 2: Load config + Categorize
        async with start_span("ingestion.stage2", {
            "synap.ingestion_id": ingestion_id or "",
            "synap.instance_id": instance_id,
            "synap.content_len": len(request.content) if request.content else 0,
            "synap.nature": stage1_output.nature.value if stage1_output.nature else None,
        }):
            async with start_span("ingestion.stage2.load_config", {
                "synap.ingestion_id": ingestion_id or "",
                "synap.instance_id": instance_id,
            }):
                t0 = time.monotonic()
                config = await self._config_loader.get_config(
                    instance_id=instance_id,
                )
                set_current_span_attributes({
                    "synap.config_loaded": config is not None,
                    "synap.config_version": config.metadata.config_version if config else None,
                    "synap.load_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                })

            # Bypass LLM categorization — derive extraction priorities directly from MACA config.
            # Categories are never queried by retrieval/ranking/compaction (dead metadata).
            stage2_output = _build_stage2_from_config(config, stage1_output.input_id)
            set_current_span_attributes({
                "synap.category_count": 0,
                "synap.categories": "bypassed",
                "synap.categorize_latency_ms": 0,
                "synap.extraction_priorities_source": "maca_config",
            })

        logger.info(
            "ingestion_stage2_bypassed ingestion_id=%s request_id=%s extraction_priorities=%s",
            ingestion_id,
            request.request_id,
            [p.value for p in stage2_output.extraction_priorities],
        )

        # Resolve DCT and detect temporal content
        from synap.cloud.agents.ingestion.services.temporal_tagging_service import (
            TemporalTaggingService, resolve_dct,
        )

        dct, dct_source = resolve_dct(request)
        temporal_service = TemporalTaggingService()
        temporal_flags = temporal_service.extract_temporal_flags(
            text=request.content if isinstance(request.content, str) else "",
            dct=dct,
        )

        # The decision is Jev's, one message at a time; the regex flags above
        # are kept for the log and are the fallback. See temporal_gate.
        from synap.cloud.agents.ingestion.services import temporal_gate
        gate = await temporal_gate.decide(
            request.content if isinstance(request.content, str) else "",
            dct, correlation_id=request.request_id or "",
        )
        has_temporal_content = gate.needs_rules

        set_current_span_attributes({
            "synap.dct_source": dct_source,
            "synap.dct_value": dct.isoformat() if dct else None,
            "synap.has_temporal_content": has_temporal_content,
            "synap.temporal_gate_source": gate.source,
            "synap.temporal_gate_max_score": gate.max_score,
            "synap.temporal_flags": str(temporal_flags.to_dict()),
        })

        logger.info(
            "ingestion_temporal_detection ingestion_id=%s dct_source=%s has_temporal=%s "
            "gate=%s gate_score=%s gate_checked=%d/%d regex=%s",
            ingestion_id,
            dct_source,
            has_temporal_content,
            gate.source,
            None if gate.max_score is None else round(gate.max_score, 3),
            gate.messages_checked,
            gate.messages_total,
            temporal_flags.has_temporal_content,
        )

        # Stage 3: Extract memories
        # For chunked documents, prepend preceding summary for temporal context
        extraction_content = request.content
        if stage1_output.preceding_summary:
            extraction_content = (
                f"[CONTEXT FROM PREVIOUS PART OF DOCUMENT]\n"
                f"{stage1_output.preceding_summary}\n\n"
                f"[CURRENT SECTION TO ANALYZE]\n"
                f"{request.content}"
            )

        async with start_span("ingestion.stage3", {
            "synap.ingestion_id": ingestion_id or "",
            "synap.instance_id": instance_id,
            "synap.category_count": len(stage2_output.categories),
            "synap.extraction_priorities": ",".join(
                p.value if hasattr(p, "value") else str(p)
                for p in stage2_output.extraction_priorities
            )[:256],
            "synap.is_chunked": stage1_output.is_chunked,
            "synap.chunk_index": stage1_output.chunk_index,
        }):
            t0 = time.monotonic()
            stage3_output = await self._extractor.extract(
                content=extraction_content,
                stage2_output=stage2_output,
                document_id=request.document_id or ingestion_id,
                dct=dct,
                has_temporal_content=has_temporal_content,
                request_scope=request.scope,
                correlation_id=request.request_id,
                # Client-provided call-analysis JSON from the transcript
                # ingest endpoint (spec §4.5) — used as hints only.
                analysis_hints=(request.metadata or {}).get("analysis"),
                # The instance's MACA config, loaded above. Until 2026-09-17 it
                # was used only to pick memory types and never reached the
                # extraction prompts; extract() applies it when the instance is
                # enabled in SYNAP_MACA_EXTRACTION_INSTANCES.
                config=config,
            )
            set_current_span_attributes({
                "synap.memories_extracted": len(stage3_output.memories),
                "synap.entities_extracted": len(stage3_output.entities),
                "synap.relationships_extracted": len(stage3_output.relationships),
                "synap.extract_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # Stamp document_created_at on all extracted memories so it persists
        # through to RDBMS, ChromaDB, and Neo4j without threading through method signatures
        #
        # `pii_detected` rides along for the same reason. It is a fact about
        # the document, decided by the ingress gate before this pipeline was
        # entered (PII Protection), so extraction has no business
        # deciding it and only has to carry it.
        _pii_detected = (request.metadata or {}).get("pii_detected")
        # The day the conversation happened, and only when the client said so.
        #
        # This used to filter `dct` through a source label, because `resolve_dct`
        # always returned a date and the last one it could reach was our own
        # ingest clock. That is no longer true: it returns None when nothing in
        # the request says when the conversation happened, so there is nothing
        # left to filter and the value can be stamped as it comes.
        #
        # Ingesting one benchmark session twice through the transcript route
        # showed it: with turn timestamps the stored date was right; with no
        # dates anywhere it was our own clock, on every memory. Almost every
        # row carried a value in this field, so "we do not know" had
        # effectively stopped being sayable.
        for memory in stage3_output.memories:
            memory.document_created_at = dct
            if _pii_detected is not None:
                memory.pii_detected = bool(_pii_detected)

        # The event window the model wrote (WHEN the event happened) is checked
        # against DCT before it is stored: bad dates are dropped, never guessed.
        from synap.cloud.agents.ingestion.services.event_window import validate_event_windows
        _windowed = validate_event_windows(stage3_output.memories, dct)
        logger.info(
            "ingestion_event_windows ingestion_id=%s with_window=%d of=%d",
            ingestion_id, _windowed, len(stage3_output.memories),
        )

        # PII Protection, the way out.
        #
        # Two separate things happen here and they carry different weight.
        #
        # The safety re-check is not a duplicate of the check at the door. The
        # incoming text is long and messy, and "my adhar is one two three four"
        # is hard to catch. What comes back is a short clean sentence, "Their
        # Aadhaar is 123456789012", which is easy to catch. Extraction tidies
        # the text up, and tidy text is far easier to match, so this genuinely
        # finds things the first pass cannot (feature spec 8).
        #
        # The drop rule is a quality decision, not a safety one. By the time it
        # runs the value is already gone, so if it gets one wrong we keep an
        # unhelpful row and nothing leaks. That is why it is allowed to be
        # simple.
        stage3_output.memories = await self._apply_pii_exit_rules(
            stage3_output.memories, request
        )


        # Stage 5: Generate hypothetical questions via dedicated LLM call.
        # Must run before serialization so questions are in the Stage4 payload.
        # Enabled by default — set ENABLE_QUESTION_EMBEDDINGS=false to disable.
        enable_hqe = os.environ.get("ENABLE_QUESTION_EMBEDDINGS", "true").lower() in ("true", "1", "yes")
        if self._hqe_generator and stage3_output.memories and enable_hqe:
            async with start_span("ingestion.stage5_hqe", {
                "synap.ingestion_id": ingestion_id or "",
                "synap.memory_count": len(stage3_output.memories),
            }):
                t0_hqe = time.monotonic()
                stage3_output.memories = await self._hqe_generator.generate(
                    memories=stage3_output.memories,
                    request_scope=request.scope,
                    correlation_id=request.request_id,
                )
                question_total = sum(len(m.hypothetical_questions) for m in stage3_output.memories)
                set_current_span_attributes({
                    "synap.hqe_questions_generated": question_total,
                    "synap.hqe_latency_ms": round((time.monotonic() - t0_hqe) * 1000, 2),
                })

        logger.info(
            "ingestion_stage3_completed ingestion_id=%s request_id=%s memories=%s entities=%s relationships=%s",
            ingestion_id,
            request.request_id,
            len(stage3_output.memories),
            len(stage3_output.entities),
            len(stage3_output.relationships),
        )

        # The storage view, and it runs **last** on purpose.
        #
        # Extraction ran on the model view, so a `hide from the model` value is
        # sitting in these memories as a label while our own store is supposed
        # to hold the real thing. The value is recovered from the alias table
        # rather than remembered, because this worker never saw the original
        # document.
        #
        # ⚠️ It used to run before Stage 5, and that sent the restored real
        # value straight back to a model provider: HQE builds its prompt from
        # `memory.content`, so a client who chose "hide from the model"
        # specifically to keep phone numbers away from OpenAI had them shipped
        # there one step later. Everything upstream of this line must see the
        # alias. Nothing between here and persistence calls a model.
        stage3_output.memories = await self._apply_pii_storage_view(
            stage3_output.memories, request
        )

        return {
            "ingestion_id": payload["ingestion_id"],
            "request": request.model_dump(mode="json"),
            "stage1_output": stage1_output.model_dump(mode="json"),
            "stage2_output": stage2_output.model_dump(mode="json"),
            "stage3_output": stage3_output.model_dump(mode="json"),
            "maca_config_version": payload.get("maca_config_version"),
        }

    async def _apply_pii_storage_view(self, memories, request):
        """Put back the values our own store holds in the clear (Cloud PII).

        The real body is the ``pii_ingestion`` hook (``storage_view``). With no
        PII feature registered the memories pass through unchanged.
        """
        out = await edition.acall("pii_ingestion", "storage_view", memories, request)
        return memories if out is None else out

    async def _apply_pii_exit_rules(self, memories, request):
        """The safety re-check and the drop rule (Cloud PII).

        The real body is the ``pii_ingestion`` hook (``exit_rules``). With no
        PII feature registered every memory is kept.
        """
        out = await edition.acall("pii_ingestion", "exit_rules", memories, request)
        return memories if out is None else out
