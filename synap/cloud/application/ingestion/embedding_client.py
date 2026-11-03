"""
Sentence-Transformers Embedding Client

Self-hosted embedding generation using sentence-transformers library.
Provides local inference without external API dependencies.
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


# The exact Hugging Face commit the loader may fetch, per model. Asking for a
# branch or tag would let the publisher (or whoever takes over their account)
# change what runs at the next restart, and could change the vectors so new
# memories stop matching old ones. The files that decide the vectors (weights,
# tokenizer, pooling and sentence-transformers config) are byte-identical in
# every commit of this model from Dec 2024 to the one below.
PINNED_EMBEDDING_REVISIONS = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
}

_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


class UnpinnedEmbeddingModelError(RuntimeError):
    """The model to load has no exact revision, so it is not loaded."""


def resolve_embedding_revision(model_name: str) -> str:
    """The exact commit to load ``model_name`` at, or raise.

    ``EMBEDDING_MODEL_REVISION`` wins when set; otherwise the model must be in
    ``PINNED_EMBEDDING_REVISIONS``. Only a full 40-character commit hash is
    accepted: a branch name, a tag or a short hash can move or be re-pointed.
    """
    revision = (os.environ.get("EMBEDDING_MODEL_REVISION") or "").strip().lower()
    if revision:
        if not _COMMIT_HASH.fullmatch(revision):
            raise UnpinnedEmbeddingModelError(
                f"EMBEDDING_MODEL_REVISION={revision!r} is not a full 40-character "
                f"commit hash. Set it to the exact commit of {model_name} to load."
            )
        return revision
    pinned = PINNED_EMBEDDING_REVISIONS.get(model_name)
    if pinned is None:
        raise UnpinnedEmbeddingModelError(
            f"No pinned revision for embedding model {model_name!r}. Set "
            f"EMBEDDING_MODEL_REVISION to the exact commit hash to load, so the "
            f"model files cannot change under the server."
        )
    return pinned


# --- embed_perf JSONL logger (opt-in via SYNAP_EMBED_PERF_LOG_ENABLED) ---
# Captures per-call latency so we can compare distributions across embedding
# models without depending on the broader OTEL pipeline. Writes one JSONL line
# per encode() call, rotated daily. Bypasses the broken alloy→tempo path.
_EMBED_PERF_LOG_ENABLED = os.environ.get(
    "SYNAP_EMBED_PERF_LOG_ENABLED", "false"
).lower() in ("true", "1", "yes")
_EMBED_PERF_LOG_DIR = os.environ.get("SYNAP_EMBED_PERF_LOG_DIR", "/tmp/synap_embed_perf")
_embed_perf_lock = None


def _embed_perf_log(model_name: str, op: str, batch_size: int, dim: int, total_ms: float) -> None:
    if not _EMBED_PERF_LOG_ENABLED:
        return
    global _embed_perf_lock
    try:
        import json
        import pathlib
        import threading
        from datetime import datetime, timezone
        if _embed_perf_lock is None:
            _embed_perf_lock = threading.Lock()
        ts = datetime.now(timezone.utc)
        path = pathlib.Path(_EMBED_PERF_LOG_DIR) / f"embed_perf_{ts.strftime('%Y-%m-%d')}.jsonl"
        rec = {
            "ts": ts.isoformat(),
            "model": model_name,
            "op": op,
            "batch_size": batch_size,
            "dim": dim,
            "total_ms": round(total_ms, 3),
            "ms_per_item": round(total_ms / max(batch_size, 1), 3),
        }
        with _embed_perf_lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except PermissionError:
                fallback = pathlib.Path("/tmp") / path.name
                with open(fallback, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class SentenceTransformerEmbeddingClient:
    """
    Self-hosted embedding client using sentence-transformers.

    Features:
    - Local inference (no external API calls)
    - Model: Snowflake/snowflake-arctic-embed-l-v2.0 (1024 dimensions, multilingual)
    - Query prefix support for asymmetric retrieval models
    - ONNX INT8 acceleration (GIL-free, ~2x faster on CPU)
    - Batch processing with configurable batch size
    - HuggingFace model caching
    - Compatible with EmbeddingClient protocol

    Environment Variables:
    - EMBEDDING_MODEL: Model name (default: Snowflake/snowflake-arctic-embed-l-v2.0)
    - EMBEDDING_MODEL_REVISION: Full 40-character commit hash to load. Required
      for any model not in PINNED_EMBEDDING_REVISIONS; the loader refuses to
      start without one. Overrides the built-in pin when set.
    - EMBEDDING_TRUST_REMOTE_CODE: Allow the model to run its own code (default: false)
    - EMBEDDING_BATCH_SIZE: Batch size for encoding (default: 32)
    - TRANSFORMERS_CACHE: HuggingFace transformers cache directory
    - HF_HOME: HuggingFace home directory
    - HUGGINGFACE_HUB_CACHE: HuggingFace Hub cache directory
    - HUGGINGFACE_API_TOKEN: Optional HuggingFace API token
    """

    _model = None  # Singleton model instance
    _onnx_session = None  # ONNX INT8 session for fast passage encoding (~2x CPU speedup)
    _tokenizer = None  # Shared tokenizer for ONNX path
    _onnx_init_attempted = False
    _lock = None  # Thread lock for model initialization

    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
        batch_size: int = 32,
        dimension: int = 1024,
        query_prefix: str = "query: ",
    ):
        """
        Initialize the sentence-transformers embedding client.

        Args:
            model_name: Name of the sentence-transformers model
            batch_size: Batch size for encoding multiple texts
            dimension: Expected embedding dimension (for fallback)
            query_prefix: Prefix prepended to queries for asymmetric retrieval models
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self.query_prefix = query_prefix
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the sentence-transformers model.

        Uses singleton pattern to load model once and reuse across instances.
        Thread-safe initialization with lazy loading.
        """
        if SentenceTransformerEmbeddingClient._model is not None:
            return

        # Initialize thread lock if needed
        if SentenceTransformerEmbeddingClient._lock is None:
            import threading
            SentenceTransformerEmbeddingClient._lock = threading.Lock()

        # Thread-safe model loading
        with SentenceTransformerEmbeddingClient._lock:
            if SentenceTransformerEmbeddingClient._model is not None:
                return

            # Outside the try below on purpose: an unpinned model must stop the
            # server, not fall back to zero vectors.
            revision = resolve_embedding_revision(self.model_name)

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    f"Loading sentence-transformers model: {self.model_name} @ {revision}"
                )

                # Set HuggingFace cache directories from environment
                cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")

                # Load model with caching, at the pinned commit. arctic-l-v2.0 is a
                # plain XLM-RoBERTa model with no code of its own (vectors are
                # identical with and without remote code), so remote code stays
                # off unless the operator asks for it for another model.
                SentenceTransformerEmbeddingClient._model = SentenceTransformer(
                    self.model_name,
                    cache_folder=cache_dir,
                    revision=revision,
                    trust_remote_code=os.environ.get(
                        "EMBEDDING_TRUST_REMOTE_CODE", "false"
                    ).lower() in ("true", "1", "yes"),
                )

                logger.info(f"Successfully loaded model: {self.model_name}")
                logger.info(f"Model dimension: {SentenceTransformerEmbeddingClient._model.get_sentence_embedding_dimension()}")

            except Exception as e:
                logger.warning(
                    f"Failed to load sentence-transformers model '{self.model_name}': {e}. "
                    f"Will use zero vectors for embeddings (dimension={self.dimension})."
                )
                SentenceTransformerEmbeddingClient._model = None

    async def embed(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generate embedding vector for a single text.

        Uses ONNX INT8 when available (GIL-free, ~2x faster), falls back to PyTorch.

        Args:
            text: The text to generate an embedding for
            is_query: If True, prepend query prefix (for asymmetric retrieval models)

        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.dimension

        if SentenceTransformerEmbeddingClient._model is None:
            logger.warning(f"Model not available, returning zero vector for text: {text[:50]}...")
            return [0.0] * self.dimension

        # For queries with prefix, ONNX path doesn't apply the prefix,
        # so route through embed_batch which handles it
        if is_query and self.query_prefix:
            results = await self.embed_batch([text], is_query=True)
            return results[0]

        # Try ONNX path (GIL-free, ~2x faster) when enabled via env var.
        # Disabled by default: lazy ONNX export spikes memory (~7GB) and OOM-kills
        # on small instances. Enable only when the instance has >=16GB RAM, or
        # pre-export the ONNX model outside the app process.
        cls = SentenceTransformerEmbeddingClient
        if os.environ.get("EMBEDDING_ONNX_ENABLED", "false").lower() in ("true", "1", "yes"):
            if not cls._onnx_init_attempted:
                self._initialize_onnx()

            if cls._onnx_session is not None and cls._tokenizer is not None:
                try:
                    results = await self._embed_onnx([text])
                    return results[0]
                except Exception as e:
                    logger.warning(f"ONNX single embed failed, falling back to PyTorch: {e}")

        return await self._embed_pytorch(text)

    async def _embed_pytorch(self, text: str) -> List[float]:
        """PyTorch fallback for single text embedding."""
        try:
            import asyncio
            import time
            t0 = time.perf_counter()
            embedding = await asyncio.to_thread(
                SentenceTransformerEmbeddingClient._model.encode,
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
            _embed_perf_log(self.model_name, "pytorch_single", 1, self.dimension,
                            (time.perf_counter() - t0) * 1000)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            return [0.0] * self.dimension

    async def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Uses ONNX INT8 when available (GIL-free, ~2x faster), falls back to PyTorch.

        Args:
            texts: List of texts to generate embeddings for
            is_query: If True, prepend query prefix (for asymmetric retrieval models)

        Returns:
            List of embedding vectors, one per input text
        """
        if not texts:
            return []

        if SentenceTransformerEmbeddingClient._model is None:
            logger.warning(f"Model not available, returning zero vectors for {len(texts)} texts")
            return [[0.0] * self.dimension for _ in texts]

        encode_texts = [f"{self.query_prefix}{t}" for t in texts] if is_query and self.query_prefix else texts

        # Try ONNX path (GIL-free, ~2x faster) when enabled via env var.
        cls = SentenceTransformerEmbeddingClient
        if os.environ.get("EMBEDDING_ONNX_ENABLED", "false").lower() in ("true", "1", "yes"):
            if not cls._onnx_init_attempted:
                self._initialize_onnx()

            if cls._onnx_session is not None and cls._tokenizer is not None:
                try:
                    return await self._embed_onnx(encode_texts)
                except Exception as e:
                    logger.warning(f"ONNX batch embed failed, falling back to PyTorch: {e}")

        return await self._embed_batch_pytorch(encode_texts)

    async def _embed_batch_pytorch(self, texts: List[str]) -> List[List[float]]:
        """PyTorch fallback for batch embedding. Texts should already have query prefix applied."""
        try:
            import asyncio
            import time
            t0 = time.perf_counter()
            embeddings = await asyncio.to_thread(
                SentenceTransformerEmbeddingClient._model.encode,
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
            _embed_perf_log(self.model_name, "pytorch_batch", len(texts), self.dimension,
                            (time.perf_counter() - t0) * 1000)

            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return [list(emb) for emb in embeddings]

        except Exception as e:
            logger.error(f"Error generating embeddings for batch of {len(texts)} texts: {e}")
            return [[0.0] * self.dimension for _ in texts]

    def _initialize_onnx(self):
        """Load the pre-baked ONNX INT8 passage encoder from EMBEDDING_ONNX_DIR.

        Previously this method exported the PyTorch model to ONNX at runtime via
        ``torch.onnx.export`` + ``quantize_dynamic``. That path was fragile:
            (1) it spiked memory by ~7GB during export (OOM-killed small
                containers); and
            (2) recent transformers releases broke ``torch.onnx.export`` for the
                arctic-l-v2 XLM-RoBERTa variant via the new SDPA mask code
                (``masking_utils.sdpa_mask`` IndexError on traced tensors).

        New approach: the Dockerfile has an `onnx-downloader` build stage that
        fetches ``onnx/model_int8.onnx`` and the tokenizer files from the
        Snowflake HF repo at image build time, and the production stage copies
        them into ``/app/models/onnx/arctic-l-v2.0/`` (path exposed via the
        ``EMBEDDING_ONNX_DIR`` env var). This method just opens the file. No
        export, no runtime memory spike, no transformers-version coupling.

        Verified: cosine 0.97-0.98 vs PyTorch, 5.6× speedup at
        batch=1, 2.3-2.5× at batch=16.
        """
        cls = SentenceTransformerEmbeddingClient
        if cls._onnx_init_attempted:
            return

        # Double-checked locking: multiple stage4 workers can call embed_batch()
        # concurrently on the first invocation. Same pattern as _initialize_model().
        with cls._lock:
            if cls._onnx_init_attempted:
                return
            cls._onnx_init_attempted = True

            onnx_dir = os.environ.get("EMBEDDING_ONNX_DIR")
            if not onnx_dir:
                logger.warning(
                    "EMBEDDING_ONNX_DIR is not set — ONNX path disabled. "
                    "Falling back to PyTorch. Set EMBEDDING_ONNX_DIR to the "
                    "directory containing model_int8.onnx + tokenizer files."
                )
                return

            int8_path = os.path.join(onnx_dir, "model_int8.onnx")
            if not os.path.exists(int8_path):
                logger.warning(
                    f"ONNX model file not found at {int8_path} — falling back "
                    f"to PyTorch. Either the Dockerfile onnx-downloader stage "
                    f"did not populate the directory, or EMBEDDING_ONNX_DIR "
                    f"points to the wrong location."
                )
                return

            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                logger.info(
                    f"Loading pre-baked ONNX INT8 passage encoder from {onnx_dir}..."
                )

                # Tokenizer loaded from the same directory the Dockerfile
                # populated. trust_remote_code=False on purpose — the tokenizer
                # is a vanilla XLMRobertaTokenizerFast, no custom code needed.
                tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

                # Single-threaded session: arctic-l-v2 INT8 inference is
                # latency-sensitive per call, and we run multiple sessions in
                # parallel via the chroma_executor anyway. Letting ORT spin up
                # its internal thread-pool per call adds startup cost.
                # CPUExecutionProvider — no GPU assumed.
                session = ort.InferenceSession(
                    int8_path,
                    providers=["CPUExecutionProvider"],
                )

                cls._onnx_session = session
                cls._tokenizer = tokenizer

                int8_size_mb = os.path.getsize(int8_path) / 1e6
                logger.info(
                    f"ONNX INT8 passage encoder ready: model_size={int8_size_mb:.0f}MB, "
                    f"inputs={[i.name for i in session.get_inputs()]}, "
                    f"outputs={[o.name for o in session.get_outputs()]}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load pre-baked ONNX passage encoder "
                    f"(will use PyTorch): {type(e).__name__}: {e}"
                )

    async def _embed_onnx(self, texts: List[str]) -> List[List[float]]:
        """ONNX INT8 batch encoding — GIL-free, ~2-5× faster than PyTorch on CPU.

        Uses the model's canonical ``sentence_embedding`` output when the
        pre-baked ONNX graph exposes it (current arctic-l-v2 INT8 graph from
        Snowflake does — see ``session.get_outputs()`` log line in
        ``_initialize_onnx``). The graph already applies the same CLS-pooling
        + L2-normalization that the PyTorch path's sentence-transformers
        pooling layer does, so we don't re-do it on numpy.

        Falls back to ``token_embeddings`` + manual CLS pooling for ONNX
        models that only expose the raw transformer output (older models
        exported with our previous custom torch.onnx.export path).

        Caller must check that _onnx_session and _tokenizer are available
        before calling.
        """
        import asyncio
        import time
        import numpy as np

        session = SentenceTransformerEmbeddingClient._onnx_session
        tokenizer = SentenceTransformerEmbeddingClient._tokenizer

        # Detect whether the graph exposes a pre-pooled ``sentence_embedding``
        # output. The Snowflake-published ONNX exports a (token_embeddings,
        # sentence_embedding) tuple; our older custom export only produced
        # last_hidden_state.
        output_names = [o.name for o in session.get_outputs()]
        has_sentence_embedding = "sentence_embedding" in output_names

        def _encode_onnx():
            enc = tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            out = session.run(
                None,
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )
            if has_sentence_embedding:
                # Index by name to avoid relying on output ordering (some
                # exports list sentence_embedding first, others second).
                idx = output_names.index("sentence_embedding")
                # Already pooled + L2-normalized inside the ONNX graph —
                # matches PyTorch sentence-transformers exactly.
                return out[idx].tolist()
            # Fallback: raw transformer output, do CLS pooling + L2 norm manually.
            pooled = out[0][:, 0, :]
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / (norms + 1e-9)
            return normalized.tolist()

        t0 = time.perf_counter()
        result = await asyncio.to_thread(_encode_onnx)
        _embed_perf_log(self.model_name, "onnx_batch", len(texts), self.dimension,
                        (time.perf_counter() - t0) * 1000)
        return result

    @classmethod
    def shutdown(cls):
        """Cleanup ONNX session on application shutdown."""
        if cls._onnx_session is not None:
            logger.info("Cleaning up ONNX session")
            cls._onnx_session = None
            cls._tokenizer = None


from synap.cloud.application.memory_architecture.configurators.storage import EMBEDDING_DIMENSIONS

# Query prefixes for asymmetric retrieval models (documents need no prefix).
# v1.5 is symmetric (no prefix). v2.0 is asymmetric — XLM-R-style "query: " prefix.
_MODEL_QUERY_PREFIXES = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": "query: ",
}


def get_sentence_transformer_client(
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> SentenceTransformerEmbeddingClient:
    """
    Factory function to create a sentence-transformers embedding client.

    Reads configuration from environment variables:
    - EMBEDDING_MODEL (default: Snowflake/snowflake-arctic-embed-l-v2.0)
    - EMBEDDING_BATCH_SIZE (default: 32)

    Args:
        model_name: Override model name (uses env var if not provided)
        batch_size: Override batch size (uses env var if not provided)

    Returns:
        SentenceTransformerEmbeddingClient instance
    """
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL",
        "Snowflake/snowflake-arctic-embed-l-v2.0"
    )

    batch_size = batch_size or int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

    dimension = EMBEDDING_DIMENSIONS.get(model_name, EMBEDDING_DIMENSIONS.get("default", 1024))
    query_prefix = _MODEL_QUERY_PREFIXES.get(model_name, "")

    logger.info(
        f"Creating SentenceTransformerEmbeddingClient: "
        f"model={model_name}, batch_size={batch_size}, "
        f"dimension={dimension}, "
        f"query_prefix={'yes' if query_prefix else 'no'}"
    )

    return SentenceTransformerEmbeddingClient(
        model_name=model_name,
        batch_size=batch_size,
        dimension=dimension,
        query_prefix=query_prefix,
    )
