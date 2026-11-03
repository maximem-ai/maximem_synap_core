"""
Unit tests for SentenceTransformerEmbeddingClient.

Tests the self-hosted sentence-transformers embedding client.

NOTE (2026-06): The embedding client was reshaped by the arctic-embed-l-v2.0
upgrade (PR #419) and the standalone-embedding-service refactor. The old
``use_process_pool`` / ``_process_pool`` machinery was removed; the client now
exposes a ``query_prefix`` and an optional (env-gated) ONNX path, and defaults
to ``Snowflake/snowflake-arctic-embed-l-v2.0`` (1024-dim). These tests target
that current in-process API.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from synap.cloud.application.ingestion.embedding_client import (
    SentenceTransformerEmbeddingClient,
    get_sentence_transformer_client,
)


class TestSentenceTransformerEmbeddingClient:
    """Test suite for SentenceTransformerEmbeddingClient."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton model + ONNX state before each test."""
        SentenceTransformerEmbeddingClient._model = None
        SentenceTransformerEmbeddingClient._onnx_session = None
        SentenceTransformerEmbeddingClient._tokenizer = None
        SentenceTransformerEmbeddingClient._onnx_init_attempted = False
        SentenceTransformerEmbeddingClient._lock = None
        yield
        # Cleanup after test
        SentenceTransformerEmbeddingClient._model = None
        SentenceTransformerEmbeddingClient._onnx_session = None
        SentenceTransformerEmbeddingClient._tokenizer = None
        SentenceTransformerEmbeddingClient._onnx_init_attempted = False
        SentenceTransformerEmbeddingClient._lock = None

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_initialization_success(self, mock_sentence_transformer):
        """Test successful model initialization."""
        # Mock the SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Create client
        client = SentenceTransformerEmbeddingClient(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            batch_size=32,
        )

        # Verify model was loaded
        assert SentenceTransformerEmbeddingClient._model is not None
        mock_sentence_transformer.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_failure_fallback(self):
        """Test fallback behavior when model loading fails."""
        # Create client with non-existent model (will fail to load)
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")

            client = SentenceTransformerEmbeddingClient(
                model_name="non-existent-model",
            )

            # Model should be None after failed initialization
            assert SentenceTransformerEmbeddingClient._model is None

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_embed_single_text(self, mock_sentence_transformer):
        """Test embedding a single text."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embedding = np.array([0.1] * 384)
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        # Create client and embed
        client = SentenceTransformerEmbeddingClient()
        result = await client.embed("test text")

        # Verify
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once()

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_embed_batch(self, mock_sentence_transformer):
        """Test embedding multiple texts in batch."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_embeddings = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        # Create client and embed batch (no query prefix -> texts unchanged)
        client = SentenceTransformerEmbeddingClient(batch_size=32, query_prefix="")
        texts = ["text 1", "text 2", "text 3"]
        result = await client.embed_batch(texts)

        # Verify
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(emb) == 384 for emb in result)
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            convert_to_tensor=False,
            show_progress_bar=False,
        )

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_embed_batch_empty(self, mock_sentence_transformer):
        """Test embedding empty list returns empty list."""
        mock_sentence_transformer.return_value = MagicMock()
        client = SentenceTransformerEmbeddingClient()
        result = await client.embed_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_fallback_zero_vector(self):
        """Test fallback to zero vector when model unavailable."""
        # Create client with failed model loading
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")

            client = SentenceTransformerEmbeddingClient(
                dimension=384,
            )

            # Embed should return zero vector
            result = await client.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 384
            assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    async def test_embed_batch_fallback_zero_vectors(self):
        """Test fallback to zero vectors for batch when model unavailable."""
        # Create client with failed model loading
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Model not found")

            client = SentenceTransformerEmbeddingClient(
                dimension=384,
            )

            # Embed batch should return zero vectors
            result = await client.embed_batch(["text 1", "text 2"])

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(len(emb) == 384 for emb in result)
            assert all(all(x == 0.0 for x in emb) for emb in result)

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_embed_empty_text(self, mock_sentence_transformer):
        """Test embedding empty text returns zero vector."""
        mock_sentence_transformer.return_value = MagicMock()
        client = SentenceTransformerEmbeddingClient(
            dimension=384,
        )
        result = await client.embed("")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(x == 0.0 for x in result)

    @pytest.mark.asyncio
    @patch.dict("os.environ", {
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EMBEDDING_BATCH_SIZE": "64",
    })
    @patch("sentence_transformers.SentenceTransformer")
    async def test_factory_function_with_env_vars(self, mock_sentence_transformer):
        """Test factory function reads configuration from environment variables."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Create client via factory
        client = get_sentence_transformer_client()

        # Verify configuration from env vars
        assert client.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert client.batch_size == 64
        # all-MiniLM is symmetric -> no query prefix
        assert client.query_prefix == ""
        assert client.dimension == 384

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_factory_function_with_overrides(self, mock_sentence_transformer):
        """Test factory function with parameter overrides."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Create client via factory with overrides
        client = get_sentence_transformer_client(
            model_name="custom-model",
            batch_size=16,
        )

        # Verify overrides were applied
        assert client.model_name == "custom-model"
        assert client.batch_size == 16
        # Unknown model -> default dimension, no query prefix
        assert client.dimension == 1024
        assert client.query_prefix == ""

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_singleton_model_reuse(self, mock_sentence_transformer):
        """Test that model is loaded once and reused across instances."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Create two clients
        client1 = SentenceTransformerEmbeddingClient()
        client2 = SentenceTransformerEmbeddingClient()

        # Model should only be loaded once
        assert mock_sentence_transformer.call_count == 1

    @pytest.mark.asyncio
    @patch("sentence_transformers.SentenceTransformer")
    async def test_embed_error_handling(self, mock_sentence_transformer):
        """Test error handling during embedding generation."""
        # Mock the model to raise an error during encode
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Encoding error")
        mock_sentence_transformer.return_value = mock_model

        # Create client and attempt to embed (dimension drives the zero-vector
        # fallback length on encode failure)
        client = SentenceTransformerEmbeddingClient(dimension=384)
        result = await client.embed("test text")

        # Should fall back to zero vector
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(x == 0.0 for x in result)


class TestGetSentenceTransformerClient:
    """Test suite for get_sentence_transformer_client factory function."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    @patch("sentence_transformers.SentenceTransformer")
    async def test_default_configuration(self, mock_sentence_transformer):
        """Test factory function with default configuration."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_sentence_transformer.return_value = mock_model

        # Reset singleton
        SentenceTransformerEmbeddingClient._model = None
        SentenceTransformerEmbeddingClient._lock = None

        # Create client via factory
        client = get_sentence_transformer_client()

        # Verify defaults (arctic-embed-l-v2.0, 1024-dim, asymmetric query prefix)
        assert client.model_name == "Snowflake/snowflake-arctic-embed-l-v2.0"
        assert client.batch_size == 32
        assert client.dimension == 1024
        assert client.query_prefix == "query: "

        # Cleanup
        SentenceTransformerEmbeddingClient._model = None
