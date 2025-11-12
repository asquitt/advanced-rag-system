"""Tests for embedding service."""
import pytest
import numpy as np
from src.embeddings.embedding_service import EmbeddingService


@pytest.fixture
def embedder():
    """Create embedding service for tests."""
    return EmbeddingService()


def test_embed_single(embedder):
    """Test embedding a single text."""
    texts = ["Hello world"]
    embeddings = embedder.embed(texts)

    assert embeddings.shape == (1, embedder.dimension)
    assert embeddings.dtype == np.float32


def test_embed_batch(embedder):
    """Test batch embedding."""
    texts = [f"Document {i}" for i in range(10)]
    embeddings = embedder.embed(texts)

    assert embeddings.shape == (10, embedder.dimension)


def test_embed_query(embedder):
    """Test query embedding."""
    query = "test query"
    embedding = embedder.embed_query(query)

    assert embedding.shape == (embedder.dimension,)


def test_empty_input(embedder):
    """Test with empty input."""
    embeddings = embedder.embed([])
    assert embeddings.shape == (0,)


def test_embedding_normalized(embedder):
    """Test that embeddings are normalized."""
    texts = ["test"]
    embeddings = embedder.embed(texts)

    # L2 norm should be ~1.0
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 0.01
