"""Tests for embedding service."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.embeddings.embedding_service import EmbeddingService


@pytest.fixture
def embedder():
    """Create embedding service for tests without cache."""
    return EmbeddingService(use_cache=False)


@pytest.fixture
def embedder_with_mock_cache():
    """Create embedding service with mocked Redis cache."""
    with patch("src.embeddings.embedding_service.redis.Redis") as mock_redis:
        mock_cache = MagicMock()
        mock_cache.ping.return_value = True
        mock_cache.get.return_value = None
        mock_cache.setex.return_value = True
        mock_redis.return_value = mock_cache

        service = EmbeddingService(use_cache=True)
        service.cache = mock_cache
        yield service


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


def test_initialization_with_cache(embedder_with_mock_cache):
    """Test initialization with cache enabled."""
    assert embedder_with_mock_cache.use_cache is True
    assert embedder_with_mock_cache.cache is not None


@patch("src.embeddings.embedding_service.redis.Redis")
def test_initialization_cache_failure(mock_redis):
    """Test initialization when Redis is unavailable."""
    mock_redis.return_value.ping.side_effect = Exception("Connection failed")

    service = EmbeddingService(use_cache=True)

    # Should disable cache on failure
    assert service.use_cache is False


def test_cache_key_generation(embedder):
    """Test cache key generation."""
    text = "test text"
    key = embedder._cache_key(text)

    assert isinstance(key, str)
    assert key.startswith("emb:v1:")
    # Same text should generate same key
    assert embedder._cache_key(text) == key


def test_embed_with_cache_miss(embedder_with_mock_cache):
    """Test embedding with cache miss."""
    embedder_with_mock_cache.cache.get.return_value = None

    texts = ["test text"]
    embeddings = embedder_with_mock_cache.embed(texts)

    # Should generate embeddings
    assert embeddings.shape == (1, embedder_with_mock_cache.dimension)

    # Should attempt to cache
    embedder_with_mock_cache.cache.setex.assert_called()


def test_embed_with_cache_hit(embedder_with_mock_cache):
    """Test embedding with cache hit."""
    # Mock cached embedding
    cached_emb = np.random.randn(384).astype(np.float32)
    embedder_with_mock_cache.cache.get.return_value = cached_emb.tobytes()

    texts = ["cached text"]
    embeddings = embedder_with_mock_cache.embed(texts)

    # Should return cached embedding
    assert embeddings.shape == (1, 384)
    assert np.allclose(embeddings[0], cached_emb)


def test_embed_skip_cache(embedder_with_mock_cache):
    """Test embedding with cache disabled via parameter."""
    texts = ["test"]
    embeddings = embedder_with_mock_cache.embed(texts, check_cache=False)

    # Should not check cache
    embedder_with_mock_cache.cache.get.assert_not_called()
    assert embeddings.shape == (1, embedder_with_mock_cache.dimension)


def test_embed_cache_error_handling(embedder_with_mock_cache):
    """Test that cache errors don't break embedding."""
    # Mock cache.get to raise exception
    embedder_with_mock_cache.cache.get.side_effect = Exception("Cache error")

    texts = ["test"]
    # Should still generate embeddings despite cache error
    embeddings = embedder_with_mock_cache.embed(texts)

    assert embeddings.shape == (1, embedder_with_mock_cache.dimension)


def test_embed_mixed_cache_hits(embedder_with_mock_cache):
    """Test embedding with mixed cache hits and misses."""

    def mock_get(key):
        # First text is cached, second is not
        if b"first" in key.encode():
            cached_emb = np.random.randn(384).astype(np.float32)
            return cached_emb.tobytes()
        return None

    embedder_with_mock_cache.cache.get = mock_get

    texts = ["first text", "second text"]
    embeddings = embedder_with_mock_cache.embed(texts)

    # Should return both embeddings in correct order
    assert embeddings.shape == (2, embedder_with_mock_cache.dimension)


def test_embed_batch_size(embedder):
    """Test that batch_size parameter is respected."""
    assert embedder.batch_size == 32

    # Create with custom batch size
    custom_embedder = EmbeddingService(batch_size=16, use_cache=False)
    assert custom_embedder.batch_size == 16


def test_embed_show_progress(embedder):
    """Test embedding with progress bar enabled."""
    texts = ["test"] * 5
    # Should not raise error with show_progress=True
    embeddings = embedder.embed(texts, show_progress=True)
    assert embeddings.shape == (5, embedder.dimension)


def test_embed_deterministic(embedder):
    """Test that same input produces same embedding."""
    text = "consistent text"
    emb1 = embedder.embed([text])
    emb2 = embedder.embed([text])

    assert np.allclose(emb1, emb2)


def test_embed_different_texts(embedder):
    """Test that different texts produce different embeddings."""
    text1 = ["machine learning"]
    text2 = ["deep learning"]

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)

    # Embeddings should be different
    assert not np.allclose(emb1, emb2)


def test_embed_query_consistency(embedder):
    """Test that embed_query matches embed."""
    text = "test query"

    emb_batch = embedder.embed([text])
    emb_query = embedder.embed_query(text)

    # Should produce same embedding
    assert np.allclose(emb_batch[0], emb_query)


def test_model_dimension(embedder):
    """Test that model dimension is correct."""
    # bge-small-en-v1.5 has 384 dimensions
    assert embedder.dimension == 384


def test_custom_model():
    """Test initialization with different model."""
    # Note: This test uses the default model since we can't easily test
    # with a different model without downloading it
    embedder = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5", use_cache=False
    )
    assert embedder.dimension > 0


def test_embed_special_characters(embedder):
    """Test embedding text with special characters."""
    texts = ["Hello! How are you?", "Test @#$% special", "Unicode: café résumé"]
    embeddings = embedder.embed(texts)

    assert embeddings.shape == (3, embedder.dimension)


def test_embed_long_text(embedder):
    """Test embedding very long text."""
    long_text = "word " * 500  # Very long text
    embeddings = embedder.embed([long_text])

    assert embeddings.shape == (1, embedder.dimension)


def test_embed_order_preservation(embedder):
    """Test that embedding order matches input order."""
    texts = ["first", "second", "third"]
    embeddings = embedder.embed(texts)

    # Verify by checking each embedding individually
    for i, text in enumerate(texts):
        single_emb = embedder.embed([text])[0]
        assert np.allclose(embeddings[i], single_emb, atol=1e-5)
