"""
Shared pytest fixtures for testing.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import List


@pytest.fixture
def sample_texts():
    """Sample document texts for testing."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text data",
        "Computer vision focuses on image and video analysis",
        "Reinforcement learning uses rewards and penalties",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing (384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing."""
    np.random.seed(42)
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = MagicMock()

    # Mock get_collections
    mock.get_collections.return_value = Mock(collections=[])

    # Mock create_collection
    mock.create_collection.return_value = None

    # Mock upsert
    mock.upsert.return_value = None

    # Mock search results
    mock_result = Mock()
    mock_result.payload = {"text": "Test document", "source": "test"}
    mock_result.score = 0.95
    mock.search.return_value = [mock_result]

    # Mock get_collection
    mock_collection = Mock()
    mock_collection.points_count = 5
    mock.get_collection.return_value = mock_collection

    return mock


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock = MagicMock()

    # Mock get/set operations
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True

    # Mock ping for health check
    mock.ping.return_value = True

    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    mock = MagicMock()

    # Mock encode to return random embeddings
    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        np.random.seed(42)
        return np.random.randn(len(texts), 384).astype(np.float32)

    mock.encode = mock_encode

    return mock


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for testing."""
    mock = MagicMock()

    # Mock predict to return scores
    def mock_predict(pairs, **kwargs):
        return np.array([0.9 - i * 0.1 for i in range(len(pairs))])

    mock.predict = mock_predict

    return mock


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents for testing."""
    return ["doc1", "doc2", "doc3", "doc4", "doc5"]


@pytest.fixture
def sample_relevant_docs():
    """Sample relevant documents for testing."""
    return {"doc1", "doc3", "doc6"}


@pytest.fixture
def sample_search_results():
    """Sample search results with scores."""
    return [
        {"text": "Machine learning is AI", "score": 0.95, "metadata": {}},
        {"text": "Deep learning networks", "score": 0.88, "metadata": {}},
        {"text": "NLP for text", "score": 0.75, "metadata": {}},
    ]
