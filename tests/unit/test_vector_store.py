"""
Unit tests for VectorStore.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.retrieval.vector_store import VectorStore


class TestVectorStore:
    """Test VectorStore class."""

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_initialization(self, mock_qdrant_class, mock_qdrant_client):
        """Test vector store initialization."""
        mock_qdrant_class.return_value = mock_qdrant_client

        store = VectorStore(
            collection_name="test_collection",
            host="localhost",
            port=6333,
            vector_size=384,
        )

        # Check initialization
        assert store.collection_name == "test_collection"
        assert store.vector_size == 384
        mock_qdrant_class.assert_called_once_with(host="localhost", port=6333)

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_create_collection_new(self, mock_qdrant_class, mock_qdrant_client):
        """Test creating a new collection."""
        mock_qdrant_class.return_value = mock_qdrant_client

        # Mock get_collections to return empty list (collection doesn't exist)
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        store = VectorStore(collection_name="new_collection")

        # Should create the collection
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "new_collection"

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_create_collection_existing(self, mock_qdrant_class, mock_qdrant_client):
        """Test when collection already exists."""
        mock_qdrant_class.return_value = mock_qdrant_client

        # Mock get_collections to return existing collection
        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_qdrant_client.get_collections.return_value = Mock(
            collections=[mock_collection]
        )

        store = VectorStore(collection_name="existing_collection")

        # Should NOT create the collection
        mock_qdrant_client.create_collection.assert_not_called()

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_documents(
        self, mock_qdrant_class, mock_qdrant_client, sample_texts, sample_embeddings
    ):
        """Test adding documents to vector store."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        store = VectorStore()

        # Add documents
        store.add(sample_texts, sample_embeddings)

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args

        assert call_args[1]["collection_name"] == "documents"
        points = call_args[1]["points"]
        assert len(points) == len(sample_texts)

        # Check first point structure
        assert "id" in points[0].__dict__ or hasattr(points[0], "id")
        assert "vector" in points[0].__dict__ or hasattr(points[0], "vector")
        assert "payload" in points[0].__dict__ or hasattr(points[0], "payload")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_documents_with_metadata(
        self, mock_qdrant_class, mock_qdrant_client, sample_texts, sample_embeddings
    ):
        """Test adding documents with metadata."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        store = VectorStore()

        # Add metadata
        metadata = [{"source": f"doc_{i}"} for i in range(len(sample_texts))]

        store.add(sample_texts, sample_embeddings, metadata)

        # Verify metadata is included
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]["points"]

        # Check that payload includes both text and metadata
        for i, point in enumerate(points):
            assert point.payload["text"] == sample_texts[i]
            assert point.payload["source"] == f"doc_{i}"

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_search(
        self, mock_qdrant_class, mock_qdrant_client, sample_query_embedding
    ):
        """Test searching for similar documents."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        # Mock search results
        mock_result1 = Mock()
        mock_result1.payload = {"text": "Result 1", "source": "test"}
        mock_result1.score = 0.95

        mock_result2 = Mock()
        mock_result2.payload = {"text": "Result 2"}
        mock_result2.score = 0.88

        mock_qdrant_client.search.return_value = [mock_result1, mock_result2]

        store = VectorStore()

        # Search
        results = store.search(sample_query_embedding, limit=2)

        # Verify search was called correctly
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["collection_name"] == "documents"
        assert call_args[1]["limit"] == 2

        # Check results structure
        assert len(results) == 2
        assert results[0]["text"] == "Result 1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"source": "test"}
        assert results[1]["text"] == "Result 2"
        assert results[1]["score"] == 0.88
        assert results[1]["metadata"] == {}

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_search_with_numpy_array(
        self, mock_qdrant_class, mock_qdrant_client, sample_query_embedding
    ):
        """Test search handles numpy arrays correctly."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.return_value = []

        store = VectorStore()

        # Search with numpy array
        results = store.search(sample_query_embedding, limit=5)

        # Verify numpy array was converted to list
        call_args = mock_qdrant_client.search.call_args
        query_vector = call_args[1]["query_vector"]
        assert isinstance(query_vector, list)

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_search_with_list(self, mock_qdrant_class, mock_qdrant_client):
        """Test search handles lists correctly."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.search.return_value = []

        store = VectorStore()

        # Search with list
        query_list = [0.1] * 384
        results = store.search(query_list, limit=5)

        # Verify list was used directly
        call_args = mock_qdrant_client.search.call_args
        query_vector = call_args[1]["query_vector"]
        assert query_vector == query_list

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_count(self, mock_qdrant_class, mock_qdrant_client):
        """Test getting document count."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        # Mock get_collection
        mock_collection_info = Mock()
        mock_collection_info.points_count = 42
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        store = VectorStore(collection_name="test_collection")

        count = store.count()

        assert count == 42
        mock_qdrant_client.get_collection.assert_called_with("test_collection")

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_create_collection_error_handling(self, mock_qdrant_class):
        """Test error handling in collection creation."""
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        # Mock get_collections to raise an exception
        mock_client.get_collections.side_effect = Exception("Connection error")

        # Should handle exception gracefully
        store = VectorStore()

        # Verify exception was caught and handled
        assert store.collection_name == "documents"

    @patch("src.retrieval.vector_store.QdrantClient")
    def test_add_with_numpy_embeddings(
        self, mock_qdrant_class, mock_qdrant_client, sample_texts
    ):
        """Test adding documents with numpy embeddings."""
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])

        store = VectorStore()

        # Create numpy embeddings
        embeddings = np.random.randn(len(sample_texts), 384).astype(np.float32)

        store.add(sample_texts, embeddings)

        # Verify embeddings were converted to lists
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]["points"]

        for point in points:
            assert isinstance(point.vector, list)
