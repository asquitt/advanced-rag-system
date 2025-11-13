"""
Unit tests for HybridRetriever.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetriever:
    """Test HybridRetriever class."""

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_initialization(
        self, mock_embedding_service, mock_vector_store, mock_sparse_retriever
    ):
        """Test hybrid retriever initialization."""
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedding_service.return_value = mock_embedder

        retriever = HybridRetriever(collection_name="test", dense_weight=0.7)

        assert retriever.dense_weight == 0.7
        assert retriever.documents == []
        mock_embedding_service.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_sparse_retriever.assert_called_once()

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_index_documents(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_texts,
        sample_embeddings,
    ):
        """Test indexing documents."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed.return_value = sample_embeddings
        mock_embedding_service.return_value = mock_embedder

        mock_vector = Mock()
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever()

        # Index documents
        retriever.index(sample_texts)

        # Verify calls
        mock_embedder.embed.assert_called_once_with(sample_texts, show_progress=True)
        mock_vector.add.assert_called_once_with(sample_texts, sample_embeddings, None)
        mock_sparse.index.assert_called_once_with(sample_texts)
        assert retriever.documents == sample_texts

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_index_with_metadata(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_texts,
        sample_embeddings,
    ):
        """Test indexing documents with metadata."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed.return_value = sample_embeddings
        mock_embedding_service.return_value = mock_embedder

        mock_vector = Mock()
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever()

        metadata = [{"source": f"doc_{i}"} for i in range(len(sample_texts))]

        # Index with metadata
        retriever.index(sample_texts, metadata)

        # Verify metadata passed to vector store
        mock_vector.add.assert_called_once_with(
            sample_texts, sample_embeddings, metadata
        )

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_hybrid(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test hybrid retrieval combining dense and sparse."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        # Mock dense results
        mock_vector = Mock()
        mock_vector.search.return_value = [
            {"text": "Doc A", "score": 0.9},
            {"text": "Doc B", "score": 0.8},
            {"text": "Doc C", "score": 0.7},
        ]
        mock_vector_store.return_value = mock_vector

        # Mock sparse results
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            ("Doc B", 10.0),
            ("Doc C", 8.0),
            ("Doc D", 6.0),
        ]
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever(dense_weight=0.5)

        # Retrieve
        results = retriever.retrieve("test query", top_k=3)

        # Verify calls
        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_vector.search.assert_called_once_with(sample_query_embedding, limit=6)
        mock_sparse.retrieve.assert_called_once_with("test query", top_k=6)

        # Check results
        assert len(results) <= 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_score_normalization(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test that scores are normalized correctly."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        # Mock dense results with different scales
        mock_vector = Mock()
        mock_vector.search.return_value = [
            {"text": "Doc A", "score": 1.0},
            {"text": "Doc B", "score": 0.5},
        ]
        mock_vector_store.return_value = mock_vector

        # Mock sparse results with different scales
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            ("Doc A", 100.0),
            ("Doc C", 50.0),
        ]
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever(dense_weight=0.5)

        results = retriever.retrieve("test", top_k=5)

        # Doc A should have highest score (appears in both with high scores)
        assert results[0]["text"] == "Doc A"

        # All scores should be > 0
        assert all(r["score"] > 0 for r in results)

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_dense_weight_zero(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test retrieval with dense_weight=0 (sparse only)."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        mock_vector = Mock()
        mock_vector.search.return_value = [{"text": "Dense Doc", "score": 0.9}]
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [("Sparse Doc", 10.0)]
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever(dense_weight=0.0)

        results = retriever.retrieve("test", top_k=1)

        # Should prioritize sparse results only
        assert results[0]["text"] == "Sparse Doc"

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_dense_weight_one(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test retrieval with dense_weight=1 (dense only)."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        mock_vector = Mock()
        mock_vector.search.return_value = [{"text": "Dense Doc", "score": 0.9}]
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [("Sparse Doc", 10.0)]
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever(dense_weight=1.0)

        results = retriever.retrieve("test", top_k=1)

        # Should prioritize dense results only
        assert results[0]["text"] == "Dense Doc"

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_empty_results(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test retrieval with empty results."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        mock_vector = Mock()
        mock_vector.search.return_value = []
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = []
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever()

        results = retriever.retrieve("test", top_k=5)

        # Should return empty list
        assert results == []

    @patch("src.retrieval.hybrid_retriever.SparseRetriever")
    @patch("src.retrieval.hybrid_retriever.VectorStore")
    @patch("src.retrieval.hybrid_retriever.EmbeddingService")
    def test_retrieve_top_k_limiting(
        self,
        mock_embedding_service,
        mock_vector_store,
        mock_sparse_retriever,
        sample_query_embedding,
    ):
        """Test that top_k correctly limits results."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.embed_query.return_value = sample_query_embedding
        mock_embedding_service.return_value = mock_embedder

        # Return many results
        mock_vector = Mock()
        mock_vector.search.return_value = [
            {"text": f"Dense Doc {i}", "score": 0.9 - i * 0.1} for i in range(10)
        ]
        mock_vector_store.return_value = mock_vector

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            (f"Sparse Doc {i}", 10.0 - i) for i in range(10)
        ]
        mock_sparse_retriever.return_value = mock_sparse

        retriever = HybridRetriever()

        # Request only top 3
        results = retriever.retrieve("test", top_k=3)

        assert len(results) == 3
