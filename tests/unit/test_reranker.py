"""
Unit tests for Reranker.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.retrieval.reranker import Reranker


class TestReranker:
    """Test Reranker class."""

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_initialization(self, mock_cross_encoder):
        """Test reranker initialization."""
        reranker = Reranker(model_name="test-model")

        mock_cross_encoder.assert_called_once_with("test-model")
        assert reranker.model is not None

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_initialization_default_model(self, mock_cross_encoder):
        """Test reranker initialization with default model."""
        reranker = Reranker()

        mock_cross_encoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_basic(self, mock_cross_encoder, mock_cross_encoder):
        """Test basic reranking."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.9, 0.7, 0.6, 0.5])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        query = "machine learning"
        documents = [
            "Doc 1 about ML",
            "Doc 2 about ML",
            "Doc 3 about NLP",
            "Doc 4 about CV",
            "Doc 5 about data",
        ]

        results = reranker.rerank(query, documents, top_k=3)

        # Verify predict was called correctly
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 5
        assert all(pair[0] == query for pair in pairs)
        assert [pair[1] for pair in pairs] == documents

        # Check results
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # Results should be sorted by score (descending)
        doc, score = results[0]
        assert doc == "Doc 2 about ML"  # highest score (0.9)
        assert score == 0.9

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_ordering(self, mock_cross_encoder, mock_cross_encoder):
        """Test that results are ordered by score."""
        # Setup mock with specific scores
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.9, 0.5, 0.1])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        documents = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]
        results = reranker.rerank("test", documents, top_k=5)

        # Check ordering (should be C, B, D, A, E)
        expected_order = [
            ("Doc C", 0.9),
            ("Doc B", 0.7),
            ("Doc D", 0.5),
            ("Doc A", 0.3),
            ("Doc E", 0.1),
        ]

        for i, (expected_doc, expected_score) in enumerate(expected_order):
            doc, score = results[i]
            assert doc == expected_doc
            assert score == pytest.approx(expected_score)

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_top_k(self, mock_cross_encoder, mock_cross_encoder):
        """Test that top_k limits results correctly."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        documents = ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]

        # Test different top_k values
        for k in [1, 3, 5]:
            results = reranker.rerank("test", documents, top_k=k)
            assert len(results) == k

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_empty_documents(self, mock_cross_encoder):
        """Test reranking with empty document list."""
        reranker = Reranker()

        results = reranker.rerank("test query", [], top_k=5)

        # Should return empty list
        assert results == []

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_single_document(self, mock_cross_encoder, mock_cross_encoder):
        """Test reranking with single document."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.85])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        documents = ["Single doc"]
        results = reranker.rerank("test", documents, top_k=5)

        # Should return one result
        assert len(results) == 1
        assert results[0][0] == "Single doc"
        assert results[0][1] == 0.85

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_top_k_larger_than_documents(
        self, mock_cross_encoder, mock_cross_encoder
    ):
        """Test when top_k is larger than number of documents."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.4])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = reranker.rerank("test", documents, top_k=10)

        # Should return all 3 documents
        assert len(results) == 3

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_query_document_pairs(self, mock_cross_encoder, mock_cross_encoder):
        """Test that query-document pairs are created correctly."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5, 0.6])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        query = "specific query"
        documents = ["Doc A", "Doc B"]

        reranker.rerank(query, documents, top_k=2)

        # Check pairs passed to predict
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [["specific query", "Doc A"], ["specific query", "Doc B"]]

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_score_types(self, mock_cross_encoder, mock_cross_encoder):
        """Test that scores are numeric types."""
        # Setup mock with different numeric types
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = reranker.rerank("test", documents, top_k=3)

        # All scores should be numeric
        for doc, score in results:
            assert isinstance(score, (int, float, np.number))

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_rerank_preserves_document_content(
        self, mock_cross_encoder, mock_cross_encoder
    ):
        """Test that document content is preserved."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5, 0.6, 0.7])
        mock_cross_encoder.return_value = mock_model

        reranker = Reranker()

        # Documents with special characters and formatting
        documents = [
            "Document with special chars: !@#$%",
            "Document with numbers: 12345",
            "Document with unicode: café résumé",
        ]

        results = reranker.rerank("test", documents, top_k=3)

        # Check that all documents are present in results
        result_docs = [doc for doc, score in results]
        assert set(result_docs) == set(documents)
