"""
Unit tests for RetrievalEvaluator.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, MagicMock
from src.evaluation.evaluator import RetrievalEvaluator


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    retriever = Mock()

    def mock_retrieve(query, top_k=5):
        # Return different results based on query
        if "machine learning" in query.lower():
            return [
                {"text": "Machine learning is AI", "score": 0.95},
                {"text": "Deep learning networks", "score": 0.88},
                {"text": "NLP for text", "score": 0.75},
                {"text": "Computer vision", "score": 0.65},
                {"text": "Data science", "score": 0.55},
            ][:top_k]
        else:
            return [
                {"text": "Python programming", "score": 0.92},
                {"text": "Java language", "score": 0.85},
                {"text": "JavaScript web", "score": 0.78},
                {"text": "C++ systems", "score": 0.70},
                {"text": "Ruby on Rails", "score": 0.60},
            ][:top_k]

    retriever.retrieve = mock_retrieve
    return retriever


class TestRetrievalEvaluator:
    """Test RetrievalEvaluator class."""

    def test_initialization(self, mock_retriever):
        """Test evaluator initialization."""
        evaluator = RetrievalEvaluator(mock_retriever)
        assert evaluator.retriever == mock_retriever

    def test_evaluate_single_query(self, mock_retriever):
        """Test evaluation with single query."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Machine learning is AI", "Deep learning networks"}]

        results = evaluator.evaluate(queries, relevant_docs, k_values=[3, 5])

        # Check structure
        assert "mrr" in results
        assert "precision" in results
        assert "recall" in results
        assert "ndcg" in results

        # Check precision keys
        assert "P@3" in results["precision"]
        assert "P@5" in results["precision"]

        # Check that MRR is calculated (should be 1.0 as first doc is relevant)
        assert results["mrr"] == 1.0

        # Check that precision is reasonable (2 out of first 3 are relevant)
        assert results["precision"]["P@3"] == pytest.approx(2 / 3)

    def test_evaluate_multiple_queries(self, mock_retriever):
        """Test evaluation with multiple queries."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning", "programming languages"]
        relevant_docs = [
            {"Machine learning is AI", "Deep learning networks"},
            {"Python programming", "Java language"},
        ]

        results = evaluator.evaluate(queries, relevant_docs, k_values=[5])

        # Check that metrics are averaged across queries
        assert "P@5" in results["precision"]
        assert "R@5" in results["recall"]
        assert "NDCG@5" in results["ndcg"]

        # MRR should be 1.0 (both queries have relevant doc at rank 1)
        assert results["mrr"] == 1.0

    def test_evaluate_default_k_values(self, mock_retriever):
        """Test evaluation with default k values."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Machine learning is AI"}]

        # Don't specify k_values, should use default
        results = evaluator.evaluate(queries, relevant_docs)

        # Check default k values [1, 3, 5, 10]
        assert "P@1" in results["precision"]
        assert "P@3" in results["precision"]
        assert "P@5" in results["precision"]
        assert "P@10" in results["precision"]

    def test_evaluate_no_relevant_docs(self, mock_retriever):
        """Test evaluation when no relevant docs are retrieved."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Unrelated document"}]

        results = evaluator.evaluate(queries, relevant_docs, k_values=[5])

        # All metrics should be 0
        assert results["mrr"] == 0.0
        assert results["precision"]["P@5"] == 0.0
        assert results["recall"]["R@5"] == 0.0
        assert results["ndcg"]["NDCG@5"] == 0.0

    def test_evaluate_partial_relevance(self, mock_retriever):
        """Test evaluation with partial relevance."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Machine learning is AI", "Unrelated doc"}]

        results = evaluator.evaluate(queries, relevant_docs, k_values=[5])

        # Should have partial metrics
        # Precision: 1/5 = 0.2
        assert results["precision"]["P@5"] == 0.2
        # Recall: 1/2 = 0.5
        assert results["recall"]["R@5"] == 0.5
        # MRR should be 1.0 (relevant doc at rank 1)
        assert results["mrr"] == 1.0

    def test_save_results(self, mock_retriever):
        """Test saving results to file."""
        evaluator = RetrievalEvaluator(mock_retriever)

        results = {
            "mrr": 0.85,
            "precision": {"P@5": 0.75},
            "recall": {"R@5": 0.80},
            "ndcg": {"NDCG@5": 0.82},
        }

        # Use temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            evaluator.save_results(results, temp_file)

            # Verify file was created and contains correct data
            assert os.path.exists(temp_file)

            with open(temp_file, "r") as f:
                loaded_results = json.load(f)

            assert loaded_results == results
            assert loaded_results["mrr"] == 0.85
            assert loaded_results["precision"]["P@5"] == 0.75

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_evaluate_empty_queries(self, mock_retriever):
        """Test evaluation with empty query list."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = []
        relevant_docs = []

        # Should handle empty lists gracefully
        # This will cause division by zero, so we expect it to fail
        # or return NaN values
        with pytest.raises(ZeroDivisionError):
            results = evaluator.evaluate(queries, relevant_docs, k_values=[5])

    def test_evaluate_different_k_values(self, mock_retriever):
        """Test evaluation with various k values."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Machine learning is AI", "Deep learning networks"}]

        results = evaluator.evaluate(queries, relevant_docs, k_values=[1, 2, 3])

        # Check all k values are present
        assert "P@1" in results["precision"]
        assert "P@2" in results["precision"]
        assert "P@3" in results["precision"]

        # Precision@1 should be 1.0 (first doc is relevant)
        assert results["precision"]["P@1"] == 1.0
        # Precision@2 should be 1.0 (both first 2 docs are relevant)
        assert results["precision"]["P@2"] == 1.0

    def test_evaluate_uses_text_as_id(self, mock_retriever):
        """Test that evaluator uses 'text' field as document ID."""
        evaluator = RetrievalEvaluator(mock_retriever)

        queries = ["machine learning"]
        relevant_docs = [{"Machine learning is AI"}]

        # The mock returns results with "text" field
        # Evaluator should use this for comparison
        results = evaluator.evaluate(queries, relevant_docs, k_values=[1])

        # First result matches relevant doc exactly
        assert results["precision"]["P@1"] == 1.0
