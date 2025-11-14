"""
Unit tests for SparseRetriever.
"""
import pytest
import numpy as np
from src.retrieval.sparse_retriever import SparseRetriever


class TestSparseRetriever:
    """Test SparseRetriever class."""

    def test_initialization(self):
        """Test sparse retriever initialization."""
        retriever = SparseRetriever()
        assert retriever.bm25 is None
        assert retriever.documents == []

    def test_index_documents(self, sample_texts):
        """Test indexing documents."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        assert retriever.bm25 is not None
        assert retriever.documents == sample_texts
        assert len(retriever.documents) == len(sample_texts)

    def test_index_empty_documents(self):
        """Test indexing empty document list."""
        retriever = SparseRetriever()
        retriever.index([])

        assert retriever.bm25 is not None
        assert retriever.documents == []

    def test_retrieve_before_indexing(self):
        """Test retrieval before indexing raises error."""
        retriever = SparseRetriever()

        with pytest.raises(ValueError, match="Must index documents first"):
            retriever.retrieve("test query")

    def test_retrieve_basic(self, sample_texts):
        """Test basic retrieval."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        results = retriever.retrieve("machine learning", top_k=3)

        # Should return list of tuples (document, score)
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # Check that results are from the indexed documents
        for doc, score in results:
            assert doc in sample_texts
            assert isinstance(score, (float, np.floating))

    def test_retrieve_top_k(self, sample_texts):
        """Test retrieval with different top_k values."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        # Test different k values
        for k in [1, 3, 5]:
            results = retriever.retrieve("neural networks", top_k=k)
            assert len(results) == min(k, len(sample_texts))

    def test_retrieve_more_than_available(self, sample_texts):
        """Test retrieval when top_k exceeds document count."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        # Request more documents than available
        results = retriever.retrieve("test", top_k=100)
        assert len(results) == len(sample_texts)

    def test_retrieve_relevance_order(self):
        """Test that results are ordered by relevance."""
        retriever = SparseRetriever()

        docs = [
            "Python programming language",
            "Java programming language",
            "Machine learning with Python",
            "Deep learning neural networks",
            "Data science and analytics",
        ]

        retriever.index(docs)

        # Query should match first doc best
        results = retriever.retrieve("Python programming", top_k=3)

        # First result should contain "Python programming"
        top_doc, top_score = results[0]
        assert "Python programming" in top_doc

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_case_insensitive(self):
        """Test that retrieval is case-insensitive."""
        retriever = SparseRetriever()

        docs = [
            "Machine Learning Algorithms",
            "Deep Learning Networks",
            "Natural Language Processing",
        ]

        retriever.index(docs)

        # Test with different cases
        results_lower = retriever.retrieve("machine learning", top_k=1)
        results_upper = retriever.retrieve("MACHINE LEARNING", top_k=1)
        results_mixed = retriever.retrieve("Machine Learning", top_k=1)

        # All should return the same top document
        assert results_lower[0][0] == results_upper[0][0] == results_mixed[0][0]

    def test_retrieve_with_single_word_query(self, sample_texts):
        """Test retrieval with single word query."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        results = retriever.retrieve("learning", top_k=3)

        assert len(results) == 3
        # Should find documents containing "learning"
        for doc, score in results:
            assert isinstance(doc, str)
            assert isinstance(score, (float, np.floating))

    def test_retrieve_with_multi_word_query(self, sample_texts):
        """Test retrieval with multi-word query."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        results = retriever.retrieve("machine learning algorithms", top_k=3)

        assert len(results) == 3
        # Scores should be non-negative
        for doc, score in results:
            assert score >= 0

    def test_retrieve_empty_query(self, sample_texts):
        """Test retrieval with empty query."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        results = retriever.retrieve("", top_k=3)

        # Should still return results (BM25 handles empty queries)
        assert len(results) == 3

    def test_retrieve_non_matching_query(self, sample_texts):
        """Test retrieval with query that doesn't match well."""
        retriever = SparseRetriever()
        retriever.index(sample_texts)

        # Query with words not in documents
        results = retriever.retrieve("quantum physics astronomy", top_k=3)

        # Should still return results, but with lower scores
        assert len(results) == 3
        # Scores should be relatively low
        for _, score in results:
            assert score >= 0

    def test_multiple_indexing(self, sample_texts):
        """Test indexing multiple times overwrites previous index."""
        retriever = SparseRetriever()

        # First index
        retriever.index(sample_texts[:3])
        assert len(retriever.documents) == 3

        # Second index should overwrite
        retriever.index(sample_texts)
        assert len(retriever.documents) == len(sample_texts)

        results = retriever.retrieve("test", top_k=5)
        assert len(results) == len(sample_texts)

    def test_tokenization(self):
        """Test that documents are tokenized correctly."""
        retriever = SparseRetriever()

        docs = ["This is a test document", "Another test"]

        retriever.index(docs)

        # BM25 should tokenize by splitting on spaces and lowercasing
        assert retriever.documents == docs
        assert retriever.bm25 is not None
