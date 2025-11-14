"""
Unit tests for RetrievalPipeline.
"""
import pytest
from unittest.mock import Mock, patch
from src.retrieval.retrieval_pipeline import RetrievalPipeline


class TestRetrievalPipeline:
    """Test RetrievalPipeline class."""

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_initialization_with_reranking(
        self, mock_hybrid_retriever, mock_reranker
    ):
        """Test pipeline initialization with reranking enabled."""
        pipeline = RetrievalPipeline(
            collection_name="test", use_reranking=True, dense_weight=0.7
        )

        mock_hybrid_retriever.assert_called_once_with(
            collection_name="test", dense_weight=0.7
        )
        mock_reranker.assert_called_once()
        assert pipeline.use_reranking is True

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_initialization_without_reranking(self, mock_hybrid_retriever):
        """Test pipeline initialization with reranking disabled."""
        pipeline = RetrievalPipeline(use_reranking=False)

        assert pipeline.use_reranking is False
        assert not hasattr(pipeline, "reranker") or pipeline.reranker is None

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_index(self, mock_hybrid_retriever, sample_texts):
        """Test indexing documents."""
        mock_retriever = Mock()
        mock_hybrid_retriever.return_value = mock_retriever

        pipeline = RetrievalPipeline(use_reranking=False)

        pipeline.index(sample_texts)

        mock_retriever.index.assert_called_once_with(sample_texts, None)

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_index_with_metadata(self, mock_hybrid_retriever, sample_texts):
        """Test indexing documents with metadata."""
        mock_retriever = Mock()
        mock_hybrid_retriever.return_value = mock_retriever

        pipeline = RetrievalPipeline(use_reranking=False)

        metadata = [{"source": f"doc_{i}"} for i in range(len(sample_texts))]
        pipeline.index(sample_texts, metadata)

        mock_retriever.index.assert_called_once_with(sample_texts, metadata)

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_without_reranking(self, mock_hybrid_retriever):
        """Test retrieval without reranking."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {"text": "Doc 1", "score": 0.9},
            {"text": "Doc 2", "score": 0.8},
            {"text": "Doc 3", "score": 0.7},
        ]
        mock_hybrid_retriever.return_value = mock_retriever

        pipeline = RetrievalPipeline(use_reranking=False)

        results = pipeline.retrieve("test query", top_k=3)

        # Should call hybrid retriever with top_k=3
        mock_retriever.retrieve.assert_called_once_with("test query", top_k=3)

        assert len(results) == 3
        assert results[0]["text"] == "Doc 1"

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_with_reranking(self, mock_hybrid_retriever, mock_reranker):
        """Test retrieval with reranking."""
        # Mock hybrid retriever
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {"text": "Doc 1", "score": 0.9},
            {"text": "Doc 2", "score": 0.8},
            {"text": "Doc 3", "score": 0.7},
            {"text": "Doc 4", "score": 0.6},
        ]
        mock_hybrid_retriever.return_value = mock_retriever

        # Mock reranker
        mock_rerank_instance = Mock()
        mock_rerank_instance.rerank.return_value = [
            ("Doc 2", 0.95),
            ("Doc 1", 0.85),
            ("Doc 3", 0.75),
        ]
        mock_reranker.return_value = mock_rerank_instance

        pipeline = RetrievalPipeline(use_reranking=True)

        results = pipeline.retrieve("test query", top_k=3, rerank_top_k=20)

        # Should retrieve rerank_top_k results first
        mock_retriever.retrieve.assert_called_once_with("test query", top_k=20)

        # Should rerank with correct documents
        mock_rerank_instance.rerank.assert_called_once()
        call_args = mock_rerank_instance.rerank.call_args
        assert call_args[0][0] == "test query"
        assert call_args[0][1] == ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
        assert call_args[1]["top_k"] == 3

        # Check reranked results
        assert len(results) == 3
        assert results[0]["text"] == "Doc 2"
        assert results[0]["score"] == 0.95

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_with_reranking_empty_results(
        self, mock_hybrid_retriever, mock_reranker
    ):
        """Test retrieval with reranking when no results found."""
        # Mock hybrid retriever returning empty results
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_hybrid_retriever.return_value = mock_retriever

        mock_rerank_instance = Mock()
        mock_reranker.return_value = mock_rerank_instance

        pipeline = RetrievalPipeline(use_reranking=True)

        results = pipeline.retrieve("test query", top_k=3)

        # Should not call reranker if no results
        mock_rerank_instance.rerank.assert_not_called()
        assert results == []

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_limits_top_k(self, mock_hybrid_retriever):
        """Test that top_k correctly limits results."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {"text": f"Doc {i}", "score": 1.0 - i * 0.1} for i in range(10)
        ]
        mock_hybrid_retriever.return_value = mock_retriever

        pipeline = RetrievalPipeline(use_reranking=False)

        results = pipeline.retrieve("test", top_k=3)

        # Should only return 3 results
        assert len(results) == 3

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_rerank_top_k_default(
        self, mock_hybrid_retriever, mock_reranker
    ):
        """Test default rerank_top_k value."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {"text": f"Doc {i}", "score": 0.9} for i in range(5)
        ]
        mock_hybrid_retriever.return_value = mock_retriever

        mock_rerank_instance = Mock()
        mock_rerank_instance.rerank.return_value = [
            (f"Doc {i}", 0.8) for i in range(5)
        ]
        mock_reranker.return_value = mock_rerank_instance

        pipeline = RetrievalPipeline(use_reranking=True)

        # Don't specify rerank_top_k
        results = pipeline.retrieve("test", top_k=5)

        # Should use default rerank_top_k=20
        mock_retriever.retrieve.assert_called_once_with("test", top_k=20)

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_score_conversion_to_float(
        self, mock_hybrid_retriever, mock_reranker
    ):
        """Test that scores are converted to float."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"text": "Doc 1", "score": 0.9}]
        mock_hybrid_retriever.return_value = mock_retriever

        mock_rerank_instance = Mock()
        # Return numpy float
        import numpy as np

        mock_rerank_instance.rerank.return_value = [("Doc 1", np.float32(0.95))]
        mock_reranker.return_value = mock_rerank_instance

        pipeline = RetrievalPipeline(use_reranking=True)

        results = pipeline.retrieve("test", top_k=1)

        # Score should be converted to Python float
        assert isinstance(results[0]["score"], float)
        assert results[0]["score"] == pytest.approx(0.95)

    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_default_parameters(self, mock_hybrid_retriever):
        """Test default initialization parameters."""
        pipeline = RetrievalPipeline()

        # Check defaults
        mock_hybrid_retriever.assert_called_once_with(
            collection_name="documents", dense_weight=0.5
        )
        assert pipeline.use_reranking is True

    @patch("src.retrieval.retrieval_pipeline.Reranker")
    @patch("src.retrieval.retrieval_pipeline.HybridRetriever")
    def test_retrieve_preserves_document_content(
        self, mock_hybrid_retriever, mock_reranker
    ):
        """Test that document content is preserved through reranking."""
        # Documents with special content
        special_docs = [
            {"text": "Doc with special chars: !@#$%", "score": 0.9},
            {"text": "Doc with unicode: café résumé", "score": 0.8},
        ]

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = special_docs
        mock_hybrid_retriever.return_value = mock_retriever

        mock_rerank_instance = Mock()
        mock_rerank_instance.rerank.return_value = [
            ("Doc with unicode: café résumé", 0.95),
            ("Doc with special chars: !@#$%", 0.85),
        ]
        mock_reranker.return_value = mock_rerank_instance

        pipeline = RetrievalPipeline(use_reranking=True)

        results = pipeline.retrieve("test", top_k=2)

        # Check content is preserved
        assert results[0]["text"] == "Doc with unicode: café résumé"
        assert results[1]["text"] == "Doc with special chars: !@#$%"
