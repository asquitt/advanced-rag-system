"""
Unit tests for evaluation metrics.
"""
import pytest
import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)


class TestPrecisionAtK:
    """Test precision@k metric."""

    def test_precision_all_relevant(self):
        """Test when all retrieved documents are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3", "doc4"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_none_relevant(self):
        """Test when no retrieved documents are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_precision_partial(self):
        """Test with partial relevance."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6"}
        # 2 out of 5 are relevant
        assert precision_at_k(retrieved, relevant, 5) == 0.4

    def test_precision_k_less_than_retrieved(self):
        """Test when k is less than retrieved list length."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc5"}
        # Only doc1 in first 3
        assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)

    def test_precision_zero_k(self):
        """Test with k=0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}
        assert precision_at_k(retrieved, relevant, 0) == 0.0

    def test_precision_empty_retrieved(self):
        """Test with empty retrieved list."""
        retrieved = []
        relevant = {"doc1", "doc2"}
        assert precision_at_k(retrieved, relevant, 5) == 0.0


class TestRecallAtK:
    """Test recall@k metric."""

    def test_recall_all_relevant_retrieved(self):
        """Test when all relevant documents are retrieved."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, 4) == 1.0

    def test_recall_none_retrieved(self):
        """Test when no relevant documents are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_recall_partial(self):
        """Test with partial recall."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6"}
        # 2 out of 3 relevant docs retrieved
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)

    def test_recall_k_less_than_retrieved(self):
        """Test when k is less than retrieved list length."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc5"}
        # Only doc1 in first 3
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_recall_empty_relevant(self):
        """Test with empty relevant set."""
        retrieved = ["doc1", "doc2"]
        relevant = set()
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_recall_empty_retrieved(self):
        """Test with empty retrieved list."""
        retrieved = []
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, 5) == 0.0


class TestMeanReciprocalRank:
    """Test MRR metric."""

    def test_mrr_first_position(self):
        """Test when relevant doc is always first."""
        retrieved_lists = [["doc1", "doc2"], ["doc3", "doc4"]]
        relevant_sets = [{"doc1"}, {"doc3"}]
        assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == 1.0

    def test_mrr_no_relevant(self):
        """Test when no relevant documents are retrieved."""
        retrieved_lists = [["doc1", "doc2"], ["doc3", "doc4"]]
        relevant_sets = [{"doc5"}, {"doc6"}]
        assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == 0.0

    def test_mrr_mixed_positions(self):
        """Test with relevant docs at different positions."""
        retrieved_lists = [
            ["doc1", "doc2", "doc3"],  # relevant at position 1
            ["doc4", "doc5", "doc1"],  # relevant at position 3
            ["doc2", "doc1", "doc3"],  # relevant at position 2
        ]
        relevant_sets = [{"doc1", "doc4"}, {"doc1", "doc2"}, {"doc3", "doc5"}]
        # RR: 1/1 + 1/3 + 1/1 = 2.333... / 3 = 0.777...
        expected_mrr = (1.0 + 1 / 3 + 1.0) / 3
        assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == pytest.approx(
            expected_mrr
        )

    def test_mrr_empty_lists(self):
        """Test with empty lists."""
        retrieved_lists = []
        relevant_sets = []
        assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == 0.0

    def test_mrr_single_query(self):
        """Test with single query."""
        retrieved_lists = [["doc1", "doc2", "doc3"]]
        relevant_sets = [{"doc2"}]
        assert mean_reciprocal_rank(retrieved_lists, relevant_sets) == 0.5


class TestNDCGAtK:
    """Test NDCG@k metric."""

    def test_ndcg_perfect_ranking(self):
        """Test with perfect ranking (all relevant docs first)."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}
        assert ndcg_at_k(retrieved, relevant, 5) == 1.0

    def test_ndcg_no_relevant(self):
        """Test when no relevant documents are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_ndcg_partial_ranking(self):
        """Test with imperfect ranking."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6"}
        score = ndcg_at_k(retrieved, relevant, 5)
        # DCG: 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG: 1/log2(2) + 1/log2(3) = 1.0 + 0.630... = 1.630...
        expected_dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4)
        expected_idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        expected_ndcg = expected_dcg / expected_idcg
        assert score == pytest.approx(expected_ndcg)

    def test_ndcg_k_less_than_retrieved(self):
        """Test when k is less than retrieved list length."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc3", "doc4"}
        # Only doc3 in first 3 positions
        score = ndcg_at_k(retrieved, relevant, 3)
        # DCG: 1/log2(4) = 0.5
        # IDCG: 1/log2(2) + 1/log2(3) = 1.630...
        expected_dcg = 1.0 / np.log2(4)
        expected_idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        expected_ndcg = expected_dcg / expected_idcg
        assert score == pytest.approx(expected_ndcg)

    def test_ndcg_empty_relevant(self):
        """Test with empty relevant set."""
        retrieved = ["doc1", "doc2"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0

    def test_ndcg_more_relevant_than_k(self):
        """Test when there are more relevant docs than k."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc4", "doc5", "doc6"}
        # k=3, but 5 relevant docs
        # DCG: 1/log2(2) + 1/log2(3) = 1.630...
        # IDCG: same (only 3 positions available)
        score = ndcg_at_k(retrieved, relevant, 3)
        assert score == pytest.approx(1.0)

    def test_ndcg_k_zero(self):
        """Test with k=0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}
        assert ndcg_at_k(retrieved, relevant, 0) == 0.0
