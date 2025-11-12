"""
Evaluation metrics for retrieval quality.
"""
from typing import List, Set
import numpy as np


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate precision@k.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        Precision at k
    """
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_retrieved / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate recall@k.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        Recall at k
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant)
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(retrieved_lists: List[List[str]], relevant_sets: List[Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_lists: List of retrieved document lists for each query
        relevant_sets: List of relevant document sets for each query
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        k: Cut-off rank
        
    Returns:
        NDCG at k
    """
    retrieved_at_k = retrieved[:k]
    
    # DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved_at_k, 1):
        if doc in relevant:
            dcg += 1.0 / np.log2(i + 1)
    
    # IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    # Test metrics
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc6"}
    
    print("Evaluation Metrics Test")
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}\n")
    
    print(f"Precision@5: {precision_at_k(retrieved, relevant, 5):.3f}")
    print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.3f}")
    print(f"NDCG@5: {ndcg_at_k(retrieved, relevant, 5):.3f}")
    
    # MRR example
    retrieved_lists = [
        ["doc1", "doc2", "doc3"],
        ["doc4", "doc5", "doc1"],
        ["doc2", "doc1", "doc3"]
    ]
    relevant_sets = [
        {"doc1", "doc4"},
        {"doc1", "doc2"},
        {"doc3", "doc5"}
    ]
    print(f"\nMRR: {mean_reciprocal_rank(retrieved_lists, relevant_sets):.3f}")
