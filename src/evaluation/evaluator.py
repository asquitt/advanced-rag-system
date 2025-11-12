"""
Evaluation framework for retrieval system.
"""
from typing import List, Dict, Set
import json
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k
)


class RetrievalEvaluator:
    """Evaluate retrieval system performance."""
    
    def __init__(self, retriever):
        """
        Initialize evaluator.
        
        Args:
            retriever: Retrieval system to evaluate
        """
        self.retriever = retriever
    
    def evaluate(
        self,
        queries: List[str],
        relevant_docs: List[Set[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval system.
        
        Args:
            queries: List of test queries
            relevant_docs: List of relevant document sets for each query
            k_values: K values to evaluate at
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            "precision": {k: [] for k in k_values},
            "recall": {k: [] for k in k_values},
            "ndcg": {k: [] for k in k_values},
            "mrr_scores": []
        }
        
        retrieved_lists = []
        
        # Retrieve for each query
        for query, relevant in zip(queries, relevant_docs):
            retrieved_results = self.retriever.retrieve(query, top_k=max(k_values))
            retrieved_ids = [r['text'] for r in retrieved_results]  # Using text as ID for now
            
            retrieved_lists.append(retrieved_ids)
            
            # Calculate metrics at different k values
            for k in k_values:
                results["precision"][k].append(precision_at_k(retrieved_ids, relevant, k))
                results["recall"][k].append(recall_at_k(retrieved_ids, relevant, k))
                results["ndcg"][k].append(ndcg_at_k(retrieved_ids, relevant, k))
        
        # Calculate MRR
        mrr = mean_reciprocal_rank(retrieved_lists, relevant_docs)
        
        # Aggregate results
        summary = {
            "mrr": mrr,
            "precision": {},
            "recall": {},
            "ndcg": {}
        }
        
        for k in k_values:
            summary["precision"][f"P@{k}"] = sum(results["precision"][k]) / len(queries)
            summary["recall"][f"R@{k}"] = sum(results["recall"][k]) / len(queries)
            summary["ndcg"][f"NDCG@{k}"] = sum(results["ndcg"][k]) / len(queries)
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    from src.retrieval.retrieval_pipeline import RetrievalPipeline
    
    # Create test data
    pipeline = RetrievalPipeline(collection_name="eval_test")
    
    docs = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing analyzes text",
        "Computer vision processes images"
    ]
    
    pipeline.index(docs)
    
    # Test queries with known relevant docs
    test_queries = [
        "programming languages",
        "artificial intelligence and machine learning"
    ]
    
    relevant_sets = [
        {"Python is a programming language"},
        {"Machine learning is a subset of AI", "Deep learning uses neural networks"}
    ]
    
    # Evaluate
    evaluator = RetrievalEvaluator(pipeline)
    results = evaluator.evaluate(test_queries, relevant_sets)
    
    print("\nEvaluation Results:")
    print(f"MRR: {results['mrr']:.3f}")
    for metric in ['precision', 'recall', 'ndcg']:
        print(f"\n{metric.upper()}:")
        for k, v in results[metric].items():
            print(f"  {k}: {v:.3f}")
