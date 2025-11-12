"""
Complete retrieval pipeline with optional re-ranking.
"""
from typing import List, Dict
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class RetrievalPipeline:
    """Complete retrieval pipeline with hybrid search and re-ranking."""

    def __init__(
        self,
        collection_name: str = "documents",
        use_reranking: bool = True,
        dense_weight: float = 0.5,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            collection_name: Vector store collection
            use_reranking: Whether to use re-ranking
            dense_weight: Weight for dense vs sparse
        """
        self.hybrid_retriever = HybridRetriever(
            collection_name=collection_name, dense_weight=dense_weight
        )
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = Reranker()

    def index(self, documents: List[str], metadata: List[Dict] = None):
        """Index documents."""
        self.hybrid_retriever.index(documents, metadata)

    def retrieve(
        self, query: str, top_k: int = 5, rerank_top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve documents with optional re-ranking.

        Args:
            query: Query string
            top_k: Final number of results
            rerank_top_k: Number to retrieve before re-ranking

        Returns:
            List of results with scores
        """
        # First stage: hybrid retrieval
        initial_top_k = rerank_top_k if self.use_reranking else top_k
        results = self.hybrid_retriever.retrieve(query, top_k=initial_top_k)

        # Second stage: re-ranking
        if self.use_reranking and len(results) > 0:
            documents = [r["text"] for r in results]
            reranked = self.reranker.rerank(query, documents, top_k=top_k)

            return [{"text": doc, "score": float(score)} for doc, score in reranked]

        return results[:top_k]


if __name__ == "__main__":
    # Test complete pipeline
    print("Testing retrieval pipeline with re-ranking...\n")

    pipeline = RetrievalPipeline(collection_name="test_pipeline", use_reranking=True)

    docs = [
        "Python programming for web development with Django and Flask",
        "Machine learning model training and evaluation techniques",
        "Deep learning architectures for computer vision tasks",
        "Natural language processing with transformer models",
        "Data science workflow from collection to visualization",
        "Reinforcement learning algorithms for game playing",
        "Statistical analysis methods for data exploration",
    ]

    pipeline.index(docs)

    query = "machine learning model evaluation"

    print(f"Query: {query}\n")
    results = pipeline.retrieve(query, top_k=3)

    print("Results (with re-ranking):")
    for i, r in enumerate(results, 1):
        print(f"{i}. ({r['score']:.3f}) {r['text']}")
