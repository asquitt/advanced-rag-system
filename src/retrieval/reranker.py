"""
Re-ranking using cross-encoder models.
"""
from sentence_transformers import CrossEncoder
from typing import List, Tuple


class Reranker:
    """Cross-encoder based re-ranker for improved precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize re-ranker.

        Args:
            model_name: Cross-encoder model name
        """
        print(f"Loading re-ranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Re-ranker loaded")

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Query string
            documents: List of documents to re-rank
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]


if __name__ == "__main__":
    # Test re-ranker
    reranker = Reranker()

    query = "machine learning applications"

    # Simulated retrieval results (not perfectly ordered)
    docs = [
        "Computer vision for image processing tasks",
        "Machine learning applied to healthcare diagnostics",
        "Natural language models for text generation",
        "Reinforcement learning in robotics applications",
        "Deep learning applications in finance",
    ]

    print(f"Query: {query}\n")
    print("Before re-ranking:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc}")

    # Re-rank
    reranked = reranker.rerank(query, docs, top_k=3)

    print("\nAfter re-ranking:")
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"{i}. ({score:.3f}) {doc}")
