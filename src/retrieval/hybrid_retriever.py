"""
Hybrid retriever combining dense and sparse methods.
"""
from typing import List, Dict, Tuple
import numpy as np
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    """Hybrid retrieval using both dense (vector) and sparse (BM25)."""

    def __init__(self, collection_name: str = "documents", dense_weight: float = 0.5):
        """
        Initialize hybrid retriever.

        Args:
            collection_name: Vector store collection name
            dense_weight: Weight for dense scores (0-1), sparse gets (1-weight)
        """
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore(
            collection_name=collection_name, vector_size=self.embedder.dimension
        )
        self.sparse_retriever = SparseRetriever()
        self.dense_weight = dense_weight
        self.documents = []

    def index(self, documents: List[str], metadata: List[Dict] = None):
        """
        Index documents using both dense and sparse methods.

        Args:
            documents: List of document texts
            metadata: Optional metadata
        """
        print(f"Indexing {len(documents)} documents with hybrid retrieval...")

        self.documents = documents

        # Dense indexing
        embeddings = self.embedder.embed(documents, show_progress=True)
        self.vector_store.add(documents, embeddings, metadata)

        # Sparse indexing
        self.sparse_retriever.index(documents)

        print(f"Hybrid indexing complete")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve using hybrid approach.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of results with hybrid scores
        """
        # Get dense results
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, limit=top_k * 2)

        # Get sparse results
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)

        # Combine scores
        hybrid_scores = {}

        # Add dense scores
        max_dense = max([r["score"] for r in dense_results]) if dense_results else 1.0
        for r in dense_results:
            text = r["text"]
            normalized_score = r["score"] / max_dense
            hybrid_scores[text] = self.dense_weight * normalized_score

        # Add sparse scores
        max_sparse = (
            max([score for _, score in sparse_results]) if sparse_results else 1.0
        )
        for text, score in sparse_results:
            normalized_score = score / max_sparse
            if text in hybrid_scores:
                hybrid_scores[text] += (1 - self.dense_weight) * normalized_score
            else:
                hybrid_scores[text] = (1 - self.dense_weight) * normalized_score

        # Sort by hybrid score
        sorted_results = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [{"text": text, "score": score} for text, score in sorted_results]


if __name__ == "__main__":
    # Test hybrid retrieval
    retriever = HybridRetriever(collection_name="test_hybrid", dense_weight=0.5)

    docs = [
        "Python is a versatile programming language used in data science",
        "Machine learning algorithms learn patterns from data",
        "Deep neural networks have multiple hidden layers",
        "Natural language processing deals with text and language",
        "Computer vision enables machines to understand images",
    ]

    retriever.index(docs)

    query = "machine learning data"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    print("Hybrid results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. (hybrid: {r['score']:.3f}) {r['text']}")
