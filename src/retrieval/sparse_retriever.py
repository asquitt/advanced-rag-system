"""
Sparse retrieval using BM25.
"""
from rank_bm25 import BM25Okapi
from typing import List, Tuple


class SparseRetriever:
    """BM25-based sparse retrieval."""

    def __init__(self):
        """Initialize sparse retriever."""
        self.bm25 = None
        self.documents = []

    def index(self, documents: List[str]):
        """
        Index documents for BM25.

        Args:
            documents: List of document texts
        """
        self.documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Indexed {len(documents)} documents for BM25")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve documents using BM25.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Must index documents first")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = [(self.documents[i], scores[i]) for i in top_indices]

        return results


if __name__ == "__main__":
    # Test BM25
    retriever = SparseRetriever()

    docs = [
        "Python programming language for data science",
        "Machine learning algorithms and models",
        "Deep learning neural networks",
        "Natural language processing techniques",
        "Computer vision image recognition",
    ]

    retriever.index(docs)

    query = "machine learning algorithms"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. (BM25: {score:.3f}) {doc}")
