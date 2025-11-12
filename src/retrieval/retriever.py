"""
Basic document retriever combining embeddings and vector store.
"""
from typing import List, Dict
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore


class BasicRetriever:
    """Simple retriever using dense embeddings."""

    def __init__(self, collection_name: str = "documents"):
        """Initialize retriever."""
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore(
            collection_name=collection_name, vector_size=self.embedder.dimension
        )

    def index(self, documents: List[str], metadata: List[Dict] = None):
        """
        Index documents.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        print(f"Indexing {len(documents)} documents...")
        embeddings = self.embedder.embed(documents, show_progress=True)
        self.vector_store.add(documents, embeddings, metadata)
        print(f"Indexed {self.vector_store.count()} total documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, limit=top_k)
        return results


if __name__ == "__main__":
    # Test end-to-end
    retriever = BasicRetriever(collection_name="test_retriever")

    # Index some documents
    docs = [
        "Python is a high-level programming language",
        "Machine learning is a subset of AI",
        "Neural networks are inspired by the human brain",
        "Docker containers provide isolated environments",
        "REST APIs enable communication between services",
    ]

    retriever.index(docs)

    # Query
    query = "What is machine learning?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    print("\nTop results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. (score: {r['score']:.3f}) {r['text']}")
