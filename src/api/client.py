"""
API client for testing.
"""
import requests
from typing import List, Optional


class RAGClient:
    """Client for RAG API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client."""
        self.base_url = base_url

    def health(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def index(
        self, documents: List[str], metadata: Optional[List[dict]] = None
    ) -> dict:
        """Index documents."""
        payload = {"documents": documents}
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(f"{self.base_url}/index", json=payload)
        response.raise_for_status()
        return response.json()

    def query(self, query: str, top_k: int = 5) -> dict:
        """Query documents."""
        payload = {"query": query, "top_k": top_k}

        response = requests.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Test client
    client = RAGClient()

    # Check health
    print("Health check:", client.health())

    # Index documents
    docs = [
        "Python is a high-level programming language",
        "Machine learning enables computers to learn from data",
        "Docker containers provide isolated environments",
    ]

    result = client.index(docs)
    print(f"\nIndexed: {result}")

    # Query
    response = client.query("machine learning", top_k=2)
    print(f"\nQuery: {response['query']}")
    print(f"Found {response['count']} results:")
    for i, r in enumerate(response["results"], 1):
        print(f"{i}. ({r['score']:.3f}) {r['text']}")
