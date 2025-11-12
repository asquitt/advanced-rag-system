"""
Vector store wrapper for Qdrant.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np
import uuid


class VectorStore:
    """Qdrant vector database wrapper."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 384  # bge-small dimension
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            host: Qdrant host
            port: Qdrant port
            vector_size: Embedding dimension
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = vector_size
        
        self._create_collection()
    
    def _create_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection setup: {e}")
    
    def add(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: Document texts
            embeddings: Document embeddings
            metadata: Optional metadata for each document
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        points = []
        for text, embedding, meta in zip(texts, embeddings, metadata):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload={"text": text, **meta}
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: np.ndarray, limit: int = 5):
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            limit: Number of results to return
            
        Returns:
            List of search results with scores
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            limit=limit
        )
        
        return [
            {
                "text": r.payload["text"],
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"}
            }
            for r in results
        ]
    
    def count(self) -> int:
        """Get number of documents in collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count


if __name__ == "__main__":
    # Test vector store
    store = VectorStore(collection_name="test")
    
    # Create dummy data
    texts = ["Hello world", "Machine learning", "Vector database"]
    embeddings = np.random.randn(3, 384).astype(np.float32)
    
    # Add documents
    store.add(texts, embeddings)
    print(f"Added {store.count()} documents")
    
    # Search
    query_vec = np.random.randn(384).astype(np.float32)
    results = store.search(query_vec, limit=2)
    
    print("\nSearch results:")
    for r in results:
        print(f"  {r['text']}: {r['score']:.3f}")
