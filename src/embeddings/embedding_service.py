"""
Embedding service with batch processing support.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingService:
    """Generate embeddings using a local model with batch processing."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for encoding
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(
        self, 
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (dimension,)
        """
        return self.embed([query])[0]


if __name__ == "__main__":
    import time
    
    service = EmbeddingService()
    
    # Test single query
    query = "What is machine learning?"
    query_emb = service.embed_query(query)
    print(f"Query embedding shape: {query_emb.shape}")
    
    # Test batch processing
    test_texts = [f"Document {i} about various topics" for i in range(100)]
    
    start = time.time()
    embeddings = service.embed(test_texts, show_progress=True)
    elapsed = time.time() - start
    
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    print(f"Time: {elapsed:.2f}s ({len(test_texts)/elapsed:.1f} docs/sec)")
