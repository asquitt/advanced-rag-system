"""
Basic embedding service using sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingService:
    """Generate embeddings using a local model."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings


if __name__ == "__main__":
    # Simple test
    service = EmbeddingService()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence"
    ]
    
    embeddings = service.embed(test_texts)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"First embedding (truncated): {embeddings[0][:5]}")
