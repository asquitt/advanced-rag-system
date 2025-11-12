"""
Embedding service with Redis caching.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import hashlib
import redis


class EmbeddingService:
    """Generate embeddings with caching support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        use_cache: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for encoding
            use_cache: Whether to use Redis cache
            redis_host: Redis host
            redis_port: Redis port
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Setup cache
        if use_cache:
            try:
                self.cache = redis.Redis(
                    host=redis_host, port=redis_port, db=0, decode_responses=False
                )
                self.cache.ping()
                print(f"Redis cache connected")
            except Exception as e:
                print(f"Redis unavailable: {e}")
                self.use_cache = False

        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return f"emb:v1:{hashlib.md5(text.encode()).hexdigest()}"

    def embed(
        self, texts: List[str], show_progress: bool = False, check_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings with caching.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            check_cache: Whether to check cache

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache if enabled
        if self.use_cache and check_cache:
            for idx, text in enumerate(texts):
                cache_key = self._cache_key(text)
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        emb = np.frombuffer(cached, dtype=np.float32)
                        embeddings.append((idx, emb))
                        continue
                except Exception:
                    pass

                uncached_texts.append(text)
                uncached_indices.append(idx)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Cache new embeddings
            if self.use_cache and check_cache:
                for text, emb in zip(uncached_texts, new_embeddings):
                    cache_key = self._cache_key(text)
                    try:
                        self.cache.setex(
                            cache_key,
                            604800,  # 7 days
                            emb.astype(np.float32).tobytes(),
                        )
                    except Exception:
                        pass

            # Add to results
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, emb))

        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query])[0]


if __name__ == "__main__":
    import time

    service = EmbeddingService(use_cache=True)

    test_texts = ["machine learning", "deep learning", "natural language processing"]

    # Cold cache
    print("Cold cache (first run):")
    start = time.time()
    emb1 = service.embed(test_texts, check_cache=False)
    cold_time = time.time() - start
    print(f"Time: {cold_time*1000:.2f}ms")

    # Warm cache
    print("\nWarm cache (second run):")
    start = time.time()
    emb2 = service.embed(test_texts, check_cache=True)
    warm_time = time.time() - start
    print(f"Time: {warm_time*1000:.2f}ms")

    print(f"\nSpeedup: {cold_time/warm_time:.1f}x faster with caching")
    print(f"Embeddings match: {np.allclose(emb1, emb2)}")
