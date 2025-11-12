"""
Configuration for retrieval system.
"""
from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    
    # Model settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    
    # Retrieval settings
    dense_weight: float = 0.5  # Balance between dense (0) and sparse (1)
    top_k: int = 5
    batch_size: int = 32
    
    # Vector store settings
    collection_name: str = "documents"
    vector_host: str = "localhost"
    vector_port: int = 6333
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
