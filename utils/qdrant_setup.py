from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, 
    HnswConfigDiff, OptimizersConfigDiff
)
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_documents")
VECTOR_SIZE = 1024  # BGE-Large dimension

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_HOST)

def create_collection():
    client = get_qdrant_client()
    
    # Check if already exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists ✅")
        return client

    # Create collection with HNSW index
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        ),
        hnsw_config=HnswConfigDiff(
            m=16,              # connections per node
            ef_construct=100,  # build time accuracy
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created ✅")
    return client

if __name__ == "__main__":
    create_collection()
