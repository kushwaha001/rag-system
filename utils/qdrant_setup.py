from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    HnswConfigDiff, OptimizersConfigDiff,
    PayloadSchemaType
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
        _ensure_payload_indexes(client)
        return client

    # Create collection with HNSW index
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created ✅")
    _ensure_payload_indexes(client)
    return client

def _ensure_payload_indexes(client: QdrantClient):
    """Create payload indexes for fast filtered scrolls and counts."""
    existing_info = client.get_collection(COLLECTION_NAME)
    existing_indexes = set(existing_info.payload_schema.keys()) if existing_info.payload_schema else set()

    for field, schema_type in [
        ("section", PayloadSchemaType.KEYWORD),
        ("source",  PayloadSchemaType.KEYWORD),
    ]:
        if field not in existing_indexes:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema_type,
                )
                print(f"  ✅ Payload index created: {field}")
            except Exception:
                pass  # index may already exist under a different name

if __name__ == "__main__":
    create_collection()
