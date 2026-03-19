from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from utils.embeddings import get_embedding_service
from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_document(file_path: str) -> dict:
    """
    Full ingestion pipeline:
    File → Docling parse → Chunk → BGE-Large embed → Qdrant store
    """
    print(f"📄 Parsing document: {file_path}")

    # Step 1 — Parse with Docling
    converter = DocumentConverter()
    result = converter.convert(file_path)
    full_text = result.document.export_to_markdown()
    print(f"✅ Parsed {len(full_text)} characters")

    # Step 2 — Chunk
    chunks = chunk_text(full_text, chunk_size=512, overlap=50)
    print(f"✅ Created {len(chunks)} chunks")

    # Step 3 — Embed
    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents(chunks)
    print(f"✅ Embedded {len(embeddings)} chunks")

    # Step 4 — Store in Qdrant
    client = get_qdrant_client()
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "source": file_path,
                "chunk_index": i
            }
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"✅ Stored {len(points)} chunks in Qdrant")

    return {
        "file": file_path,
        "chunks": len(chunks),
        "status": "success"
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = ingest_document(sys.argv[1])
        print(result)
    else:
        print("Usage: python3 ingestion/pipeline.py <path_to_file>")
