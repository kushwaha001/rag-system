from qdrant_client import QdrantClient
from utils.embeddings import get_embedding_service
from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from typing import List, Dict

def dense_search(query: str, top_k: int = 10) -> List[Dict]:
    """Dense retrieval using BGE-Large embeddings"""
    embedding_service = get_embedding_service()
    query_vector = embedding_service.embed_query(query)

    client = get_qdrant_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score": r.score,
            "retriever": "dense"
        }
        for r in results
    ]

def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    scores = {}
    texts = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            key = result["text"][:100]
            if key not in scores:
                scores[key] = 0.0
                texts[key] = result
            scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for key in sorted_keys:
        item = texts[key].copy()
        item["rrf_score"] = scores[key]
        fused.append(item)

    return fused

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    print(f"🔍 Retrieving for: {query}")

    dense_results = dense_search(query, top_k=top_k * 3)
    print(f"✅ Dense retrieval: {len(dense_results)} results")

    fused_results = reciprocal_rank_fusion([dense_results])

    for r in fused_results:
        r["reranker_score"] = r.get("rrf_score", 0.0)

    return fused_results[:top_k]


def retrieve_multi(queries: List[str], top_k: int = 20) -> List[Dict]:
    """Retrieve for multiple queries with a single batched embedding call."""
    from utils.embeddings import get_embedding_service
    embedding_service = get_embedding_service()

    # Embed all queries in one GPU batch
    vectors = embedding_service.model.encode_queries(queries)
    import numpy as np
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()

    client = get_qdrant_client()
    seen_texts: set = set()
    all_chunks: List[Dict] = []

    for vector in vectors:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        vector = [float(x) for x in vector]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True
        ).points

        for r in results:
            text = r.payload["text"].strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_chunks.append({
                    "text": text,
                    "source": r.payload["source"],
                    "chunk_index": r.payload["chunk_index"],
                    "score": r.score,
                    "rrf_score": r.score,
                    "reranker_score": r.score,
                    "retriever": "dense"
                })

    return all_chunks

if __name__ == "__main__":
    results = retrieve("What is machine learning?", top_k=3)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"RRF Score:      {r.get('rrf_score', 0):.4f}")
        print(f"Reranker Score: {r.get('reranker_score', 0):.4f}")
        print(f"Text: {r['text'][:200]}")
        print(f"Source: {r['source']}")
