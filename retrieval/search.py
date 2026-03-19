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
    """
    Full retrieval pipeline:
    Dense search → RRF fusion → BGE Reranker → top_k results
    """
    print(f"🔍 Retrieving for: {query}")

    # Step 1 — Dense retrieval (get more candidates for reranker)
    dense_results = dense_search(query, top_k=top_k * 3)
    print(f"✅ Dense retrieval: {len(dense_results)} results")

    # Step 2 — RRF fusion
    fused_results = reciprocal_rank_fusion([dense_results])
    print(f"✅ RRF fusion: {len(fused_results)} results")

    # Step 3 — Rerank with BGE cross-encoder
    from utils.reranker import get_reranker
    reranker = get_reranker()
    reranked = reranker.rerank(query, fused_results, top_k=top_k)
    print(f"✅ Reranked: {len(reranked)} results")

    return reranked

if __name__ == "__main__":
    results = retrieve("What is machine learning?", top_k=3)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"RRF Score:      {r.get('rrf_score', 0):.4f}")
        print(f"Reranker Score: {r.get('reranker_score', 0):.4f}")
        print(f"Text: {r['text'][:200]}")
        print(f"Source: {r['source']}")
