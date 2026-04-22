from FlagEmbedding import FlagReranker
from typing import List, Dict
import os

class RerankerService:
    def __init__(self):
        print("Loading BGE Reranker...")
        self.reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True
        )
        print("BGE Reranker loaded ✅")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank chunks using cross-encoder scoring.
        Cross-encoder looks at query AND chunk together — much more accurate than embeddings alone.
        """
        if not chunks:
            return []

        # Build query-chunk pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Score each pair
        scores = self.reranker.compute_score(pairs, normalize=True)

        # Attach reranker scores — handle both list and single numpy scalar returns
        import numpy as np
        if not isinstance(scores, (list, np.ndarray)):
            scores = [scores]
        scores = list(scores)
        for i, chunk in enumerate(chunks):
            chunk["reranker_score"] = float(scores[i])

        # Sort by reranker score
        reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

        return reranked[:top_k]

_reranker = None

def get_reranker() -> RerankerService:
    global _reranker
    if _reranker is None:
        _reranker = RerankerService()
    return _reranker
