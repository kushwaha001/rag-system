from FlagEmbedding import FlagModel
from typing import List
import numpy as np
import torch

class EmbeddingService:
    def __init__(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = "cuda:1"
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        print(f"🔁 Loading BGE model on {device}...")

        self.model = FlagModel(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=torch.cuda.is_available(),
            devices=[device]
        )

        print(f"✅ BGE-Large loaded on {device}")

    # =========================
    # DOCUMENT EMBEDDINGS
    # =========================
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode_single_device(
            texts,
            batch_size=16
        )

        # 🔥 Ensure clean python list of floats
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # Safety: ensure each embedding is flat list
        cleaned = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            cleaned.append([float(x) for x in emb])

        return cleaned

    # =========================
    # QUERY EMBEDDING (CRITICAL FIX)
    # =========================
    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode_queries([query])

        # 🔥 FIX 1: convert numpy → list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # 🔥 FIX 2: flatten [[...]] → [...]
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            embedding = embedding[0]

        # 🔥 FIX 3: force float (avoid scalar conversion errors)
        embedding = [float(x) for x in embedding]

        return embedding


# =========================
# SINGLETON
# =========================
_service = None

def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service