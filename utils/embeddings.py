from FlagEmbedding import FlagModel
from typing import List

class EmbeddingService:
    def __init__(self):
        print("Loading BGE-Large model...")
        self.model = FlagModel(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True
        )
        print("BGE-Large loaded ✅")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=32)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode_queries([query])
        return embedding[0].tolist()

_service = None

def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
