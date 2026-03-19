import redis
import numpy as np
import json
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))

class SemanticCache:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False
        )
        self.client.ping()
        print("Redis cache connected ✅")

    def _cosine_similarity(self, a: list, b: list) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_cached_response(self, query: str, query_embedding: list):
        try:
            keys = self.client.keys("query_embedding:*")
            best_similarity = 0.0
            best_response = None

            for key in keys:
                cached_data = self.client.get(key)
                if not cached_data:
                    continue
                cached = json.loads(cached_data)
                similarity = self._cosine_similarity(
                    query_embedding,
                    cached.get("embedding", [])
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = cached.get("response")

            if best_similarity >= SIMILARITY_THRESHOLD and best_response:
                print(f"🎯 Cache HIT (similarity: {best_similarity:.3f})")
                return best_response

            print(f"❌ Cache MISS (best similarity: {best_similarity:.3f})")
            return None

        except Exception as e:
            print(f"Cache lookup error: {e}")
            return None

    def cache_response(self, query: str, query_embedding: list, response: dict):
        try:
            key = f"query_embedding:{hashlib.md5(query.encode()).hexdigest()}"
            data = json.dumps({
                "query": query,
                "embedding": query_embedding,
                "response": response
            })
            self.client.setex(key, CACHE_TTL, data)
            print(f"💾 Response cached (TTL: {CACHE_TTL}s)")
        except Exception as e:
            print(f"Cache store error: {e}")

    def clear_cache(self):
        keys = self.client.keys("query_embedding:*")
        if keys:
            self.client.delete(*keys)
        print(f"🗑️ Cleared {len(keys)} cached responses")

    def cache_stats(self) -> dict:
        keys = self.client.keys("query_embedding:*")
        return {
            "cached_queries": len(keys),
            "ttl_seconds": CACHE_TTL,
            "similarity_threshold": SIMILARITY_THRESHOLD
        }

_cache = None

def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
