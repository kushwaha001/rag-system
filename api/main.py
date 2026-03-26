from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.qdrant_setup import create_collection
import httpx
import os
import time

load_dotenv()

app = FastAPI(title="Industrial RAG System", version="1.0.0")

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))

@app.on_event("startup")
async def startup():
    create_collection()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    file_path: str

class FolderIngestRequest(BaseModel):
    folder_path: str
    use_vlm: bool = False

@app.get("/health")
async def health():
    return {"status": "ok", "model": VLLM_MODEL}

@app.get("/services")
async def check_services():
    results = {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{VLLM_HOST}/health", timeout=5)
            results["vllm"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        results["vllm"] = "unreachable"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{QDRANT_HOST}/collections", timeout=5)
            results["qdrant"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        results["qdrant"] = "unreachable"
    try:
        from utils.cache import get_cache
        get_cache()
        results["redis"] = "ok"
    except Exception:
        results["redis"] = "unreachable"
    return results

@app.get("/cache/stats")
async def cache_stats():
    from utils.cache import get_cache
    return get_cache().cache_stats()

@app.delete("/cache/clear")
async def clear_cache():
    from utils.cache import get_cache
    get_cache().clear_cache()
    return {"status": "cache cleared"}

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        from ingestion.pipeline import ingest_document
        result = ingest_document(req.file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/folder")
async def ingest_folder_endpoint(req: FolderIngestRequest):
    try:
        from ingestion.pipeline import ingest_folder
        result = ingest_folder(req.folder_path, use_vlm=req.use_vlm)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(req: QueryRequest):
    try:
        start_time = time.time()

        from utils.embeddings import get_embedding_service
        from utils.cache import get_cache

        embedding_service = get_embedding_service()
        query_embedding = embedding_service.embed_query(req.question)
        cache = get_cache()

        cached = cache.get_cached_response(req.question, query_embedding)
        if cached:
            cached["cached"] = True
            cached["latency_ms"] = round((time.time() - start_time) * 1000)
            return cached

        from retrieval.search import retrieve
        chunks = retrieve(req.question, top_k=req.top_k)

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        context = "\n\n".join([
            f"[Source {i+1}]: {c['text']}"
            for i, c in enumerate(chunks)
        ])

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
Always cite which source you used.

Context:
{context}

Question: {req.question}

Answer:"""

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.1
                }
            )
            result = response.json()
            answer = result["choices"][0]["message"]["content"]

        from utils.evaluator import evaluate_response
        evaluation = await evaluate_response(
            question=req.question,
            answer=answer,
            context=context
        )

        verdict = evaluation.get("verdict", "FAIL")
        faithfulness = evaluation.get("faithfulness", 0)

        final_response = {
            "question": req.question,
            "answer": answer,
            "verdict": verdict,
            "cached": False,
            "latency_ms": round((time.time() - start_time) * 1000),
            "scores": {
                "faithfulness": faithfulness,
                "hallucination": evaluation.get("hallucination"),
                "completeness": evaluation.get("completeness")
            },
            "sources": [
                {
                    "text": c["text"][:200],
                    "source": c["source"],
                    "reranker_score": c.get("reranker_score", 0)
                }
                for c in chunks
            ]
        }

        if verdict == "PASS" and faithfulness >= FAITHFULNESS_THRESHOLD:
            cache.cache_response(req.question, query_embedding, final_response)
        else:
            final_response["reason"] = evaluation.get("reason", "Low quality response")

        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))