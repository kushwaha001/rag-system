from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.qdrant_setup import create_collection
import httpx
import os
import time
from fastapi.responses import FileResponse
from utils.docx_generator import generate_docx
from utils.pdf_generator import generate_pdf
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="Industrial RAG System", version="1.0.0")

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve frontend files
@app.on_event("startup")
async def startup():
    create_collection()
    print("✅ Qdrant ready")

# ── SERVE FRONTEND ────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

# Mount static files
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="static")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    file_path: str

class FolderIngestRequest(BaseModel):
    folder_path: str
    use_vlm: bool = False

class PaperRequest(BaseModel):
    topic: str
    difficulty: str
    num_questions: int
    types: list[str]

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

        # =========================
        # STEP 0 — EMBEDDING
        # =========================
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.embed_query(req.question)

        # 🔥 CACHE COMPLETELY DISABLED
        # (removed all cache usage)

        # =========================
        # STEP 1 — RETRIEVE
        # =========================
        from retrieval.search import retrieve
        raw_chunks = retrieve(req.question, top_k=50)

        if not raw_chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        # =========================
        # STEP 2 — DEDUP
        # =========================
        seen = set()
        chunks = []
        for c in raw_chunks:
            text = c.get("text", "").strip()
            if text and text not in seen:
                seen.add(text)
                chunks.append(c)

        # =========================
        # STEP 3 — RERANK (DISABLED 🔥)
        # =========================
        for c in chunks:
            c["reranker_score"] = c.get("rrf_score", c.get("score", 0.0))

        chunks = chunks[:8]

        # =========================
        # STEP 4 — BUILD CONTEXT
        # =========================
        context = "\n\n".join([
            f"[Source {i+1} | Score: {c['reranker_score']:.2f}]\n{c['text']}"
            for i, c in enumerate(chunks)
        ])

        # =========================
        # STEP 5 — LLM CALL
        # =========================
        prompt = f"""You are a helpful assistant. Answer ONLY using the provided context.
Always cite sources like [Source 1].

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

        # =========================
        # STEP 6 — EVALUATION
        # =========================
        from utils.evaluator import evaluate_response
        evaluation = await evaluate_response(
            question=req.question,
            answer=answer,
            context=context
        )

        verdict = evaluation.get("verdict", "FAIL")
        faithfulness = evaluation.get("faithfulness", 0)

        # =========================
        # FINAL RESPONSE
        # =========================
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
                    "source": c.get("source", ""),
                    "reranker_score": c["reranker_score"]
                }
                for c in chunks
            ]
        }

        if verdict != "PASS" or faithfulness < FAITHFULNESS_THRESHOLD:
            final_response["reason"] = evaluation.get("reason", "Low quality response")

        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate-paper")
async def generate_paper(req: PaperRequest):
    try:
        start_time = time.time()

        from retrieval.search import retrieve

        # =========================
        # STEP 1 — RETRIEVE CONTEXT
        # =========================
        query_text = f"{req.topic} concepts explanation technical details"
        raw_chunks = retrieve(query_text, top_k=50)

        if not raw_chunks:
            raise HTTPException(status_code=404, detail="No relevant content found")

        # Deduplicate + clean
        seen = set()
        chunks = []
        for c in raw_chunks:
            text = c.get("text", "").strip()
            if text and text not in seen:
                seen.add(text)
                chunks.append(c)

        context = "\n\n".join([c["text"] for c in chunks[:20]])

        # =========================
        # STEP 2 — DISTRIBUTION
        # =========================
        total = req.num_questions

        mcq = int(total * 0.5) if "mcq" in req.types else 0
        short = int(total * 0.3) if "short" in req.types else 0
        long = total - (mcq + short)

        # =========================
        # STEP 3 — PROMPT
        # =========================
        prompt = f"""
You are an expert exam paper setter.

Generate a structured question paper.

Requirements:
- Topic: {req.topic}
- Difficulty: {req.difficulty}
- Total Questions: {req.num_questions}

Breakdown:
- MCQs: {mcq}
- Short Answer: {short}
- Long Answer: {long}

STRICT RULES:
- Use ONLY the given context
- Do NOT hallucinate
- Avoid repetition
- Questions must be technical

Return ONLY valid JSON. No explanation.

FORMAT:
{{
  "mcq": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "answer": "A"
    }}
  ],
  "short": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ],
  "long": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}

Context:
{context}
"""

        # =========================
        # STEP 4 — LLM CALL
        # =========================
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1200
                }
            )

        result = response.json()
        output_text = result["choices"][0]["message"]["content"]

        # =========================
        # STEP 5 — SAFE JSON PARSE
        # =========================
        import json

        def try_parse(text):
            try:
                return json.loads(text)
            except:
                return None

        paper = try_parse(output_text)

        # 🔥 Retry once if failed
        if paper is None:
            retry_prompt = prompt + "\n\nREMEMBER: OUTPUT ONLY JSON."
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{VLLM_HOST}/v1/chat/completions",
                    json={
                        "model": VLLM_MODEL,
                        "messages": [{"role": "user", "content": retry_prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1200
                    }
                )
            result = response.json()
            output_text = result["choices"][0]["message"]["content"]
            paper = try_parse(output_text)

        if paper is None:
            paper = {
                "error": "Failed to generate structured paper",
                "raw_output": output_text[:1000]
            }

        # =========================
        # RESPONSE
        # =========================
        return {
            "topic": req.topic,
            "difficulty": req.difficulty,
            "paper": paper,
            "latency_ms": round((time.time() - start_time) * 1000)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download/docx")
async def download_docx(data: dict):
    try:
        paper = data.get("paper")
        topic = data.get("topic", "Unknown")
        difficulty = data.get("difficulty", "Unknown")

        if not paper:
            raise HTTPException(status_code=400, detail="Missing paper data")

        filepath = generate_docx(paper, topic, difficulty)

        return FileResponse(
            path=filepath,
            filename="question_paper.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download/pdf")
async def download_pdf(data: dict):
    try:
        paper = data.get("paper")
        topic = data.get("topic", "Unknown")
        difficulty = data.get("difficulty", "Unknown")

        if not paper:
            raise HTTPException(status_code=400, detail="Missing paper data")

        filepath = generate_pdf(paper, topic, difficulty)

        return FileResponse(
            path=filepath,
            filename="question_paper.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))