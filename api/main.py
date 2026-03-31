from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.qdrant_setup import create_collection
import httpx
import os
import time
import asyncio
import json
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

class GenerateRequest(BaseModel):
    prompt: str
    top_k: int = 10
    max_tokens: int = 2048
    temperature: float = 0.3

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
                    "max_tokens": 2048,
                    "temperature": 0.1
                }
            )
            result = response.json()
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"vLLM error: {result['error'].get('message', str(result['error']))}")
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
        import random
        start_time = time.time()

        from retrieval.search import retrieve_multi

        # =========================
        # STEP 1 — RETRIEVE CONTEXT (multiple angles, single batched embedding)
        # =========================
        queries = [
            f"{req.topic} concepts definitions principles",
            f"{req.topic} mechanisms processes how it works",
            f"{req.topic} applications examples real world",
            f"{req.topic} analysis comparison advantages disadvantages",
        ]

        all_chunks = retrieve_multi(queries, top_k=20)

        if not all_chunks:
            raise HTTPException(status_code=404, detail="No relevant content found")

        # Shuffle so each generation run uses chunks in a different order
        random.shuffle(all_chunks)

        # =========================
        # STEP 2 — DISTRIBUTION
        # =========================
        total = req.num_questions

        # Normalize types: true_false/fill_blank → mcq, diagram → long
        normalized_types = []
        for t in req.types:
            if t in ('mcq', 'true_false', 'fill_blank'):
                if 'mcq' not in normalized_types:
                    normalized_types.append('mcq')
            elif t == 'short':
                if 'short' not in normalized_types:
                    normalized_types.append('short')
            elif t in ('long', 'diagram'):
                if 'long' not in normalized_types:
                    normalized_types.append('long')
        if not normalized_types:
            normalized_types = ['mcq']

        mcq = int(total * 0.5) if "mcq" in normalized_types else 0
        short = int(total * 0.3) if "short" in normalized_types else 0
        long = (total - mcq - short) if "long" in normalized_types else 0

        # Redistribute any unallocated questions
        unallocated = total - mcq - short - long
        if unallocated > 0:
            if "mcq" in normalized_types:
                mcq += unallocated
            elif "short" in normalized_types:
                short += unallocated
            else:
                long += unallocated

        # =========================
        # STEP 3 — PARALLEL GENERATION (one request per question)
        # vLLM continuous batching handles all requests simultaneously
        # =========================
        focus_angles = [
            "definitions, terminology, and core concepts",
            "working mechanisms, processes, and how things operate",
            "real-world applications, use cases, and examples",
            "comparisons, trade-offs, advantages and disadvantages",
            "problem-solving, troubleshooting, and critical thinking",
        ]

        answer_letters = ["a", "b", "c", "d"]

        def build_prompt(q_type: str, index: int) -> str:
            chunk_start = (index * 3) % len(all_chunks)
            ctx_chunks = all_chunks[chunk_start:chunk_start + 3]
            if len(ctx_chunks) < 3:
                ctx_chunks += all_chunks[:3 - len(ctx_chunks)]
            context = "\n\n".join([c["text"][:500] for c in ctx_chunks])
            focus = focus_angles[index % len(focus_angles)]
            correct = random.choice(answer_letters)

            if q_type == "mcq":
                return f"""You are an exam setter. Read the context carefully and generate 1 MCQ question.

STRICT RULES:
- The question MUST be based ONLY on information present in the context below
- Do NOT use any outside knowledge
- The correct answer MUST be option "{correct}"
- The other 3 options must be plausible but clearly wrong based on the context
- Focus on: {focus}
- Difficulty: {req.difficulty}

Return ONLY valid JSON, no extra text:
{{"question": "<question from context>", "options": {{"a": "<option>", "b": "<option>", "c": "<option>", "d": "<option>"}}, "answer": "{correct}"}}

Context:
{context}"""

            elif q_type == "short":
                return f"""You are an exam setter. Read the context carefully and generate 1 short answer question.

STRICT RULES:
- The question and answer MUST be based ONLY on information in the context below
- Do NOT use any outside knowledge
- Focus on: {focus}
- Difficulty: {req.difficulty}

Return ONLY valid JSON, no extra text:
{{"question": "<question from context>", "answer": "<concise answer from context>"}}

Context:
{context}"""

            else:
                return f"""You are an exam setter. Read the context carefully and generate 1 long answer question.

STRICT RULES:
- The question and answer MUST be based ONLY on information in the context below
- Do NOT use any outside knowledge
- Focus on: {focus}
- Difficulty: {req.difficulty}

Return ONLY valid JSON, no extra text:
{{"question": "<question from context>", "answer": "<detailed answer from context>"}}

Context:
{context}"""

        async def generate_one(client: httpx.AsyncClient, q_type: str, index: int):
            prompt = build_prompt(q_type, index)
            try:
                resp = await client.post(
                    f"{VLLM_HOST}/v1/chat/completions",
                    json={
                        "model": VLLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }
                )
                result = resp.json()
                if "error" in result:
                    return None, q_type
                text = result["choices"][0]["message"]["content"].strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start == -1 or end == 0:
                    return None, q_type
                return json.loads(text[start:end]), q_type
            except Exception:
                return None, q_type

        # Build task list: mcq × mcq_count, short × short_count, long × long_count
        tasks_spec = (
            [("mcq", i) for i in range(mcq)] +
            [("short", i) for i in range(short)] +
            [("long", i) for i in range(long)]
        )

        async with httpx.AsyncClient(timeout=120) as client:
            results = await asyncio.gather(*[
                generate_one(client, q_type, idx)
                for idx, (q_type, i) in enumerate(tasks_spec)
            ])

        all_mcq = [r for r, t in results if r and t == "mcq"]
        all_short = [r for r, t in results if r and t == "short"]
        all_long = [r for r, t in results if r and t == "long"]

        # =========================
        # STEP 4 — MERGE RESULTS
        # =========================
        paper = {
            "mcq": all_mcq[:mcq],
            "short": all_short[:short],
            "long": all_long[:long]
        }

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "topic": req.topic,
            "difficulty": req.difficulty,
            "paper": paper,
            "latency_ms": round((time.time() - start_time) * 1000)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        from utils.embeddings import get_embedding_service
        from retrieval.search import retrieve

        # Retrieve relevant chunks
        chunks = retrieve(req.prompt, top_k=req.top_k)

        context = "\n\n".join([
            f"[Source {i+1}]: {c['text']}"
            for i, c in enumerate(chunks[:req.top_k])
        ])

        prompt = f"""You are a helpful technical assistant. Use the following context to answer.

Context:
{context}

Task: {req.prompt}

Response:"""

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature
                }
            )
            result = response.json()
            answer = result["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "sources": [{"text": c["text"][:200], "source": c["source"]} for c in chunks],
            "status": "success"
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