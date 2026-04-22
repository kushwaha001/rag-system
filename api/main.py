from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

UPLOAD_DIR = "/tmp/rag_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    source_filter: str = "all"   # "all" | "chat" | "prompt"

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
    instructions: str = ""
    selected_sources: list[str] = []   # empty = use all ingested docs

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


@app.get("/ingestion-status")
async def ingestion_status(folder_path: str):
    try:
        from ingestion.pipeline import get_ingested_files, SUPPORTED_EXTENSIONS

        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

        all_files = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.startswith("~$"):
                    continue
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    all_files.append(os.path.join(root, f))

        ingested = get_ingested_files()

        ingested_files = sorted([f for f in all_files if f in ingested])
        pending_files  = sorted([f for f in all_files if f not in ingested])

        return {
            "folder": folder_path,
            "total": len(all_files),
            "ingested_count": len(ingested_files),
            "pending_count": len(pending_files),
            "ingested": ingested_files,
            "pending": pending_files,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    section: str = Form("chat")
):
    if section not in ("chat", "paper", "prompt"):
        raise HTTPException(status_code=400, detail="section must be chat, paper, or prompt")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.html', '.csv'}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    section_dir = os.path.join(UPLOAD_DIR, section)
    os.makedirs(section_dir, exist_ok=True)
    filepath = os.path.join(section_dir, file.filename)

    # Only flag as duplicate if this exact destination path is already in Qdrant
    # (same filename from a different vehicle/system folder is NOT a duplicate)
    from ingestion.pipeline import get_ingested_files
    if filepath in get_ingested_files():
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' is already uploaded in this section. Delete it first to replace."
        )

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    try:
        from ingestion.pipeline import ingest_document
        result = ingest_document(filepath, section=section)
        result["section"] = section
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents(section: str = None):
    try:
        from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
        client = get_qdrant_client()

        docs = {}
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=["source", "filename", "section"]
            )

            for r in results:
                if not r.payload:
                    continue
                source = r.payload.get("source", "")
                doc_section = r.payload.get("section", "all")
                filename = r.payload.get("filename", source.split("/")[-1])

                if section and doc_section != section:
                    continue

                if source not in docs:
                    docs[source] = {
                        "source": source,
                        "filename": filename,
                        "section": doc_section,
                        "chunk_count": 0
                    }
                docs[source]["chunk_count"] += 1

            if offset is None:
                break

        return {"documents": list(docs.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def delete_document(source: str):
    try:
        from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue

        client = get_qdrant_client()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            )
        )

        if source.startswith(UPLOAD_DIR):
            try:
                os.remove(source)
            except Exception:
                pass

        return {"status": "deleted", "source": source}
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
        section_filter = req.source_filter if req.source_filter in ("chat", "prompt") else None
        raw_chunks = retrieve(req.question, top_k=50, section_filter=section_filter)

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
        # STEP 3 — RERANK + PATH BOOST
        # =========================
        try:
            from utils.reranker import get_reranker
            from retrieval.search import path_boost_after_rerank
            reranker = get_reranker()
            # Get wider pool from reranker so path_boost has room to promote right docs
            chunks = reranker.rerank(req.question, chunks, top_k=15)
            # Boost chunks whose file path matches query keywords (model codes etc.)
            # This corrects cases where a different vehicle's manual scores high on text alone
            chunks = path_boost_after_rerank(req.question, chunks)
            chunks = chunks[:5]
        except Exception as rerank_err:
            print(f"⚠️ Reranker failed: {rerank_err} — falling back to RRF score ranking")
            from retrieval.search import path_boost_after_rerank
            chunks = sorted(chunks, key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)[:15]
            chunks = path_boost_after_rerank(req.question, chunks)[:5]

        # =========================
        # STEP 4 — BUILD CONTEXT
        # =========================
        context = "\n\n".join([
            f"[Source {i+1}]\n{c['text']}"
            for i, c in enumerate(chunks)
        ])

        # =========================
        # STEP 5 — LLM CALL
        # =========================
        system_msg = (
            "You are a precise technical assistant. Answer questions based ONLY on the "
            "provided context. If the context does not contain enough information to answer "
            "the question, say: 'I don't have enough information in the provided documents "
            "to answer this.' Do not use outside knowledge. Cite sources using [Source N] notation."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer based only on the context above:"

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
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
                    "reranker_score": c.get("reranker_score", c.get("rrf_score", c.get("score", 0)))
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
        # Four angles tuned for technical/vehicle manuals:
        # broad topic + 3 specific angles that match the language in workshop manuals
        queries = [
            req.topic,
            f"{req.topic} components parts specifications construction",
            f"{req.topic} working principle operation procedure steps",
            f"{req.topic} maintenance inspection troubleshooting diagnosis",
        ]

        all_chunks = retrieve_multi(
            queries,
            top_k=20,
            topic=req.topic,
            selected_sources=req.selected_sources,
        )

        if not all_chunks:
            raise HTTPException(
                status_code=404,
                detail=("No content found in the selected documents."
                        if req.selected_sources
                        else "No relevant content found")
            )

        # Use reranker as the quality gate — skip the raw-score pre-filter
        # (raw embedding scores are not comparable across RRF-fused results)
        if not all_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found relevant enough to '{req.topic}'. Try a different topic or upload relevant documents."
            )

        # =========================
        # RERANK — sort by cross-encoder score against the topic so the most
        # relevant chunks are at the front of the list
        # =========================
        # For question paper we skip the cross-encoder (slow, designed for single-answer
        # precision). Path boost on RRF scores is sufficient — it already hard-filters
        # the correct vehicle/system and boosts relevant chunks.
        from retrieval.search import path_boost_after_rerank
        all_chunks = sorted(all_chunks, key=lambda x: x.get("rrf_score", x.get("score", 0)), reverse=True)
        all_chunks = path_boost_after_rerank(req.topic, all_chunks)[:40]

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

        # Shuffle chunks within each source independently, then interleave sources
        # so every question gets a random chunk from a random part of a random source.
        from itertools import zip_longest as _zip_longest
        _source_groups: dict = {}
        for _c in all_chunks:
            _src = _c.get("source", "")
            _source_groups.setdefault(_src, []).append(_c)
        # Shuffle within each source so we don't always start from the top
        for _src_chunks in _source_groups.values():
            random.shuffle(_src_chunks)
        # Round-robin interleave across sources
        interleaved_chunks = [
            _c
            for _group in _zip_longest(*_source_groups.values())
            for _c in _group
            if _c is not None
        ]
        _n_chunks = len(interleaved_chunks)

        def build_prompt(q_type: str, index: int) -> str:
            # Pick 3 chunks randomly spread across the interleaved pool —
            # each question starts at a different random offset within its slice.
            slice_size = max(1, _n_chunks // max(total, 1))
            slice_start = index * slice_size
            # Pick a random position within this question's slice
            offset = random.randint(0, max(0, slice_size - 1))
            chunk_start = (slice_start + offset) % _n_chunks
            ctx_chunks = interleaved_chunks[chunk_start:chunk_start + 3]
            if len(ctx_chunks) < 3:
                ctx_chunks = ctx_chunks + interleaved_chunks[:3 - len(ctx_chunks)]
            context = "\n\n".join([c["text"][:800] for c in ctx_chunks])
            focus = focus_angles[index % len(focus_angles)]
            correct = random.choice(answer_letters)

            extra = f"\n- Additional instructions: {req.instructions}" if req.instructions.strip() else ""

            if q_type == "mcq":
                return f"""You are an exam setter. Read the context carefully and generate 1 MCQ question.

STRICT RULES:
- The question MUST be based ONLY on information present in the context below
- Do NOT use any outside knowledge
- The correct answer MUST be option "{correct}"
- The other 3 options must be plausible but clearly wrong based on the context
- Focus on: {focus}
- Difficulty: {req.difficulty}{extra}

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
- Difficulty: {req.difficulty}{extra}

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
- Difficulty: {req.difficulty}{extra}

Return ONLY valid JSON, no extra text:
{{"question": "<question from context>", "answer": "<detailed answer from context>"}}

Context:
{context}"""

        # Token budget per type — MCQ needs ~150, short ~250, long ~450
        _max_tokens_map = {"mcq": 200, "short": 280, "long": 450}

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
                        "max_tokens": _max_tokens_map.get(q_type, 300)
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

        async with httpx.AsyncClient(timeout=300) as client:
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
        unique_sources = list({
            c.get("source", "") for c in all_chunks if c.get("source")
        })
        sources = [
            {"filename": s.split("/")[-1], "source": s}
            for s in unique_sources
        ]

        return {
            "topic": req.topic,
            "difficulty": req.difficulty,
            "paper": paper,
            "sources": sources,
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