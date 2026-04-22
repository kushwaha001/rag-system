# RAG System — Project Overview

## What Is This?

A production **Retrieval-Augmented Generation (RAG)** system built for vehicle/industrial technical manual Q&A. It ingests PDF, DOCX, PPTX, TXT, MD, HTML, and CSV files, stores chunked embeddings in Qdrant, and serves three use cases from a single web UI:

1. **Chat** — Ask questions, get answers grounded in your documents with source citations.
2. **Question Paper Generator** — Generate MCQ / Short / Long answer exam papers from document content.
3. **Custom Prompt** — Free-form prompt with document context injected automatically.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Qwen/Qwen2.5-14B-Instruct-AWQ via vLLM |
| Embeddings | BAAI/bge-large-en-v1.5 (FlagEmbedding, 1024-dim) |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder, FP16) |
| Vector DB | Qdrant (HNSW index, cosine similarity) |
| Cache | Redis (semantic similarity cache, cosine threshold 0.95) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS single-page app (no framework) |
| Doc Parsing | Docling (PDF/DOCX/PPTX → markdown) |
| Export | python-docx + reportlab (DOCX and PDF output) |

---

## Ports

| Service | Port |
|---|---|
| vLLM OpenAI-compatible API | 8000 |
| Qdrant vector DB | 6333 |
| Redis cache | 6379 |
| FastAPI (RAG API + Frontend) | 8001 |

---

## Directory Structure

```
rag-system/
├── api/
│   └── main.py              # FastAPI app — all HTTP endpoints
├── ingestion/
│   └── pipeline.py          # Document parsing, chunking, embedding, Qdrant upsert
├── retrieval/
│   └── search.py            # Dense search, RRF fusion, path scoring, reranking
├── utils/
│   ├── embeddings.py        # BGE-Large embedding service (singleton)
│   ├── reranker.py          # BGE Reranker v2-m3 service (singleton)
│   ├── qdrant_setup.py      # Qdrant client, collection creation, payload indexes
│   ├── cache.py             # Redis semantic cache
│   ├── evaluator.py         # LLM-as-judge faithfulness evaluator
│   ├── pdf_generator.py     # Export question paper as PDF (reportlab)
│   └── docx_generator.py    # Export question paper as DOCX (python-docx)
├── frontend/
│   └── index.html           # Single-file SPA (all HTML + CSS + JS inline)
├── docker-compose.yml       # Docker setup for Qdrant + Redis + vLLM + rag-api
├── Dockerfile               # Container definition for rag-api service
├── .env.example             # Template for environment variables
├── STARTUP.txt              # Step-by-step startup commands (3 terminals)
└── pyproject.toml / uv.lock # Python dependency management (uv)
```

---

## End-to-End Flow

### Ingestion Flow

```
File on disk
    └─► parse_document()          [Docling → markdown + image references]
    └─► chunk_text()              [256-word chunks, 64-word overlap]
    └─► embed_documents()         [BGE-Large, batch=128, FP16 on GPU]
    └─► Qdrant upsert()           [payload: text, source, filename, section, chunk_index]
```

Each chunk stores its `section` tag (`"chat"`, `"paper"`, `"prompt"`, or `"all"`) so the retriever can filter by which UI section uploaded it.

### Query / Chat Flow

```
User question
    └─► retrieve()
        ├─► dense_search()×3 variations  [BGE embed → Qdrant HNSW query]
        ├─► reciprocal_rank_fusion()      [RRF merges 3 result lists]
        └─► source_targeted_search()      [path-match → prepend targeted chunks]
    └─► BGE Reranker (cross-encoder)      [top-15 → re-score → top-5]
    └─► path_boost_after_rerank()         [vehicle/system folder boost + hard filter]
    └─► Build context string
    └─► vLLM chat completion              [Qwen2.5-14B-AWQ, temp=0.1]
    └─► evaluate_response()               [LLM-as-judge: faithfulness / hallucination]
    └─► Return answer + sources + scores
```

### Question Paper Flow

```
Topic + settings
    └─► retrieve_multi()
        ├─► 4 query angles embedded in single batch
        ├─► Qdrant search per angle (top_k=20 each)
        ├─► RRF fusion across all angles
        └─► source_targeted_search() prepended
    └─► path_boost_after_rerank()         [top-40 chunks]
    └─► Interleave chunks across sources  [round-robin shuffle]
    └─► asyncio.gather()                  [one vLLM call per question, all parallel]
    └─► Parse JSON per question
    └─► Return paper {mcq, short, long} + sources
```

---

## Key Algorithms

### Two-Level Path Scoring

Documents live in a folder hierarchy:
```
LVG/  <vehicle_folder>/  <system_folder>/  filename.ext
e.g.  3. MG 413W MPFI /  9. Steering sys /  Field manual.pdf
```

When a query arrives, tokens are extracted and matched against:
- **Vehicle folder** → `vehicle_score`
- **System folder + filename** → `system_score`

**Hard vehicle filter**: if any chunk has `vehicle_score >= 2`, all chunks from other vehicles are discarded entirely. This prevents cross-vehicle contamination (e.g. SAFARI manual appearing for MG 413W queries).

### Abbreviation Expansion

Folder names use abbreviations (`SUSP SYS`, `LUB`, `ENG`). The `ABBREV` dict expands these bidirectionally so `"lubrication"` in a query matches `"LUB"` in a folder path and vice versa.

### Reciprocal Rank Fusion (RRF)

Multiple query variations or angles are searched independently. RRF merges ranked lists:
```
score(chunk) = Σ  1 / (60 + rank_in_list)
```
Chunks appearing high in multiple lists get boosted; single-list flukes are suppressed.

### Semantic Cache (Redis)

Chat answers are cached by embedding similarity. On each query, the new embedding is compared to all cached embeddings via cosine similarity. If best match ≥ 0.95, the cached answer is returned immediately without hitting vLLM.

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve frontend SPA |
| GET | `/health` | Basic health check |
| GET | `/services` | Check vLLM + Qdrant + Redis status |
| POST | `/upload` | Upload file, ingest into Qdrant (multipart, section param) |
| POST | `/ingest` | Ingest file by server-side path |
| POST | `/ingest/folder` | Ingest entire folder recursively |
| GET | `/ingestion-status` | List ingested vs pending files in a folder |
| GET | `/documents` | List all ingested documents (filter by section) |
| DELETE | `/documents` | Delete document from Qdrant + disk |
| POST | `/query` | Chat Q&A with retrieval + reranking + evaluation |
| POST | `/generate-paper` | Generate question paper |
| POST | `/generate` | Custom prompt with RAG context |
| POST | `/download/docx` | Export paper as DOCX |
| POST | `/download/pdf` | Export paper as PDF |
| GET | `/cache/stats` | Redis cache statistics |
| DELETE | `/cache/clear` | Flush Redis cache |

---

## Environment Variables (`.env`)

```
VLLM_HOST=http://localhost:8000
VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
QDRANT_HOST=http://localhost:6333
QDRANT_COLLECTION=rag_documents
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600                  # Cache expiry in seconds
SIMILARITY_THRESHOLD=0.95       # Cosine threshold for cache hit
FAITHFULNESS_THRESHOLD=0.8      # Min score to mark answer as PASS
```

---

## Frontend Sections

The single-page app (`frontend/index.html`) has three tabs:

- **Chat tab** — Text input, source mode toggle (All Sources / Uploaded Only), answer display with sources accordion, evaluation scores (faithfulness / hallucination / completeness).
- **Question Paper tab** — Topic, difficulty, question count (1–200), type checkboxes (MCQ / Short / Long / True-False / Fill-blank / Diagram / Numerical), additional instructions textarea, preview with answer key toggle, download DOCX/PDF buttons.
- **Custom Prompt tab** — Free-form prompt input, source mode toggle, document list with delete option.

---

## Startup (Summary)

See `STARTUP.txt` for exact commands. In brief:

1. `docker start qdrant redis`
2. Start vLLM: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq --port 8000`
3. Start API: `.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8001`
4. Open browser: `http://localhost:8001`

---

## Important Notes

- **Duplicate detection** is path-exact: same filename from different vehicle folders is NOT a duplicate. Only the same absolute path (`/tmp/rag_uploads/chat/file.pdf`) triggers a conflict.
- **Cross-encoder reranker** is used only for `/query` (chat). The `/generate-paper` endpoint skips it for speed — RRF + path_boost is sufficient for question diversity.
- **Embeddings and reranker** are singletons loaded once at first use, not at startup (lazy init). Loading takes ~10–15 seconds.
- **vLLM continuous batching**: all questions in a paper are sent as parallel `asyncio.gather()` calls — vLLM processes them in a single batch, which is much faster than sequential generation.
- **Qdrant data** persists at `/home/hpc25/rag-system/qdrant_data/` via Docker volume mount.
