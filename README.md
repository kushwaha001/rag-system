# 🔍 Industrial RAG System

A production-grade Retrieval-Augmented Generation (RAG) system built with open-source components.

## 🏗️ Architecture
```
Document
    ↓
Docling (parse) → BGE-Large (embed) → Qdrant (store)

Query
    ↓
Redis Cache Check
    ↓
BGE-Large (embed query)
    ↓
Qdrant Dense Retrieval
    ↓
RRF Fusion
    ↓
BGE Reranker
    ↓
Qwen2.5-32B (generate)
    ↓
LLM Evaluator (PASS/FAIL)
    ↓
Answer + Sources + Scores
```

## 🧩 Components

| Component | Model/Tool | Purpose |
|---|---|---|
| LLM | Qwen2.5-32B-Instruct-AWQ | Answer generation |
| Embeddings | BGE-Large-en-v1.5 | Semantic search |
| Reranker | BGE-Reranker-v2-m3 | Result accuracy |
| Vector DB | Qdrant | Chunk storage |
| Cache | Redis | Semantic caching |
| Parser | Docling | Document parsing |
| API | FastAPI | REST endpoints |
| Serving | vLLM | LLM inference |

## ⚙️ Requirements

- Ubuntu 20.04+
- NVIDIA GPU with 24GB+ VRAM
- CUDA 12.1+
- Python 3.11+
- Docker + Docker Compose

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/rag-system.git
cd rag-system
```

### 2. Setup environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start all services
```bash
docker-compose up -d qdrant redis
```

### 4. Start vLLM
```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.60 \
  --port 8000
```

### 5. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Start RAG API
```bash
uvicorn api.main:app --reload --port 9000
```

### 7. Verify
```bash
curl http://localhost:9000/services
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /services | Check all services |
| POST | /ingest | Ingest a document |
| POST | /query | Query the RAG system |
| GET | /cache/stats | Cache statistics |
| DELETE | /cache/clear | Clear cache |

## 📄 Ingest a Document
```bash
curl -X POST http://localhost:9000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

## 🔍 Query
```bash
curl -X POST http://localhost:9000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here?"}'
```

## 📁 Project Structure
```
rag-system/
├── api/
│   └── main.py          # FastAPI endpoints
├── ingestion/
│   └── pipeline.py      # Document ingestion
├── retrieval/
│   └── search.py        # Retrieval + RRF + Reranker
├── utils/
│   ├── embeddings.py    # BGE-Large service
│   ├── reranker.py      # BGE Reranker service
│   ├── evaluator.py     # LLM evaluator
│   ├── cache.py         # Redis semantic cache
│   └── qdrant_setup.py  # Qdrant collection setup
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## 📄 License
MIT
