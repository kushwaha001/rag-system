# Industrial RAG System

An AI-powered question paper generation and document Q&A system built with vLLM, Qdrant, Redis, and FastAPI.

---

## What It Does

- **Question Paper Generator** — Generate MCQ, short answer, and long answer questions from ingested documents
- **Custom Q&A** — Ask questions and get answers grounded in your document knowledge base
- **Document Ingestion** — Ingest PDFs and DOCX files into the vector database
- **Download Papers** — Export generated papers as PDF or DOCX

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1x 16GB VRAM | 2x 24GB VRAM |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 100GB free |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| NVIDIA Driver | 520+ | 570+ |
| CUDA | 11.8+ | 12.x |

> The system uses `Qwen/Qwen2.5-14B-Instruct-AWQ` by default (~9GB VRAM). For higher quality, switch to the 32B model (requires 2x 24GB GPUs).

---

## Architecture

```
Browser
   |
   v
FastAPI (port 9000)  -->  vLLM (port 8000)  -->  Qwen LLM
   |
   |-->  Qdrant (port 6333)  -->  Vector Search
   -->  Redis  (port 6379)  -->  Cache
```

---

## Deployment

There are two ways to deploy: **Docker** (recommended for other machines) or **Manual** (recommended for development/local use).

---

## Option A — Docker Deployment (Recommended)

### Prerequisites

**1. Install Docker**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

**2. Install Docker Compose plugin**
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu focal stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-compose-plugin
docker compose version
```

**3. Install NVIDIA Container Toolkit** (for GPU access inside Docker)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**4. Verify GPU is accessible to Docker**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed.

---

### Setup

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd rag-system
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```

> The `.env.example` already has the correct Docker service hostnames (`qdrant`, `redis`, `vllm`). Only change the model name or thresholds if needed.

**3. Build and start all services**
```bash
docker compose up --build -d
```

**4. Watch the logs** (vLLM takes 5-10 minutes to download and load the model on first run)
```bash
docker compose logs -f vllm
```

Wait until you see:
```
Application startup complete.
```

**5. Verify everything is running**
```bash
docker compose ps
curl http://localhost:9000/health
curl http://localhost:6333/collections
```

**6. Open the app**

Visit `http://localhost:9000` in your browser.

---

### Stopping and Starting

```bash
# Stop all services
docker compose down

# Start again (no rebuild needed)
docker compose up -d

# Rebuild after code changes
docker compose up --build -d
```

---

### Changing the Model

Edit `.env`:
```
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
```

Then restart vLLM:
```bash
docker compose restart vllm
```

---

## Option B — Manual Deployment (Local / Development)

Use this if you want to run without Docker or already have services running.

### Prerequisites

- Python 3.11+
- Docker (for Qdrant and Redis only)
- NVIDIA GPU with drivers installed

### Setup

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd rag-system
```

**2. Create virtual environment and install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**3. Create your `.env` file**
```bash
cp .env.example .env
```

Edit `.env` and change service hostnames to `localhost`:
```
VLLM_HOST=http://localhost:8000
QDRANT_HOST=http://localhost:6333
REDIS_HOST=localhost
```

**4. Start Qdrant**
```bash
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

**5. Start Redis**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

**6. Start vLLM** (Terminal 1)
```bash
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000
```

Wait until you see `Application startup complete` before proceeding.

**7. Start FastAPI** (Terminal 2)
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 9000
```

**8. Open the app**

Visit `http://localhost:9000` in your browser.

---

## Expose Publicly (Optional)

To share the app over the internet without opening router ports:

**Install cloudflared**
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/cloudflared
```

**Create a public tunnel** (Terminal 3)
```bash
cloudflared tunnel --url http://localhost:9000
```

You will get a temporary public URL like `https://xxxx.trycloudflare.com`.

> This URL changes every time you restart cloudflared. For a permanent URL, create a free Cloudflare account and set up a named tunnel.

---

## Ingesting Documents

Once the system is running, ingest your documents via the API:

**Single file:**
```bash
curl -X POST http://localhost:9000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/absolute/path/to/document.pdf"}'
```

**Entire folder:**
```bash
curl -X POST http://localhost:9000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/absolute/path/to/folder"}'
```

Supported formats: `.pdf`, `.docx`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/health` | System health check |
| GET | `/services` | Check vLLM, Qdrant, Redis status |
| POST | `/ingest` | Ingest a single document |
| POST | `/ingest/folder` | Ingest all documents in a folder |
| POST | `/query` | Ask a question (RAG) |
| POST | `/generate-paper` | Generate a question paper |
| POST | `/download/pdf` | Download paper as PDF |
| POST | `/download/docx` | Download paper as DOCX |

---

## Troubleshooting

**`choices` key error when generating paper**
- vLLM is not running or the model name in `.env` does not match the loaded model
- Check: `curl http://localhost:8000/v1/models`
- Make sure `VLLM_MODEL` in `.env` matches exactly

**Port already in use**
```bash
docker ps -a | grep <port>
docker rm -f <container-name>
```

**Qdrant has no collections after restart**
- Make sure you mounted the correct data volume
- Docker: check `docker inspect qdrant | grep Mounts`
- Manual: verify `qdrant_data/collections/` exists

**vLLM runs out of GPU memory**
- Lower `--gpu-memory-utilization` to `0.75`
- Or use a smaller model: `Qwen/Qwen2.5-7B-Instruct-AWQ`

**Docker cannot access GPU**
- NVIDIA Container Toolkit must be installed (see Prerequisites)
- Run: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
