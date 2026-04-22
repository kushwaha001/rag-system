# RAG System — User Guide

A document-grounded question-answering and question-paper generation system. Upload your PDFs, DOCX, or PPTX files and chat with them, or auto-generate exam papers from their content.

---

## 1. What You Need Before Starting

| Requirement | Why | How to check |
|---|---|---|
| **Docker** (with Compose v2) | Runs all services | `docker --version` and `docker compose version` |
| **NVIDIA GPU** (≥ 24 GB VRAM recommended) | Runs the Qwen 14B model and reranker | `nvidia-smi` |
| **nvidia-container-toolkit** | Lets Docker use the GPU | `docker info \| grep -i nvidia` |
| **Disk space** (~ 60 GB free) | Model weights, image layers, ingested documents | `df -h .` |
| **Ports free**: 8000, 8001, 6333, 6379 | vLLM, API+UI, Qdrant, Redis | `ss -tlnp \| grep -E '8000\|8001\|6333\|6379'` |

If you don't have an NVIDIA GPU, the system won't run — vLLM needs one.

---

## 2. One-Time System Setup (first install only)

**2a. Install NVIDIA container toolkit** (lets Docker access the GPU):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

**2b. Configure Docker data directory** (needed if root filesystem has < 15 GB free — Docker image layers need ~10 GB):
```bash
sudo mkdir -p /home/$USER/docker-data
echo '{"data-root":"/home/'$USER'/docker-data","runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}},"default-runtime":"nvidia"}' | sudo tee /etc/docker/daemon.json
```

**2c. Restart Docker:**
```bash
sudo systemctl restart docker
```

Verify both are set:
```bash
docker info | grep -E 'Runtimes|Default Runtime|Docker Root Dir'
```
You should see `nvidia` in Runtimes and your home directory as Docker Root Dir.

---

## 3. One-Command Install

From the project root:

```bash
./start.sh
```

That's it. The script:
1. Checks prerequisites (Docker, Compose, NVIDIA runtime).
2. Creates `qdrant_data/` (vector DB) and `uploads/` (temporary file landing zone).
3. Builds the API container and pulls Qdrant, Redis, and vLLM images.
4. Starts everything in the background.

**First run downloads ~30 GB** (the vLLM image + the Qwen 14B model weights). Expect **3–5 minutes** of warm-up after containers start before vLLM is ready to serve requests.

Watch progress:
```bash
./start.sh logs
```

When you see `vllm ✔` in the status check, you're ready:
```bash
./start.sh status
```

Open the UI:
```
http://localhost:8001
```

---

## 4. The Three Tabs

### 3.1 Chat

Ask any question. The system:
1. Rewrites your query into several angles (so `"how does ABS work?"` also searches for `"anti-lock braking system"`).
2. Searches Qdrant for the most relevant chunks.
3. Reranks them with a cross-encoder.
4. Feeds the top chunks to the LLM as context.
5. Streams the answer back with citations (click a citation number to see the source chunk).

**Tips**
- Ask specific questions. `"What is the oil change interval for the MG 413W?"` works better than `"tell me about maintenance"`.
- If the answer says *"I don't have enough context,"* either the topic isn't in your ingested docs or you need to ingest more.

### 3.2 Question Paper (new: document selector)

Generate an exam paper on any topic from your ingested documents.

Fill in:
- **Topic** — what the paper is about (e.g., `"Fuel System"`).
- **Difficulty** — Easy / Medium / Hard.
- **Number of questions** — 1–30.
- **Question types** — MCQ, Short Answer, Long Answer, True/False (any combination).
- **Extra instructions** (optional) — e.g., `"focus on diagnostic procedures"`.

**Knowledge Source** (this is the new selector):

| Mode | What it does | Use when |
|---|---|---|
| **All Documents** (default) | Searches every ingested document | You want the broadest coverage |
| **Pick Documents** | Searches *only* the documents you tick | You want a paper *exclusively* from a specific manual / chapter / vehicle |

In **Pick Documents** mode:
- Use the filter box to narrow the list.
- Click `Select All` / `Clear` for bulk actions.
- You must tick at least one document before clicking **Generate**.

Once generated, you can download the paper as **PDF** or **DOCX**.

### 3.3 Custom Prompt

Same retrieval pipeline as Chat, but you write the full system + user prompt yourself. Useful when you want a specific output format (summaries, tables, JSON, etc.) without the default chat persona.

---

## 5. Ingesting Documents

From the **Documents** panel on the left:
1. Click **Upload** and pick one or more `.pdf`, `.docx`, or `.pptx` files.
2. Each file is parsed, chunked, embedded, and indexed into Qdrant.
3. A progress bar shows each file's stage (*parsing → chunking → embedding → indexing*).
4. When done, the file appears in the document list.

**Folder-aware indexing**: if you upload files through the *folder upload* path, the folder structure (e.g., `LVG/MG 413W/FUEL SYS/manual.pdf`) is stored alongside the chunks. The retriever uses this to boost path-matching hits — so a query about *"MG 413W fuel system"* will prefer chunks from the `MG 413W/FUEL SYS/` folder.

**Removing a document**: click the trash icon next to its name. This deletes every chunk for that file from Qdrant.

---

## 6. Everyday Commands

```bash
./start.sh          # start everything (or restart after reboot)
./start.sh stop     # stop all containers (data is preserved)
./start.sh restart  # bounce all services
./start.sh logs     # tail logs from all containers
./start.sh status   # show container + health status
./start.sh clean    # WARNING: deletes containers + Redis cache (qdrant_data is kept)
```

---

## 7. Troubleshooting

### *"NVIDIA runtime not detected"* at startup
Install the NVIDIA container toolkit:
```
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```
Then restart Docker: `sudo systemctl restart docker`.

### vLLM container keeps restarting / fails with OOM
The default model (Qwen 14B AWQ) needs ~18 GB VRAM. If you have less:
```bash
# in the project root
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ ./start.sh
```
Or lower GPU utilization:
```bash
VLLM_GPU_UTIL=0.70 ./start.sh
```

### UI loads but chat hangs
Check that vLLM finished loading:
```bash
./start.sh status
curl http://localhost:8000/health
```
Model load takes 3–5 min on first run.

### *"Port already in use"*
Something else is on 8001 (or 6333 / 6379 / 8000). Find it:
```bash
ss -tlnp | grep 8001
```
Either stop that process or edit the port in `docker-compose.yml`.

### Question paper generation says *"No relevant content found"*
- You're in **Pick Documents** mode and your selected docs don't cover the topic → switch to **All Documents** or pick broader files.
- The topic isn't well represented in your ingested corpus → ingest more material on that subject.

### My ingested documents disappeared after restart
They shouldn't — Qdrant data is stored in `./qdrant_data/`. If you ran `./start.sh clean`, note that it preserves `qdrant_data/` but removes Redis cache. If you manually deleted `qdrant_data/`, the collection is gone and you need to re-upload.

### I want to wipe everything and start fresh
```bash
./start.sh clean
rm -rf qdrant_data uploads
./start.sh
```

---

## 8. Architecture in One Paragraph

When you ask a question, the **FastAPI** service (`api/main.py`) rewrites it into multiple angles, embeds each with **BGE-Large**, searches **Qdrant** with a vehicle/system path-aware filter, fuses the results with Reciprocal Rank Fusion, reranks the top candidates with a **BGE-reranker cross-encoder**, applies a path-match boost, and sends the top chunks plus your question to **vLLM** (Qwen 14B AWQ). The answer streams back to the browser, with citations you can click to jump to the source chunk. **Redis** caches semantically-similar queries so repeat questions are instant.

For deep architecture details, see `PROJECT_OVERVIEW.md` and `CODEBASE.txt`.

---

## 9. Where Things Live

| Path | What's in it |
|---|---|
| `./qdrant_data/` | Vector DB (persists across restarts — back this up) |
| `./uploads/` | Transient upload landing zone (cleared on `clean`) |
| `./api/main.py` | All HTTP endpoints |
| `./retrieval/search.py` | Multi-angle retrieval + reranking |
| `./ingestion/pipeline.py` | Document parsing and chunking |
| `./frontend/index.html` | The entire UI (single-page) |
| `./docker-compose.yml` | Service topology |
| `./start.sh` | One-command launcher |

---

## 10. Getting Help

- **Logs**: `./start.sh logs` — always the first place to look.
- **Service health**: `./start.sh status`.
- **Project details**: `PROJECT_OVERVIEW.md` (architecture + flows), `CODEBASE.txt` (file-by-file reference).
