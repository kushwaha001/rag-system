from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct
from utils.embeddings import get_embedding_service
from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from dotenv import load_dotenv
import uuid
import os
import tempfile
import base64
import asyncio
import concurrent.futures
import signal

load_dotenv()

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.html', '.csv'}

# Files known to cause issues - add problematic filenames here
SKIP_FILES = {
    "Service Trg Handout of TATA Safari Strome Strome GS-800.pdf",  # example
}
def run_async(coro):
    """Run async coroutine in a new thread with its own event loop"""
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_ingested_files() -> set:
    """Get set of already ingested file paths from Qdrant"""
    try:
        client = get_qdrant_client()
        ingested = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["source"]
            )
            for r in results:
                if r.payload and "source" in r.payload:
                    ingested.add(r.payload["source"])
            if offset is None:
                break
        print(f"📦 Found {len(ingested)} already ingested file paths in Qdrant")
        return ingested
    except Exception as e:
        print(f"⚠️ Could not check ingested files: {e}")
        return set()


def render_pdf_page_as_image(pdf_path: str, page_num: int) -> str:
    """Render a PDF page as PNG image, return temp file path"""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pix.save(tmp.name)
    doc.close()
    return tmp.name


async def caption_page_with_vlm(image_path: str, page_num: int, source: str) -> str:
    """Send a page image to VLM for captioning"""
    import httpx
    VLM_HOST = os.getenv("VLM_HOST", "http://localhost:8002")
    VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{VLM_HOST}/v1/chat/completions",
                json={
                    "model": VLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """You are analyzing a page from a technical automotive/engineering document.
Describe in detail:
1. Type of content (diagram, schematic, table, chart, photo, text)
2. Main components, parts, or elements shown
3. Technical specifications, measurements, or values visible
4. Any text, labels, numbers, or codes visible
5. What this information is used for technically
Be specific and technical in your description."""
                                }
                            ]
                        }
                    ],
                    "max_tokens": 512
                }
            )
            result = response.json()
            caption = result["choices"][0]["message"]["content"]
            return f"[Page {page_num+1} from {os.path.basename(source)}]: {caption}"
    except Exception as e:
        print(f"  ❌ VLM caption failed for page {page_num}: {e}")
        return ""


async def caption_images_with_vlm(images: list, file_path: str) -> list:
    import httpx
    VLM_HOST = os.getenv("VLM_HOST", "http://localhost:8002")
    VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    captions = []
    for img_data in images:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_data["image"].save(tmp.name)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            os.unlink(tmp_path)
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{VLM_HOST}/v1/chat/completions",
                    json={
                        "model": VLM_MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{encoded}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": """Analyze this technical image and describe:
1. Type of diagram or image
2. Main components and elements
3. Technical information conveyed
4. Any visible text, numbers or labels
Be specific and technical."""
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 512
                    }
                )
                result = response.json()
                caption = result["choices"][0]["message"]["content"]
                existing = img_data.get("caption", "")
                full_caption = f"[Technical Image on page {img_data['page']}]"
                if existing:
                    full_caption += f" Docling caption: {existing}."
                full_caption += f" VLM description: {caption}"
                captions.append(full_caption)
                print(f"  ✅ Captioned image on page {img_data['page']}")
        except Exception as e:
            print(f"  ❌ Caption failed: {e}")
    return captions


def parse_document(file_path: str):
    """Simple parse without VLM"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {'.txt', '.csv'}:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(), []
    converter = DocumentConverter()
    result = converter.convert(file_path)
    full_text = result.document.export_to_markdown()
    images = []
    for element in result.document.pictures:
        try:
            img = element.get_image(result.document)
            if img:
                images.append({
                    "image": img,
                    "caption": element.caption_text(result.document) or "",
                    "page": getattr(element, 'page_no', 0)
                })
        except Exception:
            pass
    return full_text, images


def parse_document_with_vlm(file_path: str) -> tuple:
    """
    Parse document using Docling.
    For PDF pages where OCR fails (diagrams/images), use VLM to caption them.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in {'.txt', '.csv'}:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(), []

    converter = DocumentConverter()
    result = converter.convert(file_path)
    full_text = result.document.export_to_markdown()

    if ext != '.pdf':
        return full_text, []

    image_captions = []
    try:
        import fitz
        doc = fitz.open(file_path)
        print(f"  📖 PDF has {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            text_on_page = page.get_text().strip()

            is_diagram_page = (
                len(image_list) > 0 and len(text_on_page) < 100
            ) or (
                len(text_on_page) < 50 and page.rect.width > 0
            )

            if is_diagram_page:
                print(f"  🖼️  Page {page_num+1} appears to be diagram — sending to VLM...")
                try:
                    img_path = render_pdf_page_as_image(file_path, page_num)
                    caption = run_async(caption_page_with_vlm(img_path, page_num, file_path))
                    os.unlink(img_path)
                    if caption:
                        image_captions.append(caption)
                        print(f"  ✅ Captioned page {page_num+1}")
                except Exception as e:
                    print(f"  ❌ Failed page {page_num+1}: {e}")

        doc.close()
    except Exception as e:
        print(f"  ⚠️ PyMuPDF processing failed: {e}")

    return full_text, image_captions


def ingest_document(file_path: str, use_vlm: bool = False) -> dict:
    print(f"📄 Parsing: {os.path.basename(file_path)}")

    def timeout_handler(signum, frame):
        raise TimeoutError("File processing timed out after 5 minutes")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout per file

        try:
            if use_vlm and file_path.lower().endswith('.pdf'):
                full_text, image_captions = parse_document_with_vlm(file_path)
            else:
                full_text, images = parse_document(file_path)
                image_captions = []
                if use_vlm and images:
                    print(f"  🖼️  Found {len(images)} images — sending to VLM...")
                    image_captions = run_async(caption_images_with_vlm(images, file_path))
        finally:
            signal.alarm(0)  # Cancel timeout

        chunks = []
        if full_text and len(full_text.strip()) >= 50:
            chunks = chunk_text(full_text, chunk_size=512, overlap=50)
            print(f"  📝 {len(chunks)} text chunks")

        all_chunks = chunks + image_captions
        if not all_chunks:
            return {"file": file_path, "chunks": 0, "status": "skipped", "reason": "No content"}

        embedding_service = get_embedding_service()
        embeddings = embedding_service.embed_documents(all_chunks)

        client = get_qdrant_client()
        points = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk_type = "image_caption" if i >= len(chunks) else "text"
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "chunk_index": i,
                    "chunk_type": chunk_type,
                    "file_type": os.path.splitext(file_path)[1].lower()
                }
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  ✅ {len(points)} chunks ({len(chunks)} text + {len(image_captions)} image captions)")

        return {
            "file": file_path,
            "chunks": len(points),
            "text_chunks": len(chunks),
            "image_chunks": len(image_captions),
            "status": "success"
        }

    except TimeoutError as e:
        print(f"  ⏱️ Skipped (timeout): {str(e)}")
        return {"file": file_path, "chunks": 0, "status": "skipped", "reason": "timeout"}
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:100]}")
        return {"file": file_path, "chunks": 0, "status": "failed", "reason": str(e)[:200]}


def ingest_folder(folder_path: str, use_vlm: bool = False) -> dict:
    print(f"\n📁 Scanning: {folder_path}")
    print(f"🎨 VLM mode: {'ON' if use_vlm else 'OFF'}")

    # Step 1: Collect all supported files
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS and not filename.startswith('~$'):
                all_files.append(os.path.join(root, filename))

    print(f"📊 Found {len(all_files)} supported files")

    # Step 2: Skip already ingested files
    already_ingested = get_ingested_files()
    remaining = [f for f in all_files if f not in already_ingested]

    print(f"⏭️  Skipping {len(all_files) - len(remaining)} already ingested files")
    print(f"📋 Processing {len(remaining)} remaining files")

    # Step 3: Init results
    results = {"success": [], "failed": [], "skipped": []}

    # Step 4: Handle VLM startup
    vlm_was_already_running = False

    if use_vlm:
        import httpx as _httpx
        try:
            r = _httpx.get("http://localhost:8002/health", timeout=3)
            if r.status_code == 200:
                print("✅ VLM already running — skipping startup")
                vlm_was_already_running = True
            else:
                raise Exception("not healthy")
        except:
            from utils.vlm_manager import stop_vllm_14b, start_vlm
            print("\n🔄 Switching to VLM mode...")
            stop_vllm_14b()
            if not start_vlm():
                print("❌ VLM failed — falling back to text-only")
                use_vlm = False

    # Step 5: Process files
    try:
        for i, filepath in enumerate(remaining):
            filename = os.path.basename(filepath)

            if filename in SKIP_FILES:
                print(f"\n[{i+1}/{len(remaining)}] ⏭️ Skipping blacklisted: {filename}")
                results["skipped"].append({
                    "file": filepath,
                    "chunks": 0,
                    "status": "skipped",
                    "reason": "blacklisted"
                })
                continue

            print(f"\n[{i+1}/{len(remaining)}]")
            result = ingest_document(filepath, use_vlm=use_vlm)

            if result["status"] == "success":
                results["success"].append(result)
            elif result["status"] == "failed":
                results["failed"].append(result)
            else:
                results["skipped"].append(result)

    finally:
        if use_vlm and not vlm_was_already_running:
            print("\n🔄 Restoring 14B LLM...")
            from utils.vlm_manager import stop_vlm, start_vllm_14b
            stop_vlm()
            start_vllm_14b()

    # Step 6: Summary
    summary = {
        "folder": folder_path,
        "vlm_used": use_vlm,
        "total_files": len(all_files),
        "already_ingested": len(all_files) - len(remaining),
        "processed": len(remaining),
        "successful": len(results["success"]),
        "failed": len(results["failed"]),
        "skipped": len(results["skipped"]),
        "total_chunks": sum(r["chunks"] for r in results["success"]),
        "image_chunks": sum(r.get("image_chunks", 0) for r in results["success"]),
        "failed_files": [r["file"] for r in results["failed"]],
        "status": "complete"
    }

    print(f"\n{'='*50}")
    print(f"✅ Complete! {summary['successful']} files, {summary['total_chunks']} chunks")
    print(f"   Image captions: {summary['image_chunks']}")
    print(f"   Already ingested: {summary['already_ingested']}")
    print(f"{'='*50}")

    return summary