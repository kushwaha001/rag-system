from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct
from utils.embeddings import get_embedding_service
from utils.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from dotenv import load_dotenv

import uuid
import os
import logging

load_dotenv()

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("INGESTION")

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.html', '.csv'}

# =========================
# CHUNKING
# =========================
def chunk_text(text: str, chunk_size: int = 256, overlap: int = 64):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# =========================
# GET INGESTED FILES
# =========================
def get_ingested_files() -> set:
    try:
        client = get_qdrant_client()
        ingested = set()
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=["source"]
            )

            for r in results:
                if r.payload and "source" in r.payload:
                    ingested.add(r.payload["source"])

            if offset is None:
                break

        logger.info(f"📦 Found {len(ingested)} already ingested files")
        return ingested

    except Exception as e:
        logger.warning(f"⚠️ Could not check ingested files: {e}")
        return set()

# =========================
# PARSE DOCUMENT (NO SIGNAL)
# =========================
def parse_document(file_path: str):
    try:
        ext = os.path.splitext(file_path)[1].lower()

        # Handle txt/csv directly — Docling doesn't support these
        if ext in {'.txt', '.csv'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(), []

        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        full_text = doc.export_to_markdown() or ""
        image_references = []

        for picture in doc.pictures:
            try:
                page_num = getattr(picture, 'page_no', 0)
                page_num = int(page_num) if page_num else 0

                caption = ""
                try:
                    caption = picture.caption_text(doc) or ""
                except:
                    pass

                # Get surrounding text elements on same/adjacent pages
                prev_text = ""
                next_text = ""
                all_texts = []

                for text_elem in doc.texts:
                    try:
                        elem_page = getattr(text_elem, 'page_no', 0)
                        elem_page = int(elem_page) if elem_page else 0
                        text = text_elem.text.strip() if hasattr(text_elem, 'text') else ""
                        if len(text) > 20:
                            all_texts.append((elem_page, text))
                    except Exception:
                        pass

                all_texts.sort(key=lambda x: x[0])
                same_page = [t for p, t in all_texts if p == page_num]
                prev_page = [t for p, t in all_texts if p == page_num - 1]
                next_page_t = [t for p, t in all_texts if p == page_num + 1]

                if same_page:
                    words = " ".join(same_page).split()
                    mid = len(words) // 2
                    prev_text = " ".join(words[max(0, mid-75):mid])
                    next_text = " ".join(words[mid:min(len(words), mid+75)])
                elif prev_page or next_page_t:
                    prev_text = " ".join(prev_page[-1].split()[:100]) if prev_page else ""
                    next_text = " ".join(next_page_t[0].split()[:100]) if next_page_t else ""

                prev_clean = prev_text[:300] if prev_text else "Not available"
                next_clean = next_text[:300] if next_text else "Not available"

                reference = f"""Image context:
- File: {os.path.basename(file_path)}
- Page: {page_num + 1}
- Section context: {prev_clean}
- Following explanation: {next_clean}"""

                if caption:
                    reference += f"\n- Caption: {caption}"

                reference += f"\nThis image likely represents: content related to {prev_clean[:80]} as described in the surrounding technical documentation."

                image_references.append(reference)
                logger.info(f"🖼️ Image on page {page_num+1} referenced with context")

            except Exception as e:
                logger.warning(f"⚠️ Image processing failed: {e}")

        return full_text, image_references

    except Exception as e:
        logger.error(f"❌ Failed parsing {file_path}: {e}")
        return "", []

# =========================
# INGEST SINGLE DOCUMENT
# =========================
def ingest_document(file_path: str, section: str = "all"):
    logger.info(f"📄 Parsing: {os.path.basename(file_path)}")

    try:
        full_text, image_references = parse_document(file_path)

        # =========================
        # TEXT CHUNKS
        # =========================
        chunks = []
        if full_text and len(full_text.strip()) >= 50:
            chunks = chunk_text(full_text)
            logger.info(f"📝 {len(chunks)} text chunks")

        if image_references:
            logger.info(f"🖼️ {len(image_references)} image references")

        all_chunks = chunks + image_references

        if not all_chunks:
            return {"file": file_path, "chunks": 0, "status": "skipped"}

        # =========================
        # EMBEDDING (BATCHED)
        # =========================
        embedder = get_embedding_service()

        batch_size = 128
        embeddings = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            logger.info(f"⚡ Embedding batch {i//batch_size + 1}")
            embeddings.extend(embedder.embed_documents(batch))

        # =========================
        # STORE IN QDRANT
        # =========================
        client = get_qdrant_client()

        points = []
        for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
            chunk_type = "image_reference" if i >= len(chunks) else "text"

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk,
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "chunk_index": i,
                    "chunk_type": chunk_type,
                    "file_type": os.path.splitext(file_path)[1].lower(),
                    "section": section
                }
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)

        logger.info(f"✅ Stored {len(points)} chunks")

        return {
            "file": file_path,
            "chunks": len(points),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ Failed: {str(e)[:100]}")
        return {"file": file_path, "chunks": 0, "status": "failed"}

# =========================
# INGEST FOLDER
# =========================
def ingest_folder(folder_path: str, use_vlm: bool = False) -> dict:
    logger.info(f"📁 Scanning: {folder_path}")

    all_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            # Skip Office temp/lock files (start with ~$)
            if f.startswith('~$'):
                continue
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                all_files.append(os.path.join(root, f))

    logger.info(f"📊 Found {len(all_files)} files")

    ingested = get_ingested_files()
    remaining = [f for f in all_files if f not in ingested]

    logger.info(f"📋 Processing {len(remaining)} new files")

    total_chunks = 0

    for i, file in enumerate(remaining):
        logger.info(f"[{i+1}/{len(remaining)}]")

        res = ingest_document(file)
        total_chunks += res.get("chunks", 0)

    logger.info("=" * 50)
    logger.info(f"✅ DONE | Total chunks: {total_chunks}")
    logger.info("=" * 50)

    return {"status": "success", "chunks": total_chunks}