"""
Document Endpoint — Upload PDF → Ekstrak (PyMuPDF) → Chunk → Embed → Simpan ke Qdrant.

Struktur IDENTIK dengan audio endpoint:
  POST   /document/upload             → upload & proses PDF (sync)
  GET    /document/course/{course_id} → list dokumen per course
  DELETE /document/course/{course_id} → hapus semua dokumen course dari Qdrant + registry
  GET    /document/tasks              → semua task (filter opsional by status)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.document import DocumentUploadResponse, DocumentTaskListResponse
from app.config import settings
from app.utils.logger import get_logger
import uuid, os, shutil, json, asyncio
from datetime import datetime

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_DOC_MB = settings.MAX_DOCUMENT_MB
TASKS_FILE = os.path.join("output", "doc_tasks.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tasks() -> dict:
    if not os.path.exists(TASKS_FILE):
        return {}
    with open(TASKS_FILE, "r") as f:
        return json.load(f)


def _save_task(task_id: str, data: dict):
    tasks = _load_tasks()
    tasks[task_id] = data
    os.makedirs("output", exist_ok=True)
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload & Ekstraksi PDF → Simpan ke Qdrant",
    description=f"""
Upload file PDF, ekstrak teks via **PyMuPDF** (zero DLL dependency),
chunking via **RecursiveCharacterTextSplitter** (LangChain),
generate embedding via **Gemini**, simpan ke **Qdrant** (append mode).

**Batasan:** File maks **{MAX_DOC_MB}MB** — hanya `.pdf`

**Alur:**
1. Validasi & simpan PDF sementara
2. PyMuPDF → ekstrak teks + deteksi heading via font size → `build_sections()`
3. `RecursiveCharacterTextSplitter` → chunk_size=1000, overlap=150
   - Setiap chunk di-enrich: `SECTION: {{title}}\\n---\\n{{konten}}`
4. Gemini → generate embedding vectors
5. Qdrant → simpan dengan metadata `course_id`, `document_id`, `section`, `page`
    """,
)
async def upload_document(
    file: UploadFile = File(..., description=f"File PDF (maks {MAX_DOC_MB}MB)"),
    course_id: str = Query(..., description="ID course (wajib)"),
    doc_title: str = Query(default=None, description="Judul dokumen (opsional)"),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Hanya PDF yang didukung. Diterima: '{ext}'",
        )

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_DOC_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar ({size_mb:.1f}MB). Maks {MAX_DOC_MB}MB.",
        )

    task_id     = str(uuid.uuid4())   # internal registry only
    document_id = str(uuid.uuid4())   # ID unik per dokumen, dikembalikan ke user
    temp_path   = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        f"[DocUpload] START | file={file.filename} ({size_mb:.1f}MB) | "
        f"course_id={course_id} | document_id={document_id}"
    )

    try:
        # ── Step 1: Ekstraksi PDF via PyMuPDF ────────────────────────────────
        logger.info("[DocUpload] Step 1/3 — PyMuPDF extraction + build_sections...")
        from app.services.docling_service import DoclingService

        pdf_svc = DoclingService()
        if not pdf_svc.is_available():
            raise RuntimeError(
                "PyMuPDF atau langchain-text-splitters tidak terinstall.\n"
                "Jalankan: pip install PyMuPDF langchain-text-splitters"
            )

        loop = asyncio.get_event_loop()
        doc_result = await loop.run_in_executor(
            None,
            lambda: pdf_svc.extract_pdf(temp_path, doc_title=doc_title or file.filename),
        )

        sections   = doc_result.get("sections", [])
        raw_chunks = doc_result.get("chunks", [])

        logger.info(
            f"[DocUpload] ✅ Step 1 Done | "
            f"pages={doc_result['total_pages']} | "
            f"chars={doc_result['total_chars']} | "
            f"sections={len(sections)} | "
            f"chunks={len(raw_chunks)}"
        )

        if not raw_chunks:
            raise ValueError(
                "Tidak ada chunk yang dihasilkan. "
                "Pastikan PDF berisi teks yang dapat dibaca (bukan scan/image only)."
            )

        # ── Step 2: Gemini Embedding ──────────────────────────────────────────
        logger.info("[DocUpload] Step 2/3 — Gemini Embedding...")
        from app.services.embedding_service import EmbeddingService

        chunk_dicts = [
            {
                "text": chunk["content"],
                "metadata": {
                    "course_id":   course_id,
                    "document_id": document_id,
                    "task_id":     task_id,
                    "source_file": doc_result["source_file"],
                    "doc_title":   doc_result["doc_title"],
                    "type":        "document",
                    "section":     chunk["metadata"]["section"],
                    "page":        chunk["metadata"]["page"],
                    "chunk_id":    chunk["metadata"]["chunk_id"],
                    "total_words": chunk["metadata"]["total_words"],
                    "chunk_index": i,
                },
            }
            for i, chunk in enumerate(raw_chunks)
        ]

        embed_svc = EmbeddingService()
        embedded  = await embed_svc.embed_chunks(chunk_dicts)
        logger.info(f"[DocUpload] ✅ Step 2 Done | vectors={len(embedded)}")

        # ── Step 3: Simpan ke Qdrant ──────────────────────────────────────────
        logger.info("[DocUpload] Step 3/3 — Saving to Qdrant...")
        from app.services.qdrant_service import QdrantService

        qdrant        = QdrantService()
        chunks_stored = await qdrant.upsert_vectors(embedded)
        logger.info(f"[DocUpload] ✅ Step 3 Done | stored={chunks_stored}")

        # Simpan ke registry internal (task_id dipakai sebagai key)
        _save_task(task_id, {
            "task_id":        task_id,
            "document_id":    document_id,
            "status":         "done",
            "course_id":      course_id,
            "filename":       file.filename,
            "doc_title":      doc_result["doc_title"],
            "total_pages":    doc_result["total_pages"],
            "total_chars":    doc_result["total_chars"],
            "total_sections": len(sections),
            "total_chunks":   len(raw_chunks),
            "chunks_stored":  chunks_stored,
            "created_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        chunks_preview = [
            {
                "chunk_index":    i,
                "section":        c["metadata"]["section"],
                "page":           c["metadata"]["page"],
                "chunk_id":       c["metadata"]["chunk_id"],
                "total_words":    c["metadata"]["total_words"],
                "content_preview": (
                    c["content"][:200] + "..."
                    if len(c["content"]) > 200
                    else c["content"]
                ),
            }
            for i, c in enumerate(raw_chunks)
        ]

        return DocumentUploadResponse(
            document_id=document_id,
            status="done",
            course_id=course_id,
            source_file=doc_result["source_file"],
            doc_title=doc_result["doc_title"],
            filename=file.filename,
            size_mb=round(size_mb, 2),
            total_pages=doc_result["total_pages"],
            total_chars=doc_result["total_chars"],
            total_sections=len(sections),
            total_chunks=len(raw_chunks),
            chunks_stored=chunks_stored,
            chunks=chunks_preview,
            message=(
                f"PDF '{file.filename}' berhasil diproses "
                f"({len(sections)} sections, {len(raw_chunks)} chunks) "
                f"dan disimpan ke Qdrant."
            ),
        )

    except Exception as e:
        _save_task(task_id, {
            "task_id":     task_id,
            "document_id": document_id,
            "status":      "error",
            "course_id":   course_id,
            "filename":    file.filename,
            "error":       str(e),
            "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.error(f"[DocUpload] ❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── GET /course/{course_id} ───────────────────────────────────────────────────

@router.get(
    "/course/{course_id}",
    response_model=DocumentTaskListResponse,
    summary="Lihat Semua Dokumen berdasarkan course_id",
)
async def get_docs_by_course(course_id: str):
    tasks     = _load_tasks()
    task_list = [t for t in tasks.values() if t.get("course_id") == course_id]
    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    if not task_list:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada dokumen untuk course_id '{course_id}'.",
        )

    return DocumentTaskListResponse(total=len(task_list), tasks=task_list)


# ── DELETE /course/{course_id} ────────────────────────────────────────────────

@router.delete(
    "/course/{course_id}",
    summary="Hapus Semua Dokumen berdasarkan course_id",
    description="""
Hapus semua vector dokumen di **Qdrant** dan registry lokal berdasarkan `course_id`.

> ⚠️ Hanya menghapus vector bertipe `document`. Vector audio tidak terpengaruh.
    """,
)
async def delete_docs_by_course(course_id: str):
    tasks   = _load_tasks()
    related = [t for t in tasks.values() if t.get("course_id") == course_id]

    if not related:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada dokumen untuk course_id '{course_id}'.",
        )

    try:
        from app.services.qdrant_service import QdrantService
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        qdrant = QdrantService()
        loop   = asyncio.get_event_loop()

        count_result = await loop.run_in_executor(
            None,
            lambda: qdrant.client.count(
                collection_name=qdrant.collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="course_id", match=MatchValue(value=course_id)),
                        FieldCondition(key="type",      match=MatchValue(value="document")),
                    ]
                ),
                exact=True,
            ),
        )
        total = count_result.count

        if total > 0:
            await loop.run_in_executor(
                None,
                lambda: qdrant.client.delete(
                    collection_name=qdrant.collection,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="course_id", match=MatchValue(value=course_id)),
                                FieldCondition(key="type",      match=MatchValue(value="document")),
                            ]
                        )
                    ),
                ),
            )

        logger.info(f"[DocDelete] {total} vectors dihapus | course_id={course_id}")

        deleted_ids = []
        for tid, tdata in list(tasks.items()):
            if tdata.get("course_id") == course_id:
                deleted_ids.append(tid)
                del tasks[tid]

        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        return {
            "course_id":        course_id,
            "status":           "deleted",
            "deleted_vectors":  total,
            "deleted_tasks":    len(deleted_ids),
            "task_ids_deleted": deleted_ids,
            "message":          f"Semua dokumen course '{course_id}' berhasil dihapus.",
        }

    except Exception as e:
        logger.error(f"[DocDelete] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /tasks ────────────────────────────────────────────────────────────────

@router.get(
    "/tasks",
    response_model=DocumentTaskListResponse,
    summary="Lihat Semua Task Dokumen",
)
async def list_all_doc_tasks(
    status: str = Query(default=None, description="Filter: 'done' atau 'error'"),
):
    tasks     = _load_tasks()
    task_list = list(tasks.values())

    if status:
        task_list = [t for t in task_list if t.get("status") == status]

    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return DocumentTaskListResponse(total=len(task_list), tasks=task_list)