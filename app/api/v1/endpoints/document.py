"""
Document Endpoint — Upload PDF → Ekstrak (Docling/PyMuPDF) → Chunk → Embed → Simpan ke Qdrant.

Arsitektur ASYNC via Celery:
  POST   /document/upload             → kirim ke docling_worker via Celery → return task_id
  GET    /document/status/{task_id}   → cek status task
  GET    /document/course/{course_id} → list dokumen per course
  DELETE /document/course/{course_id} → hapus semua dokumen course dari Qdrant + registry
  GET    /document/tasks              → semua task

File di-upload ke shared_temp volume → docling_worker container bisa akses path yang sama.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.document import DocumentTaskListResponse
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
    summary="Upload PDF → Simpan ke Qdrant (Async via Celery)",
    description=f"""
Upload file PDF, kirim ke **docling_worker** via Celery, return `task_id`.

**Alur async:**
1. Validasi & simpan PDF ke shared volume
2. Dispatch ke `docling_worker` (Celery queue: `document`)
3. Return `task_id` + status `pending`
4. Poll `GET /document/status/{{task_id}}` untuk cek hasil

**Batasan:** File maks **{MAX_DOC_MB}MB** — hanya `.pdf`
    """,
)
async def upload_document(
    file: UploadFile = File(..., description=f"File PDF (maks {MAX_DOC_MB}MB)"),
    course_id: str = Query(..., description="ID course (wajib)"),
    doc_title: str = Query(default=None, description="Judul dokumen (opsional)"),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Hanya PDF yang didukung. Diterima: '{ext}'")

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_DOC_MB:
        raise HTTPException(status_code=413, detail=f"File terlalu besar ({size_mb:.1f}MB). Maks {MAX_DOC_MB}MB.")

    task_id     = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    from app.workers.document_worker import extract_document_task

    celery_task = extract_document_task.apply_async(
        kwargs={
            "task_id": task_id,
            "pdf_path": temp_path,
            "course_id": course_id,
            "doc_title": doc_title or file.filename,
        },
        task_id=task_id,
        queue="document",
    )

    task_data = {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "document_id": document_id,
        "status": "pending",
        "course_id": course_id,
        "filename": file.filename,
        "doc_title": doc_title or file.filename,
        "size_mb": round(size_mb, 2),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_task(task_id, task_data)

    logger.info(f"[DocUpload] Task dispatched | task_id={task_id} | celery_id={celery_task.id}")
    return task_data


# ── GET /status/{task_id} ─────────────────────────────────────────────────────

@router.get(
    "/status/{task_id}",
    summary="Cek Status Task Dokumen",
    description="""
Poll status task yang sedang diproses docling_worker.

**Status:** `pending` → `processing` → `done` / `error`
    """,
)
async def get_task_status(task_id: str):
    from app.workers.celery_app import celery

    tasks = _load_tasks()
    local = tasks.get(task_id)

    celery_result = celery.AsyncResult(task_id)
    celery_state = celery_result.state

    state_map = {
        "PENDING": "pending",
        "STARTED": "processing",
        "SUCCESS": "done",
        "FAILURE": "error",
        "RETRY": "processing",
        "REVOKED": "error",
    }

    if celery_state == "SUCCESS" and local and local.get("status") != "done":
        result = celery_result.result or {}
        local.update({
            "status": "done",
            "doc_title": result.get("doc_title"),
            "total_pages": result.get("total_pages"),
            "total_chars": result.get("total_chars"),
            "total_chunks": result.get("total_chunks"),
            "chunks_stored": result.get("chunks_stored"),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        _save_task(task_id, local)

    elif celery_state == "FAILURE" and local and local.get("status") != "error":
        local.update({
            "status": "error",
            "error": str(celery_result.result),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        _save_task(task_id, local)

    if not local:
        return {"task_id": task_id, "status": state_map.get(celery_state, "pending"), "celery_state": celery_state}

    tasks = _load_tasks()
    return tasks.get(task_id, {"task_id": task_id, "status": state_map.get(celery_state, "pending")})


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
        raise HTTPException(status_code=404, detail=f"Tidak ada dokumen untuk course_id '{course_id}'.")
    return DocumentTaskListResponse(total=len(task_list), tasks=task_list)


# ── DELETE /course/{course_id} ────────────────────────────────────────────────

@router.delete(
    "/course/{course_id}",
    summary="Hapus Semua Dokumen berdasarkan course_id",
)
async def delete_docs_by_course(course_id: str):
    tasks   = _load_tasks()
    related = [t for t in tasks.values() if t.get("course_id") == course_id]
    if not related:
        raise HTTPException(status_code=404, detail=f"Tidak ada dokumen untuk course_id '{course_id}'.")

    try:
        from app.services.qdrant_service import QdrantService
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        qdrant = QdrantService()
        loop   = asyncio.get_event_loop()

        count_result = await loop.run_in_executor(
            None,
            lambda: qdrant.client.count(
                collection_name=qdrant.collection,
                count_filter=Filter(must=[
                    FieldCondition(key="course_id", match=MatchValue(value=course_id)),
                    FieldCondition(key="type",      match=MatchValue(value="document")),
                ]),
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
                        filter=Filter(must=[
                            FieldCondition(key="course_id", match=MatchValue(value=course_id)),
                            FieldCondition(key="type",      match=MatchValue(value="document")),
                        ])
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
    status: str = Query(default=None, description="Filter: 'pending', 'processing', 'done', 'error'"),
):
    tasks     = _load_tasks()
    task_list = list(tasks.values())
    if status:
        task_list = [t for t in task_list if t.get("status") == status]
    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return DocumentTaskListResponse(total=len(task_list), tasks=task_list)