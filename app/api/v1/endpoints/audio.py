"""
Audio Endpoint — Upload & Transcribe Audio → Chunk → Embed → Simpan ke Qdrant.

Arsitektur ASYNC via Celery:
  POST /transcribe  → kirim ke whisper_worker via Celery → return task_id + status
  POST /record      → sama, alias untuk rekaman browser
  GET  /status/{task_id}    → cek status task (pending/processing/done/error)
  GET  /course/{course_id}  → list task per course
  GET  /tasks               → semua task
  DELETE /course/{course_id} → hapus data Qdrant + registry

File di-upload ke shared_temp volume → whisper_worker container bisa akses path yang sama.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.audio import TaskListResponse
from app.config import settings
from app.utils.logger import get_logger
import uuid, os, shutil, json
from datetime import datetime

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TASKS_FILE = os.path.join("output", "tasks.json")
MAX_AUDIO_MB = settings.MAX_AUDIO_MB


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


def _validate_audio_file(file: UploadFile, extra_ext: set = None):
    allowed = ALLOWED_EXTENSIONS | (extra_ext or set())
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{ext}'. Gunakan: {', '.join(sorted(allowed))}",
        )
    return ext


def _check_size(file: UploadFile) -> float:
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > MAX_AUDIO_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File terlalu besar ({size_mb:.1f}MB). Maks {MAX_AUDIO_MB}MB. "
                f"Gunakan format MP3 atau WAV 16kHz mono untuk rekaman 3 menit."
            ),
        )
    return size_mb


def _save_temp(file: UploadFile, task_id: str) -> str:
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_path


def _dispatch_audio_task(
    task_id: str,
    temp_path: str,
    language: str,
    course_id: str,
    filename: str,
    size_mb: float,
    audio_type: str = "audio",
) -> dict:
    from app.workers.audio_worker import transcribe_audio_task

    celery_task = transcribe_audio_task.apply_async(
        kwargs={
            "task_id": task_id,
            "audio_path": temp_path,
            "language": language,
            "course_id": course_id,
        },
        task_id=task_id,
        queue="audio",
    )

    logger.info(
        f"[Audio] Task dispatched → whisper_worker | "
        f"task_id={task_id} | file={filename} ({size_mb:.1f}MB) | "
        f"celery_id={celery_task.id}"
    )

    return {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "pending",
        "course_id": course_id,
        "source_file": filename,
        "type": audio_type,
        "size_mb": round(size_mb, 2),
        "language_requested": language,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── POST /transcribe ──────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    summary="Upload & Transkripsi Audio → Simpan ke Qdrant (Async)",
    description=f"""
Upload file audio, kirim ke **whisper_worker** via Celery, return `task_id`.

**Alur async:**
1. Upload & validasi file (maks **{MAX_AUDIO_MB}MB**)
2. Simpan ke shared volume → dispatch ke `whisper_worker`
3. Return `task_id` + status `pending`
4. Poll `GET /audio/status/{{task_id}}` untuk cek hasil

**Format:** `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac`
    """,
)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(default="auto", description="Bahasa: 'auto', 'id', 'en'"),
    course_id: str = Query(..., description="ID course dari Laravel (wajib)"),
):
    _validate_audio_file(file)
    size_mb = _check_size(file)
    task_id = str(uuid.uuid4())
    temp_path = _save_temp(file, task_id)
    task_data = _dispatch_audio_task(
        task_id=task_id,
        temp_path=temp_path,
        language=language,
        course_id=course_id,
        filename=file.filename,
        size_mb=size_mb,
        audio_type="audio",
    )
    _save_task(task_id, task_data)
    return task_data


# ── POST /record ──────────────────────────────────────────────────────────────

@router.post(
    "/record",
    summary="Upload Rekaman Audio → Simpan ke Qdrant (Async)",
    description=f"""
Endpoint untuk upload file rekaman audio.
Mendukung format `.mp3` dan `.aac` selain format audio standar lainnya.

**Format yang didukung:** `.mp3` `.aac` `.wav` `.m4a` `.ogg` `.flac`
    """,
)
async def record_audio(
    file: UploadFile = File(...),
    language: str = Query(default="auto", description="Bahasa: 'auto', 'id', 'en'"),
    course_id: str = Query(..., description="ID course dari Laravel (wajib)"),
):
    extra = {".mp3", ".aac"}
    ext = _validate_audio_file(file, extra_ext=extra)
    if not ext:
        file.filename = f"recording_{uuid.uuid4().hex[:8]}.mp3"
    size_mb = _check_size(file)
    task_id = str(uuid.uuid4())
    temp_path = _save_temp(file, task_id)
    task_data = _dispatch_audio_task(
        task_id=task_id,
        temp_path=temp_path,
        language=language,
        course_id=course_id,
        filename=file.filename,
        size_mb=size_mb,
        audio_type="audio_record",
    )
    _save_task(task_id, task_data)
    return task_data


# ── GET /status/{task_id} ─────────────────────────────────────────────────────

@router.get(
    "/status/{task_id}",
    summary="Cek Status Task Audio",
    description="""
Poll status task yang sedang diproses whisper_worker.

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
            "language": result.get("language"),
            "duration": result.get("duration"),
            "total_chunks": result.get("total_chunks"),
            "chunks_stored": result.get("chunks_stored"),
            "transcription_time": result.get("transcription_time"),
            "realtime_factor": result.get("realtime_factor"),
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
    response_model=TaskListResponse,
    summary="Lihat Semua Task berdasarkan course_id",
)
async def get_tasks_by_course(course_id: str):
    tasks = _load_tasks()
    task_list = [t for t in tasks.values() if t.get("course_id") == course_id]
    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    if not task_list:
        raise HTTPException(status_code=404, detail=f"Tidak ada task untuk course_id '{course_id}'.")
    return TaskListResponse(total=len(task_list), tasks=task_list)


# ── GET /tasks ────────────────────────────────────────────────────────────────

@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="Lihat Semua Task Audio",
)
async def list_all_tasks(
    status: str = Query(default=None, description="Filter: 'pending', 'processing', 'done', 'error'"),
):
    tasks = _load_tasks()
    task_list = list(tasks.values())
    if status:
        task_list = [t for t in task_list if t.get("status") == status]
    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return TaskListResponse(total=len(task_list), tasks=task_list)


# ── DELETE /course/{course_id} ────────────────────────────────────────────────

@router.delete(
    "/course/{course_id}",
    summary="Hapus Semua Data Audio berdasarkan course_id",
)
async def delete_by_course_id(course_id: str):
    tasks = _load_tasks()
    related_tasks = [t for t in tasks.values() if t.get("course_id") == course_id]
    if not related_tasks:
        raise HTTPException(status_code=404, detail=f"Tidak ada data untuk course_id '{course_id}'.")

    try:
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        deleted_vectors = await qdrant.delete_by_course_id(course_id)
        logger.info(f"[Delete] Qdrant: {deleted_vectors} vectors | course_id={course_id}")

        task_ids_deleted = []
        for tid, task_data in list(tasks.items()):
            if task_data.get("course_id") == course_id:
                task_ids_deleted.append(tid)
                del tasks[tid]

        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        return {
            "course_id": course_id,
            "status": "deleted",
            "deleted_vectors": deleted_vectors,
            "deleted_tasks": len(task_ids_deleted),
            "task_ids_deleted": task_ids_deleted,
            "message": f"Semua data audio course '{course_id}' berhasil dihapus.",
        }
    except Exception as e:
        logger.error(f"[Delete] Error course_id={course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))