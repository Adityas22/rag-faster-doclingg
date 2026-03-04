"""
Generate Endpoint — RAG content generation via Celery + Redis.

Arsitektur ASYNC (sama dengan document/audio):
  POST   /generate/flashcard           → dispatch ke generate_worker → return task_id
  POST   /generate/quiz/mc             → dispatch ke generate_worker → return task_id
  POST   /generate/quiz/essay          → dispatch ke generate_worker → return task_id
  POST   /generate/summary             → dispatch ke generate_worker → return task_id
  POST   /generate/material            → dispatch ke generate_worker → return task_id
  GET    /generate/status/{task_id}    → cek status dari Redis
  GET    /generate/tasks               → list semua task generate

Flow per endpoint:
  1. Terima GenerateRequest dari client
  2. Generate task_id (UUID)
  3. Simpan status 'pending' ke Redis
  4. Dispatch task ke celery_generate worker (queue: generate)
  5. Return GenerateTaskResponse dengan task_id + status_url
  6. Client poll GET /generate/status/{task_id}
  7. Worker selesai → simpan result ke Redis
  8. GET status → return GenerateStatusResponse dengan result

PENTING: request.content_type di-override per endpoint via model_copy()
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from app.schemas.generate import (
    GenerateRequest,
    GenerateTaskResponse,
    GenerateStatusResponse,
)
from app.services.redis_service import (
    save_task_result,
    get_task_result,
    list_tasks,
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _dispatch_generate_task(
    request: GenerateRequest,
    content_type: str,
) -> GenerateTaskResponse:
    """
    Dispatch task generate ke Celery queue 'generate' dan simpan
    status awal ke Redis. Dipakai oleh semua endpoint POST.

    Args:
        request: GenerateRequest dari client
        content_type: jenis konten yang di-override per endpoint

    Returns:
        GenerateTaskResponse dengan task_id dan status_url
    """
    from app.workers.generate_worker import generate_content_task

    # Override content_type sesuai endpoint (Pydantic v2 safe)
    req = request.model_copy(update={"content_type": content_type})

    task_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Simpan status awal ke Redis sebelum dispatch ───────────────────────────
    save_task_result(task_id, {
        "task_id": task_id,
        "status": "pending",
        "content_type": content_type,
        "course_id": req.course_id,
        "source_file": req.source_file,
        "topic": req.topic,
        "count": req.count,
        "difficulty": req.difficulty,
        "language": req.language,
        "created_at": created_at,
    })

    # ── Dispatch ke Celery queue 'generate' ───────────────────────────────────
    celery_task = generate_content_task.apply_async(
        kwargs={
            "task_id": task_id,
            "course_id": req.course_id,
            "content_type": content_type,
            "topic": req.topic,
            "source_file": req.source_file,
            "count": req.count,
            "difficulty": req.difficulty,
            "language": req.language,
        },
        task_id=task_id,
        queue="generate",
    )

    # ── Update celery_task_id ke Redis ─────────────────────────────────────────
    save_task_result(task_id, {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "pending",
        "content_type": content_type,
        "course_id": req.course_id,
        "source_file": req.source_file,
        "topic": req.topic,
        "count": req.count,
        "difficulty": req.difficulty,
        "language": req.language,
        "created_at": created_at,
    })

    logger.info(
        f"[Generate] Dispatched | task_id={task_id} | celery_id={celery_task.id} "
        f"| type={content_type} | course={req.course_id} | topic={req.topic!r}"
    )

    return GenerateTaskResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        status="pending",
        content_type=content_type,
        course_id=req.course_id,
        source_file=req.source_file,
        topic=req.topic,
        count=req.count,
        difficulty=req.difficulty,
        language=req.language,
        created_at=created_at,
        status_url=f"/api/v1/generate/status/{task_id}",
    )


# ── POST /flashcard ────────────────────────────────────────────────────────────

@router.post(
    "/flashcard",
    response_model=GenerateTaskResponse,
    summary="Generate Flashcard (Async)",
    description="""
Generate flashcard dari materi di Qdrant — diproses async via Celery.

**Flow:**
1. POST request ini → return `task_id` + `status_url`
2. Poll `GET /generate/status/{task_id}` hingga status = `done`
3. Ambil hasil dari field `result.flashcards`

**Format kartu:**
- **front** — heading / kata kunci singkat (tampilan depan)
- **back** — penjelasan detail (tampilan belakang)

`course_id` dan `source_file` opsional — kosong = ambil dari seluruh data (global).
`difficulty` tidak perlu diisi.
    """,
)
async def generate_flashcard(request: GenerateRequest):
    try:
        return _dispatch_generate_task(request, content_type="flashcard")
    except Exception as e:
        logger.error(f"[Generate] Dispatch flashcard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /quiz/mc ──────────────────────────────────────────────────────────────

@router.post(
    "/quiz/mc",
    response_model=GenerateTaskResponse,
    summary="Generate Quiz Pilihan Ganda (Async)",
    description="""
Generate soal pilihan ganda (A/B/C/D + kunci jawaban + penjelasan) — async via Celery.

**Flow:**
1. POST → return `task_id`
2. Poll `GET /generate/status/{task_id}`
3. Hasil ada di `result.items`

`difficulty`: `easy` | `medium` | `hard`
    """,
)
async def generate_quiz_mc(request: GenerateRequest):
    try:
        return _dispatch_generate_task(request, content_type="quiz_mc")
    except Exception as e:
        logger.error(f"[Generate] Dispatch quiz_mc error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /quiz/essay ───────────────────────────────────────────────────────────

@router.post(
    "/quiz/essay",
    response_model=GenerateTaskResponse,
    summary="Generate Quiz Essay (Async)",
    description="""
Generate soal essay (pertanyaan + contoh jawaban + poin-poin kunci) — async via Celery.

**Flow:**
1. POST → return `task_id`
2. Poll `GET /generate/status/{task_id}`
3. Hasil ada di `result.items`

`difficulty`: `easy` | `medium` | `hard`
    """,
)
async def generate_quiz_essay(request: GenerateRequest):
    try:
        return _dispatch_generate_task(request, content_type="quiz_essay")
    except Exception as e:
        logger.error(f"[Generate] Dispatch quiz_essay error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /summary ──────────────────────────────────────────────────────────────

@router.post(
    "/summary",
    response_model=GenerateTaskResponse,
    summary="Generate Ringkasan Materi (Async)",
    description="""
Buat ringkasan dari materi di Qdrant — async via Celery.

**Flow:**
1. POST → return `task_id`
2. Poll `GET /generate/status/{task_id}`
3. Hasil ada di `result.summary` dan `result.key_points`

`difficulty` tidak perlu diisi.
`count` = jumlah key_points yang dihasilkan.
    """,
)
async def generate_summary(request: GenerateRequest):
    try:
        return _dispatch_generate_task(request, content_type="summary")
    except Exception as e:
        logger.error(f"[Generate] Dispatch summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /material ─────────────────────────────────────────────────────────────

@router.post(
    "/material",
    response_model=GenerateTaskResponse,
    summary="Generate Materi Pelajaran Lengkap (Async)",
    description="""
Generate materi lengkap (judul, pembuka, isi, contoh, latihan) — async via Celery.

**Flow:**
1. POST → return `task_id`
2. Poll `GET /generate/status/{task_id}`
3. Hasil ada di `result.title`, `result.content`, `result.examples`, dll.

`difficulty` tidak perlu diisi.
`count` tidak berlaku untuk material.
    """,
)
async def generate_material(request: GenerateRequest):
    try:
        return _dispatch_generate_task(request, content_type="material")
    except Exception as e:
        logger.error(f"[Generate] Dispatch material error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /status/{task_id} ──────────────────────────────────────────────────────

@router.get(
    "/status/{task_id}",
    response_model=GenerateStatusResponse,
    summary="Cek Status Task Generate",
    description="""
Poll status task generate yang sedang diproses oleh celery_generate worker.

**Status lifecycle:**
```
pending → processing → done
                     ↘ error
```

Jika `status = done`, field `result` akan berisi konten yang di-generate.
Jika `status = error`, field `error` berisi pesan error.

Sumber data: **Redis** (key: `gen_task:{task_id}`)
    """,
)
async def get_generate_status(task_id: str):
    # ── Ambil dari Redis ───────────────────────────────────────────────────────
    data = get_task_result(task_id)

    if data:
        return GenerateStatusResponse(**{
            k: v for k, v in data.items()
            if k in GenerateStatusResponse.model_fields
        })

    # ── Fallback: cek Celery backend langsung (jika Redis key expired) ─────────
    try:
        from app.workers.celery_app import celery
        celery_result = celery.AsyncResult(task_id)
        celery_state = celery_result.state

        state_map = {
            "PENDING":  "pending",
            "STARTED":  "processing",
            "SUCCESS":  "done",
            "FAILURE":  "error",
            "RETRY":    "processing",
            "REVOKED":  "error",
        }

        status = state_map.get(celery_state, "pending")

        if celery_state == "SUCCESS":
            result_data = celery_result.result or {}
            return GenerateStatusResponse(
                task_id=task_id,
                status="done",
                result=result_data,
            )

        if celery_state == "FAILURE":
            return GenerateStatusResponse(
                task_id=task_id,
                status="error",
                error=str(celery_result.result),
            )

        return GenerateStatusResponse(task_id=task_id, status=status)

    except Exception as e:
        logger.warning(f"[Generate] Status fallback error: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' tidak ditemukan atau sudah expired (TTL 1 jam).",
        )


# ── GET /tasks ─────────────────────────────────────────────────────────────────

@router.get(
    "/tasks",
    summary="List Semua Task Generate",
    description="""
Ambil semua task generate yang masih ada di Redis (TTL 1 jam).

Gunakan query param `status` untuk filter:
- `pending` — belum diproses
- `processing` — sedang diproses worker
- `done` — selesai, result tersedia
- `error` — gagal
    """,
)
async def list_generate_tasks(
    status: str = Query(
        default=None,
        description="Filter by status: pending | processing | done | error",
    ),
    content_type: str = Query(
        default=None,
        description="Filter by content_type: flashcard | quiz_mc | quiz_essay | summary | material",
    ),
):
    tasks = list_tasks(prefix="gen_task:")

    if status:
        tasks = [t for t in tasks if t.get("status") == status]

    if content_type:
        tasks = [t for t in tasks if t.get("content_type") == content_type]

    return {
        "total": len(tasks),
        "tasks": tasks,
    }