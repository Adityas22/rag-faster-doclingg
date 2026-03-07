"""
Generate Endpoint — Material, Flashcard, dan Quiz via Celery + Redis.

Skenario:
  1. User upload PDF/Audio → sudah tersimpan di Qdrant
  2. POST /generate/material   → generate materi → simpan di Laravel (MySQL)
  3. POST /generate/flashcard  → generate flashcard dari Qdrant (opsional + material_context)
  4. POST /generate/quiz       → generate quiz dari Qdrant (opsional + material_context)
                                  quiz_type: essay | multiple_choice_single | multiple_choice_multiple

Semua endpoint ASYNC via Celery:
  POST   → dispatch → return {task_id, status_url}
  GET    /generate/status/{task_id}  → poll hasil dari Redis
  GET    /generate/tasks             → list semua task

Status lifecycle: pending → processing → done / error
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.schemas.generate import (
    GenerateMaterialRequest,
    GenerateFlashcardRequest,
    GenerateQuizRequest,
    GenerateTaskResponse,
    GenerateStatusResponse,
)
from app.services.redis_service import save_task_result, get_task_result, list_tasks
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/material
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/material",
    response_model=GenerateTaskResponse,
    summary="Generate Materi Pelajaran Lengkap (Async)",
    description="""
Generate materi pembelajaran dari PDF/Audio yang sudah di-upload ke Qdrant.
Cukup kirim `course_id` — semua file dalam course tersebut akan digunakan sebagai sumber.

**Output JSON** (tersedia saat `status=done`):
```json
{
  "title": "Judul materi",
  "introduction": "Paragraf pembuka",
  "sections": [{"heading": "Sub-topik", "body": "Penjelasan..."}],
  "content": "Isi materi lengkap",
  "key_points": ["Poin 1", "Poin 2"],
  "summary": "Kesimpulan singkat"
}
```

**Flow:**
1. POST request → return `task_id` + `status_url`
2. Poll `GET /generate/status/{task_id}` hingga `status = done`
3. Ambil `result` → simpan di backend Laravel (MySQL)
""",
)
async def generate_material(request: GenerateMaterialRequest):
    from app.workers.generate_worker import generate_material_task

    task_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_task_result(task_id, {
        "task_id": task_id,
        "status": "pending",
        "generate_type": "material",
        "course_id": request.course_id,
        "topic": request.topic,
        "language": request.language,
        "created_at": created_at,
    })

    try:
        celery_task = generate_material_task.apply_async(
            kwargs={
                "task_id": task_id,
                "course_id": request.course_id,
                "topic": request.topic,
                "language": request.language,
            },
            task_id=task_id,
            queue="generate",
        )
    except Exception as e:
        logger.error(f"[Generate] Dispatch material error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    save_task_result(task_id, {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "pending",
        "generate_type": "material",
        "course_id": request.course_id,
        "topic": request.topic,
        "language": request.language,
        "created_at": created_at,
    })

    logger.info(f"[Generate] Material dispatched | task_id={task_id} | course={request.course_id}")

    return GenerateTaskResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        status="pending",
        generate_type="material",
        course_id=request.course_id,
        topic=request.topic,
        language=request.language,
        created_at=created_at,
        status_url=f"/api/v1/generate/status/{task_id}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/flashcard
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/flashcard",
    response_model=GenerateTaskResponse,
    summary="Generate Flashcard (Async, Jumlah Dinamis)",
    description="""
Generate flashcard dari materi di Qdrant.

**Parameter penting:**
- `count`: jumlah flashcard yang diinginkan (1–50), **default 10**
- `material_context` *(opsional)*: teks `content` dari hasil generate materi  
  (dari Laravel) untuk meningkatkan akurasi flashcard

**Output** (`result.flashcards`):
```json
[
  {"front": "Kata kunci", "back": "Penjelasan 2-4 kalimat"}
]
```
""",
)
async def generate_flashcard(request: GenerateFlashcardRequest):
    from app.workers.generate_worker import generate_flashcard_task

    task_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_task_result(task_id, {
        "task_id": task_id,
        "status": "pending",
        "generate_type": "flashcard",
        "course_id": request.course_id,
        "source_file": request.source_file,
        "topic": request.topic,
        "count": request.count,
        "language": request.language,
        "created_at": created_at,
    })

    try:
        celery_task = generate_flashcard_task.apply_async(
            kwargs={
                "task_id": task_id,
                "course_id": request.course_id,
                "source_file": request.source_file,
                "topic": request.topic,
                "material_context": request.material_context,
                "count": request.count,
                "language": request.language,
            },
            task_id=task_id,
            queue="generate",
        )
    except Exception as e:
        logger.error(f"[Generate] Dispatch flashcard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    save_task_result(task_id, {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "pending",
        "generate_type": "flashcard",
        "course_id": request.course_id,
        "source_file": request.source_file,
        "topic": request.topic,
        "count": request.count,
        "language": request.language,
        "created_at": created_at,
    })

    logger.info(
        f"[Generate] Flashcard dispatched | task_id={task_id} | "
        f"course={request.course_id} | count={request.count}"
    )

    return GenerateTaskResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        status="pending",
        generate_type="flashcard",
        course_id=request.course_id,
        source_file=request.source_file,
        topic=request.topic,
        count=request.count,
        language=request.language,
        created_at=created_at,
        status_url=f"/api/v1/generate/status/{task_id}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/quiz
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/quiz",
    response_model=GenerateTaskResponse,
    summary="Generate Quiz (Async, Tipe & Jumlah Dinamis)",
    description="""
Generate soal quiz dari materi di Qdrant.

**`quiz_type`** (wajib pilih salah satu):
| Nilai | Deskripsi |
|-------|-----------|
| `essay` | Soal esai dengan contoh jawaban + poin penilaian |
| `multiple_choice_single` | Pilihan ganda A/B/C/D, **1 jawaban benar** |
| `multiple_choice_multiple` | Pilihan ganda A/B/C/D/E, **>1 jawaban bisa benar** |

**Parameter penting:**
- `count`: jumlah soal (1–50), **default 5**
- `difficulty`: `easy` | `medium` | `hard`
- `material_context` *(opsional)*: teks `content` dari materi di Laravel

**Output** (`result.items`) struktur per tipe:
- `essay`: `{question, sample_answer, key_points[], score_weight}`
- `multiple_choice_single`: `{question, options[], correct_answer, explanation}`
- `multiple_choice_multiple`: `{question, options[], correct_answers[], explanation}`
""",
)
async def generate_quiz(request: GenerateQuizRequest):
    from app.workers.generate_worker import generate_quiz_task

    task_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_task_result(task_id, {
        "task_id": task_id,
        "status": "pending",
        "generate_type": "quiz",
        "quiz_type": request.quiz_type,
        "course_id": request.course_id,
        "source_file": request.source_file,
        "topic": request.topic,
        "count": request.count,
        "difficulty": request.difficulty,
        "language": request.language,
        "created_at": created_at,
    })

    try:
        celery_task = generate_quiz_task.apply_async(
            kwargs={
                "task_id": task_id,
                "course_id": request.course_id,
                "source_file": request.source_file,
                "topic": request.topic,
                "material_context": request.material_context,
                "quiz_type": request.quiz_type,
                "count": request.count,
                "difficulty": request.difficulty,
                "language": request.language,
            },
            task_id=task_id,
            queue="generate",
        )
    except Exception as e:
        logger.error(f"[Generate] Dispatch quiz error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    save_task_result(task_id, {
        "task_id": task_id,
        "celery_task_id": celery_task.id,
        "status": "pending",
        "generate_type": "quiz",
        "quiz_type": request.quiz_type,
        "course_id": request.course_id,
        "source_file": request.source_file,
        "topic": request.topic,
        "count": request.count,
        "difficulty": request.difficulty,
        "language": request.language,
        "created_at": created_at,
    })

    logger.info(
        f"[Generate] Quiz dispatched | task_id={task_id} | type={request.quiz_type} | "
        f"course={request.course_id} | count={request.count} | difficulty={request.difficulty}"
    )

    return GenerateTaskResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        status="pending",
        generate_type="quiz",
        quiz_type=request.quiz_type,
        course_id=request.course_id,
        source_file=request.source_file,
        topic=request.topic,
        count=request.count,
        difficulty=request.difficulty,
        language=request.language,
        created_at=created_at,
        status_url=f"/api/v1/generate/status/{task_id}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /generate/status/{task_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status/{task_id}",
    response_model=GenerateStatusResponse,
    summary="Cek Status Task Generate",
    description="""
Poll status task generate yang sedang diproses oleh worker.

**Status lifecycle:**
```
pending → processing → done
                     ↘ error
```

Jika `status = done`, field `result` berisi output (material / flashcards / quiz items).
Jika `status = error`, field `error` berisi pesan error.

Data diambil dari **Redis** (TTL: 1 jam).
""",
)
async def get_generate_status(task_id: str):
    data = get_task_result(task_id)

    if data:
        return GenerateStatusResponse(**{
            k: v for k, v in data.items()
            if k in GenerateStatusResponse.model_fields
        })

    # Fallback ke Celery backend langsung
    try:
        from app.workers.celery_app import celery
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

        return GenerateStatusResponse(
            task_id=task_id,
            status=state_map.get(celery_state, "pending"),
        )

    except Exception as e:
        logger.warning(f"[Generate] Status fallback error: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' tidak ditemukan atau sudah expired (TTL 1 jam).",
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /generate/tasks
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/tasks",
    summary="List Semua Task Generate",
    description="""
Ambil semua task generate yang masih ada di Redis (TTL 1 jam).

Filter tersedia:
- `status`: `pending` | `processing` | `done` | `error`
- `generate_type`: `material` | `flashcard` | `quiz`
- `quiz_type`: `essay` | `multiple_choice_single` | `multiple_choice_multiple`
""",
)
async def list_generate_tasks(
    status: str = Query(default=None, description="Filter by status"),
    generate_type: str = Query(default=None, description="Filter: material | flashcard | quiz"),
    quiz_type: str = Query(default=None, description="Filter: essay | multiple_choice_single | multiple_choice_multiple"),
):
    tasks = list_tasks(prefix="gen_task:")

    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    if generate_type:
        tasks = [t for t in tasks if t.get("generate_type") == generate_type]
    if quiz_type:
        tasks = [t for t in tasks if t.get("quiz_type") == quiz_type]

    return {
        "total": len(tasks),
        "tasks": tasks,
    }