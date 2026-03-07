"""
Celery Worker — Async generate task untuk Material, Flashcard, dan Quiz.

Task yang tersedia:
  - generate_material_task   → generate materi lengkap dari Qdrant
  - generate_flashcard_task  → generate flashcard (jumlah dinamis)
  - generate_quiz_task       → generate quiz (essay | mc_single | mc_multiple, jumlah dinamis)

Status lifecycle: pending → processing → done / error
Hasil disimpan ke Redis: key = gen_task:{task_id}
"""
import asyncio
from datetime import datetime

from app.workers.celery_app import celery
from app.services.redis_service import save_task_result, update_task_status
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: run async coroutine dalam event loop baru (Celery sync context)
# ─────────────────────────────────────────────────────────────────────────────

def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Generate Materi
# ─────────────────────────────────────────────────────────────────────────────

@celery.task(
    bind=True,
    name="app.workers.generate_worker.generate_material_task",
    max_retries=2,
    default_retry_delay=10,
)
def generate_material_task(
    self,
    task_id: str,
    course_id: str,
    topic: str | None = None,
    language: str = "id",
):
    """
    Generate materi pembelajaran dari konten di Qdrant berdasarkan course_id.
    Semua file dalam course tersebut digunakan sebagai sumber materi.
    Output JSON ini yang nantinya disimpan di backend Laravel (MySQL).
    """
    logger.info(
        f"[MatWorker] ▶ START | task_id={task_id} | course={course_id} | "
        f"topic={topic!r} | lang={language}"
    )

    update_task_status(task_id, "processing", extra={
        "generate_type": "material",
        "course_id": course_id,
        "topic": topic,
        "language": language,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "celery_task_id": self.request.id,
    })
    self.update_state(state="STARTED", meta={"step": "retrieving_context", "task_id": task_id})

    try:
        from app.services.langgraph_service import run_generate_material
        from app.schemas.generate import GenerateMaterialRequest

        request = GenerateMaterialRequest(
            course_id=course_id,
            topic=topic,
            language=language,
        )

        result = _run_async(run_generate_material(request))

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "done",
            "generate_type": "material",
            "course_id": course_id,
            "topic": topic,
            "language": language,
            "result": result,
            "finished_at": finished_at,
        })

        logger.info(f"[MatWorker] ✅ DONE | task_id={task_id}")
        return {"task_id": task_id, "status": "done"}

    except Exception as exc:
        logger.error(f"[MatWorker] ❌ ERROR | task_id={task_id}: {exc}", exc_info=True)

        if self.request.retries < self.max_retries:
            update_task_status(task_id, "processing", extra={
                "retry": self.request.retries + 1,
                "last_error": str(exc),
            })
            raise self.retry(exc=exc, countdown=10)

        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "error",
            "generate_type": "material",
            "course_id": course_id,
            "topic": topic,
            "error": str(exc),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise exc


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Generate Flashcard
# ─────────────────────────────────────────────────────────────────────────────

@celery.task(
    bind=True,
    name="app.workers.generate_worker.generate_flashcard_task",
    max_retries=2,
    default_retry_delay=10,
)
def generate_flashcard_task(
    self,
    task_id: str,
    course_id: str | None,
    source_file: str | None = None,
    topic: str | None = None,
    material_context: str | None = None,
    count: int = 10,
    language: str = "id",
):
    """
    Generate flashcard dari konten Qdrant.
    material_context (opsional): teks materi dari Laravel untuk hasil lebih akurat.
    count: jumlah flashcard yang diinginkan (dinamis, 1–50).
    """
    logger.info(
        f"[FCWorker] ▶ START | task_id={task_id} | course={course_id} | "
        f"topic={topic!r} | count={count}"
    )

    update_task_status(task_id, "processing", extra={
        "generate_type": "flashcard",
        "course_id": course_id,
        "source_file": source_file,
        "topic": topic,
        "count": count,
        "language": language,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "celery_task_id": self.request.id,
    })
    self.update_state(state="STARTED", meta={"step": "retrieving_context", "task_id": task_id})

    try:
        from app.services.langgraph_service import run_generate_flashcard
        from app.schemas.generate import GenerateFlashcardRequest

        request = GenerateFlashcardRequest(
            course_id=course_id,
            source_file=source_file,
            topic=topic,
            material_context=material_context,
            count=count,
            language=language,
        )

        result = _run_async(run_generate_flashcard(request))

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "done",
            "generate_type": "flashcard",
            "course_id": course_id,
            "source_file": source_file,
            "topic": topic,
            "count": count,
            "language": language,
            "result": result,
            "finished_at": finished_at,
        })

        logger.info(f"[FCWorker] ✅ DONE | task_id={task_id} | cards={result.get('count')}")
        return {"task_id": task_id, "status": "done"}

    except Exception as exc:
        logger.error(f"[FCWorker] ❌ ERROR | task_id={task_id}: {exc}", exc_info=True)

        if self.request.retries < self.max_retries:
            update_task_status(task_id, "processing", extra={
                "retry": self.request.retries + 1,
                "last_error": str(exc),
            })
            raise self.retry(exc=exc, countdown=10)

        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "error",
            "generate_type": "flashcard",
            "course_id": course_id,
            "topic": topic,
            "count": count,
            "error": str(exc),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise exc


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Generate Quiz
# ─────────────────────────────────────────────────────────────────────────────

@celery.task(
    bind=True,
    name="app.workers.generate_worker.generate_quiz_task",
    max_retries=2,
    default_retry_delay=10,
)
def generate_quiz_task(
    self,
    task_id: str,
    course_id: str | None,
    quiz_type: str,
    source_file: str | None = None,
    topic: str | None = None,
    material_context: str | None = None,
    count: int = 5,
    difficulty: str = "medium",
    language: str = "id",
):
    """
    Generate quiz dari konten Qdrant.

    quiz_type:
      - "essay"                    → soal esai
      - "multiple_choice_single"   → pilihan ganda 1 jawaban benar
      - "multiple_choice_multiple" → pilihan ganda bisa >1 jawaban benar

    material_context (opsional): teks materi dari Laravel.
    count: jumlah soal (dinamis, 1–50).
    """
    logger.info(
        f"[QuizWorker] ▶ START | task_id={task_id} | course={course_id} | "
        f"type={quiz_type} | topic={topic!r} | count={count} | difficulty={difficulty}"
    )

    update_task_status(task_id, "processing", extra={
        "generate_type": "quiz",
        "quiz_type": quiz_type,
        "course_id": course_id,
        "source_file": source_file,
        "topic": topic,
        "count": count,
        "difficulty": difficulty,
        "language": language,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "celery_task_id": self.request.id,
    })
    self.update_state(state="STARTED", meta={"step": "retrieving_context", "task_id": task_id})

    try:
        from app.services.langgraph_service import run_generate_quiz
        from app.schemas.generate import GenerateQuizRequest

        request = GenerateQuizRequest(
            course_id=course_id,
            source_file=source_file,
            topic=topic,
            material_context=material_context,
            quiz_type=quiz_type,
            count=count,
            difficulty=difficulty,
            language=language,
        )

        result = _run_async(run_generate_quiz(request))

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "done",
            "generate_type": "quiz",
            "quiz_type": quiz_type,
            "course_id": course_id,
            "source_file": source_file,
            "topic": topic,
            "count": count,
            "difficulty": difficulty,
            "language": language,
            "result": result,
            "finished_at": finished_at,
        })

        logger.info(f"[QuizWorker] ✅ DONE | task_id={task_id} | items={result.get('count')}")
        return {"task_id": task_id, "status": "done"}

    except Exception as exc:
        logger.error(f"[QuizWorker] ❌ ERROR | task_id={task_id}: {exc}", exc_info=True)

        if self.request.retries < self.max_retries:
            update_task_status(task_id, "processing", extra={
                "retry": self.request.retries + 1,
                "last_error": str(exc),
            })
            raise self.retry(exc=exc, countdown=10)

        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "error",
            "generate_type": "quiz",
            "quiz_type": quiz_type,
            "course_id": course_id,
            "topic": topic,
            "count": count,
            "error": str(exc),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        raise exc