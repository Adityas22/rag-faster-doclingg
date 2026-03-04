"""
Celery Worker — Async content generation task.

Flow lengkap:
  1. FastAPI endpoint dispatch task ini via apply_async
  2. Worker ini dijalankan oleh container celery_generate (queue: generate)
  3. Hasil disimpan ke Redis via RedisService (key: gen_task:{task_id})
  4. FastAPI GET /generate/status/{task_id} ambil hasil dari Redis

Status lifecycle:
  pending → processing → done / error
"""
import asyncio
from datetime import datetime

from app.workers.celery_app import celery
from app.services.redis_service import save_task_result, update_task_status
from app.utils.logger import get_logger

logger = get_logger(__name__)


@celery.task(
    bind=True,
    name="app.workers.generate_worker.generate_content_task",
    max_retries=2,
    default_retry_delay=10,
)
def generate_content_task(
    self,
    task_id: str,
    course_id: str | None,
    content_type: str,
    topic: str | None = None,
    source_file: str | None = None,
    count: int = 5,
    difficulty: str | None = None,
    language: str = "id",
):
    """
    Celery task untuk generate konten AI secara async.

    Dipanggil dari endpoint POST /generate/{content_type}
    Hasilnya disimpan ke Redis dengan key gen_task:{task_id}
    """
    logger.info(
        f"[GenWorker] ▶ START | task_id={task_id} | type={content_type} "
        f"| course={course_id} | topic={topic!r} | count={count}"
    )

    # ── Tandai status PROCESSING di Redis ─────────────────────────────────────
    update_task_status(task_id, "processing", extra={
        "content_type": content_type,
        "course_id": course_id,
        "source_file": source_file,
        "topic": topic,
        "count": count,
        "difficulty": difficulty,
        "language": language,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "celery_task_id": self.request.id,
    })

    # Update Celery state juga (untuk Flower monitoring)
    self.update_state(state="STARTED", meta={
        "step": "retrieving_context",
        "task_id": task_id,
    })

    try:
        from app.services.langgraph_service import run_generate_workflow
        from app.schemas.generate import GenerateRequest

        request = GenerateRequest(
            course_id=course_id,
            source_file=source_file,
            content_type=content_type,
            topic=topic,
            count=count,
            difficulty=difficulty,
            language=language,
        )

        # Jalankan coroutine di event loop baru (Celery worker adalah sync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_generate_workflow(request))
        finally:
            loop.close()

        # ── Simpan hasil sukses ke Redis ───────────────────────────────────────
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "done",
            "content_type": content_type,
            "course_id": course_id,
            "source_file": source_file,
            "topic": topic,
            "count": count,
            "difficulty": difficulty,
            "language": language,
            "result": result,
            "finished_at": finished_at,
        })

        logger.info(f"[GenWorker] ✅ DONE | task_id={task_id} | type={content_type}")
        return {"task_id": task_id, "status": "done"}

    except Exception as exc:
        logger.error(f"[GenWorker] ❌ ERROR | task_id={task_id}: {exc}", exc_info=True)

        # Cek apakah masih bisa retry
        if self.request.retries < self.max_retries:
            # Update status ke processing (akan retry)
            update_task_status(task_id, "processing", extra={
                "retry": self.request.retries + 1,
                "last_error": str(exc),
            })
            raise self.retry(exc=exc, countdown=10)

        # Retry habis → tandai error di Redis
        save_task_result(task_id, {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "error",
            "content_type": content_type,
            "course_id": course_id,
            "source_file": source_file,
            "topic": topic,
            "error": str(exc),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Raise agar Celery juga mencatat FAILURE
        raise exc