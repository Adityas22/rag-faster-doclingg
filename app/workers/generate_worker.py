"""
Celery Worker — Async content generation task (opsional).
Digunakan jika generate membutuhkan waktu lama (model lokal besar).
"""
import asyncio
from app.workers.celery_app import celery
from app.utils.logger import get_logger

logger = get_logger(__name__)


@celery.task(
    bind=True,
    name="app.workers.generate_worker.generate_content_task",
    max_retries=2,
    default_retry_delay=5,
)
def generate_content_task(
    self,
    task_id: str,
    course_id: str,
    content_type: str,
    topic: str = None,
    count: int = 5,
    difficulty: str = "medium",
    language: str = "id",
):
    """
    Task async untuk generate konten jika proses LLM membutuhkan waktu lama.
    Biasanya di-trigger jika model Llama lokal lambat.
    """
    logger.info(f"[GenWorker] task_id={task_id} | type={content_type} | course={course_id}")
    self.update_state(state="STARTED", meta={"step": "retrieving_context"})

    try:
        from app.services.langgraph_service import run_generate_workflow
        from app.schemas.generate import GenerateRequest

        request = GenerateRequest(
            course_id=course_id,
            content_type=content_type,
            topic=topic,
            count=count,
            difficulty=difficulty,
            language=language,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_generate_workflow(request))
        loop.close()

        logger.info(f"[GenWorker] ✅ Generation done for task_id={task_id}")
        return result

    except Exception as exc:
        logger.error(f"[GenWorker] ❌ Error task_id={task_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=5)
