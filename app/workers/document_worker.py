"""
Celery Worker — PDF document extraction + token chunking + embedding task.
"""
import os
import asyncio
from app.workers.celery_app import celery
from app.utils.logger import get_logger

logger = get_logger(__name__)


@celery.task(
    bind=True,
    name="app.workers.document_worker.extract_document_task",
    max_retries=3,
    default_retry_delay=15,
)
def extract_document_task(
    self,
    task_id: str,
    pdf_path: str,
    course_id: str,
    doc_title: str = None,
):
    """
    Async task: Extract PDF → token chunk → embed → simpan ke Qdrant (append mode).
    """
    logger.info(f"[DocWorker] task_id={task_id} | file={pdf_path} | course={course_id}")
    self.update_state(state="STARTED", meta={"step": "extracting"})

    try:
        # ── 1. Ekstraksi PDF ─────────────────────────────────────────────────
        from app.services.docling_service import DoclingService
        docling = DoclingService()
        doc_result = docling.extract_pdf(pdf_path, doc_title=doc_title)
        logger.info(
            f"[DocWorker] Extracted: {doc_result['total_pages']} pages, "
            f"{doc_result['total_chars']} chars"
        )

        self.update_state(state="STARTED", meta={"step": "chunking"})

        # ── 2. Token-Based Chunking ──────────────────────────────────────────
        from app.utils.chunker import chunk_text
        chunks = chunk_text(doc_result["full_text"])

        self.update_state(state="STARTED", meta={"step": "embedding", "chunks": len(chunks)})

        # ── 3. Embedding ─────────────────────────────────────────────────────
        from app.services.embedding_service import EmbeddingService
        embed_svc = EmbeddingService()

        chunk_dicts = [
            {
                "text": c,
                "metadata": {
                    "source_file": doc_result["source_file"],
                    "doc_title": doc_result["doc_title"],
                    "course_id": course_id,
                    "type": "document",
                    "task_id": task_id,
                    "chunk_index": i,
                },
            }
            for i, c in enumerate(chunks)
        ]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embedded = loop.run_until_complete(embed_svc.embed_chunks(chunk_dicts))

        self.update_state(state="STARTED", meta={"step": "storing"})

        # ── 4. Simpan ke Qdrant (Append Mode) ───────────────────────────────
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        stored = loop.run_until_complete(qdrant.upsert_vectors(embedded))
        loop.close()

        logger.info(f"[DocWorker] ✅ Stored {stored} chunks for task_id={task_id}")

        # ── 5. Cleanup ───────────────────────────────────────────────────────
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return {
            "source_file": doc_result["source_file"],
            "course_id": course_id,
            "doc_title": doc_result["doc_title"],
            "total_pages": doc_result["total_pages"],
            "total_chars": doc_result["total_chars"],
            "total_chunks": len(chunks),
            "chunks_stored": stored,
        }

    except Exception as exc:
        logger.error(f"[DocWorker] ❌ Error task_id={task_id}: {exc}", exc_info=True)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise self.retry(exc=exc, countdown=15)
