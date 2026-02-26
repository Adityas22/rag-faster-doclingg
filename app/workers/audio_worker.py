"""
Celery Worker — Audio transcription task (async).
Menggunakan segment-aware token chunking dari notebook.
"""
import os
import asyncio
from app.workers.celery_app import celery
from app.utils.logger import get_logger

logger = get_logger(__name__)


@celery.task(
    bind=True,
    name="app.workers.audio_worker.transcribe_audio_task",
    max_retries=3,
    default_retry_delay=10,
)
def transcribe_audio_task(
    self,
    task_id: str,
    audio_path: str,
    language: str = "auto",
    course_id: str = None,
):
    """
    Async task: Transcribe audio → chunk (token+timestamp) → embed → simpan ke Qdrant.

    Steps:
    1. WhisperService → transkripsi + denoising
    2. Segment-aware token chunker → chunk dengan timestamp
    3. EmbeddingService → generate vectors (Gemini)
    4. QdrantService → simpan vectors (append mode)
    5. Cleanup temp file
    """
    logger.info(f"[AudioWorker] task_id={task_id} | file={audio_path} | lang={language}")
    self.update_state(state="STARTED", meta={"step": "transcribing"})

    try:
        # ── 1. Transkripsi + Denoising ───────────────────────────────────────
        from app.services.whisper_service import WhisperService
        ws = WhisperService.get_instance()
        result = ws.transcribe_to_dict(audio_path, language=language)
        logger.info(
            f"[AudioWorker] Transcribed: {len(result['segments'])} segs | "
            f"lang={result['language']} | duration={result['duration']}s"
        )

        self.update_state(state="STARTED", meta={"step": "chunking"})

        # ── 2. Token-Based Chunking (segment-aware) ──────────────────────────
        from app.utils.chunker import chunk_segments, chunk_text

        segments = result.get("segments", [])
        if segments:
            segment_chunks = chunk_segments(segments)
            chunk_dicts = [
                {
                    "text": sc["text"],
                    "metadata": {
                        "source_file": result["source_file"],
                        "course_id": course_id,
                        "type": "audio",
                        "language": result["language"],
                        "task_id": task_id,
                        "chunk_index": sc["chunk_index"],
                        "timestamp_start": sc.get("timestamp_start"),
                        "timestamp_end": sc.get("timestamp_end"),
                        "token_count": sc.get("token_count", 0),
                        "language_detected": result["language"],
                    },
                }
                for sc in segment_chunks
            ]
        else:
            plain_chunks = chunk_text(result["content"])
            chunk_dicts = [
                {
                    "text": c,
                    "metadata": {
                        "source_file": result["source_file"],
                        "course_id": course_id,
                        "type": "audio",
                        "language": result["language"],
                        "task_id": task_id,
                        "chunk_index": i,
                    },
                }
                for i, c in enumerate(plain_chunks)
            ]

        self.update_state(state="STARTED", meta={"step": "embedding", "chunks": len(chunk_dicts)})

        # ── 3. Embedding ─────────────────────────────────────────────────────
        from app.services.embedding_service import EmbeddingService
        embed_svc = EmbeddingService()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embedded = loop.run_until_complete(embed_svc.embed_chunks(chunk_dicts))

        self.update_state(state="STARTED", meta={"step": "storing"})

        # ── 4. Simpan ke Qdrant (Append Mode) ───────────────────────────────
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        stored = loop.run_until_complete(qdrant.upsert_vectors(embedded))
        loop.close()

        logger.info(f"[AudioWorker] ✅ Stored {stored} chunks for task_id={task_id}")

        # ── 5. Cleanup ───────────────────────────────────────────────────────
        if os.path.exists(audio_path):
            os.remove(audio_path)

        result["chunks_stored"] = stored
        result["total_chunks"] = len(chunk_dicts)
        return result

    except Exception as exc:
        logger.error(f"[AudioWorker] ❌ Error task_id={task_id}: {exc}", exc_info=True)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise self.retry(exc=exc, countdown=10)
