"""
Audio Endpoint — Upload & Transcribe Audio → Chunk → Embed → Simpan ke Qdrant.

Batasan:
- File audio maks 10MB (cukup untuk rekaman ±3 menit dengan kualitas standar)
- Format: .mp3 .wav .m4a .ogg .flac .aac
- Chunking berbasis token + timestamp segmen (dari notebook)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.schemas.audio import (
    TranscribeFullResponse,
    TranscribeStatusResponse,
    TaskListResponse,
    AudioInfoResponse,
)
from app.config import settings
from app.utils.logger import get_logger
import uuid, os, shutil, json
from datetime import datetime

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TASKS_FILE = os.path.join("output", "tasks.json")

# 10MB untuk rekaman audio (±3 menit @ 128kbps ≈ 2.8MB, cukup besar)
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


def _validate_audio_file(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{ext}'. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}",
        )


# ── POST /transcribe ──────────────────────────────────────────────────────────
@router.post(
    "/transcribe",
    response_model=TranscribeFullResponse,
    summary="Upload & Transkripsi Audio → Simpan ke Qdrant",
    description="""
Proses lengkap dalam satu request:

1. Upload & validasi file audio (**maks 10MB** — cukup untuk ±3 menit rekaman)
2. Denoising audio (noisereduce spectral gating)
3. Transkripsi via Faster Whisper (beam_size=1, greedy decoding)
4. Chunking berbasis token + timestamp segmen
5. Generate embedding via Gemini
6. Simpan ke Qdrant Cloud dengan `course_id` (append mode — ID global berlanjut)
7. Return hasil lengkap

**Format audio:** `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.aac`

**Tips rekaman 3 menit:**
- MP3 128kbps ≈ 2.8MB ✅
- WAV 16kHz mono ≈ 5.8MB ✅
- WAV 44.1kHz stereo ≈ 31MB ❌ (gunakan MP3 atau WAV mono 16kHz)
    """,
)
async def transcribe_audio(
    file: UploadFile = File(..., description=f"File audio rekaman (maks {MAX_AUDIO_MB}MB, ±3 menit)"),
    language: str = Query(default="auto", description="Bahasa: 'auto', 'id', 'en'"),
    course_id: str = Query(..., description="ID course dari Laravel (wajib)"),
):
    _validate_audio_file(file)

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_AUDIO_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File terlalu besar ({size_mb:.1f}MB). Maks {MAX_AUDIO_MB}MB. "
                f"Untuk rekaman 3 menit gunakan format MP3 atau WAV 16kHz mono."
            ),
        )

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        f"[Transcribe] START | file={file.filename} ({size_mb:.1f}MB) | course_id={course_id}"
    )

    try:
        # ── Step 1: Transkripsi ──────────────────────────────────────────────
        logger.info("[Transcribe] Step 1/3 — Faster Whisper (dengan denoising)...")
        from app.services.whisper_service import WhisperService
        ws = WhisperService.get_instance()
        transcribe_result = ws.transcribe_to_dict(temp_path, language=language)

        logger.info(
            f"[Transcribe] ✅ Step 1 Done | "
            f"lang={transcribe_result['language']} | "
            f"segs={len(transcribe_result['segments'])} | "
            f"duration={transcribe_result['duration']}s | "
            f"RTF={transcribe_result.get('realtime_factor', '?')}x"
        )

        # ── Step 2: Chunking Token-Based + Timestamp ─────────────────────────
        logger.info("[Transcribe] Step 2/3 — Token-based chunking + Gemini Embedding...")
        from app.utils.chunker import chunk_segments, chunk_text
        from app.services.embedding_service import EmbeddingService

        # Gunakan segment-aware chunking jika tersedia
        segments = transcribe_result.get("segments", [])
        if segments:
            segment_chunks = chunk_segments(segments)
            chunk_dicts = [
                {
                    "text": sc["text"],
                    "metadata": {
                        "course_id": course_id,
                        "task_id": task_id,
                        "source_file": file.filename,
                        "type": "audio",
                        "language": transcribe_result["language"],
                        "duration": transcribe_result["duration"],
                        "chunk_index": sc["chunk_index"],
                        "timestamp_start": sc.get("timestamp_start"),
                        "timestamp_end": sc.get("timestamp_end"),
                        "token_count": sc.get("token_count", 0),
                        "language_detected": transcribe_result["language"],
                    },
                }
                for sc in segment_chunks
            ]
        else:
            # Fallback plain text chunking
            plain_chunks = chunk_text(transcribe_result["content"])
            chunk_dicts = [
                {
                    "text": chunk,
                    "metadata": {
                        "course_id": course_id,
                        "task_id": task_id,
                        "source_file": file.filename,
                        "type": "audio",
                        "language": transcribe_result["language"],
                        "duration": transcribe_result["duration"],
                        "chunk_index": i,
                        "language_detected": transcribe_result["language"],
                    },
                }
                for i, chunk in enumerate(plain_chunks)
            ]

        logger.info(f"[Transcribe] Total chunks: {len(chunk_dicts)}")

        embed_svc = EmbeddingService()
        embedded = await embed_svc.embed_chunks(chunk_dicts)
        logger.info(f"[Transcribe] ✅ Step 2 Done | vectors={len(embedded)}")

        # ── Step 3: Simpan ke Qdrant (Append Mode) ───────────────────────────
        logger.info("[Transcribe] Step 3/3 — Saving to Qdrant Cloud (append mode)...")
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        chunks_stored = await qdrant.upsert_vectors(embedded)
        logger.info(
            f"[Transcribe] ✅ Step 3 Done | stored={chunks_stored} | course_id={course_id}"
        )

        # Registry lokal
        _save_task(task_id, {
            "task_id": task_id,
            "status": "done",
            "course_id": course_id,
            "source_file": file.filename,
            "language": transcribe_result["language"],
            "duration": transcribe_result["duration"],
            "transcription_time": transcribe_result.get("transcription_time"),
            "realtime_factor": transcribe_result.get("realtime_factor"),
            "total_chunks": len(chunk_dicts),
            "chunks_stored": chunks_stored,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        return TranscribeFullResponse(
            task_id=task_id,
            status="done",
            course_id=course_id,
            source_file=transcribe_result["source_file"],
            content=transcribe_result["content"],
            language=transcribe_result["language"],
            language_probability=transcribe_result["language_probability"],
            duration=transcribe_result["duration"],
            transcription_time=transcribe_result.get("transcription_time"),
            realtime_factor=transcribe_result.get("realtime_factor"),
            segments=transcribe_result["segments"],
            total_chunks=len(chunk_dicts),
            chunks_stored=chunks_stored,
        )

    except Exception as e:
        _save_task(task_id, {
            "task_id": task_id,
            "status": "error",
            "course_id": course_id,
            "source_file": file.filename,
            "error": str(e),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.error(f"[Transcribe] ❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── POST /record — alias untuk rekaman langsung ───────────────────────────────
@router.post(
    "/record",
    response_model=TranscribeFullResponse,
    summary="Upload Rekaman Audio Langsung (Voice Recording ±3 menit)",
    description="""
Endpoint khusus untuk **rekaman suara langsung** dari user (voice recording ±3 menit).

Sama dengan `/transcribe` tapi:
- Limit **10MB** (sesuai durasi rekaman 3 menit)
- Menerima format `.webm` `.ogg` `.wav` `.mp4` dari browser MediaRecorder API
- Otomatis denoising sebelum transkripsi

**Browser MediaRecorder output:**
| Browser | Format | Ukuran 3 menit |
|---------|--------|----------------|
| Chrome  | webm/opus | ~1-2MB ✅ |
| Firefox | ogg/opus  | ~1-2MB ✅ |
| Safari  | mp4/aac   | ~3-5MB ✅ |
    """,
)
async def record_audio(
    file: UploadFile = File(..., description="File rekaman audio dari browser (maks 10MB)"),
    language: str = Query(default="auto", description="Bahasa: 'auto', 'id', 'en'"),
    course_id: str = Query(..., description="ID course dari Laravel (wajib)"),
):
    # Extend allowed extensions untuk format browser recording
    extended_ext = ALLOWED_EXTENSIONS | {".webm", ".mp4", ".3gp"}
    ext = os.path.splitext(file.filename or "record.webm")[1].lower()
    if not ext:
        # Assign extension default jika tidak ada
        file.filename = f"recording_{uuid.uuid4().hex[:8]}.webm"
        ext = ".webm"

    if ext not in extended_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{ext}'. Browser recording: .webm .ogg .mp4 .3gp",
        )

    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_AUDIO_MB:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Rekaman terlalu besar ({size_mb:.1f}MB). "
                f"Maks {MAX_AUDIO_MB}MB (±3 menit). "
                f"Pastikan durasi tidak melebihi 3 menit."
            ),
        )

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.TEMP_DIR, f"{task_id}_{file.filename}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        f"[Record] START | file={file.filename} ({size_mb:.1f}MB) | course_id={course_id}"
    )

    try:
        from app.services.whisper_service import WhisperService
        from app.utils.chunker import chunk_segments, chunk_text
        from app.services.embedding_service import EmbeddingService
        from app.services.qdrant_service import QdrantService

        ws = WhisperService.get_instance()
        transcribe_result = ws.transcribe_to_dict(temp_path, language=language)

        logger.info(
            f"[Record] Transcribed | lang={transcribe_result['language']} | "
            f"segs={len(transcribe_result['segments'])} | duration={transcribe_result['duration']}s"
        )

        # Segment-aware chunking
        segments = transcribe_result.get("segments", [])
        if segments:
            segment_chunks = chunk_segments(segments)
            chunk_dicts = [
                {
                    "text": sc["text"],
                    "metadata": {
                        "course_id": course_id,
                        "task_id": task_id,
                        "source_file": file.filename,
                        "type": "audio_record",
                        "language": transcribe_result["language"],
                        "duration": transcribe_result["duration"],
                        "chunk_index": sc["chunk_index"],
                        "timestamp_start": sc.get("timestamp_start"),
                        "timestamp_end": sc.get("timestamp_end"),
                        "token_count": sc.get("token_count", 0),
                        "language_detected": transcribe_result["language"],
                    },
                }
                for sc in segment_chunks
            ]
        else:
            plain_chunks = chunk_text(transcribe_result["content"])
            chunk_dicts = [
                {
                    "text": chunk,
                    "metadata": {
                        "course_id": course_id,
                        "task_id": task_id,
                        "source_file": file.filename,
                        "type": "audio_record",
                        "language": transcribe_result["language"],
                        "duration": transcribe_result["duration"],
                        "chunk_index": i,
                    },
                }
                for i, chunk in enumerate(plain_chunks)
            ]

        embed_svc = EmbeddingService()
        embedded = await embed_svc.embed_chunks(chunk_dicts)

        qdrant = QdrantService()
        chunks_stored = await qdrant.upsert_vectors(embedded)

        _save_task(task_id, {
            "task_id": task_id,
            "status": "done",
            "course_id": course_id,
            "source_file": file.filename,
            "type": "audio_record",
            "language": transcribe_result["language"],
            "duration": transcribe_result["duration"],
            "transcription_time": transcribe_result.get("transcription_time"),
            "total_chunks": len(chunk_dicts),
            "chunks_stored": chunks_stored,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        logger.info(f"[Record] ✅ Done | stored={chunks_stored} | course_id={course_id}")

        return TranscribeFullResponse(
            task_id=task_id,
            status="done",
            course_id=course_id,
            source_file=transcribe_result["source_file"],
            content=transcribe_result["content"],
            language=transcribe_result["language"],
            language_probability=transcribe_result["language_probability"],
            duration=transcribe_result["duration"],
            transcription_time=transcribe_result.get("transcription_time"),
            realtime_factor=transcribe_result.get("realtime_factor"),
            segments=transcribe_result["segments"],
            total_chunks=len(chunk_dicts),
            chunks_stored=chunks_stored,
        )

    except Exception as e:
        _save_task(task_id, {
            "task_id": task_id,
            "status": "error",
            "course_id": course_id,
            "source_file": file.filename,
            "error": str(e),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.error(f"[Record] ❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── GET: Task berdasarkan course_id ───────────────────────────────────────────
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
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada task untuk course_id '{course_id}'.",
        )

    return TaskListResponse(total=len(task_list), tasks=task_list)


# ── GET: Semua Task ───────────────────────────────────────────────────────────
@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="Lihat Semua Task",
)
async def list_all_tasks(
    status: str = Query(default=None, description="Filter: 'done' atau 'error'"),
):
    tasks = _load_tasks()
    task_list = list(tasks.values())

    if status:
        task_list = [t for t in task_list if t.get("status") == status]

    task_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return TaskListResponse(total=len(task_list), tasks=task_list)


# ── DELETE: Hapus berdasarkan course_id ───────────────────────────────────────
@router.delete(
    "/course/{course_id}",
    summary="Hapus Semua Data Audio berdasarkan course_id",
)
async def delete_by_course_id(course_id: str):
    tasks = _load_tasks()
    related_tasks = [t for t in tasks.values() if t.get("course_id") == course_id]

    if not related_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada data untuk course_id '{course_id}'.",
        )

    try:
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        deleted_vectors = await qdrant.delete_by_course_id(course_id)
        logger.info(f"[Delete] Qdrant: {deleted_vectors} vectors | course_id={course_id}")

        task_ids_deleted = []
        for task_id, task_data in list(tasks.items()):
            if task_data.get("course_id") == course_id:
                task_ids_deleted.append(task_id)
                del tasks[task_id]

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
