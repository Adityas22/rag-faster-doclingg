"""
Generate Endpoint — RAG content generation.

PENTING: request.content_type TIDAK bisa di-mutate langsung di Pydantic v2.
Solusi: gunakan request.model_copy(update={...}) untuk setiap endpoint.
"""
from fastapi import APIRouter, HTTPException
from app.schemas.generate import (
    GenerateRequest,
    GenerateFlashcardResponse,
    GenerateQuizResponse,
    GenerateSummaryResponse,
    GenerateMaterialResponse,
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _log_scope(req: GenerateRequest) -> str:
    if req.course_id and req.source_file:
        return f"course:{req.course_id}+file:{req.source_file}"
    if req.course_id:
        return f"course:{req.course_id}"
    if req.source_file:
        return f"file:{req.source_file}"
    return "global"


# ── Flashcard ─────────────────────────────────────────────────────────────────
@router.post(
    "/flashcard",
    response_model=GenerateFlashcardResponse,
    response_model_exclude_none=True,
    summary="Generate Flashcard",
    description="""
Generate flashcard dari materi di Qdrant.

**Format kartu:**
- **front** — heading / kata kunci singkat (tampilan depan)
- **back** — penjelasan detail (tampilan belakang)

`course_id` dan `source_file` opsional — kosong = ambil dari seluruh data (global).
`difficulty` tidak perlu diisi.
    """,
)
async def generate_flashcard(request: GenerateRequest):
    req = request.model_copy(update={"content_type": "flashcard"})
    logger.info(f"[Generate] flashcard | {_log_scope(req)} | topic={req.topic} | count={req.count}")
    try:
        from app.services.langgraph_service import run_generate_workflow
        result = await run_generate_workflow(req)
        return GenerateFlashcardResponse(**result)
    except Exception as e:
        logger.error(f"Generate flashcard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Quiz MC ───────────────────────────────────────────────────────────────────
@router.post(
    "/quiz/mc",
    response_model=GenerateQuizResponse,
    response_model_exclude_none=True,
    summary="Generate Quiz Pilihan Ganda",
    description="""
Generate soal pilihan ganda (A/B/C/D + kunci jawaban + penjelasan).

`difficulty`: `easy` | `medium` | `hard`
    """,
)
async def generate_quiz_mc(request: GenerateRequest):
    req = request.model_copy(update={"content_type": "quiz_mc"})
    logger.info(f"[Generate] quiz_mc | {_log_scope(req)} | diff={req.difficulty} | count={req.count}")
    try:
        from app.services.langgraph_service import run_generate_workflow
        result = await run_generate_workflow(req)
        
        result["difficulty"] = req.difficulty 
        
        return GenerateQuizResponse(**result)
    except Exception as e:
        logger.error(f"Generate quiz_mc error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Quiz Essay ────────────────────────────────────────────────────────────────
@router.post(
    "/quiz/essay",
    response_model=GenerateQuizResponse,
    response_model_exclude_none=True,
    summary="Generate Quiz Essay",
    description="""
Generate soal essay (pertanyaan + contoh jawaban + poin-poin kunci).

`difficulty`: `easy` | `medium` | `hard`
    """,
)
async def generate_quiz_essay(request: GenerateRequest):
    req = request.model_copy(update={"content_type": "quiz_essay"})
    logger.info(f"[Generate] quiz_essay | {_log_scope(req)} | diff={req.difficulty} | count={req.count}")
    try:
        from app.services.langgraph_service import run_generate_workflow
        result = await run_generate_workflow(req)
        
        result["difficulty"] = req.difficulty
        
        return GenerateQuizResponse(**result)
    except Exception as e:
        logger.error(f"Generate quiz_essay error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
@router.post(
    "/summary",
    response_model=GenerateSummaryResponse,
    response_model_exclude_none=True,
    summary="Generate Ringkasan Materi",
    description="""
Buat ringkasan dari materi di Qdrant.

`difficulty` tidak perlu diisi.
`count` tidak berlaku untuk summary.
    """,
)
async def generate_summary(request: GenerateRequest):
    req = request.model_copy(update={"content_type": "summary"})
    logger.info(f"[Generate] summary | {_log_scope(req)}")
    try:
        from app.services.langgraph_service import run_generate_workflow
        result = await run_generate_workflow(req)
        return GenerateSummaryResponse(**result)
    except Exception as e:
        logger.error(f"Generate summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Material ──────────────────────────────────────────────────────────────────
@router.post(
    "/material",
    response_model=GenerateMaterialResponse,
    response_model_exclude_none=True,
    summary="Generate Materi Pelajaran Lengkap",
    description="""
Generate materi lengkap (judul, pembuka, isi, contoh, latihan).

`difficulty` tidak perlu diisi.
`count` tidak berlaku untuk material.
    """,
)
async def generate_material(request: GenerateRequest):
    req = request.model_copy(update={"content_type": "material"})
    logger.info(f"[Generate] material | {_log_scope(req)}")
    try:
        from app.services.langgraph_service import run_generate_workflow
        result = await run_generate_workflow(req)
        return GenerateMaterialResponse(**result)
    except Exception as e:
        logger.error(f"Generate material error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
