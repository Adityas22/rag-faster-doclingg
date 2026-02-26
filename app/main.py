from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.config import settings
from app.api.v1.router import router as v1_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan: startup & shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"   Whisper model : {settings.WHISPER_MODEL_SIZE} ({settings.WHISPER_DEVICE})")
    # logger.info(f"   Qdrant        : {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info(f"   Qdrant        : {settings.QDRANT_URL}")
    logger.info(f"   Redis         : {settings.REDIS_URL}")
    yield
    # Shutdown
    logger.info("👋 Shutting down...")


# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## 🎓 AI Course Generator — FastAPI Backend

Backend AI untuk platform **LKS Kodeno**. Semua request dikirim dari Laravel backend.

### Komponen Utama
| Komponen | Peran |
|---|---|
| 🎙️ **Faster Whisper** | Speech-to-text dari audio/voice recording |
| 📄 **Docling** | Ekstraksi teks dari PDF |
| 🔢 **Gemini Embedding** | Generate vector untuk semantic search |
| 🗄️ **Qdrant** | Vector database untuk pencarian semantik |
| ✨ **Llama / Gemini** | Generate flashcard, quiz, ringkasan |
| 🔄 **LangGraph** | Orchestrasi workflow AI |
| ⚡ **Celery + Redis** | Async task queue untuk proses berat |

### Alur Kerja
1. Laravel mengirim upload audio/PDF ke FastAPI
2. FastAPI memasukkan task ke Redis Queue
3. Celery Worker memproses secara async (transkrip → embed → simpan Qdrant)
4. Laravel polling status via `task_id`
5. Ketika selesai, generate konten menggunakan RAG dari Qdrant
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(v1_router, prefix="/api/v1")


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"], summary="API Info")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
    }
