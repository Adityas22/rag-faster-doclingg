from fastapi import APIRouter
from app.api.v1.endpoints import health, audio, document, generate

router = APIRouter()

router.include_router(health.router,     prefix="/health",    tags=["🔧 Health"])
router.include_router(audio.router,      prefix="/audio",     tags=["🎙️ Audio"])
router.include_router(document.router,   prefix="/document",  tags=["📄 Document"])
router.include_router(generate.router,   prefix="/generate",  tags=["✨ Generate"])
