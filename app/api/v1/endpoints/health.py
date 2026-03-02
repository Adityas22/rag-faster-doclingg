from fastapi import APIRouter
from app.config import settings
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("", tags=["Health"])
async def health_check():
    status = {"status": "ok", "version": settings.APP_VERSION, "services": {}}

    # Redis
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL, socket_connect_timeout=3)
        r.ping()
        status["services"]["redis"] = {"status": "ok"}
    except Exception as e:
        status["services"]["redis"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5)
        cols = client.get_collections()
        status["services"]["qdrant"] = {
            "status": "ok",
            "collections": [c.name for c in cols.collections],
        }
    except Exception as e:
        status["services"]["qdrant"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    # Ollama
    try:
        import requests
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            status["services"]["ollama"] = {
                "status": "ok",
                "model": settings.OLLAMA_MODEL,
                "available_models": models,
            }
        else:
            raise Exception(f"HTTP {r.status_code}")
    except Exception as e:
        status["services"]["ollama"] = {
            "status": "unavailable",
            "error": str(e),
            "fallback": "gemini" if settings.GEMINI_API_KEY else "none",
        }

    # Gemini
    status["services"]["gemini"] = {
        "status": "configured" if settings.GEMINI_API_KEY else "not_configured",
        "role": "embedding + llm_fallback",
    }

    return status
