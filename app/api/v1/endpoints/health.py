from fastapi import APIRouter
from app.config import settings
import time, psutil, platform

router = APIRouter()


@router.get(
    "/",
    summary="Health Check",
    description="Cek status server, resource usage, dan koneksi ke services eksternal.",
)
async def health_check():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    # Cek Qdrant
    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=2,
        )
        qc.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    # Cek Redis
    redis_ok = False
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "timestamp": int(time.time()),
        "system": {
            "platform": platform.system(),
            "cpu_percent": cpu,
            "memory_percent": mem.percent,
            "memory_available_mb": round(mem.available / 1024 / 1024, 1),
        },
        "services": {
            "qdrant": "ok" if qdrant_ok else "unreachable",
            "redis": "ok" if redis_ok else "unreachable",
        },
        "config": {
            "whisper_model": settings.WHISPER_MODEL_SIZE,
            "whisper_device": settings.WHISPER_DEVICE,
            "qdrant_collection": settings.QDRANT_COLLECTION,
        },
    }

@router.get(
    "/qdrant",
    summary="Cek Status Qdrant Collection + Payload Indexes",
    description="Debug endpoint: tampilkan info collection dan daftar payload indexes yang ada.",
)
async def qdrant_health():
    """
    Cek apakah collection dan payload indexes sudah benar.
    Payload indexes WAJIB ada untuk filter by course_id, source_file, dll.
    """
    try:
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()

        info = await qdrant.get_collection_info()

        # Ambil daftar indexes yang sudah ada
        raw_info = qdrant.client.get_collection(settings.QDRANT_COLLECTION)
        payload_schema = raw_info.payload_schema or {}
        existing_indexes = list(payload_schema.keys())

        required_indexes = ["course_id", "source_file", "type", "task_id", "chunk_index"]
        missing = [f for f in required_indexes if f not in existing_indexes]

        return {
            "status": "ok",
            "collection_info": info,
            "payload_indexes": {
                "existing": existing_indexes,
                "required": required_indexes,
                "missing": missing,
                "all_present": len(missing) == 0,
            },
            "message": (
                "✅ Semua index tersedia" if not missing
                else f"⚠️ Index belum ada: {missing}. Akan dibuat otomatis saat QdrantService init."
            ),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
