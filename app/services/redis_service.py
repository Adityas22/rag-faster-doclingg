"""
RedisService — Menyimpan dan mengambil hasil task generate dari Redis.

Dipakai oleh:
  - generate_worker.py  → simpan result setelah LLM selesai
  - generate.py endpoint → ambil result untuk GET /status/{task_id}

Key pattern: gen_task:{task_id}
TTL: 3600 detik (1 jam) — sesuai celery result_expires
"""
import json
import redis
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# TTL hasil task di Redis (detik)
TASK_TTL = 3600


def _get_client() -> redis.Redis:
    """Buat Redis client dari REDIS_URL di settings."""
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def save_task_result(task_id: str, data: dict, ttl: int = TASK_TTL) -> None:
    """
    Simpan hasil task ke Redis.

    Args:
        task_id: ID task (UUID)
        data: dict berisi status, result, error, metadata
        ttl: waktu hidup key di Redis (detik)
    """
    try:
        client = _get_client()
        key = f"gen_task:{task_id}"
        client.setex(key, ttl, json.dumps(data, ensure_ascii=False))
        logger.info(f"[Redis] Saved task result | key={key} | status={data.get('status')}")
    except Exception as e:
        logger.error(f"[Redis] Failed to save task_id={task_id}: {e}", exc_info=True)


def get_task_result(task_id: str) -> dict | None:
    """
    Ambil hasil task dari Redis.

    Returns:
        dict jika ditemukan, None jika tidak ada / expired
    """
    try:
        client = _get_client()
        key = f"gen_task:{task_id}"
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.error(f"[Redis] Failed to get task_id={task_id}: {e}", exc_info=True)
        return None


def update_task_status(task_id: str, status: str, extra: dict = None, ttl: int = TASK_TTL) -> None:
    """
    Update status task yang sudah ada di Redis.
    Jika key belum ada, buat entry baru dengan status tersebut.

    Args:
        task_id: ID task
        status: string status baru ('pending', 'processing', 'done', 'error')
        extra: dict tambahan yang di-merge ke data yang sudah ada
        ttl: reset TTL ke nilai ini
    """
    existing = get_task_result(task_id) or {"task_id": task_id}
    existing["status"] = status
    if extra:
        existing.update(extra)
    save_task_result(task_id, existing, ttl=ttl)


def list_tasks(prefix: str = "gen_task:") -> list[dict]:
    """
    Ambil semua task generate dari Redis (untuk debugging / admin).

    Returns:
        List dict task, diurutkan dari yang terbaru
    """
    try:
        client = _get_client()
        keys = client.keys(f"{prefix}*")
        tasks = []
        for key in keys:
            raw = client.get(key)
            if raw:
                try:
                    tasks.append(json.loads(raw))
                except Exception:
                    pass
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks
    except Exception as e:
        logger.error(f"[Redis] Failed to list tasks: {e}", exc_info=True)
        return []


def delete_task(task_id: str) -> bool:
    """Hapus task dari Redis. Returns True jika berhasil dihapus."""
    try:
        client = _get_client()
        key = f"gen_task:{task_id}"
        deleted = client.delete(key)
        return deleted > 0
    except Exception as e:
        logger.error(f"[Redis] Failed to delete task_id={task_id}: {e}", exc_info=True)
        return False