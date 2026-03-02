"""
Celery + Redis setup untuk async task queue.
"""
from celery import Celery
from app.config import settings

celery = Celery(
    "ai_course_generator",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.workers.audio_worker",
        "app.workers.document_worker",
        "app.workers.generate_worker",
    ],
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Jakarta",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                   # ACK setelah task selesai (lebih aman)
    worker_prefetch_multiplier=1,          # Proses 1 task per worker (hindari OOM)
    result_expires=3600,                   # Hasil disimpan 1 jam
    task_soft_time_limit=600,              # 10 menit soft limit
    task_time_limit=900,                   # 15 menit hard limit
    task_routes={
        "app.workers.audio_worker.*": {"queue": "audio"},
        "app.workers.document_worker.*": {"queue": "document"},
        "app.workers.generate_worker.*": {"queue": "generate"},
    },
)
