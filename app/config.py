import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    APP_NAME: str = "AI Course Generator"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8000", "http://localhost:3000"]

    # ── Whisper ───────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_CONDITION_ON_PREV: bool = False
    WHISPER_CHUNK_LENGTH_S: int = 30
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_KEEP_IN_MEMORY: bool = False

    # ── Denoising ─────────────────────────────────────────
    DENOISE_ENABLED: bool = True
    DENOISE_PROP_DECREASE: float = 0.75
    DENOISE_STATIONARY: bool = False

    # ── Gemini (embedding + LLM fallback) ─────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_EMBEDDING_MODEL: str = ""
    GEMINI_GENERATE_MODEL: str = ""

    # ── Llama (Ollama) — PRIMARY LLM ──────────────────────
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT: int = 120

    # ── Docling ───────────────────────────────────────────
    DOCLING_OCR_ENABLED: bool = True
    DOCLING_TABLE_STRUCTURE: bool = True

    # ── Qdrant ────────────────────────────────────────────
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "course_documents"
    QDRANT_EMBEDDING_DIM: int = 3072

    # ── Redis / Celery ────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"

    # ── Chunking ──────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    BATCH_EMBED_SIZE: int = 100

    # ── Upload ────────────────────────────────────────────
    TEMP_DIR: str = "./temp"
    OUTPUT_DIR: str = "./output"
    MAX_UPLOAD_MB: int = 100
    MAX_AUDIO_MB: int = 10
    MAX_DOCUMENT_MB: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)