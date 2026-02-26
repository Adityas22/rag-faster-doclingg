import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    APP_NAME: str = "AI Course Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8000", "http://localhost:3000"]

    # ── Whisper ───────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "base"       # tiny | base | small | medium | large-v2
    WHISPER_DEVICE: str = "cpu"            # cpu | cuda
    WHISPER_COMPUTE_TYPE: str = "int8"     # int8 | float16 | float32
    # Optimasi kecepatan (dari notebook)
    WHISPER_BEAM_SIZE: int = 1             # 1=greedy (cepat), 5=beam (akurat)
    WHISPER_CONDITION_ON_PREV: bool = False # False = lebih cepat, cegah hallucination
    WHISPER_CHUNK_LENGTH_S: int = 30        # detik per chunk audio
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6

    # WINDOWS DLL CONFLICT: set False agar model di-release setelah tiap request
    # Set True hanya jika TIDAK menggunakan Docling di server yang sama
    WHISPER_KEEP_IN_MEMORY: bool = False

    # ── Denoising ─────────────────────────────────────────
    DENOISE_ENABLED: bool = True
    DENOISE_PROP_DECREASE: float = 0.75
    DENOISE_STATIONARY: bool = False       # False=noise dinamis (lebih umum)

    # ── Gemini ────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_EMBEDDING_MODEL: str = ""
    GEMINI_GENERATE_MODEL: str = ""

    # ── Llama (Ollama) ────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── Qdrant Cloud ──────────────────────────────────────
    QDRANT_URL: str = ""                   # https://xxxx.cloud.qdrant.io:6333
    QDRANT_API_KEY: str = ""              # API Key dari Qdrant Cloud dashboard
    QDRANT_COLLECTION: str = "course_documents"
    QDRANT_EMBEDDING_DIM: int = 3072       # gemini-embedding-001 default

    # ── Redis / Celery ────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Chunking (berbasis token, dari notebook) ──────────
    CHUNK_SIZE: int = 500                  # token per chunk
    CHUNK_OVERLAP: int = 50               # token overlap antar chunk
    BATCH_EMBED_SIZE: int = 100

    # ── Upload ────────────────────────────────────────────
    TEMP_DIR: str = "./temp"
    OUTPUT_DIR: str = "./output"
    MAX_UPLOAD_MB: int = 100              # default umum
    MAX_AUDIO_MB: int = 10               # audio rekaman 3 menit maks 10MB
    MAX_DOCUMENT_MB: int = 10            # dokumen PDF maks 10MB

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Pastikan folder temp & output ada
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
