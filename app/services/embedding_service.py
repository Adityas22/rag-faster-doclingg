"""
EmbeddingService — Generate vector embeddings via Google Gemini API.

Rate limit handling (free tier: 100 requests/menit):
- Batch size 10 per request
- Delay 0.7s antar batch
- Retry otomatis saat 429 — deteksi via string matching (kompatibel semua versi google-genai)
"""
import asyncio
import re
from typing import List, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

TASK_TYPE_MAP = {
    "retrieval_document":  "RETRIEVAL_DOCUMENT",
    "retrieval_query":     "RETRIEVAL_QUERY",
    "semantic_similarity": "SEMANTIC_SIMILARITY",
    "classification":      "CLASSIFICATION",
    "clustering":          "CLUSTERING",
}

BATCH_SIZE_SAFE = 10    # item per API call
BATCH_DELAY_SEC = 0.7   # jeda antar batch → ~85 req/menit, aman di bawah limit 100
MAX_RETRY       = 5     # maks retry saat 429
RETRY_BASE_SEC  = 32.0  # fallback wait jika retryDelay tidak terbaca dari response


def _is_rate_limit_error(e: Exception) -> bool:
    """
    Deteksi 429 / RESOURCE_EXHAUSTED dari exception.

    Gunakan string matching saja — TIDAK pakai atribut seperti .status_code
    karena nama atribut berbeda antar versi google-genai:
      - versi lama : e.status_code
      - versi baru : tidak ada .status_code, cukup str(e) berisi '429'
    """
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _parse_retry_delay(e: Exception) -> float:
    """
    Baca retryDelay dari pesan error Gemini.
    Contoh pesan: 'Please retry in 30.363255193s.'
    Contoh dict:  "'retryDelay': '30s'"
    Fallback ke RETRY_BASE_SEC jika tidak ditemukan.
    """
    msg = str(e)

    # Format 1: "Please retry in 14.88s"
    m = re.search(r'retry in ([\d.]+)s', msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0

    # Format 2: "'retryDelay': '30s'"
    m = re.search(r"retryDelay['\"]:\s*['\"]?([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 2.0

    return RETRY_BASE_SEC


class EmbeddingService:
    _dimension: Optional[int] = None

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY belum diset di .env")

        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model  = settings.GEMINI_EMBEDDING_MODEL

    # ── Single embed ──────────────────────────────────────────────────────────

    async def embed_text(
        self, text: str, task_type: str = "retrieval_document"
    ) -> List[float]:
        """Embed satu teks dengan retry otomatis saat 429."""
        task = TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT")

        for attempt in range(1, MAX_RETRY + 1):
            try:
                loop     = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.embed_content(
                        model=self.model,
                        contents=text,
                        config={"task_type": task},
                    ),
                )
                vector = response.embeddings[0].values
                if EmbeddingService._dimension is None:
                    EmbeddingService._dimension = len(vector)
                    logger.info(f"Embedding dimension: {EmbeddingService._dimension}")
                return list(vector)

            except Exception as e:
                if _is_rate_limit_error(e):
                    wait = _parse_retry_delay(e)
                    logger.warning(
                        f"[Embed] 429 rate limit (attempt {attempt}/{MAX_RETRY}). "
                        f"Waiting {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                    if attempt == MAX_RETRY:
                        raise
                else:
                    raise

    # ── One batch dengan retry ────────────────────────────────────────────────

    async def _embed_one_batch(
        self,
        batch: List[str],
        task: str,
        batch_num: int,
        total_batches: int,
    ) -> List[List[float]]:
        """Kirim satu batch ke Gemini dengan retry otomatis saat 429."""
        for attempt in range(1, MAX_RETRY + 1):
            try:
                loop     = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda b=batch: self.client.models.embed_content(
                        model=self.model,
                        contents=b,
                        config={"task_type": task},
                    ),
                )
                vectors = [list(emb.values) for emb in response.embeddings]

                if EmbeddingService._dimension is None and vectors:
                    EmbeddingService._dimension = len(vectors[0])
                    logger.info(f"Embedding dimension: {EmbeddingService._dimension}")

                logger.info(
                    f"[Embed] Batch {batch_num}/{total_batches} ✅ "
                    f"({len(batch)} items)"
                )
                return vectors

            except Exception as e:
                if _is_rate_limit_error(e):
                    wait = _parse_retry_delay(e)
                    logger.warning(
                        f"[Embed] Batch {batch_num}/{total_batches} — "
                        f"429 rate limit (attempt {attempt}/{MAX_RETRY}). "
                        f"Waiting {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                    if attempt == MAX_RETRY:
                        logger.error(
                            f"[Embed] Batch {batch_num} gagal setelah {MAX_RETRY}x retry."
                        )
                        raise
                else:
                    raise

        return []

    # ── Batch semua teks ──────────────────────────────────────────────────────

    async def embed_batch(
        self, texts: List[str], task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """
        Embed banyak teks:
        - BATCH_SIZE_SAFE item per API call
        - Delay BATCH_DELAY_SEC antar batch
        - Retry otomatis saat 429
        """
        if not texts:
            return []

        task          = TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT")
        all_vectors:  List[List[float]] = []
        total_batches = (len(texts) + BATCH_SIZE_SAFE - 1) // BATCH_SIZE_SAFE

        logger.info(
            f"[Embed] Starting {len(texts)} texts | "
            f"{total_batches} batches × {BATCH_SIZE_SAFE} items | "
            f"delay={BATCH_DELAY_SEC}s/batch"
        )

        for batch_num, i in enumerate(
            range(0, len(texts), BATCH_SIZE_SAFE), start=1
        ):
            batch   = texts[i : i + BATCH_SIZE_SAFE]
            vectors = await self._embed_one_batch(
                batch, task, batch_num, total_batches
            )
            all_vectors.extend(vectors)

            # Delay antar batch (kecuali batch terakhir)
            if batch_num < total_batches:
                await asyncio.sleep(BATCH_DELAY_SEC)

        logger.info(f"[Embed] ✅ Done: {len(all_vectors)} vectors total")
        return all_vectors

    # ── embed_chunks ──────────────────────────────────────────────────────────

    async def embed_chunks(
        self, chunks: List[dict], metadata: dict = None
    ) -> List[dict]:
        """Embed list of chunk dicts. Setiap chunk harus punya key 'text'."""
        if not chunks:
            return []

        chunk_texts     = [c["text"] for c in chunks]
        chunk_metadatas = [c.get("metadata") or {} for c in chunks]

        vectors = await self.embed_batch(chunk_texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(vectors)} vectors vs {len(chunks)} chunks."
            )

        result = []
        for i in range(len(chunks)):
            combined_meta                = {**(metadata or {}), **chunk_metadatas[i]}
            combined_meta["chunk_index"] = i
            result.append({
                "text":     chunk_texts[i],
                "vector":   vectors[i],
                "metadata": combined_meta,
            })

        return result

    @classmethod
    def get_dimension(cls) -> int:
        return cls._dimension or settings.QDRANT_EMBEDDING_DIM