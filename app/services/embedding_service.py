"""
EmbeddingService — Generate vector embeddings via Google Gemini API.
"""
import asyncio
from typing import List, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

TASK_TYPE_MAP = {
    "retrieval_document": "RETRIEVAL_DOCUMENT",
    "retrieval_query": "RETRIEVAL_QUERY",
    "semantic_similarity": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


class EmbeddingService:
    _dimension: Optional[int] = None

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY belum diset di .env")

        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_EMBEDDING_MODEL

    async def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Embed satu teks, return vector list[float]."""
        task = TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"task_type": task},
            ),
        )
        vector = response.embeddings[0].values

        # Cache dimension
        if EmbeddingService._dimension is None:
            EmbeddingService._dimension = len(vector)
            logger.info(f"Embedding dimension detected: {EmbeddingService._dimension}")

        return list(vector)

    async def embed_batch(
        self, texts: List[str], task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """
        Embed banyak teks sekaligus menggunakan batching.
        Batas Gemini API: 100 item per request.
        """
        task = TASK_TYPE_MAP.get(task_type, "RETRIEVAL_DOCUMENT")
        batch_size = settings.BATCH_EMBED_SIZE
        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self.client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config={"task_type": task},
                ),
            )
            for emb in response.embeddings:
                all_vectors.append(list(emb.values))

        return all_vectors

    async def embed_chunks(
        self, chunks: List[dict], metadata: dict = None
    ) -> List[dict]:
        if not chunks:
            return []

        # Simpan teks dan metadata sebelum embed
        chunk_texts = [c["text"] for c in chunks]
        chunk_metadatas = [c.get("metadata") or {} for c in chunks]

        # Embed semua teks
        vectors = await self.embed_batch(chunk_texts)

        # Validasi jumlah vector == jumlah chunk
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(vectors)} vectors vs {len(chunks)} chunks."
            )

        # Gabungkan teks + vector + metadata per chunk
        result = []
        for i in range(len(chunks)):
            # chunk_metadatas[i] lebih prioritas dari fallback metadata
            combined_meta = {**(metadata or {}), **chunk_metadatas[i]}
            combined_meta["chunk_index"] = i

            result.append({
                "text": chunk_texts[i],
                "vector": vectors[i],
                "metadata": combined_meta,
            })

        return result

    @classmethod
    def get_dimension(cls) -> int:
        return cls._dimension or settings.QDRANT_EMBEDDING_DIM
