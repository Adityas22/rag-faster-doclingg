"""
QdrantService — Vector DB operations using Qdrant Cloud.
- Append mode (ID global berlanjut)
- Payload indexes wajib untuk filter (course_id dll)
- search() tanpa score_threshold default agar tidak 0 hasil
"""
import asyncio
import time
from typing import List, Optional, Dict, Any
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantService:

    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )
        self.collection = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.QDRANT_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"✅ Qdrant collection created: {self.collection}")
        else:
            logger.debug(f"Qdrant collection exists: {self.collection}")

        # Wajib: buat payload index agar filter bisa jalan di Qdrant Cloud
        self._ensure_indexes()

    def _ensure_indexes(self):
        """
        Buat payload index untuk field yang difilter.
        Qdrant Cloud MENSYARATKAN index untuk setiap field yang dipakai di Filter().
        Aman dipanggil berkali-kali — jika sudah ada, tidak error.
        """
        from qdrant_client.models import PayloadSchemaType

        indexes = [
            ("course_id",   PayloadSchemaType.KEYWORD),
            ("source_file", PayloadSchemaType.KEYWORD),
            ("type",        PayloadSchemaType.KEYWORD),
            ("task_id",     PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
        ]

        for field_name, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(f"Payload index ensured: {field_name}")
            except Exception as e:
                err = str(e).lower()
                if "already exists" in err or "conflict" in err or "400" in err:
                    pass  # index sudah ada, tidak masalah
                else:
                    logger.warning(f"Index '{field_name}' warning: {e}")

    def _get_existing_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count or 0
        except Exception:
            return 0

    async def upsert_vectors(self, items: List[dict]) -> int:
        """
        Simpan vectors ke Qdrant dengan Append Mode.
        ID global berlanjut dari jumlah points yang sudah ada.
        """
        from qdrant_client.models import PointStruct

        if not items:
            return 0

        loop = asyncio.get_event_loop()
        existing_count = await loop.run_in_executor(None, self._get_existing_count)
        logger.info(
            f"[Qdrant] Upserting {len(items)} vectors | "
            f"ID: {existing_count} → {existing_count + len(items) - 1}"
        )

        upload_ts = time.time()
        points = []

        for i, item in enumerate(items):
            global_id = existing_count + i
            meta = item.get("metadata", {})

            points.append(
                PointStruct(
                    id=global_id,
                    vector=item["vector"],
                    payload={
                        "text": item["text"],
                        "original_text": item["text"],
                        "global_chunk_id": global_id,
                        "upload_timestamp": upload_ts,
                        **meta,
                    },
                )
            )

        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.collection,
                points=points,
            ),
        )

        logger.info(
            f"✅ Upserted {len(points)} vectors "
            f"(total: {existing_count + len(points)} points)"
        )
        return len(points)

    async def search(
        self,
        vector: List[float],
        course_id: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[dict]:
        """
        Similarity search dengan filter opsional.

        Mode filter:
        - course_id + source_file → filter keduanya (AND)
        - course_id saja          → filter by course
        - source_file saja        → filter by file
        - keduanya None           → global (tanpa filter, seluruh collection)

        score_threshold=None → tidak ada threshold, selalu return top_k hasil.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must_conditions = []
        if course_id:
            must_conditions.append(
                FieldCondition(key="course_id", match=MatchValue(value=course_id))
            )
        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        loop = asyncio.get_event_loop()

        search_kwargs: dict = dict(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        results = await loop.run_in_executor(
            None,
            lambda: self.client.search(**search_kwargs),
        )

        logger.debug(f"[Qdrant] Search → {len(results)} results | course_id={course_id}")

        return [
            {
                "id": str(r.id),
                "score": round(r.score, 4),
                "text": r.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in r.payload.items()
                    if k not in ("text", "original_text")
                },
            }
            for r in results
        ]

    async def retrieve_all_context(
        self,
        course_id: Optional[str] = None,
        source_file: Optional[str] = None,
        max_tokens: int = 100_000,
    ) -> Dict[str, Any]:
        """Ambil SEMUA data dari Qdrant dengan pagination."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        loop = asyncio.get_event_loop()

        must_conditions = []
        if course_id:
            must_conditions.append(
                FieldCondition(key="course_id", match=MatchValue(value=course_id))
            )
        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )

        scroll_filter = Filter(must=must_conditions) if must_conditions else None

        all_points = []
        offset = None

        while True:
            result = await loop.run_in_executor(
                None,
                lambda off=offset: self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=off,
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            points, next_offset = result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return {
                "full_context": "",
                "total_points": 0,
                "total_tokens": 0,
                "source_files": [],
                "chunks_detail": [],
            }

        all_points.sort(key=lambda p: (
            p.payload.get("source_file", ""),
            p.payload.get("chunk_index", 0),
        ))

        from datetime import timedelta
        context_parts = []
        chunks_detail = []
        source_files: set = set()
        current_source = None

        for point in all_points:
            payload = point.payload
            src = payload.get("source_file", "unknown")
            source_files.add(src)

            if src != current_source:
                current_source = src
                context_parts.append(f"\n=== Sumber: {src} ===")

            ts_start = payload.get("timestamp_start")
            ts_end = payload.get("timestamp_end")
            if ts_start is not None and ts_end is not None:
                ts_label = (
                    f"[{str(timedelta(seconds=int(ts_start)))} - "
                    f"{str(timedelta(seconds=int(ts_end)))}]"
                )
            else:
                ts_label = f"[Chunk {payload.get('chunk_index', '?')}]"

            text = payload.get("text", "")
            context_parts.append(f"{ts_label} {text}")

            chunks_detail.append({
                "id": point.id,
                "source_file": src,
                "chunk_index": payload.get("chunk_index"),
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "token_count": payload.get("token_count", 0),
            })

        full_context = "\n".join(context_parts)

        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens = len(enc.encode(full_context))
            if total_tokens > max_tokens:
                tokens = enc.encode(full_context)
                full_context = enc.decode(tokens[:max_tokens])
                total_tokens = max_tokens
        except ImportError:
            total_tokens = len(full_context) // 4

        return {
            "full_context": full_context,
            "total_points": len(all_points),
            "total_tokens": total_tokens,
            "source_files": sorted(source_files),
            "chunks_detail": chunks_detail,
        }

    async def delete_by_course_id(self, course_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        loop = asyncio.get_event_loop()

        count_result = await loop.run_in_executor(
            None,
            lambda: self.client.count(
                collection_name=self.collection,
                count_filter=Filter(
                    must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
                ),
                exact=True,
            ),
        )
        total = count_result.count
        logger.info(f"[Qdrant] Found {total} vectors | course_id={course_id}")

        if total == 0:
            return 0

        await loop.run_in_executor(
            None,
            lambda: self.client.delete(
                collection_name=self.collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
                    )
                ),
            ),
        )

        logger.info(f"[Qdrant] ✅ Deleted {total} vectors | course_id={course_id}")
        return total

    async def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
            "config": {
                "distance": str(info.config.params.vectors.distance),
                "dimension": info.config.params.vectors.size,
            },
        }
