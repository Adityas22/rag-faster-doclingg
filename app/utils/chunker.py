"""
Text Chunking Utility — Recursive Token-Based Chunker (dari notebook faster_whisper).
Menggunakan tiktoken untuk penghitungan token yang akurat (bukan estimasi karakter).
"""
import re
from typing import List, Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RecursiveTokenChunker:
    """
    Chunker berbasis token (tiktoken cl100k_base).
    - Prioritaskan split berdasarkan segmen waktu audio jika tersedia
    - Fallback ke word-based split untuk plain text
    Sesuai implementasi notebook faster_whisper_mp3_extraction.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.warning("tiktoken not installed. Falling back to character-based chunking.")
            self.encoding = None

    def _token_len(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        # fallback estimasi: 1 token ≈ 4 chars
        return len(text) // 4

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking berbasis segmen (audio dengan timestamp)
    # ─────────────────────────────────────────────────────────────────────────
    def chunk_segments(self, segments: List[dict]) -> List[dict]:
        """
        Chunk berdasarkan segmen waktu audio.

        Args:
            segments: List of {"text": str, "start": float, "end": float}

        Returns:
            List of {
                "text": str, "chunk_index": int,
                "timestamp_start": float, "timestamp_end": float,
                "token_count": int
            }
        """
        if not segments:
            return []

        groups = self._group_segments_by_tokens(segments)
        chunks = []

        for i, group in enumerate(groups):
            chunk_text = " ".join(seg["text"].strip() for seg in group)
            token_count = self._token_len(chunk_text)
            timestamp_start = group[0].get("start")
            timestamp_end = group[-1].get("end")

            chunks.append({
                "text": chunk_text,
                "chunk_index": i,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "token_count": token_count,
            })

            logger.debug(
                f"Chunk {i+1:03d} | "
                f"start={timestamp_start:.1f}s end={timestamp_end:.1f}s | "
                f"{token_count} tokens | {len(group)} segs"
            )

        logger.info(
            f"Segment chunking done: {len(segments)} segs → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _group_segments_by_tokens(self, segments: List[dict]) -> List[List[dict]]:
        """Group segmen audio berdasarkan batas token (dengan overlap)."""
        groups = []
        current_group: List[dict] = []
        current_tokens = 0

        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_tokens = self._token_len(seg.get("text", ""))

            if current_tokens + seg_tokens > self.chunk_size and current_group:
                groups.append(current_group)

                # Overlap: ambil segmen terakhir yg masih cukup untuk overlap
                overlap_group: List[dict] = []
                overlap_tokens = 0
                for seg_back in reversed(current_group):
                    t = self._token_len(seg_back.get("text", ""))
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap_group.insert(0, seg_back)
                        overlap_tokens += t
                    else:
                        break

                current_group = overlap_group
                current_tokens = overlap_tokens

            current_group.append(seg)
            current_tokens += seg_tokens
            i += 1

        if current_group:
            groups.append(current_group)

        return groups

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking plain text (fallback, tanpa timestamp)
    # ─────────────────────────────────────────────────────────────────────────
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk plain text berdasarkan token count.

        Args:
            text: Teks yang akan dipecah

        Returns:
            List of chunk strings
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if self._token_len(text) <= self.chunk_size:
            return [text]

        chunks = self._split_by_words(text)
        chunks = [c.strip() for c in chunks if c.strip()]

        logger.debug(
            f"Text chunking: {self._token_len(text)} tokens → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _split_by_words(self, text: str) -> List[str]:
        """Word-based split dengan token overlap."""
        chunks: List[str] = []
        current_words: List[str] = []
        current_tokens = 0

        words = text.split()
        i = 0

        while i < len(words):
            word = words[i]
            word_tokens = self._token_len(word)

            if current_tokens + word_tokens > self.chunk_size and current_words:
                chunks.append(" ".join(current_words))

                # Overlap
                overlap_words: List[str] = []
                overlap_tokens = 0
                for w in reversed(current_words):
                    wt = self._token_len(w)
                    if overlap_tokens + wt <= self.chunk_overlap:
                        overlap_words.insert(0, w)
                        overlap_tokens += wt
                    else:
                        break

                current_words = overlap_words
                current_tokens = overlap_tokens

            current_words.append(word)
            current_tokens += word_tokens
            i += 1

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks


# ── Singleton chunker ──────────────────────────────────────────────────────────
_chunker: Optional[RecursiveTokenChunker] = None


def get_chunker() -> RecursiveTokenChunker:
    global _chunker
    if _chunker is None:
        _chunker = RecursiveTokenChunker()
    return _chunker


# ── Public helpers ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Chunk plain text. Wrapper untuk backward compatibility dengan endpoint/worker.
    """
    if chunk_size or chunk_overlap:
        chunker = RecursiveTokenChunker(
            chunk_size=chunk_size or settings.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        )
    else:
        chunker = get_chunker()
    return chunker.chunk_text(text)


def chunk_segments(segments: List[dict]) -> List[dict]:
    """
    Chunk audio segments (dengan timestamp). 
    Digunakan oleh whisper service setelah transkripsi.
    """
    return get_chunker().chunk_segments(segments)


def count_tokens(text: str) -> int:
    """Hitung token aktual menggunakan tiktoken."""
    return get_chunker()._token_len(text)
