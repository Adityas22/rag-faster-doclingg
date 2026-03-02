from pydantic import BaseModel
from typing import Optional, List


# ── Chunk preview (mirip SegmentResult di audio) ──────────────────────────────

class DocumentChunkPreview(BaseModel):
    chunk_index:     int
    section:         str
    page:            Optional[int] = None
    chunk_id:        int
    total_words:     int
    content_preview: str


# ── Upload Response (mirip TranscribeFullResponse di audio) ───────────────────

class DocumentUploadResponse(BaseModel):
    document_id:    str
    status:         str
    course_id:      str
    source_file:    str
    doc_title:      str
    filename:       str
    size_mb:        float
    total_pages:    int
    total_chars:    int
    total_sections: int
    total_chunks:   int
    chunks_stored:  int
    chunks:         List[DocumentChunkPreview] = []
    message:        str


# ── Task list (sama persis dengan audio TaskListResponse) ─────────────────────

class DocumentTaskListResponse(BaseModel):
    total: int
    tasks: List[dict] = []