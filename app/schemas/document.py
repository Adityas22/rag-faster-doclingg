from pydantic import BaseModel, Field
from typing import Optional, List


# ── Upload / Extract ──────────────────────────────────────────────────────────
class DocumentUploadResponse(BaseModel):
    """Response saat upload selesai diproses (sync)."""
    task_id: str
    status: str
    filename: str
    size_mb: float
    course_id: str
    doc_title: str
    total_pages: int
    total_chars: int
    total_chunks: int
    chunks_stored: int
    message: str


# ── Task Registry ─────────────────────────────────────────────────────────────
class DocumentTask(BaseModel):
    task_id: str
    status: str
    course_id: str
    filename: str
    doc_title: Optional[str] = None
    total_pages: Optional[int] = None
    total_chars: Optional[int] = None
    total_chunks: Optional[int] = None
    chunks_stored: Optional[int] = None
    error: Optional[str] = None
    created_at: str


class DocumentTaskListResponse(BaseModel):
    total: int
    tasks: List[dict] = []


# ── Backward compat (masih dipakai worker) ────────────────────────────────────
class DocumentExtractResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    size_mb: float
    course_id: str
    message: str


class DocumentResult(BaseModel):
    source_file: str
    course_id: str
    doc_title: str
    total_pages: int
    total_chars: int
    total_chunks: int
    chunks_stored: int
    language: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[DocumentResult] = None
    error: Optional[str] = None

# alias
DocumentFullResponse = DocumentUploadResponse
