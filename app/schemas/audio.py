from pydantic import BaseModel, Field
from typing import Optional, List, Any


class SegmentResult(BaseModel):
    start: float
    end: float
    text: str


class TranscribeFullResponse(BaseModel):
    task_id: str
    status: str
    course_id: str
    source_file: str
    content: str
    language: str
    language_probability: float
    duration: float
    transcription_time: Optional[float] = Field(None, description="Waktu transkripsi (detik)")
    realtime_factor: Optional[float] = Field(None, description="Kecepatan relatif (RTF): >1=lebih cepat dari real-time")
    segments: List[SegmentResult] = []
    total_chunks: int
    chunks_stored: int


class TranscribeStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    total: int
    tasks: List[dict] = []


class AudioInfoResponse(BaseModel):
    file_name: str
    format: str
    file_size_mb: float
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_formatted: Optional[str] = None
