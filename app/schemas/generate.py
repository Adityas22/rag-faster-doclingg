from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Any


# ── Request ───────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    course_id: Optional[str] = Field(
        default=None,
        description=(
            "ID course (opsional). "
            "Diisi → ambil data dari course tersebut saja. "
            "Kosong / null → global (seluruh Qdrant)."
        ),
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Filter by nama file tertentu (opsional).",
    )
    content_type: Literal["flashcard", "quiz_mc", "quiz_essay", "summary", "material"] = Field(
        default="flashcard",
        description="Di-override otomatis oleh masing-masing endpoint.",
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Topik spesifik yang ingin di-generate (opsional). "
            "Jika diisi, sistem hanya mengambil chunk yang relevan dengan topik ini."
        ),
    )
    count: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Jumlah item (flashcard / soal / key_points / examples+exercises).",
    )
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default=None,
        description="Tingkat kesulitan — hanya untuk quiz_mc dan quiz_essay.",
    )
    language: Literal["id", "en"] = Field(
        default="id",
        description="'id' = Bahasa Indonesia, 'en' = English.",
    )

    @model_validator(mode="after")
    def _normalize_empty_strings(self) -> "GenerateRequest":
        """
        String kosong ("") dinormalisasi ke None agar tidak lolos sebagai
        filter Qdrant yang tidak valid (course_id="" akan filter nothing).
        """
        if isinstance(self.course_id, str) and self.course_id.strip() == "":
            self.course_id = None
        if isinstance(self.source_file, str) and self.source_file.strip() == "":
            self.source_file = None
        if isinstance(self.topic, str) and self.topic.strip() == "":
            self.topic = None
        return self

    # PENTING: format examples di sini adalah request body langsung —
    # field langsung di root JSON, TIDAK dibungkus "value" atau "summary".
    # Format {"summary": ..., "value": {...}} adalah format Swagger UI lama
    # yang TIDAK kompatibel dengan Pydantic v2 dan akan membuat topic tidak terbaca.
    model_config = {
        "json_schema_extra": {
            "example": {
                "course_id": "course-abc",
                "topic": "pisang",
                "count": 5,
                "language": "id",
            }
        }
    }


# ── Flashcard ─────────────────────────────────────────────────────────────────
class FlashcardItem(BaseModel):
    front: str = Field(..., description="Heading / kata kunci (tampilan depan kartu)")
    back: str = Field(..., description="Penjelasan detail 1-2 kalimat (tampilan belakang kartu)")


class GenerateFlashcardResponse(BaseModel):
    count: int
    flashcards: List[Any]
    context_chunks_used: int
    context_scope: str


# ── Quiz ──────────────────────────────────────────────────────────────────────
class QuizMCOption(BaseModel):
    label: str
    text: str


class QuizMCItem(BaseModel):
    question: str
    options: List[QuizMCOption]
    correct_answer: str
    explanation: Optional[str] = None


class QuizEssayItem(BaseModel):
    question: str
    sample_answer: str
    key_points: List[str] = []


class GenerateQuizResponse(BaseModel):
    content_type: str
    count: int
    difficulty: Optional[str] = None  # Hanya ada di sini
    items: List[Any]
    context_chunks_used: int
    context_scope: str


# ── Summary ───────────────────────────────────────────────────────────────────
class GenerateSummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    key_points_count: int
    context_chunks_used: int
    context_scope: str


# ── Material ──────────────────────────────────────────────────────────────────
class GenerateMaterialResponse(BaseModel):
    title: str
    introduction: str
    content: str
    examples: List[str] = []
    exercises: List[str] = []
    examples_count: int
    exercises_count: int
    context_chunks_used: int
    context_scope: str