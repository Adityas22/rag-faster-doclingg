"""
Schemas untuk endpoint /generate.

Skenario:
  1. Generate Materi  → POST /generate/material
     Input: course_id, topic, language
     Output: structured material JSON (title, introduction, content, sections, key_points, summary)

  2. Generate Flashcard & Quiz → POST /generate/flashcard | /generate/quiz
     Input: course_id, source_file (opsional), topic, material_context (opsional)
            quiz_type: "essay" | "multiple_choice_single" | "multiple_choice_multiple"
            count: jumlah soal/flashcard (dinamis)
     Output: flashcards[] / quiz items[]

Async via Celery + Redis:
  POST  → dispatch → return task_id
  GET   /generate/status/{task_id} → poll hasil
"""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Any


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class GenerateMaterialRequest(BaseModel):
    """
    Request untuk generate materi dari PDF/Audio yang sudah di-upload ke Qdrant.
    Cukup dengan course_id — semua file dalam course tersebut akan digunakan.
    Hasil materi ini (JSON) nantinya disimpan di backend Laravel (MySQL).
    """
    course_id: str = Field(
        ...,
        description="ID course dari Laravel (wajib). Semua file dalam course ini akan digunakan.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topik spesifik yang ingin di-generate. Kosong = seluruh materi dalam course.",
    )
    language: Literal["id", "en"] = Field(
        default="id",
        description="'id' = Bahasa Indonesia, 'en' = English.",
    )

    @model_validator(mode="after")
    def _normalize(self) -> "GenerateMaterialRequest":
        for field in ("topic",):
            v = getattr(self, field)
            if isinstance(v, str) and v.strip() == "":
                setattr(self, field, None)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "course_id": "course-abc123",
                "topic": "Fotosintesis",
                "language": "id",
            }
        }
    }


class GenerateFlashcardRequest(BaseModel):
    """
    Request untuk generate flashcard dari materi yang sudah ada di Qdrant.
    Bisa menggunakan material_data (JSON dari Laravel) atau langsung course_id.
    """
    course_id: Optional[str] = Field(
        default=None,
        description="ID course untuk filter data di Qdrant.",
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Filter berdasarkan nama file (opsional).",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topik spesifik untuk flashcard (opsional).",
    )
    material_context: Optional[str] = Field(
        default=None,
        description=(
            "Konteks materi tambahan dari Laravel (opsional). "
            "Isi dengan field 'content' dari hasil generate materi "
            "untuk hasil flashcard yang lebih akurat."
        ),
    )
    count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah flashcard yang diinginkan (1–50).",
    )
    language: Literal["id", "en"] = Field(
        default="id",
        description="'id' = Bahasa Indonesia, 'en' = English.",
    )

    @model_validator(mode="after")
    def _normalize(self) -> "GenerateFlashcardRequest":
        for field in ("course_id", "source_file", "topic", "material_context"):
            v = getattr(self, field)
            if isinstance(v, str) and v.strip() == "":
                setattr(self, field, None)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "course_id": "course-abc123",
                "topic": "Fotosintesis",
                "count": 15,
                "language": "id",
            }
        }
    }


class GenerateQuizRequest(BaseModel):
    """
    Request untuk generate quiz dengan tipe soal dinamis.

    quiz_type:
      - "essay"                    → soal esai dengan contoh jawaban + poin kunci
      - "multiple_choice_single"   → pilihan ganda, 1 jawaban benar (A/B/C/D)
      - "multiple_choice_multiple" → pilihan ganda, bisa lebih dari 1 jawaban benar
    """
    course_id: Optional[str] = Field(
        default=None,
        description="ID course untuk filter data di Qdrant.",
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Filter berdasarkan nama file (opsional).",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topik spesifik untuk soal (opsional).",
    )
    material_context: Optional[str] = Field(
        default=None,
        description=(
            "Konteks materi dari Laravel (opsional). "
            "Diisi dengan content dari hasil generate materi."
        ),
    )
    quiz_type: Literal["essay", "multiple_choice_single", "multiple_choice_multiple"] = Field(
        default="multiple_choice_single",
        description=(
            "'essay' = soal uraian, "
            "'multiple_choice_single' = pilihan ganda 1 jawaban, "
            "'multiple_choice_multiple' = pilihan ganda bisa >1 jawaban benar."
        ),
    )
    count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Jumlah soal yang diinginkan (1–50).",
    )
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default="medium",
        description="Tingkat kesulitan soal.",
    )
    language: Literal["id", "en"] = Field(
        default="id",
        description="'id' = Bahasa Indonesia, 'en' = English.",
    )

    @model_validator(mode="after")
    def _normalize(self) -> "GenerateQuizRequest":
        for field in ("course_id", "source_file", "topic", "material_context"):
            v = getattr(self, field)
            if isinstance(v, str) and v.strip() == "":
                setattr(self, field, None)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "course_id": "course-abc123",
                "topic": "Fotosintesis",
                "quiz_type": "multiple_choice_single",
                "count": 10,
                "difficulty": "medium",
                "language": "id",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK RESPONSE (saat dispatch ke Celery)
# ─────────────────────────────────────────────────────────────────────────────

class GenerateTaskResponse(BaseModel):
    """Response saat task berhasil di-dispatch ke Celery."""
    task_id: str = Field(..., description="UUID task — gunakan untuk poll status")
    celery_task_id: str = Field(..., description="ID internal Celery task")
    status: Literal["pending"] = Field(default="pending")
    generate_type: str = Field(..., description="'material' | 'flashcard' | 'quiz'")
    quiz_type: Optional[str] = Field(
        default=None,
        description="Hanya diisi jika generate_type='quiz': essay | multiple_choice_single | multiple_choice_multiple",
    )
    course_id: Optional[str] = None
    source_file: Optional[str] = None  # hanya untuk flashcard & quiz
    topic: Optional[str] = None
    count: Optional[int] = None
    difficulty: Optional[str] = None
    language: str
    created_at: str
    status_url: str = Field(..., description="URL untuk poll status task")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS RESPONSE (saat poll GET /status/{task_id})
# ─────────────────────────────────────────────────────────────────────────────

class GenerateStatusResponse(BaseModel):
    """
    Response untuk GET /generate/status/{task_id}.
    Field 'result' berisi output generate jika status == 'done'.
    """
    task_id: str
    celery_task_id: Optional[str] = None
    status: Literal["pending", "processing", "done", "error"]
    generate_type: Optional[str] = None
    quiz_type: Optional[str] = None
    course_id: Optional[str] = None
    source_file: Optional[str] = None
    topic: Optional[str] = None
    count: Optional[int] = None
    difficulty: Optional[str] = None
    language: Optional[str] = None
    result: Optional[Any] = Field(
        default=None,
        description=(
            "Hasil generate — tersedia jika status='done'. "
            "Struktur berbeda tergantung generate_type: "
            "material={title,introduction,content,sections,key_points,summary}, "
            "flashcard={count,flashcards[]}, "
            "quiz={quiz_type,count,difficulty,items[]}"
        ),
    )
    error: Optional[str] = Field(default=None, description="Pesan error jika status='error'")
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# RESULT MODELS — struktur output untuk tiap generate_type
# ─────────────────────────────────────────────────────────────────────────────

# ── Material ──────────────────────────────────────────────────────────────────

class MaterialSection(BaseModel):
    """Bagian/sub-topik dalam materi."""
    heading: str = Field(..., description="Judul sub-topik")
    body: str = Field(..., description="Isi penjelasan sub-topik")


class MaterialResult(BaseModel):
    """
    Hasil generate materi — ini yang disimpan di Laravel (MySQL).
    Semua field di-serialize sebagai JSON di database.
    """
    title: str = Field(..., description="Judul materi")
    introduction: str = Field(..., description="Paragraf pembuka")
    sections: List[MaterialSection] = Field(
        default=[],
        description="Daftar sub-topik dengan heading + body",
    )
    content: str = Field(..., description="Isi materi lengkap dalam satu blok teks")
    key_points: List[str] = Field(
        default=[],
        description="Poin-poin penting dari materi",
    )
    summary: str = Field(..., description="Kesimpulan/ringkasan singkat")
    context_chunks_used: int = Field(default=0)
    context_scope: str = Field(default="unknown")


# ── Flashcard ─────────────────────────────────────────────────────────────────

class FlashcardItem(BaseModel):
    """Satu kartu flashcard."""
    front: str = Field(..., description="Kata kunci / konsep (tampilan depan kartu)")
    back: str = Field(..., description="Penjelasan 2–4 kalimat (tampilan belakang kartu)")


class FlashcardResult(BaseModel):
    """Hasil generate flashcard."""
    count: int
    flashcards: List[FlashcardItem]
    context_chunks_used: int = 0
    context_scope: str = "unknown"


# ── Quiz — Essay ──────────────────────────────────────────────────────────────

class QuizEssayItem(BaseModel):
    """Satu soal essay."""
    question: str = Field(..., description="Pertanyaan esai")
    sample_answer: str = Field(..., description="Contoh jawaban ideal")
    key_points: List[str] = Field(default=[], description="Poin-poin penilaian")
    score_weight: Optional[int] = Field(
        default=None,
        description="Bobot nilai (opsional, 1–100)",
    )


# ── Quiz — Multiple Choice Single ────────────────────────────────────────────

class MCOption(BaseModel):
    """Satu pilihan jawaban."""
    label: str = Field(..., description="Label pilihan: A, B, C, D")
    text: str = Field(..., description="Teks pilihan jawaban")


class QuizMCSingleItem(BaseModel):
    """Satu soal MC dengan 1 jawaban benar."""
    question: str
    options: List[MCOption] = Field(..., description="4 pilihan jawaban")
    correct_answer: str = Field(..., description="Label jawaban benar, misal 'A'")
    explanation: Optional[str] = Field(default=None, description="Penjelasan mengapa benar")


# ── Quiz — Multiple Choice Multiple ──────────────────────────────────────────

class QuizMCMultipleItem(BaseModel):
    """Satu soal MC dengan >1 jawaban benar."""
    question: str
    options: List[MCOption] = Field(..., description="4–6 pilihan jawaban")
    correct_answers: List[str] = Field(
        ...,
        description="Daftar label jawaban benar, misal ['A', 'C']",
    )
    explanation: Optional[str] = Field(default=None, description="Penjelasan")


# ── Quiz Result ───────────────────────────────────────────────────────────────

class QuizResult(BaseModel):
    """Hasil generate quiz — semua tipe."""
    quiz_type: str = Field(
        ...,
        description="essay | multiple_choice_single | multiple_choice_multiple",
    )
    count: int
    difficulty: Optional[str] = None
    items: List[Any] = Field(
        ...,
        description=(
            "List soal. Struktur tiap item tergantung quiz_type:\n"
            "- essay: QuizEssayItem\n"
            "- multiple_choice_single: QuizMCSingleItem\n"
            "- multiple_choice_multiple: QuizMCMultipleItem"
        ),
    )
    context_chunks_used: int = 0
    context_scope: str = "unknown"