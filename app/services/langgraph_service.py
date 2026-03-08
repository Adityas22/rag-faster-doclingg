"""
Hybrid RAG Workflow — Generate Material, Flashcard, dan Quiz.

Skenario:
  1. Generate Materi  → run_generate_material(request)
     Output: title, introduction, sections[], content, key_points[], summary

  2. Generate Flashcard → run_generate_flashcard(request)
     Input opsional: material_context (JSON dari Laravel)
     Output: flashcards[] dengan front + back

  3. Generate Quiz → run_generate_quiz(request)
     quiz_type: essay | multiple_choice_single | multiple_choice_multiple
     Input opsional: material_context (JSON dari Laravel)
     Output: items[] sesuai tipe

Filtering:
  - Score gap detection + keyword relevance check
  - Threshold adaptif: best_score * 0.75
"""
import asyncio
import json
import re
from datetime import timedelta
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Konfigurasi Filtering ─────────────────────────────────────────────────────
_summary_cache: dict = {}
SCORE_RATIO_THRESHOLD = 0.75
MIN_SCORE_ABSOLUTE = 0.40
SCORE_GAP_CUTOFF = 0.15
TOP_RELEVANT_CHUNKS = 8   # sedikit lebih banyak untuk materi lengkap


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(v: Optional[str]) -> Optional[str]:
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _scope_label(course_id: Optional[str], source_file: Optional[str]) -> str:
    if course_id and source_file:
        return f"course:{course_id}|file:{source_file}"
    if course_id:
        return f"course:{course_id}"
    if source_file:
        return f"file:{source_file}"
    return "global"


def _cache_key(course_id, source_file, total_points):
    return f"{course_id or '__all__'}|{source_file or '__all__'}|{total_points}"


# ── Query builder ─────────────────────────────────────────────────────────────
def _build_search_query(
    topic: Optional[str],
    generate_type: str,
    course_id: Optional[str],
    source_file: Optional[str],
    quiz_type: Optional[str] = None,
) -> str:
    type_hints = {
        "material": "penjelasan lengkap, konsep utama, definisi, contoh, penerapan",
        "flashcard": "konsep utama, definisi, poin kunci, istilah penting",
        "quiz_essay": "konsep mendalam, proses, analisis, evaluasi, argumentasi",
        "quiz_multiple_choice_single": "konsep, fakta, perbandingan, prinsip yang bisa diujikan",
        "quiz_multiple_choice_multiple": "konsep, fakta ganda, perbandingan, relasi antar konsep",
    }

    key = f"quiz_{quiz_type}" if generate_type == "quiz" and quiz_type else generate_type
    hint = type_hints.get(key, "materi pembelajaran")

    if topic:
        return f"{topic}: {hint}"
    elif course_id and source_file:
        return f"seluruh materi course {course_id} file {source_file}: {hint}"
    elif course_id:
        return f"seluruh materi course {course_id}: {hint}"
    elif source_file:
        return f"isi file {source_file}: {hint}"
    else:
        return f"rangkuman seluruh materi pembelajaran: {hint}"


# ── Topic-aware relevance filter ──────────────────────────────────────────────
def _filter_relevant_chunks(
    results: list,
    topic: Optional[str] = None,
    max_chunks: int = TOP_RELEVANT_CHUNKS,
) -> tuple:
    if not results:
        return [], False

    best_score = results[0].get("score", 0)
    adaptive_threshold = max(best_score * SCORE_RATIO_THRESHOLD, MIN_SCORE_ABSOLUTE)
    after_threshold = [r for r in results if r.get("score", 0) >= adaptive_threshold]

    if not after_threshold:
        logger.warning(
            f"[HybridRAG] Threshold terlalu ketat (best={best_score:.3f}), fallback top-3"
        )
        after_threshold = results[:3]

    # Score gap detection
    gap_cut = len(after_threshold)
    for i in range(1, len(after_threshold)):
        prev_score = after_threshold[i - 1].get("score", 0)
        curr_score = after_threshold[i].get("score", 0)
        if (prev_score - curr_score) >= SCORE_GAP_CUTOFF:
            gap_cut = i
            break
    after_gap = after_threshold[:gap_cut]

    # Keyword check
    if topic:
        topic_words = [w.lower() for w in topic.split() if len(w) >= 3]
        on_topic = [c for c in after_gap if any(w in c.get("text", "").lower() for w in topic_words)]
        off_topic = [c for c in after_gap if c not in on_topic]
        selected = on_topic[:max_chunks] if on_topic else after_gap[:max_chunks]
        if on_topic and off_topic:
            logger.info(f"[HybridRAG] Keyword filter '{topic}': kept {len(on_topic)}, removed {len(off_topic)}")
    else:
        selected = after_gap[:max_chunks]

    logger.info(
        f"[HybridRAG] Filter: {len(results)} raw → {len(after_threshold)} threshold → "
        f"{len(after_gap)} gap → {len(selected)} final | best={best_score:.3f}"
    )
    return selected, best_score >= MIN_SCORE_ABSOLUTE


# ── Similarity Search ─────────────────────────────────────────────────────────
async def _similarity_search(
    query: str,
    course_id: Optional[str],
    source_file: Optional[str],
    top_k: int = 20,
) -> list:
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant_service import QdrantService

    embed_svc = EmbeddingService()
    qdrant_svc = QdrantService()

    query_vector = await embed_svc.embed_text(query, task_type="retrieval_query")
    results = await qdrant_svc.search(
        vector=query_vector,
        course_id=course_id,
        source_file=source_file,
        top_k=top_k,
    )
    return results


# ── Format Chunks ─────────────────────────────────────────────────────────────
def _format_chunks(results: list) -> str:
    if not results:
        return "Tidak ada chunk relevan ditemukan."
    lines = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score = r.get("score", 0)
        ts_s = meta.get("timestamp_start")
        ts_e = meta.get("timestamp_end")
        if ts_s is not None and ts_e is not None:
            label = f"[{str(timedelta(seconds=int(ts_s)))} - {str(timedelta(seconds=int(ts_e)))}]"
        else:
            label = f"[Chunk {meta.get('chunk_index', i)}]"
        lines.append(f"{label} (relevance={score:.3f})\n{r['text']}")
    return "\n\n".join(lines)


# ── Parse JSON ────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\n?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"[HybridRAG] JSON parse error: {e}\nRaw: {text[:400]}")
        raise ValueError(f"LLM response bukan valid JSON: {e}")


# ── Generate via Gemini ───────────────────────────────────────────────────────
async def _generate_with_gemini(prompt: str, max_tokens: int = 4096) -> str:
    from google import genai as genai_new
    client = genai_new.Client(api_key=settings.GEMINI_API_KEY)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.GEMINI_GENERATE_MODEL,
            contents=prompt,
        ),
    )
    return response.text


# ── Build Global Summary ──────────────────────────────────────────────────────
async def _build_global_summary(
    course_id: Optional[str],
    source_file: Optional[str],
    language: str,
    total_points_hint: int = 0,
    summary_max_tokens: int = 3000,
) -> str:
    from app.services.qdrant_service import QdrantService
    from google import genai as genai_new

    ck = _cache_key(course_id, source_file, total_points_hint)
    if ck in _summary_cache:
        logger.info(f"[HybridRAG] Summary cache hit | key={ck}")
        return _summary_cache[ck]

    qdrant_svc = QdrantService()
    all_data = await qdrant_svc.retrieve_all_context(
        course_id=course_id,
        source_file=source_file,
        max_tokens=summary_max_tokens,
    )
    raw_context = all_data.get("full_context", "")
    total_points = all_data.get("total_points", 0)

    if not raw_context or total_points == 0:
        return ""

    ck = _cache_key(course_id, source_file, total_points)
    if ck in _summary_cache:
        return _summary_cache[ck]

    lang_instruction = "Jawab dalam Bahasa Indonesia." if language == "id" else "Answer in English."
    prompt = f"""Kamu adalah asisten yang meringkas isi materi pembelajaran.

Berikut adalah seluruh konten:
{raw_context}

---
Tugas: Buat RINGKASAN GLOBAL singkat (maksimal 300 kata) yang mencakup:
- Topik-topik utama yang dibahas
- Poin kunci dari setiap topik
- Alur atau struktur materi secara keseluruhan

{lang_instruction}"""

    client = genai_new.Client(api_key=settings.GEMINI_API_KEY)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.GEMINI_GENERATE_MODEL,
            contents=prompt,
        ),
    )
    global_summary = response.text.strip()
    _summary_cache[ck] = global_summary
    return global_summary


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_material_prompt(
    topic: Optional[str],
    language: str,
    global_summary: str,
    relevant_context: str,
    has_relevant_chunks: bool,
) -> str:
    output_lang = "Bahasa Indonesia" if language == "id" else "English"
    topic_label = f'"{topic}"' if topic else "seluruh materi"
    topic_constraint = (
        f'🎯 TOPIK: Buat materi HANYA tentang {topic_label}.\n'
        f'   LARANG membahas topik lain meskipun ada dalam konteks.'
        if topic else
        "Buat materi dari SELURUH konten yang tersedia secara komprehensif."
    )
    source_rule = (
        "⚠️ ATURAN: Gunakan HANYA informasi dari KONTEKS RELEVAN di bawah."
        if has_relevant_chunks else
        "Gunakan KONTEKS GLOBAL sebagai referensi utama."
    )

    return f"""Kamu adalah penulis modul pembelajaran profesional yang membuat materi ajar LENGKAP dan DETAIL.

=== KONTEKS GLOBAL ===
{global_summary or "Tidak tersedia."}

=== KONTEKS RELEVAN (SUMBER UTAMA) ===
{relevant_context}

=== INSTRUKSI WAJIB ===
{topic_constraint}
{source_rule}
- Jawab dalam {output_lang}
- Output HANYA JSON valid, TANPA teks tambahan, TANPA markdown
- WAJIB menghasilkan konten yang PANJANG, DETAIL, dan KAYA informasi
- Setiap section MINIMAL 5-8 kalimat penjabaran penuh
- Jangan buat poin-poin singkat — tulis dalam paragraf yang mengalir
- introduction MINIMAL 4-5 kalimat
- content MINIMAL 8-10 paragraf panjang yang menggabungkan semua section
- key_points BERISI 5-8 poin SINGKAT seperti butir abstrak: maks 10 kata per poin, padat, langsung ke inti — BUKAN kalimat panjang atau paragraf
- summary MINIMAL 4-5 kalimat penutup yang komprehensif
- learning_objectives MINIMAL 5 tujuan pembelajaran yang terukur
- prerequisites MINIMAL 3 prasyarat pengetahuan
- real_world_applications MINIMAL 4 contoh penerapan nyata yang detail
- common_mistakes MINIMAL 4 kesalahan umum beserta penjelasan cara mengatasinya
- further_reading MINIMAL 3 saran pengembangan materi

Tugas: Buatkan modul pembelajaran LENGKAP dan KOMPREHENSIF tentang {topic_label}.
Bayangkan ini adalah bab dalam buku teks — setiap bagian harus tuntas dan mendalam.

Format JSON (SEMUA field WAJIB diisi dengan konten PANJANG dan DETAIL):
{{
  "title": "Judul materi yang spesifik dan deskriptif",

  "introduction": "Paragraf pembuka PANJANG (minimal 4-5 kalimat) yang menjelaskan: (1) apa topik ini, (2) mengapa penting dipelajari, (3) konteks dalam bidang ilmu yang lebih luas, (4) apa yang akan dipelajari dalam materi ini, (5) manfaat nyata setelah mempelajari topik ini.",

  "learning_objectives": [
    "Setelah mempelajari materi ini, peserta didik mampu menjelaskan konsep X secara lengkap",
    "Peserta didik mampu mengidentifikasi komponen-komponen utama dari Y",
    "Peserta didik mampu membedakan antara A dan B berdasarkan karakteristiknya",
    "Peserta didik mampu menerapkan prinsip Z dalam konteks nyata",
    "Peserta didik mampu menganalisis permasalahan terkait topik ini"
  ],

  "prerequisites": [
    "Pemahaman dasar tentang konsep X sebelum mempelajari materi ini",
    "Pengetahuan mengenai Y sebagai fondasi untuk memahami topik",
    "Kemampuan dasar dalam Z yang dibutuhkan untuk mengikuti materi"
  ],

  "sections": [
    {{
      "heading": "1. Pengertian dan Definisi [Nama Topik]",
      "body": "Penjelasan PANJANG dan DETAIL (minimal 5-8 kalimat penuh). Jelaskan definisi lengkap dari berbagai sudut pandang, asal-usul konsep, bagaimana para ahli mendefinisikannya, dan apa yang membuat konsep ini unik. Sertakan detail spesifik dari konteks yang tersedia. Jangan hanya satu atau dua kalimat — kembangkan setiap poin menjadi paragraf yang bermakna dan informatif bagi pembaca.",
      "key_concept": "Konsep inti yang harus dipahami dari section ini"
    }},
    {{
      "heading": "2. Sejarah dan Perkembangan",
      "body": "Penjelasan panjang tentang latar belakang historis, bagaimana konsep ini berkembang dari waktu ke waktu, siapa tokoh-tokoh penting yang berkontribusi, dan bagaimana pemahaman tentang topik ini berevolusi hingga saat ini.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "3. Komponen dan Elemen Utama",
      "body": "Uraian mendalam tentang bagian-bagian penyusun, elemen-elemen kunci, dan karakteristik utama dari topik ini. Jelaskan setiap komponen secara terpisah namun tunjukkan bagaimana mereka saling berkaitan dan membentuk satu kesatuan yang utuh.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "4. Prinsip dan Mekanisme Kerja",
      "body": "Penjelasan rinci tentang bagaimana topik ini bekerja, prinsip-prinsip yang mendasarinya, mekanisme atau proses yang terlibat, dan apa yang terjadi secara sistematis. Gunakan analogi atau perumpamaan jika membantu pemahaman.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "5. Jenis, Klasifikasi, dan Variasi",
      "body": "Penjelasan tentang berbagai jenis, tipe, atau klasifikasi yang ada dalam topik ini. Uraikan perbedaan dan persamaan antara masing-masing jenis, kapan masing-masing digunakan, dan apa kelebihan serta kekurangannya.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "6. Penerapan dan Implementasi Praktis",
      "body": "Bagaimana topik ini diterapkan dalam kehidupan nyata atau dalam praktik profesional. Berikan contoh-contoh konkret, langkah-langkah implementasi, dan skenario penggunaan yang umum dijumpai. Jelaskan juga tantangan yang sering dihadapi dalam penerapannya.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "7. Kelebihan, Keterbatasan, dan Pertimbangan",
      "body": "Analisis mendalam tentang kelebihan dan manfaat dari topik ini, keterbatasan atau kekurangan yang perlu diperhatikan, serta faktor-faktor yang perlu dipertimbangkan dalam penggunaannya. Berikan perspektif yang seimbang dan objektif.",
      "key_concept": "Konsep inti section ini"
    }},
    {{
      "heading": "8. Hubungan dengan Konsep Lain",
      "body": "Jelaskan bagaimana topik ini berkaitan dengan konsep-konsep lain dalam bidang yang sama, bagaimana ia mempengaruhi atau dipengaruhi oleh faktor-faktor lain, serta posisinya dalam kerangka ilmu yang lebih besar.",
      "key_concept": "Konsep inti section ini"
    }}
  ],

  "content": "INI ADALAH FIELD TERPENTING — tulis isi materi LENGKAP dalam satu blok teks panjang yang mengalir seperti bab buku teks. MINIMAL 10-15 paragraf. Gabungkan semua section di atas menjadi narasi yang kohesif dan komprehensif. Mulai dari pengenalan, bahas setiap aspek secara mendalam, berikan contoh-contoh spesifik, jelaskan mekanisme dan prinsip kerja, uraikan berbagai jenis dan klasifikasi, dan tutup dengan implikasi praktis. Setiap paragraf harus berisi minimal 4-6 kalimat yang substantif. Jangan gunakan bullet point — tulis dalam paragraf prosa yang mengalir dan enak dibaca.",

  "key_points": [
    "Kalimat pendek 1 — satu fakta/konsep inti, maks 10 kata",
    "Kalimat pendek 2 — satu fakta/konsep inti, maks 10 kata",
    "Kalimat pendek 3 — satu fakta/konsep inti, maks 10 kata",
    "Kalimat pendek 4 — satu fakta/konsep inti, maks 10 kata",
    "Kalimat pendek 5 — satu fakta/konsep inti, maks 10 kata"
  ],

  "real_world_applications": [
    {{
      "context": "Nama bidang atau konteks penerapan pertama",
      "description": "Penjelasan detail (3-4 kalimat) tentang bagaimana topik ini diterapkan dalam konteks ini, apa manfaatnya, dan contoh konkret yang bisa dijumpai"
    }},
    {{
      "context": "Nama bidang atau konteks penerapan kedua",
      "description": "Penjelasan detail tentang penerapan di bidang ini beserta contoh nyata"
    }},
    {{
      "context": "Nama bidang atau konteks penerapan ketiga",
      "description": "Penjelasan detail tentang penerapan di bidang ini beserta contoh nyata"
    }},
    {{
      "context": "Nama bidang atau konteks penerapan keempat",
      "description": "Penjelasan detail tentang penerapan di bidang ini beserta contoh nyata"
    }}
  ],

  "common_mistakes": [
    {{
      "mistake": "Deskripsi kesalahan umum pertama yang sering dilakukan",
      "explanation": "Penjelasan mengapa ini salah dan bagaimana cara menghindari atau memperbaikinya (2-3 kalimat)"
    }},
    {{
      "mistake": "Deskripsi kesalahan umum kedua",
      "explanation": "Penjelasan dan cara mengatasinya"
    }},
    {{
      "mistake": "Deskripsi kesalahan umum ketiga",
      "explanation": "Penjelasan dan cara mengatasinya"
    }},
    {{
      "mistake": "Deskripsi kesalahan umum keempat",
      "explanation": "Penjelasan dan cara mengatasinya"
    }}
  ],

  "summary": "Paragraf penutup PANJANG (minimal 4-5 kalimat) yang merangkum: (1) inti dari seluruh materi yang telah dibahas, (2) mengapa topik ini penting, (3) apa poin-poin kunci yang harus diingat, (4) bagaimana pengetahuan ini dapat diterapkan, (5) langkah selanjutnya yang bisa dilakukan untuk memperdalam pemahaman.",

  "further_reading": [
    "Topik lanjutan pertama yang disarankan untuk dipelajari berikutnya beserta alasannya",
    "Topik lanjutan kedua yang berkaitan erat dengan materi ini",
    "Konsep yang perlu dipelajari untuk pemahaman lebih mendalam",
    "Aspek praktis atau teknis yang bisa dijelajahi lebih lanjut"
  ]
}}"""


def _build_flashcard_prompt(
    topic: Optional[str],
    count: int,
    language: str,
    global_summary: str,
    relevant_context: str,
    has_relevant_chunks: bool,
    material_context: Optional[str] = None,
) -> str:
    output_lang = "Bahasa Indonesia" if language == "id" else "English"
    topic_label = f'"{topic}"' if topic else "materi"

    extra_context = ""
    if material_context:
        extra_context = f"\n=== MATERI REFERENSI (dari database) ===\n{material_context}\n"

    return f"""Kamu adalah asisten AI yang membuat flashcard pembelajaran.

=== KONTEKS GLOBAL ===
{global_summary or "Tidak tersedia."}
{extra_context}
=== KONTEKS RELEVAN (SUMBER UTAMA) ===
{relevant_context}

=== INSTRUKSI ===
- Buat TEPAT {count} flashcard tentang {topic_label}
- Gunakan HANYA informasi dari konteks di atas
- Jawab dalam {output_lang}
- Output HANYA JSON valid, TANPA teks tambahan

Format JSON:
{{
  "flashcards": [
    {{
      "front": "Kata kunci / konsep singkat (1 baris)",
      "back": "Penjelasan 2-4 kalimat yang bersumber dari konteks"
    }}
  ]
}}"""


def _build_quiz_prompt(
    topic: Optional[str],
    quiz_type: str,
    count: int,
    difficulty: Optional[str],
    language: str,
    global_summary: str,
    relevant_context: str,
    has_relevant_chunks: bool,
    material_context: Optional[str] = None,
) -> str:
    output_lang = "Bahasa Indonesia" if language == "id" else "English"
    topic_label = f'"{topic}"' if topic else "materi"
    diff_map = {"easy": "mudah", "medium": "menengah", "hard": "sulit"}
    diff_label = diff_map.get(difficulty or "medium", "menengah")

    extra_context = ""
    if material_context:
        extra_context = f"\n=== MATERI REFERENSI (dari database) ===\n{material_context}\n"

    header = f"""Kamu adalah asisten AI yang membuat soal latihan pembelajaran.

=== KONTEKS GLOBAL ===
{global_summary or "Tidak tersedia."}
{extra_context}
=== KONTEKS RELEVAN (SUMBER UTAMA) ===
{relevant_context}

=== INSTRUKSI ===
- Buat TEPAT {count} soal tentang {topic_label} dengan tingkat kesulitan {diff_label}
- Gunakan HANYA informasi dari konteks di atas
- Jawab dalam {output_lang}
- Output HANYA JSON valid, TANPA teks tambahan
"""

    if quiz_type == "essay":
        return header + f"""
Tugas: Buat {count} soal ESAI tingkat {diff_label} tentang {topic_label}.

Format JSON:
{{
  "items": [
    {{
      "question": "Pertanyaan esai yang membutuhkan jawaban panjang",
      "sample_answer": "Contoh jawaban ideal (2-5 kalimat) dari konteks",
      "key_points": ["Poin penilaian 1", "Poin penilaian 2", "Poin penilaian 3"],
      "score_weight": 20
    }}
  ]
}}"""

    elif quiz_type == "multiple_choice_single":
        return header + f"""
Tugas: Buat {count} soal PILIHAN GANDA (1 jawaban benar) tingkat {diff_label} tentang {topic_label}.

Format JSON:
{{
  "items": [
    {{
      "question": "Pertanyaan pilihan ganda?",
      "options": [
        {{"label": "A", "text": "Pilihan A"}},
        {{"label": "B", "text": "Pilihan B"}},
        {{"label": "C", "text": "Pilihan C"}},
        {{"label": "D", "text": "Pilihan D"}}
      ],
      "correct_answer": "A",
      "explanation": "Penjelasan mengapa A benar berdasarkan materi"
    }}
  ]
}}"""

    elif quiz_type == "multiple_choice_multiple":
        return header + f"""
Tugas: Buat {count} soal PILIHAN GANDA MULTIPLE (lebih dari 1 jawaban bisa benar) tingkat {diff_label} tentang {topic_label}.
Setiap soal harus memiliki 2–3 jawaban benar dari 5–6 pilihan.

Format JSON:
{{
  "items": [
    {{
      "question": "Manakah pernyataan berikut yang BENAR tentang ...? (pilih semua yang benar)",
      "options": [
        {{"label": "A", "text": "Pilihan A"}},
        {{"label": "B", "text": "Pilihan B"}},
        {{"label": "C", "text": "Pilihan C"}},
        {{"label": "D", "text": "Pilihan D"}},
        {{"label": "E", "text": "Pilihan E"}}
      ],
      "correct_answers": ["A", "C", "E"],
      "explanation": "Penjelasan mengapa A, C, E benar berdasarkan materi"
    }}
  ]
}}"""

    # fallback
    return header


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

async def run_generate_material(request) -> dict:
    """
    Generate materi pembelajaran lengkap dari konten di Qdrant berdasarkan course_id.
    Semua file dalam course tersebut digunakan — tidak perlu filter source_file.
    Output ini yang akan disimpan di backend Laravel (MySQL) sebagai JSON.
    """
    course_id = _normalize(getattr(request, "course_id", None))
    topic = _normalize(getattr(request, "topic", None))
    language = getattr(request, "language", "id")
    scope = _scope_label(course_id, None)  # source_file selalu None untuk material

    query = _build_search_query(topic, "material", course_id, None)
    logger.info(f"[Material] ▶ START | scope={scope} | topic={topic!r} | query={query!r}")

    # Step 1: Similarity Search
    raw_results = await _similarity_search(query, course_id, source_file=None, top_k=20)
    if not raw_results:
        raise RuntimeError(
            f"Tidak ada data di Qdrant untuk course_id='{course_id}'. "
            "Pastikan file sudah di-upload dan diproses."
        )

    # Step 2: Filter
    results, has_relevant = _filter_relevant_chunks(raw_results, topic=topic, max_chunks=TOP_RELEVANT_CHUNKS)
    relevant_context = _format_chunks(results)

    # Step 3: Global Summary (pakai course_id saja)
    global_summary = await _build_global_summary(course_id, None, language, len(raw_results))

    # Step 4: Prompt & Generate — naikkan max_tokens agar output panjang tidak terpotong
    prompt = _build_material_prompt(topic, language, global_summary, relevant_context, has_relevant)
    raw_output = await _generate_with_gemini(prompt, max_tokens=8192)

    # Step 5: Parse
    parsed = _parse_json(raw_output)
    ctx_count = len(results)

    sections   = parsed.get("sections", [])
    key_points = parsed.get("key_points", [])
    logger.info(
        f"[Material] ✅ DONE | scope={scope} | sections={len(sections)} "
        f"| key_points={len(key_points)} | content_len={len(parsed.get('content', ''))}"
    )

    return {
        "title":                   parsed.get("title", ""),
        "introduction":            parsed.get("introduction", ""),
        "learning_objectives":     parsed.get("learning_objectives", []),
        "prerequisites":           parsed.get("prerequisites", []),
        "sections":                sections,
        "content":                 parsed.get("content", ""),
        "key_points":              key_points,
        "real_world_applications": parsed.get("real_world_applications", []),
        "common_mistakes":         parsed.get("common_mistakes", []),
        "summary":                 parsed.get("summary", ""),
        "further_reading":         parsed.get("further_reading", []),
        "context_chunks_used":     ctx_count,
        "context_scope":           scope,
    }


async def run_generate_flashcard(request) -> dict:
    """
    Generate flashcard dari konten Qdrant.
    Opsional: material_context dari Laravel untuk hasil lebih akurat.
    """
    course_id = _normalize(getattr(request, "course_id", None))
    source_file = _normalize(getattr(request, "source_file", None))
    topic = _normalize(getattr(request, "topic", None))
    material_context = _normalize(getattr(request, "material_context", None))
    count = getattr(request, "count", 10)
    language = getattr(request, "language", "id")
    scope = _scope_label(course_id, source_file)

    query = _build_search_query(topic, "flashcard", course_id, source_file)
    logger.info(f"[Flashcard] ▶ START | scope={scope} | topic={topic!r} | count={count}")

    raw_results = await _similarity_search(query, course_id, source_file, top_k=20)
    if not raw_results:
        raise RuntimeError(f"Tidak ada data di Qdrant untuk scope '{scope}'.")

    results, has_relevant = _filter_relevant_chunks(raw_results, topic=topic, max_chunks=TOP_RELEVANT_CHUNKS)
    relevant_context = _format_chunks(results)

    global_summary = ""
    if course_id or source_file:
        global_summary = await _build_global_summary(course_id, source_file, language, len(raw_results))

    prompt = _build_flashcard_prompt(
        topic, count, language, global_summary, relevant_context, has_relevant, material_context
    )
    raw_output = await _generate_with_gemini(prompt, max_tokens=4096)
    parsed = _parse_json(raw_output)

    cards = parsed.get("flashcards", [])
    logger.info(f"[Flashcard] ✅ DONE | scope={scope} | count={len(cards)}")

    return {
        "count": len(cards),
        "flashcards": cards,
        "context_chunks_used": len(results),
        "context_scope": scope,
    }


async def run_generate_quiz(request) -> dict:
    """
    Generate quiz dengan tipe soal dinamis:
      - essay
      - multiple_choice_single
      - multiple_choice_multiple
    """
    course_id = _normalize(getattr(request, "course_id", None))
    source_file = _normalize(getattr(request, "source_file", None))
    topic = _normalize(getattr(request, "topic", None))
    material_context = _normalize(getattr(request, "material_context", None))
    quiz_type = getattr(request, "quiz_type", "multiple_choice_single")
    count = getattr(request, "count", 5)
    difficulty = getattr(request, "difficulty", "medium")
    language = getattr(request, "language", "id")
    scope = _scope_label(course_id, source_file)

    query = _build_search_query(topic, "quiz", course_id, source_file, quiz_type)
    logger.info(f"[Quiz] ▶ START | scope={scope} | type={quiz_type} | topic={topic!r} | count={count}")

    raw_results = await _similarity_search(query, course_id, source_file, top_k=20)
    if not raw_results:
        raise RuntimeError(f"Tidak ada data di Qdrant untuk scope '{scope}'.")

    results, has_relevant = _filter_relevant_chunks(raw_results, topic=topic, max_chunks=TOP_RELEVANT_CHUNKS)
    relevant_context = _format_chunks(results)

    global_summary = ""
    if course_id or source_file:
        global_summary = await _build_global_summary(course_id, source_file, language, len(raw_results))

    prompt = _build_quiz_prompt(
        topic, quiz_type, count, difficulty, language,
        global_summary, relevant_context, has_relevant, material_context
    )
    raw_output = await _generate_with_gemini(prompt, max_tokens=4096)
    parsed = _parse_json(raw_output)

    items = parsed.get("items", [])
    logger.info(f"[Quiz] ✅ DONE | scope={scope} | type={quiz_type} | count={len(items)}")

    return {
        "quiz_type": quiz_type,
        "count": len(items),
        "difficulty": difficulty,
        "items": items,
        "context_chunks_used": len(results),
        "context_scope": scope,
    }


# ── Cache Management ──────────────────────────────────────────────────────────
def clear_summary_cache(
    course_id: Optional[str] = None,
    source_file: Optional[str] = None,
):
    global _summary_cache
    if course_id is None and source_file is None:
        n = len(_summary_cache)
        _summary_cache.clear()
        logger.info(f"[HybridRAG] All cache cleared ({n} entries)")
        return

    keys_to_del = [
        k for k in list(_summary_cache.keys())
        if (course_id and k.startswith(f"{course_id}|"))
        or (source_file and f"|{source_file}|" in k)
    ]
    for k in keys_to_del:
        del _summary_cache[k]
    if keys_to_del:
        logger.info(f"[HybridRAG] Cache cleared: {len(keys_to_del)} entries")