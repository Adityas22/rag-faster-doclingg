"""
Hybrid RAG Workflow — dengan topic-aware relevance filtering

Perbaikan utama:
  - Score gap detection: potong chunk saat ada loncatan score besar (topik berganti)
  - Keyword relevance check: chunk yang tidak mengandung kata dari topic di-penalti
  - Threshold adaptif: minimum score = best_score * 0.75 (relatif, bukan absolut)
  - Jika topic ada, LLM diperintahkan KERAS untuk hanya bicara tentang topic tsb
"""
import asyncio
from datetime import timedelta
from typing import Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
_summary_cache: dict = {}

# Threshold adaptif: chunk harus minimal (best_score * ratio) untuk dipakai
SCORE_RATIO_THRESHOLD = 0.75   # chunk harus >= 75% skor terbaik
# Minimum score absolut — chunk di bawah ini selalu dibuang
MIN_SCORE_ABSOLUTE = 0.40
# Gap: jika skor turun lebih dari ini antara chunk berurutan, potong
SCORE_GAP_CUTOFF = 0.15
# Max chunk yang dipakai untuk generate
TOP_RELEVANT_CHUNKS = 6


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


def _cache_key(course_id: Optional[str], source_file: Optional[str], total_points: int) -> str:
    return f"{course_id or '__all__'}|{source_file or '__all__'}|{total_points}"


# ── Query builder ─────────────────────────────────────────────────────────────
def _build_search_query(
    topic: Optional[str],
    content_type: str,
    course_id: Optional[str],
    source_file: Optional[str],
) -> str:
    type_hints = {
        "flashcard":  "konsep utama, definisi, poin kunci, istilah penting",
        "quiz_mc":    "konsep, fakta, perbandingan, prinsip yang bisa diujikan",
        "quiz_essay": "konsep mendalam, proses, analisis, evaluasi",
        "summary":    "ringkasan, poin penting, kesimpulan, inti materi",
        "material":   "penjelasan lengkap, contoh, penerapan, latihan",
    }
    hint = type_hints.get(content_type, "materi pembelajaran")

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
    results: list[dict],
    topic: Optional[str] = None,
    max_chunks: int = TOP_RELEVANT_CHUNKS,
) -> tuple[list[dict], bool]:
    """
    Filter chunk dengan 3 strategi bertingkat:

    1. THRESHOLD ADAPTIF: chunk harus >= best_score * SCORE_RATIO_THRESHOLD
       DAN >= MIN_SCORE_ABSOLUTE
       → Ini memastikan chunk yang jauh di bawah yang terbaik tidak ikut masuk

    2. SCORE GAP DETECTION: jika skor turun tajam (>= SCORE_GAP_CUTOFF) antara
       chunk berurutan, potong di sana — ini menandai pergantian topik
       → Contoh: [0.82, 0.79, 0.75, 0.55, 0.52] → potong setelah 0.75

    3. KEYWORD CHECK (jika topic ada): chunk yang tidak mengandung
       kata dari topic SAMA SEKALI mendapat penalti — dipindahkan ke belakang
       Ini safety net terakhir agar konten tidak melenceng

    Returns (filtered_chunks, has_high_relevance)
    """
    if not results:
        return [], False

    best_score = results[0].get("score", 0)

    # --- Strategy 1: threshold adaptif ---
    adaptive_threshold = max(best_score * SCORE_RATIO_THRESHOLD, MIN_SCORE_ABSOLUTE)
    after_threshold = [r for r in results if r.get("score", 0) >= adaptive_threshold]

    if not after_threshold:
        # Tidak ada yang lolos threshold → fallback ke top-3 saja
        logger.warning(
            f"[HybridRAG] Threshold terlalu ketat (best={best_score:.3f}, "
            f"threshold={adaptive_threshold:.3f}), fallback top-3"
        )
        after_threshold = results[:3]

    # --- Strategy 2: score gap detection ---
    gap_cut = len(after_threshold)
    for i in range(1, len(after_threshold)):
        prev_score = after_threshold[i - 1].get("score", 0)
        curr_score = after_threshold[i].get("score", 0)
        gap = prev_score - curr_score
        if gap >= SCORE_GAP_CUTOFF:
            gap_cut = i
            logger.info(
                f"[HybridRAG] Score gap detected at index {i}: "
                f"{prev_score:.3f} → {curr_score:.3f} (gap={gap:.3f}) → cutting here"
            )
            break
    after_gap = after_threshold[:gap_cut]

    # --- Strategy 3: keyword check (jika topic ada) ---
    if topic:
        # Ekstrak kata-kata kunci dari topic (minimal 3 karakter)
        topic_words = [w.lower() for w in topic.split() if len(w) >= 3]

        def _chunk_contains_topic(chunk: dict) -> bool:
            text = chunk.get("text", "").lower()
            return any(word in text for word in topic_words)

        on_topic   = [c for c in after_gap if _chunk_contains_topic(c)]
        off_topic  = [c for c in after_gap if not _chunk_contains_topic(c)]

        if on_topic:
            # Ada chunk yang mengandung kata topic → buang yang off-topic
            if off_topic:
                logger.info(
                    f"[HybridRAG] Keyword filter '{topic}': "
                    f"kept {len(on_topic)}, removed {len(off_topic)} off-topic chunks"
                )
            selected = on_topic[:max_chunks]
        else:
            # Tidak ada yang mengandung kata topic — mungkin topic ada di embedding
            # tapi beda kata. Pakai semua hasil gap filter, jangan buang.
            logger.warning(
                f"[HybridRAG] No chunk contains topic keywords {topic_words}, "
                f"using gap-filtered results as-is"
            )
            selected = after_gap[:max_chunks]
    else:
        selected = after_gap[:max_chunks]

    has_relevant = best_score >= MIN_SCORE_ABSOLUTE

    logger.info(
        f"[HybridRAG] Filter result: {len(results)} raw → "
        f"{len(after_threshold)} threshold → "
        f"{len(after_gap)} gap → "
        f"{len(selected)} final | best_score={best_score:.3f} | "
        f"adaptive_threshold={adaptive_threshold:.3f}"
    )

    return selected, has_relevant


# ── STEP: Similarity Search ───────────────────────────────────────────────────
async def _similarity_search(
    query: str,
    course_id: Optional[str],
    source_file: Optional[str],
    top_k: int = 15,
) -> list[dict]:
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant_service import QdrantService

    embed_svc  = EmbeddingService()
    qdrant_svc = QdrantService()

    query_vector = await embed_svc.embed_text(query, task_type="retrieval_query")
    results = await qdrant_svc.search(
        vector=query_vector,
        course_id=course_id,
        source_file=source_file,
        top_k=top_k,
    )
    return results


# ── Global Summary ────────────────────────────────────────────────────────────
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

    raw_context  = all_data.get("full_context", "")
    total_points = all_data.get("total_points", 0)

    logger.info(
        f"[HybridRAG] retrieve_all_context | "
        f"scope={_scope_label(course_id, source_file)} | "
        f"{total_points} points | {all_data.get('total_tokens', 0):,} tokens"
    )

    if not raw_context or total_points == 0:
        return ""

    ck = _cache_key(course_id, source_file, total_points)
    if ck in _summary_cache:
        logger.info(f"[HybridRAG] Summary cache hit (updated key) | key={ck}")
        return _summary_cache[ck]

    logger.info(f"[HybridRAG] Building global summary | {total_points} points...")
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
    loop   = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.GEMINI_GENERATE_MODEL,
            contents=prompt,
        ),
    )
    global_summary = response.text.strip()
    _summary_cache[ck] = global_summary
    logger.info(f"[HybridRAG] Summary built & cached | {len(global_summary)} chars")
    return global_summary


# ── Format Chunks ─────────────────────────────────────────────────────────────
def _format_chunks(results: list[dict]) -> str:
    if not results:
        return "Tidak ada chunk relevan ditemukan."
    lines = []
    for i, r in enumerate(results, 1):
        meta  = r.get("metadata", {})
        score = r.get("score", 0)
        ts_s  = meta.get("timestamp_start")
        ts_e  = meta.get("timestamp_end")
        if ts_s is not None and ts_e is not None:
            ts_label = (
                f"[{str(timedelta(seconds=int(ts_s)))} - "
                f"{str(timedelta(seconds=int(ts_e)))}]"
            )
        else:
            ts_label = f"[Chunk {meta.get('chunk_index', i)}]"
        lines.append(f"{ts_label} (relevance={score:.3f})\n{r['text']}")
    return "\n\n".join(lines)


# ── Build Final Prompt ────────────────────────────────────────────────────────
def _build_prompt(
    content_type: str,
    topic: Optional[str],
    count: int,
    difficulty: Optional[str],
    language: str,
    global_summary: str,
    relevant_context: str,
    has_relevant_chunks: bool,
) -> str:
    lang_map    = {"id": "Bahasa Indonesia", "en": "English"}
    output_lang = lang_map.get(language, language)
    diff_map    = {"easy": "mudah", "medium": "menengah", "hard": "sulit"}
    diff_label  = diff_map.get(difficulty or "", "menengah")

    # Instruksi topic yang sangat eksplisit
    if topic:
        topic_constraint = (
            f"🎯 TOPIK: Kamu HANYA boleh membuat konten tentang \"{topic}\".\n"
            f"   LARANG KERAS membahas topik lain meskipun ada dalam konteks.\n"
            f"   Jika konteks tidak cukup tentang \"{topic}\", katakan materi tidak tersedia."
        )
    else:
        topic_constraint = "Buat konten dari keseluruhan materi yang tersedia."

    if has_relevant_chunks:
        source_rule = (
            "⚠️ ATURAN KERAS: Gunakan HANYA informasi dari KONTEKS RELEVAN di bawah.\n"
            "   DILARANG menggunakan pengetahuan umum LLM di luar konteks yang diberikan."
        )
    else:
        source_rule = "Gunakan KONTEKS GLOBAL sebagai referensi — tetap dalam scope materi."

    context_block = f"""=== KONTEKS GLOBAL (orientasi topik yang ada dalam materi) ===
{global_summary if global_summary else "Tidak tersedia."}

=== KONTEKS RELEVAN — SUMBER UTAMA ===
{relevant_context}"""

    petunjuk = f"""{topic_constraint}
{source_rule}
- Jawab dalam {output_lang}
- Output HANYA JSON valid, TANPA teks tambahan, TANPA markdown"""

    if content_type == "flashcard":
        instruksi = f"""Tugas: Buatkan TEPAT {count} flashcard tentang "{topic or 'materi'}" dari KONTEKS RELEVAN.

- "front": kata kunci / konsep tentang {topic or 'materi'} (bersumber dari konteks)
- "back": penjelasan 3-5 kalimat yang HARUS bersumber langsung dari konteks

Format JSON:
{{
  "flashcards": [
    {{
      "front": "Kata kunci dari materi",
      "back": "Penjelasan 3-5 kalimat dari konteks."
    }}
  ]
}}"""

    elif content_type == "quiz_mc":
        instruksi = f"""Tugas: Buatkan TEPAT {count} soal pilihan ganda tingkat {diff_label} tentang "{topic or 'materi'}".
Soal HARUS dari konten spesifik dalam KONTEKS RELEVAN.

Format JSON:
{{
  "items": [
    {{
      "question": "Pertanyaan dari materi...",
      "options": [
        {{"label": "A", "text": "..."}},
        {{"label": "B", "text": "..."}},
        {{"label": "C", "text": "..."}},
        {{"label": "D", "text": "..."}}
      ],
      "correct_answer": "A",
      "explanation": "Penjelasan dari materi..."
    }}
  ]
}}"""

    elif content_type == "quiz_essay":
        instruksi = f"""Tugas: Buatkan TEPAT {count} soal essay tingkat {diff_label} tentang "{topic or 'materi'}".
Soal dan jawaban HARUS dari konten spesifik dalam KONTEKS RELEVAN.

Format JSON:
{{
  "items": [
    {{
      "question": "Pertanyaan dari materi...",
      "sample_answer": "Jawaban dari materi...",
      "key_points": ["poin dari materi 1", "poin dari materi 2"]
    }}
  ]
}}"""

    elif content_type == "summary":
        instruksi = f"""Tugas: Buat ringkasan tentang "{topic or 'materi'}" dari KONTEKS RELEVAN.
Hasilkan TEPAT {count} poin penting dari konteks.

Format JSON:
{{
  "summary": "Paragraf ringkasan dari materi...",
  "key_points": ["Poin 1 dari materi", "Poin 2 dari materi"]
}}"""

    elif content_type == "material":
        instruksi = f"""Tugas: Buat materi pelajaran tentang "{topic or 'materi'}" dari KONTEKS RELEVAN.
Hasilkan TEPAT {count} contoh dan {count} soal latihan dari materi.

Format JSON:
{{
  "title": "Judul berdasarkan konten",
  "introduction": "Pembuka dari materi...",
  "content": "Isi materi dari konten...",
  "examples": ["Contoh 1 dari materi", "Contoh 2 dari materi"],
  "exercises": ["Soal 1 dari materi", "Soal 2 dari materi"]
}}"""

    else:
        instruksi = f"Tugas: Buat konten '{content_type}' tentang '{topic or 'materi'}' dari konteks."

    return f"""Kamu adalah asisten AI yang membuat konten pembelajaran EKSKLUSIF dari materi yang diberikan.

{context_block}

=== INSTRUKSI ===
{instruksi}

{petunjuk}"""


# ── Generate via Gemini ───────────────────────────────────────────────────────
async def _generate_with_gemini(prompt: str) -> str:
    from google import genai as genai_new
    client = genai_new.Client(api_key=settings.GEMINI_API_KEY)
    loop   = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.GEMINI_GENERATE_MODEL,
            contents=prompt,
        ),
    )
    return response.text


# ── Parse JSON ────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    import json, re
    text = re.sub(r"```(?:json)?\n?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"[HybridRAG] JSON parse error: {e}\nRaw: {text[:300]}")
        raise ValueError(f"LLM response bukan valid JSON: {e}")


# ── Public Entry Point ────────────────────────────────────────────────────────
async def run_generate_workflow(request) -> dict:
    """
    Hybrid RAG dengan topic-aware relevance filtering.

    Perbaikan:
    1. Query = topic + content_type hint
    2. Filter bertingkat: threshold adaptif → gap detection → keyword check
    3. Keyword check: chunk yang tidak mengandung kata topic dibuang
    4. Prompt: topic constraint eksplisit di instruksi LLM
    """
    course_id    = _normalize(getattr(request, "course_id", None))
    source_file  = _normalize(getattr(request, "source_file", None))
    topic        = _normalize(getattr(request, "topic", None))
    content_type = request.content_type
    count        = getattr(request, "count", 5)
    difficulty   = getattr(request, "difficulty", None)
    language     = getattr(request, "language", "id")

    scope = _scope_label(course_id, source_file)
    query = _build_search_query(topic, content_type, course_id, source_file)

    logger.info(
        f"[HybridRAG] ▶ START | type={content_type} | scope={scope} | "
        f"topic={topic!r} | query={query!r} | count={count}"
    )

    # ── STEP 1: Similarity Search top_k=15 ────────────────────────────────────
    logger.info(f"[HybridRAG] Step 1: Similarity search | top_k=15")
    raw_results = await _similarity_search(
        query=query,
        course_id=course_id,
        source_file=source_file,
        top_k=15,
    )

    if not raw_results:
        raise RuntimeError(
            f"Tidak ada data di Qdrant untuk scope '{scope}'. "
            f"Pastikan file sudah di-upload dan diproses terlebih dahulu."
        )

    scores_preview = [round(r.get("score", 0), 3) for r in raw_results]
    logger.info(f"[HybridRAG] Step 1 done | scores={scores_preview}")

    # ── STEP 2: Topic-aware filtering ─────────────────────────────────────────
    results, has_relevant = _filter_relevant_chunks(
        raw_results,
        topic=topic,
        max_chunks=TOP_RELEVANT_CHUNKS,
    )
    relevant_context = _format_chunks(results)

    # ── STEP 3: Global Summary ────────────────────────────────────────────────
    # Hanya diambil jika course_id/source_file ada (scope terbatas)
    # Untuk global scope, skip summary agar tidak menambah noise
    global_summary = ""
    if course_id or source_file:
        logger.info(f"[HybridRAG] Step 3: Global summary | scope={scope}")
        global_summary = await _build_global_summary(
            course_id=course_id,
            source_file=source_file,
            language=language,
            total_points_hint=len(raw_results),
            summary_max_tokens=3000,
        )
        logger.info(f"[HybridRAG] Step 3 done | {len(global_summary)} chars")
    else:
        logger.info(f"[HybridRAG] Step 3: Skipped (global scope, topic filtering sufficient)")

    # ── STEP 4: Build Prompt ──────────────────────────────────────────────────
    prompt = _build_prompt(
        content_type=content_type,
        topic=topic,
        count=count,
        difficulty=difficulty,
        language=language,
        global_summary=global_summary,
        relevant_context=relevant_context,
        has_relevant_chunks=has_relevant,
    )

    # ── STEP 5: Generate ──────────────────────────────────────────────────────
    logger.info(f"[HybridRAG] Step 5: Generating | model={settings.GEMINI_GENERATE_MODEL}")
    raw_output = await _generate_with_gemini(prompt)
    logger.info(f"[HybridRAG] Step 5 done | {len(raw_output)} chars")

    # ── STEP 6: Parse JSON ────────────────────────────────────────────────────
    parsed    = _parse_json(raw_output)
    ctx_count = len(results)

    logger.info(
        f"[HybridRAG] ✅ DONE | type={content_type} | scope={scope} | "
        f"chunks_used={ctx_count} | topic={topic!r}"
    )

    # ── Format response ───────────────────────────────────────────────────────
    if content_type == "flashcard":
        cards = parsed.get("flashcards", [])
        return {
            "count": len(cards),
            "flashcards": cards,
            "context_chunks_used": ctx_count,
            "context_scope": scope,
        }

    elif content_type in ("quiz_mc", "quiz_essay"):
        items = parsed.get("items", [])
        return {
            "content_type": content_type,
            "count": len(items),
            "difficulty": difficulty,
            "items": items,
            "context_chunks_used": ctx_count,
            "context_scope": scope,
        }

    elif content_type == "summary":
        key_points = parsed.get("key_points", [])
        return {
            "summary": parsed.get("summary", ""),
            "key_points": key_points,
            "key_points_count": len(key_points),
            "context_chunks_used": ctx_count,
            "context_scope": scope,
        }

    elif content_type == "material":
        examples  = parsed.get("examples", [])
        exercises = parsed.get("exercises", [])
        return {
            "title":           parsed.get("title", ""),
            "introduction":    parsed.get("introduction", ""),
            "content":         parsed.get("content", ""),
            "examples":        examples,
            "exercises":       exercises,
            "examples_count":  len(examples),
            "exercises_count": len(exercises),
            "context_chunks_used": ctx_count,
            "context_scope":   scope,
        }

    return {
        "result": parsed,
        "context_chunks_used": ctx_count,
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
        logger.info(
            f"[HybridRAG] Cache cleared: {len(keys_to_del)} entries | "
            f"course_id={course_id} source_file={source_file}"
        )
