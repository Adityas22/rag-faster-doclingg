"""
DoclingService — PDF extraction dengan strategi hybrid:

  Strategi 1 (UTAMA): Docling
    - Parsing teks terstruktur, TABEL (Markdown), dan teks dalam GAMBAR
    - Dijalankan via subprocess terpisah agar torch DLL error tidak crash server
    - Hasil di-cache ke JSON → request berikutnya langsung dari cache (cepat)

  Strategi 2 (FALLBACK): PyMuPDF (fitz)
    - Dipakai jika Docling gagal (timeout, DLL error, dll)
    - Mendukung teks + tabel via fitz.find_tables()
    - Zero torch/CUDA dependency

Requirements:
  pip install PyMuPDF langchain-text-splitters docling
"""

import os
import re
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

from app.utils.logger import get_logger

logger = get_logger(__name__)

DOCLING_TIMEOUT = 300  # 5 menit timeout untuk parsing Docling


# ==============================================================================
# DOCLING WORKER SCRIPT — dijalankan sebagai subprocess terpisah
# ==============================================================================

DOCLING_WORKER_SCRIPT = '''
import sys
import json

def run(pdf_path, output_path, do_ocr):
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.generate_page_images = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result   = converter.convert(pdf_path)
        doc_json = result.document.export_to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False)

        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    pdf_path    = sys.argv[1]
    output_path = sys.argv[2]
    do_ocr      = sys.argv[3].lower() == "true"
    run(pdf_path, output_path, do_ocr)
'''


def _run_docling_subprocess(pdf_path: str, output_json: str, do_ocr: bool = False) -> bool:
    """
    Jalankan Docling di subprocess terpisah.
    Return True jika berhasil, False jika gagal/timeout.
    """
    # Tulis worker script ke file temp
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(DOCLING_WORKER_SCRIPT)
        worker_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, worker_path, pdf_path, output_json, str(do_ocr)],
            timeout=DOCLING_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and os.path.exists(output_json):
            logger.info("[Docling] ✅ Subprocess selesai")
            return True
        else:
            logger.warning(f"[Docling] ❌ Subprocess gagal: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"[Docling] ⏱ Timeout setelah {DOCLING_TIMEOUT}s")
        return False
    except Exception as e:
        logger.warning(f"[Docling] ❌ Subprocess error: {e}")
        return False
    finally:
        try:
            os.unlink(worker_path)
        except Exception:
            pass


# ==============================================================================
# DOCLING: Parse tabel → Markdown
# ==============================================================================

def _docling_table_to_markdown(table_node: dict) -> str:
    cells = table_node.get("data", {}).get("table_cells", [])
    if not cells:
        return ""

    max_row, max_col = 0, 0
    for cell in cells:
        max_row = max(max_row, cell.get("end_row_offset_idx", 1))
        max_col = max(max_col, cell.get("end_col_offset_idx", 1))

    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    for cell in cells:
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        text = cell.get("text", "").strip().replace("\n", " ")
        if r < max_row and c < max_col and not grid[r][c]:
            grid[r][c] = text

    md_lines = []
    for i, row in enumerate(grid):
        md_lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * max_col) + "|")

    return "\n".join(md_lines)


# ==============================================================================
# DOCLING: Build sections dari doc_json
# ==============================================================================

def _build_sections_docling(doc_json: dict) -> List[dict]:
    """
    Traverse doc_json Docling → list sections dengan teks, tabel, dan gambar.
    Identik dengan implementasi di notebook PDF_Parser.ipynb.
    """
    texts_lookup    = {item["self_ref"]: item for item in doc_json.get("texts", [])}
    groups_lookup   = {item["self_ref"]: item for item in doc_json.get("groups", [])}
    tables_lookup   = {item["self_ref"]: item for item in doc_json.get("tables", [])}
    pictures_lookup = {item["self_ref"]: item for item in doc_json.get("pictures", [])}

    sections: List[dict] = []
    current_section: Optional[dict] = None
    buffer_text: List[str] = []

    SKIP_LABELS    = {"page_header", "footnote"}
    CONTENT_LABELS = {"text", "paragraph", "list_item", "formula", "title", "caption"}

    def clean(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip() if text else ""

    def sentence_end(text: str) -> bool:
        return text.strip().endswith(('.', '!', '?', ':', '"', '\u201d'))

    def flush(target):
        nonlocal buffer_text
        if buffer_text:
            target["content"].append(" ".join(buffer_text))
            buffer_text = []

    def traverse(node, cur):
        nonlocal sections, buffer_text

        if "$ref" in node:
            ref = node["$ref"]

            if ref in texts_lookup:
                item  = texts_lookup[ref]
                label = item.get("label")
                page_no = item["prov"][0].get("page_no") if item.get("prov") else None

                if label == "formula":
                    raw = item.get("orig", "").strip()
                elif label == "list_item":
                    raw = item.get("orig", "") or item.get("text", "")
                else:
                    raw = item.get("text", "").strip()

                if not raw or label in SKIP_LABELS:
                    return cur

                content = clean(raw)

                if label == "section_header":
                    if cur:
                        flush(cur)
                        sections.append(cur)
                    cur = {"title": content, "content": [], "page_start": page_no}
                    buffer_text = []

                elif label in CONTENT_LABELS:
                    if cur is None:
                        cur = {"title": "PREFACE", "content": [], "page_start": page_no}
                    if label == "formula":
                        flush(cur)
                        cur["content"].append(f"\n[FORMULA]: {content}\n")
                    elif label == "list_item":
                        flush(cur)
                        cur["content"].append(content)
                    else:
                        buffer_text.append(content)
                        if sentence_end(content):
                            flush(cur)

            elif ref in groups_lookup:
                for child in groups_lookup[ref].get("children", []):
                    cur = traverse(child, cur)
                return cur

            elif ref in tables_lookup:
                if cur:
                    flush(cur)
                md = _docling_table_to_markdown(tables_lookup[ref])
                if md:
                    if cur is None:
                        cur = {"title": "PREFACE", "content": [], "page_start": 1}
                    cur["content"].append(f"\n{md}\n")
                return cur

            elif ref in pictures_lookup:
                pic = pictures_lookup[ref]
                caption = ""
                for cr in pic.get("captions", []):
                    cap_ref = cr.get("$ref", "")
                    if cap_ref in texts_lookup:
                        caption = clean(texts_lookup[cap_ref].get("text", ""))
                        break
                if caption:
                    if cur is None:
                        cur = {"title": "PREFACE", "content": [], "page_start": 1}
                    flush(cur)
                    cur["content"].append(f"\n[GAMBAR]: {caption}\n")
                return cur

        for child in node.get("children", []):
            cur = traverse(child, cur)
        return cur

    current_section = traverse(doc_json["body"], current_section)
    if current_section:
        flush(current_section)
        sections.append(current_section)

    # Post-processing
    final, pending = [], []
    for s in sections:
        tu = s["title"].upper()
        if any(w in tu for w in ["REFERENCE", "BIBLIOGRAPHY", "DAFTAR PUSTAKA"]):
            continue
        if not s["content"]:
            pending.append(s["title"])
            continue
        if pending:
            s["title"] = " > ".join(pending) + " > " + s["title"]
            pending = []
        final.append(s)

    return final


# ==============================================================================
# PYMUPDF FALLBACK: Tabel → Markdown
# ==============================================================================

def _pymupdf_table_to_markdown(table) -> str:
    try:
        rows = table.extract()
        if not rows:
            return ""
        md_lines = []
        for i, row in enumerate(rows):
            cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_lines.append("|" + "|".join(["---"] * len(cells)) + "|")
        return "\n".join(md_lines)
    except Exception:
        return ""


# ==============================================================================
# PYMUPDF FALLBACK: Extract blocks + tabel
# ==============================================================================

def _extract_blocks_pymupdf(pdf_path: str):
    import fitz
    doc    = fitz.open(pdf_path)
    blocks = []
    tables = {}

    for page_no, page in enumerate(doc, start=1):
        try:
            pt = page.find_tables()
            if pt and pt.tables:
                tables[page_no] = [_pymupdf_table_to_markdown(t) for t in pt.tables]
                tables[page_no] = [m for m in tables[page_no] if m]
        except Exception:
            pass

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    fn      = span.get("font", "")
                    flags   = span.get("flags", 0)
                    is_bold = bool(flags & 16) or "bold" in fn.lower()
                    blocks.append({
                        "text":      text,
                        "page_no":   page_no,
                        "font_size": round(span.get("size", 12), 1),
                        "is_bold":   is_bold,
                    })
    doc.close()
    return blocks, tables


def _detect_heading_threshold(blocks: List[dict]) -> float:
    if not blocks:
        return 14.0
    sizes    = sorted(b["font_size"] for b in blocks)
    max_size = sizes[-1]
    if len(blocks) <= 20:
        return max_size + 1.0  # PDF pendek: matikan heading agar semua jadi konten
    idx      = min(int(len(sizes) * 0.85), len(sizes) - 1)
    p85      = sizes[idx]
    median   = sizes[len(sizes) // 2]
    thresh   = max(p85, median + 1.0, 12.0)
    return thresh if thresh < max_size else max_size + 1.0


def _build_sections_pymupdf(blocks: List[dict], tables: dict) -> List[dict]:
    if not blocks:
        return []

    threshold         = _detect_heading_threshold(blocks)
    injected_pages    = set()
    sections          = []
    current_section   = None
    buffer            = []

    def is_heading(b):
        t = b["text"].strip()
        if b["font_size"] < threshold:   return False
        if len(t) < 3 or len(t) > 150:  return False
        if re.match(r'^[\d\s\.\-\•\*\(\)]+$', t): return False
        if t.endswith(('.', '!', '?', ',', ';')): return False
        return True

    def clean(t):
        return re.sub(r'\s+', ' ', t).strip()

    def flush():
        nonlocal buffer
        if buffer and current_section is not None:
            combined = " ".join(buffer).strip()
            if combined:
                current_section["content"].append(combined)
            buffer = []

    def inject_tables(page_no):
        if page_no in tables and page_no not in injected_pages:
            injected_pages.add(page_no)
            if current_section is not None:
                for md in tables[page_no]:
                    current_section["content"].append(f"\n{md}\n")

    for block in blocks:
        text    = clean(block["text"])
        page_no = block["page_no"]
        if not text:
            continue
        inject_tables(page_no)

        if is_heading(block):
            flush()
            if current_section:
                sections.append(current_section)
            current_section = {"title": text, "content": [], "page_start": page_no}
            buffer = []
        else:
            if current_section is None:
                current_section = {"title": "PREFACE", "content": [], "page_start": page_no}
            buffer.append(text)
            if text.endswith(('.', '!', '?', ':', '"', '\u201d')):
                flush()

    flush()
    if current_section:
        sections.append(current_section)

    # Post-processing
    final, pending = [], []
    for s in sections:
        tu = s["title"].upper()
        if any(w in tu for w in ["REFERENCE", "BIBLIOGRAPHY", "DAFTAR PUSTAKA"]):
            continue
        if not s["content"]:
            pending.append(s["title"])
            continue
        if pending:
            s["title"] = " > ".join(pending) + " > " + s["title"]
            pending = []
        final.append(s)

    # Fallback: kumpulkan semua teks + tabel langsung dari blocks
    if not final:
        all_text = " ".join(clean(b["text"]) for b in blocks if b["text"].strip())
        tbl_content = [md for pn in sorted(tables) for md in tables[pn]]
        content = ([all_text] if all_text else []) + [f"\n{m}\n" for m in tbl_content]
        if content:
            final = [{"title": "DOCUMENT", "content": content, "page_start": 1}]

    return final


# ==============================================================================
# Semantic Chunking (shared)
# ==============================================================================

def create_semantic_chunks(
    sections: List[dict],
    source_filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[dict]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise RuntimeError("Jalankan: pip install langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    for sec in sections:
        section_text = "\n\n".join(sec["content"])
        for i, chunk_content in enumerate(splitter.split_text(section_text)):
            all_chunks.append({
                "content": f"SECTION: {sec['title']}\n---\n{chunk_content}",
                "metadata": {
                    "source":      source_filename,
                    "section":     sec["title"],
                    "page":        sec.get("page_start"),
                    "chunk_id":    i,
                    "total_words": len(chunk_content.split()),
                },
            })
    return all_chunks


# ==============================================================================
# DoclingService — Main Class
# ==============================================================================

class DoclingService:
    """
    Hybrid PDF extraction:
    - Utama  : Docling (via subprocess) → teks + tabel + teks dalam gambar
    - Fallback: PyMuPDF → teks + tabel (jika Docling gagal/DLL error)
    Nama class dipertahankan agar tidak perlu ubah import di project.
    """

    def is_available(self) -> bool:
        try:
            import fitz  # noqa
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa
            return True
        except ImportError as e:
            logger.warning(f"[PDF] Dependency missing: {e}")
            return False

    def _get_cache_path(self, pdf_path: str) -> str:
        p = Path(pdf_path)
        return str(p.parent / f"{p.stem}_docling_cache.json")

    def extract_pdf(
        self,
        pdf_path: str,
        doc_title: Optional[str] = None,
        do_ocr: bool = False,
    ) -> dict:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not self.is_available():
            raise RuntimeError(
                "PyMuPDF atau langchain-text-splitters tidak terinstall.\n"
                "Jalankan: pip install PyMuPDF langchain-text-splitters"
            )

        pdf_name   = Path(pdf_path).name
        cache_path = self._get_cache_path(pdf_path)
        logger.info(f"[PDF] Extracting: {pdf_name}")

        # ── Coba Docling (via subprocess atau cache) ─────────────────────────
        doc_json  = None
        used_mode = "pymupdf"

        # Load dari cache jika ada
        if os.path.exists(cache_path):
            logger.info(f"[Docling] ✅ Load dari cache: {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    doc_json = json.load(f)
                used_mode = "docling_cache"
            except Exception as e:
                logger.warning(f"[Docling] Cache rusak: {e}")
                doc_json = None

        # Jalankan Docling subprocess jika belum ada cache
        if doc_json is None:
            logger.info("[Docling] Mencoba Docling via subprocess...")
            success = _run_docling_subprocess(pdf_path, cache_path, do_ocr)
            if success:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        doc_json = json.load(f)
                    used_mode = "docling"
                    logger.info("[Docling] ✅ Docling berhasil")
                except Exception as e:
                    logger.warning(f"[Docling] Gagal baca hasil: {e}")
                    doc_json = None

        # ── Build sections ───────────────────────────────────────────────────
        import fitz
        doc        = fitz.open(pdf_path)
        page_count = doc.page_count
        full_text  = "\n".join(page.get_text() for page in doc)
        doc.close()

        if doc_json is not None:
            # Pakai hasil Docling
            sections = _build_sections_docling(doc_json)
            logger.info(f"[Docling] build_sections: {len(sections)} sections")
        else:
            # Fallback PyMuPDF
            logger.warning("[PDF] Docling gagal → fallback PyMuPDF")
            used_mode = "pymupdf"
            blocks, tables = _extract_blocks_pymupdf(pdf_path)
            sections       = _build_sections_pymupdf(blocks, tables)
            logger.info(f"[PyMuPDF] build_sections: {len(sections)} sections")

        # Fallback akhir: gunakan full_text mentah
        if not sections and full_text.strip():
            logger.warning("[PDF] Semua metode gagal → fallback full_text mentah")
            sections = [{"title": "DOCUMENT", "content": [full_text.strip()], "page_start": 1}]

        if not sections:
            raise ValueError(
                "Tidak ada teks yang dapat diekstrak. "
                "Pastikan PDF berisi teks (bukan scan/image only)."
            )

        chunks = create_semantic_chunks(
            sections=sections,
            source_filename=pdf_name,
            chunk_size=1000,
            chunk_overlap=150,
        )

        logger.info(
            f"[PDF] ✅ Done [{used_mode}]: {page_count} pages | "
            f"{len(full_text)} chars | "
            f"{len(sections)} sections | "
            f"{len(chunks)} chunks"
        )

        return {
            "source_file":    pdf_name,
            "doc_title":      doc_title or Path(pdf_path).stem,
            "total_pages":    page_count,
            "total_chars":    len(full_text),
            "full_text":      full_text,
            "total_sections": len(sections),
            "sections":       sections,
            "chunks":         chunks,
            "extraction_mode": used_mode,  # info debug: "docling" / "docling_cache" / "pymupdf"
        }