"""
DoclingService — Microservices Edition.
Docling jalan LANGSUNG (bukan subprocess) karena container Linux terisolasi.
do_ocr=True dan do_table_structure=True aktif penuh.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, List

from app.utils.logger import get_logger

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, PdfFormatOption
from docling.document_converter import DocumentConverter

logger = get_logger(__name__)

_OCR_ENABLED = os.environ.get("DOCLING_OCR_ENABLED", "true").lower() == "true"
_TABLE_STRUCTURE = os.environ.get("DOCLING_TABLE_STRUCTURE", "true").lower() == "true"


def _run_docling(pdf_path: str, output_json: str) -> bool:
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = _OCR_ENABLED
        pipeline_options.do_table_structure = _TABLE_STRUCTURE
        pipeline_options.generate_page_images = False

        logger.info(f"[Docling] OCR={_OCR_ENABLED}, TableStructure={_TABLE_STRUCTURE}")

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(pdf_path)
        doc_json = result.document.export_to_dict()

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False)

        logger.info("[Docling] ✅ Selesai")
        return True

    except Exception as e:
        logger.warning(f"[Docling] ❌ Error: {e}")
        return False


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


def _build_sections_docling(doc_json: dict) -> List[dict]:
    texts_lookup    = {item["self_ref"]: item for item in doc_json.get("texts", [])}
    groups_lookup   = {item["self_ref"]: item for item in doc_json.get("groups", [])}
    tables_lookup   = {item["self_ref"]: item for item in doc_json.get("tables", [])}
    pictures_lookup = {item["self_ref"]: item for item in doc_json.get("pictures", [])}

    sections: List[dict] = []
    current_section: Optional[dict] = None
    buffer_text: List[str] = []

    SKIP_LABELS    = {"page_header", "footnote"}
    CONTENT_LABELS = {"text", "paragraph", "list_item", "formula", "title", "caption"}

    def clean(text):
        return re.sub(r'\s+', ' ', text).strip() if text else ""

    def sentence_end(text):
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
                from docling.datamodel.document import Table
                table_obj = Table.model_validate(tables_lookup[ref])
                md = table_obj.export_to_markdown() # Ini akan menghasilkan tabel | yang rapi
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


def _extract_blocks_pymupdf(pdf_path: str):
    import fitz
    doc = fitz.open(pdf_path)
    blocks, tables = [], {}
    for page_no, page in enumerate(doc, start=1):
        try:
            pt = page.find_tables()
            if pt and pt.tables:
                tables[page_no] = [m for m in [_pymupdf_table_to_markdown(t) for t in pt.tables] if m]
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
                    fn = span.get("font", "")
                    flags = span.get("flags", 0)
                    blocks.append({
                        "text": text,
                        "page_no": page_no,
                        "font_size": round(span.get("size", 12), 1),
                        "is_bold": bool(flags & 16) or "bold" in fn.lower(),
                    })
    doc.close()
    return blocks, tables


def _build_sections_pymupdf(blocks: List[dict], tables: dict) -> List[dict]:
    if not blocks:
        return []
    sizes = sorted(b["font_size"] for b in blocks)
    max_size = sizes[-1]
    if len(blocks) <= 20:
        threshold = max_size + 1.0
    else:
        p85 = sizes[min(int(len(sizes) * 0.85), len(sizes) - 1)]
        median = sizes[len(sizes) // 2]
        thresh = max(p85, median + 1.0, 12.0)
        threshold = thresh if thresh < max_size else max_size + 1.0

    injected_pages, sections = set(), []
    current_section, buffer = None, []

    def is_heading(b):
        t = b["text"].strip()
        if b["font_size"] < threshold: return False
        if len(t) < 3 or len(t) > 150: return False
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
        text = clean(block["text"])
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

    if not final:
        all_text = " ".join(clean(b["text"]) for b in blocks if b["text"].strip())
        tbl_content = [md for pn in sorted(tables) for md in tables[pn]]
        content = ([all_text] if all_text else []) + [f"\n{m}\n" for m in tbl_content]
        if content:
            final = [{"title": "DOCUMENT", "content": content, "page_start": 1}]

    return final


def create_semantic_chunks(sections, source_filename, chunk_size=1000, chunk_overlap=150):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
                    "source": source_filename,
                    "section": sec["title"],
                    "page": sec.get("page_start"),
                    "chunk_id": i,
                    "total_words": len(chunk_content.split()),
                },
            })
    return all_chunks


class DoclingService:
    def __init__(self):
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        )
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        # ↓ WAJIB True agar OCR pada gambar/scan bisa jalan
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        
        # Gunakan EasyOCR agar teks dalam gambar ikut diambil
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["id", "en"],  # sesuaikan bahasa dokumen
            use_gpu=False,      # ganti True jika ada GPU
        )
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_pdf(self, pdf_path: str, doc_title: str = None):
        logger.info(f"[DoclingService] Extracting: {pdf_path}")

        conv_result = self.doc_converter.convert(pdf_path)

        # export_to_markdown() → tabel otomatis jadi format | kolom | baris |
        full_text_markdown = conv_result.document.export_to_markdown()

        total_pages = len(conv_result.pages)

        # Kumpulkan markdown tabel secara terpisah (untuk logging/debug)
        tables_markdown = []
        for table in conv_result.document.tables:
            md = table.export_to_markdown()
            if md:
                tables_markdown.append(md)

        logger.info(
            f"[DoclingService] ✅ {total_pages} pages | "
            f"{len(full_text_markdown)} chars | "
            f"{len(tables_markdown)} tables"
        )

        return {
            "source_file": os.path.basename(pdf_path),   # ← basename saja, bukan full path
            "doc_title": doc_title if doc_title else os.path.basename(pdf_path),
            "full_text": full_text_markdown,
            "total_pages": total_pages,
            "total_chars": len(full_text_markdown),
            "tables_markdown": tables_markdown,           # ← KEY INI yang dipanggil worker
            "tables_count": len(tables_markdown),
        }