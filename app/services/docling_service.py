"""
DoclingService — PDF text extraction using Docling.

WINDOWS DLL CONFLICT FIX:
Docling menggunakan PyTorch (torch). Faster Whisper juga menggunakan PyTorch
tapi via CTranslate2. Di Windows, keduanya bisa bentrok di DLL (c10.dll)
jika keduanya di-load dalam proses yang sama secara bersamaan.

Solusi:
1. JANGAN simpan self.converter sebagai instance variable permanen
2. Load DocumentConverter hanya saat extract_pdf() dipanggil
3. Setelah selesai, del converter dan panggil gc.collect() untuk release DLL
4. Ini memastikan torch dari Docling sudah dilepas sebelum Whisper jalan
"""
import os
import gc
from pathlib import Path
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DoclingService:

    def is_available(self) -> bool:
        """Cek apakah docling terinstall tanpa load model."""
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_pdf(self, pdf_path: str, doc_title: Optional[str] = None) -> dict:
        """
        Ekstrak teks dari PDF.

        PENTING: converter di-load dan di-release dalam method ini saja.
        Tidak disimpan sebagai self.converter agar torch DLL dilepas
        setelah method ini selesai — mencegah konflik dengan Faster Whisper.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not self.is_available():
            raise RuntimeError(
                "Docling tidak terinstall. Jalankan: pip install docling"
            )

        converter = None
        try:
            logger.info(f"[Docling] Loading converter for: {Path(pdf_path).name}")
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()

            result = converter.convert(pdf_path)
            doc = result.document
            full_text = doc.export_to_markdown()

            # Per-page extraction (best-effort)
            pages = []
            page_count = 0
            try:
                for page in doc.pages:
                    page_text_parts = []
                    for item, _ in doc.iterate_items(page_no=page.page_no):
                        if hasattr(item, "text") and item.text:
                            page_text_parts.append(item.text)
                    pages.append({
                        "page_num": page.page_no + 1,
                        "text": " ".join(page_text_parts).strip(),
                    })
                    page_count += 1
            except Exception:
                pages = [{"page_num": 1, "text": full_text}]
                page_count = 1

            logger.info(
                f"[Docling] ✅ Extracted: {page_count} pages, {len(full_text)} chars"
            )

            return {
                "source_file": Path(pdf_path).name,
                "doc_title": doc_title or Path(pdf_path).stem,
                "total_pages": page_count,
                "total_chars": len(full_text),
                "full_text": full_text,
                "pages": pages,
            }

        finally:
            # Selalu release converter setelah selesai
            # Ini penting untuk Windows agar torch DLL dilepas dari memori
            if converter is not None:
                try:
                    del converter
                except Exception:
                    pass
            try:
                del result
                del doc
            except Exception:
                pass
            gc.collect()
            logger.debug("[Docling] Converter released from memory (DLL freed)")