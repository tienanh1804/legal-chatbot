"""Extract plain text from PDF, DOCX, and images (Gemini vision OCR)."""

from __future__ import annotations

import io
import logging
import mimetypes
import os
import shutil
import subprocess
import zipfile
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def extract_pdf_pages(path: str) -> Tuple[int, List[Tuple[int, str]]]:
    """Return (page_count, [(page_1_based, text), ...])."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF (pymupdf) is required for PDF support") from e

    doc = fitz.open(path)
    pages: List[Tuple[int, str]] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append((i + 1, text.strip()))
    finally:
        doc.close()
    return len(pages), pages


def _is_ooxml_word_file(path: str) -> bool:
    """True if this is a Word 2007+ file (ZIP package with word/document.xml)."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            return "word/document.xml" in names
    except (zipfile.BadZipFile, OSError):
        return False


def extract_docx_text(path: str) -> str:
    """Extract all text from a DOCX / OOXML Word file (no reliable page numbers)."""
    try:
        import docx
    except ImportError as e:
        raise RuntimeError("python-docx is required for DOCX support") from e

    document = docx.Document(path)
    parts: List[str] = []
    for p in document.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


_LEGACY_DOC_NO_TOOL_MSG = (
    "File .doc (Word 97–2003) cần công cụ antiword để đọc. "
    "Trong Docker image đã có sẵn; nếu chạy local Windows, hãy cài antiword hoặc chuyển sang .docx/PDF."
)


def extract_legacy_doc_binary(path: str) -> str:
    """
    Trích văn bản từ .doc nhị phân (Word 97–2003) bằng antiword (CLI).

    Trên Linux/Docker: cài gói `antiword`. Windows: cần bản antiword trong PATH hoặc dùng Docker.
    """
    exe = shutil.which("antiword")
    if not exe:
        raise RuntimeError("ANTIWORD_NOT_FOUND")

    cmd: List[str] = [exe]
    utf8_map = "/usr/share/antiword/UTF-8.txt"
    if os.path.isfile(utf8_map):
        cmd.extend(["-m", utf8_map])
    cmd.append(os.path.abspath(path))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError("Đọc file .doc quá lâu (timeout).") from e

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise ValueError(f"antiword không đọc được file: {err or proc.returncode}")

    text = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    return text


def _image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def ocr_image_with_gemini(path: str) -> str:
    """OCR via Gemini multimodal (no local tesseract)."""
    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow and google-generativeai required for image OCR") from e

    try:
        from core.config import GEMINI_API_KEYS, GEMINI_MODEL
    except ImportError:
        from backend.core.config import GEMINI_API_KEYS, GEMINI_MODEL

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY configured for image OCR")

    genai.configure(api_key=GEMINI_API_KEYS[0])
    model = genai.GenerativeModel(GEMINI_MODEL)

    raw = _image_bytes(path)
    img = Image.open(io.BytesIO(raw))
    prompt = (
        "Bạn là OCR. Chỉ trả về toàn bộ văn bản đọc được từ ảnh, "
        "giữ nguyên xuống dòng hợp lý. Không giải thích, không thêm lời dẫn."
    )
    response = model.generate_content([prompt, img])
    text = (response.text or "").strip()
    if not text:
        logger.warning("Gemini OCR returned empty text for %s", path)
    return text


def extract_text_by_path(path: str, mime: str | None = None) -> Tuple[int, List[Tuple[int, int, str]]]:
    """
    Return page_count and list of (page_start, page_end, text_segment).

    For DOCX/images, page_start/end are approximate (1-based).
    """
    mime = mime or _guess_mime(path)
    ext = os.path.splitext(path)[1].lower()

    if mime == "application/pdf" or ext == ".pdf":
        n, pages = extract_pdf_pages(path)
        segments: List[Tuple[int, int, str]] = []
        for pnum, txt in pages:
            if txt:
                segments.append((pnum, pnum, txt))
        return n, segments

    word_mimes = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    )
    is_word_by_ext = ext in (".docx", ".doc")
    is_word_by_mime = mime in word_mimes or "wordprocessingml" in (mime or "")

    if is_word_by_ext or is_word_by_mime:
        body = ""
        if _is_ooxml_word_file(path):
            try:
                body = extract_docx_text(path)
            except Exception as e:
                logger.exception("OOXML Word extract failed for %s", path)
                raise ValueError(
                    "Không đọc được file Word (.docx/OOXML). Thử lưu lại .docx hoặc PDF. "
                    f"Chi tiết: {e}"
                ) from e
        elif ext == ".doc":
            try:
                body = extract_legacy_doc_binary(path)
            except RuntimeError as e:
                if "ANTIWORD_NOT_FOUND" in str(e):
                    raise ValueError(_LEGACY_DOC_NO_TOOL_MSG) from e
                raise
        else:
            try:
                body = extract_docx_text(path)
            except Exception as e:
                logger.exception("Word extract failed for %s", path)
                raise ValueError(
                    "Không đọc được file Word. Thử lưu lại .docx hoặc PDF. "
                    f"Chi tiết: {e}"
                ) from e

        return 1, [(1, 1, body)] if body.strip() else (1, [])

    if mime.startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        txt = ocr_image_with_gemini(path)
        return 1, [(1, 1, txt)] if txt.strip() else (1, [])

    raise ValueError(f"Unsupported file type: {mime} ({path})")
