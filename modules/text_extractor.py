from pathlib import Path

import re
import zipfile
import time

# Some code patterns intentionally repeat small import guards across modules
# (optional OCR and parser libraries). Silence duplicate-code warnings here.
# pylint: disable=duplicate-code,R0801

try:
    import pdfplumber  # type: ignore
except (ImportError, ModuleNotFoundError):
    pdfplumber = None

try:
    from docx import Document  # type: ignore
except (ImportError, ModuleNotFoundError):
    Document = None

try:
    # pdfminer fallback (often available in environments with pdf support)
    from pdfminer.high_level import extract_text as _pdfminer_extract_text  # type: ignore
except (ImportError, ModuleNotFoundError):
    _pdfminer_extract_text = None

# Optional OCR stack: pdf2image for rendering PDFs and pytesseract for OCR
try:
    from pdf2image import convert_from_path, convert_from_bytes  # type: ignore
except (ImportError, ModuleNotFoundError):
    convert_from_path = None
    convert_from_bytes = None

try:
    import pytesseract  # type: ignore
except (ImportError, ModuleNotFoundError):
    pytesseract = None

try:
    from PIL import Image  # type: ignore
except (ImportError, ModuleNotFoundError):
    Image = None


def extract_text(path: Path) -> str:  # pylint: disable=too-many-branches
    """Extract text from a file with retry logic for timing issues.
    
    Includes validation to ensure the file is fully written and readable
    before attempting extraction.
    """
    # Validate file exists and is readable with retry logic
    if not _wait_for_file(path):
        raise ValueError(f"File not ready or not accessible: {path}")
    
    suffix = path.suffix.lower()
    result = ""
    if suffix == ".pdf":
        # prefer pdfplumber, fall back to pdfminer
        if pdfplumber is not None:
            result = _extract_pdf(path)
            if not result:
                # try OCR if no text extracted
                ocr_text = _extract_pdf_ocr(path)
                if ocr_text:
                    result = ocr_text
        elif _pdfminer_extract_text is not None:
            try:
                result = _pdfminer_extract_text(str(path)) or ""
            except (OSError, ValueError):
                result = ""
            if not result:
                result = _extract_pdf_ocr(path)
        else:
            # No PDF-specific extractor available; try OCR as last resort
            result = _extract_pdf_ocr(path)
    elif suffix == ".docx":
        # prefer python-docx, else try a lightweight zip/xml fallback
        if Document is not None:
            result = _extract_docx(path)
        else:
            result = _extract_docx_fallback(path)
    elif suffix == ".txt":
        result = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return result


def _extract_pdf(path: Path) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _extract_pdf_ocr(path: Path, max_pages: int = 3, dpi: int = 200) -> str:
    """Attempt OCR on a PDF using pdf2image + pytesseract.

    This is a best-effort fallback and requires external dependencies
    (`poppler` for pdf2image) and Python packages. It will try up to
    `max_pages` pages to limit cost.
    """
    if pytesseract is None or Image is None:
        return ""

    images = None
    try:
        if convert_from_path is not None:
            images = convert_from_path(str(path), dpi=dpi, first_page=1, last_page=max_pages)
        elif convert_from_bytes is not None:
            data = path.read_bytes()
            images = convert_from_bytes(data, dpi=dpi, first_page=1, last_page=max_pages)
    except (OSError, RuntimeError, ValueError):
        images = None

    if not images:
        return ""

    texts = []
    for img in images:
        try:
            if hasattr(pytesseract, "image_to_string"):
                texts.append(pytesseract.image_to_string(img))
            else:
                # Unexpected pytesseract API; skip
                texts.append("")
        except (OSError, RuntimeError, ValueError):
            texts.append("")

    return "\n".join(t.strip() for t in texts if t.strip())


def _extract_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_docx_fallback(path: Path) -> str:
    """Very small fallback for .docx files that reads word/document.xml.
    This handles simple docx files created by common editors but is not a
    full replacement for python-docx.
    """
    try:
        with zipfile.ZipFile(path, "r") as z:
            try:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
            except KeyError:
                return ""
        # strip tags naivey
        text = xml
        # Replace paragraph and break tags with newlines
        text = text.replace("</w:p>", "\n")
        # Remove other tags
        text = re.sub(r"<[^>]+>", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except (zipfile.BadZipFile, OSError):
        return ""


def _wait_for_file(path: Path, max_retries: int = 3, retry_delay: float = 0.1) -> bool:
    """Wait for file to be fully written and accessible.
    
    This helps handle timing issues where the file may not be immediately
    readable after being saved (e.g., network file systems or slow I/O).
    
    Args:
        path: Path to the file to wait for
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        True if file is accessible, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Check if file exists
            if not path.exists():
                time.sleep(retry_delay)
                continue
            
            # Check if file has non-zero size (ensures it was fully written)
            if path.stat().st_size == 0:
                time.sleep(retry_delay)
                continue
            
            # Try to open and read first few bytes to verify accessibility
            with open(path, 'rb') as f:
                # Read just the first byte to verify file is readable
                f.read(1)
            
            # File is accessible
            return True
            
        except (OSError, IOError, PermissionError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    
    return False
