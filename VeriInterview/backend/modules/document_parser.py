"""
Universal Document Parser — extracts clean text from uploaded PDF, DOCX, DOC, TXT, and Markdown files.

Uses PyMuPDF (fitz) for fast, accurate PDF text extraction, falling back to pypdf.
Uses python-docx for DOCX files.
Handles various text encodings (utf-8, utf-8-sig, latin-1, cp1252).
Cleans non-printable/corrupted bytes so raw binary is never returned.
"""
from __future__ import annotations
import io
import re
import string


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable control characters."""
    if not text:
        return ""
    # Replace null bytes and non-printable control characters (keep \n, \r, \t)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Normalize line endings
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse 3+ newlines to 2
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _is_binary_garbage(text: str) -> bool:
    """Check if the extracted text looks like binary garbage rather than readable text."""
    if not text or len(text.strip()) == 0:
        return True
    sample = text[:1000]
    # Check proportion of printable characters
    printable_count = sum(1 for c in sample if c in string.printable)
    ratio = printable_count / len(sample)
    return ratio < 0.70


def parse_pdf_bytes(content: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz) or pypdf fallback."""
    # 1. Try PyMuPDF (fitz) - best layout preservation and font handling
    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(stream=content, filetype="pdf")
        pages_text = []
        for page in doc:
            t = page.get_text()
            if t:
                pages_text.append(t)
        doc.close()
        full_text = "\n\n".join(pages_text).strip()
        if full_text and not _is_binary_garbage(full_text):
            return _clean_text(full_text)
    except Exception:
        pass

    # 2. Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(content))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        full_text = "\n\n".join(pages_text).strip()
        if full_text and not _is_binary_garbage(full_text):
            return _clean_text(full_text)
    except Exception:
        pass

    return ""


def parse_docx_bytes(content: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        import docx  # type: ignore
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        full_text = "\n\n".join(paragraphs).strip()
        if full_text and not _is_binary_garbage(full_text):
            return _clean_text(full_text)
    except Exception:
        pass
    return ""


def parse_text_bytes(content: bytes) -> str:
    """Decode raw text bytes trying common encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            decoded = content.decode(enc)
            if not _is_binary_garbage(decoded):
                return _clean_text(decoded)
        except (UnicodeDecodeError, Exception):
            continue
    return _clean_text(content.decode("utf-8", errors="ignore"))


def parse_document_bytes(filename: str, content: bytes) -> str:
    """
    Universal document parser.
    Auto-detects format by filename extension AND magic bytes.
    Returns clean plain text.
    """
    lower = filename.lower()
    is_pdf = lower.endswith(".pdf") or content[:5] == b"%PDF-"
    is_docx = lower.endswith(".docx") or (content[:4] == b"PK\x03\x04" and lower.endswith(".docx"))
    is_doc = lower.endswith(".doc")

    if is_pdf:
        extracted = parse_pdf_bytes(content)
        if extracted and not _is_binary_garbage(extracted):
            return extracted

    if is_docx:
        extracted = parse_docx_bytes(content)
        if extracted and not _is_binary_garbage(extracted):
            return extracted

    # If it's a legacy .doc or binary file where docx/pdf didn't match
    if is_doc:
        # Try docx in case it's actually docx with .doc extension
        extracted = parse_docx_bytes(content)
        if extracted and not _is_binary_garbage(extracted):
            return extracted
        # Extract ASCII strings from binary file
        strings = re.findall(rb'[\x20-\x7e\n\r\t]{4,}', content)
        if strings:
            decoded = b"\n".join(strings).decode("ascii", errors="ignore")
            if not _is_binary_garbage(decoded):
                return _clean_text(decoded)

    # Text / Markdown fallback
    text = parse_text_bytes(content)
    if not _is_binary_garbage(text):
        return text

    # Final fallback: strip non-printable bytes
    return _clean_text(content.decode("utf-8", errors="ignore"))
