"""
Feature 2 — Resume Parsing (Master Spec Section 5). Supports PDF, DOCX, TXT.
Delegates keyword/skill extraction to the Resume-Matcher adapter (Section 3.1).
"""
from __future__ import annotations
import io
from backend.adapters.resume_matcher import extract_skill_profile
from backend.schemas import SkillProfile


def parse_resume_bytes(filename: str, content: bytes) -> str:
    lower = filename.lower()
    try:
        if lower.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if lower.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return content.decode("utf-8", errors="ignore")


def build_profile(resume_text: str, job_description: str, resume_claims_raw: list[dict]) -> SkillProfile:
    return extract_skill_profile(resume_text, job_description, resume_claims_raw)
