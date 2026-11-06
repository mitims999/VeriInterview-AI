"""
Feature 2 — Resume Parsing (Master Spec Section 5). Supports PDF, DOCX, TXT.
Delegates keyword/skill extraction to the Resume-Matcher adapter (Section 3.1).
"""
from __future__ import annotations
import io
from backend.adapters.resume_matcher import extract_skill_profile
from backend.schemas import SkillProfile


from backend.modules.document_parser import parse_document_bytes


def parse_resume_bytes(filename: str, content: bytes) -> str:
    return parse_document_bytes(filename, content)


def build_profile(resume_text: str, job_description: str, resume_claims_raw: list[dict]) -> SkillProfile:
    return extract_skill_profile(resume_text, job_description, resume_claims_raw)
