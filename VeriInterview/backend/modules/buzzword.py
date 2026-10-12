"""
Build Spec Section 3.5 — Keyword/Buzzword-vs-Job-Role Module.

Built on Resume-Matcher's keyword extractor (Section 3.1), applied to
transcript answers rather than the resume. Low coverage is CAPABILITY
evidence by default, only feeding integrity risk when combined with a
high resume-claimed expertise + low technical depth pattern (handled in
evidence_fusion.py, not here).
"""
from __future__ import annotations
from backend.adapters.resume_matcher import _extract_keywords


def buzzword_coverage(job_description: str, candidate_answers: list[str]) -> dict:
    expected = _extract_keywords(job_description)
    combined_answers = " ".join(candidate_answers).lower()
    used = [k for k in expected if k.lower() in combined_answers]
    missing = [k for k in expected if k not in used]
    score = 100.0 * (len(used) / len(expected)) if expected else 100.0
    return {
        "buzzword_coverage_score": round(score, 1),
        "expected_keywords": expected,
        "used_keywords": used,
        "missing_keywords": missing,
        "evidence": [f"Candidate used {len(used)} of {len(expected)} role-critical terms across all answers."] if expected else [],
    }
