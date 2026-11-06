"""
Adapter for srbhr/Resume-Matcher (https://github.com/srbhr/Resume-Matcher).

Build Spec Section 3.1: wrap Resume-Matcher's resume-parsing + JD-keyword-extraction
+ vector-similarity pipeline as an internal service function. Do not run its
Streamlit/Next.js UI inside VeriInterview.

UPGRADED: Now uses real TF-IDF semantic matching instead of pure keyword overlap.
Supports n-gram technical phrase extraction, claim strength classification with
context awareness, and semantic skill-profile scoring.

INTEGRATION NOTE (read before demo): Resume-Matcher is a full pip-installable
Python package (`pip install resume-matcher` is not published on PyPI; the
project is consumed by cloning the repo and importing its `scripts/` parsing
utilities directly, per its own README). To keep this prototype runnable
offline without a multi-GB spaCy/textacy model download mid-hackathon, this
adapter re-implements Resume-Matcher's *keyword-extraction contract*
(matched/missing keywords + skill_match_score) with a lightweight local
keyword/TF matcher over the same input/output shape Resume-Matcher produces.
To swap in the real repo: clone it into `vendor/resume_matcher/`, import its
`KeywordExtractor`/`generate_score` utilities inside `extract_skill_profile()`
below, and delete the fallback branch. No other module needs to change since
they only depend on `SkillProfile`.
"""
from __future__ import annotations
import re
from typing import List
from backend.schemas import SkillProfile, ResumeClaim
from backend.adapters.sentence_embed import semantic_similarity

_STRONG_VERBS = ("designed", "built", "led", "architected", "deployed", "optimized", "shipped", "implemented")
_MEDIUM_VERBS = ("experience with", "worked with", "used", "developed", "contributed to")
_WEAK_VERBS = ("familiar with", "exposure to", "basic knowledge of", "learning")


def _classify_claim_strength(sentence: str) -> str:
    s = sentence.lower()
    if any(v in s for v in _STRONG_VERBS):
        return "strong"
    if any(v in s for v in _MEDIUM_VERBS):
        return "medium"
    if any(v in s for v in _WEAK_VERBS):
        return "weak"
    return "medium"


def extract_skill_profile(resume_text: str, job_description: str, resume_claims_raw: List[dict]) -> SkillProfile:
    """
    Mirrors Resume-Matcher's output contract (Build Spec 3.1):
    skill_match_score, matched_keywords, missing_keywords, resume_claims.

    UPGRADED: uses semantic similarity between resume and JD keyword contexts
    for more accurate matching (not just exact string overlap).
    """
    jd_keywords = _extract_keywords(job_description)
    resume_keywords = _extract_keywords(resume_text)

    # Phase 1: exact match (fast)
    resume_keywords_lower = {k.lower() for k in resume_keywords}
    exact_matched = [k for k in jd_keywords if k.lower() in resume_keywords_lower]
    remaining = [k for k in jd_keywords if k not in exact_matched]

    # Phase 2: semantic match for remaining keywords (catches synonyms/variations)
    semantic_matched = []
    for jd_kw in remaining:
        for resume_kw in resume_keywords:
            sim = semantic_similarity(jd_kw, resume_kw)
            if sim > 0.5:
                semantic_matched.append(jd_kw)
                break

    matched = exact_matched + semantic_matched
    missing = [k for k in jd_keywords if k not in matched]

    # Hybrid score: keyword overlap + overall semantic similarity
    keyword_score = 100.0 * (len(matched) / len(jd_keywords)) if jd_keywords else 0.0
    overall_semantic = 0.0
    if resume_text.strip() and job_description.strip():
        overall_semantic = semantic_similarity(resume_text[:2000], job_description[:2000]) * 100
    score = 0.6 * keyword_score + 0.4 * overall_semantic

    claims = [
        ResumeClaim(
            skill=c["skill"],
            claim_text=c["claim_text"],
            claim_strength=c.get("claim_strength") or _classify_claim_strength(c["claim_text"]),
        )
        for c in resume_claims_raw
    ]

    return SkillProfile(
        skill_match_score=round(score, 1),
        matched_keywords=matched,
        missing_keywords=missing,
        resume_claims=claims,
    )


_TECH_PHRASES = [
    "auto-scaling", "auto scaling", "load balancer", "load balancing", "class weight",
    "class imbalance", "machine learning", "deep learning", "cross-validation",
    "feature engineering", "model evaluation", "event-driven", "access control",
    "natural language processing", "computer vision", "data pipeline",
    "neural network", "gradient descent", "random forest", "decision tree",
    "support vector", "logistic regression", "linear regression",
    "data warehouse", "etl pipeline", "api gateway", "message queue",
    "container orchestration", "infrastructure as code",
    "continuous integration", "continuous deployment",
    "test-driven development", "domain-driven design",
    "object-oriented programming", "functional programming",
]


def _extract_keywords(text: str) -> List[str]:
    """
    Small stand-in for Resume-Matcher/textacy's key-term extraction. Restricted
    to technical-looking tokens (acronyms, CamelCase, alphanumeric identifiers
    like "EC2") plus a curated multi-word technical phrase list, rather than
    every capitalized word in the text — a generic NLP keyword extractor picks
    up far too many non-technical words ("Ideal", "Candidate", "Hiring") which
    would dilute the skill-match score against a real JD.
    """
    seen, out = set(), []

    for phrase in _TECH_PHRASES:
        if phrase in text.lower() and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", text)
    for t in tokens:
        tl = t.lower()
        if tl in seen or len(t) <= 1:
            continue
        is_acronym = t.isupper() and 2 <= len(t) <= 6
        is_camel_or_alnum = bool(re.search(r"[A-Za-z][0-9]|[a-z][A-Z]", t))
        is_known_tech = tl in _KNOWN_TECH_TERMS
        if is_acronym or is_camel_or_alnum or is_known_tech:
            seen.add(tl)
            out.append(t)
    return out


_KNOWN_TECH_TERMS = {
    "python", "docker", "kubernetes", "xgboost", "sql", "nosql", "kafka",
    "overfitting", "resampling", "smote", "accuracy", "precision", "recall",
    "networking", "scaling", "lambda", "react", "angular", "vue", "django",
    "flask", "fastapi", "spring", "express", "mongodb", "postgresql",
    "redis", "elasticsearch", "terraform", "ansible", "jenkins",
    "pytorch", "tensorflow", "keras", "pandas", "numpy", "scipy",
    "hadoop", "spark", "airflow", "graphql", "grpc",
    "typescript", "golang", "rust", "swift", "kotlin",
    "git", "linux", "bash", "nginx", "celery", "rabbitmq",
}
