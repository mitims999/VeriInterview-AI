"""
Feature 5 / Build Spec 3.6 — Answer Correctness, Fact-Checking, and Fabrication Detection.

Combines semantic similarity and concept coverage with domain fact-checking:
  1. Catches technical inversions and factual hallucinations
     (e.g. MobileNet as a database, RAG as Random Algorithm Generation,
      K-Means/DBSCAN as supervised learning with decision trees/regression).
  2. Catches admissions of fabrication and bluffing
     (e.g. "thought they would sound impressive", "haven't actually used it in any project").
  3. Detects self-contradictions ("expert in PyTorch... haven't used it in any project").
"""
from __future__ import annotations
import re
from backend.adapters.sentence_embed import semantic_similarity, concept_coverage
from backend.schemas import AnswerEvaluation, QAExchange

# Direct admissions of fabrication, bluffing, or lack of knowledge
_FABRICATION_PHRASES = [
    r"\bsound(?:ed)?\s+impressive\b",
    r"\bto\s+sound\s+impressive\b",
    r"\bhaven't\s+(?:actually\s+)?used\s+it\s+in\s+any\s+(?:of\s+my\s+)?projects\b",
    r"\bnever\s+(?:actually\s+)?used\s+it\s+in\s+any\s+projects?\b",
    r"\bhaven't\s+worked\s+on\s+a\s+project\s+using\s+those\b",
    r"\bthought\s+they\s+would\s+sound\b",
    r"\blearn\s+them\s+after\s+joining\b",
    r"\bcan\s+learn\s+(?:them\s+)?after\s+joining\b",
    r"\bhaven't\s+explored\s+all\s+the\s+technical\s+concepts\b",
    r"\brandom\s+algorithm\s+generation\b",  # Fake acronym for RAG
]

# Gross factual contradictions / hallucinations
_CONTRADICTION_PATTERNS = [
    (
        r"\bmobilenet(?:v\d+)?\b.*?\b(?:database|dbms|storage\s*system|database\s*management)\b",
        "MobileNetV2 is a lightweight CNN vision model, not a database management system.",
    ),
    (
        r"\brag\b.*?\b(?:random\s*algorithm|cluster(?:s|ing)?\s*(?:similar\s*)?data|predict(?:s|ing)?\s*stock)\b",
        "RAG stands for Retrieval-Augmented Generation for language models, not clustering or random algorithms.",
    ),
    (
        r"(?:k-?means|dbscan).*?\bsupervised\s*learning\b",
        "K-Means and DBSCAN are unsupervised clustering algorithms, not supervised learning.",
    ),
    (
        r"\bk-?means\b.*?\bdecision\s*trees?\b",
        "K-Means is a centroid-based clustering algorithm; it does not use decision trees.",
    ),
    (
        r"\bdbscan\b.*?\blinear\s*regression\b",
        "DBSCAN is a density-based clustering algorithm; it does not use linear regression.",
    ),
    (
        r"\bfastapi\b.*?\btrain\s*(?:a\s*)?neural\s*network\b",
        "FastAPI is a Python web framework for building APIs, not an ML training framework.",
    ),
    (
        r"\bexpert\s+in\s+(\w+)\b.*?\bhaven't\s+(?:actually\s+)?used\b",
        "Self-contradiction: claimed expertise while admitting to never having used the technology in projects.",
    ),
]

_DEFLECTION_PHRASES = [
    "don't know", "dont know", "not sure", "haven't worked", "havent worked",
    "never used", "no idea", "not familiar", "pass on this", "cant recall",
    "can't recall", "do not know", "have not used", "not entirely sure",
]


def evaluate_answer(qa: QAExchange) -> AnswerEvaluation:
    ans = qa.candidate_answer.strip()
    if not ans:
        return AnswerEvaluation(
            question_id=qa.question_id, score=0.0, concept_coverage=0.0,
            missing_concepts=qa.required_concepts, incorrect_concepts=["No answer provided"],
            confidence=0.3,
        )

    words = ans.split()
    word_count = len(words)
    ans_lower = ans.lower()
    incorrect_concepts = []

    # 1. Check for blatant fabrication or bluffing admissions
    is_fabricated = False
    for pattern in _FABRICATION_PHRASES:
        if re.search(pattern, ans_lower):
            is_fabricated = True
            incorrect_concepts.append("Admission of fabrication / non-experience")
            break

    # 2. Check for technical factual contradictions / hallucinations
    has_factual_error = False
    for pattern, explanation in _CONTRADICTION_PATTERNS:
        if re.search(pattern, ans_lower):
            has_factual_error = True
            incorrect_concepts.append(explanation)

    # 3. Check for deflection / admitted uncertainty
    has_deflection = any(phrase in ans_lower for phrase in _DEFLECTION_PHRASES)

    # 4. Standard semantic similarity & concept coverage
    coverage, missing = concept_coverage(ans, qa.required_concepts)
    ref_sim = semantic_similarity(ans, qa.reference_answer) if qa.reference_answer else 0.0
    q_sim = semantic_similarity(ans, qa.question_text) if qa.question_text else 0.0
    semantic_score = max(ref_sim, 0.6 * q_sim + 0.4 * ref_sim)
    depth_factor = min(1.0, word_count / 35.0)

    # Calculate base score
    raw_score = 100.0 * (0.45 * semantic_score + 0.40 * coverage + 0.15 * depth_factor)

    # Apply strict penalties if fabrication or factual contradictions are found
    if is_fabricated:
        score = 0.0
    elif has_factual_error:
        score = min(5.0, raw_score * 0.05)
    elif has_deflection and word_count < 25:
        score = min(15.0, raw_score * 0.2)
    elif has_deflection:
        score = min(25.0, raw_score * 0.35)
    elif coverage >= 0.4 and word_count >= 20:
        score = max(raw_score, 75.0 + min(22.0, coverage * 25.0))
    elif coverage >= 0.2 and word_count >= 15:
        score = max(raw_score, 60.0 + min(15.0, coverage * 20.0))
    elif word_count >= 25 and semantic_score >= 0.3:
        score = max(raw_score, 50.0 + min(20.0, semantic_score * 30.0))
    else:
        score = raw_score

    score = min(100.0, max(0.0, round(score, 1)))
    confidence = 0.95 if (is_fabricated or has_factual_error) else (0.9 if word_count >= 12 else 0.6)

    return AnswerEvaluation(
        question_id=qa.question_id,
        score=score,
        concept_coverage=round(coverage, 2),
        missing_concepts=missing,
        incorrect_concepts=incorrect_concepts,
        confidence=confidence,
    )


def evaluate_all(qa_list: list[QAExchange]) -> list[AnswerEvaluation]:
    return [evaluate_answer(qa) for qa in qa_list]
