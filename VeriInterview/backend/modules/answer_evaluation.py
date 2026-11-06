"""
Feature 4/5 — Answer Correctness, Fact-Checking, and Fabrication Detection
(merged: Member 4's multi-dimension scorer + Member 2's domain fact-check pass).

Scoring dimensions (Member 4):
  1. Semantic similarity  — sentence-transformers or TF-IDF cosine fallback
  2. Concept coverage     — what fraction of required concepts appear
  3. Semantic depth       — uniqueness of vocabulary relative to reference,
                            penalising verbatim copying and rewarding elaboration
  4. Length adequacy      — penalise answers that are too short (< threshold words)
                            or suspiciously long / padded

Fact-check pass (Member 2), applied on top — never raises the score, only caps it:
  1. Catches technical inversions and factual hallucinations
     (e.g. MobileNet as a database, RAG as Random Algorithm Generation,
      K-Means/DBSCAN as supervised learning with decision trees/regression).
  2. Catches admissions of fabrication and bluffing
     (e.g. "thought they would sound impressive", "haven't actually used it in any project").
  3. Detects self-contradictions ("expert in PyTorch... haven't used it in any project").

The composite score (0-100) and confidence are passed downstream to:
  - skill_verification.py   (per-skill verdict)
  - ai_answer_detector.py   (ai_likelihood piped back into AnswerEvaluation)
  - pipeline.py             (feeds evidence_fusion + capability score)

Design rule: this module NEVER makes a hiring decision. It produces evidence
scores and flags for human reviewer inspection only.
"""
from __future__ import annotations
import re
from statistics import mean
from backend.adapters.sentence_embed import semantic_similarity, concept_coverage
from backend.config import ANSWER_EVAL_SEMANTIC_WEIGHT, ANSWER_EVAL_CONCEPT_WEIGHT
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


# ── Thresholds (all configurable in config.py if needed) ─────────────────────
MIN_WORDS_BASIC = 10          # Basic questions: penalise if fewer words than this
MIN_WORDS_INTERMEDIATE = 20   # Intermediate / Advanced questions
MAX_WORDS_PADDING_FACTOR = 4  # If candidate answer is >4x reference length, flag padding
DEPTH_UNIQUE_WEIGHT = 0.6     # Weight for unique vocab fraction in depth score
DEPTH_OVERLAP_WEIGHT = 0.4    # Weight for reference overlap in depth score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lower-cased word tokens, letters + digits only."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _type_token_ratio(tokens: list[str]) -> float:
    """Lexical richness: unique types / total tokens (0-1)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _sentence_lengths(text: str) -> list[int]:
    """Split into sentences and return their word-counts."""
    sentences = re.split(r"[.!?]+", text.strip())
    return [len(s.split()) for s in sentences if s.strip()]


def _length_adequacy(answer: str, reference: str, difficulty: str) -> float:
    """
    Returns a 0-1 score:
      - 1.0  if answer length is in the expected range
      - <1.0 penalised for being too short
      - <1.0 slightly penalised for being excessively padded
    """
    a_words = len(answer.split())
    r_words = max(1, len(reference.split()))
    min_words = MIN_WORDS_BASIC if difficulty.lower() == "basic" else MIN_WORDS_INTERMEDIATE

    # Too short
    if a_words < min_words:
        return max(0.1, a_words / min_words)

    # Excessively padded (>4x reference) — suspicious of filler / copy-paste
    if a_words > r_words * MAX_WORDS_PADDING_FACTOR:
        excess = a_words / (r_words * MAX_WORDS_PADDING_FACTOR)
        return max(0.5, 1.0 - 0.1 * (excess - 1))

    return 1.0


def _semantic_depth(answer: str, reference: str) -> float:
    """
    Measures how conceptually rich the answer is vs the reference.

    Two components:
      A) Unique vocabulary fraction of the answer itself (TTR) — rewards
         elaboration over scripted recitation.
      B) Proportion of reference's non-stopword vocabulary that the answer
         also uses — ensures depth is domain-relevant, not just verbose.

    Returns a 0-1 score.
    """
    STOP = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
        "or", "but", "for", "so", "be", "by", "as", "if", "do", "not",
        "was", "are", "with", "this", "that", "have", "from", "you", "we",
        "they", "i", "my", "your", "its", "he", "she", "his", "her",
        "when", "than", "then", "can", "will", "would", "could", "should",
    }
    a_tokens = [t for t in _tokenise(answer) if t not in STOP]
    r_tokens = [t for t in _tokenise(reference) if t not in STOP]

    ttr = _type_token_ratio(a_tokens)                         # A: own richness

    if r_tokens:
        r_vocab = set(r_tokens)
        a_vocab = set(a_tokens)
        overlap = len(a_vocab & r_vocab) / len(r_vocab)       # B: domain overlap
    else:
        overlap = 0.5

    depth = DEPTH_UNIQUE_WEIGHT * ttr + DEPTH_OVERLAP_WEIGHT * overlap
    return round(min(1.0, depth), 3)


# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate_answer(qa: QAExchange) -> AnswerEvaluation:
    """
    Multi-dimension answer evaluation (Member 4) with a domain fact-check pass
    (Member 2) layered on top. The fact-check pass can only cap the score down
    (fabrication admission, factual contradiction, deflection) — it never raises it.

    Returns AnswerEvaluation with:
      score              — 0-100 composite correctness, capped by fact-check findings
      concept_coverage   — 0-1 fraction of required concepts present
      missing_concepts   — which required concepts were absent
      incorrect_concepts — fabrication/contradiction findings from the fact-check pass
      semantic_depth     — 0-1 lexical/semantic richness vs reference
      length_adequacy    — 0-1 length appropriateness
      confidence         — 0-1 evaluator confidence (driven by answer length / findings)
      ai_likelihood      — 0-1 initial estimate; refined by ai_answer_detector later
    """
    ans = qa.candidate_answer.strip()
    if not ans:
        return AnswerEvaluation(
            question_id=qa.question_id,
            score=0.0,
            concept_coverage=0.0,
            missing_concepts=list(qa.required_concepts),
            incorrect_concepts=["No answer provided"],
            confidence=0.2,
            semantic_depth=0.0,
            length_adequacy=0.0,
            ai_likelihood=0.0,
        )

    word_count = len(ans.split())
    ans_lower = ans.lower()
    incorrect_concepts: list[str] = []

    # ── Member 2 fact-check pass: fabrication admissions, factual contradictions, deflection ──
    is_fabricated = False
    for pattern in _FABRICATION_PHRASES:
        if re.search(pattern, ans_lower):
            is_fabricated = True
            incorrect_concepts.append("Admission of fabrication / non-experience")
            break

    has_factual_error = False
    for pattern, explanation in _CONTRADICTION_PATTERNS:
        if re.search(pattern, ans_lower):
            has_factual_error = True
            incorrect_concepts.append(explanation)

    has_deflection = any(phrase in ans_lower for phrase in _DEFLECTION_PHRASES)

    # ── Member 4 multi-dimension scoring ──
    sem_sim = semantic_similarity(qa.candidate_answer, qa.reference_answer)
    cov, missing = concept_coverage(qa.candidate_answer, qa.required_concepts)
    depth = _semantic_depth(qa.candidate_answer, qa.reference_answer)
    adequacy = _length_adequacy(qa.candidate_answer, qa.reference_answer, qa.difficulty)

    # If required concepts are specified, scale raw score so that answers lacking
    # technical concepts cannot score high on conversational overlap alone.
    cov_gate = (0.4 + 0.6 * cov) if qa.required_concepts else 1.0
    raw_score = 100.0 * (
        ANSWER_EVAL_SEMANTIC_WEIGHT * 0.8 * sem_sim
        + ANSWER_EVAL_CONCEPT_WEIGHT * 0.8 * cov
        + 0.2 * depth
    ) * cov_gate
    base_score = min(100.0, raw_score * adequacy)  # length adequacy only penalises, never boosts

    # Fact-check findings cap the score down; they never raise it above the base score.
    if is_fabricated:
        score = 0.0
    elif has_factual_error:
        score = min(5.0, base_score * 0.05)
    elif has_deflection and word_count < 25:
        score = min(15.0, base_score * 0.2)
    elif has_deflection:
        score = min(25.0, base_score * 0.35)
    else:
        score = base_score
    score = round(min(100.0, max(0.0, score)), 1)

    # Evaluator confidence: higher for longer, substantive answers; a confirmed
    # fact-check finding is itself high-confidence evidence.
    if word_count >= 25:
        confidence = 0.95
    elif word_count >= 12:
        confidence = 0.80
    else:
        confidence = 0.55
    if is_fabricated or has_factual_error:
        confidence = max(confidence, 0.95)

    return AnswerEvaluation(
        question_id=qa.question_id,
        score=score,
        concept_coverage=round(cov, 2),
        missing_concepts=missing,
        incorrect_concepts=incorrect_concepts,
        confidence=round(confidence, 2),
        semantic_depth=depth,
        length_adequacy=round(adequacy, 2),
        ai_likelihood=0.0,  # filled in by ai_answer_detector.run()
    )


def evaluate_all(qa_list: list[QAExchange]) -> list[AnswerEvaluation]:
    return [evaluate_answer(qa) for qa in qa_list]


# ── Aggregate helpers consumed by pipeline.py ─────────────────────────────────

def answer_summary(evaluations: list[AnswerEvaluation]) -> dict:
    """
    Returns a brief dict consumed by the pipeline for reporting.
    Not a risk score — purely descriptive capability information.
    """
    if not evaluations:
        return {"average_score": 0.0, "average_depth": 0.0, "average_coverage": 0.0}
    scores = [e.score for e in evaluations]
    depths = [e.semantic_depth for e in evaluations]
    coverages = [e.concept_coverage for e in evaluations]
    return {
        "average_score": round(mean(scores), 1),
        "average_depth": round(mean(depths), 3),
        "average_coverage": round(mean(coverages), 2),
        "questions_evaluated": len(evaluations),
        "strong_answers": sum(1 for s in scores if s >= 70),
        "weak_answers": sum(1 for s in scores if s < 40),
    }
