"""
Feature 5 / Build Spec 3.6 — Answer Correctness / Expected-Answer Matching.

Combines semantic similarity (sentence-transformers adapter) with key-concept
coverage. Never relies on keyword matching alone (Master Spec Section 8).
"""
from __future__ import annotations
from backend.adapters.sentence_embed import semantic_similarity, concept_coverage
from backend.config import ANSWER_EVAL_SEMANTIC_WEIGHT, ANSWER_EVAL_CONCEPT_WEIGHT
from backend.schemas import AnswerEvaluation, QAExchange


def evaluate_answer(qa: QAExchange) -> AnswerEvaluation:
    if not qa.candidate_answer.strip():
        return AnswerEvaluation(
            question_id=qa.question_id, score=0.0, concept_coverage=0.0,
            missing_concepts=qa.required_concepts, incorrect_concepts=[], confidence=0.3,
        )

    semantic_score = semantic_similarity(qa.candidate_answer, qa.reference_answer)
    coverage, missing = concept_coverage(qa.candidate_answer, qa.required_concepts)

    score = 100.0 * (ANSWER_EVAL_SEMANTIC_WEIGHT * semantic_score + ANSWER_EVAL_CONCEPT_WEIGHT * coverage)
    confidence = 0.9 if len(qa.candidate_answer.split()) >= 8 else 0.6

    return AnswerEvaluation(
        question_id=qa.question_id,
        score=round(score, 1),
        concept_coverage=round(coverage, 2),
        missing_concepts=missing,
        incorrect_concepts=[],
        confidence=confidence,
    )


def evaluate_all(qa_list: list[QAExchange]) -> list[AnswerEvaluation]:
    return [evaluate_answer(qa) for qa in qa_list]
