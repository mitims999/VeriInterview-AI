"""
Candidate Capability Score — reported separately from Interview Integrity Risk
(Build Spec Section 4: "never merge them into one number").

Combines average answer correctness with buzzword/role-keyword coverage.
"""
from __future__ import annotations
from backend.schemas import AnswerEvaluation


def capability_score(evaluations: list[AnswerEvaluation], buzzword_coverage_score: float) -> float:
    if not evaluations:
        return 0.0
    avg_answer_score = sum(e.score for e in evaluations) / len(evaluations)
    # If the candidate fails the technical questions (score < 40),
    # buzzword name-dropping must not artificially inflate capability!
    if avg_answer_score < 40:
        return round(avg_answer_score, 1)
    return round(0.85 * avg_answer_score + 0.15 * min(avg_answer_score, buzzword_coverage_score), 1)
