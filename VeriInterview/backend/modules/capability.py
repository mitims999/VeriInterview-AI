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
    return round(0.7 * avg_answer_score + 0.3 * buzzword_coverage_score, 1)
