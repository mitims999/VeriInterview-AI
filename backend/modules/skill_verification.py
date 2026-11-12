"""
Feature 3 — Resume-to-Interview Skill Verification (Master Spec Section 6).

Aggregates per-question answer evaluation into a per-skill verdict. A skill
claimed on the resume but never tested is NOT_ASSESSED, never DOES_NOT_KNOW.
"""
from __future__ import annotations
from collections import defaultdict
from backend.schemas import AnswerEvaluation, QAExchange, SkillVerdict, ResumeClaim

_STRENGTH_FLOOR = {"weak": 40, "medium": 55, "strong": 70}  # expected score floor implied by claim strength


def verify_skills(
    qa_list: list[QAExchange],
    evaluations: list[AnswerEvaluation],
    resume_claims: list[ResumeClaim],
) -> list[SkillVerdict]:
    eval_by_q = {e.question_id: e for e in evaluations}
    by_skill: dict[str, list[tuple[QAExchange, AnswerEvaluation]]] = defaultdict(list)
    for qa in qa_list:
        ev = eval_by_q.get(qa.question_id)
        if ev:
            by_skill[qa.skill].append((qa, ev))

    claimed_skills = {c.skill: c for c in resume_claims}
    verdicts = []

    all_skills = set(claimed_skills) | set(by_skill)
    for skill in sorted(all_skills):
        pairs = by_skill.get(skill, [])
        if not pairs:
            verdicts.append(SkillVerdict(skill=skill, verdict="NOT_ASSESSED", evidence_score=0.0, questions_used=[]))
            continue

        avg_score = sum(e.score for _, e in pairs) / len(pairs)
        claim = claimed_skills.get(skill)
        floor = _STRENGTH_FLOOR.get(claim.claim_strength, 50) if claim else 50

        if avg_score >= 85:
            verdict = "STRONGLY_VERIFIED"
        elif avg_score >= 70:
            verdict = "VERIFIED"
        elif avg_score >= 50:
            verdict = "PARTIALLY_VERIFIED"
        elif avg_score >= 30:
            verdict = "WEAKLY_SUPPORTED"
        else:
            verdict = "WEAKLY_SUPPORTED"

        # Mismatch: strong resume claim contradicted by poor demonstrated performance
        if claim and claim.claim_strength == "strong" and avg_score < floor - 20:
            verdict = "RESUME_INTERVIEW_MISMATCH"

        verdicts.append(SkillVerdict(
            skill=skill, verdict=verdict, evidence_score=round(avg_score, 1),
            questions_used=[qa.question_id for qa, _ in pairs],
        ))

    return verdicts


def profile_consistency_risk(verdicts: list[SkillVerdict], resume_claims: list[ResumeClaim]) -> float:
    """
    0-100 risk feeding the fusion engine's profile_consistency_risk weight.

    Only medium/strong resume claims are eligible to generate mismatch risk.
    A "weak" claim (e.g. "familiar with AWS") sets a low expectation already —
    a weak candidate performing weakly against a weak claim is a CAPABILITY
    signal, not an integrity signal, and must not raise risk (Master Spec
    Section 6: genuine-but-weak candidates must never be flagged as suspicious).
    """
    claimed = {c.skill: c for c in resume_claims}
    eligible = {s: c for s, c in claimed.items() if c.claim_strength in ("medium", "strong")}
    if not eligible:
        return 0.0
    mismatch_weight = 0.0
    total_weight = 0.0
    strength_w = {"medium": 1.0, "strong": 1.5}
    for v in verdicts:
        claim = eligible.get(v.skill)
        if not claim:
            continue
        w = strength_w.get(claim.claim_strength, 1.0)
        total_weight += w
        if v.verdict == "RESUME_INTERVIEW_MISMATCH":
            mismatch_weight += w
        elif v.verdict == "WEAKLY_SUPPORTED":
            mismatch_weight += w * 0.5
    if total_weight == 0:
        return 0.0
    return round(min(100.0, 100.0 * mismatch_weight / total_weight), 1)
