"""
Feature 9 — Response Timing / Prompting Anomalies (Master Spec Section 12).

Deterministic heuristic: flag latency spikes, especially when paired with a
sudden jump in answer quality (classic "reading a prompted answer" pattern).
Contributes into profile_consistency_risk / knowledge_ownership as behavioral
corroboration rather than standing alone.
"""
from __future__ import annotations
from statistics import mean, pstdev
from backend.schemas import QAExchange, AnswerEvaluation, ModuleResult, Evidence, Quality


def analyze_timing(qa_list: list[QAExchange], evaluations: list[AnswerEvaluation]) -> ModuleResult:
    if len(qa_list) < 2:
        return ModuleResult(
            feature="response_timing", score=0, confidence=0.0, level="UNAVAILABLE", status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="timing", severity="low", message="Not enough answered questions to assess timing patterns.")],
        )

    eval_by_q = {e.question_id: e for e in evaluations}
    latencies = [qa.response_latency_sec for qa in qa_list]
    lm, ls = mean(latencies), (pstdev(latencies) if len(latencies) > 1 else 1.0)
    ls = ls or 1.0

    evidence = []
    risk_points = []
    for qa in qa_list:
        z = (qa.response_latency_sec - lm) / ls
        ev = eval_by_q.get(qa.question_id)
        spike = z > 1.8
        quality_jump = ev and ev.score >= 80
        if spike and quality_jump:
            risk_points.append(90)
            evidence.append(Evidence(
                type="latency_quality_spike", severity="high", question_id=qa.question_id,
                message=f"Latency spiked to {qa.response_latency_sec:.1f}s (z={z:.1f}) on {qa.question_id}, "
                        f"immediately followed by an unusually polished answer (score {ev.score}).",
            ))
        elif spike:
            risk_points.append(45)
            evidence.append(Evidence(
                type="latency_spike", severity="medium", question_id=qa.question_id,
                message=f"Response latency on {qa.question_id} ({qa.response_latency_sec:.1f}s) is well above the candidate's own average ({lm:.1f}s).",
            ))
        else:
            risk_points.append(5)

    score = sum(risk_points) / len(risk_points)
    level = "LOW" if score <= 30 else ("MEDIUM" if score <= 60 else "HIGH")
    if not evidence:
        evidence.append(Evidence(type="timing", severity="low", message="Response latencies remained consistent throughout the interview."))

    return ModuleResult(
        feature="response_timing", score=round(score, 1), confidence=0.75, level=level,
        evidence=evidence, quality=Quality(data_quality=1.0, baseline_sufficient=True),
    )
