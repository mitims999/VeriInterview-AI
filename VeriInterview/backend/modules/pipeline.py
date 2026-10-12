"""
End-to-end orchestration: takes one interview record (matching the synthetic
fixture schema in data/synthetic/*.json, or an equivalent record built live
from the interviewer-created interview) and runs the full VeriInterview
pipeline described in Master Spec Section 3 (core workflow diagram).

This is the single place that wires every module together; it contains no
scoring logic of its own beyond passing data between modules, per the
"keep modules independent, create clear interfaces" rule (Master Spec 41).
"""
from __future__ import annotations
from backend.schemas import QAExchange, ResumeClaim
from backend.modules import answer_evaluation, skill_verification, buzzword, response_timing, capability
from backend.modules import knowledge_ownership, evidence_fusion, ai_answer_detector
from backend.adapters import opensmile_adapter, proctoring_adapter, syncnet_adapter, insightface_adapter
from backend.modules.resume import build_profile


def run_full_analysis(record: dict) -> dict:
    resume_claims = [ResumeClaim(**c) for c in record.get("resume_claims", [])]
    qa_list = [QAExchange(**qa) for qa in record.get("qa_list", [])]

    # Feature 2/3: resume profile + skill verification
    profile = build_profile(record.get("resume_text", ""), record.get("job_description", ""), record.get("resume_claims", []))

    # Feature 4 (Member 4): multi-dimension answer correctness / expected-answer matching
    evaluations = answer_evaluation.evaluate_all(qa_list)

    # Feature 7 (Member 4): AI answer detection — runs after evaluations, mutates
    # each evaluation's ai_likelihood in-place as a side-effect
    ai_result = ai_answer_detector.run(qa_list, evaluations)

    # Feature 3 aggregation
    verdicts = skill_verification.verify_skills(qa_list, evaluations, resume_claims)
    profile_risk_score = skill_verification.profile_consistency_risk(verdicts, resume_claims)

    # Section 3.5: buzzword coverage (capability signal)
    bw = buzzword.buzzword_coverage(record.get("job_description", ""), [qa.candidate_answer for qa in qa_list])

    # Feature 9: response timing
    timing_result = response_timing.analyze_timing(qa_list, evaluations)

    # Feature 8: voice modulation / speech behavior (openSMILE adapter)
    speech_result = opensmile_adapter.compute(record.get("speech_features"))

    # Feature 7: gaze / behavior (Proctoring-AI adapter)
    behavior_result = proctoring_adapter.compute(record.get("behavior_signals"))

    # Feature 6: lip-sync (SyncNet adapter)
    sync_result = syncnet_adapter.compute(record.get("av_signals"))

    # Face identity (InsightFace adapter)
    identity_result = insightface_adapter.compute(record.get("identity_signals"))

    # Feature 10: knowledge ownership / CCST
    # Gate each counterfactual by how well the candidate scored on the ORIGINAL
    # question for that skill, so a weak-but-honest candidate isn't penalized
    # for failing a follow-up they never claimed to master (see module docstring).
    eval_by_q = {e.question_id: e for e in evaluations}
    original_by_skill = {qa.skill: eval_by_q[qa.question_id].score for qa in qa_list if qa.question_id in eval_by_q}
    counterfactuals = []
    for cf in record.get("counterfactuals", []):
        cf = dict(cf)
        cf.setdefault("original_answer_score", original_by_skill.get(cf.get("skill"), 70))
        counterfactuals.append(cf)
    ko_result = knowledge_ownership.evaluate_counterfactuals(counterfactuals)

    # Fold response-timing evidence into profile_consistency module result so the
    # fusion engine's 6 weighted components (Build Spec Section 4) stay intact,
    # while timing evidence still surfaces on the report timeline.
    profile_consistency_result = _blend_profile_and_timing(profile_risk_score, verdicts, timing_result)

    cap_score = capability.capability_score(evaluations, bw["buzzword_coverage_score"])

    assessment = evidence_fusion.fuse(
        candidate_name=record.get("candidate_name", "Unknown Candidate"),
        identity=identity_result,
        sync=sync_result,
        behavior=behavior_result,
        speech=speech_result,
        knowledge_ownership=ko_result,
        profile_consistency=profile_consistency_result,
        capability_score=cap_score,
    )

    return {
        "assessment": assessment.model_dump(),
        "skill_profile": profile.model_dump(),
        "skill_verdicts": [v.model_dump() for v in verdicts],
        "answer_evaluations": [e.model_dump() for e in evaluations],
        "answer_summary": answer_evaluation.answer_summary(evaluations),
        "ai_detection": ai_result.model_dump(),
        "buzzword_coverage": bw,
        "qa_list": [qa.model_dump() for qa in qa_list],
    }


def _blend_profile_and_timing(profile_risk_score: float, verdicts, timing_result):
    from backend.schemas import ModuleResult, Evidence, Quality
    mismatch_evidence = [
        Evidence(type="skill_mismatch", severity="high" if v.verdict == "RESUME_INTERVIEW_MISMATCH" else "medium",
                  message=f"{v.skill}: resume claim not supported by interview performance (verdict: {v.verdict}).")
        for v in verdicts if v.verdict in ("RESUME_INTERVIEW_MISMATCH", "WEAKLY_SUPPORTED")
    ]
    combined_score = min(100.0, 0.6 * profile_risk_score + 0.4 * timing_result.score) if timing_result.status == "OK" else profile_risk_score
    level = "LOW" if combined_score <= 30 else ("MEDIUM" if combined_score <= 60 else "HIGH")
    evidence = mismatch_evidence + timing_result.evidence
    if not evidence:
        evidence = [Evidence(type="profile_consistency", severity="low", message="Resume claims and interview timing are consistent.")]
    return ModuleResult(
        feature="profile_consistency", score=round(combined_score, 1),
        confidence=round((0.8 + timing_result.confidence) / 2, 2) if timing_result.status == "OK" else 0.7,
        level=level, evidence=evidence, quality=Quality(data_quality=1.0, baseline_sufficient=True),
    )
