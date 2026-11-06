"""
End-to-end orchestration: takes one interview record (matching the synthetic
fixture schema in data/synthetic/*.json, or an equivalent record built live
from the interviewer-created interview) and runs the full VeriInterview
pipeline described in Master Spec Section 3 (core workflow diagram).

This is the single place that wires every module together; it contains no
scoring logic of its own beyond passing data between modules, per the
"keep modules independent, create clear interfaces" rule (Master Spec 41).

Owned by Member 5. Integration notes for the other four members' modules —
full rationale in evidence_fusion.py's module docstring:
  - Member 3's deepfake_fusion is computed and returned as
    `deepfake_assessment` (and surfaced by xai_explainer as an "Authenticity"
    group) but does NOT change the 6 evidence_fusion weights, to avoid
    double-counting identity/sync evidence and to preserve the non-negotiable
    5-scenario e2e tests that key on identity_risk/sync_risk independently.
  - Member 4's AI-answer-detection result is blended into knowledge_ownership
    (both ask "does this answer reflect genuine ownership?").
  - Member 2's keyword/buzzword-stuffing cross-reference is blended into
    profile_consistency alongside the existing skill-mismatch + timing blend
    (all three ask "is the resume consistent with the interview?").
"""
from __future__ import annotations
from backend.schemas import QAExchange, ResumeClaim, ModuleResult, Evidence, Quality
from backend.modules import answer_evaluation, skill_verification, buzzword, response_timing, capability
from backend.modules import knowledge_ownership, evidence_fusion, ai_answer_detector, keyword_detector, xai_explainer
from backend.adapters import opensmile_adapter, proctoring_adapter, syncnet_adapter, insightface_adapter
from backend.modules.resume import build_profile
from backend.modules.deepfake_fusion import fuse_deepfake


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

    # Feature F8 (Member 2): buzzword-stuffing / hollow-claims / resume-interview
    # cross-reference, run over the same resume + transcript + evaluations.
    transcript_text = " ".join(qa.candidate_answer for qa in qa_list)
    kw_report = keyword_detector.detect_all(
        resume_text=record.get("resume_text", ""),
        transcript_text=transcript_text,
        qa_list=[qa.model_dump() for qa in qa_list],
        evaluations=[e.model_dump() for e in evaluations],
    )

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

    # Blend Member 4's AI-answer-detection signal into knowledge ownership —
    # both ask whether the candidate's answer reflects genuine understanding.
    knowledge_ownership_result = _blend_knowledge_ownership_and_ai_detection(ko_result, ai_result)

    # Fold response-timing evidence AND Member 2's keyword/cross-reference
    # suspicion score into profile_consistency so the fusion engine's 6
    # weighted components (Build Spec Section 4) stay intact, while all three
    # signals still surface their own evidence on the report timeline.
    profile_consistency_result = _blend_profile_consistency(profile_risk_score, verdicts, timing_result, kw_report)

    cap_score = capability.capability_score(evaluations, bw["buzzword_coverage_score"])

    assessment = evidence_fusion.fuse(
        candidate_name=record.get("candidate_name", "Unknown Candidate"),
        identity=identity_result,
        sync=sync_result,
        behavior=behavior_result,
        speech=speech_result,
        knowledge_ownership=knowledge_ownership_result,
        profile_consistency=profile_consistency_result,
        capability_score=cap_score,
    )

    # Member 3: Deepfake Sub-Score from identity + lip-sync + eye cues. Report-only
    # (see this module's and evidence_fusion.py's docstrings for why it does not
    # also change the 6 fusion weights above) but surfaced prominently by xai_explainer.
    deepfake = fuse_deepfake(identity_result, sync_result, record.get("eye_authenticity"))

    explanation = xai_explainer.explain(assessment, deepfake)

    return {
        "deepfake_assessment": deepfake.model_dump(),
        "assessment": assessment.model_dump(),
        "explanation": explanation.model_dump(),
        "skill_profile": profile.model_dump(),
        "skill_verdicts": [v.model_dump() for v in verdicts],
        "answer_evaluations": [e.model_dump() for e in evaluations],
        "answer_summary": answer_evaluation.answer_summary(evaluations),
        "ai_detection": ai_result.model_dump(),
        "keyword_detection": kw_report,
        "buzzword_coverage": bw,
        "qa_list": [qa.model_dump() for qa in qa_list],
    }


def _blend_knowledge_ownership_and_ai_detection(ko_result: ModuleResult, ai_result) -> ModuleResult:
    """
    Combine Member 5's original CCST knowledge-ownership score with Member 4's
    AI-answer-detection score. Both measure the same underlying question —
    "does this answer reflect the candidate's own understanding?" — from
    different angles (counterfactual stress-test vs. stylometric/behavioral
    AI-generation signals), so they are corroborating evidence for one
    component rather than two separate fusion weights.
    """
    if ai_result.status != "OK":
        return ko_result
    if ko_result.status != "OK":
        # No counterfactuals were run; AI-detection alone still carries the component.
        ai_score = ai_result.overall_ai_likelihood * 100.0
        level = "LOW" if ai_score <= 30 else ("MEDIUM" if ai_score <= 60 else "HIGH")
        return ModuleResult(
            feature="knowledge_ownership", score=round(ai_score, 1), confidence=ai_result.confidence,
            level=level, evidence=ai_result.evidence or [
                Evidence(type="ai_answer_detection", severity="low", message="No AI-answer-detection signals triggered.")],
            quality=Quality(data_quality=1.0, baseline_sufficient=True),
        )

    ai_score = ai_result.overall_ai_likelihood * 100.0
    # Weighted toward the CCST result (it's a more direct probe), AI-detection corroborates.
    combined_score = min(100.0, 0.65 * ko_result.score + 0.35 * ai_score)
    level = "LOW" if combined_score <= 30 else ("MEDIUM" if combined_score <= 60 else "HIGH")
    evidence = list(ko_result.evidence) + list(ai_result.evidence)
    return ModuleResult(
        feature="knowledge_ownership", score=round(combined_score, 1),
        confidence=round((ko_result.confidence + ai_result.confidence) / 2, 2),
        level=level, evidence=evidence, quality=ko_result.quality,
    )


def _blend_profile_consistency(profile_risk_score: float, verdicts, timing_result, kw_report: dict) -> ModuleResult:
    mismatch_evidence = [
        Evidence(type="skill_mismatch", severity="high" if v.verdict == "RESUME_INTERVIEW_MISMATCH" else "medium",
                  message=f"{v.skill}: resume claim not supported by interview performance (verdict: {v.verdict}).")
        for v in verdicts if v.verdict in ("RESUME_INTERVIEW_MISMATCH", "WEAKLY_SUPPORTED")
    ]

    keyword_score = kw_report.get("resume_suspicion_score", 0.0)
    keyword_evidence = []
    if kw_report.get("hollow_claims"):
        keyword_evidence.append(Evidence(
            type="hollow_claims", severity="low",
            message=f"Resume contains {len(kw_report['hollow_claims'])} vague/hollow self-descriptor(s) without supporting project context.",
        ))
    if kw_report.get("buzzword_analysis", {}).get("is_stuffed"):
        keyword_evidence.append(Evidence(
            type="buzzword_stuffing", severity="medium",
            message="Resume keyword density suggests buzzword-stuffing relative to substantiated claims.",
        ))
    cross_ref = kw_report.get("cross_reference", {})
    if cross_ref.get("mismatch_count", 0) > 0:
        keyword_evidence.append(Evidence(
            type="cross_reference_mismatch", severity="high",
            message=f"{cross_ref['mismatch_count']} resume skill(s) claimed but contradicted or unsupported when tested in the interview.",
        ))

    if profile_risk_score >= 40:
        combined_score = max(profile_risk_score, keyword_score * 0.6)
    else:
        timing_component = timing_result.score if timing_result.status == "OK" else profile_risk_score
        combined_score = min(100.0, 0.45 * profile_risk_score + 0.30 * timing_component + 0.25 * keyword_score)

    level = "LOW" if combined_score <= 30 else ("MEDIUM" if combined_score <= 60 else "HIGH")
    evidence = mismatch_evidence + timing_result.evidence + keyword_evidence
    if not evidence:
        evidence = [Evidence(type="profile_consistency", severity="low", message="Resume claims and interview timing are consistent.")]
    return ModuleResult(
        feature="profile_consistency", score=round(combined_score, 1),
        confidence=round((0.8 + timing_result.confidence) / 2, 2) if timing_result.status == "OK" else 0.7,
        level=level, evidence=evidence, quality=Quality(data_quality=1.0, baseline_sufficient=True),
    )
