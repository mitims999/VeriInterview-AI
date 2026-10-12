"""Unit tests for individual modules (Master Spec Section 26)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.adapters import opensmile_adapter, proctoring_adapter, syncnet_adapter, insightface_adapter
from backend.adapters.resume_matcher import extract_skill_profile
from backend.modules import skill_verification, knowledge_ownership
from backend.schemas import QAExchange, AnswerEvaluation, ResumeClaim


def test_resume_extraction_matches_tech_keywords():
    profile = extract_skill_profile(
        resume_text="Designed and deployed AWS architecture using EC2 and S3.",
        job_description="Looking for someone with AWS, EC2, S3 and Kubernetes experience.",
        resume_claims_raw=[{"skill": "AWS", "claim_text": "Designed and deployed AWS architecture.", "claim_strength": "strong"}],
    )
    assert "AWS" in profile.matched_keywords
    assert "Kubernetes" in profile.missing_keywords
    assert 0 <= profile.skill_match_score <= 100


def test_speech_behavior_insufficient_data_is_unavailable():
    result = opensmile_adapter.compute([{"question_id": "Q1", "pitch_hz": 150, "loudness_db": -20,
                                          "speech_rate_wpm": 120, "pause_ratio": 0.2, "speech_silence_ratio": 0.7}])
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == 0.0


def test_speech_behavior_flags_deviation():
    baseline = [dict(question_id=f"Q{i}", pitch_hz=170, loudness_db=-18, speech_rate_wpm=128, pause_ratio=0.15, speech_silence_ratio=0.78) for i in range(2)]
    anomaly = [dict(question_id="Q3", pitch_hz=150, loudness_db=-13, speech_rate_wpm=190, pause_ratio=0.03, speech_silence_ratio=0.95)]
    result = opensmile_adapter.compute(baseline + anomaly)
    assert result.status == "OK"
    assert result.level in ("MEDIUM", "HIGH")


def test_proctoring_phone_detection_is_strong_evidence_even_with_low_gaze_deviation():
    result = proctoring_adapter.compute({"gaze_off_screen_pct": 0.02, "head_pose_variance": "normal",
                                          "multi_person_detected": False, "phone_detected": True})
    assert result.level == "HIGH"
    assert result.score >= 80


def test_proctoring_gaze_alone_never_exceeds_medium():
    result = proctoring_adapter.compute({"gaze_off_screen_pct": 0.95, "head_pose_variance": "high",
                                          "multi_person_detected": False, "phone_detected": False})
    assert result.level != "HIGH"


def test_syncnet_large_offset_is_high_risk():
    result = syncnet_adapter.compute({"av_offset_frames": 9, "sync_confidence": 0.3})
    assert result.level == "HIGH"


def test_insightface_identity_drift_flagged():
    result = insightface_adapter.compute({"min_frame_similarity_to_anchor": 0.71, "face_lost_pct": 0.18})
    assert result.level == "HIGH"


def test_skill_not_tested_is_not_assessed_not_does_not_know():
    verdicts = skill_verification.verify_skills(
        qa_list=[], evaluations=[],
        resume_claims=[ResumeClaim(skill="Kubernetes", claim_text="Familiar with Kubernetes.", claim_strength="weak")],
    )
    assert verdicts[0].verdict == "NOT_ASSESSED"


def test_weak_claim_never_creates_mismatch_risk():
    verdicts = [skill_verification.SkillVerdict(skill="AWS", verdict="WEAKLY_SUPPORTED", evidence_score=20, questions_used=["Q1"])]
    risk = skill_verification.profile_consistency_risk(verdicts, [ResumeClaim(skill="AWS", claim_text="Familiar with AWS.", claim_strength="weak")])
    assert risk == 0.0


def test_knowledge_ownership_no_counterfactuals_is_unavailable():
    result = knowledge_ownership.evaluate_counterfactuals([])
    assert result.status == "INSUFFICIENT_EVIDENCE"


def test_knowledge_ownership_low_original_score_is_not_penalized():
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "Machine Learning", "followup_question_id": "Q3F", "perturbed_assumption": "x",
        "candidate_followup_answer": "not sure", "expected_new_concepts": ["SMOTE"], "expected_retained_concepts": ["F1"],
        "original_answer_score": 20,
    }])
    assert result.level == "LOW"


def test_knowledge_ownership_high_original_score_with_failed_followup_is_high_risk():
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "Machine Learning", "followup_question_id": "Q3F", "perturbed_assumption": "x",
        "candidate_followup_answer": "not sure at all", "expected_new_concepts": ["SMOTE", "oversampling"], "expected_retained_concepts": ["F1"],
        "original_answer_score": 90,
    }])
    assert result.level == "HIGH"
