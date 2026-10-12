"""
Integration test: resume -> question -> answer -> evaluation -> skill verification ->
risk fusion (Master Spec Section 26).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.modules.pipeline import run_full_analysis

RECORD = {
    "candidate_name": "Test Candidate",
    "job_description": "Backend Engineer with AWS and Python experience, EC2 and S3.",
    "resume_text": "Designed and deployed AWS architecture using EC2 and S3.",
    "resume_claims": [{"skill": "AWS", "claim_text": "Designed and deployed AWS architecture using EC2 and S3.", "claim_strength": "strong"}],
    "qa_list": [{
        "question_id": "Q1", "question_text": "How did you use AWS?", "skill": "AWS",
        "difficulty": "Basic", "expected_proficiency": "Intermediate",
        "reference_answer": "A strong answer mentions EC2 for compute and S3 for storage.",
        "required_concepts": ["EC2", "S3"],
        "candidate_answer": "I used EC2 for compute and S3 for storage.",
        "response_latency_sec": 2.0, "answer_duration_sec": 20.0,
    }],
    "speech_features": [
        {"question_id": "Q1", "pitch_hz": 170, "loudness_db": -18, "speech_rate_wpm": 128, "pause_ratio": 0.15, "speech_silence_ratio": 0.78},
        {"question_id": "Q2", "pitch_hz": 171, "loudness_db": -18, "speech_rate_wpm": 129, "pause_ratio": 0.15, "speech_silence_ratio": 0.78},
    ],
    "behavior_signals": {"gaze_off_screen_pct": 0.05, "head_pose_variance": "normal", "multi_person_detected": False, "phone_detected": False},
    "identity_signals": {"min_frame_similarity_to_anchor": 0.98, "face_lost_pct": 0.0},
    "av_signals": {"av_offset_frames": 1, "sync_confidence": 0.9},
    "counterfactuals": [],
}


def test_full_pipeline_produces_valid_report():
    result = run_full_analysis(RECORD)
    assessment = result["assessment"]
    assert 0 <= assessment["risk_score"] <= 100
    assert assessment["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0 <= assessment["confidence"] <= 1
    assert "recommended_action" in assessment
    assert result["skill_verdicts"][0]["skill"] == "AWS"
    assert result["answer_evaluations"][0]["question_id"] == "Q1"


def test_missing_answer_still_produces_a_report_without_crashing():
    record = dict(RECORD)
    record["qa_list"] = [dict(RECORD["qa_list"][0], candidate_answer="")]
    result = run_full_analysis(record)
    assert result["answer_evaluations"][0]["score"] == 0.0
    assert result["assessment"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_no_media_signals_reduces_confidence_but_does_not_crash():
    record = dict(RECORD)
    record["av_signals"] = None
    record["identity_signals"] = None
    record["behavior_signals"] = None
    result = run_full_analysis(record)
    assessment = result["assessment"]
    assert assessment["components"]["sync_risk"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert 0 <= assessment["confidence"] <= 1
