"""
Tests for the real-video path of adapters/proctoring_adapter.py (M3 task P-6),
including the R1 rule: steady (nervous) gaze aversion from the first answer must
score lower than a spike on later questions, and gaze alone never reaches HIGH.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.adapters import proctoring_adapter as pa


def _sig(per_q, head="normal", objects=None, coverage=0.95):
    pct = sum(q[1] for q in per_q) / len(per_q)
    s = {"gaze_status": "ok", "gaze_off_screen_pct": pct, "head_pose_variance": head, "yaw_std": 5, "pitch_std": 4,
         "gaze_coverage": coverage, "baseline_gaze_off_pct": per_q[0][1], "gaze_events": [],
         "per_question": [{"question_id": q, "frames": 300, "gaze_off_pct": v, "dominant_direction": "image_left"} for q, v in per_q]}
    s.update(objects or {"objects_status": "ok", "multi_person_detected": False, "phone_detected": False, "object_events": []})
    return s


def test_calm_candidate_is_low():
    r = pa.compute(_sig([("Q1", 0.05), ("Q2", 0.06), ("Q3", 0.04)]))
    assert r.level == "LOW" and r.status == "OK"


def test_nervous_steady_aversion_scores_below_question_spike_r1():
    nervous = pa.compute(_sig([("Q1", 0.35), ("Q2", 0.38), ("Q3", 0.33)], head="elevated"))
    spike = pa.compute(_sig([("Q1", 0.05), ("Q2", 0.60), ("Q3", 0.05)]))
    assert nervous.score < spike.score
    assert nervous.level != "HIGH" and any("nervousness" in e.message for e in nervous.evidence)
    q2 = [e for e in spike.evidence if e.question_id == "Q2"]
    assert q2 and "Q2" in q2[0].message and "first answer" in q2[0].message


def test_gaze_alone_never_exceeds_medium():
    r = pa.compute(_sig([("Q1", 0.0), ("Q2", 1.0), ("Q3", 1.0)], head="high"))
    assert r.level == "MEDIUM" and r.score <= 55


def test_phone_and_second_person_are_strong_with_timestamps():
    obj = {"objects_status": "ok", "multi_person_detected": True, "phone_detected": True,
           "object_events": [{"type": "phone", "t": 120.0, "t_end": 130.0, "question_id": "Q2"},
                             {"type": "multi_person", "t": 200.0, "t_end": 215.0, "question_id": "Q3"}]}
    r = pa.compute(_sig([("Q1", 0.05), ("Q2", 0.05), ("Q3", 0.05)], objects=obj))
    assert r.level == "HIGH" and r.score >= 85 and r.confidence >= 0.85
    phone = [e for e in r.evidence if e.type == "phone"][0]
    assert phone.timestamp == 120.0 and phone.question_id == "Q2"


def test_objects_only_when_gaze_insufficient():
    s = {"gaze_status": "insufficient_frames", "gaze_coverage": 0.1, "objects_status": "ok",
         "multi_person_detected": False, "phone_detected": True,
         "object_events": [{"type": "phone", "t": 5.0, "t_end": 9.0, "question_id": None}]}
    r = pa.compute(s)
    assert r.level == "HIGH" and r.status == "OK"


def test_nothing_usable_is_insufficient_and_model_missing_is_unavailable():
    assert pa.compute({"gaze_status": "insufficient_frames", "objects_status": "no_frames"}).status == "INSUFFICIENT_EVIDENCE"
    assert pa.compute({"status_reason": "model_unavailable"}).status == "UNAVAILABLE"
    assert pa.compute(None).status == "INSUFFICIENT_EVIDENCE"
