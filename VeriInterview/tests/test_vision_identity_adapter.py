"""
Tests for the real-video path of adapters/insightface_adapter.py (M3 task I-3).
The legacy fixture path is covered by test_unit_modules / test_e2e_demo_scenarios.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.adapters import insightface_adapter as ia


def _sig(**kw):
    base = {"face_lost_pct": 0.02, "multi_face_pct": 0.0, "frames_sampled": 120, "min_frame_similarity_to_anchor": 0.9,
            "mean_similarity": 0.95, "below_threshold_pct": 0.0, "distinct_identities": 1, "anchor_quality": 0.95,
            "anchor_method": "enrollment_window", "good_frames": 115, "good_frame_pct": 0.96, "drift_events": [],
            "status_reason": None}
    return {**base, **kw}


def test_consistent_session_is_low_with_high_confidence():
    r = ia.compute(_sig())
    assert r.level == "LOW" and r.status == "OK" and r.confidence >= 0.85
    assert r.evidence[0].severity == "low"


def test_partial_mismatch_scales_through_medium_to_high():
    assert ia.compute(_sig(below_threshold_pct=0.03)).level == "LOW"
    assert ia.compute(_sig(below_threshold_pct=0.10)).level == "MEDIUM"
    assert ia.compute(_sig(below_threshold_pct=0.30)).level == "HIGH"


def test_identity_switch_is_high_with_timestamped_evidence():
    ev = [{"t": 312.5, "t_end": 600.0, "question_id": "Q4", "similarity": 0.05}]
    r = ia.compute(_sig(distinct_identities=2, below_threshold_pct=0.45, drift_events=ev))
    assert r.level == "HIGH" and r.score >= 75
    drift = [e for e in r.evidence if e.type == "identity_drift"][0]
    assert drift.question_id == "Q4" and drift.timestamp == 312.5 and "312.5s" in drift.message
    assert any(e.type == "identity_switch" for e in r.evidence)


def test_poor_visibility_lowers_confidence_not_risk():
    good = ia.compute(_sig())
    poor = ia.compute(_sig(face_lost_pct=0.4, good_frame_pct=0.3, good_frames=12, anchor_quality=0.5))
    assert poor.score == good.score
    assert poor.confidence < good.confidence - 0.3
    assert any(e.type == "face_lost" and "not counted as risk" in e.message for e in poor.evidence)


def test_status_reasons_map_to_unavailable_or_insufficient():
    assert ia.compute(None).status == "INSUFFICIENT_EVIDENCE"
    assert ia.compute({"status_reason": "no_face", "face_lost_pct": 1.0}).status == "INSUFFICIENT_EVIDENCE"
    assert ia.compute({"status_reason": "insufficient_good_frames"}).status == "INSUFFICIENT_EVIDENCE"
    assert ia.compute({"status_reason": "media_error"}).status == "INSUFFICIENT_EVIDENCE"
    m = ia.compute({"status_reason": "model_unavailable", "detail": "x"})
    assert m.status == "UNAVAILABLE" and m.level == "UNAVAILABLE" and m.score == 0


def test_analyze_video_reports_bad_file_instead_of_raising(tmp_path):
    bad = tmp_path / "junk.mp4"
    bad.write_bytes(b"not a video")
    s = ia.analyze_video(str(bad))
    assert s["status_reason"] in ("media_error", "model_unavailable")
    assert ia.compute(s).status in ("INSUFFICIENT_EVIDENCE", "UNAVAILABLE")


def test_reference_mismatch_is_high_even_when_session_is_consistent():
    r = ia.compute(_sig(reference_status="ok", reference_similarity=0.12, reference_match_pct=0.0))
    assert r.level == "HIGH" and r.score >= 70
    assert r.evidence[0].type == "reference_mismatch" and "reference photo" in r.evidence[0].message


def test_reference_weak_match_is_medium_and_good_match_stays_low():
    weak = ia.compute(_sig(reference_status="ok", reference_similarity=0.38, reference_match_pct=0.6))
    assert weak.level == "MEDIUM" and weak.evidence[0].type == "reference_weak_match"
    good = ia.compute(_sig(reference_status="ok", reference_similarity=0.72, reference_match_pct=1.0))
    assert good.level == "LOW" and any(e.type == "reference_match" for e in good.evidence)


def test_unusable_reference_photo_adds_no_risk():
    base = ia.compute(_sig())
    r = ia.compute(_sig(reference_status="no_face"))
    assert r.score == base.score and any(e.type == "reference_photo" for e in r.evidence)
