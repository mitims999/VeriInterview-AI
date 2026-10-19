"""
Tests for the real-video path of adapters/syncnet_adapter.py (M3 task S-4).
Signal values mirror the S-3 GRID calibration (docs/m3/calibration/syncnet_grid.md).
The legacy fixture path is covered by test_unit_modules / test_e2e_demo_scenarios.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.adapters import syncnet_adapter as sa


def _seg(i, offset, lse_c, q=None):
    return {"start": i * 10.0, "end": i * 10.0 + 9.5, "offset": offset, "lse_c": lse_c, "n_frames": 237, "question_id": q}


def _sig(offsets, lse, offset=None, stability=1.0, speech=36.0, qs=None):
    segs = [_seg(i, o, l, (qs or [None] * len(offsets))[i]) for i, (o, l) in enumerate(zip(offsets, lse))]
    return {"av_offset_frames": offsets[0] if offset is None else offset, "lse_c": sum(lse) / len(lse),
            "offset_stability": stability, "speech_seconds_analyzed": speech, "per_segment": segs, "status_reason": None}


def test_genuine_is_low_with_good_confidence():
    r = sa.compute(_sig([1, 1, 2, 1], [5.9, 6.1, 5.7, 6.0]))
    assert r.level == "LOW" and r.confidence > 0.9 and r.status == "OK"


def test_offset_bands():
    assert sa.compute(_sig([4, 4, 4, 4], [6, 6, 6, 6])).level == "MEDIUM"
    assert sa.compute(_sig([11, 11, 12, 11], [6, 6, 6, 6])).level == "HIGH"


def test_low_quality_camera_lowers_confidence_but_not_risk():
    r = sa.compute(_sig([1, 1, 1, 2], [0.9, 1.2, 1.2, 1.4], stability=1.0))
    assert r.level == "LOW" and r.confidence <= 0.3
    assert any(e.type == "sync_confidence" and "not counted as risk" in e.message for e in r.evidence)


def test_audio_swap_low_confidence_and_unstable_is_high():
    r = sa.compute(_sig([4, -6, 12, 5], [1.35, 1.1, 1.16, 1.5], offset=4, stability=0.5))
    assert r.level == "HIGH" and any(e.type == "av_no_sync" for e in r.evidence)
    assert "not come from the person on camera" in [e for e in r.evidence if e.type == "av_no_sync"][0].message


def test_partial_manipulation_flags_the_answer_with_time_and_question():
    r = sa.compute(_sig([1, -6, -6, 2], [4.6, 4.7, 5.0, 6.3], offset=1, stability=0.5, qs=["Q1", "Q2", "Q2", "Q3"]))
    assert r.level == "HIGH"
    segs = [e for e in r.evidence if e.type == "av_offset_segment"]
    assert len(segs) == 2 and segs[0].question_id == "Q2" and segs[0].timestamp == 10.0


def test_short_speech_reduces_confidence():
    long = sa.compute(_sig([1, 1], [6, 6], speech=36.0))
    short = sa.compute(_sig([1, 1], [6, 6], speech=8.0))
    assert short.score == long.score and short.confidence < long.confidence


def test_status_reasons():
    for reason in ("no_audio", "no_speech", "no_face", "too_short", "media_error"):
        r = sa.compute({"status_reason": reason})
        assert r.status == "INSUFFICIENT_EVIDENCE" and r.score == 0
    assert sa.compute({"status_reason": "model_unavailable"}).status == "UNAVAILABLE"
    assert sa.compute(None).status == "INSUFFICIENT_EVIDENCE"


def test_evidence_never_claims_a_deepfake_verdict():
    r = sa.compute(_sig([4, -6, 12, 5], [1.35, 1.1, 1.16, 1.5], offset=4, stability=0.5))
    assert not any("deepfake" in e.message.lower() for e in r.evidence)
