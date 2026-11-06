"""
Unit tests for backend/vision/syncnet_runner.py (M3 task S-2): windowing,
budget, upstream crop geometry and aggregation. No model, no data.
Real-model checks on GRID (genuine / shifted audio) are in docs/m3/M3_PROGRESS.md
and the S-3 calibration.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("cv2")
from backend.vision import syncnet_runner as sr  # noqa: E402
from backend.vision.media import Segment  # noqa: E402


def test_speech_windows_merge_pauses_drop_short_and_split_long():
    speech = [Segment(0.5, 2.0), Segment(3.0, 4.5),        # merged across a 1 s pause -> 0.5..4.5
              Segment(10.0, 11.0),                         # isolated 1 s -> dropped (< 2 s)
              Segment(20.0, 45.0)]                         # 25 s -> 3 windows of <= 10 s
    w = sr.speech_windows(speech)
    assert w[0] == Segment(0.5, 4.5)
    assert len(w) == 4 and all(x.duration <= 10.0 + 1e-9 for x in w)
    assert sum(x.duration for x in w[1:]) == pytest.approx(25.0)


def test_budget_keeps_everything_under_cap_and_spreads_when_over():
    short = [Segment(i * 20, i * 20 + 5) for i in range(4)]
    assert sr.budget_windows(short, 60) == short
    many = [Segment(i * 20, i * 20 + 10) for i in range(30)]          # 300 s of speech
    kept = sr.budget_windows(many, 60)
    assert sum(x.duration for x in kept) <= 60
    assert kept[0].start == 0 and kept[-1].start > 100                 # spread, not just the opening


def test_crop_matches_upstream_geometry():
    img = np.zeros((300, 400, 3), np.uint8)
    img[100:200, 150:250] = 255                                        # a 100 px "face"
    crops = sr.crop_faces([img], np.array([[150, 100, 250, 200]], float), crop_scale=0.4)
    assert crops.shape == (1, 224, 224, 3)
    # upstream crop: rows [cy-bs, cy+bs*1.8), cols [cx-bs*1.4, cx+bs*1.4): face fills the top-middle
    c = crops[0, :, :, 0]
    assert c[40, 112] == 255 and c[200, 112] == 0 and c[40, 5] == 0


def test_aggregate_weighted_median_stability_and_legacy_keys():
    segs = [{"offset": 1, "lse_c": 6.0, "lse_d": 7.0, "n_frames": 250},
            {"offset": 2, "lse_c": 5.0, "lse_d": 7.5, "n_frames": 200},
            {"offset": 9, "lse_c": 3.0, "lse_d": 9.0, "n_frames": 50},
            {"start": 0, "end": 3, "skipped": "no_face"}]
    a = sr.aggregate(segs, speech_s=30.0)
    assert a["av_offset_frames"] == 1
    assert a["offset_stability"] == pytest.approx(0.9)                 # 450 of 500 frames within +/-1
    assert a["speech_seconds_analyzed"] == 20.0 and a["segments_skipped_no_face"] == 1
    assert 0 < a["sync_confidence"] <= 1 and a["status_reason"] is None


def test_aggregate_reports_no_face_and_too_short():
    assert sr.aggregate([{"start": 0, "end": 3, "skipped": "no_face"}], 3.0)["status_reason"] == "no_face"
    short = sr.aggregate([{"offset": 0, "lse_c": 6.0, "lse_d": 7.0, "n_frames": 60}], 3.0)
    assert short["status_reason"] == "too_short"


def test_budget_gives_every_question_a_window():
    from backend.vision.media import QuestionSegment
    windows = [Segment(i * 12.0, i * 12.0 + 10.0) for i in range(150)]          # 30 min of continuous answers
    qs = [QuestionSegment(f"Q{k}", k * 180.0, (k + 1) * 180.0) for k in range(10)]
    kept = sr.budget_windows(windows, 30.0, qs)
    assert sum(w.duration for w in kept) <= 30.0 + 1e-9
    covered = {q.question_id for q in qs for w in kept if q.start <= (w.start + w.end) / 2 < q.end}
    assert covered == {q.question_id for q in qs}                                  # no answer skipped
