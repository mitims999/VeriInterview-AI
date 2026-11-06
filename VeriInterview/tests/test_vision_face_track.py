"""
Unit tests for backend/vision/face_track.py (M3 task G-5).

Tracker, IoU, primary-face choice and quality gating are tested with
synthetic boxes (no model, no data). The last test loads the real InsightFace
pack and skips when the weights / vision deps are not installed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import face_track as ft
from backend.vision.face_track import FaceObs, IouTracker, Track


def _obs(t, x, y=50.0, size=100.0, det=0.9, sharp=200.0, yaw=None):
    return FaceObs(t=t, frame_index=int(t * 10), bbox=np.array([x, y, x + size, y + size], np.float32),
                   kps=None, det_score=det, sharpness=sharp, yaw=yaw)


def test_iou_basic_cases():
    a = np.array([0, 0, 10, 10], np.float32)
    assert ft.iou(a, a) == pytest.approx(1.0)
    assert ft.iou(a, np.array([20, 20, 30, 30], np.float32)) == 0.0
    assert ft.iou(a, np.array([5, 0, 15, 10], np.float32)) == pytest.approx(50 / 150)


def test_tracker_follows_two_people_separately():
    tr = IouTracker()
    for i in range(10):  # candidate drifts slowly, a second person stays put on the right
        tr.update([_obs(i * 0.5, 50 + i * 2), _obs(i * 0.5, 400)])
    assert len(tr.tracks) == 2
    assert sorted(len(t.obs) for t in tr.tracks) == [10, 10]


def test_tracker_starts_new_track_after_long_gap_or_jump():
    tr = IouTracker(max_gap_s=1.0)
    tr.update([_obs(0.0, 50)])
    tr.update([_obs(0.5, 52)])          # same track
    tr.update([_obs(3.0, 54)])          # same place but missing for 2.5 s -> new track
    tr.update([_obs(3.5, 600)])         # big jump -> new track
    assert [len(t.obs) for t in tr.tracks] == [2, 1, 1]


def test_primary_is_longest_track_then_largest():
    long_small = Track(0, [_obs(i, 0, size=80) for i in range(10)])
    short_big = Track(1, [_obs(i, 300, size=200) for i in range(3)])
    assert ft.pick_primary([long_small, short_big]) == 0
    tie_small = Track(2, [_obs(i, 0, size=80) for i in range(5)])
    tie_big = Track(3, [_obs(i, 300, size=150) for i in range(5)])
    assert ft.pick_primary([tie_small, tie_big]) == 3
    assert ft.pick_primary([]) is None


def test_quality_blur_is_relative_so_uniformly_soft_video_is_kept():
    # a deepfake-like video that is soft throughout must NOT be excluded as "blurry"
    soft = Track(0, [_obs(i, 0, sharp=45.0) for i in range(10)])
    ft.assess_track_quality(soft)
    assert all(o.quality_ok for o in soft.obs)
    # one motion-blurred frame inside an otherwise sharp track is excluded
    mixed = Track(1, [_obs(i, 0, sharp=300.0) for i in range(9)] + [_obs(9, 0, sharp=60.0)])
    ft.assess_track_quality(mixed)
    assert [o.quality_ok for o in mixed.obs] == [True] * 9 + [False]
    assert mixed.obs[-1].quality_issues == ("blurry",)


def test_quality_flags_small_low_confidence_and_turned_away_faces():
    tr = Track(0, [_obs(0, 0), _obs(1, 0, size=30), _obs(2, 0, det=0.3), _obs(3, 0, yaw=70.0)])
    ft.assess_track_quality(tr)
    assert [o.quality_issues for o in tr.obs] == [(), ("too_small",), ("low_confidence",), ("turned_away",)]


def test_track_result_percentages():
    r = ft.TrackResult(tracks=[], frames_sampled=10, frames_with_face=8, frames_multi_face=1,
                       sample_fps=2.0, primary_track_id=None)
    assert r.face_lost_pct == pytest.approx(0.2) and r.multi_face_pct == pytest.approx(0.1)
    empty = ft.TrackResult([], 0, 0, 0, 2.0, None)
    assert empty.face_lost_pct == 1.0 and empty.multi_face_pct == 0.0


def test_real_model_finds_no_face_in_blank_frame():
    pytest.importorskip("insightface")
    try:
        app = ft.get_face_app()
    except ft.ModelUnavailable:
        pytest.skip("buffalo_l weights not downloaded")
    from backend.vision.media import Frame
    frame = Frame(index=0, t=0.0, image=np.zeros((240, 320, 3), np.uint8))
    assert ft.detect(frame, app, embed=True, pose=True) == []
