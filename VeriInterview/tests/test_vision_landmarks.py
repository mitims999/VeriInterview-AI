"""Tests for backend/vision/landmarks.py (M3 task P-1)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import landmarks as L  # noqa: E402


def _rot(pitch=0.0, yaw=0.0, roll=0.0):
    p, y, r = (math.radians(v) for v in (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    return m


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (15, 0, 0), (0, 30, 0), (0, 0, -20), (10, -25, 5)])
def test_matrix_to_euler_round_trips(pitch, yaw, roll):
    got = L.matrix_to_euler(_rot(pitch, yaw, roll))
    assert got == pytest.approx((pitch, yaw, roll), abs=1e-6)


def test_landmarker_on_blank_frame_returns_none():
    pytest.importorskip("mediapipe")
    from backend.vision.face_track import ModelUnavailable
    from backend.vision.media import Frame
    try:
        lm = L.Landmarker()
    except ModelUnavailable:
        pytest.skip("face_landmarker.task not downloaded")
    with lm:
        assert lm.process(Frame(0, 0.0, np.zeros((240, 320, 3), np.uint8))) is None


def test_solvepnp_port_recovers_a_frontal_face_and_a_turn():
    pytest.importorskip("cv2")
    from backend.vision import head_pose as H
    w, h, f = 640, 480, 640.0

    def project(pitch, yaw):
        r = _rot(pitch, yaw, 0)[:3, :3]
        model = H.MODEL_POINTS * np.array([1.0, 1.0, 1.0])
        cam = (r @ model.T).T + np.array([0, 0, -3000.0])            # y-up, camera looks down -z
        pts = np.zeros((478, 3), np.float32)
        for idx, (x, y, z) in zip(H.MP_INDICES, cam):
            pts[idx, :2] = (w / 2 + f * x / -z, h / 2 - f * y / -z)    # image y points down
        return pts

    for pitch, yaw in [(0, 0), (0, 25), (12, -15)]:
        got = H.solve(project(pitch, yaw), (h, w))
        assert got[0] == pytest.approx(pitch, abs=2.0) and got[1] == pytest.approx(yaw, abs=2.0)


def test_pose_from_obs_prefers_the_mediapipe_matrix():
    from backend.vision import head_pose as H
    from backend.vision.landmarks import LandmarkObs
    obs = LandmarkObs(t=0, frame_index=0, points=np.zeros((478, 3), np.float32), blend={}, matrix=_rot(5, 20, 0),
                      face_px=100, n_faces=1)
    assert H.pose_from_obs(obs, (480, 640)) == pytest.approx((5, 20, 0), abs=1e-6)
