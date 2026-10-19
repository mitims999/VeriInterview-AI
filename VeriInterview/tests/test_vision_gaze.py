"""
Unit tests for backend/vision/gaze.py (M3 task P-3): the ported Proctoring-AI
eye-ratio rule, blink skipping, head-pose override, debounce, per-question
stats. Synthetic landmarks; no model, no data.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import gaze  # noqa: E402
from backend.vision.landmarks import LandmarkObs  # noqa: E402
from backend.vision.media import QuestionSegment  # noqa: E402


def _points(iris_x=0.5, iris_y=0.5):
    """Two eyes 40 px wide; iris placed at a fraction of the eye width/height."""
    p = np.zeros((478, 3), np.float32)
    for e, x0 in ((gaze.EYES["A"], 100.0), (gaze.EYES["B"], 200.0)):
        left, right = (e["c1"], e["c2"]) if e is gaze.EYES["A"] else (e["c1"], e["c2"])
        p[e["c1"], :2] = (x0, 100) if e is gaze.EYES["A"] else (x0, 100)
        p[e["c2"], :2] = (x0 + 40, 100)
        p[e["top"], :2] = (x0 + 20, 90)
        p[e["bottom"], :2] = (x0 + 20, 110)
        p[e["iris"], :2] = (x0 + 40 * iris_x, 90 + 20 * iris_y)
    return p


def _matrix(pitch=0.0, yaw=0.0):
    p, y = math.radians(pitch), math.radians(yaw)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    return m


def _obs(t, iris_x=0.5, iris_y=0.5, yaw=0.0, pitch=0.0, blink=0.0):
    return LandmarkObs(t=t, frame_index=int(t * 10), points=_points(iris_x, iris_y),
                       blend={"eyeBlinkLeft": blink, "eyeBlinkRight": blink}, matrix=_matrix(pitch, yaw),
                       face_px=150.0, n_faces=1)


def test_upstream_eye_rule_with_both_eyes_agreeing():
    assert gaze.eye_direction(_points(0.5, 0.5)) == "center"
    assert gaze.eye_direction(_points(0.85, 0.5)) == "image_right"     # x_ratio = 0.85/0.15 = 5.7 > 3
    assert gaze.eye_direction(_points(0.15, 0.5)) == "image_left"      # x_ratio = 0.18 < 0.33
    assert gaze.eye_direction(_points(0.5, 0.15)) == "up"              # y_ratio = 0.18 < 0.33
    p = _points(0.85, 0.5)
    p[gaze.EYES["B"]["iris"], 0] = 200 + 20                            # second eye centred -> disagree
    assert gaze.eye_direction(p) == "center"


def test_blink_is_not_read_as_looking_up():
    s = gaze.frame_state(_obs(0, iris_y=0.05, blink=0.9), (480, 640))
    assert s["blink"] and not s["off"]


def test_head_pose_overrides_eyes_including_looking_down():
    assert gaze.frame_state(_obs(0, yaw=40), (480, 640))["direction"] == "image_left"
    assert gaze.frame_state(_obs(0, yaw=-40), (480, 640))["direction"] == "image_right"
    assert gaze.frame_state(_obs(0, pitch=-35), (480, 640))["direction"] == "down"
    assert not gaze.frame_state(_obs(0, yaw=10, pitch=-10), (480, 640))["off"]


def test_debounce_drops_short_glances_keeps_long_ones():
    states = [{"t": i / 10, "off": 3 <= i < 5 or 10 <= i < 20, "direction": "x"} for i in range(30)]
    out = gaze.debounce(states, min_off_s=0.5, fps=10)
    assert [s["off"] for s in out[3:5]] == [False, False]
    assert all(s["off"] for s in out[10:20])


def test_analyze_rates_events_and_per_question_baseline():
    fps = 10
    stream = []
    for i in range(600):                                               # 60 s
        t = i / fps
        looking_away = 40 <= t < 46                                    # one 6 s look to the side in Q2
        stream.append((t, _obs(t, iris_x=0.9 if looking_away else 0.5), (480, 640)))
    qs = [QuestionSegment("Q1", 0, 30), QuestionSegment("Q2", 30, 60)]
    out = gaze.analyze(stream, fps, qs)
    assert out["gaze_status"] == "ok"
    assert out["gaze_off_screen_pct"] == pytest.approx(0.1, abs=0.01)
    assert out["baseline_gaze_off_pct"] == 0.0
    q2 = [q for q in out["per_question"] if q["question_id"] == "Q2"][0]
    assert q2["gaze_off_pct"] == pytest.approx(0.2, abs=0.02) and q2["dominant_direction"] == "image_right"
    assert out["gaze_events"][0]["question_id"] == "Q2" and out["gaze_events"][0]["t"] == pytest.approx(40.0)
    assert out["head_pose_variance"] == "normal"


def test_invalid_frames_are_excluded_and_too_few_frames_reported():
    fps = 10
    stream = [(i / fps, _obs(i / fps), (480, 640)) for i in range(100)]
    out = gaze.analyze(stream, fps, valid=lambda t: t < 2.0)          # face clearly visible only for 2 s
    assert out["gaze_status"] == "insufficient_frames"
    out2 = gaze.analyze(stream + [(10.0 + i / fps, None, (480, 640)) for i in range(100)], fps)
    assert out2["gaze_coverage"] == pytest.approx(0.5)


def test_head_variance_labels():
    fps = 10
    def wobble(amp):  # sine yaw: std = amp / sqrt(2); thresholds elevated 8, high 15 degrees
        return [(i / fps, _obs(i / fps, yaw=amp * math.sin(i / 3)), (480, 640)) for i in range(300)]
    assert gaze.analyze(wobble(5), fps)["head_pose_variance"] == "normal"       # std 3.5
    assert gaze.analyze(wobble(15), fps)["head_pose_variance"] == "elevated"    # std 10.6
    assert gaze.analyze(wobble(28), fps)["head_pose_variance"] == "high"        # std 19.8
