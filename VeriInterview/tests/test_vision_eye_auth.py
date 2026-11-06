"""
Unit tests for backend/vision/eye_auth.py (M3 task P-4) with the R1 rules:
a high blink rate and mouth movement in silence never count; only too-few /
metronomic blinks and a static mouth during speech are flagged.
Synthetic landmark streams; no model, no data.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import eye_auth as E  # noqa: E402
from backend.vision.landmarks import LandmarkObs  # noqa: E402
from backend.vision.media import Segment  # noqa: E402

FPS = 10


def _pts(eye_open=1.0, mouth=0.0):
    p = np.zeros((478, 3), np.float32)
    for c1, c2, u1, l1, u2, l2 in E.EAR_EYES:
        x0 = 100.0 if c1 == 33 else 200.0
        p[c1, :2], p[c2, :2] = (x0, 100), (x0 + 30, 100)
        for u, l, dx in ((u1, l1, 10), (u2, l2, 20)):
            p[u, :2], p[l, :2] = (x0 + dx, 100 - 5 * eye_open), (x0 + dx, 100 + 5 * eye_open)
    p[E.MOUTH_LEFT, :2], p[E.MOUTH_RIGHT, :2] = (130, 200), (190, 200)
    p[E.INNER_LIP_TOP, :2], p[E.INNER_LIP_BOTTOM, :2] = (160, 200 - 30 * mouth), (160, 200 + 30 * mouth)
    return p


def _stream(seconds, blink_times=(), talking=lambda t: False):
    out = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        closed = any(b <= t < b + 0.2 for b in blink_times)
        mouth = 0.15 * abs(np.sin(2 * np.pi * 3 * t)) if talking(t) else 0.0
        out.append((t, LandmarkObs(t=t, frame_index=i, points=_pts(0.15 if closed else 1.0, mouth),
                                   blend={"eyeBlinkLeft": 0.9 if closed else 0.05, "eyeBlinkRight": 0.9 if closed else 0.05},
                                   matrix=None, face_px=150, n_faces=1)))
    return out


def _natural_blinks(seconds, mean_gap=4.0, seed=0):
    rng, t, out = np.random.default_rng(seed), 1.0, []
    while t < seconds - 1:
        out.append(t)
        t += rng.uniform(0.3, 2.0) * mean_gap
    return out


def test_natural_blinking_is_not_flagged_and_ear_agrees():
    r = E.analyze(_stream(120, _natural_blinks(120)), FPS)
    assert 8 <= r["blink_rate_per_min"] <= 30
    assert not r["too_few_blinks"] and not r["metronomic_blinks"]
    assert r["blink_ear_ratio_closed_vs_open"] < 0.5        # blendshape-closed frames really have low EAR


def test_no_blinking_for_two_minutes_is_flagged():
    r = E.analyze(_stream(120, []), FPS)
    assert r["blink_judgeable"] and r["too_few_blinks"]


def test_short_clip_is_not_judged_for_blink_rate():
    r = E.analyze(_stream(30, []), FPS)
    assert not r["blink_judgeable"] and not r["too_few_blinks"]


def test_metronomic_blinking_is_flagged():
    r = E.analyze(_stream(120, [1 + 4.0 * k for k in range(29)]), FPS)
    assert r["metronomic_blinks"]


def test_nervous_fast_blinking_adds_no_flag_r1():
    r = E.analyze(_stream(120, _natural_blinks(120, mean_gap=1.5, seed=3)), FPS)
    assert r["blink_rate_per_min"] > 30 and r["high_blink_rate"]
    assert not r["too_few_blinks"] and not r["metronomic_blinks"]


def test_static_mouth_during_speech_is_flagged_but_silent_mouth_movement_is_ignored():
    speech = [Segment(10, 40)]
    frozen = E.analyze(_stream(60, _natural_blinks(60)), FPS, speech)                         # mouth never moves
    talking = E.analyze(_stream(60, _natural_blinks(60), talking=lambda t: 10 <= t < 40), FPS, speech)
    fidget = E.analyze(_stream(60, _natural_blinks(60), talking=lambda t: t < 10 or t >= 40), FPS, speech)
    assert frozen["mouth_speech_mismatch_pct"] > 0.9
    assert talking["mouth_speech_mismatch_pct"] == 0.0
    assert fidget["mouth_speech_mismatch_pct"] > 0.9 and talking["mouth_speech_mismatch_pct"] == 0.0


def test_invalid_frames_excluded():
    r = E.analyze(_stream(60), FPS, valid=lambda t: t < 5)
    assert r["status_reason"] == "insufficient_frames"
