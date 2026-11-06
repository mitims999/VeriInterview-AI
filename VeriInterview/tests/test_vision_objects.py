"""Tests for backend/vision/objects.py (M3 task P-5): persistence rule + model smoke test."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("cv2")
from backend.vision import objects as O  # noqa: E402
from backend.vision.media import Frame, QuestionSegment  # noqa: E402


def test_single_sample_detection_is_not_an_alert():
    t = [float(i) for i in range(10)]
    assert O._persistent_runs(t, [False, True, False, False, True, True, True, False, False, False], 2) == [(4.0, 6.0)]
    assert O._persistent_runs(t, [True] + [False] * 9, 2) == []


def test_analyze_uses_persistence_and_reports_events(monkeypatch):
    from backend.config import GAZE_THRESHOLDS
    assert GAZE_THRESHOLDS.object_persist_samples == 3
    seq = {0: [], 1: [{"cls": O.PHONE, "score": 0.9, "box": (0, 0, 5, 5)}], 2: [], 3: [],
           4: [{"cls": O.PERSON, "score": 0.9, "box": (0, 0, 9, 9)}, {"cls": O.PERSON, "score": 0.8, "box": (50, 0, 9, 9)}],
           5: [{"cls": O.PERSON, "score": 0.9, "box": (0, 0, 9, 9)}, {"cls": O.PERSON, "score": 0.8, "box": (50, 0, 9, 9)}],
           6: [{"cls": O.PERSON, "score": 0.9, "box": (0, 0, 9, 9)}, {"cls": O.PERSON, "score": 0.8, "box": (50, 0, 9, 9)}],
           7: [], 8: [], 9: []}                                                  # 3 consecutive multi-person samples
    frames = [Frame(i, float(i), np.zeros((4, 4, 3), np.uint8)) for i in range(10)]
    it = iter(seq[i] for i in range(10))
    monkeypatch.setattr(O, "detect", lambda img: next(it))
    monkeypatch.setattr(O, "get_net", lambda: (None, None, 416, "fake"))
    out = O.analyze(frames, [QuestionSegment("Q1", 0, 3), QuestionSegment("Q2", 3, 10)])
    assert out["multi_person_detected"] and not out["phone_detected"]       # lone phone frame ignored
    ev = out["object_events"][0]
    assert ev["type"] == "multi_person" and ev["t"] == 4.0 and ev["question_id"] == "Q2"


def test_real_model_sees_nothing_in_a_blank_frame():
    from backend.vision.face_track import ModelUnavailable
    try:
        O.get_net()
    except ModelUnavailable:
        pytest.skip("YOLO weights not downloaded")
    assert O.detect(np.zeros((416, 416, 3), np.uint8)) == []
