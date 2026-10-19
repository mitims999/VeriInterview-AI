"""
Dense face landmarks for the gaze / head-pose / blink / mouth analyzers
(M3 task P-1), via the MediaPipe Tasks Face Landmarker (Apache-2.0).

Per frame it returns 478 landmarks (468 face mesh + 10 iris points), the
ARKit-style blendshape scores we use (eyeBlink*, jawOpen, mouthClose), and the
facial transformation matrix MediaPipe estimates for head pose.

Runs in VIDEO mode on full frames (MediaPipe has its own face detector and
tracks landmarks between frames, ~10-20 ms/frame on CPU). It keeps up to two
faces and returns the largest, i.e. the candidate (the second one is
reported via `n_faces`).
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np

from backend.config import MODELS_DIR
from backend.vision.face_track import ModelUnavailable
from backend.vision.media import Frame

BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthClose", "eyeLookOutLeft", "eyeLookInLeft",
               "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookOutRight", "eyeLookInRight", "eyeLookUpRight", "eyeLookDownRight")


@dataclass
class LandmarkObs:
    t: float
    frame_index: int
    points: np.ndarray             # 478 x 3: x, y in pixels; z in MediaPipe's relative units
    blend: dict                    # subset of blendshape scores (0-1)
    matrix: Optional[np.ndarray]   # 4 x 4 facial transformation matrix
    face_px: float                 # face height in pixels (landmark extent)
    n_faces: int


def matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from a 4x4 / 3x3 rotation (x-right, y-up, z-towards-camera)."""
    r = np.asarray(m)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch, yaw, roll = math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], sy), math.atan2(r[1, 0], r[0, 0])
    else:
        pitch, yaw, roll = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class Landmarker:
    """Stateful VIDEO-mode landmarker: create one per video (timestamps must increase)."""

    def __init__(self, num_faces: int = 2):
        path = os.path.join(MODELS_DIR, "mediapipe", "face_landmarker.task")
        if not os.path.exists(path):
            raise ModelUnavailable("MediaPipe face_landmarker.task missing: run python scripts/download_models.py")
        try:
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise ModelUnavailable("mediapipe not installed: pip install -r requirements-vision.txt") from e
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path), running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame: Frame) -> Optional[LandmarkObs]:
        import mediapipe as mp
        h, w = frame.image.shape[:2]
        ms = max(self._last_ms + 1, int(round(frame.t * 1000)))
        self._last_ms = ms
        rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.face_landmarks:
            return None
        faces = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lms], np.float32)
            faces.append((float(pts[:, 1].max() - pts[:, 1].min()), i, pts))
        face_px, i, pts = max(faces)
        blend = {}
        if res.face_blendshapes:
            blend = {c.category_name: float(c.score) for c in res.face_blendshapes[i] if c.category_name in BLENDSHAPES}
        matrix = np.asarray(res.facial_transformation_matrixes[i]) if res.facial_transformation_matrixes else None
        return LandmarkObs(t=frame.t, frame_index=frame.index, points=pts, blend=blend, matrix=matrix,
                           face_px=face_px, n_faces=len(faces))


def iter_landmarks(frames: Iterable[Frame]) -> Iterator[tuple[Frame, Optional[LandmarkObs]]]:
    """(frame, landmarks or None) for each frame, with one landmarker for the whole sequence."""
    with Landmarker() as lm:
        for f in frames:
            yield f, lm.process(f)


_LOCK = threading.Lock()  # MediaPipe graphs are not shared across threads; one Landmarker per video/thread
