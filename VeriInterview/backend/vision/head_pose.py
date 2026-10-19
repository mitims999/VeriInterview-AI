"""
Head pose from six facial landmarks (M3 task P-2), ported from
vardanagarwal/Proctoring-AI `head_pose_estimation.py` (MIT).

Kept from upstream: the generic 6-point 3D face model (nose tip, chin, eye
outer corners, mouth corners), a pinhole camera with focal length = image
width and principal point = image centre, no lens distortion, and
cv2.solvePnP. Changed: landmarks come from MediaPipe instead of the dlib-style
68-point model, and the result is true Euler angles (Rodrigues + RQDecomp3x3)
instead of upstream's 2-D line-slope angles with a fixed 48 degree cut-off.

Sign convention (matches landmarks.matrix_to_euler): yaw > 0 = turned to the
image's left, pitch > 0 = looking up, roll > 0 = head tilted counter-clockwise
in the image.

PRIMARY vs FALLBACK (task P-2 measurement, docs/m3/M3_PROGRESS.md): this
generic 6-point fit tracks head motion (yaw correlation 0.76-0.95 with
MediaPipe) but has a systematic yaw bias of +8 to +17 degrees. MediaPipe's own
transformation matrix (468-point fit) agrees with InsightFace's independent
3D-68 pose (AVLips 19.3 vs 19.6 degrees; this fit gives 30.0). So the gaze
module uses `pose_from_obs()`: MediaPipe matrix first, this solvePnP only
when no matrix is available.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Proctoring-AI model points (y up). Order: nose tip, chin, image-left eye corner,
# image-right eye corner, image-left mouth corner, image-right mouth corner.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

# Matching MediaPipe face-mesh indices (subject's right = image left on a webcam).
MP_INDICES = (1, 152, 33, 263, 61, 291)


def solve(points_px: np.ndarray, image_size: tuple[int, int]) -> Optional[tuple[float, float, float]]:
    """(pitch, yaw, roll) degrees from MediaPipe landmarks in pixels; None if solvePnP fails."""
    import cv2
    h, w = image_size
    image_points = np.ascontiguousarray(points_px[list(MP_INDICES), :2], dtype=np.float64)
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    # upstream model is y-up while image y points down: flip y so solvePnP sees one convention
    model = MODEL_POINTS * np.array([1.0, -1.0, -1.0])
    ok, rvec, _ = cv2.solvePnP(model, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    # back to a y-up, z-towards-camera frame to match MediaPipe's transformation matrix convention
    flip = np.diag([1.0, -1.0, -1.0])
    r = flip @ rot @ flip
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def pose_from_obs(obs, image_size: tuple[int, int]) -> Optional[tuple[float, float, float]]:
    """(pitch, yaw, roll) for a landmarks.LandmarkObs: MediaPipe matrix if present, else the solvePnP port."""
    from backend.vision.landmarks import matrix_to_euler
    if getattr(obs, "matrix", None) is not None:
        return matrix_to_euler(obs.matrix)
    return solve(obs.points, image_size)
