"""
Gaze / head-pose behaviour over an interview (M3 task P-3), porting the
eye-direction rule of vardanagarwal/Proctoring-AI `eye_tracker.py` (MIT).

Upstream rule (kept): for each eye, compare the pupil's position between the
eye corners, x_ratio = dist(left corner) / dist(right corner) and
y_ratio = dist(top lid) / dist(bottom lid). x_ratio > 3 or < 0.33 means a
sideways look, y_ratio < 0.33 means looking up, and both eyes must agree.
Changed: the pupil is MediaPipe's iris centre (468 / 473) instead of a
thresholded dark blob, and directions are reported in IMAGE terms.

Additions (measured on GRID / OEP, see docs/m3/M3_PROGRESS.md):
- Blink frames (eyeBlink blendshape > threshold) are skipped for eye
  direction, because closing lids collapse the y_ratio and would read as "up".
- Head pose (MediaPipe matrix, head_pose.pose_from_obs) beyond the yaw/pitch
  limits also counts as facing away, including looking DOWN at notes.
- Debounce: an off-screen state must last >= min_off_screen_s.
- Optional `valid(t)` gate: only frames where the face detector sees a clear
  face count (MediaPipe still fits landmarks on partly covered faces).

Output feeds `behavior_signals` (adapters/proctoring_adapter.py). Gaze is
BEHAVIOUR evidence only. It never enters the deepfake score (decision D3).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence

import numpy as np

from backend.config import EYE_AUTH_THRESHOLDS, GAZE_THRESHOLDS
from backend.vision import head_pose
from backend.vision.landmarks import LandmarkObs
from backend.vision.media import QuestionSegment, question_at

# MediaPipe indices; "A" = image-left eye, "B" = image-right eye (verified: iris 468 sits nearest 33 in 100 % of frames)
EYES = {"A": dict(iris=468, c1=33, c2=133, top=159, bottom=145),
        "B": dict(iris=473, c1=362, c2=263, top=386, bottom=374)}


def eye_ratios(points: np.ndarray) -> dict:
    """{eye: (x_ratio, y_ratio)} with upstream's definitions, in image coordinates."""
    out = {}
    for name, e in EYES.items():
        cx, cy = points[e["iris"], 0], points[e["iris"], 1]
        left, right = sorted((points[e["c1"], 0], points[e["c2"], 0]))
        top, bottom = points[e["top"], 1], points[e["bottom"], 1]
        out[name] = ((cx - left) / max(1e-6, right - cx), (cy - top) / max(1e-6, bottom - cy))
    return out


def eye_direction(points: np.ndarray) -> str:
    """'center' | 'image_right' | 'image_left' | 'up' (both eyes must agree, as upstream)."""
    g = GAZE_THRESHOLDS
    dirs = []
    for xr, yr in eye_ratios(points).values():
        if xr > g.eye_x_ratio_right:
            dirs.append("image_right")
        elif xr < g.eye_x_ratio_left:
            dirs.append("image_left")
        elif yr < g.eye_y_ratio_up:
            dirs.append("up")
        else:
            dirs.append("center")
    return dirs[0] if dirs[0] == dirs[1] else "center"


def is_blink(obs: LandmarkObs) -> bool:
    b = obs.blend
    return max(b.get("eyeBlinkLeft", 0.0), b.get("eyeBlinkRight", 0.0)) > EYE_AUTH_THRESHOLDS.blink_closed_score


def frame_state(obs: LandmarkObs, image_size: tuple[int, int]) -> dict:
    g = GAZE_THRESHOLDS
    pose = head_pose.pose_from_obs(obs, image_size)
    pitch, yaw = (pose[0], pose[1]) if pose else (0.0, 0.0)
    blink = is_blink(obs)
    direction = "center"
    if abs(yaw) > g.head_yaw_off_deg:
        direction = "image_left" if yaw > 0 else "image_right"
    elif pitch > g.head_pitch_off_deg:
        direction = "up"
    elif pitch < -g.head_pitch_off_deg:
        direction = "down"
    elif not blink:
        direction = eye_direction(obs.points)
    return {"t": obs.t, "off": direction != "center", "direction": direction, "yaw": yaw, "pitch": pitch, "blink": blink}


def debounce(states: list[dict], min_off_s: float, fps: float) -> list[dict]:
    """Turn off-screen runs shorter than min_off_s back to on-screen."""
    min_n = max(1, int(round(min_off_s * fps)))
    out = [dict(s) for s in states]
    i = 0
    while i < len(out):
        if out[i]["off"]:
            j = i
            while j < len(out) and out[j]["off"]:
                j += 1
            if j - i < min_n:
                for k in range(i, j):
                    out[k]["off"], out[k]["direction"] = False, "center"
            i = j
        else:
            i += 1
    return out


def _variance_label(yaw_std: float, pitch_std: float) -> str:
    g, s = GAZE_THRESHOLDS, max(yaw_std, pitch_std)
    return "high" if s >= g.head_std_high_deg else ("elevated" if s >= g.head_std_elevated_deg else "normal")


def analyze(stream: Sequence[tuple[float, Optional[LandmarkObs], tuple[int, int]]], fps: float,
            questions: Sequence[QuestionSegment] = (), valid: Optional[Callable[[float], bool]] = None,
            sampled: Optional[int] = None) -> dict:
    """Gaze part of behavior_signals. stream: (t, landmarks or None, (h, w)) per sampled frame."""
    g = GAZE_THRESHOLDS
    states = [frame_state(o, size) for t, o, size in stream if o is not None and (valid is None or valid(t))]
    total = sampled if sampled is not None else len(stream)
    if len(states) < max(10, fps * 5):
        return {"gaze_status": "insufficient_frames", "gaze_coverage": round(len(states) / max(1, total), 3)}
    states = debounce(states, g.min_off_screen_s, fps)
    off = np.array([s["off"] for s in states])
    yaw = np.array([s["yaw"] for s in states])
    pitch = np.array([s["pitch"] for s in states])

    def summary(sub):
        dirs = Counter(s["direction"] for s in sub if s["off"])
        return {"frames": len(sub), "gaze_off_pct": round(float(np.mean([s["off"] for s in sub])), 3),
                "dominant_direction": dirs.most_common(1)[0][0] if dirs else None}

    out = {
        "gaze_status": "ok",
        "gaze_off_screen_pct": round(float(off.mean()), 3),
        "head_pose_variance": _variance_label(float(yaw.std()), float(pitch.std())),
        "yaw_std": round(float(yaw.std()), 2), "pitch_std": round(float(pitch.std()), 2),
        "yaw_mean": round(float(yaw.mean()), 2), "pitch_mean": round(float(pitch.mean()), 2),
        "gaze_coverage": round(len(states) / max(1, total), 3),
        "blink_frame_pct": round(float(np.mean([s["blink"] for s in states])), 3),
    }
    # long off-screen episodes for the evidence timeline (>= 2 s)
    events, i = [], 0
    while i < len(states):
        if states[i]["off"]:
            j = i
            while j < len(states) and states[j]["off"]:
                j += 1
            if states[j - 1]["t"] - states[i]["t"] + 1 / fps >= 2.0:
                d = Counter(s["direction"] for s in states[i:j]).most_common(1)[0][0]
                events.append({"t": round(states[i]["t"], 2), "t_end": round(states[j - 1]["t"], 2), "direction": d,
                               "question_id": question_at(states[i]["t"], questions) if questions else None})
            i = j
        else:
            i += 1
    out["gaze_events"] = events[:20]

    if questions:
        per_q = []
        for q in questions:
            sub = [s for s in states if q.start <= s["t"] < q.end]
            if sub:
                per_q.append({"question_id": q.question_id, **summary(sub)})
        out["per_question"] = per_q
        if per_q:
            out["baseline_gaze_off_pct"] = per_q[0]["gaze_off_pct"]       # candidate's own first answer
    else:
        first = [s for s in states if s["t"] - states[0]["t"] <= 60.0]
        out["baseline_gaze_off_pct"] = summary(first)["gaze_off_pct"]
    return out
