"""
Eye-region AUTHENTICITY cues for the deepfake sub-score (M3 task P-4).

Requirement R1 (owner): a nervous genuine candidate must not look like a fake.
Stress RAISES blink rate and causes lip-biting / fidgeting, so both cues here
are one-sided and only fire on patterns typical of synthesised faces:

1. Blinking (MediaPipe eyeBlink blendshapes, cross-checked with the eye-aspect
   ratio from the landmarks):
   - too few blinks: < `too_few_blinks_per_min` over >= `min_eye_visible_s`
     of visible eyes. Face-swap / talking-head generators often under-blink
     (Li et al., "In Ictu Oculi", 2018).
   - metronomic blinking: inter-blink intervals with coefficient of variation
     < `metronomic_ibi_cv` (human blinking is irregular).
   - a HIGH blink rate adds 0 risk; it is reported as a possible stress sign.
2. Mouth vs speech: speech is present (VAD/Whisper) for >= `mouth_mismatch_window_s`
   while the mouth shows no articulation (frozen or replayed face, audio
   dubbed over a still face). Only checked WHILE SPEECH IS PRESENT; mouth
   movement in silence (lip-biting, nervous habits) is ignored.

Gaze direction is NOT used here (decision D3).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from backend.config import EYE_AUTH_THRESHOLDS
from backend.vision.landmarks import LandmarkObs
from backend.vision.media import Segment

# eye-aspect-ratio points (MediaPipe): (corner, corner, upper1, lower1, upper2, lower2)
EAR_EYES = ((33, 133, 160, 144, 158, 153), (362, 263, 385, 380, 387, 373))
INNER_LIP_TOP, INNER_LIP_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291


def ear(points: np.ndarray) -> float:
    """Mean eye-aspect ratio of both eyes (Soukupova & Cech 2016) from MediaPipe landmarks."""
    vals = []
    for c1, c2, u1, l1, u2, l2 in EAR_EYES:
        p = points[:, :2]
        horiz = np.linalg.norm(p[c1] - p[c2])
        vals.append((np.linalg.norm(p[u1] - p[l1]) + np.linalg.norm(p[u2] - p[l2])) / (2.0 * max(horiz, 1e-6)))
    return float(np.mean(vals))


def mouth_open(points: np.ndarray) -> float:
    """Inner-lip gap / mouth width (scale-free)."""
    p = points[:, :2]
    return float(np.linalg.norm(p[INNER_LIP_TOP] - p[INNER_LIP_BOTTOM]) / max(np.linalg.norm(p[MOUTH_LEFT] - p[MOUTH_RIGHT]), 1e-6))


def blink_score(obs: LandmarkObs) -> float:
    return max(obs.blend.get("eyeBlinkLeft", 0.0), obs.blend.get("eyeBlinkRight", 0.0))


def detect_blinks(ts: np.ndarray, closed: np.ndarray, fps: float) -> list[float]:
    """Blink onset times: rising edges of `closed`, merging re-openings shorter than 2 frames."""
    onsets, last_close_end = [], -1e9
    i = 0
    while i < len(closed):
        if closed[i]:
            j = i
            while j < len(closed) and closed[j]:
                j += 1
            if ts[i] - last_close_end > 2.0 / fps:
                onsets.append(float(ts[i]))
            last_close_end = ts[j - 1]
            i = j
        else:
            i += 1
    return onsets


def analyze(stream: Sequence[tuple[float, Optional[LandmarkObs]]], fps: float,
            speech: Sequence[Segment] = (), valid: Optional[Callable[[float], bool]] = None) -> dict:
    """eye_authenticity dict for deepfake_fusion. stream: (t, landmarks or None) per sampled frame."""
    cfg = EYE_AUTH_THRESHOLDS
    obs = [(t, o) for t, o in stream if o is not None and (valid is None or valid(t))]
    out: dict = {"frames_used": len(obs), "eye_visible_s": round(len(obs) / fps, 1)}
    if len(obs) < fps * 10:
        out["status_reason"] = "insufficient_frames"
        return out
    ts = np.array([t for t, _ in obs])
    bs = np.array([blink_score(o) for _, o in obs])
    ears = np.array([ear(o.points) for _, o in obs])
    closed = bs > cfg.blink_closed_score
    onsets = detect_blinks(ts, closed, fps)
    minutes = len(obs) / fps / 60.0
    rate = len(onsets) / minutes if minutes > 0 else 0.0
    ibi = np.diff(onsets)
    cv = float(ibi.std() / ibi.mean()) if len(ibi) >= 5 and ibi.mean() > 0 else None
    # cross-check: blendshape-closed frames should have a clearly lower EAR than open frames
    agree = float(np.median(ears[closed]) / np.median(ears[~closed])) if closed.any() and (~closed).any() else None
    out.update({
        "blink_count": len(onsets), "blink_rate_per_min": round(rate, 1),
        "blink_ibi_cv": None if cv is None else round(cv, 3),
        "blink_ear_ratio_closed_vs_open": None if agree is None else round(agree, 3),
        "blink_judgeable": len(obs) / fps >= cfg.min_eye_visible_s,
    })
    out["too_few_blinks"] = bool(out["blink_judgeable"] and rate < cfg.too_few_blinks_per_min)
    out["metronomic_blinks"] = bool(cv is not None and cv < cfg.metronomic_ibi_cv)
    out["high_blink_rate"] = bool(rate > 30.0)          # stress indicator only, never risk (R1)

    # mouth vs speech: only while speech is present
    if speech:
        mo = np.array([mouth_open(o.points) for _, o in obs])
        win = max(2, int(round(cfg.mouth_mismatch_window_s * fps)))
        speech_frames = mismatch_frames = 0
        for seg in speech:
            idx = np.flatnonzero((ts >= seg.start) & (ts < seg.end))
            speech_frames += len(idx)
            for k in range(0, len(idx) - win + 1, win):
                w = mo[idx[k:k + win]]
                if w.max() - w.min() < 0.04 and w.mean() < 0.08:     # no articulation: static, closed mouth
                    mismatch_frames += win
        out["speech_frames_with_face"] = speech_frames
        out["mouth_speech_mismatch_pct"] = round(mismatch_frames / speech_frames, 3) if speech_frames else None
    return out
