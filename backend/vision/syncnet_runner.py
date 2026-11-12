"""
Runs SyncNet over the speech in an interview video (M3 task S-2) and builds
the `av_signals` dict consumed by adapters/syncnet_adapter.py.

Pipeline per speech window (port of syncnet_python run_pipeline.py, MIT):
  1. Decode the window at 25 fps.
  2. Detect faces with the shared SCRFD detector on every Nth frame, take the
     largest face, linearly interpolate the box in between (upstream also
     interpolates tracks), median-smooth centre and size (kernel 13).
  3. Crop exactly like `crop_video`: square of half-size bs around the centre,
     padded by bs*(1+2*crop_scale), resized to 224x224.
  4. Score the crops + the window's 16 kHz audio with SyncNet.

Windows come from speech segments (VAD or Whisper), merged across short
pauses, at least `syncnet_min_segment_s` long, split at
`syncnet_max_window_s`, and capped at `syncnet_max_seconds_total` per video
(spread evenly over the interview). Windows with no usable face are skipped
and reported, never scored.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from backend.config import SYNC_THRESHOLDS, VISION_BUDGET, VAD_SETTINGS
from backend.vision import media
from backend.vision.media import QuestionSegment, Segment, question_at

FPS = 25


def speech_windows(speech: Sequence[Segment], merge_gap_s: float = 2.0) -> list[Segment]:
    """Merge speech across short pauses, drop short bits, split long stretches."""
    b = VISION_BUDGET
    merged: list[list[float]] = []
    for s in speech:
        if merged and s.start - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], s.end)
        else:
            merged.append([s.start, s.end])
    out = []
    for s, e in merged:
        if e - s < b.syncnet_min_segment_s:
            continue
        n = int(np.ceil((e - s) / b.syncnet_max_window_s))
        step = (e - s) / n
        out += [Segment(s + i * step, s + (i + 1) * step) for i in range(n)]
    return out


def budget_windows(windows: Sequence[Segment], max_total_s: float,
                   questions: Sequence[QuestionSegment] = ()) -> list[Segment]:
    """Keep at most max_total_s of windows.

    With question timings, every answer gets an equal share (at least one
    window of >= syncnet_min_segment_s), so one manipulated answer cannot be
    skipped. Without them, windows are spread evenly over the interview.
    """
    if questions:
        share = max(VISION_BUDGET.syncnet_min_segment_s, max_total_s / len(questions))
        keep = []
        for q in questions:
            used = 0.0
            for w in windows:
                mid = (w.start + w.end) / 2
                if q.start <= mid < q.end and used < share:
                    take = min(w.duration, share - used)
                    if take >= VISION_BUDGET.syncnet_min_segment_s:
                        keep.append(Segment(w.start, w.start + take))
                        used += take
        return keep
    total = sum(w.duration for w in windows)
    if total <= max_total_s:
        return list(windows)
    keep, used = [], 0.0
    stride = total / max_total_s
    for i, w in enumerate(windows):
        if i % max(1, int(round(stride))) == 0 and used + w.duration <= max_total_s:
            keep.append(w)
            used += w.duration
    return keep


def crop_faces(frames: list[np.ndarray], boxes: np.ndarray, crop_scale: float) -> np.ndarray:
    """Upstream crop_video geometry. boxes: T x 4 (x1, y1, x2, y2), already smoothed."""
    import cv2
    out = np.empty((len(frames), 224, 224, 3), np.uint8)
    for i, (img, (x1, y1, x2, y2)) in enumerate(zip(frames, boxes)):
        bs = max(y2 - y1, x2 - x1) / 2
        bsi = int(bs * (1 + 2 * crop_scale))
        padded = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = (y1 + y2) / 2 + bsi, (x1 + x2) / 2 + bsi
        face = padded[int(my - bs):int(my + bs * (1 + 2 * crop_scale)), int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
        out[i] = cv2.resize(face, (224, 224))
    return out


def face_boxes(frames: list[np.ndarray], det_every: int, app=None) -> Optional[np.ndarray]:
    """Largest-face box per frame (detected every Nth frame, interpolated, median-smoothed). None if mostly faceless."""
    from scipy import signal
    from backend.vision import face_track
    from backend.vision.media import Frame
    app = app or face_track.get_face_app()
    idx, det = [], []
    for i in range(0, len(frames), det_every):
        faces = face_track.detect(Frame(i, i / FPS, frames[i]), app)
        if faces:
            idx.append(i)
            det.append(faces[0].bbox)
    n_checked = len(range(0, len(frames), det_every))
    if not idx or len(idx) < 0.5 * n_checked:
        return None
    det = np.array(det)
    t = np.arange(len(frames))
    # centre x, centre y, half-size: interpolate then smooth like upstream
    cx = np.interp(t, idx, (det[:, 0] + det[:, 2]) / 2)
    cy = np.interp(t, idx, (det[:, 1] + det[:, 3]) / 2)
    hs = np.interp(t, idx, np.maximum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) / 2)
    if len(frames) >= 13:
        cx, cy, hs = (signal.medfilt(v, kernel_size=13) for v in (cx, cy, hs))
    return np.stack([cx - hs, cy - hs, cx + hs, cy + hs], axis=1)


def score_window(path: str, w: Segment, info: media.VideoInfo, app=None) -> dict:
    from backend.vision import syncnet
    frames = [f.image for f in media.iter_frames(path, fps=FPS, start=w.start, end=w.end,
                                                 max_width=VISION_BUDGET.max_frame_width, info=info)]
    boxes = face_boxes(frames, VISION_BUDGET.syncnet_det_every, app)
    if boxes is None:
        return {"start": round(w.start, 2), "end": round(w.end, 2), "skipped": "no_face"}
    crops = crop_faces(frames, boxes, VISION_BUDGET.syncnet_crop_scale)
    audio = media.load_audio(path, start=w.start, end=w.end, info=info)
    s = syncnet.score(crops, audio, vshift=SYNC_THRESHOLDS.vshift)
    return {"start": round(w.start, 2), "end": round(w.end, 2), "offset": s.offset, "lse_c": round(s.confidence, 3),
            "lse_d": round(s.min_dist, 3), "n_frames": s.n_frames}


def aggregate(segments: list[dict], speech_s: float) -> dict:
    """Frame-weighted summary across scored windows."""
    cfg = SYNC_THRESHOLDS
    scored = [s for s in segments if "offset" in s]
    if not scored:
        return {"status_reason": "no_face", "per_segment": segments, "speech_seconds_detected": round(speech_s, 1)}
    w = np.array([s["n_frames"] for s in scored], float)
    offs = np.array([s["offset"] for s in scored], float)
    order = np.argsort(offs)
    cum = np.cumsum(w[order])
    median_off = int(offs[order][np.searchsorted(cum, cum[-1] / 2)])
    lse_c = float(np.average([s["lse_c"] for s in scored], weights=w))
    analysed = float(w.sum() / FPS)
    out = {
        "av_offset_frames": median_off,
        "sync_confidence": round(float(np.clip(lse_c / cfg.lse_c_good, 0.0, 1.0)), 3),  # legacy 0-1 key
        "lse_c": round(lse_c, 3),
        "lse_d": round(float(np.average([s["lse_d"] for s in scored], weights=w)), 3),
        "offset_stability": round(float(w[np.abs(offs - median_off) <= 1].sum() / w.sum()), 3),
        "speech_seconds_analyzed": round(analysed, 1),
        "speech_seconds_detected": round(speech_s, 1),
        "segments_skipped_no_face": sum(1 for s in segments if s.get("skipped") == "no_face"),
        "per_segment": segments,
        "status_reason": None,
    }
    if analysed < cfg.min_speech_seconds:
        out["status_reason"] = "too_short"
    return out


def analyze(path: str, speech: Optional[Sequence[Segment]] = None,
            questions: Sequence[QuestionSegment] = (), app=None) -> dict:
    """av_signals for one video. `speech`: Whisper/VAD segments (computed with the energy VAD if None)."""
    info = media.probe(path)
    if not info.has_audio:
        return {"status_reason": "no_audio"}
    if speech is None:
        audio = media.load_audio(path, info=info)
        v = VAD_SETTINGS
        speech = media.detect_speech(audio, rel_threshold=v.rel_threshold, min_snr_db=v.min_snr_db,
                                     merge_gap_s=v.merge_gap_s, min_speech_s=v.min_speech_s)
    speech_s = sum(s.duration for s in speech)
    if not speech:
        return {"status_reason": "no_speech", "speech_seconds_detected": 0.0}
    windows = budget_windows(speech_windows(speech), VISION_BUDGET.syncnet_max_seconds_total, questions)
    if not windows:
        return {"status_reason": "too_short", "speech_seconds_detected": round(speech_s, 1)}
    segments = []
    for w in windows:
        seg = score_window(path, w, info, app)
        seg["question_id"] = question_at((w.start + w.end) / 2, questions) if questions else None
        segments.append(seg)
    return aggregate(segments, speech_s)
