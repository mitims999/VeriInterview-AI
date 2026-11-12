"""
Runs the full M3 vision analysis on one interview video (task N-1) and returns
the signal dicts the adapters consume:

    {"av_signals", "identity_signals", "behavior_signals", "eye_authenticity", "meta"}

One pass over the media, shared between analyzers:
  1. probe + speech segments (given by M1's Whisper path, or the energy VAD)
  2. face tracking at identity_fps with ArcFace + pose  -> identity (InsightFace)
  3-5 run in parallel threads after step 2:
  3. MediaPipe landmarks at landmarks_fps (gated on clear faces from step 2)
     -> gaze/head pose (behaviour) AND blink/mouth cues (eye authenticity)
  4. YOLOv3 person/phone at objects_fps                -> behaviour
  5. SyncNet over speech windows                       -> lip-sync
Each step that cannot run (missing model, no audio, no face) yields a
`status_reason` instead of numbers. Nothing is ever made up (decision D4,
Master Spec 27/28). Results are cached by video SHA-256 (+ reference photo +
config) in storage/features/ (decision D6). Only derived numbers are stored,
never frames or audio.

`derive_av_signals(video_bytes)` keeps the original entry point used by
main.py /api/analyze, now backed by the real analysis.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import tempfile
import time
from typing import Callable, Optional, Sequence

from backend.config import FEATURES_DIR, VAD_SETTINGS, VISION_BUDGET

CACHE_VERSION = "m3-v1"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_key(path: str, reference_photo, questions) -> str:
    h = hashlib.sha256((_sha256_file(path) + CACHE_VERSION + VISION_BUDGET.model_dump_json()).encode())
    if reference_photo is not None:
        data = reference_photo if isinstance(reference_photo, (bytes, bytearray)) else open(reference_photo, "rb").read() \
            if isinstance(reference_photo, (str, os.PathLike)) else bytes(memoryview(reference_photo))
        h.update(hashlib.sha256(data).digest())
    h.update(json.dumps([(q.question_id, q.start, q.end) for q in questions]).encode())
    return h.hexdigest()


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def analyze_video(path: str, questions: Sequence = (), speech: Optional[Sequence] = None,
                  reference_photo=None, use_cache: bool = True, run_objects: bool = True,
                  progress: Optional[Callable[[str, float], None]] = None) -> dict:
    """Full M3 analysis. `progress(stage, fraction)` is called as stages start/finish (for job APIs)."""
    report = progress or (lambda stage, frac: None)
    from backend.adapters import insightface_adapter, syncnet_adapter
    from backend.vision import eye_auth, face_track, gaze, landmarks, media, objects

    t_start = time.time()
    key = _cache_key(path, reference_photo, questions)
    cache_path = os.path.join(FEATURES_DIR, f"{key}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            out = json.load(f)
        out["meta"]["cache_hit"] = True
        return out

    timings: dict = {}
    try:
        info = media.probe(path)
    except media.MediaError as e:
        bad = {"status_reason": "media_error", "detail": str(e)}
        return {"av_signals": bad, "identity_signals": bad, "behavior_signals": bad, "eye_authenticity": bad,
                "meta": {"error": str(e)}}
    end = min(info.duration or VISION_BUDGET.max_seconds_analyzed, VISION_BUDGET.max_seconds_analyzed)

    report("speech", 0.02)
    # 1. speech
    t0 = time.time()
    if speech is None and info.has_audio:
        try:
            v = VAD_SETTINGS
            speech = media.detect_speech(media.load_audio(path, end=end, info=info), rel_threshold=v.rel_threshold,
                                         min_snr_db=v.min_snr_db, merge_gap_s=v.merge_gap_s, min_speech_s=v.min_speech_s)
        except media.MediaError:
            speech = []
    speech = list(speech or [])
    timings["speech"] = round(time.time() - t0, 1)

    report("identity", 0.05)
    # 2. face tracking + identity
    t0 = time.time()
    track = None
    try:
        track = face_track.track_video(path, sample_fps=VISION_BUDGET.identity_fps, embed=True, pose=True, end=end)
        identity_signals = insightface_adapter.analyze_video(path, questions, track_result=track, reference_photo=reference_photo)
    except face_track.ModelUnavailable as e:
        identity_signals = {"status_reason": "model_unavailable", "detail": str(e)}
    timings["identity"] = round(time.time() - t0, 1)

    valid = None
    if track is not None:
        good = sorted(o.t for tr in track.tracks for o in tr.obs if o.quality_ok)
        step = 1.0 / track.sample_fps

        def valid(t, good=good, step=step):         # a clear SCRFD face within one identity sample of t
            i = bisect.bisect_left(good, t - step)
            return i < len(good) and good[i] <= t + step

    # 3-5 are independent once identity has produced the face track: run them in parallel threads
    # (each spends its time in native code: MediaPipe, OpenCV DNN, PyTorch). N-4: ~1.6x faster end to end.
    def landmark_stage():
        t0 = time.time()
        try:
            fps = VISION_BUDGET.landmarks_fps
            stream = [(f.t, o, f.image.shape[:2]) for f, o in landmarks.iter_landmarks(
                media.iter_frames(path, fps=fps, max_width=VISION_BUDGET.max_frame_width, end=end, info=info))]
            res = (gaze.analyze(stream, fps, questions, valid=valid),
                   eye_auth.analyze([(t, o) for t, o, _ in stream], fps, speech, valid=valid))
        except face_track.ModelUnavailable as e:
            res = ({"gaze_status": "model_unavailable"}, {"status_reason": "model_unavailable", "detail": str(e)})
        timings["landmarks"] = round(time.time() - t0, 1)
        report("landmarks_done", 0.0)
        return res

    def object_stage():
        t0 = time.time()
        res = {}
        if run_objects:
            try:
                res = objects.analyze(media.iter_frames(path, fps=VISION_BUDGET.objects_fps,
                                                        max_width=VISION_BUDGET.max_frame_width, end=end, info=info), questions)
            except face_track.ModelUnavailable:
                res = {"objects_status": "model_unavailable"}
        timings["objects"] = round(time.time() - t0, 1)
        report("objects_done", 0.0)
        return res

    def sync_stage():
        t0 = time.time()
        res = {"status_reason": "no_audio"} if not info.has_audio else             syncnet_adapter.analyze_video(path, speech=speech, questions=questions)
        timings["sync"] = round(time.time() - t0, 1)
        report("sync_done", 0.0)
        return res

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    report("gaze_objects_lipsync", 0.35)
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_lm, f_obj, f_sync = pool.submit(landmark_stage), pool.submit(object_stage), pool.submit(sync_stage)
        (gaze_signals, eye_authenticity), object_signals, av_signals = f_lm.result(), f_obj.result(), f_sync.result()
    timings["parallel_wall"] = round(time.time() - t0, 1)
    behavior_signals = {**gaze_signals, **object_signals}
    if behavior_signals.get("gaze_status") == "model_unavailable" and behavior_signals.get("objects_status") in (None, "model_unavailable"):
        behavior_signals = {"status_reason": "model_unavailable"}

    report("finishing", 0.97)
    out = _jsonable({
        "av_signals": av_signals, "identity_signals": identity_signals,
        "behavior_signals": behavior_signals, "eye_authenticity": eye_authenticity,
        "meta": {"duration_s": info.duration, "analysed_s": end, "has_audio": info.has_audio,
                 "speech_s": round(sum(s.duration for s in speech), 1), "timings_s": timings,
                 "total_s": round(time.time() - t_start, 1), "cache_hit": False, "cache_version": CACHE_VERSION},
    })
    if use_cache:
        os.makedirs(FEATURES_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)
    return out


def derive_av_signals(video_bytes: bytes) -> tuple[dict, dict]:
    """Original main.py entry point: (av_signals, identity_signals) for an uploaded video, now from real analysis."""
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
        res = analyze_video(tmp, run_objects=False)
        return res["av_signals"], res["identity_signals"]
    finally:
        os.remove(tmp)
