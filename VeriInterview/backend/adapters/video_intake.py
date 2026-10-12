"""
Converts an uploaded interview video file into the `av_signals` /
`identity_signals` shape consumed by `syncnet_adapter.py` and
`insightface_adapter.py`.

INTEGRATION NOTE: this environment cannot decode video frames or run
SyncNet/InsightFace inference (no GPU, no OpenCV/torch model weights
downloaded). This function derives DETERMINISTIC placeholder signals from the
uploaded file's bytes (same file -> same result every time) purely so the
upload slot in the UI is fully wired end-to-end. To go live, replace the body
of `derive_av_signals()` with:
  1. `run_pipeline.py` + `run_syncnet.py` (joonson/syncnet_python) on the saved
     video -> real av_offset_frames / sync_confidence.
  2. InsightFace face embeddings per sampled frame, compared to a first-frame
     anchor -> real min_frame_similarity_to_anchor / face_lost_pct.
No other module needs to change; both adapters already consume this exact
output shape.
"""
from __future__ import annotations
import hashlib


def derive_av_signals(video_bytes: bytes) -> tuple[dict, dict]:
    digest = hashlib.sha256(video_bytes).digest()
    b = list(digest)

    av_offset_frames = b[0] % 10                       # 0-9 frames
    sync_confidence = 0.5 + (b[1] / 255.0) * 0.45       # 0.5-0.95
    min_similarity = 0.70 + (b[2] / 255.0) * 0.29       # 0.70-0.99
    face_lost_pct = (b[3] / 255.0) * 0.2                # 0-0.2

    av_signals = {"av_offset_frames": av_offset_frames, "sync_confidence": round(sync_confidence, 2)}
    identity_signals = {"min_frame_similarity_to_anchor": round(min_similarity, 2), "face_lost_pct": round(face_lost_pct, 2)}
    return av_signals, identity_signals
