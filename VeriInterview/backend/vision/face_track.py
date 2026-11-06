"""
Shared face detection + tracking for all M3 analyzers (decision D2).

One InsightFace `buffalo_l` model pack (SCRFD-10GF detector + ArcFace R50
embeddings + 3D-68 landmarks for head pose) replaces SyncNet's S3FD detector
and Proctoring-AI's OpenCV-DNN detector, so each frame is detected once.
Embeddings (~130 ms/face on CPU) and 3D pose (~40 ms) only run when the
caller asks for them; detection alone is ~50 ms/frame at det_size 320.

The IoU tracker is a port of joonson/syncnet_python `run_pipeline.py::track_shot`
(MIT): a detection joins the track whose last box overlaps it with IoU > 0.5,
as long as the track has not been missing for longer than `max_gap_s`.

Quality gating (config.FACE_QUALITY) marks faces that are too small,
low-confidence, turned away or motion-blurred relative to the candidate's
own median. Callers skip those frames instead of turning them into risk
(R1 rule 4). Uniform blur is NOT excluded, because it is a deepfake artifact.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from backend.config import FACE_QUALITY, MODELS_DIR, VISION_BUDGET
from backend.vision.media import Frame


class ModelUnavailable(RuntimeError):
    """Vision dependency or weights missing; adapters report UNAVAILABLE instead of failing."""


@dataclass
class FaceObs:
    t: float
    frame_index: int
    bbox: np.ndarray                     # x1, y1, x2, y2 (pixels, in the analysed frame)
    kps: Optional[np.ndarray]            # 5 x 2 keypoints (eyes, nose, mouth corners)
    det_score: float
    sharpness: float                     # variance of Laplacian on the 112 px face crop
    embedding: Optional[np.ndarray] = None   # 512-d L2-normalised ArcFace embedding (if requested)
    pitch: Optional[float] = None            # degrees, from the 3D-68 landmarks (if requested)
    yaw: Optional[float] = None
    roll: Optional[float] = None
    quality_ok: bool = True
    quality_issues: tuple[str, ...] = ()

    @property
    def size(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


@dataclass
class Track:
    track_id: int
    obs: list[FaceObs] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.obs[0].t

    @property
    def end(self) -> float:
        return self.obs[-1].t

    @property
    def mean_size(self) -> float:
        return float(np.mean([o.size for o in self.obs]))

    def good(self) -> list[FaceObs]:
        return [o for o in self.obs if o.quality_ok]


@dataclass
class TrackResult:
    tracks: list[Track]
    frames_sampled: int
    frames_with_face: int
    frames_multi_face: int
    sample_fps: float
    primary_track_id: Optional[int]

    @property
    def face_lost_pct(self) -> float:
        return 1.0 - self.frames_with_face / self.frames_sampled if self.frames_sampled else 1.0

    @property
    def multi_face_pct(self) -> float:
        return self.frames_multi_face / self.frames_sampled if self.frames_sampled else 0.0

    @property
    def primary(self) -> Optional[Track]:
        return next((t for t in self.tracks if t.track_id == self.primary_track_id), None)


# ---------------------------------------------------------------- model

_APP = None
_APP_LOCK = threading.Lock()


def get_face_app():
    """Lazy, process-wide InsightFace app (CPU). Raises ModelUnavailable if deps/weights are missing."""
    global _APP
    with _APP_LOCK:
        if _APP is None:
            root = os.path.join(MODELS_DIR, "insightface")
            if not os.path.isdir(os.path.join(root, "models", "buffalo_l")):
                raise ModelUnavailable("InsightFace buffalo_l weights missing: run python scripts/download_models.py")
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise ModelUnavailable("insightface not installed: pip install -r requirements-vision.txt") from e
            app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"],
                               allowed_modules=["detection", "recognition", "landmark_3d_68"])
            size = VISION_BUDGET.face_det_size
            app.prepare(ctx_id=-1, det_size=(size, size))
            _APP = app
        return _APP


def sharpness(image: np.ndarray, bbox: np.ndarray) -> float:
    """Variance of Laplacian on the face crop resized to 112x112 (scale-normalised)."""
    import cv2
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    gray = cv2.cvtColor(cv2.resize(image[y1:y2, x1:x2], (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect(frame: Frame, app=None, embed: bool = False, pose: bool = False) -> list[FaceObs]:
    """All faces in one frame, largest first. `embed` / `pose` run the extra models per face."""
    from insightface.app.common import Face
    app = app or get_face_app()
    bboxes, kpss = app.det_model.detect(frame.image, max_num=0, metric="default")
    out = []
    for i in range(bboxes.shape[0]):
        bbox = np.asarray(bboxes[i, :4], np.float32)
        kps = None if kpss is None else np.asarray(kpss[i], np.float32)
        obs = FaceObs(t=frame.t, frame_index=frame.index, bbox=bbox, kps=kps,
                      det_score=float(bboxes[i, 4]), sharpness=sharpness(frame.image, bbox))
        if (embed or pose) and kps is not None:
            face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            if embed:
                app.models["recognition"].get(frame.image, face)
                obs.embedding = np.asarray(face.normed_embedding, np.float32)
            if pose:
                app.models["landmark_3d_68"].get(frame.image, face)
                obs.pitch, obs.yaw, obs.roll = (float(v) for v in face.pose)
        out.append(obs)
    out.sort(key=lambda o: -o.size)
    return out


def assess_track_quality(track: Track) -> None:
    """Set quality_ok / quality_issues on every observation, relative to the track's own sharpness."""
    q = FACE_QUALITY
    ref = float(np.median([o.sharpness for o in track.obs])) if track.obs else 0.0
    for o in track.obs:
        issues = []
        if o.det_score < q.min_det_score:
            issues.append("low_confidence")
        if o.size < q.min_face_px:
            issues.append("too_small")
        if o.yaw is not None and abs(o.yaw) > q.max_abs_yaw_deg:
            issues.append("turned_away")
        if o.sharpness < q.min_sharpness_abs or o.sharpness < q.min_sharpness_rel * ref:
            issues.append("blurry")
        o.quality_issues, o.quality_ok = tuple(issues), not issues


def embed_photo(image: np.ndarray, app=None) -> dict:
    """Embedding of the main face in a still reference photo (task I-4).

    Returns {"embedding", "det_score", "face_px", "faces"} or {"error": reason}.
    The largest face is used; extra faces are reported so the caller can warn.
    """
    if image is None or image.ndim != 3:
        return {"error": "unreadable_image"}
    faces = detect(Frame(index=0, t=0.0, image=image), app or get_face_app(), embed=True)
    if not faces or faces[0].embedding is None:
        return {"error": "no_face"}
    main = faces[0]
    if main.det_score < FACE_QUALITY.min_det_score or main.size < FACE_QUALITY.min_face_px:
        return {"error": "face_too_small_or_unclear", "det_score": main.det_score, "face_px": main.size}
    return {"embedding": main.embedding, "det_score": main.det_score, "face_px": main.size, "faces": len(faces)}


# ---------------------------------------------------------------- tracking

def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class IouTracker:
    """Greedy IoU tracker (port of syncnet_python track_shot)."""

    def __init__(self, iou_threshold: float = 0.5, max_gap_s: float = 1.0):
        self.iou_threshold, self.max_gap_s = iou_threshold, max_gap_s
        self.tracks: list[Track] = []

    def update(self, faces: Iterable[FaceObs]) -> None:
        faces = list(faces)
        if not faces:
            return
        t = faces[0].t
        live = [tr for tr in self.tracks if t - tr.end <= self.max_gap_s]
        pairs = sorted(((iou(tr.obs[-1].bbox, f.bbox), ti, fi) for ti, tr in enumerate(live) for fi, f in enumerate(faces)),
                       reverse=True)
        used_t, used_f = set(), set()
        for score, ti, fi in pairs:
            if score <= self.iou_threshold:
                break
            if ti in used_t or fi in used_f:
                continue
            live[ti].obs.append(faces[fi])
            used_t.add(ti)
            used_f.add(fi)
        for fi, f in enumerate(faces):
            if fi not in used_f:
                self.tracks.append(Track(track_id=len(self.tracks), obs=[f]))


def pick_primary(tracks: list[Track]) -> Optional[int]:
    """The candidate: the track with the most observations, ties broken by larger average face size."""
    if not tracks:
        return None
    return max(tracks, key=lambda tr: (len(tr.obs), tr.mean_size)).track_id


def track_frames(frames: Iterable[Frame], sample_fps: float, app=None, embed: bool = False,
                 pose: bool = False, max_gap_s: float = 1.0) -> TrackResult:
    app = app or get_face_app()
    tracker = IouTracker(max_gap_s=max_gap_s)
    n = with_face = multi = 0
    for frame in frames:
        n += 1
        faces = detect(frame, app, embed=embed, pose=pose)
        with_face += bool(faces)
        multi += sum(f.det_score >= FACE_QUALITY.min_det_score for f in faces) >= 2
        tracker.update(faces)
    for tr in tracker.tracks:
        assess_track_quality(tr)
    return TrackResult(tracks=tracker.tracks, frames_sampled=n, frames_with_face=with_face, frames_multi_face=multi,
                       sample_fps=sample_fps, primary_track_id=pick_primary(tracker.tracks))


def track_video(path: str, sample_fps: float = 2.0, embed: bool = False, pose: bool = False,
                max_width: Optional[int] = None, start: float = 0.0, end: Optional[float] = None) -> TrackResult:
    """Detect + track faces through a video at `sample_fps`, capped at VISION_BUDGET.max_seconds_analyzed."""
    from backend.vision import media
    info = media.probe(path)
    limit = start + VISION_BUDGET.max_seconds_analyzed
    end = min(end if end is not None else (info.duration or limit), limit)
    frames = media.iter_frames(path, fps=sample_fps, max_width=max_width or VISION_BUDGET.max_frame_width,
                               start=start, end=end, info=info)
    return track_frames(frames, sample_fps, embed=embed, pose=pose)
