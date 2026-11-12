"""
M3 video-analysis API (deepfake sub-score + identity, lip-sync, behaviour, eye cues).

Video analysis takes minutes on CPU (a 3-min interview takes ~2.5 min), so it runs as a background job:

    POST   /api/vision/jobs          multipart: video, [reference_photo], [questions JSON]  -> 202 {job_id, ...}
    GET    /api/vision/jobs/{id}     status + stage + progress; full result when status == "done"
    DELETE /api/vision/jobs/{id}     forget the job and its result (privacy)
    GET    /api/vision/health        which models/dependencies are installed (UI can grey out checks)

Privacy (Master Spec 29): the uploaded video and photo exist only as temp files while
the job runs and are deleted afterwards. The result holds derived numbers and plain-English
evidence only. Jobs expire from memory after JOB_TTL_S. One analysis runs at a time (CPU-bound);
others queue and report their queue position.

Every result component is a ModuleResult (score, confidence, level, evidence[], status), as
everywhere else in VeriInterview. Nothing here is a verdict. The deepfake sub-score is
"report-only" (open question Q1 with M5): it does not change the main risk score.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.auth.deps import require_user, require_writable
from backend.config import MODELS_DIR
from backend.database.models import User
from backend.schemas import Evidence, ModuleResult

router = APIRouter(prefix="/api/vision", tags=["vision (M3)"])

ALLOWED_VIDEO = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg"}
ALLOWED_PHOTO = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_PHOTO_BYTES = 10 * 1024 * 1024
JOB_TTL_S = 3600
MAX_JOBS = 50
_job_owner: dict[str, int] = {}   # job_id -> organization_id (jobs are in-memory, like the jobs)

# progress fractions per stage reported by video_intake.analyze_video
_STAGE_FRACTION = {"queued": 0.0, "starting": 0.01, "speech": 0.02, "identity": 0.05,
                   "gaze_objects_lipsync": 0.35, "finishing": 0.97, "done": 1.0}
_STAGE_LABEL = {"queued": "Waiting for another analysis to finish", "starting": "Starting",
                "speech": "Finding speech", "identity": "Checking face identity",
                "gaze_objects_lipsync": "Checking lip-sync, gaze, blinks and surroundings",
                "finishing": "Combining results", "done": "Done", "error": "Failed"}


# ---------------------------------------------------------------- response contract

class QuestionWindow(BaseModel):
    question_id: str
    start: float
    end: float


class VisionResult(BaseModel):
    deepfake: ModuleResult              # the Deepfake Sub-Score card (details.sub_scores = per-check breakdown)
    identity: ModuleResult              # face identity / reference photo
    sync: ModuleResult                  # lip-sync
    eye_authenticity: ModuleResult      # blink pattern + mouth vs speech (weak cues)
    behavior: ModuleResult              # gaze, head pose, phone, extra people (behaviour, not deepfake)
    timeline: list[Evidence]            # all non-low evidence with a timestamp, sorted by time
    meta: dict                          # duration, speech seconds, timings, cache hit


class VisionJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str
    stage_label: str
    progress: float                     # 0-1
    queue_position: int = 0             # jobs ahead of this one (0 = running or next)
    created_at: float
    updated_at: float
    video_filename: Optional[str] = None
    reference_photo: bool = False
    error: Optional[str] = None
    result: Optional[VisionResult] = None


class VisionHealth(BaseModel):
    ready: bool
    models: dict[str, bool]
    dependencies: dict[str, bool]
    checks_available: dict[str, bool]


# ---------------------------------------------------------------- job manager

_jobs: dict[str, VisionJob] = {}
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m3-vision")


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = time.time()


def _expire() -> None:
    now = time.time()
    with _lock:
        for jid in [j for j, job in _jobs.items() if job.status in ("done", "error") and now - job.updated_at > JOB_TTL_S]:
            _jobs.pop(jid, None)
        while len(_jobs) > MAX_JOBS:
            oldest = min((j for j in _jobs.values() if j.status in ("done", "error")), key=lambda j: j.updated_at, default=None)
            if oldest is None:
                break
            _jobs.pop(oldest.job_id, None)


def build_result(vision: dict) -> VisionResult:
    """Adapters + deepfake fusion over video_intake.analyze_video() output."""
    from backend.adapters import insightface_adapter, proctoring_adapter, syncnet_adapter
    from backend.modules.deepfake_fusion import eye_authenticity_result, fuse_deepfake
    identity = insightface_adapter.compute(vision.get("identity_signals"))
    sync = syncnet_adapter.compute(vision.get("av_signals"))
    behavior = proctoring_adapter.compute(vision.get("behavior_signals"))
    eye = eye_authenticity_result(vision.get("eye_authenticity"))
    deepfake = fuse_deepfake(identity, sync, vision.get("eye_authenticity"))
    timeline = sorted((e for m in (identity, sync, behavior, eye) for e in m.evidence
                       if e.timestamp is not None and e.severity != "low"), key=lambda e: e.timestamp)
    return VisionResult(deepfake=deepfake, identity=identity, sync=sync, eye_authenticity=eye, behavior=behavior,
                        timeline=timeline, meta=vision.get("meta", {}))


def _run(job_id: str, video_path: str, photo_bytes: Optional[bytes], questions: list) -> None:
    from backend.adapters import video_intake
    from backend.vision.media import QuestionSegment
    lm_obj_sync_done = {"n": 0}

    def progress(stage: str, frac: float) -> None:
        if stage.endswith("_done"):                    # the three parallel stages finishing
            lm_obj_sync_done["n"] += 1
            p = _STAGE_FRACTION["gaze_objects_lipsync"] + 0.2 * lm_obj_sync_done["n"]
            _update(job_id, progress=round(min(0.95, p), 2))
        else:
            _update(job_id, stage=stage, stage_label=_STAGE_LABEL.get(stage, stage),
                    progress=_STAGE_FRACTION.get(stage, frac))

    try:
        _update(job_id, status="running", stage="starting", stage_label=_STAGE_LABEL["starting"],
                progress=_STAGE_FRACTION["starting"], queue_position=0)
        qs = [QuestionSegment(q["question_id"], float(q["start"]), float(q["end"])) for q in questions]
        vision = video_intake.analyze_video(video_path, questions=qs, reference_photo=photo_bytes, progress=progress)
        _update(job_id, status="done", stage="done", stage_label=_STAGE_LABEL["done"], progress=1.0,
                result=build_result(vision))
    except Exception as e:  # report, never crash the server; details stay in the job, no media logged
        _update(job_id, status="error", stage="error", stage_label=_STAGE_LABEL["error"], error=f"{type(e).__name__}: {e}")
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass
        _refresh_queue_positions()


def _refresh_queue_positions() -> None:
    with _lock:
        queued = sorted((j for j in _jobs.values() if j.status == "queued"), key=lambda j: j.created_at)
        for i, j in enumerate(queued):
            j.queue_position = i + (1 if any(x.status == "running" for x in _jobs.values()) else 0)


# ---------------------------------------------------------------- endpoints

@router.post("/jobs", response_model=VisionJob, status_code=202)
async def create_job(video: UploadFile = File(...), reference_photo: Optional[UploadFile] = File(None),
                     questions: str = Form("[]"), user: User = Depends(require_writable)):
    """Start a video analysis. `questions` (optional): JSON list of {question_id, start, end} in seconds."""
    _expire()
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO:
        raise HTTPException(415, f"Unsupported video type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO))}")
    try:
        q = json.loads(questions or "[]")
        assert isinstance(q, list)
        for item in q:
            assert {"question_id", "start", "end"} <= set(item) and float(item["end"]) > float(item["start"]) >= 0
    except (ValueError, AssertionError, TypeError):
        raise HTTPException(422, "questions must be a JSON list of {question_id, start, end} with end > start >= 0")

    photo_bytes = None
    if reference_photo is not None and reference_photo.filename:
        pext = os.path.splitext(reference_photo.filename)[1].lower()
        if pext not in ALLOWED_PHOTO:
            raise HTTPException(415, f"Unsupported photo type '{pext}'. Allowed: {', '.join(sorted(ALLOWED_PHOTO))}")
        photo_bytes = await reference_photo.read()
        if len(photo_bytes) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "Reference photo is larger than 10 MB")

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    size = 0
    with os.fdopen(fd, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                f.close()
                os.remove(tmp_path)
                raise HTTPException(413, "Video is larger than 500 MB")
            f.write(chunk)
    if size == 0:
        os.remove(tmp_path)
        raise HTTPException(422, "Empty video upload")

    now = time.time()
    job = VisionJob(job_id=uuid.uuid4().hex, status="queued", stage="queued", stage_label=_STAGE_LABEL["queued"],
                    progress=0.0, created_at=now, updated_at=now, video_filename=video.filename,
                    reference_photo=photo_bytes is not None)
    with _lock:
        _jobs[job.job_id] = job
        _job_owner[job.job_id] = user.organization_id
    _refresh_queue_positions()
    _executor.submit(_run, job.job_id, tmp_path, photo_bytes, q)
    return job


@router.get("/jobs/{job_id}", response_model=VisionJob)
def get_job(job_id: str, user: User = Depends(require_user)):
    with _lock:
        job = _jobs.get(job_id)
        if job is None or _job_owner.get(job_id) != user.organization_id:
            raise HTTPException(404, "Unknown or expired job")
        return job.model_copy()


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, user: User = Depends(require_writable)):
    with _lock:
        job = _jobs.get(job_id)
        if job is None or _job_owner.get(job_id) != user.organization_id:
            raise HTTPException(404, "Unknown or expired job")
        if job.status in ("queued", "running"):
            raise HTTPException(409, "Job is still running; delete it after it finishes")
        _jobs.pop(job_id, None)
        _job_owner.pop(job_id, None)


@router.get("/health", response_model=VisionHealth)
def health():
    import importlib.util
    m = MODELS_DIR
    models = {
        "insightface_buffalo_l": os.path.isdir(os.path.join(m, "insightface", "models", "buffalo_l")),
        "syncnet_v2": os.path.exists(os.path.join(m, "syncnet", "syncnet_v2.model")),
        "mediapipe_face_landmarker": os.path.exists(os.path.join(m, "mediapipe", "face_landmarker.task")),
        "yolov3": os.path.exists(os.path.join(m, "yolo", "yolov3.weights")),
        "yolov3_tiny": os.path.exists(os.path.join(m, "yolo", "yolov3-tiny.weights")),
    }
    deps = {name: importlib.util.find_spec(name) is not None
            for name in ("cv2", "insightface", "onnxruntime", "mediapipe", "torch", "python_speech_features", "imageio_ffmpeg")}
    checks = {
        "face_identity": models["insightface_buffalo_l"] and deps["insightface"] and deps["onnxruntime"],
        "lip_sync": models["syncnet_v2"] and deps["torch"] and deps["python_speech_features"] and models["insightface_buffalo_l"],
        "gaze_and_eye_cues": models["mediapipe_face_landmarker"] and deps["mediapipe"],
        "phone_and_people": (models["yolov3"] or models["yolov3_tiny"]) and deps["cv2"],
    }
    return VisionHealth(ready=all(checks.values()) and deps["imageio_ffmpeg"], models=models, dependencies=deps,
                        checks_available=checks)
