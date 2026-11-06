"""
Person / phone detection (M3 task P-5), porting vardanagarwal/Proctoring-AI
`person_and_phone.py` (MIT): YOLOv3 on COCO, class 0 = person, class 67 =
cell phone, score threshold + NMS IoU 0.5, "more than one person" and
"phone detected" alerts.

Changed: runs the original Darknet weights through OpenCV `cv2.dnn` instead of
a TensorFlow re-implementation (no TF dependency). Default model is
YOLOv3-tiny (35 MB); full YOLOv3 is used automatically if downloaded
(`scripts/download_models.py --full-yolo`). Alerts must persist for
`object_persist_samples` consecutive samples (upstream alerts per frame), so a
single false detection never becomes evidence.

These are STRONG integrity signals (Build Spec 3.3), but they are BEHAVIOUR
evidence (proctoring_adapter), never deepfake evidence.
"""
from __future__ import annotations

import os
import threading
from typing import Iterable, Optional, Sequence

import numpy as np

from backend.config import GAZE_THRESHOLDS, MODELS_DIR
from backend.vision.face_track import ModelUnavailable
from backend.vision.media import Frame, QuestionSegment, question_at

PERSON, PHONE = 0, 67
_NET = None
_LOCK = threading.Lock()


def get_net():
    """(net, output_layer_names, input_size, model_name). Full YOLOv3 if present, else YOLOv3-tiny."""
    global _NET
    with _LOCK:
        if _NET is None:
            import cv2
            d = os.path.join(MODELS_DIR, "yolo")
            for name in ("yolov3", "yolov3-tiny"):
                cfg, w = os.path.join(d, f"{name}.cfg"), os.path.join(d, f"{name}.weights")
                if os.path.exists(cfg) and os.path.exists(w):
                    net = cv2.dnn.readNetFromDarknet(cfg, w)
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    from backend.config import VISION_BUDGET
                    _NET = (net, net.getUnconnectedOutLayersNames(), VISION_BUDGET.yolo_input, name)
                    break
            else:
                raise ModelUnavailable("YOLO weights missing: run python scripts/download_models.py")
        return _NET


def detect(image: np.ndarray, score_threshold: Optional[float] = None, nms_iou: Optional[float] = None) -> list[dict]:
    """Persons and phones in one BGR image: [{"cls", "score", "box": (x, y, w, h)}]."""
    import cv2
    g = GAZE_THRESHOLDS
    thr = g.yolo_score if score_threshold is None else score_threshold
    net, outs_names, size, _ = get_net()
    h, w = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, 1 / 255.0, (size, size), swapRB=True, crop=False))
    boxes, scores, classes = [], [], []
    for out in net.forward(outs_names):
        for row in out:
            cls_scores = row[5:]
            for c in (PERSON, PHONE):
                s = float(row[4] * cls_scores[c])
                if s >= thr:
                    cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                    boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                    scores.append(s)
                    classes.append(c)
    keep = []
    for c in (PERSON, PHONE):
        idx = [i for i, k in enumerate(classes) if k == c]
        if idx:
            sel = cv2.dnn.NMSBoxes([boxes[i] for i in idx], [scores[i] for i in idx], thr, g.yolo_nms_iou if nms_iou is None else nms_iou)
            keep += [idx[int(j)] for j in np.array(sel).flatten()]
    return [{"cls": classes[i], "score": round(scores[i], 3), "box": tuple(boxes[i])} for i in keep]


def _persistent_runs(times: list[float], flags: list[bool], min_n: int) -> list[tuple[float, float]]:
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i >= min_n:
                runs.append((times[i], times[j - 1]))
            i = j
        else:
            i += 1
    return runs


def analyze(frames: Iterable[Frame], questions: Sequence[QuestionSegment] = (),
            extra_faces: Optional[dict] = None) -> dict:
    """Person/phone part of behavior_signals. extra_faces: {t: n_faces} from the face tracker (optional)."""
    g = GAZE_THRESHOLDS
    times, multi, phone = [], [], []
    for f in frames:
        dets = detect(f.image)
        n_person = sum(d["cls"] == PERSON for d in dets)
        if extra_faces:
            n_person = max(n_person, extra_faces.get(round(f.t, 2), 0))
        times.append(f.t)
        multi.append(n_person >= 2)
        phone.append(any(d["cls"] == PHONE for d in dets))
    if not times:
        return {"objects_status": "no_frames"}
    multi_runs = _persistent_runs(times, multi, g.object_persist_samples)
    phone_runs = _persistent_runs(times, phone, g.object_persist_samples)

    def events(runs, kind):
        return [{"type": kind, "t": round(a, 2), "t_end": round(b, 2),
                 "question_id": question_at(a, questions) if questions else None} for a, b in runs[:10]]

    return {
        "objects_status": "ok", "objects_model": get_net()[3], "object_samples": len(times),
        "multi_person_detected": bool(multi_runs), "phone_detected": bool(phone_runs),
        "multi_person_pct": round(float(np.mean(multi)), 3), "phone_pct": round(float(np.mean(phone)), 3),
        "object_events": events(multi_runs, "multi_person") + events(phone_runs, "phone"),
    }
