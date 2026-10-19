"""
Adapter for deepinsight/insightface (https://github.com/deepinsight/insightface).

Master Spec Section 9 / Build Spec Section 2: face embedding + identity
consistency across the interview.

Two entry points (M3 decision D1):
- `analyze_video(path, questions)` runs the real model: the shared face
  tracker (backend/vision/face_track.py, buffalo_l SCRFD + ArcFace) and the
  identity analysis (backend/vision/identity.py), and returns `identity_signals`.
- `compute(identity_signals)` turns those signals into a ModuleResult. It
  accepts both the real-video signals and the legacy synthetic-fixture shape
  (`min_frame_similarity_to_anchor`, `face_lost_pct` only), so the demo
  candidates keep working unchanged.

Scoring rules for real video:
- Risk comes from the share of good-quality frames that do not match the
  enrollment anchor (calibrated ArcFace threshold) and from a confident
  identity switch (>= 2 distinct people, each present for several seconds).
- Missing or poor data (no face, too few usable frames, weak anchor, face
  lost) lowers CONFIDENCE and is reported, but never adds risk (Master Spec 28).
- Limitation: this checks consistency WITHIN the session. A face-swap that runs
  from the first frame keeps a consistent identity and is not caught here;
  lip-sync and eye-authenticity cues cover that case (deepfake_fusion.py).
"""
from __future__ import annotations

from typing import Optional, Sequence

from backend.config import IDENTITY_THRESHOLDS, RISK_THRESHOLDS, VISION_BUDGET
from backend.schemas import Evidence, ModuleResult, Quality


def analyze_video(path: str, questions: Sequence = (), track_result=None, reference_photo=None) -> dict:
    """Real InsightFace analysis of a video file -> identity_signals.

    reference_photo: optional photo of the registered candidate (file path,
    raw image bytes, or a BGR array), e.g. from the resume or an ID (task I-4).
    It is processed in memory and never stored.

    Never raises for missing models or unreadable media: it returns a dict with
    `status_reason` set, which `compute()` turns into UNAVAILABLE /
    INSUFFICIENT_EVIDENCE.
    """
    from backend.vision import face_track, identity, media
    try:
        if track_result is None:
            track_result = face_track.track_video(path, sample_fps=VISION_BUDGET.identity_fps, embed=True, pose=True)
        reference = None
        if reference_photo is not None:
            reference = face_track.embed_photo(_load_image(reference_photo))
        return identity.analyze(track_result, questions, reference=reference)
    except face_track.ModelUnavailable as e:
        return {"status_reason": "model_unavailable", "detail": str(e)}
    except media.MediaError as e:
        return {"status_reason": "media_error", "detail": str(e)}


def _load_image(photo):
    """Path, bytes or BGR array -> BGR array (None if unreadable)."""
    import cv2
    import numpy as np
    if isinstance(photo, np.ndarray):
        return photo
    if isinstance(photo, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(str(photo))


def _level(score: float) -> str:
    return "LOW" if score <= RISK_THRESHOLDS.low_max else ("MEDIUM" if score <= RISK_THRESHOLDS.medium_max else "HIGH")


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (max(x, x0) - x0) / (x1 - x0)
    return ys[-1]


def _unavailable(message: str, status: str = "INSUFFICIENT_EVIDENCE") -> ModuleResult:
    return ModuleResult(
        feature="face_identity", score=0, confidence=0.0, level="UNAVAILABLE", status=status,
        evidence=[Evidence(type="face_identity", severity="low", message=message)],
        quality=Quality(data_quality=0.0, baseline_sufficient=False),
    )


def compute(identity_signals: Optional[dict]) -> ModuleResult:
    if not identity_signals:
        return _unavailable("No video available for face-identity analysis.")
    reason = identity_signals.get("status_reason")
    if reason == "model_unavailable":
        return _unavailable("Face-identity model not installed; identity check skipped.", status="UNAVAILABLE")
    if reason == "media_error":
        return _unavailable("Video could not be decoded for face-identity analysis.")
    if reason == "no_face":
        return _unavailable("No face was visible in the video; identity consistency could not be assessed.")
    if reason == "insufficient_good_frames":
        return _unavailable("Too few clear, front-facing frames to build a reliable identity reference.")
    if "below_threshold_pct" in identity_signals:
        return _compute_real(identity_signals)
    return _compute_legacy(identity_signals)


def _compute_real(s: dict) -> ModuleResult:
    cfg = IDENTITY_THRESHOLDS
    below = float(s.get("below_threshold_pct", 0.0))
    distinct = int(s.get("distinct_identities", 1))
    anchor_q = float(s.get("anchor_quality", 1.0))
    good_pct = float(s.get("good_frame_pct", 1.0))
    lost = float(s.get("face_lost_pct", 0.0))
    good_frames = int(s.get("good_frames", 0))

    # share of non-matching frames -> risk: 0 -> 5, medium_pct -> just MEDIUM, high_pct -> just HIGH, 50% -> 90
    risk = _interp(below, [0.0, cfg.below_threshold_medium_pct, cfg.below_threshold_high_pct, 0.5],
                   [5.0, RISK_THRESHOLDS.low_max + 1.0, RISK_THRESHOLDS.medium_max + 1.0, 90.0])
    if distinct >= 2:
        risk = max(risk, 75.0)
    risk = min(95.0, risk)

    # confidence: what we could actually see, never risk
    confidence = 0.92
    confidence *= min(1.0, 0.5 + good_pct)                      # many skipped frames -> less sure
    confidence *= min(1.0, 0.4 + anchor_q)                       # weak enrollment anchor -> less sure
    if lost > cfg.face_lost_medium_pct:
        confidence *= 0.85
    if good_frames < 20:
        confidence *= 0.75
    confidence = max(0.2, min(0.95, confidence))

    evidence: list[Evidence] = []
    if distinct >= 2:
        evidence.append(Evidence(type="identity_switch", severity="high",
                                 message=f"{distinct} distinct faces each appeared for several seconds as the main on-camera person."))
    for ev in s.get("drift_events", [])[:5]:
        q = f" during {ev['question_id']}" if ev.get("question_id") else ""
        evidence.append(Evidence(type="identity_drift", severity="high" if distinct >= 2 else "medium",
                                 question_id=ev.get("question_id"), timestamp=ev.get("t"),
                                 message=f"Face did not match the session's opening face from {ev['t']:.1f}s to "
                                         f"{ev['t_end']:.1f}s{q} (lowest similarity {ev['similarity']:.2f}, "
                                         f"same-person threshold {cfg.same_person_cosine:.2f})."))
    # Reference photo (I-4): the only identity check that catches a face that is consistent all
    # session but is not the registered candidate (proxy or full-session face-swap).
    ref_status = s.get("reference_status")
    if ref_status == "ok":
        ref_sim = float(s["reference_similarity"])
        thr = cfg.reference_match_cosine
        if ref_sim < thr:
            risk = max(risk, 70.0)
            evidence.insert(0, Evidence(type="reference_mismatch", severity="high",
                                        message=f"The face in the interview does not match the candidate's reference photo "
                                                f"(similarity {ref_sim:.2f}, same-person threshold {thr:.2f}; "
                                                f"{s.get('reference_match_pct', 0) * 100:.0f}% of frames matched)."))
        elif ref_sim < thr + cfg.reference_uncertain_margin:
            risk = max(risk, RISK_THRESHOLDS.low_max + 10.0)
            evidence.insert(0, Evidence(type="reference_weak_match", severity="medium",
                                        message=f"Only a weak match to the candidate's reference photo (similarity {ref_sim:.2f}); "
                                                f"an old or low-quality photo can also cause this."))
        else:
            evidence.append(Evidence(type="reference_match", severity="low",
                                     message=f"Interview face matches the candidate's reference photo (similarity {ref_sim:.2f})."))
        if s.get("reference_extra_faces", 0) > 0:
            evidence.append(Evidence(type="reference_photo", severity="low",
                                     message="Reference photo contains more than one face; the largest one was used."))
    elif ref_status:
        evidence.append(Evidence(type="reference_photo", severity="low",
                                 message=f"Reference photo could not be used ({ref_status.replace('_', ' ')}); "
                                         f"identity was checked within the session only."))

    if not any(e.severity != "low" for e in evidence) and not any(e.type == "identity_drift" for e in evidence):
        evidence.append(Evidence(type="identity_drift", severity="low",
                                 message=f"Face matched the session's opening face throughout "
                                         f"({good_frames} clear frames, {below * 100:.0f}% below threshold)."))
    if lost > cfg.face_lost_medium_pct:
        evidence.append(Evidence(type="face_lost", severity="low",
                                 message=f"No face visible in {lost * 100:.0f}% of sampled frames; confidence reduced (not counted as risk)."))
    if anchor_q < 0.6:
        evidence.append(Evidence(type="anchor_quality", severity="low",
                                 message="Opening frames were inconsistent, so the identity reference is weak; confidence reduced."))

    return ModuleResult(
        feature="face_identity", score=round(risk, 1), confidence=round(confidence, 2), level=_level(risk),
        evidence=evidence, quality=Quality(data_quality=round(good_pct, 2), baseline_sufficient=anchor_q >= 0.6),
    )


def _compute_legacy(identity_signals: dict) -> ModuleResult:
    """Synthetic-fixture path (pre-M3 contract), unchanged so demo candidates keep their outcomes."""
    min_similarity = identity_signals.get("min_frame_similarity_to_anchor", 0.95)  # cosine sim, 1.0 = identical
    face_lost_pct = identity_signals.get("face_lost_pct", 0.0)

    drift = max(0.0, 1.0 - min_similarity)  # 0 = no drift, higher = more identity drift
    risk = min(100.0, drift * 400 + face_lost_pct * 50)

    evidence = []
    if drift > 0.15:
        evidence.append(Evidence(type="identity_drift", severity="high",
                                  message=f"Face embedding similarity dropped to {min_similarity:.2f} against session anchor."))
    elif drift > 0.05:
        evidence.append(Evidence(type="identity_drift", severity="medium",
                                  message=f"Minor face embedding drift observed (min similarity {min_similarity:.2f})."))
    else:
        evidence.append(Evidence(type="identity_drift", severity="low",
                                  message="Face identity remained consistent with session anchor throughout."))

    if face_lost_pct > 0.1:
        evidence.append(Evidence(type="face_lost", severity="medium",
                                  message=f"Face tracking lost for {face_lost_pct*100:.0f}% of the session."))

    level = "LOW" if risk <= 30 else ("MEDIUM" if risk <= 60 else "HIGH")
    return ModuleResult(
        feature="face_identity", score=round(risk, 1), confidence=0.9 if face_lost_pct < 0.2 else 0.6, level=level,
        evidence=evidence, quality=Quality(data_quality=1.0 - face_lost_pct, baseline_sufficient=True),
    )
