"""
Adapter for vardanagarwal/Proctoring-AI (https://github.com/vardanagarwal/Proctoring-AI).

Build Spec Section 3.3: eye tracking, head pose, person/phone detection,
aggregated into a behavior_risk_score.

Two entry points (M3 decision D1):
- `analyze_video(path, questions, track_result)` runs the ported Proctoring-AI
  logic on the video: gaze + head pose (backend/vision/gaze.py on MediaPipe
  landmarks) and person/phone detection (backend/vision/objects.py, YOLOv3),
  and returns `behavior_signals`.
- `compute(behavior_signals)` turns those signals into a ModuleResult. It accepts
  the real-video signals and the legacy synthetic-fixture shape, so the demo
  candidates keep working unchanged.

Rules (Build Spec 3.3 + requirement R1):
- Gaze / head-pose are WEAK evidence, capped so they alone can never exceed
  MEDIUM. For real video the gaze term is mainly the candidate's deviation from
  their OWN first-answer baseline, so a nervous candidate who glances away
  throughout is not scored like someone who looks away only on hard questions.
- Multiple people / a phone in frame are STRONG evidence (direct violations).
- Behaviour evidence only: nothing here enters the deepfake score (D3).
"""
from __future__ import annotations

from typing import Optional, Sequence

from backend.config import RISK_THRESHOLDS, VISION_BUDGET
from backend.schemas import Evidence, ModuleResult, Quality

WEAK_CAP = 55.0
HEAD_BONUS = {"normal": 0.0, "elevated": 10.0, "high": 20.0}


def analyze_video(path: str, questions: Sequence = (), track_result=None) -> dict:
    """Ported Proctoring-AI analysis -> behavior_signals. Never raises for missing models or bad media."""
    from backend.vision import face_track, gaze, landmarks, media, objects
    try:
        info = media.probe(path)
        end = min(info.duration or VISION_BUDGET.max_seconds_analyzed, VISION_BUDGET.max_seconds_analyzed)
        valid = None
        if track_result is not None:
            good = sorted(o.t for tr in track_result.tracks for o in tr.obs if o.quality_ok)
            step = 1.0 / track_result.sample_fps
            import bisect

            def valid(t, good=good, step=step):       # a clear SCRFD face within one sample of t
                i = bisect.bisect_left(good, t - step)
                return i < len(good) and good[i] <= t + step
        fps = VISION_BUDGET.landmarks_fps
        stream = [(f.t, o, f.image.shape[:2]) for f, o in landmarks.iter_landmarks(
            media.iter_frames(path, fps=fps, max_width=VISION_BUDGET.max_frame_width, end=end, info=info))]
        signals = gaze.analyze(stream, fps, questions, valid=valid)
        obj_frames = media.iter_frames(path, fps=VISION_BUDGET.objects_fps, max_width=VISION_BUDGET.max_frame_width,
                                       end=end, info=info)
        signals.update(objects.analyze(obj_frames, questions))
        return signals
    except face_track.ModelUnavailable as e:
        return {"status_reason": "model_unavailable", "detail": str(e)}
    except media.MediaError as e:
        return {"status_reason": "media_error", "detail": str(e)}


def _level(score: float) -> str:
    return "LOW" if score <= RISK_THRESHOLDS.low_max else ("MEDIUM" if score <= RISK_THRESHOLDS.medium_max else "HIGH")


def _unavailable(message: str, status: str = "INSUFFICIENT_EVIDENCE") -> ModuleResult:
    return ModuleResult(
        feature="behavior", score=0, confidence=0.0, level="UNAVAILABLE", status=status,
        evidence=[Evidence(type="behavior", severity="low", message=message)],
        quality=Quality(data_quality=0.0, baseline_sufficient=False),
    )


def compute(behavior_signals: Optional[dict]) -> ModuleResult:
    if not behavior_signals:
        return _unavailable("No video available for gaze/behavior analysis.")
    reason = behavior_signals.get("status_reason")
    if reason == "model_unavailable":
        return _unavailable("Gaze / object models not installed; behaviour check skipped.", status="UNAVAILABLE")
    if reason == "media_error":
        return _unavailable("Video could not be decoded for gaze/behavior analysis.")
    if "gaze_status" in behavior_signals or "objects_status" in behavior_signals:
        return _compute_real(behavior_signals)
    return _compute_legacy(behavior_signals)


def _compute_real(s: dict) -> ModuleResult:
    gaze_ok = s.get("gaze_status") == "ok"
    obj_ok = s.get("objects_status") == "ok"
    if not gaze_ok and not obj_ok:
        return _unavailable("The face was not clearly visible long enough to assess gaze or surroundings.")
    evidence: list[Evidence] = []
    risk = 0.0
    confidence = 0.65

    if gaze_ok:
        pct = float(s.get("gaze_off_screen_pct", 0.0))
        base = s.get("baseline_gaze_off_pct")
        per_q = s.get("per_question", [])
        later = [q for q in per_q[1:]] if per_q else []
        excess = max([q["gaze_off_pct"] - (base or 0.0) for q in later], default=0.0)
        rel = max(0.0, excess) if base is not None and later else max(0.0, pct - (base or 0.0))
        head = s.get("head_pose_variance", "normal")
        risk = min(WEAK_CAP, 100.0 * (0.3 * pct + 0.6 * rel) + HEAD_BONUS.get(head, 0.0))
        confidence *= min(1.0, 0.4 + float(s.get("gaze_coverage", 1.0)))

        spikes = [q for q in later if base is not None and q["gaze_off_pct"] - base >= 0.2]
        for q in spikes[:4]:
            d = (q.get("dominant_direction") or "away").replace("_", " ")
            evidence.append(Evidence(type="gaze", severity="medium", question_id=q["question_id"],
                                     message=f"During {q['question_id']} the candidate looked {d} for {q['gaze_off_pct'] * 100:.0f}% "
                                             f"of the answer, vs {base * 100:.0f}% in their first answer."))
        for ev in s.get("gaze_events", [])[:3]:
            if not spikes or ev.get("question_id") not in {q["question_id"] for q in spikes}:
                d = ev["direction"].replace("_", " ")
                evidence.append(Evidence(type="gaze", severity="low", question_id=ev.get("question_id"), timestamp=ev["t"],
                                         message=f"Looked {d} from {ev['t']:.1f}s to {ev['t_end']:.1f}s."))
        if not spikes:
            if pct >= 0.2 and base is not None and pct - base < 0.1:
                evidence.insert(0, Evidence(type="gaze", severity="low",
                                            message=f"Gaze was off-centre {pct * 100:.0f}% of the time, but consistently so from the first "
                                                    f"answer (baseline {base * 100:.0f}%); this pattern is common with nervousness."))
            else:
                evidence.insert(0, Evidence(type="gaze", severity="low",
                                            message=f"Gaze off-centre {pct * 100:.0f}% of the time; no question stood out against the candidate's own baseline."))
        if head != "normal":
            evidence.append(Evidence(type="head_pose", severity="medium" if head == "elevated" else "high",
                                     message=f"Head movement was {head} (yaw spread {s.get('yaw_std', 0):.0f}°, pitch spread {s.get('pitch_std', 0):.0f}°)."))

    if obj_ok:
        for ev in s.get("object_events", [])[:6]:
            q = f" during {ev['question_id']}" if ev.get("question_id") else ""
            if ev["type"] == "multi_person":
                evidence.append(Evidence(type="multi_person", severity="high", question_id=ev.get("question_id"), timestamp=ev["t"],
                                         message=f"More than one person was in frame from {ev['t']:.0f}s to {ev['t_end']:.0f}s{q}."))
            else:
                evidence.append(Evidence(type="phone", severity="high", question_id=ev.get("question_id"), timestamp=ev["t"],
                                         message=f"A mobile phone was visible from {ev['t']:.0f}s to {ev['t_end']:.0f}s{q}."))
        if s.get("multi_person_detected"):
            risk, confidence = max(risk, 80.0), 0.85
        if s.get("phone_detected"):
            risk, confidence = max(risk, 85.0), 0.85

    if not evidence:
        evidence.append(Evidence(type="behavior", severity="low", message="No notable gaze or surroundings anomalies detected."))
    return ModuleResult(
        feature="behavior", score=round(min(100.0, risk), 1), confidence=round(confidence, 2), level=_level(risk),
        evidence=evidence, quality=Quality(data_quality=round(float(s.get("gaze_coverage", 0.9)), 2),
                                           baseline_sufficient=s.get("baseline_gaze_off_pct") is not None),
    )


def _compute_legacy(behavior_signals: dict) -> ModuleResult:
    """Synthetic-fixture path (pre-M3 contract), unchanged so demo candidates keep their outcomes."""
    gaze_pct = behavior_signals.get("gaze_off_screen_pct", 0.0)
    head_variance = behavior_signals.get("head_pose_variance", "normal")  # normal|elevated|high
    multi_person = behavior_signals.get("multi_person_detected", False)
    phone = behavior_signals.get("phone_detected", False)

    # Weak evidence component — capped so it alone cannot exceed MEDIUM (<=60)
    weak_risk = min(55.0, gaze_pct * 100 * 0.6 + {"normal": 0, "elevated": 15, "high": 30}[head_variance])

    risk = weak_risk
    evidence = []
    if gaze_pct >= 0.5:
        evidence.append(Evidence(type="gaze", severity="high", message=f"Gaze remained off-center for {gaze_pct*100:.0f}% of answer duration."))
    elif gaze_pct >= 0.2:
        evidence.append(Evidence(type="gaze", severity="medium", message=f"Gaze remained off-center for {gaze_pct*100:.0f}% of answer duration."))
    elif gaze_pct > 0:
        evidence.append(Evidence(type="gaze", severity="low", message=f"Minor gaze deviation ({gaze_pct*100:.0f}% of duration) — within normal range."))

    if head_variance != "normal":
        evidence.append(Evidence(type="head_pose", severity="medium" if head_variance == "elevated" else "high",
                                  message=f"Head-pose variance classified as {head_variance}."))

    # Strong evidence: direct integrity violations, not capped
    if multi_person:
        risk = max(risk, 80.0)
        evidence.append(Evidence(type="multi_person", severity="high", message="Multiple people detected in frame during the interview."))
    if phone:
        risk = max(risk, 85.0)
        evidence.append(Evidence(type="phone", severity="high", message="A mobile phone was detected in frame during the interview."))

    risk = min(100.0, risk)
    level = "LOW" if risk <= 30 else ("MEDIUM" if risk <= 60 else "HIGH")
    confidence = 0.85 if (multi_person or phone) else 0.65

    return ModuleResult(
        feature="behavior", score=round(risk, 1), confidence=confidence, level=level,
        evidence=evidence or [Evidence(type="behavior", severity="low", message="No notable gaze/behavior anomalies detected.")],
        quality=Quality(data_quality=0.9, baseline_sufficient=True),
    )
