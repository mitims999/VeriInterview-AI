"""
Adapter for vardanagarwal/Proctoring-AI (https://github.com/vardanagarwal/Proctoring-AI).

Build Spec Section 3.3: use `face_detector.py`, `eye_tracker.py`,
`head_pose_estimation.py`, `mouth_opening_detector.py`, `person_and_phone.py`
as-is, aggregate per-answer into a behavior_risk_score.

INTEGRATION NOTE: Proctoring-AI's scripts run against a live webcam/video
stream via OpenCV + dlib landmarks. This prototype has no raw video, so this
adapter consumes the same aggregate signals those scripts would produce
(gaze_off_screen_pct, head_pose_variance, multi_person_detected,
phone_detected) from the synthetic fixtures. Swap `compute()` to call the
real scripts frame-by-frame over a video file when available; the output
contract is unchanged.

Rule (Build Spec 3.3): gaze/head-pose variance is WEAK evidence and must never
alone push behavior_risk_score into HIGH. multi_person_detected / phone_detected
are STRONG evidence (direct integrity violations, not nervousness proxies).
"""
from __future__ import annotations
from backend.schemas import ModuleResult, Evidence, Quality


def compute(behavior_signals: dict | None) -> ModuleResult:
    if not behavior_signals:
        return ModuleResult(
            feature="behavior", score=0, confidence=0.0, level="UNAVAILABLE", status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="behavior", severity="low", message="No video available for gaze/behavior analysis.")],
            quality=Quality(data_quality=0.0, baseline_sufficient=False),
        )

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
