"""
Adapter for deepinsight/insightface (https://github.com/deepinsight/insightface).

Master Spec Section 9 / Build Spec Section 2: face embedding + identity
consistency across the interview (carried over "as-is" module).

INTEGRATION NOTE: InsightFace needs ONNX face-recognition models (buffalo_l)
and per-frame face crops from raw video. Not available in this text-only
session. This adapter consumes the per-segment face-embedding cosine-similarity
trace that InsightFace would produce across interview frames
(data/synthetic/*.json -> "identity_signals") and converts it to a risk score.
Swap `compute()` for real embedding extraction + pairwise cosine similarity
against a first-frame identity anchor when video is available.
"""
from __future__ import annotations
from backend.schemas import ModuleResult, Evidence, Quality


def compute(identity_signals: dict | None) -> ModuleResult:
    if not identity_signals:
        return ModuleResult(
            feature="face_identity", score=0, confidence=0.0, level="UNAVAILABLE", status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="face_identity", severity="low", message="No video available for face-identity analysis.")],
            quality=Quality(data_quality=0.0, baseline_sufficient=False),
        )

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
