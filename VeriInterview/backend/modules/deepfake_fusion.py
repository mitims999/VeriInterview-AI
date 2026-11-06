"""
Deepfake Sub-Score (M3 task F-1): fuses the three media-authenticity signals
into one ModuleResult(feature="deepfake") for the evidence-fusion engine / report.

Inputs
- identity: insightface_adapter ModuleResult (within-session consistency +
  optional reference-photo match)
- sync: syncnet_adapter ModuleResult (lip-sync offset / coherence)
- eye_authenticity: backend/vision/eye_auth.analyze() dict (blink pattern,
  mouth-vs-speech). Weak cues by design (R1).
Gaze / head movement / phone / extra people are NOT inputs: they are behaviour
evidence (decision D3), so a nervous candidate cannot look like a deepfake here.

Rules (deterministic, config.DEEPFAKE_WEIGHTS)
1. Weighted mean of available inputs (identity 0.45, sync 0.40, eye 0.15),
   but never below `single_input_floor` x the strongest input, so one clear
   finding still reaches MEDIUM ("needs review") instead of being averaged
   away. Unavailable inputs give their weight to the others and lower confidence.
2. Golden rule: the sub-score can only be HIGH when >= 2 inputs are at least
   MEDIUM, OR there is one STRONG identity signal (a confident switch to a
   different person, or a mismatch with the candidate's reference photo).
   Otherwise it is capped at `single_signal_cap` (MEDIUM).
3. Corroboration: each extra HIGH input adds `corroboration_boost`.
4. Eye cues alone never exceed MEDIUM; a HIGH blink rate adds nothing and,
   with normal authenticity signals, produces a "consistent with
   nervousness" note.
The output is evidence for human review. It never says "deepfake confirmed".
"""
from __future__ import annotations

from typing import Optional

from backend.config import DEEPFAKE_WEIGHTS, EYE_AUTH_THRESHOLDS, RISK_THRESHOLDS
from backend.schemas import Evidence, ModuleResult, Quality

STRONG_IDENTITY_TYPES = {"identity_switch", "reference_mismatch"}


def _level(score: float) -> str:
    return "LOW" if score <= RISK_THRESHOLDS.low_max else ("MEDIUM" if score <= RISK_THRESHOLDS.medium_max else "HIGH")


def eye_authenticity_result(eye: Optional[dict]) -> ModuleResult:
    """Turn the eye_auth dict into a (weak) ModuleResult."""
    if not eye or eye.get("status_reason"):
        return ModuleResult(feature="eye_authenticity", score=0, confidence=0.0, level="UNAVAILABLE",
                            status="INSUFFICIENT_EVIDENCE",
                            evidence=[Evidence(type="eye_authenticity", severity="low",
                                               message="Eyes/mouth not visible long enough for blink and articulation checks.")])
    cfg, cap = EYE_AUTH_THRESHOLDS, DEEPFAKE_WEIGHTS.single_signal_cap
    risk, evidence = 5.0, []
    mm = eye.get("mouth_speech_mismatch_pct")
    if mm is not None and mm >= cfg.mouth_mismatch_medium_pct:
        risk = max(risk, min(cap, RISK_THRESHOLDS.low_max + 10.0 + 100.0 * (mm - cfg.mouth_mismatch_medium_pct)))
        evidence.append(Evidence(type="mouth_speech_mismatch", severity="medium",
                                 message=f"For {mm * 100:.0f}% of the speech the mouth showed no articulation "
                                         f"(still or replayed face while the voice continued)."))
    if eye.get("metronomic_blinks"):
        risk = max(risk, RISK_THRESHOLDS.low_max + 15.0)
        evidence.append(Evidence(type="blink_pattern", severity="medium",
                                 message=f"Blinks were unusually regular (interval variation {eye.get('blink_ibi_cv', 0):.2f}); "
                                         f"natural blinking is irregular."))
    if eye.get("too_few_blinks"):
        risk = max(risk, RISK_THRESHOLDS.low_max + 5.0)
        evidence.append(Evidence(type="blink_rate", severity="low",
                                 message=f"Very few blinks ({eye.get('blink_rate_per_min', 0):.1f}/min over {eye.get('eye_visible_s', 0):.0f}s). "
                                         f"Reading or concentrating also suppresses blinking, so this is weak evidence."))
    if not evidence:
        evidence.append(Evidence(type="eye_authenticity", severity="low",
                                 message=f"Blinking ({eye.get('blink_rate_per_min', 0):.0f}/min) and mouth articulation looked natural."))
    return ModuleResult(feature="eye_authenticity", score=round(min(cap, risk), 1), confidence=0.5,
                        level=_level(min(cap, risk)), evidence=evidence)


def fuse_deepfake(identity: ModuleResult, sync: ModuleResult, eye_authenticity: Optional[dict] = None) -> ModuleResult:
    w = DEEPFAKE_WEIGHTS
    eye = eye_authenticity_result(eye_authenticity)
    parts = {"identity": (identity, w.identity), "sync": (sync, w.sync), "eye_authenticity": (eye, w.eye_authenticity)}
    avail = {k: (m, wt) for k, (m, wt) in parts.items() if m.status == "OK"}
    sub_scores = {k: {"score": m.score, "level": m.level, "confidence": m.confidence, "status": m.status}
                  for k, (m, _) in parts.items()}

    if not avail:
        return ModuleResult(feature="deepfake", score=0, confidence=0.0, level="UNAVAILABLE", status="INSUFFICIENT_EVIDENCE",
                            evidence=[Evidence(type="deepfake", severity="low",
                                               message="No media-authenticity check could be run on this video.")],
                            quality=Quality(data_quality=0.0, baseline_sufficient=False), details={"sub_scores": sub_scores})

    total_w = sum(wt for _, wt in avail.values())
    risk = sum(m.score * wt for m, wt in avail.values()) / total_w
    # one clear finding must not be averaged away to LOW (it still needs review); caps below still apply
    risk = max(risk, w.single_input_floor * max(m.score for m, _ in avail.values()))

    n_flagged = sum(m.level in ("MEDIUM", "HIGH") for m, _ in avail.values())
    n_high = sum(m.level == "HIGH" for m, _ in avail.values())
    strong_identity = identity.status == "OK" and identity.level == "HIGH" and identity.confidence >= 0.6 and \
        any(e.type in STRONG_IDENTITY_TYPES and e.severity == "high" for e in identity.evidence)

    if n_high >= 2:
        risk *= 1.0 + w.corroboration_boost * (n_high - 1)
    if strong_identity:
        risk = max(risk, w.strong_identity_switch_floor)
    elif n_flagged < 2:
        risk = min(risk, w.single_signal_cap)
    risk = min(100.0, risk)

    coverage = total_w / (w.identity + w.sync + w.eye_authenticity)
    confs = [m.confidence for m, _ in avail.values()]
    confidence = sum(m.confidence * wt for m, wt in avail.values()) / total_w
    confidence = confidence * (0.5 + 0.5 * coverage) - (max(confs) - min(confs)) * 0.1
    confidence = round(max(0.05, min(0.95, confidence)), 2)
    level = _level(risk)

    names = {"identity": "face identity", "sync": "lip-sync", "eye_authenticity": "blink/mouth cues"}
    flagged = [names[k] for k, (m, _) in avail.items() if m.level in ("MEDIUM", "HIGH")]
    missing = [names[k] for k in parts if k not in avail]
    if level == "HIGH":
        why = "a strong identity signal" if strong_identity and n_flagged < 2 else f"{n_flagged} independent checks agreeing"
        summary = (f"Media-authenticity concerns from {why} ({', '.join(flagged)}). "
                   f"Human review of the recording is recommended.")
        sev = "high"
    elif level == "MEDIUM":
        summary = (f"Some media-authenticity signals need review ({', '.join(flagged) or 'combined signals'}); "
                   f"no second independent check confirms them.")
        sev = "medium"
    else:
        summary = "Face identity, lip-sync and blink/mouth checks look consistent with a genuine live recording."
        sev = "low"
    if missing:
        summary += f" Not assessed: {', '.join(missing)}."
    evidence = [Evidence(type="deepfake_summary", severity=sev, message=summary)]
    for k, (m, _) in parts.items():
        evidence += [e for e in m.evidence if e.severity != "low"][:4]

    if eye_authenticity and eye_authenticity.get("high_blink_rate") and level == "LOW":
        evidence.append(Evidence(type="nervousness_note", severity="low",
                                 message=f"Frequent blinking ({eye_authenticity.get('blink_rate_per_min', 0):.0f}/min) is consistent with "
                                         f"nervousness; identity, lip-sync and articulation signals are normal."))

    return ModuleResult(feature="deepfake", score=round(risk, 1), confidence=confidence, level=level, evidence=evidence,
                        quality=Quality(data_quality=round(coverage, 2), baseline_sufficient=coverage >= 0.85),
                        details={"sub_scores": sub_scores, "inputs_flagged": n_flagged, "inputs_high": n_high,
                                 "strong_identity_signal": strong_identity, "coverage": round(coverage, 2)})
