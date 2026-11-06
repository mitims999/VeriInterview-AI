"""
Adapter for joonson/syncnet_python (https://github.com/joonson/syncnet_python).

Build Spec Section 3.2: AV offset + SyncNet confidence -> sync risk. Never
reported as "deepfake confirmed".

Two entry points (M3 decision D1):
- `analyze_video(path, speech, questions)` runs the vendored SyncNet model
  over the speech in the video (backend/vision/syncnet_runner.py) and returns
  `av_signals`.
- `compute(av_signals)` turns those signals into a ModuleResult. It accepts
  the real-video signals and the legacy synthetic-fixture shape
  (`av_offset_frames`, `sync_confidence` only), so the demo candidates keep
  working unchanged.

Scoring rules (calibrated on 66 generated GRID clips, docs/m3/calibration/syncnet_grid.md):
1. |median offset|: <= 2 frames normal, 3-5 MEDIUM, >= 6 HIGH (shifts were recovered
   to within 1 frame).
2. Desynchronised windows: any speech window with |offset| >= 6 frames and usable
   confidence is flagged with its time and question. This catches one manipulated answer.
3. No consistent alignment at all: LSE-C below `lse_c_low` AND offset unstable ->
   HIGH (audio swaps: LSE-C 1.0-1.6, stability 0.25-0.75).
4. Low LSE-C with a STABLE normal offset is a poor camera / compression (240p
   genuine: LSE-C 1.2-5.4, stability 1.0): it lowers confidence, never adds risk (R1).
Offset sign: negative = audio arrives later than the lips (upstream convention).
"""
from __future__ import annotations

from typing import Optional, Sequence

from backend.config import RISK_THRESHOLDS, SYNC_THRESHOLDS
from backend.schemas import Evidence, ModuleResult, Quality


def analyze_video(path: str, speech=None, questions: Sequence = ()) -> dict:
    """Real SyncNet analysis -> av_signals. Never raises for missing models or bad media."""
    from backend.vision import face_track, media, syncnet_runner
    try:
        return syncnet_runner.analyze(path, speech=speech, questions=questions)
    except face_track.ModelUnavailable as e:
        return {"status_reason": "model_unavailable", "detail": str(e)}
    except media.MediaError as e:
        return {"status_reason": "media_error", "detail": str(e)}


def _level(score: float) -> str:
    return "LOW" if score <= RISK_THRESHOLDS.low_max else ("MEDIUM" if score <= RISK_THRESHOLDS.medium_max else "HIGH")


def _unavailable(message: str, status: str = "INSUFFICIENT_EVIDENCE") -> ModuleResult:
    return ModuleResult(
        feature="av_sync", score=0, confidence=0.0, level="UNAVAILABLE", status=status,
        evidence=[Evidence(type="av_sync", severity="low", message=message)],
        quality=Quality(data_quality=0.0, baseline_sufficient=False),
    )


_REASONS = {
    "media_error": "Video could not be decoded for lip-sync analysis.",
    "no_audio": "The video has no audio track; lip-sync could not be assessed.",
    "no_speech": "No speech was detected in the video; lip-sync could not be assessed.",
    "no_face": "No clear face was visible while the candidate was speaking; lip-sync could not be assessed.",
    "too_short": "Too little speech with a visible face to assess lip-sync reliably.",
}


def compute(av_signals: Optional[dict]) -> ModuleResult:
    if not av_signals:
        return _unavailable("No video segment available for lip-sync analysis.")
    reason = av_signals.get("status_reason")
    if reason == "model_unavailable":
        return _unavailable("Lip-sync model not installed; lip-sync check skipped.", status="UNAVAILABLE")
    if reason in _REASONS:
        return _unavailable(_REASONS[reason])
    if "lse_c" in av_signals:
        return _compute_real(av_signals)
    return _compute_legacy(av_signals)


def _offset_risk(off: int) -> float:
    c = SYNC_THRESHOLDS
    if off <= c.bonafide_offset_frames:
        return 5.0 + 5.0 * off
    if off < c.high_offset_frames:
        return RISK_THRESHOLDS.low_max + 1.0 + (off - c.bonafide_offset_frames - 1) * 12.0
    return min(90.0, RISK_THRESHOLDS.medium_max + 1.0 + (off - c.high_offset_frames) * 5.0)


def _compute_real(s: dict) -> ModuleResult:
    c = SYNC_THRESHOLDS
    off = abs(int(s.get("av_offset_frames", 0)))
    lse_c = float(s.get("lse_c", 0.0))
    stability = float(s.get("offset_stability", 1.0))
    analysed = float(s.get("speech_seconds_analyzed", 0.0))
    segs = [x for x in s.get("per_segment", []) if "offset" in x]

    risk = _offset_risk(off)
    evidence: list[Evidence] = []

    if off <= c.bonafide_offset_frames:
        evidence.append(Evidence(type="av_offset", severity="low",
                                 message=f"Lips and voice are aligned (offset {off} frame(s), within the normal range)."))
    else:
        sev = "medium" if off < c.high_offset_frames else "high"
        evidence.append(Evidence(type="av_offset", severity=sev,
                                 message=f"Voice is out of step with lip movement by {off} frames (~{off * 40} ms) "
                                         f"across the analysed speech; normal is {c.bonafide_offset_frames} or fewer."))

    # rule 2: individual windows that are clearly desynchronised while the rest may be fine
    bad = [x for x in segs if abs(x["offset"]) >= c.high_offset_frames and x["lse_c"] >= c.lse_c_low]
    if bad and off < c.high_offset_frames:
        share = sum(x["n_frames"] for x in bad) / max(1, sum(x["n_frames"] for x in segs))
        risk = max(risk, RISK_THRESHOLDS.medium_max + 1.0 if share >= 0.25 else RISK_THRESHOLDS.low_max + 15.0)
    for x in bad[:5]:
        q = f" during {x['question_id']}" if x.get("question_id") else ""
        evidence.append(Evidence(type="av_offset_segment", severity="high" if off < c.high_offset_frames else "medium",
                                 question_id=x.get("question_id"), timestamp=x.get("start"),
                                 message=f"From {x['start']:.1f}s to {x['end']:.1f}s{q} the voice was out of step with the lips "
                                         f"by {abs(x['offset'])} frames while the rest of the interview was aligned differently."))

    # rule 3: no consistent alignment anywhere (e.g. replaced / dubbed audio)
    incoherent = lse_c < c.lse_c_low and stability < c.min_offset_stability
    if incoherent:
        risk = max(risk, 65.0)
        evidence.append(Evidence(type="av_no_sync", severity="high",
                                 message=f"Lip movements did not consistently match the audio at any alignment "
                                         f"(SyncNet confidence {lse_c:.1f}, offset agreement {stability * 100:.0f}%); "
                                         f"the audio may not come from the person on camera."))
    elif stability < c.min_offset_stability and segs:
        evidence.append(Evidence(type="av_offset_unstable", severity="medium",
                                 message=f"Lip-sync offset varied between parts of the interview "
                                         f"({stability * 100:.0f}% of analysed speech agreed with the typical offset)."))

    # confidence: how much we can trust the measurement (never risk on its own, rule 4)
    if incoherent:
        confidence = 0.65            # two independent cues agree: low confidence AND unstable offset
    else:
        confidence = max(0.2, min(0.95, lse_c / c.lse_c_good))
        if lse_c < c.lse_c_low:
            evidence.append(Evidence(type="sync_confidence", severity="low",
                                     message=f"Lip-sync signal is weak (SyncNet confidence {lse_c:.1f}), typical of a low-resolution "
                                             f"or heavily compressed camera; confidence reduced, not counted as risk."))
    if analysed < 20.0:
        confidence *= 0.8
    risk = min(95.0, risk)
    return ModuleResult(
        feature="av_sync", score=round(risk, 1), confidence=round(confidence, 2), level=_level(risk), evidence=evidence,
        quality=Quality(data_quality=round(min(1.0, lse_c / c.lse_c_good), 2), baseline_sufficient=analysed >= c.min_speech_seconds),
    )


def _compute_legacy(av_signals: dict) -> ModuleResult:
    """Synthetic-fixture path (pre-M3 contract), unchanged so demo candidates keep their outcomes."""
    offset = abs(av_signals.get("av_offset_frames", 0))
    sync_confidence = av_signals.get("sync_confidence", 0.7)  # SyncNet's own confidence, 0-1

    # SyncNet bonafide range is roughly within +-1 to 2 frames at good confidence.
    if offset <= 2:
        risk = 5 + offset * 5
    elif offset <= 5:
        risk = 25 + (offset - 2) * 12
    else:
        risk = min(100, 60 + (offset - 5) * 8)

    # low SyncNet confidence widens uncertainty rather than raising risk outright
    reported_confidence = max(0.2, min(0.95, sync_confidence))

    evidence = []
    if offset <= 2:
        evidence.append(Evidence(type="av_offset", severity="low", message=f"AV offset of {offset} frame(s) within normal bonafide range."))
    elif offset <= 5:
        evidence.append(Evidence(type="av_offset", severity="medium", message=f"AV offset of {offset} frames is above typical bonafide range."))
    else:
        evidence.append(Evidence(type="av_offset", severity="high", message=f"AV offset of {offset} frames is well outside bonafide range; audio/video desync detected."))

    if sync_confidence < 0.5:
        evidence.append(Evidence(type="sync_confidence", severity="medium", message=f"SyncNet confidence low ({sync_confidence:.2f}); result should be treated cautiously."))

    level = "LOW" if risk <= 30 else ("MEDIUM" if risk <= 60 else "HIGH")
    return ModuleResult(
        feature="av_sync", score=round(risk, 1), confidence=round(reported_confidence, 2), level=level,
        evidence=evidence, quality=Quality(data_quality=sync_confidence, baseline_sufficient=True),
    )
