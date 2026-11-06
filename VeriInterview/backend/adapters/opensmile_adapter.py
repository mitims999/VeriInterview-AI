"""
Adapter for audeering/opensmile + opensmile-python
(https://github.com/audeering/opensmile, https://github.com/audeering/opensmile-python).

Authoritative spec: Voice_Prosody_Feature_Verification.md (Section 3.4 of the
Build Spec references it directly). Feature set: eGeMAPSv02 (NOT ComParE_2016).
Candidate-specific baseline built from the first few answers — never compared
to a universal "normal" voice.

LICENSING NOTE: openSMILE is free for research/non-commercial use under the
audEERING research license; commercial deployment requires a separate license.
This must be flagged in the README (Build Spec instruction 8) — done.

Real audio: extract_features() runs eGeMAPSv02 low-level descriptors and
analyze_answers() performs quality-aware scoring. compute() accepts those
typed measurements or retains the legacy fixture interface for demo records.
Synthetic fixture scores are not measurements extracted from recordings.

Section 35 compliance ("what Claude Code must NOT implement"):
- no emotion-as-cheating inference
- no gender-based thresholds
- no absolute pitch thresholds — everything is relative to the candidate's
  own first-answers baseline.
"""
from __future__ import annotations
from statistics import mean, pstdev
from backend.config import SPEECH_WEIGHTS
from backend.schemas import ModuleResult, Evidence, Quality


def _zdev(value: float, baseline_mean: float, baseline_std: float) -> float:
    """
    Relative deviation from the candidate's own baseline, not an absolute
    threshold. The baseline is built from only a handful of early answers, so
    its natural sampling variance is floored at 8% of the mean — otherwise a
    tiny, meaningless fluctuation on a 2-answer baseline would look like a
    large z-score and falsely flag a perfectly normal candidate.
    """
    min_std = max(0.08 * abs(baseline_mean), 1e-3)
    baseline_std = max(baseline_std, min_std)
    return abs(value - baseline_mean) / baseline_std


def compute(per_answer_features: list[dict] | None) -> ModuleResult:
    """
    per_answer_features: list of dicts, one per answered question, each with
    keys: pitch_hz, loudness_db, speech_rate_wpm, pause_ratio, speech_silence_ratio
    The first 2 (or up to 1/3) answers form the candidate-specific baseline.
    """
    if per_answer_features and (hasattr(per_answer_features[0], "metrics") or "metrics" in per_answer_features[0]):
        from backend.audio_schemas import AnswerAudioFeatures
        return analyze_answers([AnswerAudioFeatures.model_validate(a) for a in per_answer_features]).result

    if not per_answer_features or len(per_answer_features) < 3:
        return ModuleResult(
            feature="speech_behavior", score=0, confidence=0.0, level="UNAVAILABLE", status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="speech_behavior", severity="low", message="Insufficient answers to build a candidate speech baseline.")],
            quality=Quality(data_quality=0.0, baseline_sufficient=False),
        )

    baseline_n = min(max(2, len(per_answer_features) // 3), len(per_answer_features) - 1)
    baseline = per_answer_features[:baseline_n]
    test = per_answer_features[baseline_n:]

    def bl(key):
        vals = [a[key] for a in baseline]
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    pm, ps = bl("pitch_hz")
    lm, ls = bl("loudness_db")
    rm, rs = bl("speech_rate_wpm")
    pause_m, pause_s = bl("pause_ratio")
    ratio_m, ratio_s = bl("speech_silence_ratio")

    per_q_devs = []
    for a in test:
        pitch_dev = _zdev(a["pitch_hz"], pm, ps)
        loud_dev = _zdev(a["loudness_db"], lm, ls)
        rate_dev = _zdev(a["speech_rate_wpm"], rm, rs)
        pause_dev = _zdev(a["pause_ratio"], pause_m, pause_s)
        ratio_dev = _zdev(a["speech_silence_ratio"], ratio_m, ratio_s)

        raw = (
            SPEECH_WEIGHTS.pitch_deviation * pitch_dev
            + SPEECH_WEIGHTS.loudness_deviation * loud_dev
            + SPEECH_WEIGHTS.speech_rate_deviation * rate_dev
            + SPEECH_WEIGHTS.pause_deviation * pause_dev
            + SPEECH_WEIGHTS.speech_ratio_deviation * ratio_dev
        )
        per_q_devs.append((a.get("question_id"), raw, rate_dev, pause_dev))

    avg_raw = mean(d[1] for d in per_q_devs)
    score = min(100.0, avg_raw * 40)  # normalize z-score-ish deviation to 0-100

    evidence = []
    worst = max(per_q_devs, key=lambda d: d[1]) if per_q_devs else None
    if worst and worst[1] > 1.5:
        evidence.append(Evidence(
            type="speech_rate", severity="high" if worst[1] > 2.5 else "medium",
            question_id=worst[0],
            message=f"Speech pattern deviated significantly from candidate baseline on {worst[0]} "
                    f"(rate/pause/pitch combined deviation z~={worst[1]:.1f}).",
        ))
    else:
        evidence.append(Evidence(type="speech_rate", severity="low",
                                  message="Speech pattern remained consistent with the candidate's own baseline."))

    level = "LOW" if score <= 30 else ("MEDIUM" if score <= 60 else "HIGH")
    confidence = 0.8 if len(test) >= 2 else 0.55
    return ModuleResult(
        feature="speech_behavior", score=round(score, 1), confidence=confidence, level=level,
        evidence=evidence, quality=Quality(data_quality=0.9, baseline_sufficient=True),
    )


def extract_features(*args, **kwargs):
    """Extract real pitch/loudness/timing measurements from prepared audio."""
    from backend.modules.speech_behavior.extractor import extract_features as extract
    return extract(*args, **kwargs)


def analyze_answers(answers):
    """Return detailed per-answer scores and the existing fusion contract."""
    from backend.modules.speech_behavior.scorer import score_answers
    return score_answers(answers)
