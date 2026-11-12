"""Explainable prototype scoring. No emotion, identity or cheating inference."""
from __future__ import annotations

from statistics import mean

from backend.audio_schemas import (
    AnswerAudioFeatures, AnswerSpeechScore, SignalDeviation, SpeechBehaviorReport,
)
from backend.config import SPEECH_WEIGHTS
from backend.schemas import ModuleResult, Evidence, Quality
from .baseline import usable, feature_values, summarize
from .config import CONFIG


def score_answers(answers: list[AnswerAudioFeatures]) -> SpeechBehaviorReport:
    """Inputs are chronological answers from one candidate/session only."""
    if len({a.question_id for a in answers}) != len(answers):
        raise ValueError("Each answer must have a unique question_id.")
    if any(b.start_sec < a.end_sec - .05 for a, b in zip(answers, answers[1:])):
        raise ValueError("Answers must be chronological and must not overlap.")
    baseline, evaluated, evidence = [], [], []
    weights = {
        "pitch_mean_hz": SPEECH_WEIGHTS.pitch_deviation * .7,
        "pitch_std_hz": SPEECH_WEIGHTS.pitch_deviation * .3,
        "rms_dbfs": SPEECH_WEIGHTS.loudness_deviation * .7,
        "loudness_std": SPEECH_WEIGHTS.loudness_deviation * .3,
        "speech_rate_wpm": SPEECH_WEIGHTS.speech_rate_deviation,
        "pause_ratio": SPEECH_WEIGHTS.pause_deviation,
        "speech_ratio": SPEECH_WEIGHTS.speech_ratio_deviation,
    }
    labels = {
        "pitch_mean_hz": ("Mean pitch", "Hz"), "pitch_std_hz": ("Pitch variability", "Hz"),
        "rms_dbfs": ("Recording level", "dBFS"),
        "loudness_std": ("Loudness variability", "openSMILE relative units"),
        "speech_rate_wpm": ("Speech rate", "WPM"),
        "pause_ratio": ("Internal pause ratio", "fraction"),
        "speech_ratio": ("Speech-to-recording ratio", "fraction"),
    }
    consecutive, longest = 0, 0
    for answer in answers:
        entry = AnswerSpeechScore(question_id=answer.question_id, timestamp_sec=answer.start_sec,
                                 status="INSUFFICIENT_EVIDENCE", confidence=0)
        if not usable(answer):
            entry.evidence = ["Audio is too short or unreliable for baseline comparison."]
            consecutive = 0
        elif len(baseline) < CONFIG.baseline_answers:
            baseline.append(answer)
            entry.evidence = ["Used to establish the candidate baseline; not scored against itself."]
        else:
            reference = summarize(baseline)
            current = feature_values(answer)
            signals = []
            for key, (center, scale) in reference.items():
                value = current[key]
                if value is None:
                    continue
                signals.append(SignalDeviation(feature=key, value=value, baseline=center,
                    signed_change=value-center, unit=labels[key][1],
                    z_score=min(CONFIG.z_cap, abs(value-center)/scale)))
            coverage = sum(weights[s.feature] for s in signals)
            raw = sum(weights[s.feature] * s.z_score for s in signals) / max(coverage, 1e-8)
            score = round(min(100, raw / CONFIG.z_cap * 100), 1)
            baseline_quality = mean(a.quality.score for a in baseline)
            confidence = round(answer.quality.score * baseline_quality * coverage
                               * (.5 + .5 * min(1, len(baseline)/5)), 3)
            messages = []
            for s in signals:
                if s.z_score >= CONFIG.evidence_z:
                    label, unit = labels[s.feature]
                    message = (f"{label} changed from {s.baseline:.2f} to {s.value:.2f} {unit} "
                               f"relative to this candidate's baseline.")
                    messages.append(message)
                    evidence.append(Evidence(type=s.feature, severity="high" if s.z_score >= 3 else "medium",
                        message=message, question_id=answer.question_id, timestamp=answer.start_sec))
            if not messages:
                messages = ["Measured speech features remained close to the candidate baseline."]
            if answer.difficulty:
                messages.append(f"Question difficulty: {answer.difficulty}; speech changes may reflect task demands.")
            entry = AnswerSpeechScore(question_id=answer.question_id, timestamp_sec=answer.start_sec,
                status="OK", score=score, confidence=confidence, signals=signals, evidence=messages)
            consecutive = consecutive + 1 if score > CONFIG.medium_max else 0
            longest = max(longest, consecutive)
        evaluated.append(entry)

    valid = [a for a in evaluated if a.status == "OK"]
    if valid:
        score = round(mean(a.score for a in valid), 1)
        confidence = round(mean(a.confidence for a in valid), 3)
        level = "LOW" if score <= CONFIG.low_max else "MEDIUM" if score <= CONFIG.medium_max else "HIGH"
        result = ModuleResult(feature="speech_behavior", score=score, confidence=confidence, level=level,
            evidence=evidence or [Evidence(type="speech_behavior", severity="low",
                message="Speech behavior remained close to the candidate baseline.")],
            quality=Quality(data_quality=round(mean(a.quality.score for a in answers if usable(a)), 3),
                            baseline_sufficient=True))
    else:
        result = ModuleResult(feature="speech_behavior", score=0, confidence=0, level="UNAVAILABLE",
            status="INSUFFICIENT_EVIDENCE", quality=Quality(data_quality=0, baseline_sufficient=False),
            evidence=[Evidence(type="speech_behavior", severity="low",
                message="Need two usable baseline answers and at least one later usable answer.")])
    return SpeechBehaviorReport(result=result, baseline_question_ids=[a.question_id for a in baseline],
                                answers=evaluated, persistent_anomaly_count=longest)
