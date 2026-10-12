"""Fixed baseline built only from earlier, usable answers of one session."""
from __future__ import annotations

from statistics import median, pstdev
from backend.audio_schemas import AnswerAudioFeatures
from .config import CONFIG


def usable(answer: AnswerAudioFeatures) -> bool:
    return (answer.quality.score >= CONFIG.minimum_quality
            and answer.quality.speech_duration_sec >= CONFIG.minimum_speech_sec
            and answer.end_sec - answer.start_sec >= CONFIG.minimum_answer_sec
            and answer.metrics.pitch_mean_hz is not None)


def feature_values(answer: AnswerAudioFeatures) -> dict[str, float | None]:
    m = answer.metrics
    return {
        "pitch_mean_hz": m.pitch_mean_hz, "pitch_std_hz": m.pitch_std_hz,
        "rms_dbfs": m.rms_dbfs, "loudness_std": m.loudness_std,
        "speech_rate_wpm": m.speech_rate_wpm,
        "pause_ratio": m.pause_ratio, "speech_ratio": m.speech_ratio,
    }


def summarize(answers: list[AnswerAudioFeatures]) -> dict[str, tuple[float, float]]:
    rows = [feature_values(a) for a in answers]
    summary = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r[key] is not None]
        if len(vals) != len(rows):
            continue
        center = median(vals)
        if key.startswith("pitch"):
            floor = max(CONFIG.pitch_floor_hz, .08 * abs(center))
        elif key == "rms_dbfs":
            floor = CONFIG.loudness_floor_db
        elif key == "speech_rate_wpm":
            floor = max(CONFIG.rate_floor_wpm, .08 * abs(center))
        elif key == "loudness_std":
            floor = max(.02, .15 * abs(center))
        else:
            floor = CONFIG.ratio_floor
        summary[key] = (center, max(pstdev(vals), floor))
    return summary
