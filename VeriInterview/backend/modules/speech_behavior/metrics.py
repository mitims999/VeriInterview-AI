"""Timing measurements from VAD; rates use recognized words, not F0 segments."""
from __future__ import annotations

import numpy as np
from backend.audio_schemas import TimeRange
from .config import CONFIG


def timing_metrics(duration: float, intervals: list[TimeRange], word_count: int | None) -> dict:
    speech = min(duration, sum(i.end_sec - i.start_sec for i in intervals))
    gaps = [b.start_sec - a.end_sec for a, b in zip(intervals, intervals[1:])
            if b.start_sec - a.end_sec >= CONFIG.minimum_pause_sec]
    active_span = intervals[-1].end_sec - intervals[0].start_sec if intervals else 0
    return dict(
        speech_duration_sec=speech, silence_duration_sec=max(0, duration - speech),
        speech_ratio=speech / duration if duration else 0,
        # Internal pauses exclude leading/trailing silence; this is distinct
        # from 1 - speech_ratio, avoiding duplicate features by definition.
        pause_ratio=sum(gaps) / active_span if active_span else 0,
        pause_count=len(gaps), pause_mean_sec=float(np.mean(gaps)) if gaps else 0,
        pause_median_sec=float(np.median(gaps)) if gaps else 0,
        pause_longest_sec=max(gaps, default=0), word_count=word_count,
        speech_rate_wpm=word_count * 60 / duration if word_count is not None and duration else None,
        articulation_rate_wpm=word_count * 60 / speech if word_count is not None and speech else None,
    )


def finite_mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None
