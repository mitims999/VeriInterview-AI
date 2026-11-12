"""Real eGeMAPSv02 LLD extraction, with pitch converted per frame to Hz."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf

from backend.adapters.audio_extractor import AudioError, AudioDependencyError
from backend.adapters.whisper_adapter import speech_intervals
from backend.audio_schemas import AcousticMetrics, AcousticEvent, AnswerAudioFeatures
from .metrics import timing_metrics, finite_mean
from .quality import assess_quality

_LOCK = Lock()


@lru_cache(maxsize=1)
def _extractor():
    try:
        import opensmile
    except ImportError as exc:
        raise AudioDependencyError("Install requirements-audio.txt to enable openSMILE.") from exc
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)


def extract_features(path: str | Path, question_id: str, *, word_count: int | None = None,
                     offset_sec: float = 0, difficulty: str | None = None) -> AnswerAudioFeatures:
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise AudioError("Acoustic analysis requires prepared mono 16 kHz audio.")
    intervals = speech_intervals(audio)
    quality = assess_quality(audio, intervals)
    timing = timing_metrics(len(audio) / sr, intervals, word_count)
    metrics = AcousticMetrics(rms_dbfs=quality.rms_dbfs, voiced_frame_ratio=0, **timing)
    events = []
    # Short/silent audio remains explicitly unmeasurable; no invented zeros
    # for pitch, jitter, shimmer or speaking rate.
    if len(audio) >= 1600 and intervals:
        with _LOCK:
            frame = _extractor().process_signal(audio, sr)
        times = frame.index.get_level_values("start").total_seconds().to_numpy()
        speech_mask = np.zeros(len(frame), dtype=bool)
        for x in intervals:
            speech_mask |= (times >= x.start_sec) & (times < x.end_sec)
        semitones = frame["F0semitoneFrom27.5Hz_sma3nz"].to_numpy()
        voiced = speech_mask & np.isfinite(semitones) & (semitones > 0)
        pitch = 27.5 * np.power(2., semitones[voiced] / 12.)
        loud = frame["Loudness_sma3"].to_numpy()[speech_mask]
        loud = loud[np.isfinite(loud)]
        if pitch.size:
            metrics.pitch_mean_hz = float(pitch.mean())
            metrics.pitch_median_hz = float(np.median(pitch))
            metrics.pitch_std_hz = float(pitch.std())
            metrics.pitch_range_hz = float(np.percentile(pitch, 95) - np.percentile(pitch, 5))
            metrics.pitch_cv = float(pitch.std() / pitch.mean())
        if loud.size:
            metrics.loudness_mean = float(loud.mean())
            metrics.loudness_std = float(loud.std())
            metrics.loudness_range = float(np.percentile(loud, 95) - np.percentile(loud, 5))
        metrics.voiced_frame_ratio = float(voiced.sum() / max(1, speech_mask.sum()))
        for key, column in [("jitter", "jitterLocal_sma3nz"), ("shimmer_db", "shimmerLocaldB_sma3nz"),
                            ("hnr_db", "HNRdBACF_sma3nz")]:
            setattr(metrics, key, finite_mean(frame[column].to_numpy()[voiced]))

        # Sustained changes in adjacent one-second windows reduce frame-level
        # pitch-tracker spikes. These describe changes, not integrity judgments.
        for sec in range(1, int(len(audio) / sr)):
            before = voiced & (times >= sec - 1) & (times < sec)
            after = voiced & (times >= sec) & (times < sec + 1)
            if before.sum() >= 20 and after.sum() >= 20:
                delta = float(np.median(semitones[after]) - np.median(semitones[before]))
                if abs(delta) >= 4:
                    events.append(AcousticEvent(type="pitch_window_change", timestamp_sec=offset_sec + sec,
                                                change=round(delta, 2), unit="semitones"))
            left, right = audio[(sec-1)*sr:sec*sr], audio[sec*sr:(sec+1)*sr]
            if np.any(speech_mask & (times >= sec-1) & (times < sec)) and np.any(
                speech_mask & (times >= sec) & (times < sec+1)
            ):
                a = np.sqrt(np.mean(left.astype(float)**2))
                b = np.sqrt(np.mean(right.astype(float)**2))
                delta = float(20 * np.log10(max(b, 1e-8) / max(a, 1e-8)))
                if abs(delta) >= 8:
                    events.append(AcousticEvent(type="level_window_change", timestamp_sec=offset_sec + sec,
                                                change=round(delta, 2), unit="dBFS difference"))
    return AnswerAudioFeatures(question_id=question_id, start_sec=offset_sec,
        end_sec=offset_sec + len(audio) / sr, difficulty=difficulty,
        metrics=metrics, quality=quality, events=events[:100])
