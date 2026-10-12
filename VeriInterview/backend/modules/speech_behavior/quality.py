"""Measurement reliability heuristic; never a calibrated cheating probability."""
from __future__ import annotations

import numpy as np
from backend.audio_schemas import AudioQuality, TimeRange


def assess_quality(audio: np.ndarray, intervals: list[TimeRange], sample_rate: int = 16000) -> AudioQuality:
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio.astype(float) ** 2))) if len(audio) else 0
    rms_dbfs = float(20 * np.log10(max(rms, 1e-8)))
    clipping = float(np.mean(np.abs(audio) >= .999)) if len(audio) else 0
    speech_duration = sum(x.end_sec - x.start_sec for x in intervals)
    warnings = []
    score = min(1., speech_duration / 5) * min(1., duration / 3)
    if speech_duration < 1:
        warnings.append("Insufficient speech for reliable acoustic analysis.")
    if duration < 3:
        warnings.append("Recording is shorter than three seconds.")
    if rms_dbfs < -45:
        score *= .4
        warnings.append("Recording level is very low.")
    if clipping > .001:
        score *= max(.05, 1 - clipping * 15)
        warnings.append("Clipping may distort pitch and loudness measurements.")

    # Frame-based speech/noise energy contrast, only when both are observed.
    # This is a rough recording-quality estimate, not a ground-truth SNR.
    size = 320
    usable = len(audio) // size * size
    frames = audio[:usable].reshape(-1, size)
    power = np.mean(frames.astype(float) ** 2, axis=1)
    centers = (np.arange(len(frames)) * size + size / 2) / sample_rate
    mask = np.zeros(len(frames), dtype=bool)
    for x in intervals:
        mask |= (centers >= x.start_sec) & (centers < x.end_sec)
    snr = None
    if np.sum(mask) >= 10 and np.sum(~mask) >= 10:
        signal = float(np.median(power[mask]))
        noise = float(np.median(power[~mask]))
        snr = float(np.clip(10 * np.log10(max(signal, 1e-12) / max(noise, 1e-12)), -40, 80))
        if snr < 10:
            score *= max(.1, (snr + 5) / 15)
            warnings.append("Low estimated speech-to-background energy contrast.")
    else:
        warnings.append("Background noise could not be estimated reliably.")
        score *= .9
    warnings.append("Single-speaker input is assumed; diarization is not performed.")
    return AudioQuality(score=round(float(np.clip(score, 0, 1)), 3), duration_sec=duration,
        speech_duration_sec=speech_duration, clipping_ratio=clipping, rms_dbfs=rms_dbfs,
        estimated_snr_db=snr, warnings=warnings)
