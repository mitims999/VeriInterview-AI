"""Real faster-whisper inference with Silero VAD and word timestamps.

Model loading is lazy, cached, and offline by default. Never substitute a
synthetic transcript when the dependency/model is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf

from backend.adapters.audio_extractor import AudioDependencyError, AudioError
from backend.audio_schemas import TimeRange, WordTimestamp, TranscriptSegment, TranscriptionResult

SAMPLE_RATE = 16000
_MODEL_LOCK = Lock()
_INFERENCE_LOCK = Lock()
_MODELS: dict = {}


def speech_intervals(audio: np.ndarray) -> list[TimeRange]:
    """Silero speech activity; unvoiced consonants are not treated as pauses."""
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
    except ImportError as exc:
        raise AudioDependencyError("Install requirements-audio.txt to enable Silero VAD.") from exc
    if audio.size < 512 or float(np.max(np.abs(audio))) < 1e-5:
        return []
    chunks = get_speech_timestamps(audio.astype(np.float32), VadOptions(
        min_speech_duration_ms=150, min_silence_duration_ms=300, speech_pad_ms=30,
    ))
    return [TimeRange(start_sec=c["start"] / SAMPLE_RATE, end_sec=c["end"] / SAMPLE_RATE)
            for c in chunks if c["end"] > c["start"]]


def get_model():
    model_name = os.environ.get("WHISPER_MODEL", "base")
    cache = os.environ.get("WHISPER_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "storage" / "models"))
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    allow_download = os.environ.get("WHISPER_ALLOW_DOWNLOAD", "0") == "1"
    key = model_name, cache, device, compute, allow_download
    with _MODEL_LOCK:
        if key not in _MODELS:
            try:
                from faster_whisper import WhisperModel
                _MODELS[key] = WhisperModel(model_name, device=device, compute_type=compute,
                    download_root=cache, local_files_only=not allow_download, cpu_threads=4)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                raise AudioDependencyError(
                    "Whisper could not load. Install requirements-audio.txt and run "
                    "scripts/download_audio_model.py for the configured WHISPER_MODEL."
                ) from exc
        return _MODELS[key]


def transcribe_audio(audio: np.ndarray, *, offset_sec: float = 0,
                     language: str | None = None, segment_prefix: str = "segment") -> TranscriptionResult:
    if audio.ndim != 1 or not np.isfinite(audio).all() or not 0 < len(audio) <= 3600 * SAMPLE_RATE:
        raise AudioError("Transcription requires finite mono 16 kHz audio, up to 60 minutes.")
    intervals = speech_intervals(audio)
    model_name = os.environ.get("WHISPER_MODEL", "base")
    duration = len(audio) / SAMPLE_RATE
    common = dict(duration_sec=duration, offset_sec=offset_sec, model=model_name,
                  speech_intervals=[TimeRange(start_sec=x.start_sec + offset_sec,
                                              end_sec=x.end_sec + offset_sec) for x in intervals])
    if not intervals:
        return TranscriptionResult(status="NO_SPEECH", **common)
    model = get_model()
    segments_out = []
    # Materialize the generator while holding the lock: inference is lazy.
    with _INFERENCE_LOCK:
        try:
            segments, info = model.transcribe(audio.astype(np.float32), language=language,
                beam_size=5, word_timestamps=True, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False)
            for i, segment in enumerate(segments):
                start, end = max(0, segment.start), min(duration, segment.end)
                if end <= start or not segment.text.strip():
                    continue
                words = []
                for w in segment.words or []:
                    ws, we = max(start, w.start), min(end, w.end)
                    if we > ws:
                        words.append(WordTimestamp(word=w.word.strip(), start_sec=offset_sec + ws,
                            end_sec=offset_sec + we, probability=float(np.clip(w.probability, 0, 1))))
                segments_out.append(TranscriptSegment(segment_id=f"{segment_prefix}-{i}",
                    start_sec=offset_sec + start, end_sec=offset_sec + end,
                    text=segment.text.strip(), words=words))
        except (ValueError, RuntimeError) as exc:
            raise AudioError("Whisper could not transcribe this audio.") from exc
    return TranscriptionResult(status="COMPLETE" if segments_out else "NO_SPEECH",
        text=" ".join(s.text for s in segments_out), language=info.language,
        language_probability=float(np.clip(info.language_probability, 0, 1)),
        segments=segments_out, **common)


def transcribe_file(path: str | Path, *, offset_sec: float = 0,
                    language: str | None = None) -> TranscriptionResult:
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise AudioError("Prepare the recording with audio_extractor before transcription.")
    return transcribe_audio(audio, offset_sec=offset_sec, language=language)
