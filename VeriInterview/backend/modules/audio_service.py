"""File orchestration. Answer boundaries come from the interviewer, never ASR guesses."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from backend.adapters.audio_extractor import AudioError, extract_audio
from backend.adapters.whisper_adapter import transcribe_file
from backend.adapters.opensmile_adapter import extract_features, analyze_answers
from backend.audio_schemas import AnswerWindow, AudioAnalysisResult, TranscriptionResult


def validate_windows(windows: list[AnswerWindow], duration_sec: float) -> None:
    if len(windows) > 100 or len({w.question_id for w in windows}) != len(windows):
        raise AudioError("Use at most 100 answers with unique question IDs.")
    if any(w.end_sec > duration_sec + .05 for w in windows):
        raise AudioError("Answer timestamps extend beyond the recording.")
    if any(b.start_sec < a.end_sec for a, b in zip(windows, windows[1:])):
        raise AudioError("Answer windows must be chronological and must not overlap.")


def words_in_window(transcript: TranscriptionResult, window: AnswerWindow):
    return [w for s in transcript.segments for w in s.words
            if window.start_sec <= (w.start_sec + w.end_sec) / 2 < window.end_sec]


def transcribe_media(source: str | Path, language: str | None = None) -> TranscriptionResult:
    with TemporaryDirectory(prefix="verivoice-transcribe-") as temp:
        audio = extract_audio(source, Path(temp) / "prepared.wav")
        return transcribe_file(audio.path, language=language)


def analyze_media(source: str | Path, windows: list[AnswerWindow] | None = None,
                  language: str | None = None) -> AudioAnalysisResult:
    with TemporaryDirectory(prefix="verivoice-analysis-") as temp:
        audio = extract_audio(source, Path(temp) / "prepared.wav")
        windows = windows or [AnswerWindow(question_id="Q1", start_sec=0, end_sec=audio.duration_sec)]
        validate_windows(windows, audio.duration_sec)
        transcript = transcribe_file(audio.path, language=language)
        answers = []
        for i, window in enumerate(windows):
            answer = extract_audio(audio.path, Path(temp) / f"answer-{i}.wav",
                                   start_sec=window.start_sec, end_sec=window.end_sec)
            # If ASR recognized speech but produced no aligned words, WPM is
            # unavailable rather than a fabricated zero.
            words = words_in_window(transcript, window)
            count = len(words) if words or transcript.status == "NO_SPEECH" else None
            answers.append(extract_features(answer.path, window.question_id, word_count=count,
                offset_sec=window.start_sec, difficulty=window.difficulty))
        return AudioAnalysisResult(transcription=transcript, answers=answers,
                                   speech_behavior=analyze_answers(answers))
