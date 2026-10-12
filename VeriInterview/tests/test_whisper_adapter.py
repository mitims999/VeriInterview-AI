from types import SimpleNamespace as NS

import numpy as np
import pytest

from backend.adapters import whisper_adapter as whisper
from backend.audio_schemas import TimeRange


def test_silent_audio_has_no_invented_transcript(monkeypatch):
    monkeypatch.setattr(whisper, "get_model", lambda: pytest.fail("Silence must not invoke Whisper"))
    result = whisper.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert result.status == "NO_SPEECH"
    assert not result.text and not result.segments


def test_words_and_segments_share_original_timeline(monkeypatch):
    monkeypatch.setattr(whisper, "speech_intervals", lambda _: [TimeRange(start_sec=.1, end_sec=.9)])
    class Model:
        def transcribe(self, audio, **kwargs):
            assert kwargs["word_timestamps"] and kwargs["vad_filter"]
            segment = NS(start=.1, end=.9, text=" Hello world", words=[
                NS(start=.1, end=.4, word=" Hello", probability=.9),
                NS(start=.4, end=.9, word=" world", probability=.8)])
            return iter([segment]), NS(language="en", language_probability=.99)
    monkeypatch.setattr(whisper, "get_model", Model)
    result = whisper.transcribe_audio(np.ones(16000, dtype=np.float32), offset_sec=12)
    assert result.text == "Hello world"
    assert result.segments[0].start_sec == pytest.approx(12.1)
    assert result.segments[0].words[-1].end_sec == pytest.approx(12.9)
    assert result.speech_intervals[0].start_sec == pytest.approx(12.1)


def test_real_silero_does_not_treat_tone_as_speech():
    t = np.arange(32000) / 16000
    result = whisper.transcribe_audio((.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    assert result.status == "NO_SPEECH"


def test_nan_audio_rejected():
    with pytest.raises(ValueError, match="finite mono"):
        whisper.transcribe_audio(np.full(16000, np.nan))
