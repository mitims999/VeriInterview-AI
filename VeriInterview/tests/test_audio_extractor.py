"""Real FFmpeg checks; no model download required."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

from backend.adapters.audio_extractor import AudioError, extract_audio


def test_stereo_decode_preserves_amplitude_and_source(tmp_path):
    src, out = tmp_path / "original.wav", tmp_path / "prepared.wav"
    t = np.arange(48000) / 48000
    signal = .2 * np.sin(2 * np.pi * 220 * t)
    sf.write(src, np.column_stack([signal, signal]), 48000, subtype="FLOAT")
    original = src.read_bytes()
    result = extract_audio(src, out)
    audio, sr = sf.read(out)
    assert sr == 16000 and audio.ndim == 1
    assert result.duration_sec == pytest.approx(1, abs=.01)
    assert np.max(np.abs(audio)) == pytest.approx(.2, abs=.005)
    assert result.source_channels == 2
    assert src.read_bytes() == original


def test_video_extract_and_answer_slice(tmp_path):
    src = tmp_path / "interview.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                    "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(src)], check=True)
    result = extract_audio(src, tmp_path / "answer.wav", start_sec=1, end_sec=2)
    assert result.duration_sec == pytest.approx(1, abs=.02)
    assert result.offset_sec == 1


def test_invalid_missing_and_out_of_range_media(tmp_path):
    src = tmp_path / "broken.mp4"
    src.write_bytes(b"not media")
    with pytest.raises(AudioError):
        extract_audio(src, tmp_path / "out.wav")
    sf.write(src.with_suffix(".wav"), np.zeros(16000), 16000)
    with pytest.raises(AudioError, match="timestamps"):
        extract_audio(src.with_suffix(".wav"), tmp_path / "out.wav", end_sec=2)
    with pytest.raises(AudioError, match="overwrite"):
        extract_audio(src, src)


def test_video_without_audio(tmp_path):
    src = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=1",
                    "-c:v", "mpeg4", str(src)], check=True)
    with pytest.raises(AudioError, match="no audio track"):
        extract_audio(src, tmp_path / "out.wav")
