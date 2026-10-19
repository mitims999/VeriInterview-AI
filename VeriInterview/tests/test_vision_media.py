"""
Unit tests for backend/vision/media.py (M3 task G-3).

The VAD / segment tests are pure numpy and always run. The decode tests build
a tiny synthetic clip with ffmpeg's built-in test sources, so they need
imageio-ffmpeg (requirements-vision.txt) but no datasets; they skip otherwise.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import media
from backend.vision.media import QuestionSegment, Segment

SR = 16000


def _noise(seconds, level=0.003, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level).astype(np.float32)


def _with_bursts(seconds, bursts):
    """Background noise plus 'speech-like' modulated tone bursts at the given (start, end) seconds."""
    x = _noise(seconds)
    t = np.arange(x.size) / SR
    for s, e in bursts:
        m = (t >= s) & (t < e)
        x[m] += 0.3 * np.sin(2 * np.pi * 180 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    return x


def test_detect_speech_finds_bursts_with_tight_boundaries():
    bursts = [(1.0, 2.5), (4.0, 6.0)]
    segs = media.detect_speech(_with_bursts(8.0, bursts))
    assert len(segs) == 2
    for seg, (s, e) in zip(segs, bursts):
        assert abs(seg.start - s) < 0.1 and abs(seg.end - e) < 0.1


def test_detect_speech_merges_short_gaps():
    segs = media.detect_speech(_with_bursts(5.0, [(1.0, 2.0), (2.15, 3.0)]))
    assert len(segs) == 1 and segs[0].start < 1.1 and segs[0].end > 2.9


def test_detect_speech_returns_nothing_for_silence_or_flat_noise():
    assert media.detect_speech(np.zeros(SR * 3, np.float32)) == []
    assert media.detect_speech(_noise(3.0, level=0.05)) == []
    assert media.detect_speech(np.zeros(100, np.float32)) == []


def test_question_mapping_and_clip():
    qs = [QuestionSegment("Q1", 0.0, 10.0), QuestionSegment("Q2", 10.0, 25.0)]
    assert media.question_at(3.0, qs) == "Q1"
    assert media.question_at(10.0, qs) == "Q2"
    assert media.question_at(30.0, qs) is None
    speech = [Segment(2.0, 12.0), Segment(20.0, 30.0)]
    assert media.clip(speech, 10.0, 25.0) == [Segment(10.0, 12.0), Segment(20.0, 25.0)]


# ---------------------------------------------------------------- decode tests

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")


def _make_clip(path, seconds=2, with_audio=True, size="320x240"):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100"]
    cmd += ["-c:v", "mpeg4", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if with_audio else []) + ["-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


def test_probe_reads_stream_metadata(tmp_path):
    info = media.probe(_make_clip(tmp_path / "a.mp4"))
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(25, abs=0.1)
    assert info.duration == pytest.approx(2.0, abs=0.15)
    assert info.has_audio and info.audio_sample_rate == 44100


def test_iter_frames_resamples_downscales_and_timestamps(tmp_path):
    p = _make_clip(tmp_path / "a.mp4")
    frames = list(media.iter_frames(p, fps=5, max_width=160))
    assert len(frames) == 10
    assert frames[0].image.shape == (120, 160, 3) and frames[0].image.dtype == np.uint8
    assert [round(f.t, 2) for f in frames[:3]] == [0.0, 0.2, 0.4]
    window = list(media.iter_frames(p, fps=5, start=1.0, end=1.6))
    assert len(window) == 3 and window[0].t == pytest.approx(1.0)


def test_load_audio_is_16k_mono_float(tmp_path):
    audio = media.load_audio(_make_clip(tmp_path / "a.mp4"))
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert audio.size / SR == pytest.approx(2.0, abs=0.1)
    assert 0.1 < np.abs(audio).max() <= 1.0


def test_missing_audio_and_bad_files_raise_typed_errors(tmp_path):
    silent_video = _make_clip(tmp_path / "noaudio.mp4", with_audio=False)
    assert not media.probe(silent_video).has_audio
    with pytest.raises(media.NoAudio):
        media.load_audio(silent_video)
    with pytest.raises(media.DecodeError):
        media.probe(str(tmp_path / "does_not_exist.mp4"))
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    with pytest.raises(media.DecodeError):
        media.probe(str(junk))
