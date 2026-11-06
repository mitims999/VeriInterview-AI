"""
Media I/O for the M3 vision pipeline: probe a video, stream frames at a target
fps, load mono audio, find speech segments, and map timestamps to interview
questions.

Everything runs through the ffmpeg binary bundled with `imageio-ffmpeg`
(no system ffmpeg needed) over pipes. No frames or audio are written to
disk (Master Spec Section 29).

Frames are yielded as BGR uint8 arrays (OpenCV / InsightFace convention).
MediaPipe callers convert with `frame.image[:, :, ::-1]`.

M1 owns `audio_extractor.py` (Whisper path). `load_audio()` here is the M3
fallback with the same 16 kHz mono convention, so either can feed SyncNet.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np


class MediaError(RuntimeError):
    """Base class for media problems the adapters turn into INSUFFICIENT_EVIDENCE."""


class DecodeError(MediaError):
    """File missing, unreadable or not a supported video."""


class NoVideo(MediaError):
    """File has no video stream."""


class NoAudio(MediaError):
    """File has no audio stream."""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration: float          # seconds (0.0 if ffmpeg could not tell)
    fps: float               # native frame rate
    width: int
    height: int
    video_codec: str
    has_audio: bool
    audio_sample_rate: Optional[int]


@dataclass(frozen=True)
class Frame:
    index: int               # index within this decode pass
    t: float                 # timestamp in seconds from the start of the file
    image: np.ndarray        # H x W x 3, uint8, BGR


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class QuestionSegment:
    question_id: str
    start: float
    end: float


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as e:  # pragma: no cover - exercised only without vision deps
        raise MediaError("imageio-ffmpeg not installed (pip install -r requirements-vision.txt)") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\S+.*?Video:\s*([^\s,]+).*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)")
_AUDIO_RE = re.compile(r"Stream #\S+.*?Audio:\s*[^,]+,\s*(\d+)\s*Hz")


def probe(path: str) -> VideoInfo:
    """Read stream metadata. Raises DecodeError / NoVideo."""
    proc = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    err = proc.stderr
    if "No such file" in err or "Invalid data found" in err or "Stream #" not in err:
        raise DecodeError(f"cannot read media file: {path}")
    vm = None
    for line in err.splitlines():
        if "Video:" in line:
            vm = _VIDEO_RE.search(line)
            if vm:
                fps_m = _FPS_RE.search(line)
                break
    if not vm:
        raise NoVideo(f"no video stream in {path}")
    dm = _DURATION_RE.search(err)
    duration = (int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))) if dm else 0.0
    am = next((m for m in (_AUDIO_RE.search(l) for l in err.splitlines() if "Audio:" in l) if m), None)
    return VideoInfo(
        path=path, duration=duration, fps=float(fps_m.group(1)) if fps_m else 0.0,
        width=int(vm.group(2)), height=int(vm.group(3)), video_codec=vm.group(1),
        has_audio=am is not None or any("Audio:" in l for l in err.splitlines()),
        audio_sample_rate=int(am.group(1)) if am else None,
    )


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def iter_frames(path: str, fps: Optional[float] = None, max_width: Optional[int] = None,
                start: float = 0.0, end: Optional[float] = None,
                info: Optional[VideoInfo] = None) -> Iterator[Frame]:
    """Stream frames as BGR arrays.

    fps: resample to this rate (None = native rate). max_width: downscale wider
    frames, keeping the aspect ratio. start/end: decode only that window (seconds).
    Memory stays bounded at one frame.
    """
    info = info or probe(path)
    w, h = info.width, info.height
    if max_width and w > max_width:
        w, h = _even(max_width), _even(info.height * max_width / info.width)
    rate = fps or info.fps or 25.0

    filters = [f"fps={rate}"] if fps else []
    if (w, h) != (info.width, info.height):
        filters.append(f"scale={w}:{h}")
    cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd += ["-an"] + (["-vf", ",".join(filters)] if filters else []) + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2)
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield Frame(index=i, t=start + i / rate, image=np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if i == 0:
        raise DecodeError(f"no frames decoded from {path}")


def load_audio(path: str, sr: int = 16000, start: float = 0.0, end: Optional[float] = None,
               info: Optional[VideoInfo] = None) -> np.ndarray:
    """Mono float32 audio in [-1, 1] at `sr` Hz. Raises NoAudio if there is no audio stream."""
    info = info or probe(path)
    if not info.has_audio:
        raise NoAudio(f"no audio stream in {path}")
    cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise NoAudio(f"could not decode audio from {path}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def detect_speech(audio: np.ndarray, sr: int = 16000, frame_s: float = 0.03, hop_s: float = 0.01,
                  min_speech_s: float = 0.20, merge_gap_s: float = 0.30, pad_s: float = 0.05,
                  min_snr_db: float = 6.0, rel_threshold: float = 0.55) -> list[Segment]:
    """Energy-based voice-activity detection (fallback when no Whisper timestamps).

    The threshold adapts to each recording: noise floor = 10th percentile of
    frame energy, speech must rise clearly above it. Returns merged, padded
    segments. Returns [] for silence or near-constant noise.
    """
    n, hop = int(frame_s * sr), int(hop_s * sr)
    if audio.size < n:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop]
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    if loud - floor < min_snr_db or loud < -55:
        return []
    active = db > floor + max(min_snr_db, rel_threshold * (loud - floor))

    # rising/falling edges of the active mask -> [start frame, end frame) runs
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    runs = zip(edges[::2], edges[1::2])
    merged: list[list[float]] = []
    for a, b in runs:
        s, e = a * hop / sr, ((b - 1) * hop + n) / sr
        if merged and s - merged[-1][1] < merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    total = audio.size / sr
    return [Segment(float(max(0.0, s - pad_s)), float(min(total, e + pad_s))) for s, e in merged if e - s >= min_speech_s]


def question_at(t: float, questions: Sequence[QuestionSegment]) -> Optional[str]:
    """question_id whose [start, end) window contains t, else None."""
    for q in questions:
        if q.start <= t < q.end:
            return q.question_id
    return None


def clip(segments: Sequence[Segment], start: float, end: float) -> list[Segment]:
    """Intersect speech segments with a window, e.g. speech inside one answer."""
    out = []
    for s in segments:
        a, b = max(s.start, start), min(s.end, end)
        if b > a:
            out.append(Segment(a, b))
    return out


def whole_video_question(info: VideoInfo) -> list[QuestionSegment]:
    """Fallback when no per-question timings exist (Task M3_TASKS Q3): one segment for the whole file."""
    return [QuestionSegment("ALL", 0.0, info.duration)]
