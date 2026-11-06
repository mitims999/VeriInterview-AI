"""Decode local media to mono 16 kHz float WAV, without gain normalization."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioError(ValueError):
    """Invalid or unsupported media; safe to present to an API caller."""


class AudioDependencyError(RuntimeError):
    """A required local binary or model is unavailable."""


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    duration_sec: float
    sample_rate: int = 16000
    channels: int = 1
    source_sample_rate: int = 0
    source_channels: int = 0
    offset_sec: float = 0.0


def _run(args: list[str], timeout: int = 120) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True,
                              timeout=timeout).stdout
    except FileNotFoundError as exc:
        raise AudioDependencyError("Install FFmpeg and ffprobe before processing media.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Media processing exceeded the time limit.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError("Unable to decode this media file. Check its format and audio track.") from exc


def probe_audio(source: str | Path) -> dict:
    source = Path(source).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise AudioError("The media file is missing or empty.")
    if source.stat().st_size > 250 * 1024 * 1024:
        raise AudioError("Media files must be no larger than 250 MiB.")
    raw = _run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                "-select_streams", "a:0", "-show_streams", "-show_format",
                "-of", "json", str(source)], timeout=30)
    try:
        data = json.loads(raw)
        streams = data.get("streams", [])
        if not streams:
            raise AudioError("This file has no audio track.")
        stream = streams[0]
        duration = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or duration <= 0:
            raise AudioError("The audio duration could not be determined.")
        if duration > 3600:
            raise AudioError("Audio must be no longer than 60 minutes.")
        return {"duration_sec": duration, "sample_rate": int(stream["sample_rate"]),
                "channels": int(stream["channels"])}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AudioError):
            raise
        raise AudioError("The file contains invalid audio metadata.") from exc


def extract_audio(source: str | Path, destination: str | Path, *,
                  start_sec: float = 0, end_sec: float | None = None) -> PreparedAudio:
    """Caller owns destination and its cleanup. Answer boundaries are explicit."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise AudioError("The prepared audio must not overwrite the original recording.")
    info = probe_audio(source)
    end_sec = info["duration_sec"] if end_sec is None else end_sec
    if not all(math.isfinite(v) for v in (start_sec, end_sec)) or not (
        0 <= start_sec < end_sec <= info["duration_sec"] + 0.05
    ):
        raise AudioError("Answer timestamps must lie within the recording and have positive duration.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # FFmpeg's default stereo downmix applies a ~3 dB boost to identical
    # channels. Explicit arithmetic averaging avoids that gain change.
    channels = info["channels"]
    mix = "+".join(f"{1 / channels:.12f}*c{i}" for i in range(channels))
    _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
          "-i", str(source), "-ss", str(start_sec), "-t", str(end_sec - start_sec),
          "-map", "0:a:0", "-vn", "-af", f"pan=mono|c0={mix}", "-ar", "16000",
          "-c:a", "pcm_f32le", str(destination)])
    decoded = probe_audio(destination)
    return PreparedAudio(destination, decoded["duration_sec"],
                         source_sample_rate=info["sample_rate"],
                         source_channels=info["channels"], offset_sec=start_sec)


def dependencies_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
