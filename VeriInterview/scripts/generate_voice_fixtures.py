"""Create controlled acoustic fixtures from a consented synthetic speech WAV.

These transformations test measurements and reliability, not cheating detection.
Generated audio stays outside Git. Supply a 16 kHz mono synthetic speech file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def generate(source: Path, output: Path):
    audio, sr = sf.read(source, dtype="float32")
    if sr != 16000 or audio.ndim != 1 or len(audio) < 16000:
        raise ValueError("Supply at least one second of synthetic mono 16 kHz speech.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite samples.")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = audio * .4
    half = len(base) // 2
    cases = {
        "baseline_1": (base, "Establish baseline; no anomaly score"),
        "baseline_2": (base.copy(), "Establish baseline; no anomaly score"),
        "normal": (base.copy(), "Small deviation from identical baseline"),
        "level_shift": (audio, "Recording level increases by approximately 8 dB"),
        "rate_pitch_shift": (resample_poly(base, 4, 5), "Pace and pitch increase together; not independent changes"),
        "pause_shift": (np.concatenate([base[:half], np.zeros(32000), base[half:]]), "Added two-second internal pause"),
        "noisy_audio": (base + rng.normal(0, .08, len(base)), "Reduced measurement reliability"),
        "clipped_audio": (np.clip(base * 20, -1, 1), "Reduced reliability due to clipping"),
        "short_audio": (base[:8000], "Insufficient speech; no anomaly score"),
        "silence": (np.zeros(64000), "No speech or pitch measurement"),
    }
    rows = []
    for index, (condition, (samples, expected)) in enumerate(cases.items(), 1):
        filename = f"VB_{index:03}.wav"
        sf.write(output / filename, samples, sr, subtype="FLOAT")
        rows.append(dict(sample_id=f"VB_{index:03}", speaker_id="synthetic_01", condition=condition,
                         filename=filename, duration_sec=len(samples) / sr, expected_behavior=expected))
    (output / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with (output / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("storage/audio_analysis/fixtures"))
    args = parser.parse_args()
    print(f"Generated {len(generate(args.source, args.output))} controlled fixtures in {args.output}")
