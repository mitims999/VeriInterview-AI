from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechConfig:
    baseline_answers: int = 2
    minimum_quality: float = .35
    minimum_speech_sec: float = 1.0
    minimum_answer_sec: float = 3.0
    z_cap: float = 4.0
    evidence_z: float = 1.5
    low_max: float = 34
    medium_max: float = 64
    minimum_pause_sec: float = .3
    # Measurement-scale floors prevent zero-variance baselines exploding.
    pitch_floor_hz: float = 8
    loudness_floor_db: float = 2
    rate_floor_wpm: float = 15
    ratio_floor: float = .05


CONFIG = SpeechConfig()
