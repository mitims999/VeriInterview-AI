"""Member 1 contracts. Times use seconds on the original media timeline."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.schemas import ModuleResult


class AudioModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class TimeRange(AudioModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self


class AnswerWindow(TimeRange):
    question_id: str = Field(min_length=1, max_length=100)
    difficulty: str | None = None


class WordTimestamp(TimeRange):
    word: str
    probability: float = Field(ge=0, le=1)


class TranscriptSegment(TimeRange):
    segment_id: str
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscriptionResult(AudioModel):
    status: Literal["COMPLETE", "NO_SPEECH"]
    text: str = ""
    language: str | None = None
    language_probability: float = Field(default=0, ge=0, le=1)
    duration_sec: float = Field(ge=0)
    offset_sec: float = Field(default=0, ge=0)
    model: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    speech_intervals: list[TimeRange] = Field(default_factory=list)


class AudioQuality(AudioModel):
    score: float = Field(ge=0, le=1)
    duration_sec: float = Field(ge=0)
    speech_duration_sec: float = Field(ge=0)
    clipping_ratio: float = Field(ge=0, le=1)
    rms_dbfs: float
    estimated_snr_db: float | None = None
    warnings: list[str] = Field(default_factory=list)


class AcousticMetrics(AudioModel):
    pitch_mean_hz: float | None = None
    pitch_median_hz: float | None = None
    pitch_std_hz: float | None = None
    pitch_range_hz: float | None = None
    pitch_cv: float | None = None
    loudness_mean: float | None = None
    loudness_std: float | None = None
    loudness_range: float | None = None
    loudness_unit: str = "openSMILE auditory loudness (relative units)"
    rms_dbfs: float
    voiced_frame_ratio: float = Field(ge=0, le=1)
    speech_duration_sec: float = Field(ge=0)
    silence_duration_sec: float = Field(ge=0)
    speech_ratio: float = Field(ge=0, le=1)
    pause_ratio: float = Field(ge=0, le=1)
    pause_count: int = Field(ge=0)
    pause_mean_sec: float = Field(ge=0)
    pause_median_sec: float = Field(ge=0)
    pause_longest_sec: float = Field(ge=0)
    word_count: int | None = None
    speech_rate_wpm: float | None = None
    articulation_rate_wpm: float | None = None
    jitter: float | None = None
    shimmer_db: float | None = None
    hnr_db: float | None = None


class AcousticEvent(AudioModel):
    type: str
    timestamp_sec: float = Field(ge=0)
    change: float
    unit: str


class AnswerAudioFeatures(AudioModel):
    question_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    difficulty: str | None = None
    metrics: AcousticMetrics
    quality: AudioQuality
    events: list[AcousticEvent] = Field(default_factory=list)
    extractor: str = "openSMILE/eGeMAPSv02"


class SignalDeviation(AudioModel):
    feature: str
    value: float
    baseline: float
    signed_change: float
    unit: str
    z_score: float = Field(ge=0)


class AnswerSpeechScore(AudioModel):
    question_id: str
    timestamp_sec: float = Field(ge=0)
    status: Literal["OK", "INSUFFICIENT_EVIDENCE"]
    score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    signals: list[SignalDeviation] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SpeechBehaviorReport(AudioModel):
    result: ModuleResult
    baseline_question_ids: list[str] = Field(default_factory=list)
    answers: list[AnswerSpeechScore] = Field(default_factory=list)
    persistent_anomaly_count: int = Field(default=0, ge=0)
    score_version: str = "speech-heuristic-v1"
    confidence_kind: str = "measurement_reliability_heuristic"


class AudioAnalysisResult(AudioModel):
    transcription: TranscriptionResult
    answers: list[AnswerAudioFeatures]
    speech_behavior: SpeechBehaviorReport
    analysis_id: str | None = None
