"""Bounded answer-level live ASR. Revisions replace earlier text, never append.

PCM packets are transport chunks, not independent media files. Each answer is
redecoded for provisional updates and finalized at an explicit answer boundary;
this avoids overlap-induced duplicated text. Maximum answer duration: 120 s.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from backend.adapters.audio_extractor import AudioError
from backend.adapters.whisper_adapter import transcribe_audio
from backend.adapters.opensmile_adapter import extract_features, analyze_answers
from backend.audio_schemas import AudioAnalysisResult, TranscriptionResult


class LiveAudioSession:
    def __init__(self):
        self.active = False
        self.question_id = None
        self.language = None
        self.difficulty = None
        self.buffer = bytearray()
        self.offset_sec = 0.
        self.last_update_samples = 0
        self.revision = 0
        self.answers = []
        self.transcripts = []

    def start_answer(self, control: dict):
        if self.active:
            raise AudioError("Finish the current answer before starting another.")
        if control.get("sample_rate") != 16000 or control.get("format") != "pcm_s16le":
            raise AudioError("Live audio must be mono signed 16-bit little-endian PCM at 16000 Hz.")
        question_id = control.get("question_id")
        if not isinstance(question_id, str) or not 1 <= len(question_id) <= 100:
            raise AudioError("Provide a question_id of 1–100 characters.")
        if any(a.question_id == question_id for a in self.answers):
            raise AudioError("Each answer must use a different question_id.")
        language = control.get("language")
        difficulty = control.get("difficulty")
        if language is not None and (not isinstance(language, str) or len(language) > 10):
            raise AudioError("Invalid language code.")
        if difficulty is not None and (not isinstance(difficulty, str) or len(difficulty) > 100):
            raise AudioError("Invalid difficulty.")
        if self.offset_sec >= 3600:
            raise AudioError("Live sessions are limited to 60 minutes.")
        if len(self.answers) >= 100:
            raise AudioError("Live sessions are limited to 100 answers.")
        self.question_id, self.language, self.difficulty = question_id, language, difficulty
        self.buffer.clear()
        self.last_update_samples, self.revision = 0, 0
        self.active = True

    def _transcribe(self):
        audio = np.frombuffer(self.buffer, dtype="<i2").astype(np.float32) / 32768
        return transcribe_audio(audio, offset_sec=self.offset_sec, language=self.language,
                                segment_prefix=self.question_id)

    def push(self, packet: bytes) -> dict | None:
        if not self.active:
            raise AudioError("Start an answer before sending audio.")
        if not packet or len(packet) % 2 or len(packet) > 64000:
            raise AudioError("Send complete PCM samples in packets of at most two seconds.")
        if len(self.buffer) + len(packet) > 120 * 16000 * 2:
            raise AudioError("Answers are limited to 120 seconds. Finish this answer and start another.")
        if self.offset_sec + (len(self.buffer) + len(packet)) / 32000 > 3600:
            raise AudioError("Live sessions are limited to 60 minutes.")
        self.buffer.extend(packet)
        samples = len(self.buffer) // 2
        if samples - self.last_update_samples < 3 * 16000:
            return None
        self.last_update_samples = samples
        self.revision += 1
        result = self._transcribe()
        return {"type": "transcript", "question_id": self.question_id,
                "revision": self.revision, "final": False,
                "transcription": result.model_dump(mode="json")}

    def end_answer(self) -> dict:
        if not self.active or not self.buffer:
            raise AudioError("No active audio answer to finalize.")
        result = self._transcribe()
        audio = np.frombuffer(self.buffer, dtype="<i2").astype(np.float32) / 32768
        with TemporaryDirectory(prefix="verivoice-live-") as temp:
            path = Path(temp) / "answer.wav"
            sf.write(path, audio, 16000, subtype="FLOAT")
            count = sum(len(s.words) for s in result.segments)
            features = extract_features(path, self.question_id,
                word_count=count if count or result.status == "NO_SPEECH" else None,
                offset_sec=self.offset_sec, difficulty=self.difficulty)
        self.answers.append(features)
        self.transcripts.append(result)
        self.offset_sec += len(audio) / 16000
        self.active = False
        self.buffer.clear()
        self.revision += 1
        return {"type": "transcript", "question_id": self.question_id, "revision": self.revision,
                "final": True, "transcription": result.model_dump(mode="json"),
                "answer": features.model_dump(mode="json"),
                "speech_behavior": analyze_answers(self.answers).model_dump(mode="json")}

    def result(self) -> AudioAnalysisResult:
        if not self.transcripts:
            raise AudioError("Record at least one answer before finishing the session.")
        languages = [t for t in self.transcripts if t.language]
        transcript = TranscriptionResult(
            status="COMPLETE" if any(t.text for t in self.transcripts) else "NO_SPEECH",
            text=" ".join(t.text for t in self.transcripts if t.text),
            duration_sec=self.offset_sec, model=self.transcripts[0].model,
            language=languages[0].language if languages else None,
            language_probability=min(t.language_probability for t in languages) if languages else 0,
            segments=[s for t in self.transcripts for s in t.segments],
            speech_intervals=[s for t in self.transcripts for s in t.speech_intervals])
        return AudioAnalysisResult(transcription=transcript, answers=self.answers,
                                   speech_behavior=analyze_answers(self.answers))
