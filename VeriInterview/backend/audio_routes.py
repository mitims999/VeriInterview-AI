"""Member 1 API. Heavy inference runs off the event loop."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from backend.adapters.audio_extractor import AudioError, AudioDependencyError, dependencies_available
from backend.audio_schemas import AnswerWindow, AudioAnalysisResult, TranscriptionResult
from backend.database.audio_models import AudioAnalysisRecord
from backend.database.models import Interview, InterviewAnswer
from backend.database.session import get_db, SessionLocal

router = APIRouter(prefix="/api", tags=["audio-intelligence"])
logger = logging.getLogger(__name__)


def parse_windows(raw: str) -> list[AnswerWindow] | None:
    try:
        windows = TypeAdapter(list[AnswerWindow]).validate_json(raw)
        return windows or None
    except ValidationError as exc:
        raise HTTPException(422, "answers must be a JSON list of question_id, start_sec and end_sec.") from exc


def persist(result: AudioAnalysisResult, db: Session, candidate_id: str | None = None,
            interview_id: int | None = None) -> AudioAnalysisResult:
    result.analysis_id = str(uuid4())
    if interview_id is not None:
        interview = db.get(Interview, interview_id)
        if interview is None:
            raise HTTPException(404, "Interview not found.")
        by_id = {str(q.id): q for q in interview.questions}
        if any(a.question_id not in by_id for a in result.answers):
            raise HTTPException(422, "Answer question IDs must belong to this interview.")
        from backend.modules.audio_service import words_in_window
        for answer in result.answers:
            window = AnswerWindow(question_id=answer.question_id, start_sec=answer.start_sec, end_sec=answer.end_sec)
            words = words_in_window(result.transcription, window)
            question = by_id[answer.question_id]
            if words:
                stored = question.answer
                if stored is None:
                    stored = InterviewAnswer(question_id=question.id)
                    db.add(stored)
                stored.transcript = " ".join(w.word for w in words)
                stored.answer_duration_sec = answer.end_sec - answer.start_sec
                # Question-end time is unknown; do not invent response latency.
    db.add(AudioAnalysisRecord(id=result.analysis_id, candidate_id=candidate_id,
        interview_id=interview_id, result_json=result.model_dump(mode="json")))
    db.commit()
    return result


async def save_upload(file: UploadFile, destination: Path) -> None:
    size = 0
    with destination.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 250 * 1024 * 1024:
                raise HTTPException(413, "Media files must be no larger than 250 MiB.")
            target.write(chunk)


@router.get("/audio/status")
def audio_status():
    packages = {}
    for name in ("opensmile", "faster-whisper", "soundfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"ffmpeg_available": dependencies_available(), "packages": packages}


@router.post("/audio/transcribe", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = File(...), language: str | None = Form(None)):
    try:
        from backend.modules.audio_service import transcribe_media
        with TemporaryDirectory(prefix="verivoice-upload-") as temp:
            source = Path(temp) / "input.media"
            await save_upload(file, source)
            return await asyncio.to_thread(transcribe_media, source, language)
    except (AudioDependencyError, ImportError) as exc:
        raise HTTPException(503, "Audio dependencies or Whisper model are unavailable. See docs/AUDIO_INTELLIGENCE.md.") from exc
    except AudioError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/speech-behavior/analyze", response_model=AudioAnalysisResult)
async def analyze(file: UploadFile = File(...), answers: str = Form("[]"),
                  language: str | None = Form(None), candidate_id: str | None = Form(None),
                  interview_id: int | None = Form(None), db: Session = Depends(get_db)):
    windows = parse_windows(answers)
    if interview_id is not None and db.get(Interview, interview_id) is None:
        raise HTTPException(404, "Interview not found.")
    try:
        from backend.modules.audio_service import analyze_media
        with TemporaryDirectory(prefix="verivoice-upload-") as temp:
            source = Path(temp) / "input.media"
            await save_upload(file, source)
            result = await asyncio.to_thread(analyze_media, source, windows, language)
        return persist(result, db, candidate_id, interview_id)
    except (AudioDependencyError, ImportError) as exc:
        raise HTTPException(503, "Audio dependencies or Whisper model are unavailable. See docs/AUDIO_INTELLIGENCE.md.") from exc
    except AudioError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/audio/analyses/{analysis_id}", response_model=AudioAnalysisResult)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    row = db.get(AudioAnalysisRecord, analysis_id)
    if not row:
        raise HTTPException(404, "Audio analysis not found.")
    return row.result_json


@router.websocket("/audio/live")
async def live(websocket: WebSocket):
    from backend.config import CORS_ORIGINS
    # Browsers send Origin; command-line/local test clients may omit it.
    origin = websocket.headers.get("origin")
    if origin and origin not in CORS_ORIGINS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        from backend.modules.audio_stream import LiveAudioSession
        session = LiveAudioSession()
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=90)
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                update = await asyncio.to_thread(session.push, message["bytes"])
                if update:
                    await websocket.send_json(update)
            else:
                control = json.loads(message.get("text") or "{}")
                action = control.get("action")
                if action in ("start", "start_answer"):
                    session.start_answer(control)
                    await websocket.send_json({"type": "ready", "question_id": session.question_id})
                elif action == "end_answer":
                    await websocket.send_json(await asyncio.to_thread(session.end_answer))
                elif action == "stop":
                    if session.active:
                        await websocket.send_json(await asyncio.to_thread(session.end_answer))
                    result = session.result()
                    with SessionLocal() as db:
                        persist(result, db)
                    await websocket.send_json({"type": "complete", "result": result.model_dump(mode="json")})
                    await websocket.close()
                    break
                else:
                    raise AudioError("Unknown audio control action.")
    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        await websocket.close(code=1008, reason="Audio session timed out.")
    except (AudioError, ValueError, ValidationError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)[:300]})
        await websocket.close(code=1008)
    except (AudioDependencyError, ImportError):
        await websocket.send_json({"type": "error", "message": "Audio dependencies or model unavailable. See docs/AUDIO_INTELLIGENCE.md."})
        await websocket.close(code=1011)
    except Exception:
        logger.exception("Live audio session failed")
        await websocket.send_json({"type": "error", "message": "Audio session failed. Please start a new session."})
        await websocket.close(code=1011)
