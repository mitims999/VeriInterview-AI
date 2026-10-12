"""
VeriInterview FastAPI application (Master Spec Section 20 / Build Spec Section 6).

Endpoints per Master Spec Section 22. Synchronous per-request execution is
acceptable for the hackathon prototype scale (single interview at a time);
heavy modules are pre-computed adapters over structured signals rather than
live media processing, so no background job queue is required to keep the
UI responsive.
"""
from __future__ import annotations
import json
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.config import CORS_ORIGINS, DATA_DIR
from backend.database.session import init_db, get_db
from backend.database import models
from backend.modules.resume import parse_resume_bytes, build_profile
from backend.modules.pipeline import run_full_analysis
from backend.modules import intake_parser
from backend.modules import smart_transcript_parser
from backend.modules import keyword_detector
from backend.adapters import video_intake

app = FastAPI(title="VeriInterview API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

SAMPLES_DIR = os.path.join(os.path.dirname(DATA_DIR), "samples")
if os.path.isdir(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Synthetic demo dataset (Master Spec Section 25 / Build Spec Section 5.2)
# ---------------------------------------------------------------------------

@app.get("/api/demo/candidates")
def list_demo_candidates():
    out = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                rec = json.load(f)
            out.append({"id": fname[:-5], "candidate_name": rec.get("candidate_name"), "label": rec.get("scenario_label", "")})
    return out


@app.get("/api/demo/candidates/{candidate_id}/report")
def demo_candidate_report(candidate_id: str):
    path = os.path.join(DATA_DIR, f"{candidate_id}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Unknown demo candidate")
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    result = run_full_analysis(record)
    result["scenario_label"] = record.get("scenario_label", "")
    result["expected_outcome"] = record.get("expected_outcome", "")
    return result


@app.get("/api/samples")
def list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return [f for f in sorted(os.listdir(SAMPLES_DIR)) if not f.startswith(".")]


# ---------------------------------------------------------------------------
# Manual intake: interviewer uploads resume + ideal-answer doc + transcript
# + (optional) interview video, gets back a live-computed report.
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze_intake(
    candidate_name: str = Form(...),
    job_description: str = Form(""),
    resume: UploadFile = File(...),
    ideal_answers: UploadFile = File(...),
    transcript: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
):
    resume_bytes = await resume.read()
    resume_text = parse_resume_bytes(resume.filename, resume_bytes)

    ideal_text = (await ideal_answers.read()).decode("utf-8", errors="ignore")
    transcript_text = (await transcript.read()).decode("utf-8", errors="ignore")

    ideal = intake_parser.parse_ideal_answers(ideal_text)
    trans = intake_parser.parse_transcript(transcript_text)
    qa_list = intake_parser.build_qa_list(ideal, trans)

    known_skills = sorted({qa["skill"] for qa in qa_list})
    resume_claims = intake_parser.infer_resume_claims(resume_text, known_skills)

    av_signals = None
    identity_signals = None
    if video is not None and video.filename:
        video_bytes = await video.read()
        if video_bytes:
            av_signals, identity_signals = video_intake.derive_av_signals(video_bytes)

    record = {
        "candidate_name": candidate_name,
        "job_description": job_description,
        "resume_text": resume_text,
        "resume_claims": resume_claims,
        "qa_list": qa_list,
        "speech_features": None,     # no raw audio captured in this intake flow
        "behavior_signals": None,    # no webcam/gaze capture in this intake flow
        "identity_signals": identity_signals,
        "av_signals": av_signals,
        "counterfactuals": [],
    }
    result = run_full_analysis(record)
    result["video_analysis_note"] = (
        "Lip-sync/identity signals were derived from the uploaded video file "
        "deterministically (no GPU/SyncNet/InsightFace decode available in this "
        "environment) — see backend/adapters/video_intake.py."
        if video is not None and video.filename
        else "No video uploaded — lip-sync and face-identity checks are marked UNAVAILABLE."
    )
    return result


# ---------------------------------------------------------------------------
# Feature 1 — Interview Setup / Question Bank
# ---------------------------------------------------------------------------

@app.post("/api/candidates")
def create_candidate(name: str = Form(...), db: Session = Depends(get_db)):
    c = models.Candidate(name=name)
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "name": c.name}


@app.post("/api/resumes/upload")
async def upload_resume(file: UploadFile = File(...), job_description: str = Form("")):
    content = await file.read()
    text = parse_resume_bytes(file.filename, content)
    # naive claim seeding: a real implementation would use an LLM extractor (Master Spec Section 31)
    claims = []
    return {"resume_text": text, "suggested_claims": claims}


@app.post("/api/interviews")
def create_interview(candidate_id: int = Form(...), job_description: str = Form(""), resume_text: str = Form(""),
                      resume_claims: str = Form("[]"), db: Session = Depends(get_db)):
    interview = models.Interview(
        candidate_id=candidate_id, job_description=job_description, resume_text=resume_text,
        resume_claims=json.loads(resume_claims), status="DRAFT",
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)
    return {"id": interview.id, "status": interview.status}


@app.get("/api/interviews/{interview_id}")
def get_interview(interview_id: int, db: Session = Depends(get_db)):
    interview = db.get(models.Interview, interview_id)
    if not interview:
        raise HTTPException(404, "Interview not found")
    return {
        "id": interview.id, "status": interview.status, "job_description": interview.job_description,
        "resume_claims": interview.resume_claims,
        "questions": [
            {"id": q.id, "question_text": q.question_text, "skill": q.skill, "difficulty": q.difficulty,
             "expected_proficiency": q.expected_proficiency, "reference_answer": q.reference_answer,
             "required_concepts": q.required_concepts, "ai_generated": q.ai_generated}
            for q in interview.questions
        ],
    }


@app.post("/api/interviews/{interview_id}/questions")
def add_question(interview_id: int, question_text: str = Form(...), skill: str = Form(""),
                  difficulty: str = Form("Basic"), expected_proficiency: str = Form("Intermediate"),
                  reference_answer: str = Form(""), required_concepts: str = Form("[]"),
                  db: Session = Depends(get_db)):
    interview = db.get(models.Interview, interview_id)
    if not interview:
        raise HTTPException(404, "Interview not found")
    q = models.Question(
        interview_id=interview_id, question_text=question_text, skill=skill, difficulty=difficulty,
        expected_proficiency=expected_proficiency, reference_answer=reference_answer,
        required_concepts=json.loads(required_concepts), order_index=len(interview.questions),
    )
    db.add(q)
    db.commit()
    db.refresh(q)
    return {"id": q.id}


@app.put("/api/questions/{question_id}")
def update_question(question_id: int, question_text: Optional[str] = Form(None), skill: Optional[str] = Form(None),
                     difficulty: Optional[str] = Form(None), expected_proficiency: Optional[str] = Form(None),
                     reference_answer: Optional[str] = Form(None), required_concepts: Optional[str] = Form(None),
                     db: Session = Depends(get_db)):
    q = db.get(models.Question, question_id)
    if not q:
        raise HTTPException(404, "Question not found")
    if question_text is not None:
        q.question_text = question_text
    if skill is not None:
        q.skill = skill
    if difficulty is not None:
        q.difficulty = difficulty
    if expected_proficiency is not None:
        q.expected_proficiency = expected_proficiency
    if reference_answer is not None:
        q.reference_answer = reference_answer
    if required_concepts is not None:
        q.required_concepts = json.loads(required_concepts)
    db.commit()
    return {"ok": True}


@app.post("/api/interviews/{interview_id}/start")
def start_interview(interview_id: int, db: Session = Depends(get_db)):
    interview = db.get(models.Interview, interview_id)
    if not interview:
        raise HTTPException(404, "Interview not found")
    interview.status = "STARTED"
    db.commit()
    return {"status": interview.status}


@app.post("/api/interviews/{interview_id}/answers")
def submit_answer(interview_id: int, question_id: int = Form(...), transcript: str = Form(...),
                   response_latency_sec: float = Form(0.0), answer_duration_sec: float = Form(0.0),
                   db: Session = Depends(get_db)):
    ans = models.InterviewAnswer(
        question_id=question_id, transcript=transcript,
        response_latency_sec=response_latency_sec, answer_duration_sec=answer_duration_sec,
    )
    db.add(ans)
    db.commit()
    return {"ok": True}


@app.get("/api/interviews/{interview_id}/report")
def interview_report(interview_id: int, db: Session = Depends(get_db)):
    interview = db.get(models.Interview, interview_id)
    if not interview:
        raise HTTPException(404, "Interview not found")

    record = {
        "candidate_name": interview.candidate.name if interview.candidate else "Unknown",
        "job_description": interview.job_description,
        "resume_text": interview.resume_text,
        "resume_claims": interview.resume_claims,
        "qa_list": [],
    }
    for q in interview.questions:
        ans = q.answer
        record["qa_list"].append({
            "question_id": str(q.id), "question_text": q.question_text, "skill": q.skill,
            "difficulty": q.difficulty, "expected_proficiency": q.expected_proficiency,
            "reference_answer": q.reference_answer, "required_concepts": q.required_concepts,
            "candidate_answer": ans.transcript if ans else "",
            "response_latency_sec": ans.response_latency_sec if ans else 0.0,
            "answer_duration_sec": ans.answer_duration_sec if ans else 0.0,
        })
    result = run_full_analysis(record)
    rec = models.RiskAssessmentRecord(interview_id=interview_id, result_json=result)
    db.add(rec)
    interview.status = "COMPLETED"
    db.commit()
    return result


# ---------------------------------------------------------------------------
# Smart Analyze: just resume + free-form transcript, no structured format.
# ---------------------------------------------------------------------------

@app.post("/api/smart-analyze")
async def smart_analyze(
    candidate_name: str = Form(...),
    job_description: str = Form(""),
    resume: UploadFile = File(...),
    transcript: UploadFile = File(...),
):
    """
    Smart analysis endpoint — accepts a resume and a free-form audio transcript
    (e.g. PDF, DOCX, TXT formats). Automatically extracts Q&A pairs, detects skills,
    cross-references resume claims with interview answers, and runs the full
    VeriInterview analysis pipeline.

    No structured [Q1] format or separate ideal-answers file needed.
    """
    from backend.modules.document_parser import parse_document_bytes

    resume_bytes = await resume.read()
    transcript_bytes = await transcript.read()

    resume_text = parse_document_bytes(resume.filename or "resume.txt", resume_bytes)
    transcript_text = parse_document_bytes(transcript.filename or "transcript.txt", transcript_bytes)

    if not resume_text.strip():
        raise HTTPException(
            400,
            "Could not extract readable text from the uploaded resume file. "
            "Please upload a standard PDF, DOCX, or TXT resume."
        )

    if not transcript_text.strip():
        raise HTTPException(
            400,
            "Could not extract readable text from the uploaded transcript file. "
            "Please upload a standard text, PDF, or DOCX transcript."
        )

    # Step 1: Parse free-form transcript into structured Q&A pairs
    qa_list = smart_transcript_parser.parse_free_transcript(
        transcript_text, resume_text
    )

    if not qa_list:
        raise HTTPException(
            400,
            "Could not extract any Q&A pairs from the transcript. "
            "Please ensure it contains interview questions and candidate answers "
            "(e.g. 'Interviewer: ... / Candidate: ...' or 'Q: ... / A: ...')."
        )

    # Step 2: Infer resume claims from extracted skills
    known_skills = sorted({qa["skill"] for qa in qa_list})
    resume_claims = intake_parser.infer_resume_claims(resume_text, known_skills)

    # Step 3: Build the analysis record and run pipeline first
    record = {
        "candidate_name": candidate_name,
        "job_description": job_description,
        "resume_text": resume_text,
        "resume_claims": resume_claims,
        "qa_list": qa_list,
        "speech_features": None,
        "behavior_signals": None,
        "identity_signals": None,
        "av_signals": None,
        "counterfactuals": [],
    }
    result = run_full_analysis(record)

    # Step 4: Run keyword detection & cross-referencing with answer evaluations
    kw_report = keyword_detector.detect_all(
        resume_text=resume_text,
        transcript_text=transcript_text,
        qa_list=qa_list,
        evaluations=result.get("answer_evaluations", []),
    )

    # Step 5: Enrich with keyword detection + cross-reference data
    result["keyword_detection"] = kw_report
    result["transcript_skills"] = smart_transcript_parser.extract_skills_from_transcript(transcript_text)
    result["parsed_questions_count"] = len(qa_list)
    result["analysis_mode"] = "smart"

    return result
