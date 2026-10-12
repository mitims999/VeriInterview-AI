# Data Model

## Persisted entities (SQLite, `backend/database/models.py`)

- **Candidate**(id, name)
- **Interview**(id, candidate_id, job_description, resume_text, resume_claims[json], status)
- **Question**(id, interview_id, question_text, skill, difficulty, expected_proficiency, reference_answer, required_concepts[json], ai_generated, order_index)
- **InterviewAnswer**(id, question_id, transcript, response_latency_sec, answer_duration_sec)
- **RiskAssessmentRecord**(id, interview_id, result_json, created_at)

This is a deliberately simplified subset of the Master Spec Section 21 entity list, sized
for a hackathon SQLite database. Migrating to Postgres only requires swapping the SQLAlchemy
connection string in `backend/database/session.py`.

## In-memory / fixture record schema (used by both the synthetic demo dataset and the
live interview report endpoint — `backend/modules/pipeline.py::run_full_analysis`)

```json
{
  "candidate_name": "string",
  "job_description": "string",
  "resume_text": "string",
  "resume_claims": [{"skill": "AWS", "claim_text": "...", "claim_strength": "weak|medium|strong"}],
  "qa_list": [{
    "question_id": "Q1", "question_text": "...", "skill": "...", "difficulty": "...",
    "expected_proficiency": "...", "reference_answer": "...", "required_concepts": ["..."],
    "candidate_answer": "...", "response_latency_sec": 2.1, "answer_duration_sec": 40.0
  }],
  "speech_features": [{"question_id": "Q1", "pitch_hz": 172, "loudness_db": -18, "speech_rate_wpm": 128, "pause_ratio": 0.14, "speech_silence_ratio": 0.78}],
  "behavior_signals": {"gaze_off_screen_pct": 0.06, "head_pose_variance": "normal|elevated|high", "multi_person_detected": false, "phone_detected": false},
  "identity_signals": {"min_frame_similarity_to_anchor": 0.98, "face_lost_pct": 0.01},
  "av_signals": {"av_offset_frames": 1, "sync_confidence": 0.92},
  "counterfactuals": [{
    "skill": "Machine Learning", "followup_question_id": "Q3F", "perturbed_assumption": "...",
    "candidate_followup_answer": "...", "expected_new_concepts": ["..."], "expected_retained_concepts": ["..."]
  }]
}
```

`speech_features`, `behavior_signals`, `identity_signals`, `av_signals` are the exact
aggregate outputs the named repos (openSMILE, Proctoring-AI, InsightFace, SyncNet) would
produce from raw audio/video; see `docs/LIMITATIONS.md` for why this prototype supplies
them as structured fixture data instead of running the models on raw media.
