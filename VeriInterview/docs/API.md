# VeriInterview API

Base URL: `http://127.0.0.1:8000`

## Health

`GET /api/health` -> `{"status": "ok"}`

## Demo dataset (synthetic candidates)

- `GET /api/demo/candidates` -> `[{id, candidate_name, label}]`
- `GET /api/demo/candidates/{id}/report` -> full `FullReport` (see below)

## Interview setup (Feature 1)

- `POST /api/candidates` (form: `name`) -> `{id, name}`
- `POST /api/resumes/upload` (multipart: `file`, form: `job_description`) -> `{resume_text, suggested_claims}`
- `POST /api/interviews` (form: `candidate_id, job_description, resume_text, resume_claims(json)`) -> `{id, status}`
- `GET  /api/interviews/{id}` -> interview + questions
- `POST /api/interviews/{id}/questions` (form fields) -> `{id}`
- `PUT  /api/questions/{id}` (partial form fields) -> `{ok}`
- `POST /api/interviews/{id}/start` -> `{status}`
- `POST /api/interviews/{id}/answers` (form: `question_id, transcript, response_latency_sec, answer_duration_sec`) -> `{ok}`
- `GET  /api/interviews/{id}/report` -> runs full analysis, returns `FullReport`, persists a `RiskAssessmentRecord`

## FullReport shape

```json
{
  "assessment": {
    "candidate_name": "...",
    "risk_score": 51.2,
    "risk_level": "MEDIUM",
    "confidence": 0.76,
    "capability_score": 51.3,
    "components": {
      "identity_risk": { "feature": "face_identity", "score": 22.5, "confidence": 0.9, "level": "LOW", "evidence": [...], "quality": {...}, "status": "OK" },
      "sync_risk": {...}, "behavior_risk": {...}, "speech_behavior_risk": {...},
      "knowledge_ownership_risk": {...}, "profile_consistency_risk": {...}
    },
    "evidence_timeline": [{"type": "...", "severity": "high", "message": "...", "question_id": "Q3"}],
    "recommended_action": "..."
  },
  "skill_profile": { "skill_match_score": 12.5, "matched_keywords": [...], "missing_keywords": [...], "resume_claims": [...] },
  "skill_verdicts": [{"skill": "AWS", "verdict": "PARTIALLY_VERIFIED", "evidence_score": 60.1, "questions_used": ["Q2","Q4"]}],
  "answer_evaluations": [{"question_id": "Q1", "score": 82, "concept_coverage": 0.84, "missing_concepts": [...], "incorrect_concepts": [], "confidence": 0.91}],
  "buzzword_coverage": {"buzzword_coverage_score": 27.0, "expected_keywords": [...], "used_keywords": [...], "missing_keywords": [...]},
  "qa_list": [{"question_id": "Q1", "question_text": "...", ...}]
}
```
