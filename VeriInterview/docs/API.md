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

## Video analysis jobs (M3: deepfake sub-score, identity, lip-sync, gaze, eye cues)

Video analysis takes minutes on CPU (about 2.5 min for a 3-min interview), so it runs as a background job.
Code: `backend/api_vision.py`. Evidence types: `docs/m3/EVIDENCE_TYPES.md`.

- `GET /api/vision/health` -> `{ready, models{...: bool}, dependencies{...: bool}, checks_available{face_identity, lip_sync, gaze_and_eye_cues, phone_and_people}}`
  Use it to grey out checks that cannot run on this machine.
- `POST /api/vision/jobs` (multipart) -> **202** `VisionJob`
  - `video` (required): `.mp4 .webm .mov .avi .mkv .m4v .mpg .mpeg`, ≤ 500 MB
  - `reference_photo` (optional): `.jpg .jpeg .png .webp .bmp`, ≤ 10 MB. A photo of the registered candidate (resume/ID). Enables the reference-photo identity check.
  - `questions` (optional form field): JSON `[{"question_id": "Q1", "start": 0, "end": 95.5}, ...]` in seconds from the start of the video. Gives per-question evidence and makes sure every answer gets a lip-sync check.
  - Errors: 415 bad file type, 413 too large, 422 empty video or bad `questions`.
- `GET /api/vision/jobs/{job_id}` -> `VisionJob` (poll every 1–2 s; 404 when unknown or expired after 1 h)
- `DELETE /api/vision/jobs/{job_id}` -> 204 (409 while running). Forgets the result.

```json
VisionJob {
  "job_id": "3f2c...",
  "status": "queued | running | done | error",
  "stage": "queued | starting | speech | identity | gaze_objects_lipsync | finishing | done | error",
  "stage_label": "Checking face identity",         // ready-to-show text
  "progress": 0.35,                                 // 0-1
  "queue_position": 0,                              // analyses ahead of this one
  "created_at": 1792000000.0, "updated_at": 1792000042.0,
  "video_filename": "interview.mp4", "reference_photo": true,
  "error": null,                                    // "Type: message" when status == "error"
  "result": null | VisionResult                     // present when status == "done"
}

VisionResult {
  "deepfake":         ModuleResult,   // Deepfake Sub-Score card
  "identity":         ModuleResult,   // face identity (+ reference photo)
  "sync":             ModuleResult,   // lip-sync
  "eye_authenticity": ModuleResult,   // blink pattern + mouth vs speech (weak cues)
  "behavior":         ModuleResult,   // gaze, head pose, phone, extra people (behaviour, NOT deepfake)
  "timeline":         [Evidence],     // every non-low finding that has a timestamp, sorted by time
  "meta":             {"duration_s", "analysed_s", "speech_s", "timings_s", "total_s", "cache_hit"}
}

ModuleResult { "feature", "score": 0-100, "confidence": 0-1, "level": "LOW|MEDIUM|HIGH|UNAVAILABLE",
               "status": "OK|INSUFFICIENT_EVIDENCE|UNAVAILABLE", "evidence": [Evidence],
               "quality": {"data_quality", "baseline_sufficient"}, "details": {...} | null }
Evidence     { "type", "severity": "low|medium|high", "message", "question_id": "Q2"|null, "timestamp": 18.0|null }
```

`deepfake.details` = `{"sub_scores": {"identity"|"sync"|"eye_authenticity": {"score", "level", "confidence", "status"}},
"inputs_flagged", "inputs_high", "strong_identity_signal", "coverage"}`. Use it for the three-bar breakdown.
`deepfake.evidence[0]` (type `deepfake_summary`) is a one-line plain-English summary for the card header.

UI rules: show `level` + `confidence` side by side (never merged); show `status != OK` as "not assessed" with the
evidence message as the reason; never present any of this as a verdict. It is a review aid.

`POST /api/analyze` (manual intake) also accepts optional `reference_photo`, now runs the same real analysis
synchronously when a video is attached, and its response includes `deepfake_assessment` (a ModuleResult, as above)
plus `video_analysis_meta`. `deepfake_assessment` is also present on every demo-candidate report.
