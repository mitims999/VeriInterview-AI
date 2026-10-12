# VeriInterview — Final Build Specification for Claude Code

> **How to use this file:** Give this entire file to Claude Code as the master build spec. Every module below names an exact GitHub repository to clone and wrap. Do NOT invent alternative models or retrain anything from scratch unless explicitly told to. Do NOT design a new architecture — implement the one specified here.

---

## 0. Project One-Liner

VeriInterview is an AI-assisted interview integrity system. It never decides hire/reject. It produces an explainable **0–100 Integrity Risk Score** (LOW / MEDIUM / HIGH) plus a separate **Candidate Capability Score**, backed by evidence from independent modules, for human review only.

**Golden rule for every module:** one weak/anomalous signal must never independently produce a "cheating" verdict. Every module returns `(score, confidence, evidence[])` — never a final decision.

---

## 1. System Architecture (authoritative diagram)

```
CANDIDATE PROFILE (resume/skills)
        │
        ▼
  PROFILE ANALYZER  ──► uses srbhr/Resume-Matcher (skill mismatch)
        │
        │
VIDEO ──────────► VISUAL ANALYZER ──► VIDEO RISK
  │                 (face/identity — deepinsight/insightface)
  │
  ├────────────► EYE GAZE / BEHAVIOR ANALYZER ──► BEHAVIOR RISK
  │                 (vardanagarwal/Proctoring-AI)
  │
  ├────────────► AUDIO ANALYZER ──► AUDIO RISK
  │                 (voice modulation — audeering/opensmile)
  │
  └────────────► AV SYNC ANALYZER ──► SYNC RISK
                    (joonson/syncnet_python)

INTERVIEW QUESTIONS + ANSWERS (+ EXPECTED ANSWER KEY, already in transcript)
        │
        ▼
   NLP ANALYZER
        │
   ┌────┴────┬─────────────┬────────────────────┐
   ▼         ▼             ▼                     ▼
Skill      Technical    Keyword/Buzzword    Answer-vs-Expected
Match      Depth        Match (JD-specific)  Match (transcript key)
   │         │             │                     │
   └────┬────┴─────────────┴─────────────────────┘
        ▼
     NLP RISK

ALL SIGNALS (Video, Behavior, Audio, Sync, NLP Risk)
        │
        ▼
  EVIDENCE FUSION
        │
        ▼
 CALIBRATED RISK MODEL
        │
        ▼
┌───────────────────┐
│ 0–30   LOW         │
│ 31–60  MEDIUM       │
│ 61–100 HIGH         │
└───────────────────┘
        │
        ▼
 EXPLANATION ENGINE
        │
        ▼
 RECRUITER DASHBOARD
```

---

## 2. Exact Repo Assignments (mandatory — do not substitute)

| # | Feature | Repository | Role |
|---|---------|------------|------|
| 1 | Resume ↔ JD skill mismatch | **https://github.com/srbhr/Resume-Matcher** | Parses resume + job description, extracts keywords/skills, computes match score |
| 2 | Lip-sync consistency | **https://github.com/joonson/syncnet_python** | Computes AV offset + confidence between candidate's mouth movement and speech audio |
| 3 | Eye gaze / head pose / behavioral anomaly | **https://github.com/vardanagarwal/Proctoring-AI** | Eye tracking (left/right/up), head pose estimation, mouth-opening detection, person/phone detection |
| 4 | Voice modulation / speech behavior | **https://github.com/audeering/opensmile** (+ `opensmile-python`) | Pitch, loudness, speech rate, pause, voice-quality features vs candidate's own baseline |
| 5 | Keyword/buzzword-vs-job-role detection | Built on top of **Resume-Matcher**'s keyword extraction (textacy-based) + a job-role keyword list module (new, not from a repo) | Flags whether candidate uses (or fails to use) role-specific technical buzzwords expected for the JD |
| 6 | Expected-answer matching | New module (no repo — uses transcript). Interviewer's expected answer/rubric is already embedded in the transcript alongside each question. Use `sentence-transformers` (https://github.com/UKPLab/sentence-transformers) to embed candidate answer vs expected answer and score semantic overlap | Compares candidate's actual answer to the pre-supplied expected-answer text for that question |
| — | Face identity / deepfake (carried over from earlier spec, keep as-is unless told otherwise) | **https://github.com/deepinsight/insightface** | Face embedding + identity consistency across the interview |
| — | Speech-to-text / transcript + timestamps | **openai/whisper** or **SYSTRAN/faster-whisper** | Produces transcript, word timestamps, pause/latency data used by modules 4–6 |

---

## 3. Module-by-Module Instructions for Claude Code

### 3.1 Skill Mismatch Module — `srbhr/Resume-Matcher`

- Clone `https://github.com/srbhr/Resume-Matcher` directly. Use its resume-parsing and JD-keyword-extraction pipeline (Python + textacy-based keyword extraction, resume/JD vector similarity).
- Input: candidate resume (PDF), job description (text/PDF).
- Output required by VeriInterview:
  ```json
  {
    "skill_match_score": 0-100,
    "matched_keywords": ["Python", "XGBoost"],
    "missing_keywords": ["Docker", "Kubernetes"],
    "resume_claims": ["Used XGBoost for imbalanced classification"]
  }
  ```
- Wrap Resume-Matcher's core parsing/matching logic as an internal service function — do not run it as its own Streamlit/Next.js app inside VeriInterview. Extract only the parsing + keyword-matching + scoring logic.
- `resume_claims` output feeds directly into the **Claim Graph Module** (Section 3.6) for CCST counterfactual generation.

### 3.2 Lip-Sync Module — `joonson/syncnet_python`

- Clone `https://github.com/joonson/syncnet_python` directly and use its pretrained SyncNet model (`download_model.sh`).
- Run `run_pipeline.py` then `run_syncnet.py` on each interview video segment.
- Capture the two native outputs: **AV offset** and **confidence/min-distance score**.
- Convert to VeriInterview format:
  ```json
  {
    "sync_risk_score": 0-100,
    "av_offset_frames": 3,
    "confidence": 0.81,
    "evidence": ["AV offset within normal bonafide range"]
  }
  ```
- Large or unstable offsets and low confidence → higher `sync_risk_score`. Never map this alone to "deepfake confirmed."

### 3.3 Eye Gaze / Behavioral Module — `vardanagarwal/Proctoring-AI`

- Clone `https://github.com/vardanagarwal/Proctoring-AI` directly.
- Use these scripts as-is (they are already modular):
  - `face_detector.py` — face detection backbone for the rest
  - `eye_tracker.py` — reports gaze direction (left / right / center / up)
  - `head_pose_estimation.py` — head orientation angles
  - `mouth_opening_detector.py` — mouth-opening baseline vs deviation
  - `person_and_phone.py` — flags multiple people or phone in frame (strong evidence, not weak)
- Aggregate per-answer into:
  ```json
  {
    "behavior_risk_score": 0-100,
    "gaze_off_screen_pct": 0.18,
    "head_pose_variance": "normal",
    "multi_person_detected": false,
    "phone_detected": false,
    "evidence": ["Gaze remained off-center for 18% of answer duration"]
  }
  ```
- Treat `gaze_off_screen_pct` and head-pose variance as **weak evidence** (per Section 7 of the original evidence model — never proof alone). Treat `multi_person_detected` / `phone_detected` as **stronger evidence** since Proctoring-AI's `person_and_phone.py` is a direct integrity violation signal, not a nervousness proxy.
- A nervous-genuine candidate (gaze drifting, pauses) must NOT alone raise behavior_risk_score into HIGH.

### 3.4 Voice Modulation Module — `audeering/opensmile`

Use the existing detailed spec already written in `Voice_Prosody_Feature_Verification.md` (already provided) as the authoritative implementation guide for this module. Summary Claude Code must follow:

- Repos: `https://github.com/audeering/opensmile`, `https://github.com/audeering/opensmile-python`
- Feature set: `eGeMAPSv02` (do not start with `ComParE_2016`)
- Build a **candidate-specific baseline** from the first few answers — never compare to a universal "normal" voice.
- Extract: pitch (F0), loudness, speaking pace (WPM via Whisper/WhisperX word timestamps), pauses, speech/silence ratio, jitter/shimmer.
- Score formula (heuristic v1):
  ```
  speech_behavior_raw =
      0.25 * pitch_deviation
    + 0.20 * loudness_deviation
    + 0.25 * speech_rate_deviation
    + 0.20 * pause_deviation
    + 0.10 * speech_ratio_deviation
  ```
  Normalize to 0–100. Output must include `score`, `confidence`, and `evidence[]` as specified in the referenced file's Section 12.
- Follow that file's Section 35 ("What Claude Code Must NOT Implement") exactly — no emotion-as-cheating, no gender-based thresholds, no absolute pitch thresholds.

### 3.5 Keyword / Buzzword-vs-Job-Role Module (new, built on Resume-Matcher's extractor)

- Reuse Resume-Matcher's **keyword/key-term extraction** (textacy-based) already pulled in Section 3.1, but apply it to the **transcript answers**, not just the resume.
- Maintain a per-job-role expected-buzzword list (e.g., for a "Backend Engineer — AWS" role: `["EC2", "S3", "Lambda", "auto-scaling", "load balancer", "IAM", "VPC"]`). This list can be auto-generated by extracting top keywords from the job description via Resume-Matcher's own JD-keyword-extraction step — no separate repo needed.
- Compare buzzwords actually used across candidate's answers vs the expected role-keyword list.
- Output:
  ```json
  {
    "buzzword_coverage_score": 0-100,
    "expected_keywords": ["EC2","S3","Lambda","IAM"],
    "used_keywords": ["EC2","S3"],
    "missing_keywords": ["Lambda","IAM"],
    "evidence": ["Candidate used 2 of 4 role-critical AWS terms across all answers"]
  }
  ```
- Low buzzword coverage is **capability evidence**, not integrity evidence by default — route it primarily into Candidate Capability, and only into Integrity Risk when combined with high resume-claimed expertise + low buzzword usage + low technical depth (a genuine mismatch pattern).

### 3.6 Expected-Answer Matching Module (new, uses transcript directly)

- The interview transcript already contains, for each question, the **expected/model answer or rubric** alongside the candidate's actual answer (per your note: "questions and answers will already be in transcript ... of what the answers must be").
- For each Q&A pair:
  1. Extract `question_text`, `expected_answer_text`, `candidate_answer_text` from the transcript.
  2. Embed both `expected_answer_text` and `candidate_answer_text` using `sentence-transformers` (e.g. `all-MiniLM-L6-v2` or a stronger SBERT/DeBERTa model).
  3. Compute cosine similarity → `answer_match_score` (0–100).
  4. Optionally use an LLM as a structured evaluator (per the "LLM as semantic extractor / reasoning analyst" principle from the original spec) to identify **missing required concepts** from the expected answer that the candidate didn't cover.
- Output:
  ```json
  {
    "answer_match_score": 0-100,
    "missing_concepts": ["class_weight", "F1-score trade-off"],
    "evidence": ["Candidate covered 3 of 5 required concepts from the expected answer"]
  }
  ```
- This module's output feeds both **Candidate Capability** (technical correctness) and, when combined with CCST (below), **Knowledge Ownership**.

### 3.7 Claim Graph + CCST Module (carried over — no repo, this is the core novelty)

- Keep this exactly as originally specified: extract claims (seeded from `resume_claims` in 3.1 and candidate answers), build a small graph with `networkx`, perturb one condition, generate a counterfactual question (LLM), compare the new answer to the original reasoning.
- Produces **Knowledge Ownership Score (KOS)** and **Causal Adaptation Consistency (CAC)** exactly as in the original project understanding doc. Do not skip or simplify this — it is the project's central novelty and must not be replaced by any of the GitHub repos above.

---

## 4. Evidence Fusion & Scoring

Keep evidence fusion deterministic and explainable — do not let any single module or the LLM directly set the final score.

```
risk = 0.20 * identity_risk (InsightFace)
      + 0.15 * sync_risk (SyncNet)
      + 0.15 * behavior_risk (Proctoring-AI)
      + 0.15 * speech_behavior_risk (openSMILE)
      + 0.20 * knowledge_ownership_risk (CCST/CAC)
      + 0.15 * profile_consistency_risk (Resume-Matcher + expected-answer match)
```

Weights are a v1 starting point — tune later with validation data. Cap any single module's contribution so one noisy signal cannot dominate (per the original evidence model).

Buckets: `0–30 LOW`, `31–60 MEDIUM`, `61–100 HIGH`.

Report **Candidate Capability** (technical depth, answer correctness, buzzword coverage) separately from **Interview Integrity Risk** — never merge them into one number.

---

## 5. Datasets Needed

### 5.1 For validating/calibrating repo-based modules (public)
- **LFW** — face verification sanity-check (InsightFace)
- **RAVDESS** (https://zenodo.org/records/1188976) — prosody/pitch/loudness extraction validation (openSMILE)
- **CREMA-D** (https://github.com/CheyneyComputerScience/CREMA-D) — auxiliary prosody validation
- **LRS2/LRS3** — optional, only if SyncNet needs re-validation on new data (pretrained model likely sufficient)

### 5.2 Project-specific synthetic datasets (must build yourself — 5 total)
1. `genuine.json` — genuine strong candidate
2. `genuine_weak.json` — genuine weak candidate
3. `ai_assisted.json` — AI-assisted candidate
4. `deepfake.json` — synthetic deepfake/impersonation demo media only
5. `nervous_genuine.json` — nervous but honest candidate

Each record should include: resume text, job description, transcript with `question / expected_answer / candidate_answer` triples, and (for A/V cases) sample audio/video clips.

---

## 6. Tech Stack (unchanged from original spec)

- Frontend: React, Vite, Tailwind, Recharts
- Backend: Python, FastAPI
- Storage (prototype): SQLite + JSON fixtures
- Graph: NetworkX
- Classical ML: scikit-learn (Isolation Forest for later anomaly refinement)

---

## 7. Instructions Claude Code Must Follow

1. Clone and use the exact repos listed in Section 2 — do not substitute or reimplement their core functionality from scratch.
2. Wrap each repo's core logic as an internal Python service/module with a clean function signature and typed (Pydantic) input/output — do not expose their original UIs (Streamlit/Next.js) inside VeriInterview.
3. Every module returns `(score, confidence, evidence[])`. No module may output a final hire/cheat decision.
4. Keep the Claim Graph / CCST / KOS / CAC module (Section 3.7) as original code — this is the project's novelty and is not covered by any repo.
5. Keep evidence fusion deterministic (weighted formula), not LLM-controlled.
6. Build the 5 synthetic test cases (Section 5.2) and verify: nervous-genuine → low integrity risk; AI-assisted → higher integrity risk despite polished delivery.
7. Dashboard must show component scores + evidence bullets + "human review" recommendation, per Section 11 of the original project understanding doc.
8. Include unit tests for each module and document licensing notes (openSMILE has commercial-use restrictions — flag this in the README).
