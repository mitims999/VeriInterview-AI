# Claude Code Master Project Specification — TCS P1 Deepfake Interview Alert Tool

## 0. How to use this file

This is the **master specification / orchestration prompt** for Claude Code Sonnet 5 Medium.

Use this file together with the separate feature-specific Markdown files supplied with the project.

The feature-specific MD files are the detailed technical specifications. This document explains:

- the TCS problem statement
- the complete product vision
- how all feature modules fit together
- what the frontend must do
- what the backend must do
- data flow
- APIs
- database design
- scoring
- risk levels
- confidence
- human review
- modularity
- integration rules
- testing
- demo requirements
- implementation order

**Important:** Do not throw away the existing feature-specific MD specifications. Read and follow all of them. The feature-specific MD files take precedence for implementation details of their own modules.

---

# 1. TCS Problem Statement

## Problem Statement ID

**P1**

## Title

**Deepfake Interview Alert Tool**

## Theme

**AI + Cyber Defense**

## Expected AI Feature

**Risk scoring + anomaly detection**

## Prototype Scope

Use sample/synthetic candidate profiles, resumes, interview questions, answers, transcripts and audio/video observations to identify mismatch and integrity-risk patterns.

## Problem

Online interviews are increasingly common for internships, campus placements and recruitment. Recruiters may face impersonation or interview-integrity risks through:

- scripted responses
- external AI assistance
- looking at answers from another screen
- unauthorized prompting
- proxy/impersonation behavior
- deepfake or face manipulation
- voice manipulation
- audio/video synchronization inconsistencies
- resume claims that cannot be demonstrated during the interview

The system should analyze multiple signals and provide an **impersonation/interview-integrity risk score with evidence**.

The system must support interviewer review and **must not make the final hiring decision**.

---

# 2. Product Vision

Build a full-stack AI-powered platform called:

# VeriInterview

### Core idea

> **VeriInterview evaluates whether the identity, claimed skills, spoken answers and observed interview behavior are consistent across the interview, and provides an explainable integrity-risk assessment for human review.**

The product is NOT:

- a simple deepfake detector
- an emotion detector
- a voice-only cheating detector
- an AI-generated-text detector
- an automatic hiring/rejection system

It is a **multimodal interview-integrity evidence platform**.

---

# 3. Core Workflow

The complete product should work like this:

```text
INTERVIEWER
   |
   +--> Create interview
   |
   +--> Upload candidate resume
   |
   +--> Enter/generate questions
   |
   +--> Enter/generate reference answers
   |
   +--> Define key concepts
   |
   +--> Map questions to skills
   |
   +--> Set expected proficiency / difficulty
   |
   +--> Start interview
             |
             v
       CANDIDATE INTERVIEW
             |
       +-----+-------------------------------+
       |                                     |
       v                                     v
    Audio capture                        Video capture
       |                                     |
       +--------+---------------+------------+
                |               |
                v               v
          Speech-to-text     Visual analysis
                |               |
                v               |
          Candidate answer      |
                |               |
       +--------+---------+     |
       |                  |     |
       v                  v     v
 Answer correctness   Resume/skill  A/V integrity
 evaluation           verification
       |                  |            |
       +------------------+------------+
                          |
                          v
                  Behavioral analysis
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
        Eye gaze      Voice/prosody   Response timing
            |             |             |
            +-------------+-------------+
                          |
                          v
                 Evidence Fusion Engine
                          |
               +----------+----------+
               |                     |
               v                     v
         Risk Score 0–100       Confidence
               |
       +-------+-------+
       |       |       |
       v       v       v
      LOW   MEDIUM    HIGH
               |
               v
        Evidence / Reasons
               |
               v
        HUMAN INTERVIEWER
```

---

# 4. Main Product Features

These are the primary modules.

## Feature 1 — Interview Setup / Question Bank

The interviewer must be able to:

1. Create an interview.
2. Upload the candidate resume.
3. Add questions manually.
4. Add expected/reference answers manually.
5. Add key concepts manually.
6. Map the question to one or more skills.
7. Specify difficulty.
8. Specify expected proficiency.
9. Optionally ask AI to generate questions from the resume.
10. Optionally ask AI to generate reference answers and key concepts.
11. Edit everything before starting the interview.

The interviewer must always remain able to modify AI-generated content.

### Example

```text
Question:
What is overfitting in machine learning?

Skill:
Machine Learning

Difficulty:
Basic

Expected proficiency:
Intermediate

Reference answer:
Overfitting occurs when a model learns the training data too closely,
including noise, and therefore generalizes poorly to unseen data.

Required concepts:
- training data
- noise
- generalization
- unseen/test data
```

---

# 5. Feature 2 — Resume Parsing + Skill Extraction

The system should parse:

- PDF
- DOCX
- TXT

Extract:

- skills
- technologies
- projects
- experience
- education
- certifications
- claimed proficiency
- explicit technical claims

Do not treat every skill equally.

Example:

```text
"Familiar with AWS"
→ weak claim

"Experience with AWS"
→ medium claim

"Designed and deployed AWS architecture"
→ strong claim
```

Create a normalized candidate skill profile.

Example:

```json
{
  "skill": "AWS",
  "claim": "Designed and deployed AWS architecture",
  "claim_strength": "strong",
  "source": "resume"
}
```

---

# 6. Feature 3 — Resume-to-Interview Skill Verification

This is a core module.

The system must answer:

> **How much evidence did the interview provide that the candidate actually possesses the skills they claimed on the resume?**

Pipeline:

```text
Resume
  ↓
Claim extraction
  ↓
Skill normalization
  ↓
Question → Skill mapping
  ↓
Candidate answer
  ↓
Answer correctness
  ↓
Concept coverage
  ↓
Per-question evidence
  ↓
Per-skill aggregation
  ↓
Resume–Interview verification
```

Possible outcomes:

```text
STRONGLY VERIFIED
VERIFIED
PARTIALLY VERIFIED
WEAKLY SUPPORTED
RESUME–INTERVIEW MISMATCH
NOT ASSESSED
```

### Important rule

If a skill is present on the resume but never tested:

```text
NOT ASSESSED
```

Never:

```text
DOES NOT KNOW
```

---

# 7. Feature 4 — Audio to Transcript

During the interview:

```text
Candidate speaks
      ↓
Audio captured
      ↓
Speech-to-text
      ↓
Transcript
      ↓
Question-specific answer record
```

Primary suggested technology:

- faster-whisper
- Whisper
- WhisperX where word-level timing/alignment is required

Store each question answer separately.

Example:

```json
{
  "question_id": "Q004",
  "start_time": 123.4,
  "end_time": 151.7,
  "transcript": "..."
}
```

---

# 8. Feature 5 — Answer Correctness / NLP Evaluation

Compare:

- reference answer
- key concepts
- candidate transcript
- question
- skill
- expected proficiency

Do NOT rely only on keyword matching.

Use:

```text
semantic similarity
+
key concept coverage
+
contradiction/error detection
+
LLM/rubric evaluation
```

Output:

```json
{
  "question_id": "Q004",
  "score": 82,
  "concept_coverage": 0.84,
  "missing_concepts": [
    "noise"
  ],
  "incorrect_concepts": [],
  "confidence": 0.91
}
```

The answer evaluation should be explainable.

---

# 9. Feature 6 — Lip-Sync / Audio-Video Consistency

Analyze whether spoken audio is reasonably synchronized with visible mouth movement.

Potential signals:

- mouth movement / phoneme timing
- audio-visual synchronization
- temporal mismatch
- face-track continuity
- abnormal frame changes

This should be an **evidence source**, not an independent cheating verdict.

Suggested architecture:

```text
Audio
 ↓
Speech / phoneme timing

Video
 ↓
Face / mouth landmarks

Both
 ↓
Temporal alignment
 ↓
Lip-sync anomaly
```

Use existing open-source/research implementations where appropriate instead of attempting to build a high-quality deepfake model from scratch during the hackathon.

The detailed feature-specific MD for A/V analysis should be followed.

---

# 10. Feature 7 — Eye Gaze / Looking Away / External Screen Detection

The system should estimate whether the candidate's visual attention pattern is consistent with normal interview behavior.

Possible measurements:

- face orientation
- head pose
- eye-gaze direction where feasible
- gaze deviation frequency
- gaze duration
- repeated direction changes
- prolonged off-screen attention

Potential scenario:

```text
question displayed
      ↓
candidate repeatedly looks toward secondary monitor
      ↓
speech continues
      ↓
timing and answer-quality signals also change
```

Do not implement:

```text
looking away = cheating
```

Instead:

```text
gaze anomaly = behavioral evidence
```

It must be combined with other signals.

---

# 11. Feature 8 — Voice Modulation / Speech Behavior

This is covered in the separate voice-prosody MD file.

Primary repository:

https://github.com/audeering/opensmile

Use openSMILE primarily as an acoustic/prosodic feature extractor.

Measure:

- pitch/F0
- loudness
- speech rate
- pitch variability
- loudness variability
- pause frequency
- pause duration
- speech/silence ratio
- voice-quality measures
- candidate-specific deviations

Output:

```json
{
  "speech_behavior_score": 68,
  "level": "HIGH",
  "confidence": 0.81
}
```

Do NOT interpret voice modulation alone as copying.

The feature is one evidence stream.

---

# 12. Feature 9 — Response Timing / Prompting Anomalies

Measure:

- time from question end to first speech
- total answer duration
- changes in response latency
- unusual delays on some questions
- changes in answer length
- sudden answer-quality changes after latency spikes

Example:

```text
Question 1:
latency 1.8 sec

Question 2:
latency 2.2 sec

Question 3:
latency 8.7 sec
answer suddenly extremely polished
```

This becomes behavioral evidence.

---

# 13. Feature 10 — Knowledge Ownership / Counterfactual Testing

This is the proposed system-level differentiator.

Do not rely only on static questions.

When the candidate makes a strong technical claim, the system can optionally generate a controlled follow-up that changes **one assumption**.

Example:

Original:

> How would you handle an imbalanced dataset?

Candidate answer establishes a reasoning chain.

Follow-up:

> What would change if the minority class became only 1% of the dataset?

The system checks whether the candidate can adapt their reasoning consistently.

Concept:

```text
Candidate claim
      ↓
Knowledge / dependency graph
      ↓
Controlled perturbation
      ↓
Follow-up question
      ↓
New answer
      ↓
Consistency analysis
```

This should be implemented as a separate module if the project has enough time.

---

# 14. Evidence Fusion Engine

This is the most important system-level component.

No individual model decides cheating.

All modules return structured evidence.

Example:

```json
{
  "resume_skill_mismatch": 0.82,
  "answer_correctness_risk": 0.74,
  "knowledge_ownership_risk": 0.61,
  "lip_sync_risk": 0.12,
  "face_identity_risk": 0.08,
  "eye_gaze_risk": 0.67,
  "speech_behavior_risk": 0.72,
  "response_latency_risk": 0.76
}
```

The engine combines the evidence.

Initial prototype weighting:

```text
Resume / answer consistency       25%
Knowledge ownership               25%
A/V identity integrity            20%
Behavioral signals                15%
Response timing                   10%
Other evidence                     5%
```

These are prototype values and must remain configurable.

---

# 15. Risk Levels

Output a 0–100 risk score.

Prototype categories:

```text
0–34   LOW
35–64  MEDIUM
65–100 HIGH
```

The thresholds must be configuration values, not hard-coded throughout the system.

---

# 16. Confidence Score

Risk and confidence must be separate.

Example:

```text
Risk = 82
Level = HIGH
Confidence = 91%
```

Meaning:

> Multiple independent evidence sources support the current assessment.

Another case:

```text
Risk = 82
Level = HIGH
Confidence = 42%
```

Meaning:

> There are suspicious signals, but the evidence is incomplete or inconsistent.

Confidence should consider:

- number of available evidence sources
- audio/video quality
- transcript confidence
- baseline sufficiency
- agreement between independent modules
- missing data
- model uncertainty

---

# 17. Human Review

The system must never automatically:

- reject the candidate
- mark them as a cheater
- make a hiring decision

Instead show:

```text
INTEGRITY RISK
78 / 100
HIGH

CONFIDENCE
88%

REVIEW RECOMMENDED

Evidence:
✓ Resume claims AWS at advanced level
✓ 2/4 AWS questions below expected level
✓ Response latency increased significantly
✓ Speech behavior anomaly detected
✓ Gaze deviation increased during those questions
○ No significant lip-sync anomaly
```

Then:

```text
Recommended action:
Conduct targeted verification.
```

---

# 18. Frontend Requirements

Build a polished web application.

Recommended stack:

- React
- Vite
- TypeScript
- Tailwind CSS or equivalent
- charting library such as Recharts
- browser MediaRecorder/Web Audio APIs

The interface should have:

## A. Interviewer Dashboard

- create interview
- candidate information
- upload resume
- question bank
- answer/reference editor
- skill mapping
- difficulty
- expected proficiency
- AI question generation
- AI reference-answer generation

## B. Interview Screen

- candidate name
- current question
- timer
- camera preview
- microphone status
- recording status
- question progress
- no sensitive debug metrics visible to candidate

## C. Live Analysis Panel

For demo mode only, interviewer/admin can view:

- transcript
- response timing
- speech behavior
- gaze
- A/V signals

## D. Final Report

Dashboard containing:

```text
Overall Integrity Risk
Confidence

Resume Verification
Knowledge Ownership
A/V Integrity
Speech Behavior
Gaze Behavior
Response Timing

Evidence Timeline
Question-by-question results
Skill-by-skill verification
Recommended human action
```

---

# 19. Final Report Layout

The final report should provide:

## Candidate Summary

```text
Candidate:
Skills claimed:
Skills assessed:
Questions answered:
```

## Integrity Summary

```text
Risk:
Level:
Confidence:
```

## Skill Verification

```text
Python      Strongly Verified
ML          Verified
AWS         Partially Verified
Kubernetes  Not Assessed
```

## Question Evaluation

```text
Q1
Score 91
✓ Correct concepts

Q2
Score 43
⚠ Missing concepts

Q3
Score 24
✗ Incorrect reasoning
```

## Behavioral Evidence

```text
Speech anomaly
Gaze anomaly
Response latency
A/V integrity
```

## Evidence Timeline

Show anomalies against interview time/questions.

---

# 20. Backend Architecture

Recommended:

**Python + FastAPI**

Suggested services/modules:

```text
backend/
├── main.py
├── config.py
├── database/
│   ├── models.py
│   └── session.py
│
├── resume/
├── questions/
├── interview/
├── transcription/
├── answer_evaluation/
├── skill_verification/
├── speech_behavior/
├── gaze/
├── av_integrity/
├── knowledge_ownership/
├── evidence_fusion/
├── reporting/
└── services/
```

Keep modules independent.

---

# 21. Database

Use SQLite for the hackathon.

Allow migration to PostgreSQL later.

Core entities:

```text
Candidate
Resume
Skill
ResumeClaim
Interview
Question
ReferenceAnswer
QuestionSkill
InterviewSession
InterviewAnswer
Transcript
AnswerEvaluation
SkillVerification
SpeechAnalysis
GazeAnalysis
AVAnalysis
BehaviorAnalysis
RiskAssessment
Evidence
```

A simplified relationship:

```text
Candidate
   |
 Resume
   |
 ResumeClaim
   |
 Skill
   |
 Question
   |
 InterviewAnswer
   |
 Evaluation
   |
 SkillVerification
   |
 RiskAssessment
```

---

# 22. API Design

Suggested endpoints:

```text
POST /api/candidates
POST /api/resumes/upload

POST /api/interviews
GET  /api/interviews/{id}

POST /api/interviews/{id}/questions
POST /api/interviews/{id}/questions/generate
PUT  /api/questions/{id}

POST /api/interviews/{id}/start
POST /api/interviews/{id}/answers

POST /api/transcription
POST /api/answer-evaluation
POST /api/skill-verification

POST /api/speech-behavior/analyze
POST /api/gaze/analyze
POST /api/av/analyze

POST /api/evidence-fusion
GET  /api/interviews/{id}/report
```

Do not force every module to execute synchronously if it makes the UI unusable.

Use background jobs where appropriate.

---

# 23. Data Contract Between Modules

Every feature must return structured JSON.

Example:

```json
{
  "feature": "speech_behavior",
  "score": 68,
  "confidence": 0.81,
  "level": "HIGH",
  "evidence": [
    {
      "type": "speech_rate",
      "severity": "medium",
      "message": "Speech rate increased 39% relative to candidate baseline."
    }
  ],
  "quality": {
    "audio_quality": 0.93,
    "baseline_sufficient": true
  }
}
```

The evidence-fusion engine must consume this structure rather than depending on feature-specific internals.

---

# 24. Storage Rule

Keep raw media separate from derived data.

```text
storage/
├── raw_audio/
├── raw_video/
├── transcripts/
├── features/
├── evaluations/
└── reports/
```

For the hackathon, local storage is acceptable.

Production system should use secure encrypted object storage.

---

# 25. Dataset / Demo Data

Use only synthetic candidate identities for the hackathon.

No real biometric data.

Prepare at least:

### Candidate A — Genuine Strong

- claims match ability
- good answers
- normal behavior

### Candidate B — Genuine Weak

- real person
- weak skills
- poor answers
- no cheating signals

This tests the difference between **poor candidate** and **suspicious candidate**.

### Candidate C — AI-assisted simulation

- strong initial answers
- controlled latency anomalies
- speech/behavior changes
- weak follow-up reasoning

### Candidate D — Deepfake / identity simulation

- synthetic/manipulated media
- A/V inconsistencies

### Candidate E — Nervous Genuine Candidate

- behavioral anomalies
- pauses
- pitch changes
- but strong knowledge and identity consistency

Expected result:

```text
Speech anomaly HIGH
Overall integrity risk not necessarily HIGH
```

This is essential for demonstrating that the system does not overreact to one feature.

---

# 26. Testing Strategy

Tests must exist at three levels.

## Unit tests

Each feature independently.

Examples:

- resume extraction
- answer scoring
- concept coverage
- speech extraction
- baseline calculation
- gaze signal
- evidence fusion

## Integration tests

Test:

```text
resume
→ question
→ answer
→ evaluation
→ skill verification
→ risk fusion
```

## End-to-end demo tests

Run complete candidate scenarios.

---

# 27. Explainability Rules

Every high or medium evidence signal should have a reason.

Bad:

```text
Speech anomaly: 72
```

Good:

```text
Speech anomaly: 72

Reason:
Speech rate increased 42% from candidate baseline
over two consecutive questions.
```

The report should distinguish:

- measured feature
- model inference
- recommended review action

Do not fabricate evidence.

---

# 28. Error Handling

The system must gracefully handle:

- empty audio
- low audio volume
- microphone failure
- transcription failure
- no face found
- face lost temporarily
- multiple faces
- poor lighting
- background noise
- unsupported resume format
- LLM/API unavailable
- missing reference answer
- skill not assessed

When data is insufficient:

```text
status = "INSUFFICIENT_EVIDENCE"
```

Do not convert missing data into risk.

---

# 29. Privacy / Security

For the hackathon:

- synthetic candidate data only
- no real biometric information
- local demo storage
- no unnecessary data collection

Do not store raw media longer than necessary in a production design.

Do not log sensitive media content in application logs.

---

# 30. Model Strategy

Do not train giant models during the hackathon.

Use existing models/tools.

Suggested:

### Speech

- faster-whisper / WhisperX
- openSMILE

### Semantic evaluation

- sentence-transformers embeddings
- LLM rubric evaluator

### Face / A-V

- OpenCV
- MediaPipe
- appropriate open-source face/lip-sync/deepfake components

### Anomaly detection

- deterministic heuristics first
- Isolation Forest if useful

### Question generation

- LLM

### Report generation

- LLM grounded only in stored evidence

---

# 31. Important Model-Orchestration Rule

LLM should NOT directly own the final risk score.

Use the LLM for:

- structured extraction
- question generation
- reference answer generation
- answer analysis
- explanations

Use deterministic / statistical logic for:

- signal normalization
- scoring
- evidence fusion
- thresholding
- confidence calculation

This keeps the system explainable and reduces hallucination risk.

---

# 32. What Makes This Project Different

Do not pitch it simply as:

> "We detect deepfakes."

Existing deepfake detectors already exist.

Pitch it as:

> **A multimodal interview-integrity engine that verifies whether a candidate's claimed identity, skills, answers and interview behavior remain consistent throughout an interview.**

Potential system-level differentiator:

### Candidate-specific knowledge ownership / counterfactual stress test

```text
Candidate claim
      ↓
Knowledge graph
      ↓
Controlled change in one assumption
      ↓
Adaptive follow-up
      ↓
Reasoning consistency
      ↓
Knowledge ownership evidence
```

Do not claim “first ever” unless prior-art research establishes it.

---

# 33. Evidence Fusion Philosophy

The central rule is:

> **No single signal can determine cheating.**

For example:

```text
voice anomaly
    alone
    ↓
weak evidence
```

But:

```text
resume mismatch
+
poor answer correctness
+
high response latency
+
speech anomaly
+
gaze anomaly
```

may justify:

```text
HIGH INTEGRITY RISK
```

This is an evidence-fusion system.

---

# 34. Final Scoring Output

Example:

```text
Candidate:
Rahul Sharma

INTEGRITY RISK
74 / 100
HIGH

CONFIDENCE
87%

Evidence:

Resume / Answer mismatch       HIGH
Knowledge ownership             HIGH
Speech behavior                 MEDIUM
Gaze behavior                   MEDIUM
A/V integrity                   LOW
Response timing                 HIGH

Recommended action:
Conduct targeted verification interview.
```

Again:

**This is a recommendation for human review, not a hiring decision.**

---

# 35. Interviewer Controls

The interviewer should be able to:

- create/edit questions
- edit generated reference answers
- edit key concepts
- select which resume skills to test
- set difficulty
- start/pause/end interview
- review transcript
- review answer scores
- review evidence
- override incorrect AI-generated question metadata
- export report

---

# 36. AI Question Generation

If interviewer chooses “Generate questions from resume”:

```text
Resume
 ↓
Skill extraction
 ↓
Claim strength
 ↓
Question generation
 ↓
Question → Skill mapping
 ↓
Reference answer
 ↓
Key concepts
 ↓
Difficulty
 ↓
Interviewer approval
```

The interviewer must approve/edit questions before interview start.

---

# 37. Question Types

Support:

- conceptual
- coding/technical
- scenario
- architecture
- troubleshooting
- project-specific
- resume-claim verification
- counterfactual follow-up

---

# 38. Coding Questions

For coding questions, do not rely on speech evaluation only.

For the hackathon, support textual/code answer evaluation as an optional future extension.

Current MVP focus:

- spoken technical questions
- spoken answers
- transcript
- semantic evaluation

---

# 39. Recommended Build Order

Claude Code should implement in this order:

## Phase 1

Project skeleton:

- frontend
- FastAPI
- DB
- config
- authentication placeholder if needed

## Phase 2

Interview setup:

- candidate
- resume upload
- question bank
- reference answer
- skill mapping

## Phase 3

Interview recording:

- microphone
- camera
- question progression
- answer recording

## Phase 4

Speech-to-text:

- Whisper
- transcript storage
- question-specific transcript

## Phase 5

Answer evaluation:

- embeddings
- concept coverage
- LLM rubric
- question score

## Phase 6

Resume skill verification:

- aggregate per skill
- mismatch report

## Phase 7

Speech behavior:

- openSMILE
- baseline
- anomaly score

## Phase 8

Gaze / video behavior

## Phase 9

A/V integrity

## Phase 10

Evidence fusion

## Phase 11

Final dashboard/report

## Phase 12

Testing + demo data + polishing

---

# 40. MVP Priority if Time Becomes Limited

Implement in this priority order:

1. Resume parsing
2. Question/reference-answer setup
3. Audio recording
4. Speech-to-text
5. Answer evaluation
6. Resume/skill verification
7. Speech behavior
8. Response timing
9. Final risk fusion
10. Gaze
11. Lip-sync/A-V
12. Advanced knowledge ownership

If time runs out, the first seven should still produce a coherent product.

---

# 41. Claude Code Development Rules

Claude Code must:

- inspect all provided `.md` feature files before coding
- reuse existing repositories/components where their licenses and integration allow
- not duplicate functionality unnecessarily
- keep modules isolated
- create clear interfaces between modules
- write tests
- avoid hard-coded secrets
- use environment variables for API keys
- document installation
- document model dependencies
- document licenses
- document dataset sources
- provide a single-command local development setup
- provide Docker support if practical
- keep the demo runnable offline where possible
- make external LLM APIs replaceable through an adapter interface

---

# 42. Repository Integration Rules

The project may contain additional feature-specific MD files such as:

```text
docs/features/
├── resume_skill_verification.md
├── voice_prosody.md
├── lip_sync.md
├── eye_gaze.md
├── transcription.md
├── answer_evaluation.md
└── ...
```

Claude Code should:

1. Read every feature MD.
2. Identify overlapping functionality.
3. Choose one canonical implementation for each component.
4. Avoid installing duplicate models doing the same task.
5. Create adapters around external repositories rather than deeply coupling the whole application to them.
6. Preserve attribution/license notices where required.
7. Create a final `docs/ARCHITECTURE.md`.

---

# 43. Required Final Project Deliverables

Claude Code must produce:

```text
/frontend
/backend
/docs
/data
/tests
/scripts
.env.example
README.md
docker-compose.yml (if practical)
Dockerfile(s)
```

And documentation:

```text
docs/
├── ARCHITECTURE.md
├── API.md
├── DATA_MODEL.md
├── MODELS.md
├── DATASETS.md
├── SECURITY.md
├── LIMITATIONS.md
└── DEMO_GUIDE.md
```

---

# 44. Required README

README must contain:

- project title
- TCS problem statement
- architecture diagram
- features
- setup
- environment variables
- model dependencies
- dataset references
- GitHub/open-source dependencies
- licensing notes
- running instructions
- test instructions
- demo instructions
- known limitations
- screenshots/placeholders if appropriate

---

# 45. Final Demo Scenario

Prepare a 3–5 candidate synthetic demo.

## Candidate 1 — Genuine strong

Expected:

```text
low risk
high confidence
strong skill verification
normal behavior
```

## Candidate 2 — Genuine but weak

Expected:

```text
low integrity risk
low technical score
no false cheating accusation
```

## Candidate 3 — AI-assisted simulation

Expected:

```text
medium/high integrity risk
resume/answer mismatch
latency anomalies
behavior anomalies
```

## Candidate 4 — Synthetic deepfake

Expected:

```text
high A/V risk
possibly high overall risk
```

## Candidate 5 — Nervous genuine candidate

Expected:

```text
speech/gaze anomaly may be high
overall integrity risk should not automatically be high
```

This demonstrates robustness.

---

# 46. Final Product Principle

The final product must answer three separate questions:

### Capability

> Does the candidate demonstrate the skills claimed on the resume?

### Authenticity

> Does the audio/video/identity appear consistent?

### Interview behavior integrity

> Are there multiple independent signals suggesting that the interaction deserves verification?

Then combine those into:

> **Interview Integrity Risk**

not:

> **Cheating = true/false**

---

# 47. Master Definition of Done

The project is done when an interviewer can:

1. Upload a synthetic candidate resume.
2. Extract claimed skills.
3. Manually enter or AI-generate questions.
4. Enter or AI-generate reference answers.
5. Map questions to skills.
6. Start the interview.
7. Record candidate audio/video.
8. Convert spoken answers to text.
9. Evaluate semantic correctness.
10. Compare candidate performance to resume claims.
11. Analyze speech behavior.
12. Analyze gaze behavior.
13. Analyze lip-sync/A-V integrity.
14. Analyze timing/behavioral anomalies.
15. Optionally run knowledge-ownership/counterfactual tests.
16. Fuse evidence.
17. Produce LOW/MEDIUM/HIGH risk.
18. Produce a confidence score.
19. Explain the reasons.
20. Show evidence question-by-question and skill-by-skill.
21. Recommend human verification where appropriate.
22. Never automatically make a hiring decision.

---

# 48. Final Instruction to Claude Code

### Build the project as a production-style modular prototype, not a collection of disconnected demos.

Use this document as the **system-level specification**.

Use the separate feature-specific `.md` files as the **implementation specifications** for their respective modules.

First inspect the repository and all provided Markdown files.

Then produce:

```text
1. architecture plan
2. implementation plan
3. project structure
4. backend
5. frontend
6. database
7. model adapters
8. feature integrations
9. evidence fusion
10. tests
11. synthetic demo dataset
12. setup documentation
13. demo workflow
```

Before implementing a feature from scratch, check whether a supplied feature MD identifies an existing open-source repository/model that can be safely reused.

Do not blindly clone external repositories. Integrate only the required components.

Keep external dependencies behind adapters.

Keep all risk weights and thresholds configurable.

Keep all evidence traceable to a specific question, timestamp, feature module and measurement.

The application must continue to work if an individual optional feature is unavailable. In that case, mark the feature:

```text
UNAVAILABLE / INSUFFICIENT EVIDENCE
```

and reduce confidence rather than manufacturing evidence.

The final project must be locally runnable and suitable for a TCS hackathon demo.

---

# 49. Final One-Line Product Definition

> **VeriInterview is a multimodal interview-integrity platform that verifies resume-claimed skills and spoken answers while analyzing identity, audio-video synchronization, gaze, speech behavior and interaction anomalies to produce an explainable risk score and confidence level for human interviewer review.**
