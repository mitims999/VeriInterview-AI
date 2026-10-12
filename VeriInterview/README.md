# VeriInterview

**TCS Problem Statement P1 — Deepfake Interview Alert Tool** (Theme: AI + Cyber Defense.
Expected AI feature: risk scoring + anomaly detection.)

> VeriInterview is a multimodal interview-integrity platform that verifies resume-claimed
> skills and spoken answers while analyzing identity, audio-video synchronization, gaze,
> speech behavior and interaction anomalies to produce an explainable **0–100 Integrity
> Risk Score** and confidence level for human interviewer review. **It never makes a
> hiring or disqualification decision.**

## Architecture

```
Resume + JD ──► Resume-Matcher adapter ──► skill claims / mismatch
Interview Q&A ──► sentence-transformers/TF-IDF ──► answer correctness ──► skill verification
Video ──► InsightFace adapter (identity) ─┐
       ──► SyncNet adapter (lip-sync)     ├──► Evidence Fusion Engine ──► Risk (0-100)
       ──► Proctoring-AI adapter (gaze)   │      (deterministic, weighted,           + Confidence
Audio ──► openSMILE adapter (voice)       │       capped, never LLM-controlled)      + Capability
Claims ──► Claim Graph + CCST (original) ─┘
                                                      │
                                                      ▼
                                       Human Interviewer Dashboard
```

Full diagram and module ownership: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Features

- Interview setup: candidates, resume upload (PDF/DOCX/TXT), question bank, reference
  answers, skill mapping, difficulty (Feature 1).
- Resume parsing + skill/claim-strength extraction via a Resume-Matcher-shaped adapter
  (Feature 2/3).
- Answer correctness via semantic similarity + concept coverage, never keyword-only
  (Feature 5).
- Resume-to-interview skill verification with 6 verdict levels, including the
  `NOT_ASSESSED` (never `DOES_NOT_KNOW`) rule for untested skills (Feature 3).
- Voice modulation (openSMILE eGeMAPSv02-shaped, candidate-specific baseline, Feature 8).
- Eye gaze / head-pose / multi-person / phone-in-frame behavior signals
  (Proctoring-AI-shaped, Feature 7).
- Lip-sync / A-V consistency (SyncNet-shaped, Feature 6).
- Face identity consistency (InsightFace-shaped).
- Response timing / prompting-anomaly detection (Feature 9).
- **Knowledge Ownership / Counterfactual Stress Test (CCST)** — the project's core
  novelty: perturbs one assumption behind a strong claim and checks whether the
  candidate's reasoning adapts consistently (Feature 10).
- Deterministic Evidence Fusion Engine — weighted, capped, corroboration-aware, never
  LLM-controlled — producing a 0-100 Integrity Risk, a separate Confidence score, and a
  separate Candidate Capability score.
- Full explainable report: skill-by-skill, question-by-question, evidence timeline,
  human-review recommendation.

## Repositories referenced (Build Spec)

| Feature | Repo |
|---|---|
| Resume/JD skill mismatch | github.com/srbhr/Resume-Matcher |
| Lip-sync | github.com/joonson/syncnet_python |
| Eye gaze / behavior | github.com/vardanagarwal/Proctoring-AI |
| Voice modulation | github.com/audeering/opensmile, opensmile-python |
| Face identity | github.com/deepinsight/insightface |
| Expected-answer matching | github.com/UKPLab/sentence-transformers |

**Important:** this prototype was built in an environment with no GPU, no camera/mic
capture, and no guaranteed access to multi-GB model downloads. Each of the repos above is
wrapped by an **adapter** (`backend/adapters/`) that implements the exact contract the
Build Spec specifies for that module, computed from structured synthetic signals that
stand in for the real model's output. Every adapter file documents precisely what one
function to replace to call the real repo once audio/video/GPU are available — see
[`docs/MODELS.md`](docs/MODELS.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
The evidence-fusion, risk-scoring, skill-verification and knowledge-ownership logic is
**not** a stub — it is real, tested, deterministic code that runs on whatever evidence
each adapter supplies.

## Setup

### Backend

```bash
cd VeriInterview
pip install -r requirements.txt
cp .env.example .env       # optional — LLM_PROVIDER=none works fully offline
python scripts/generate_fixtures.py   # (re)generates the 5 synthetic demo candidates
uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd VeriInterview/frontend
npm install
npm run dev
```

Open http://localhost:5173 — the dropdown at top-right switches between the 5 synthetic
demo candidates, each showing a live-computed Integrity Risk score, all 6 evidence
components, skill verification, question-by-question evaluation, and the evidence
timeline.

## Environment variables

See [`.env.example`](.env.example). `LLM_PROVIDER=none` (default) runs the whole system,
including the demo dataset, without any external API key.

## Model & dataset references

[`docs/MODELS.md`](docs/MODELS.md), [`docs/DATASETS.md`](docs/DATASETS.md).

## Licensing notes

openSMILE requires a **separate commercial license from audEERING for commercial use**
(research use is free). See [`docs/SECURITY.md`](docs/SECURITY.md) for this and other
repo license notes (InsightFace model weights, SyncNet).

## Running tests

```bash
cd VeriInterview
python -m pytest tests/ -v
```

20 tests: unit tests per module, an integration test across the full
resume→question→answer→evaluation→skill-verification→risk-fusion path, and end-to-end
tests over all 5 synthetic demo scenarios asserting the specific robustness properties
the demo is designed to prove (e.g. a genuine-but-weak candidate must score LOW risk
despite low capability; a nervous genuine candidate's elevated speech/gaze signals must
not alone push overall risk to HIGH).

## Demo instructions

See [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) for a full candidate-by-candidate walkthrough
and talking points.

## Known limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Project structure

```
VeriInterview/
  backend/    FastAPI app — config, schemas, database, adapters/, modules/
  frontend/   React + Vite + TypeScript + Tailwind dashboard
  data/       5 synthetic demo candidates
  docs/       architecture, API, data model, models, datasets, security, limitations, demo guide
  scripts/    generate_fixtures.py
  tests/      unit / integration / e2e tests
  storage/    sqlite db (local, gitignored)
```
