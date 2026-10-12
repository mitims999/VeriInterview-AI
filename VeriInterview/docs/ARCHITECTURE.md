# VeriInterview — Architecture

## One-line definition

VeriInterview is a multimodal interview-integrity platform that verifies resume-claimed
skills and spoken answers while analyzing identity, audio-video synchronization, gaze,
speech behavior and interaction anomalies to produce an explainable 0–100 risk score and
confidence level for human interviewer review. It never makes a hire/reject decision.

## Pipeline

```
Resume + JD --> Resume-Matcher adapter --> SkillProfile (claims, matched/missing keywords)
                                                 |
Interview Q&A --> Answer Evaluation (sentence-transformers/TF-IDF + concept coverage)
                                                 |
                         Skill Verification (per-skill verdict aggregation)
                                                 |
Video --> InsightFace adapter (identity)   ---> ModuleResult
       --> SyncNet adapter (lip-sync)      ---> ModuleResult
       --> Proctoring-AI adapter (gaze)    ---> ModuleResult
Audio --> openSMILE adapter (speech)       ---> ModuleResult
Claims --> Claim Graph + CCST (knowledge ownership) --> ModuleResult
Timing --> Response Timing analysis  --------------> folds into profile-consistency
                                                 |
                                   Evidence Fusion Engine (deterministic, weighted, capped)
                                                 |
                              0-100 Integrity Risk + Confidence + Capability Score
                                                 |
                                        Human Interviewer Dashboard
```

## Module ownership (Section 21 of the Build Spec)

| Module | Repo wrapped | File |
|---|---|---|
| Resume/JD skill mismatch | srbhr/Resume-Matcher | `backend/adapters/resume_matcher.py` |
| Lip-sync | joonson/syncnet_python | `backend/adapters/syncnet_adapter.py` |
| Gaze / behavior | vardanagarwal/Proctoring-AI | `backend/adapters/proctoring_adapter.py` |
| Voice modulation | audeering/opensmile(-python) | `backend/adapters/opensmile_adapter.py` |
| Face identity | deepinsight/insightface | `backend/adapters/insightface_adapter.py` |
| Expected-answer matching | UKPLab/sentence-transformers | `backend/adapters/sentence_embed.py` |
| Knowledge ownership / CCST | original (no repo) | `backend/modules/knowledge_ownership.py` |
| Evidence fusion | original (no repo) | `backend/modules/evidence_fusion.py` |

Every adapter file's docstring documents (a) the exact repo it wraps, (b) why this
environment computes the adapter's output from structured synthetic signals instead of
running the real model on raw media, and (c) exactly what to change to plug in the real
model when GPU/video/audio infrastructure is available. See `docs/LIMITATIONS.md`.

## Evidence contract

Every analysis module returns a `ModuleResult` (`backend/schemas.py`):
`{feature, score, confidence, level, evidence[], quality, status}`. No module ever
returns a decision. The Evidence Fusion Engine (`backend/modules/evidence_fusion.py`)
is the only place scores are combined, and it is a deterministic weighted formula —
never LLM-controlled, per Master Spec Section 31.

## Directory layout

```
VeriInterview/
  backend/        FastAPI app, config, schemas, adapters/, modules/, database/
  frontend/       React + Vite + TypeScript + Tailwind + Recharts dashboard
  data/synthetic/ 5 synthetic candidate fixtures (Master Spec Section 25)
  docs/           this file + API/DATA_MODEL/MODELS/DATASETS/SECURITY/LIMITATIONS/DEMO_GUIDE
  scripts/        generate_fixtures.py
  tests/          unit + integration tests
  storage/        sqlite db + (empty) raw_audio/raw_video/reports for a live deployment
```
