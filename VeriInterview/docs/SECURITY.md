# Security & Privacy

- **Synthetic data only.** All 5 demo candidates in `data/synthetic/` are fabricated. No
  real biometric data, headshots, voice recordings, or personal information is stored
  anywhere in this repository.
- **No hardcoded secrets.** `LLM_API_KEY` is read from the environment (`.env`, see
  `.env.example`); the app runs with zero external API keys configured by default
  (`LLM_PROVIDER=none`).
- **Local storage only.** SQLite database and any future raw media live under `storage/`,
  which is excluded from version control (`.gitignore`). A production deployment must move
  this to encrypted object storage and delete raw media on a retention schedule — this
  prototype does not implement that lifecycle.
- **No hiring decisions.** Every risk assessment ends with a human-review recommendation
  string, never an automated accept/reject action. This is enforced at the schema level:
  `RiskAssessment` has no field that could represent a final decision.
- **Licensing note — openSMILE:** the `audeering/opensmile` / `opensmile-python` feature
  extractor is free for academic/research use under the audEERING research license.
  **Commercial deployment of the speech-behavior module requires a separate commercial
  license from audEERING** — flagged here per Build Spec instruction 8. Do not ship this
  module commercially without resolving that license.
- **Licensing note — InsightFace:** the InsightFace *code* is MIT-licensed, but the
  pretrained model weights (e.g. buffalo_l) carry their own non-commercial research terms
  from the original authors/dataset licenses — verify before commercial use.
- **Licensing note — SyncNet:** joonson/syncnet_python code is **MIT** (repo `LICENSE.md`, vendored with its notice in
  `backend/vision/syncnet/`). The pretrained `syncnet_v2.model` weights are hosted by VGG Oxford, so check their terms
  before commercial use.
- **Uploaded media (M3):** interview videos and reference photos exist only as temp files while an analysis runs, and are
  deleted afterwards (`backend/api_vision.py`, `/api/analyze`). Only derived numbers and evidence text are kept (feature
  cache in `storage/features/`, gitignored). No frames, audio or face embeddings are written to disk or to logs.
- **Public calibration datasets (M3)** contain real faces and voices. They live only in gitignored `datasets/`, are used
  offline for calibration and tests, never ship, never appear in the demo, and are deleted at the end of M3 (task T-5,
  `docs/m3/M3_PROGRESS.md`). Several are non-commercial (RT-BENE, MPIIFaceGaze, AV-Deepfake1M, DeepfakeTIMIT/VidTIMIT).
- CORS is restricted to the local Vite dev origin (`backend/config.py::CORS_ORIGINS`) —
  update before any non-local deployment.
