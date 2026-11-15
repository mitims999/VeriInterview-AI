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

## Authentication, organizations and demo mode

- **Accounts.** Interviewers belong to exactly one organization. There is no public sign-up: an admin runs
  `python scripts/manage_auth.py create-org ...` / `create-user ...` (password via prompt or `VERI_PASSWORD`, never argv).
  For a fresh deployment set `BOOTSTRAP_ORG_NAME`, `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD` once; the
  first admin is created at startup if that email does not exist. Note: SQLite on a free Render tier is wiped on each
  deploy, so accounts there only persist if the bootstrap variables are set (or a persistent disk / Postgres is used).
- **Passwords** are bcrypt-hashed (min 10 chars, max 72 bytes). Unknown-email logins burn an equal bcrypt check and return
  the same message, so accounts cannot be enumerated by response or timing.
- **Sessions** are random 256-bit tokens in an `HttpOnly`, `SameSite=Lax` cookie (`Secure` when `SESSION_COOKIE_SECURE=1`).
  Only the SHA-256 of the token is stored. Absolute lifetime 12 h, idle timeout 2 h (`SESSION_TTL_HOURS`, `SESSION_IDLE_MINUTES`).
- **Rate limiting.** 5 failed sign-ins per (email, IP) per 15 min, 20 per IP; HTTP 429 with `Retry-After`. The limiter is
  in-memory (per process); use a shared store if you run several workers.
- **CSRF.** Writes carrying an `Origin` header must match `CORS_ORIGINS` (extend with `CORS_EXTRA_ORIGINS`) or the server itself.
- **Authorization.** Every `/api` route requires a session except `/api/health` and `/api/auth/login|demo`. Interviews,
  candidates, questions, answers, audio analyses, vision jobs and saved reports are scoped by `organization_id`;
  cross-organization access returns 404. Rows that predate auth have no organization and are invisible until claimed with
  `manage_auth.py claim-legacy --org <slug> --apply` (dry run by default). Note `GET /api/interviews/{id}/report` still
  persists a risk record and marks the interview COMPLETED, as before.
- **Schema changes** are additive and reversible (`backend/database/migrations.py`: nullable columns only, idempotent;
  `manage_auth.py migrate-down` removes them). Back up `storage/veriinterview.db` before first run on real data.
- **Demo mode.** `POST /api/auth/demo` opens a password-less session in a separate, flagged organization whose content is the
  fictional `data/synthetic` set. Persisting or heavy actions (creating/editing interviews, vision jobs, audio upload,
  live audio, video analysis, saving reports) return 403; the demo account cannot be signed into with a password.
- **Known limits.** Live audio uses a WebSocket authenticated by the session cookie; Vercel rewrites do not proxy WebSockets,
  so live audio on the Vercel deployment needs the socket pointed at the backend origin (pre-existing deployment limitation).
