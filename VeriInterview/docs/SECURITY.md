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
- **Licensing note — SyncNet:** joonson/syncnet_python is provided for non-commercial
  research use; a commercial deployment needs the authors' permission.
- CORS is restricted to the local Vite dev origin (`backend/config.py::CORS_ORIGINS`) —
  update before any non-local deployment.
