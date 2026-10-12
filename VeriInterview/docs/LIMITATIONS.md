# Known Limitations

1. **A/V, gaze, identity and voice adapters run on structured fixture signals, not raw
   media.** This environment has no camera/microphone capture and no GPU, so
   `syncnet_adapter.py`, `proctoring_adapter.py`, `insightface_adapter.py`, and
   `opensmile_adapter.py` compute their outputs from pre-recorded aggregate signals
   (`av_signals`, `behavior_signals`, `identity_signals`, `speech_features`) rather than
   frame-by-frame model inference on a video/audio file. Each adapter's docstring states
   exactly what to swap in to run the real model. This does not affect the fusion/scoring
   logic, which is real and fully deterministic.
2. **Resume/JD keyword extraction is a lightweight heuristic**, not Resume-Matcher's full
   textacy pipeline — it recognizes acronyms, CamelCase identifiers, and a small curated
   technical-phrase list rather than running a full NLP key-term extractor. Skill-match
   scores will be lower than a production textacy-based extractor on verbose resumes.
3. **No LLM is wired in.** Question generation, reference-answer generation, and
   counterfactual follow-up generation are LLM jobs in production (Master Spec Section 31)
   but no `LLM_API_KEY` is configured here; the synthetic fixtures pre-author what an LLM
   step would have produced. The interviewer-facing "AI generate questions" button in the
   spec is represented by the `/api/interviews/{id}/questions` manual-entry endpoint only —
   the AI-generation endpoint is not implemented in this build.
4. **Single-tenant, single-interview-at-a-time.** No auth, no multi-user concurrency
   handling, no background job queue — acceptable for a hackathon demo per Master Spec
   Section 22 ("do not force every module to execute synchronously if it makes the UI
   unusable" — here, execution is fast enough that synchronous is fine).
5. **Fusion weights are prototype values.** They live in `backend/config.py::FusionWeights`
   and are explicitly documented as tunable, per Master Spec Section 14/Build Spec Section 4.
   They have not been calibrated against a labeled dataset (none exists for this problem).
6. **Coding-question evaluation is out of scope** for this MVP (Master Spec Section 38) —
   only spoken/transcribed technical questions are evaluated.
7. **Interviewer "start interview" live recording flow (Feature 3 of Master Spec Section 4)
   is not implemented** — this build focuses on interview setup + post-hoc report
   generation from either the synthetic dataset or manually-entered interview answers, not
   live MediaRecorder/WebAudio capture in the browser.
