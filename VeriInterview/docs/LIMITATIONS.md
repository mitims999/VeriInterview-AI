# Known Limitations

1. **Vision adapters still use structured fixture signals.** SyncNet, Proctoring-AI
   and InsightFace require their real models and video processing integrations.
   Member 1's audio path now uses real FFmpeg, Whisper and openSMILE inference;
   synthetic demos retain their original feature-summary interface. See
   [audio limits and verification](AUDIO_INTELLIGENCE.md).
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
7. **Audio live capture is available separately from the full interview workflow.**
   The Audio intelligence workspace uses WebAudio PCM capture and a WebSocket for
   provisional/final transcription and prosody. It requires explicit answer boundaries,
   has a two-minute answer limit, and assumes a single speaker. It does not implement
   a complete live video interview, diarization, or automated question segmentation.
