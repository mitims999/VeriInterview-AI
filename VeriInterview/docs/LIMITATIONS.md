# Known Limitations

1. **The voice adapter (openSMILE, M1) still runs on structured fixture signals.** The M3 video checks (face identity,
   lip-sync, gaze/surroundings, blink/mouth cues) now run real models on uploaded video (`backend/vision/`, `docs/m3/`).
   The demo candidates still use their fixture signals through the same adapters.
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
8. **M3 video checks: known limits** (measured; details in `docs/m3/calibration/`):
   - **Modern lip-sync generators are not caught.** Wav2Lip / TalkLip / SadTalker / MakeItTalk fakes (AVLips): 0 of 30 flagged,
     because they are trained against SyncNet (as LipFD, NeurIPS 2024, reported). Face-swaps (31 of 32 flagged with a reference photo),
     identity switches, frozen faces and dubbed or out-of-sync audio are caught. A dedicated detector (LipFD) is the planned fix.
   - **Within-session identity cannot see a fake that runs from the first frame.** It needs the optional reference photo.
   - **Gaze is partial evidence:** it catches reading from off-screen material (OEP: 25 % off-screen vs 2 % normal), but not
     cheating that keeps the eyes on the screen. On its own it is capped at MEDIUM.
   - **Short phone glances can be missed:** a phone alert needs 3 consecutive 1-fps detections (this favours avoiding false alarms).
   - **Speed:** about 0.9x real time on a laptop CPU (a 3-min interview takes ~2.5 min on AC power, 2-4x slower on battery). No GPU path yet.
   - **Little speech means no lip-sync verdict:** reported as "not assessed", never as risk.
   - Thresholds were calibrated on public research data (GRID, LFW, OEP, COCO, DeepfakeTIMIT, AVLips), not on real interview recordings.
