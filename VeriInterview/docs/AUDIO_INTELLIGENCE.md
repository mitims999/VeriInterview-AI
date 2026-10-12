# Member 1: Audio intelligence and transcription

Branch: `audio-intelligence-and-transcription`. F1 is Whisper transcription;
F3b is speech behavior consistency analysis. The scoring/fusion owner remains
Member 5; no final fusion weights or other members' models were changed.

## Run locally

From `VeriInterview/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# macOS, if FFmpeg is missing: brew install ffmpeg
python scripts/download_audio_model.py --model base
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd VeriInterview/frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173` and select **Audio intelligence**. Browser microphone
capture requires localhost or HTTPS and an explicit permission grant. Browser
capture requests disabled automatic gain, echo cancellation and noise suppression;
devices may still apply processing. Keep the microphone position consistent.

`.env.example` lists settings, but Python does not implicitly load `.env`.
Export settings in the shell or launch Uvicorn with `--env-file .env`.
Default `WHISPER_MODEL=base`, CPU `int8`, four CPU threads. The model is cached
under ignored `storage/models/`. Inference defaults to local files only;
model downloads are explicit. `WHISPER_CACHE_DIR`, `WHISPER_DEVICE`, and
`WHISPER_COMPUTE_TYPE` are configurable. Do not set CUDA on a Mac.

The default Docker image installs the original text/demo requirements only.
To use audio in a container, install FFmpeg and `requirements-audio.txt`, cache
the model in the mounted storage volume, and point the frontend proxy at the
backend container. Container audio deployment has not been verified here.

## Uploaded recordings

The extractor accepts decodable local audio/video containers via FFmpeg, limits
media to 250 MiB and 60 minutes, rejects files without audio, and writes mono
16 kHz float WAV. It does not overwrite or normalize the original. Channel
averaging avoids FFmpeg's default stereo gain boost. Separate speakers mixed
into channels are not separated automatically.

The API uses multipart upload, not arbitrary client-supplied server paths:

```bash
curl -F 'file=@interview.mp4' -F 'language=en' \
  http://127.0.0.1:8000/api/audio/transcribe

curl -F 'file=@interview.mp4' \
  -F 'answers=[{"question_id":"Q1","start_sec":0,"end_sec":10},{"question_id":"Q2","start_sec":10,"end_sec":20},{"question_id":"Q3","start_sec":20,"end_sec":30}]' \
  http://127.0.0.1:8000/api/speech-behavior/analyze
```

Answer windows must be chronological, non-overlapping, within the recording,
and use unique question IDs (maximum 100). Without windows, the whole file is
one answer: transcription and measurements work, but a baseline anomaly score
is unavailable. Whisper segments are not treated as question boundaries.
Recordings must contain only the candidate within each answer window; diarization
is not implemented.

## Live WebSocket protocol

Connect to `/api/audio/live` (the Vite proxy enables WebSockets). Browser origins
are checked against the existing CORS allowlist. Start an answer:

```json
{"action":"start","question_id":"Q1","sample_rate":16000,"format":"pcm_s16le","language":"en"}
```

Wait for `{"type":"ready","question_id":"Q1"}`, then send binary mono PCM16
little-endian packets, at most 64,000 bytes each. WebM/Opus blobs are not raw PCM
and are rejected. The included AudioWorklet sends 2,048-sample packets.

Every approximately three seconds of received audio, the server returns a
`transcript` message with `question_id`, `revision`, `final:false`, and
`transcription`. Replace the previous provisional text for that question; do
not append it. Use `{"action":"end_answer"}` to flush/finalize the current
answer and receive `final:true`, acoustic `answer` measurements, and an updated
`speech_behavior` report. The browser flushes the last short PCM packet first.

Start another answer with the same format metadata and `action:"start_answer"`.
Use `{"action":"stop"}` to finalize any active answer, persist the session, and
receive `{"type":"complete","result":...}`. Question IDs must be unique.
All timestamps use accumulated recorded-audio time; wall-clock gaps while the
microphone is stopped are not represented or interpreted as response latency.

This is bounded answer-level near-real-time ASR, not native token streaming:
the current answer is redecoded for each provisional revision. Answers are
limited to 120 seconds (UI finishes at 115 seconds); sessions to 60 minutes.
The server processes packets sequentially, with backpressure while inference
runs. Update latency depends on the CPU/model and can exceed three seconds.
An idle connection closes after 90 seconds. A disconnected session has no
automatic resume; only an explicit completed session is persisted. Concurrent
sessions share a serialized model inference lock; this is a local prototype.

## F3b measurements and scoring

- openSMILE eGeMAPSv02 low-level descriptors supply pitch, auditory loudness,
  jitter, shimmer and HNR. F0 is converted from semitones **per frame** to Hz.
- Pitch range and loudness range are 95th–5th percentile ranges to reduce spikes.
- Auditory loudness is labeled in openSMILE relative units; waveform recording
  level is separately measured in dBFS, not calibrated physical sound pressure.
- Silero VAD supplies speech/silence intervals. Internal pauses are gaps of at
  least 300 ms between speech regions, excluding leading/trailing silence.
- Speech rate is recognized words / full answer minutes. Articulation rate is
  recognized words / VAD speech minutes. Word timestamps are ASR estimates, not
  forced-alignment guarantees; recognition errors affect these rates. Missing
  aligned words produce an unavailable rate, not an invented zero.
- One-second acoustic change events have timestamps; they are measurements,
  not cheating flags. Jitter/shimmer/HNR are supporting measurements only.

The first two **usable** answers form a fixed candidate/session-specific baseline.
They are never scored against themselves. At least one later usable answer is
needed. Each feature uses baseline median and standard deviation with a
measurement-scale variance floor. Z deviations are capped at 4. The configured
pitch/loudness/rate/pause/ratio weights are 25/20/25/20/10%; pitch and loudness
each allocate 70% to mean/level and 30% to variability. Missing features are
excluded and weights renormalized. Pause ratio measures internal gaps; speech
ratio uses total recording duration, so they are not identical complements.

The weighted capped deviation maps linearly to 0–100. Per-answer results are
averaged for the module summary; individual results remain available. Speech
thresholds are LOW ≤34, MEDIUM ≤64, HIGH >64. Consecutive high-answer counts
are exposed without automatically multiplying risk. Question difficulty is
retained as context, not an unvalidated automatic score correction.

Audio quality accounts for speech duration, short recordings, very low level,
clipping, and estimated speech/background energy contrast when measurable.
Too-short, silent, or low-quality audio is unscored. Measurement reliability is
`current_quality × mean_baseline_quality × available_feature_weight × baseline_size_factor`.
The baseline-size factor is 0.7 for two answers; this deliberately avoids high
confidence from a tiny baseline. Missing noise estimates reduce quality.
Reliability is a documented **heuristic**, not statistical calibration or a
probability of cheating. Configuration is in `speech_behavior/config.py` and
the existing `backend/config.py` speech weights.

## Storage and Member 5 handoff

`POST /api/speech-behavior/analyze` stores its structured result in the additive
SQLite `audio_analyses` table. `GET /api/audio/analyses/{analysis_id}` retrieves it.
Raw uploads/decoded temporary audio are deleted after the request. Downloaded
models, generated fixtures, runtime databases and recordings stay out of Git.

The response contains `transcription`, typed per-answer `answers`,
`speech_behavior` (baseline IDs, deviations, explanations, persistence count),
and `analysis_id`. `speech_behavior.result` is the existing `ModuleResult`
contract. Member 5 can also pass `result.answers` as `record.speech_features`:
`opensmile_adapter.compute()` recognizes the new typed measurements. Legacy
synthetic demo features remain supported and are explicitly separate.

Optional `interview_id` on upload links the analysis to an existing interview.
Use that interview's question IDs: the API stores aligned transcripts in its
answers, and `/api/interviews/{id}/report` passes the latest linked audio
measurements into the existing pipeline. Other scoring components are unchanged.
Response latency is not inferred without a question-end timestamp (Member 4).
The standalone browser audio workspace does not automatically link to an
interview; use the API for that integration.

## Verification

```bash
python -m pytest tests -q
cd frontend
node --test tests/audio-worklet.test.mjs
npm run build
npm run lint
```

Real model tests require a synthetic mono 16 kHz spoken fixture and a cached
model; the normal suite skips them explicitly if the fixture is not supplied:

```bash
AUDIO_TEST_WAV=/absolute/path/synthetic-speech.wav python -m pytest tests/test_audio_real.py -q
python scripts/generate_voice_fixtures.py --source /absolute/path/synthetic-speech.wav
```

The real tests exercise file upload → FFmpeg → Whisper → openSMILE → scoring →
SQLite retrieval, and PCM WebSocket → provisional revisions → final transcript
→ persisted session. Worklet tests verify byte order, clipping and final-packet
flush. Extraction tests caught and guard against stereo downmix gain changes.
Tests also cover clean/noisy/clipped/short/silent audio, F0 unit conversion,
baseline leakage, stable speech, measured changes and existing nervous-genuine
demo behavior. See `data/voice_behavior/README.md` for controlled fixture limits.

Current local real-inference checks emitted NumPy/Whisper matrix-operation
warnings; inspected model features were finite and transcription checks passed.
Browser microphone permission/hardware behavior requires a manual check: no
controllable browser was available during implementation.

## References and limits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): word timestamps,
  Silero VAD, CPU inference; model downloads are required once.
- [openSMILE Python usage](https://audeering.github.io/opensmile-python/usage.html):
  eGeMAPSv02 and LLDs.
- [openSMILE license](https://github.com/audeering/opensmile-python#license):
  research/educational restrictions; review commercial licensing before production.
- [FFmpeg](https://ffmpeg.org/ffmpeg.html): media extraction.

Voice behavior changes do not establish dishonesty, copying, psychological
state, or hiring eligibility. No emotion classifier, gender-specific pitch
thresholds, identity inference, or automatic hiring decision is implemented.
Weights/thresholds need project-specific calibration. Multi-speaker separation,
question detection, native continuous streaming, learned anomaly models and
change-point research remain future work.
