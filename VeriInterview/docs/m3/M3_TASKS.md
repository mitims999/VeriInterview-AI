# M3 — Deepfake Vision: Plan & Tasks

**Owner:** Member 3 · **Branch:** `deepfake_feature3`
**Features:** F3a Lip-sync (SyncNet) · F3c Eye tracking (Proctoring-AI) + Face identity (InsightFace)
**Deliverable:** real video → 3 upgraded adapters → `deepfake_fusion.py` → one **Deepfake Sub-Score**

Progress is tracked in [`M3_PROGRESS.md`](M3_PROGRESS.md). Data sources for every task: [`M3_DATA.md`](M3_DATA.md).

---

## 1. Where we are today (baseline audit)

| File | Current state | Problem |
|---|---|---|
| `backend/adapters/syncnet_adapter.py` | `compute(av_signals)` maps a fixture `{av_offset_frames, sync_confidence}` to risk | No video is read. Treats SyncNet confidence as 0–1, but real SyncNet confidence (LSE-C) is unbounded, about 7–10 on real speech. |
| `backend/adapters/insightface_adapter.py` | `compute(identity_signals)` maps `{min_frame_similarity_to_anchor, face_lost_pct}` with `drift*400` | No embeddings. The `drift*400` mapping assumes same-person cosine similarity is about 0.95, but real ArcFace (buffalo_l) same-person similarity across poses is about 0.5–0.8, so real input would always come out HIGH. |
| `backend/adapters/proctoring_adapter.py` | `compute(behavior_signals)` maps gaze %, head variance, person/phone flags | No frames are read and there is no per-question breakdown. Its weak-vs-strong evidence rules are correct and must be kept. |
| `backend/adapters/video_intake.py` | Derives **fake** signals from a SHA-256 of the uploaded bytes | Must be replaced. A placeholder like this goes against Master Spec §27 ("do not fabricate evidence"). |
| `backend/modules/deepfake_fusion.py` | Does not exist | — |
| `main.py /api/analyze` | `behavior_signals=None` always; video goes through the hash path | Needs the real video analyzer. |

**Things that must not break.** `tests/test_unit_modules.py` (4 M3 tests), `tests/test_e2e_demo_scenarios.py` (the deepfake candidate must stay identity HIGH and sync HIGH; the nervous-genuine candidate must not reach HIGH overall), `tests/test_integration_pipeline.py` (missing video gives `INSUFFICIENT_EVIDENCE`), the 5 synthetic fixtures, and the frontend keys `identity_risk / sync_risk / behavior_risk` in `ComponentBars.tsx`.

**Environment facts.** Python 3.13.5 · NVIDIA MX450 (2 GB VRAM) · **no system ffmpeg**.

**Data policy (owner decision, 2026-10-12):** the local `data/samples/` videos are **not** used. All calibration and validation uses **public online datasets** plus controlled manipulations generated from them. See [`M3_DATA.md`](M3_DATA.md) for the per-task data map, licenses and download links. Public datasets contain real faces, so they stay in gitignored `datasets/` and never enter the repo or the demo.

---

## 2. Rules from the specs that M3 must follow

1. Every adapter returns `ModuleResult(score, confidence, level, evidence[], quality, status)` and **never** a verdict (Build Spec §7.3, Master Spec §23).
2. **No single signal decides anything.** Gaze or head-pose alone must never reach HIGH (Build Spec §3.3). A nervous-genuine candidate must stay below HIGH overall (Master Spec §25/45).
3. `multi_person_detected` and `phone_detected` are **strong** evidence (Build Spec §3.3).
4. Large or unstable SyncNet offset or low confidence raises sync risk, but this must **never** be reported as "deepfake confirmed" (Build Spec §3.2).
5. Missing data never becomes risk. No face, no audio, a too-short clip or bad lighting gives `INSUFFICIENT_EVIDENCE` / `UNAVAILABLE` and lower confidence (Master Spec §28, §48).
6. All thresholds and weights live in `backend/config.py` (Master Spec §15, §48).
7. Every evidence item must be traceable to a module, a measurement, a `question_id` and a `timestamp` where possible (Master Spec §48).
8. Repos go behind adapters, the original UIs are not exposed, and attribution and licenses are kept (Build Spec §7.1–2, Master Spec §42).
9. Privacy: synthetic data only, no frames written to disk beyond temp processing, no media in logs (Master Spec §29).

---

## 3. Reference repos researched (what we take from each)

| Repo | License | What we reuse | What we change |
|---|---|---|---|
| [joonson/syncnet_python](https://github.com/joonson/syncnet_python) | MIT (code). Weights come from VGG Oxford. | `SyncNetModel.py`, and `SyncNetInstance.evaluate()`: MFCC via `python_speech_features`, ±`vshift` distance sweep, `offset = vshift − argmin(mean_dist)`, `conf = median(mean_dist) − min(mean_dist)`, framewise conf with a median filter (k=9). From `run_pipeline.py` we reuse the crop logic (`crop_scale=0.40`, median-smoothed boxes with k=13, 224×224 output, 25 fps, 16 kHz mono). Weights: `syncnet_v2.model`. | We replace the S3FD detector and PySceneDetect with the **shared InsightFace detector** (one face detector for all of M3). We call the model in-process instead of through the CLI scripts. We evaluate **per answer segment** instead of per whole video. |
| [deepinsight/insightface](https://github.com/deepinsight/insightface) (`insightface` 2.1 on PyPI, pure-python wheel, Python ≥3.10, pulls in `onnxruntime`) | MIT (code). **buffalo_l weights are non-commercial research only.** | `FaceAnalysis(name="buffalo_l")`, giving `bbox`, `kps`, `det_score`, `normed_embedding`, `pose`. SCRFD detection plus ArcFace 512-d embeddings. | Our own logic: an enrollment anchor, robust similarity statistics, identity-switch clustering and quality gating. |
| [vardanagarwal/Proctoring-AI](https://github.com/vardanagarwal/Proctoring-AI) | MIT. Its landmark model was trained on non-commercial data. | **Algorithms ported, not run as-is.** Eye tracker: pupil-vs-eye-corner `x_ratio > 3` / `< 0.33` gives right/left and `y_ratio < 0.33` gives up, and both eyes must agree. Head pose: `solvePnP` on the 6-point 3D model (nose, chin, eye corners, mouth corners), focal length = image width, ±48° cutoffs. Mouth-opening baseline-vs-deviation. `person_and_phone.py`: YOLOv3 COCO **class 0 = person, 67 = cell phone**, score ≥ 0.6, NMS IoU 0.5. | Its scripts need TensorFlow plus dlib-style 68-point landmarks plus a live webcam loop, and none of that installs cleanly on Python 3.13/Windows. We swap in **MediaPipe Face Landmarker** (478 points including iris, Apache-2.0, Python 3.13 wheels) and run YOLOv3 or YOLOv3-tiny through **OpenCV `cv2.dnn`** (same Darknet weights, no TF). We also compute true Euler angles instead of the repo's 2-D slope approximation. |
| Wav2Lip / LSE-C/LSE-D literature (arXiv 2008.10010, 2109.08061, 2405.04327) | — | Calibration anchor: real LRS2 video scores about **LSE-C 7–8, LSE-D 6.5–6.8** with SyncNet. | Used only to set the starting confidence normalisation. Final thresholds come from our own calibration (task V-3). |
| [AaronComo/LipFD](https://github.com/AaronComo/LipFD) (NeurIPS'24 "Lips Are Lying") | see repo | Reference only: it documents that SyncNet-trained generators (Wav2Lip-style) can **fool SyncNet**. | It informs the Limitations section. Possible stretch goal; not in MVP. |

> **Correction to make in `docs/SECURITY.md` / `MODELS.md`:** both currently say syncnet_python is non-commercial. The repo `LICENSE.md` is **MIT**. The weights are hosted by VGG Oxford, and their terms still need to be checked separately.

---

## 4. Target architecture

```
Uploaded video (mp4/webm)                         Question time segments
        │                                         (from M1 Whisper / transcript;
        ▼                                          fallback = whole video)
backend/vision/media.py  ── decode once (imageio-ffmpeg), 16k mono WAV, frame sampler
        │
backend/vision/face_track.py ── InsightFace SCRFD detect + IoU track + primary-face pick
        │            (shared by all three analyzers; detections cached per frame)
        ├──────────────────────┬────────────────────────────┐
        ▼                      ▼                            ▼
vision/syncnet_runner.py  vision/identity.py          vision/gaze.py (+head_pose, blink,
 25fps face crops +        ArcFace embeddings,         mouth, objects.py YOLO person/phone)
 MFCC → SyncNet            anchor, drift, clusters     MediaPipe Face Landmarker
        │                      │                            │
        ▼                      ▼                            ▼
  av_signals{}           identity_signals{}           behavior_signals{}  ◄─ same dict shapes
        │                      │                            │              as today + new
        ▼                      ▼                            ▼              optional keys
adapters/syncnet_adapter  adapters/insightface_adapter  adapters/proctoring_adapter
  .compute() → ModuleResult  .compute() → ModuleResult    .compute() → ModuleResult
        │                      │                            │
        └──────────────┬───────┘                            │
                       ▼                                    │ (authenticity-only cues:
          modules/deepfake_fusion.py ◄──────────────────────┘  blink, mouth↔audio,
                       │                                       landmark jitter)
                       ▼
             ModuleResult(feature="deepfake")  → handed to M5's pipeline / evidence_fusion
```

### Key design decisions (please review)

- **D1. Keep the `compute(signals_dict)` contract and add `analyze_video(...)`.** Each adapter gets a real `analyze_video(path, segments) -> signals_dict`, and the existing `compute()` keeps working on that dict. Fixtures, tests, the demo dropdown and M5's pipeline keep working unchanged. Real video and fixtures go through the same scoring code. ✅ **Confirmed by owner.**
- **D2. One shared face detector.** InsightFace SCRFD replaces SyncNet's S3FD and Proctoring-AI's OpenCV DNN detector, so we download one detector instead of three. MediaPipe is used only for dense landmarks on the face that is already tracked. ✅ **Confirmed by owner.**
- **D3. Gaze-off-screen is behaviour evidence, not deepfake evidence.** The Deepfake Sub-Score fuses **identity + lip-sync + eye-region *authenticity* cues**: blink-rate naturalness, mouth movement vs speech activity, and landmark jitter. "Looking away" stays in `behavior_risk`. This avoids counting the same thing twice in M5's fusion, and it keeps nervous-genuine candidates from looking like deepfakes. ✅ **Confirmed by owner, with requirement R1 below.**
- **D8. Port Proctoring-AI's algorithms instead of running its scripts.** Its scripts need TensorFlow + dlib, which don't install cleanly on Python 3.13 / Windows. We re-implement the eye-ratio rule, the solvePnP head pose and the YOLOv3 person/phone logic on MediaPipe + OpenCV `cv2.dnn`, keeping the original thresholds and attribution. ✅ **Confirmed by owner.**
- **D4. `video_intake.py`'s hash placeholder is deleted.** If models or weights are missing, the adapters return `UNAVAILABLE` with an explanatory evidence line. They never make up numbers. ✅ **Confirmed by owner.**
- **D5. Re-calibrated thresholds in config.** New `SyncThresholds`, `IdentityThresholds`, `GazeThresholds`, `DeepfakeWeights` in `config.py` (additive only). The fixture values in `data/synthetic/*.json` remain valid inputs, and we check that the e2e expectations still hold. If a fixture needs re-tuning, that goes through `scripts/generate_fixtures.py` after a heads-up to M5. ✅ **Confirmed by owner.**
- **D6. Cache by video hash.** Derived features are written to `storage/features/<sha256>.json` (no frames are stored), so re-running a report on the same video costs nothing. Video analysis runs in a worker thread. A background-job endpoint is a stretch goal. ✅ **Confirmed by owner.**
- **D7. Model weights are never committed.** They go in `storage/models/` (gitignored) and are fetched by `scripts/download_models.py`. ✅ **Confirmed by owner.**

### R1 — Hard requirement: tell nervous candidates apart from fake ones (owner, 2026-10-12)

A nervous but genuine candidate must get a **LOW Deepfake Sub-Score**. A manipulated stream must get a HIGH one **whether or not** the person also looks nervous. Research shows that stress **raises blink rate** and **increases gaze aversion** (PMC12649463, vTSST gaze-cortisol study, arXiv 2202.03033), so the naive cues point the wrong way. Design rules:

1. **No nervousness-correlated signal feeds the deepfake score.** Gaze aversion, head movement, fidgeting, speech rate, pauses and *high* blink rate are excluded. They stay in `behavior_risk` / `speech_behavior_risk`, where the weak-evidence caps already apply.
2. **The blink cue is one-sided.** It flags only *absent or too-rare* blinking (configurable, starting value < 2 blinks/min over ≥ 60 s of visible eyes) or *metronomic* blinking (inter-blink-interval coefficient of variation below a threshold), both known synthesis artifacts. A **high** blink rate contributes **0** deepfake risk and is reported as a possible stress indicator.
3. **Mouth-vs-speech is measured only while speech is present.** It flags sustained (≥ 1 s) windows where the audio contains speech but the mouth shows no articulation, or the reverse. Lip-biting or a nervous mouth when not speaking is ignored.
4. **Head motion lowers confidence, never raises risk.** Fast or fidgety movement degrades SyncNet and landmark quality, so it reduces the *confidence* of the sync and identity inputs instead of adding risk. Identity frames are quality-gated (pose, blur), so looking away cannot cause "identity drift".
5. **Rigid motion vs non-rigid motion.** The optional landmark-jitter cue measures non-rigid residuals *after* rigid head-pose alignment (Procrustes), because nervous head motion is rigid while face-swap boundary jitter is not.
6. **Per-candidate baselines** for every eye cue (first answer or enrollment window), never population norms.
7. **Explicit explanation.** When nervousness indicators are elevated and the authenticity signals are normal, the deepfake result carries an evidence line such as *"Elevated blink rate / gaze aversion are consistent with nervousness; identity, lip-sync and articulation signals are normal"*. M5's XAI layer can then show it as is.

Verified by **F-4** (separation tests on synthetic profiles + real stressed and real fake videos).

### Open questions for other members

| # | With | Question | My proposed default |
|---|---|---|---|
| Q1 | M5 | How does the Deepfake Sub-Score enter final fusion? (a) It replaces `identity_risk` + `sync_risk` (combined weight 0.35) with one `deepfake_risk` component, and the two remain visible as sub-components. (b) It is a report-only summary card and the 6 components stay as they are. | **(b) for now**, because it changes nothing in M5's weights. Switching to (a) later is a one-line change. |
| Q2 | M5 | Add an optional `details: dict \| None = None` to `ModuleResult` (backward compatible) so `deepfake` can carry `sub_scores` and per-question stats for XAI? | Yes. This is additive, and I will open the PR touching `schemas.py`. |
| Q3 | M1 | Will `audio_extractor.py` expose `extract_wav(video_path, sr=16000) -> path`? And will Whisper segments give `[{question_id, start, end}]`? | I'll write `vision/media.py` with the same signature and switch to M1's once it lands. Without segments, M3 falls back to whole-video analysis. |
| Q4 | M1 | Speech-activity (VAD) timeline. Needed for the "mouth moving vs speech" cue and to skip silent stretches in SyncNet. | Use Whisper word timestamps if available. Otherwise use a simple RMS-energy VAD in `media.py`. |
| Q5 | M5 | Does the `/api/analyze` request stay synchronous? Video analysis could take 30–120 s on CPU. | Keep it synchronous, with caching plus a frame budget (task I-4). A job endpoint is a stretch goal. |

---

## 5. Signal contracts (backward compatible — new keys are optional)

```jsonc
// av_signals  (SyncNet)
{
  "av_offset_frames": 1,            // EXISTING — median offset across speech segments
  "sync_confidence": 0.86,          // EXISTING — normalised 0-1 (from LSE-C via config)
  "lse_c": 7.4, "lse_d": 6.9,       // NEW raw SyncNet confidence / min distance
  "offset_stability": 0.9,          // NEW fraction of segments agreeing with median offset (±1)
  "speech_seconds_analyzed": 84.0,  // NEW
  "per_segment": [{"question_id":"Q1","start":3.2,"end":41.0,"offset":1,"lse_c":7.9}],
  "status_reason": null             // NEW e.g. "no_face", "no_speech", "too_short"
}
// identity_signals  (InsightFace)
{
  "min_frame_similarity_to_anchor": 0.62, // EXISTING — robust min (p5), not raw min
  "face_lost_pct": 0.03,                  // EXISTING
  "mean_similarity": 0.74, "below_threshold_pct": 0.01,
  "distinct_identities": 1,               // clusters with ≥ N seconds of presence
  "multi_face_pct": 0.0, "anchor_quality": 0.92,
  "drift_events": [{"t":312.5,"question_id":"Q4","similarity":0.21}]
}
// behavior_signals  (Proctoring-AI port)
{
  "gaze_off_screen_pct": 0.12,            // EXISTING
  "head_pose_variance": "normal",         // EXISTING (derived from yaw/pitch std vs thresholds)
  "multi_person_detected": false,         // EXISTING (YOLO person≥2 persisting ≥2 samples)
  "phone_detected": false,                // EXISTING (YOLO class 67 persisting ≥2 samples)
  "yaw_std": 6.1, "pitch_std": 4.3,
  "per_question": [{"question_id":"Q3","gaze_off_pct":0.41,"dominant_direction":"left"}],
  "baseline_gaze_off_pct": 0.08,          // candidate's own baseline (first answer)
  "blink_rate_per_min": 14.2, "blink_regularity": 0.3,   // authenticity cues → deepfake_fusion
  "mouth_speech_mismatch_pct": 0.04                       // mouth open-ness vs VAD
}
```

---

## 6. Tasks

Effort: S ≤ 2 h · M ≈ half a day · L ≈ 1 day. Tasks within a phase can run in parallel unless a "Depends" entry is listed.

### Phase 0 — Groundwork

| ID | Task | Files | Acceptance criteria | Effort |
|---|---|---|---|---|
| G-0 | **Acquire public data** per [`M3_DATA.md`](M3_DATA.md): download Tier A (GRID 4–6 talkers, LFW, DeepfakeTIMIT, OEP, RT-BENE subset, COCO person/phone subset, AVLips val), submit Tier B forms, add `datasets/` to `.gitignore`, and write `scripts/fetch_datasets.py` for the scriptable ones. | `scripts/fetch_datasets.py`, `.gitignore` | Tier A on disk, with a dataset inventory table in PROGRESS. Tier B requests submitted. | M |
| G-1 | Add deps: `opencv-python`, `mediapipe`, `insightface`, `onnxruntime`, `torch` (CPU wheel; optional CUDA), `python_speech_features`, `imageio-ffmpeg`. Use a separate `requirements-vision.txt` so the core app still installs light. | `requirements-vision.txt`, `README.md` | `pip install -r requirements-vision.txt` works on Python 3.13/Windows. The core tests still pass without it. | M |
| G-2 | `scripts/download_models.py`: syncnet_v2.model, buffalo_l (via insightface auto-download into `storage/models`), MediaPipe `face_landmarker.task`, YOLOv3-tiny cfg/weights (full YOLOv3 optional via flag). Verify checksums. Gitignore `storage/models/`, `storage/features/`. | `scripts/`, `.gitignore` | One command fetches everything. Re-running it is a no-op. | M |
| G-3 | `backend/vision/media.py`: probe the video, decode frames at a target fps (generator, bounded memory), extract 16 kHz mono WAV, RMS-energy VAD, map a timestamp to a question segment. Uses the imageio-ffmpeg binary. | `backend/vision/media.py` | Works on GRID `.mpg`, DeepfakeTIMIT/OEP `.avi` and FakeAVCeleb `.mp4`. Handles a corrupt or no-audio file by raising typed errors (`NoAudio`, `DecodeError`). | M |
| G-4 | Add M3 thresholds and weights to `config.py`: `SyncThresholds`, `IdentityThresholds`, `GazeThresholds`, `DeepfakeWeights`, `VisionBudget` (max seconds analysed, fps per analyzer). | `backend/config.py` | Additive only. Existing tests stay green. | S |
| G-5 | `backend/vision/face_track.py`: InsightFace `FaceAnalysis` singleton (lazy load, CPU provider by default), per-frame detect, IoU tracker (port of syncnet `track_shot`: IoU > 0.5, `num_failed_det`), primary-face selection (largest and longest track), quality gating (det_score, face size, Laplacian blur, |yaw| ≤ 45°). | `backend/vision/face_track.py` | Returns tracks with bbox, kps, embedding, quality per sampled frame. Missing weights raise `ModelUnavailable`. | L |

### Phase 1 — F3c Face identity (InsightFace) — *Depends: G-3, G-5*

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| I-1 | `vision/identity.py`: sample at 2 fps. Build the **enrollment anchor** as the median normed embedding of the first 10 s of good-quality frames (fall back to the best 15 frames anywhere). Compute the cosine-similarity trace, p5/mean, `below_threshold_pct` (ArcFace same-person threshold from config, starting value 0.35), and `face_lost_pct`. | Gives `identity_signals` in the §5 shape for GRID and DeepfakeTIMIT clips. Same-person threshold calibrated on **LFW pairs** (genuine vs impostor cosine distributions, documented FAR). | M |
| I-2 | **Identity-switch detection.** Agglomerative clustering on embeddings (cosine, threshold from config). Report `distinct_identities` as clusters with ≥ 3 s presence, plus `drift_events` with timestamps and question ids. | A synthetic splice of two different people gives `distinct_identities=2` and a drift event near the cut. | M |
| I-3 | Upgrade `insightface_adapter.py`. Add `analyze_video()`. Recalibrate `compute()` to use `below_threshold_pct` and `distinct_identities` when present, and keep the legacy `min_frame_similarity_to_anchor` path for fixtures. Evidence lines carry timestamps and question ids. A bad anchor gives low confidence, not risk. | The existing unit test and the e2e deepfake identity-HIGH test pass. Genuine GRID / VidTIMIT give LOW. DeepfakeTIMIT face-swaps and GRID identity splices give HIGH. Missing face gives `INSUFFICIENT_EVIDENCE`. | M |
| I-4 | **Reference photo match (promoted to MVP by owner, 2026-10-14).** Optional photo of the registered candidate (resume/ID) is embedded with ArcFace and compared with the session anchor and every frame. Below the LFW-calibrated threshold → `reference_mismatch` (risk ≥ 70, HIGH). Within the margin → `reference_weak_match` (MEDIUM). Unusable photo → note only, no risk. This is the only identity check that catches a face that is consistent all session but is not the candidate (proxy or full-session face-swap). Found in I-1 testing: DeepfakeTIMIT fakes are self-consistent (0.965). The photo is processed in memory and never stored. | DeepfakeTIMIT fake (fram1 face on fadg0) vs a real fadg0 reference photo → mismatch. Genuine reference → match. | S |

### Phase 2 — F3a Lip-sync (SyncNet) — *Depends: G-3, G-5*

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| S-1 | Vendor `SyncNetModel.py` plus a trimmed `SyncNetInstance.evaluate()` into `backend/vision/syncnet/`, with the MIT license header and attribution. Remove the ffmpeg/tempdir calls and take numpy frames and audio directly. Load with `weights_only=True`. Use CUDA if available, else CPU. | Reproduces the repo's README reference on `example.avi` (offset 3, conf ≈ 10.0, within tolerance). | M |
| S-2 | `vision/syncnet_runner.py`: build 25 fps 224×224 face crops for the primary track (port of `crop_video`: crop_scale 0.40, median-smoothed boxes with k=13). Run SyncNet **per speech segment** (VAD- or question-bounded, ≥ 2 s), and aggregate: median offset, offset stability, LSE-C/LSE-D, per-segment table. Skip segments with no face or no speech. | Gives `av_signals` in the §5 shape. Genuine GRID clips land in the "bonafide" range. | L |
| S-3 | **Calibration harness** `scripts/build_m3_testset.py` + `scripts/calibrate_syncnet.py`: from public **GRID** clips, generate the controlled set in [`M3_DATA.md` §4](M3_DATA.md) (audio shifted ±2/±5/±10 frames, partial shift, audio swap, identity splice, silence, occlusion, low quality). Also run on **AVLips** val and, when approved, **FakeAVCeleb** to measure what SyncNet misses. Record LSE-C and offset for each and write a CSV. Use it to set `SyncThresholds` (LSE-C→0–1 normalisation, offset bands). | Shifted copies are recovered within ±1 frame. Swapped audio gives low LSE-C. Thresholds are committed with a short rationale in PROGRESS. | M |
| S-4 | Upgrade `syncnet_adapter.py`. Add `analyze_video()`. `compute()` scores **offset magnitude + offset instability + low confidence while speech and face are present**. Low confidence with no speech means `INSUFFICIENT_EVIDENCE`, not risk. Keep the legacy fixture path. Evidence never says "deepfake". | The existing unit test and the e2e sync-HIGH test pass. A shifted-by-10 clip gives HIGH, genuine gives LOW. | M |

### Phase 3 — F3c Eye tracking / proctoring (Proctoring-AI port) — *Depends: G-3, G-5*

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| P-1 | `vision/landmarks.py`: MediaPipe **Tasks** `FaceLandmarker` wrapper (VIDEO mode, 478 points + iris + blendshapes + transformation matrix), run on the tracked face ROI. | Landmarks for ≥ 90 % of good frames on GRID and OEP webcam videos. | M |
| P-2 | `vision/head_pose.py`: port Proctoring-AI's 6-point `solvePnP` (MediaPipe indices: nose 1, chin 152, eye corners 33/263, mouth corners 61/291), then convert to proper yaw/pitch/roll via `Rodrigues` + `RQDecomp3x3`. Cross-check against the MediaPipe transformation matrix. | Yaw/pitch within ~10° of the MediaPipe matrix (and of BIWI / AFLW2000-3D ground truth if downloaded). | S |
| P-3 | `vision/gaze.py`: port `find_eyeball_position` onto iris centres (468/473) relative to eye corners and lids, with the same ratio rule (`x>3` / `x<0.33` / `y<0.33`, thresholds in config) and both eyes required to agree. Combine with head pose to get an "off-screen" state. **Debounce:** an off-screen period counts only if it lasts ≥ 0.5 s. Output per-question `gaze_off_pct` and dominant direction, plus the candidate's **own baseline** (first answer). | GRID (camera-facing) gives < 10 % off-screen. OEP cheating intervals show higher off-screen than non-cheating intervals (report precision/recall). Values are reported per question. | M |
| P-4 | `vision/blink.py` + mouth: blink rate from the eye-aspect ratio or MediaPipe `eyeBlink*` blendshapes, and blink-interval regularity. Mouth-opening ratio vs the VAD timeline gives `mouth_speech_mismatch_pct` (port of `mouth_opening_detector`'s baseline-vs-deviation idea). | Blink classifier validated on **RT-BENE** labels. Plausible blink rates (about 8–30/min) on GRID/OEP. Mouth-vs-speech checked against GRID word alignments. These are authenticity cues for `deepfake_fusion` only. | M |
| P-5 | `vision/objects.py`: YOLOv3(-tiny) through `cv2.dnn.readNetFromDarknet`, at 1 fps. COCO class 0 = person and 67 = phone, score ≥ 0.6, NMS 0.5 (Proctoring-AI values). A flag fires only if it **persists ≥ 2 consecutive samples**. Also count multi-face from the InsightFace detections. | Precision/recall for `person` / `cell phone` on the **COCO val2017** subset at the chosen thresholds. No false flags on GRID. Evidence includes timestamps. | M |
| P-6 | Upgrade `proctoring_adapter.py`. Add `analyze_video()`. `compute()` uses **deviation from the candidate's own baseline** when `per_question` and baseline are present (protects nervous-genuine candidates), keeps the legacy path, keeps the weak-evidence cap (≤ 55) and the strong-evidence floors (multi-person 80, phone 85), and adds per-question evidence. | All 4 existing proctoring/e2e assertions pass. Gaze-only input never exceeds MEDIUM. | M |

### Phase 4 — Deepfake fusion — *Depends: I-3, S-4, P-4*

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| F-1 | `modules/deepfake_fusion.py` → `fuse_deepfake(identity: ModuleResult, sync: ModuleResult, eye_authenticity: dict \| None) -> ModuleResult(feature="deepfake")`. Weighted (config `DeepfakeWeights`; starting values identity 0.45 / sync 0.40 / eye-authenticity 0.15). Unavailable inputs get their weight redistributed and lower confidence. **Corroboration rule:** the sub-score reaches HIGH only if ≥ 2 inputs are ≥ MEDIUM, **or** there is one strong signal (`distinct_identities ≥ 2` with high confidence). Otherwise it is capped at MEDIUM. A small boost applies when identity and sync are both HIGH (mirrors `evidence_fusion`). Gaze-off-screen is **not** an input (D3). | Deterministic and documented. Returns `sub_scores` (in `details` if Q2 is accepted, otherwise as evidence lines). | M |
| F-2 | Evidence and XAI hand-off: each evidence line names its sub-signal, measurement, question id and timestamp. Publish an **evidence-type catalog** (`identity_drift`, `identity_switch`, `face_lost`, `av_offset`, `av_offset_unstable`, `sync_confidence`, `blink_rate`, `mouth_speech_mismatch`, `gaze`, `head_pose`, `multi_person`, `phone`) so M5's `xai_explainer.py` can write plain-English text. | The catalog is in `docs/m3/EVIDENCE_TYPES.md`. | S |
| F-4 | **Nervous-vs-fake separation (R1).** (a) Synthetic profiles: *nervous-genuine* (gaze-off 35 %, head variance high, blink 35/min, long pauses, identity and sync clean) → deepfake **LOW**. *Fake-calm* (identity drift + desync + < 2 blinks/min, gaze normal) → **HIGH**. *Fake-nervous* (both) → **HIGH**. (b) Real stressed people: **OEP** subjects (exam pressure, both cheating and non-cheating intervals) → deepfake LOW for every subject. **UBFC-Phys** stress tasks T2/T3 vs rest T1 (when access is granted) → no significant shift in the deepfake score (paired test). (c) Real fakes: DeepfakeTIMIT, AVLips val, the generated GRID set → MEDIUM/HIGH. Report AUC (stressed-real vs fake) and the stress shift Δ in PROGRESS. | `tests/test_nervous_vs_fake.py` (synthetic, always runs) plus `scripts/eval_nervous_vs_fake.py` (dataset-backed). Target: 0 HIGH on OEP / UBFC-Phys, and AUC ≥ 0.9 on the stressed-real vs fake split. | M |
| F-3 | Unit tests: genuine (all LOW) gives LOW. Identity-only HIGH gives ≤ MEDIUM. Identity + sync HIGH gives HIGH. Everything unavailable gives `UNAVAILABLE`. Identity switch alone with high confidence gives HIGH. Eye cues alone never go above MEDIUM. A high blink rate adds 0 risk. Head motion only lowers confidence. | `tests/test_deepfake_fusion.py` is green. | S |

### Phase 5 — Integration — *Depends: Phase 1–4; coordinate with M5*

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| N-1 | Rewrite `video_intake.py` as `analyze_video(path, segments=None) -> {av_signals, identity_signals, behavior_signals, eye_authenticity, meta}`. Decode once, share face tracks, use the cache by sha256 (D6) and the frame budget (`VisionBudget`). Delete the hash placeholder. A missing model gives the matching signal as `None` plus a reason. | One call analyses one public test clip end to end. A second call is served from the cache. | M |
| N-2 | `main.py /api/analyze`: save the upload to a temp file, call `analyze_video`, pass the real `behavior_signals` (not `None` anymore), and update `video_analysis_note`. Accept an **optional `reference_photo` upload** (I-4) and pass it to `insightface_adapter.analyze_video`. Clean up the temp file. Frontend: optional photo field in `UploadForm.tsx` (coordinate with M5). | Uploading a public test clip (e.g. generated `genuine_long`) in the UI shows real identity, sync and gaze cards. | S (touches M5-owned flow, so it goes through a PR) |
| N-3 | Hand `deepfake_fusion` to M5 (pipeline call plus report field `deepfake_assessment`) per the Q1 decision. Optional small frontend card "Media authenticity" with the 3 sub-scores. | The e2e tests still pass. The deepfake fixture shows sub-score HIGH and the nervous-genuine fixture shows LOW. | S |
| N-4 | Performance pass: make sure a 3-minute video finishes in ≤ ~90 s on CPU (MX450 CUDA optional for SyncNet). Tune the per-analyzer fps. | Timings recorded in PROGRESS for `genuine_long` (~3 min GRID) and an OEP video. | M |

### Phase 6 — Validation, tests, docs

| ID | Task | Acceptance criteria | Effort |
|---|---|---|---|
| T-1 | Model-free unit tests for gaze ratio logic, head-pose maths, blink counting, the tracker, the identity clustering on synthetic embeddings, and the adapter `compute()` paths. | They run without weights in CI and pass. | M |
| T-2 | Model-backed tests marked `@pytest.mark.vision` (skipped if weights are missing): run over a committed **manifest** (paths + expected outcomes, no media in git) of the generated GRID set + DeepfakeTIMIT: genuine gives identity LOW and sync LOW, shift-10 gives sync HIGH, splice gives identity HIGH, splice+shift gives fusion HIGH. Skipped if `datasets/` is absent. | Green locally with weights. | M |
| T-3 | Make sure all existing tests (20) still pass, and the 5 demo candidates keep their documented outcomes. | `pytest tests/ -v` is green. | S |
| T-4 | Docs: `MODELS.md` (real status, weights, licenses, fix the SyncNet license line), `LIMITATIONS.md` (SyncNet can be fooled by SyncNet-trained generators per LipFD; webcam gaze ≠ eye-tracker accuracy; MX450 timings), `DATASETS.md` (public calibration datasets, licenses, privacy note), `docs/m3/EVIDENCE_TYPES.md`. | Docs match the code. | S |
| T-5 | **Dataset deletion gate (mandatory).** Once every box in the *Safe-to-delete gate* in [`M3_PROGRESS.md`](M3_PROGRESS.md) is ticked (results committed, everything re-creatable, M5 integration done, tests pass with and without data, weights kept), delete `VeriInterview/datasets/` (~29 GB of real faces and voices) and log the date. | `datasets/` is gone, `storage/models/` is kept, the app and tests still work, and the deletion is logged. | S |

### Stretch (only after the MVP is done)

- **X-1** Live mode: per-frame gaze and identity over the M1 WebSocket (≈ 2 fps), with incremental `ModuleResult`s.
- **X-2** Lip-forgery detector as a 4th deepfake input. **Feasibility checked 2026-10-19:** LipFD has **no licence** (all rights reserved), so it cannot be used. **AVAD** (cfeng16/audio-visual-forensics, CVPR 2023) has **MIT code**, but its weights have unstated terms; the 41 M-param checkpoint loads safely with `weights_only`, and `pca.pkl` only references numpy/sklearn classes (restricted unpickler OK). Port needed: its 2022 stack (torch 1.9, librosa 0.9, torchaudio) does not install on Py 3.13 / torch 2.14 (torchaudio ends at 2.11), so reimplement the mel-spectrogram, replace its `shell=True` ffmpeg calls with `media.py`, and feed SyncNet face crops. No reference output exists to verify a port. **Deferred:** the main live-interview attack (real-time face-swap) is already covered via the reference photo. Code + weights are kept in gitignored `datasets/vendor_eval/`.
- **X-4** Background-job endpoint `POST /api/av/analyze` → `GET /api/av/jobs/{id}` (Master Spec §22).

---

## 7. Suggested order and critical path

```
G-0 → G-1 → G-2 → G-3 → G-5 ─┬─ I-1 → I-2 → I-3 ───────────┐
            G-4 ─────────────┤                              ├─ F-1 → F-2 → F-3 → F-4 → N-1 → N-2 → N-3 → N-4 → T-2 → T-3 → T-4 → T-5
                             ├─ S-1 → S-2 → S-3 → S-4 ──────┤
                             └─ P-1 → P-2 → P-3 → P-4 → P-5 → P-6 ┘
T-1 is written alongside each phase.
```

Estimated total: about 8–10 working days for one person. If time runs short, the priority is **I (identity) → S (lip-sync) → F (fusion) → N (integration) → P (gaze)**, because the deepfake demo candidate depends on identity and sync.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Heavy dependencies on Python 3.13/Windows (torch + onnxruntime + mediapipe) | Keep them in a separate `requirements-vision.txt`. Lazy imports. Every adapter degrades to `UNAVAILABLE`. Verify installs in G-1 before writing code. |
| 2 GB VRAM | CPU by default. CUDA only for the SyncNet forward pass with small batches. |
| No ffmpeg on the system | Use `imageio-ffmpeg` (bundles a static binary). |
| No labelled deepfake data | Generate controlled positives ourselves (audio shift, audio swap, identity splice), document thresholds as prototype values, and never claim detection accuracy. |
| Editing shared files (`schemas.py`, `config.py`, `main.py`, `pipeline.py`) | Additive changes only, via small PRs with M5 tagged. |
| SyncNet is fooled by Wav2Lip-style generators | State it in LIMITATIONS. Identity and eye cues give independent coverage. |
