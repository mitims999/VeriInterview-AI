# M3 — Deepfake Vision: Progress Tracker

Plan and task definitions: [`M3_TASKS.md`](M3_TASKS.md) · Data: [`M3_DATA.md`](M3_DATA.md) · Branch: `deepfake_feature3`

**Status legend:** ⬜ not started · 🟨 in progress · ✅ done · ⛔ blocked · ⏭️ skipped/deferred

**Current phase:** M3 build complete (Phases 0–6; T-5 gated on M5 merge). X-2 feasibility checked and deferred
**Last updated:** 2026-10-12

---

## Summary

| Phase | Tasks | Done | Status |
|---|---|---|---|
| 0 Groundwork | 6 | 5 | 🟨 |
| 1 Face identity (InsightFace) | 4 | 4 | ✅ |
| 2 Lip-sync (SyncNet) | 4 | 4 | ✅ |
| 3 Eye tracking / proctoring | 6 | 6 | ✅ |
| 4 Deepfake fusion | 4 | 4 | ✅ |
| 5 Integration | 4 | 4 | ✅ |
| 6 Validation & docs | 5 | 4 | 🟨 (T-5 gated on M5 merge) |

---

## Task checklist

### Phase 0 — Groundwork
- 🟨 G-0 Acquire public datasets: **all Tier A downloaded and verified** (~33 GB). Open only for Tier B request forms + UBFC-Phys login (owner)
- ✅ G-1 `requirements-vision.txt` installs on Py 3.13 / Windows (venv `.venv`, OpenCV pinned to 4.14, all imports verified, 20/20 tests pass)
- ✅ G-2 `scripts/download_models.py` + gitignore (6 files, ~384 MB, SHA-256 pinned, every model load-tested)
- ✅ G-3 `backend/vision/media.py` (decode, WAV, VAD, segment mapping): 8 unit tests; VAD validated on 600 GRID clips (acc 0.936, prec 0.947, 0 missed)
- ✅ G-4 M3 config blocks in `config.py` (8 blocks, additive; 5 sanity tests)
- ✅ G-5 `backend/vision/face_track.py` (shared InsightFace detector + tracker): det_size 320, optional embeddings/pose, relative blur gating; 8 tests

### Phase 1 — Face identity
- ✅ I-1 Enrollment anchor + similarity trace (LFW threshold 0.30: FAR 0 %, FRR 0.4 %)
- ✅ I-2 Identity-switch clustering + drift events (6/6 splices: 2 identities, drift at the exact cut)
- ✅ I-3 `insightface_adapter.py` upgrade: 24/24 generated clips as expected; legacy fixture path unchanged
- ✅ I-4 Reference photo match (**promoted to MVP**): DeepfakeTIMIT fake caught (0.223 < 0.30); genuine 0.97

### Phase 2 — Lip-sync
- ✅ S-1 Vendor SyncNet model + evaluate: **reproduces README exactly** (offset 3, min dist 5.353, conf 10.021); 4 tests
- ✅ S-2 `syncnet_runner.py` per-speech-window crops + aggregation: GRID shifts recovered exactly (+5 → −4, −10 → +11 vs baseline +1); 5 tests
- ✅ S-3 Calibration: 78 generated clips, 66 scored; shifts recovered ±1 frame; swap vs low-quality separated by offset stability (`calibration/syncnet_grid.md`)
- ✅ S-4 `syncnet_adapter.py` upgrade: 4 calibrated rules; all genuine/silent/240p LOW, all swaps + partial shifts HIGH; 8 tests

### Phase 3 — Eye tracking / proctoring
- ✅ P-1 MediaPipe FaceLandmarker wrapper: 100% landmark coverage on GRID/OEP/AVLips; 6 tests
- ✅ P-2 Head pose: Proctoring-AI solvePnP ported (exact on synthetic projections), but **MediaPipe matrix chosen as primary** (agrees with InsightFace; port has +8–17° yaw bias on real faces); 2 tests
- ✅ P-3 Iris gaze: OEP 24 subjects; normal off-screen median 2.2 % (max 20.7 %); cheating ↑ in 17/24; reading notes 25 %; 7 tests
- ✅ P-4 Blink + mouth-vs-speech (`eye_auth.py`): one-sided R1 rules; frozen face 24–27 % vs genuine 0 %; 7 tests
- ✅ P-5 Person/phone (`objects.py`): **full YOLOv3 default** at score 0.4 (phone P 0.98 / R 0.50, multi-person P 0.97 / R 0.84, 0 false phones); tiny too weak; 3 tests
- ✅ P-6 `proctoring_adapter.py` upgrade: baseline-relative gaze (weak, cap 55), phone/person strong; OEP gaze + phone checks done; 6 tests

### Phase 4 — Deepfake fusion
- ✅ F-1 `modules/deepfake_fusion.py`: weights 0.45/0.40/0.15, single-input floor 0.75, single-signal cap 60, strong-identity floor 75, corroboration boost; `details.sub_scores`
- ✅ F-2 Evidence-type catalog for M5 XAI: `docs/m3/EVIDENCE_TYPES.md` (27 types from the code, severity, trigger, time/question, confidence-only flags, fusion rules)
- ✅ F-3 `tests/test_deepfake_fusion.py`: 12 tests (golden rule, corroboration, strong identity, unavailable inputs, eye cap, no verdict wording + R1 profiles)
- ✅ F-4 Nervous-vs-fake: synthetic + real. **OEP stressed 24/24 LOW (0 HIGH)**; DeepfakeTIMIT 31/32 flagged (AUC 1.000); **AVLips lip-sync fakes 0/30 flagged (limitation → X-2 LipFD)**

### Phase 5 — Integration
- ✅ N-1 `video_intake.analyze_video`: one pass (speech → tracks → identity → landmarks → gaze + eye cues → objects → SyncNet), sha256 cache, hash placeholder removed; `derive_av_signals()` now real
- ✅ N-2 `/api/analyze` uses real video analysis (worker thread, temp file deleted, optional `reference_photo`, real behavior + eye signals) — **touches M5's main.py: PR + review**
- ✅ N-3 `deepfake_assessment` added to every report (report-only per Q1 default; 6 fusion weights untouched). Frontend card not done (M5/UI owner)
- ✅ N-4 Performance: 3-min video **158 s on AC (0.89× real time)**, was 205 s serial / ~10 min on battery. Target of ≤ 90 s not reached on CPU (GPU optional)

### Phase 6 — Validation & docs
- ✅ T-1 Model-free unit tests: every vision module covered (media, face_track, identity, syncnet, runner, landmarks, head pose, gaze, eye_auth, objects, all 3 adapters, fusion, API)
- ✅ T-2 Model-backed `vision` tests: `tests/test_vision_e2e.py` + committed manifest `tests/data/m3_vision_manifest.csv` (10 clip expectations + bad-media case); skip without datasets/weights
- ✅ T-3 Full suite **140/140** (129 fast + 11 vision); 5 demo candidates keep their documented outcomes
- ✅ T-4 Docs updated: MODELS, LIMITATIONS (measured M3 limits), SECURITY (SyncNet MIT fix, upload deletion, dataset privacy), DATASETS, ARCHITECTURE (M3 pipeline), DEMO_GUIDE, README, API
- ⬜ T-5 **Delete `datasets/`** once every box in the *Safe-to-delete gate* below is ticked (mandatory final step)

---

## Decisions log

| Date | ID | Decision | Status |
|---|---|---|---|
| 2026-10-12 | D0 | Don't use local `data/samples/`; public online datasets only, kept out of git | ✅ Owner decision |
| 2026-10-12 | D1 | Keep `compute(signals)` contract; add `analyze_video()` per adapter | ✅ Confirmed |
| 2026-10-12 | D2 | One shared face detector (InsightFace SCRFD) replaces S3FD + OpenCV DNN | ✅ Confirmed |
| 2026-10-12 | D8 | Port Proctoring-AI algorithms onto MediaPipe + OpenCV (no TF/dlib) | ✅ Confirmed |
| 2026-10-12 | D3 | Gaze-off-screen stays behavior evidence; deepfake sub-score uses identity + sync + eye *authenticity* cues | ✅ Confirmed |
| 2026-10-16 | D11 | Person/phone detector: full YOLOv3 (248 MB) is default, YOLOv3-tiny fallback only; score 0.4 instead of upstream 0.6 | ✅ Evidence-based (P-5) |
| 2026-10-15 | D10 | Head pose: MediaPipe transformation matrix is primary; ported Proctoring-AI solvePnP is the fallback only (measured yaw bias) | ✅ Evidence-based (P-2) |
| 2026-10-14 | D9 | Reference-photo identity check (I-4) promoted from stretch to MVP | ✅ Owner decision |
| 2026-10-12 | R1 | **Hard requirement:** must separate nervous-genuine from fake (one-sided blink cue, head motion → confidence only, speech-gated mouth cue, F-4 tests) | ✅ Owner requirement |
| 2026-10-12 | D4 | Delete hash-based placeholder in `video_intake.py`; missing models → `UNAVAILABLE` | ✅ Confirmed |
| 2026-10-12 | D5 | New thresholds in `config.py` (additive); fixtures retuned only via `generate_fixtures.py` | ✅ Confirmed |
| 2026-10-12 | D6 | Cache derived features by video sha256 in `storage/features/` | ✅ Confirmed |
| 2026-10-12 | D7 | Weights in gitignored `storage/models/` via `scripts/download_models.py` | ✅ Confirmed |

## Open questions / cross-team dependencies

| # | With | Topic | Status |
|---|---|---|---|
| Q1 | M5 | How deepfake sub-score enters final fusion (replace identity+sync vs report-only) | Open — default: report-only |
| Q2 | M5 | Optional `details` field on `ModuleResult` | **Added** (additive, default None) — M5 please confirm |
| Q3 | M1 | `audio_extractor.extract_wav()` signature + Whisper question segments | Open — M3 ships own fallback |
| Q4 | M1 | Speech-activity (VAD) timeline | Open — RMS-energy fallback |
| Q5 | M5 | `/api/analyze` stays synchronous? | Open — default: sync + cache |

## Blockers

_None yet._

---

## Dataset acquisition (G-0; details in M3_DATA.md)

All data lives in gitignored `VeriInterview/datasets/`. Archives are kept compressed until a task needs them. Every download is size-checked; Zenodo/Mendeley files are also checksum-verified (MD5/SHA-256).

| Dataset | Tier | Needed for | Status | Local path | Size | Verified contents / notes |
|---|---|---|---|---|---|---|
| OEP proctoring (MSU) | A | P-1, P-3, P-5, P-6, F-4 | ✅ | `datasets/oep/OEP_database.tar.gz` | 11.8 GB | Exact byte size, full gzip stream readable. 24 subjects: 15 actors (s1–9, 17, 20–24) + 9 real exam takers (s10–16, 18, 19). Each has a webcam `.avi` (`<user>1.avi`), a wearcam `.avi` (`<user>2.avi`), `.wav` and `gt.txt`. `gt.txt` rows: `MMSS MMSS type`. Types per paper: 1 = book/notes/paper, 2 = talk to a person in the room, 3 = Internet, 4 = phone call to a friend, 5 = use phone/device. Counts in the labels: 1:192, 2:285, 3:52, 5:15, **6:21 (undocumented; treat as unknown)**, 4:0. No checksum is published (TLS chain broken on the MSU host). |
| Mendeley suspicious-behaviour | A | P-6 sanity | ✅ | `datasets/mendeley_proctor/` | 1.5 MB | SHA-256 OK. CSV has 38 columns (face, eyes, head pitch/yaw/roll, phone, gaze, pupils, label) + 2 docx tables. CC BY 4.0 |
| LFW pairs | A | I-1 | ✅ | `datasets/lfw/` (sklearn cache) | 2.3 GB | `10_folds`: 6,000 pairs (3,000 same / 3,000 different), 250×250 colour |
| COCO val2017 person/phone | A | P-5 | ✅ | `datasets/coco/person_phone/` (+ annotations zip) | 0.4 GB | 914 images: 214 phone, 250 single-person, 250 multi-person, 200 negatives. `subset.json` holds the boxes (person = COCO 1 / YOLO 0, phone = COCO 77 / YOLO 67) |
| AVLips v1.0 (raw) | A | S-3 limits, F-4 fakes, X-2 | ✅ | `datasets/avlips/AVLips v1.0.tar.xz` | 9.3 GB | Archive fully readable. 7,602 `.mp4` + 7,602 `.wav`: 3,396 real (`0_real`), 4,206 fake (`1_fake`; MakeItTalk/Wav2Lip/TalkLip/SadTalker). No license stated |
| AVLips val (preprocessed) | A | X-2 only | ✅ | `datasets/avlips/val.tar.xz` | 3.8 GB | 3,409 PNG composites (LipFD input format), **not usable by SyncNet**. Can be deleted early if space is needed |
| DeepfakeTIMIT | A | I-2, I-3, F-3, F-4 | ✅ | `datasets/deepfaketimit/` | 0.23 GB | MD5 OK. **Fakes only**: 640 `.avi` + 640 `.wav` (320 HQ, 320 LQ) across 32 VidTIMIT subjects. License CC BY-NC-ND 4.0 |
| VidTIMIT (real counterparts of the 32 subjects) | A | I-2, I-3, F-4 | ✅ | `datasets/vidtimit/` | 2.3 GB | 32 zips, MD5 + zip CRC OK. Per person: `video/<sentence>/NNN` JPEG frames (512×384) + `audio/*.wav` (32 kHz). 77k frames, 352 audio clips. CC BY-NC-ND 4.0, cite Sanderson & Lovell 2009 |
| RT-BENE (all 17 subjects) | A | P-4 | ✅ | `datasets/rt_bene/` | 0.9 GB | 35 files, MD5 OK (17 eye-image tars + blink-label CSVs + subject list). CC BY-NC-SA 4.0 |
| GRID (s1, s2, s3, s4, s15, s20 + alignments) | A | G-3, S-2..4, I-2, P-4, generated set | ✅ | `datasets/grid/{video,audio}/` + `alignments.zip` | 3.0 GB | From the **Sheffield mirror** (Zenodo was throttled to ~0.2 MB/s). Every zip passed its CRC check. 6 talkers × 1,000 `.mpg` (normal quality, 360×288) + 6 × 1,000 `.wav` (25 kHz). Alignments zip has 68k word-timing files and passed its CRC check. CC BY 4.0 |
| FakeAVCeleb | B | S-3, I-3, F-1 eval | ⬜ form not sent (owner) | | | |
| AV-Deepfake1M (subset) | B | per-segment sync | ⬜ form not sent (owner) | | | NC |
| Celeb-DF v2 | B | I-3 | ⬜ form not sent (owner) | | | |
| VoxCeleb2 (test) | B | I-1, S-2 | ⬜ form not sent (owner) | | | |
| MPIIFaceGaze | B | P-3 | ⬜ | | | NC |
| FaceForensics++ | B (opt.) | I-2 | ⬜ | | | |
| UBFC-Phys (stress T1/T2/T3 video) | B | F-4 nervous-vs-fake | ⬜ needs owner's IEEE DataPort login | | | real stressed people |
| ChaLearn First Impressions V2 | B (opt.) | F-4 | ⬜ needs owner's sign-in | | | talking-to-camera clips |
| MIT Interview Dataset | C | F-4 | ⬜ request to authors | | | 138 mock interviews |
| Generated GRID test set | — | S-3, I-2, F-3, T-2, N-4 | ⬜ (after GRID) | `datasets/m3_generated/` | ~1–3 GB | `scripts/build_m3_testset.py` |

**Disk:** about 29 GB downloaded + 2.4 GB VidTIMIT (LFW turned out to be 2.3 GB, not 0.3 GB). Free space is about 170 GB.

## Calibration results (fill in during S-3 / I-2)

| Clip | Manipulation | Offset | LSE-C | Identity p5 sim | distinct_ids | Expected | Pass? |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Performance (N-4)

| Video | Length | Identity | Landmarks (gaze + eye) | Objects (YOLOv3) | SyncNet | Total | Setup |
|---|---|---|---|---|---|---|---|
| GRID s2, 60 sentences | 179 s | 52 s | 60 s | 57 s | 37 s | **205 s** | serial, AC power, i5-1135G7 CPU |
| same | 179 s | 47 s | 80 s ‖ | 111 s ‖ | 61 s ‖ | **158 s** | stages 3–5 in parallel threads (‖) |
| GRID 36 s clips | 36 s | 50–65 s | 17–21 s | — | 50–99 s | 130–185 s | serial, battery + other jobs (throttled) |

YOLO input 320 vs 416 (COCO 914): 274 vs 468 ms/img, phone recall 0.40 vs 0.50 → **kept 416** (phone is strong evidence).

---

## 🗑️ Safe-to-delete gate for `datasets/` (task T-5, mandatory)

`datasets/` (~29 GB, **real faces and voices**) is used only for calibration and testing. The app never reads it at runtime. **Every box below must be ticked before deleting it, and it must be deleted once they are all ticked.** Do not skip this step.

**A. Results are saved in git (the numbers only):**
- ⬜ Final thresholds and weights committed in `backend/config.py` (`SyncThresholds`, `IdentityThresholds`, `GazeThresholds`, `DeepfakeWeights`, `VisionBudget`)
- ⬜ Calibration CSVs committed in `docs/m3/calibration/` (SyncNet shifts/swaps, LFW threshold/FAR, identity splices, gaze vs OEP, blink vs RT-BENE, YOLO vs COCO)
- ⬜ F-4 nervous-vs-fake report committed (AUC, stress Δ, number of HIGH on OEP / UBFC-Phys = 0)
- ⬜ Calibration, Performance and Dataset tables in this file filled in
- ⬜ CSVs checked to contain **no images, frames, audio or face embeddings**

**B. Everything can be re-created:**
- ⬜ `scripts/fetch_datasets.py` committed and able to re-download every dataset (URLs + MD5/SHA, resumable)
- ⬜ `scripts/build_m3_testset.py` + `datasets/m3_generated/manifest.csv` schema committed (the generated set can be rebuilt)
- ⬜ `M3_DATA.md` links and licenses still correct

**C. The work is finished and integrated:**
- ⬜ All M3 tasks through T-4 are ✅
- ⬜ M5 has integrated `deepfake_fusion` (N-3), and the 5 demo candidates give their documented outcomes end to end
- ⬜ Full `pytest tests/ -v` green, run once **with** datasets (vision tests executed, not skipped)
- ⬜ Branch `deepfake_feature3` merged (or the PR approved)

**D. Runtime keeps working without the datasets:**
- ⬜ `storage/models/` (~0.7 GB weights) **kept**. This folder is NOT part of the deletion.
- ⬜ Moved `datasets/` aside temporarily → `pytest tests/` still passes (vision tests show as *skipped*, not failed) and the demo + `/api/analyze` with a video still work

**E. Delete:**
- ⬜ Run `rm -rf VeriInterview/datasets/` (and empty the Recycle Bin if it was deleted from Explorer)
- ⬜ Check that `git status` is clean (nothing from `datasets/` was ever tracked: `git log --all -- datasets/` returns nothing)
- ⬜ Record the deletion date in the work log below

---

## Work log

### 2026-10-12
- Read the Master Spec, Build Spec, README and all of `docs/`. Audited the M3 adapters, `video_intake.py`, `evidence_fusion.py`, `pipeline.py`, `schemas.py`, `config.py`, the tests, the fixtures, and the frontend component keys.
- Researched syncnet_python (evaluate/pipeline internals, MIT license, weight URLs), InsightFace 2.1 (pure-python wheel, buffalo_l non-commercial weights), Proctoring-AI (eye-ratio, solvePnP, YOLOv3 class 0/67 logic, TF/dlib dependencies), MediaPipe 1.1 (Python 3.13 wheels), Wav2Lip LSE-C/LSE-D reference ranges, and LipFD (SyncNet evasion).
- Environment: Python 3.13.5, MX450 2 GB, no system ffmpeg.
- Wrote `M3_TASKS.md` and this tracker.
- Owner decision: **do not use local `data/samples/` data**. Use public online datasets only.
- Researched datasets for every task and wrote `M3_DATA.md`. Tiers: A = open now, B = request form, C = optional. Also mapped data to each task, designed a generated GRID test set, and listed model-weight sources. Found that LRS2/LRS3 are currently unavailable and that VCF has no public release.
- Plan updated: G-0 is now "acquire public datasets". Calibration and tests (I-1, I-3, S-2, S-3, P-1..P-5, N-1, N-2, N-4, T-2) now point to public data.
- **Waiting for review and decision confirmation (D1–D7) before starting G-0.**
- Owner confirmed D1, D2, D3, D8 and added requirement **R1** (nervous vs fake). Added task F-4, nervous-data sources, and the stress → blink/gaze research.
- Added `datasets/`, `storage/models/`, `storage/features/` to `VeriInterview/.gitignore`.
- Downloaded and verified the free datasets: OEP, Mendeley, LFW, COCO subset, AVLips raw + val, DeepfakeTIMIT. GRID, RT-BENE and VidTIMIT are in progress. Found the OEP cheat-type codes in the paper (code 6 undocumented). Found that DeepfakeTIMIT contains fakes only, so the matching VidTIMIT reals were queued. Found that AVLips val is preprocessed PNGs.
- Owner asked for a mandatory dataset-deletion gate. Added task **T-5** and the *Safe-to-delete gate* checklist.

### 2026-10-13
- Owner confirmed D4–D7. **All design decisions D0–D8 and requirement R1 are now confirmed.**
- GRID re-sourced from the Sheffield mirror and verified. RT-BENE is complete (35 files, MD5 OK). VidTIMIT download is in progress.

### 2026-10-14
- **G-1 done.** Created a project venv `VeriInterview/.venv` (gitignored) and installed the core requirements, then pinned `requirements-vision.txt`: torch 2.14.1+cpu, insightface 2.1, onnxruntime 1.31, mediapipe 1.1.0, opencv(-contrib)-python 4.14.0.94, python_speech_features 0.6, imageio-ffmpeg 0.6.0 (bundles ffmpeg 7.1).
  - Problem found: **OpenCV 5.0 removed `cv2.dnn.readNetFromDarknet`** (needed for the YOLOv3 person/phone port). mediapipe and insightface also pulled two different OpenCV packages. Fix: pin both packages to 4.14.0.94. `pip check` is clean.
  - Verified: MediaPipe `FaceLandmarker`, InsightFace `FaceAnalysis`, `cv2.solvePnP`/`RQDecomp3x3`, the ffmpeg binary. onnxruntime runs on CPU only (no CUDA provider); torch is the CPU build. GPU is optional and deferred.
  - The existing suite still passes (20/20). README has an "Optional: real video analysis" setup section.
- Owner asked about Google Colab. Recommendation: build and run locally. Use Colab only for an occasional heavy batch calibration run (e.g. SyncNet over thousands of AVLips clips) on a data subset. Uploading ~30 GB of real-face data to the cloud weakens the privacy plan.
- **G-2 done.** `scripts/download_models.py` (stdlib only, resumable, `--check` verify mode, `--full-yolo` option, safe zip extraction) puts weights in gitignored `storage/models/`: SyncNet v2 (54.6 MB) + `example.avi`, MediaPipe face_landmarker float16 (3.8 MB), InsightFace buffalo_l (288.6 MB zip, extracted), YOLOv3-tiny cfg + weights (35.4 MB). Sizes and SHA-256 are pinned, and `--check` passes.
  - Load tests: SyncNet loads with `torch.load(weights_only=True)` (88 tensors, no pickled code). buffalo_l gives detection + recognition + 2D-106/3D-68 landmarks + gender/age, CPU provider. MediaPipe FaceLandmarker is created with blendshapes and transformation matrices. YOLOv3-tiny runs a forward pass through `cv2.dnn` (outputs 507×85, 2028×85).
  - Network note: Windows curl (schannel) failed on Google/GitHub with `CRYPT_E_REVOCATION_OFFLINE` while the connection was unstable. The script uses Python's TLS, so it is unaffected.
- **G-3 done.** `backend/vision/media.py`: `probe()` (ffmpeg stderr parse → size, fps, duration, codec, audio presence and rate), `iter_frames()` (BGR frames through a pipe, fps resampling, max-width downscale, start/end window, one frame in memory at a time), `load_audio()` (16 kHz mono float32), `detect_speech()` (adaptive energy VAD), `question_at()` / `clip()` / `whole_video_question()`. Typed errors: `DecodeError`, `NoVideo`, `NoAudio`. Nothing is written to disk.
  - Real-format check: GRID mpeg1 360×288, DeepfakeTIMIT h264 512×384, AVLips h264 224×224, OEP mpeg4 640×480 (15.5 min decoded at 5 fps in 5.3 s). The OEP webcam `.avi` files have **no audio track** (audio is a separate `.wav`); `probe` reports this correctly.
  - VAD tuned and validated against GRID word alignments: see `docs/m3/calibration/vad_grid.md`. Default `rel_threshold=0.55`: 600 clips, acc 0.936, recall 0.954, precision 0.947, median boundary error 20/60 ms, 0 missed utterances.
  - **Data issue:** the Zenodo GRID `alignments.zip` has mismatched talker folders (s15 and s20 sentence IDs share 0/1000 with the videos). Replaced it with the Sheffield per-talker alignments (`datasets/grid/align_shef/`, all 6,000 match) and deleted the Zenodo file.
  - `tests/test_vision_media.py`: 8 tests (synthetic audio + ffmpeg-generated test clips, no datasets). Full suite 28/28. `scripts/eval_vad_grid.py` reproduces the calibration.
- **G-4 done.** Added to `backend/config.py` (additive only, existing values untouched): `MODELS_DIR`, `FEATURES_DIR`, `VisionBudget`, `VadSettings`, `FaceQuality`, `IdentityThresholds`, `SyncThresholds`, `GazeThresholds`, `EyeAuthenticityThresholds` (R1: one-sided blink rule), `DeepfakeWeights` (0.45/0.40/0.15, single-signal cap 60, identity-switch floor 75). Each value names the task that calibrates it. `tests/test_vision_config.py`: 5 checks (weights sum to 1, one signal alone cannot reach HIGH, threshold orderings, VAD default matches the media module). Suite 33/33.
- **G-5 done.** `backend/vision/face_track.py`: lazy CPU InsightFace singleton (`ModelUnavailable` if weights or deps are missing). `detect()` runs SCRFD only, plus per-face ArcFace embedding / 3D-68 pose on request. `IouTracker` is a port of syncnet `track_shot` (IoU > 0.5, gap ≤ 1 s). `pick_primary()` takes the longest track, then the largest. `assess_track_quality()` works per track. `track_video()` respects the `VisionBudget` cap.
  - Measured problems and fixes:
    1. Fixed absolute blur threshold rejected **100 % of DeepfakeTIMIT fake frames**. Blur is a fake artifact, so this would hide fakes. Sharpness on a 112 px crop varies ~6× by source (DFTIMIT 52, GRID 123, AVLips 106, OEP 298). Now relative: a frame is excluded only if it is < 0.4 × the track median, or below an absolute floor of 15. DFTIMIT usable 0 % → 100 %.
    2. Tight 224 px face crops (AVLips): det_size 640 found faces in only 11–16 % of frames. det_size 320 finds 100 %, the same as 640 on GRID/OEP, and detection is 4× faster (164 → ~45 ms).
    3. CPU cost per frame: detection ~50 ms, + ArcFace ~130 ms, + 3D pose ~40 ms. Embeddings and pose are opt-in, so identity pays ~230 ms/frame at 2 fps and other analyzers pay ~50 ms.
  - OEP 60 s: one stable primary track (120/120 frames). A second person was detected in 3 % of frames.
  - Config: `FACE_QUALITY.min_sharpness` replaced by `min_sharpness_abs`/`min_sharpness_rel`. Added `VISION_BUDGET.face_det_size=320`.
  - `tests/test_vision_face_track.py`: 8 tests (IoU, 2-person tracking, gap/jump splitting, primary choice, relative blur keeps uniformly soft fakes, small/low-conf/turned-away flags, empty-result maths, real model on a blank frame). Suite 41/41.
- All free (Tier A) datasets are complete and verified: VidTIMIT (32/32, MD5 + CRC) and RT-BENE (35/35, MD5). Total `datasets/` ≈ 33 GB.

### 2026-10-15
- LFW calibration from 2026-10-14 hung (~24 h, no output: buffered pipe, likely system sleep). Restarted as a resumable run (rows appended per pair, progress log); ~280 ms/image.
- **I-4 real-data check:** the DeepfakeTIMIT fake (fram1's face on fadg0's video) vs a real fadg0 reference photo gives sim 0.223 → HIGH `reference_mismatch` (caught). vs a fram1 photo → 0.666 match (as expected). GRID s1 session vs s1 photo 0.966 match, vs s4 −0.074 and vs s2 0.042 → mismatch. Blank photo → note only, no risk.
- **S-1 done.** `backend/vision/syncnet/model.py` is upstream `SyncNetModel.py` verbatim (MIT header + LICENSE.md). `backend/vision/syncnet/__init__.py` ports `SyncNetInstance.evaluate` to in-memory arrays (no ffmpeg/temp JPEGs), loads weights with `weights_only=True`, and returns offset, LSE-C, LSE-D and framewise confidence. On `example.avi` it gives **AV offset 3, min dist 5.353, confidence 10.021, identical to the upstream README**. A 5 s excerpt keeps offset 3 (conf 9.9). CPU speed is ~10 s per 5 s of video, ~66 s for the 12.7 s clip while LFW ran in parallel → speed work goes in N-4. Detail: MFCCs must use int16-scale audio (c0 = log energy), so float audio is rescaled ×32768. `tests/test_vision_syncnet.py`: 4 tests.
- **S-2 done.** `backend/vision/syncnet_runner.py`: speech (VAD or Whisper) → windows (merge pauses < 2 s, ≥ 2 s, split at 10 s, ≤ 60 s per video spread over the interview) → 25 fps decode → SCRFD every 5th frame + interpolation + median-13 smoothing → upstream `crop_video` geometry → SyncNet. Aggregation is frame-weighted (median offset, offset stability, LSE-C/LSE-D, legacy `av_offset_frames` + `sync_confidence` = LSE-C/7). Status reasons: `no_audio`, `no_speech`, `no_face`, `too_short`.
  - GRID s1 (6 sentences, ~16 s of speech): genuine offset **+1** (GRID baseline). Audio delayed 5 frames → **−4**. Audio advanced 10 frames → **+11**. Shifts recovered exactly; stability 1.0. **Sign: delayed audio → negative offset.**
  - LSE-C on GRID ≈ 4.6 (low resolution + pauses), below LRS2's ~7, and it does **not** drop for shifted copies (the offset search re-aligns). The offset is the desync signal; LSE-C is mainly a reliability signal. Calibrate in S-3.
  - Merge gap raised 1 s → 2 s: GRID pauses (~1.5 s) otherwise left no window ≥ 2 s. Upstream scores whole tracks including pauses.
  - Speed: ~2.5–5 s CPU per second of video (LFW was running in parallel). Covered in N-4.
- **LFW identity calibration done** (`docs/m3/calibration/identity_lfw.md`, per-pair CSV). 6,000 pairs, 0 dropped. Genuine median 0.678 (p1 0.395); impostor max **0.233**. **Threshold set to 0.30** for `same_person_cosine` and `reference_match_cosine`: FAR 0 %, FRR 0.4 %. Best balanced threshold 0.234 (99.87 %). Weak reference-match band is 0.30–0.40 (1.2 % of genuine pairs → MEDIUM note, never HIGH).
- **S-3 in progress:** `scripts/build_m3_testset.py` generated **78 clips** (6 GRID talkers × 13 manipulations, 162 MB) + `manifest.csv`. `scripts/calibrate_syncnet.py` is running on the 66 sync-relevant clips (resumable CSV).
- **P-1 done.** `backend/vision/landmarks.py`: MediaPipe Tasks FaceLandmarker in VIDEO mode, 478 points (incl. iris) in pixels, blendshape subset (eyeBlink*, jawOpen, mouthClose, eyeLook*), transformation matrix, and `matrix_to_euler()` → pitch/yaw/roll. Landmark coverage 100 % on GRID, OEP webcam and AVLips. Head-pose yaw agrees with InsightFace's independent estimate (AVLips 19.3° vs 19.6°). Frames with blink > 0.5: 1–9 %.
  - **Caveat:** MediaPipe still fits a mesh on a partly covered face (covered clip: 83 % landmark frames vs ~67 % truly visible). P-3/P-4 must gate landmark frames on a clear SCRFD face so blink and mouth cues never use hallucinated landmarks (R1).
  - Speed 53–89 ms/frame with two calibrations sharing the CPU → N-4.
- **P-2 done.** `backend/vision/head_pose.py`: Proctoring-AI 6-point solvePnP port (same 3D model, focal = width, mapped to MediaPipe indices 1/152/33/263/61/291, true Euler via Rodrigues). Exact on synthetic projections (test). On real video it tracks motion (yaw corr 0.76–0.95 vs MediaPipe) but has a **systematic yaw bias**: OEP 15.9° vs 7.9°, AVLips 30.0° vs 19.3°, GRID 21.9° vs 5.0°. InsightFace's independent 3D-68 pose agrees with MediaPipe (AVLips 19.6°, OEP 12.3°, GRID 8.2°). Decision D10: `pose_from_obs()` uses the MediaPipe matrix and falls back to solvePnP only without a matrix.
- **Phase 1 (face identity) complete: I-1..I-4 ✅.** End-to-end on the generated set: 24/24 as expected (`docs/m3/calibration/identity_generated.md`). Genuine → LOW 5.0 / conf 0.92. Splice → HIGH 90, 2 identities, drift at 18.0 s (exact cut). Face hidden for a third → LOW with confidence 0.78 (not risk). 240p → unchanged. Tests: 8 identity + 9 adapter.
- P-3 groundwork: MediaPipe iris 468 = image-left eye (near 33), 473 = image-right eye (near 263), 100 %. Camera-facing GRID: x-ratio p1–p99 = 0.58–1.50, so the upstream 3 / 0.33 cut-offs never fire (correct). OEP: x > 3 in 3.6 % of frames. **Blinks collapse the y-ratio** (p1 −0.37), so blink frames must be excluded from gaze or they read as "looking up". OEP members: 24 webcam (3.9 GB), 24 wearcam (7.9 GB), 24 wav.
- **P-3 code done, evaluation pending.** `backend/vision/gaze.py`: upstream eye-ratio rule on MediaPipe irises (both eyes must agree), blink frames skipped, head pose (MediaPipe) overrides (yaw > 30°, |pitch| > 25°, including looking DOWN at notes), 0.5 s debounce, `valid(t)` gate for frames with a clear SCRFD face, per-question stats + first-answer baseline, ≥ 2 s off-screen events for the timeline, head-variance label. 7 tests. `scripts/eval_gaze_oep.py` compares off-screen share inside vs outside OEP cheating intervals (first 10 min per subject, 3 fps).
- **S-3 early finding:** the 240p / CRF 40 clip gives **LSE-C 1.17** (genuine 4.6–6.6). A poor camera alone collapses SyncNet confidence, so in S-4 low LSE-C must lower confidence and never become risk by itself (R1).
- Housekeeping: the OEP extraction log was accidentally written to the repo root (`C:/VeriInterview-TCS/oep_extract.log`). Delete it once extraction finishes. It must not be committed.

### 2026-10-16
- **Phase 2 (lip-sync) complete: S-1..S-4 ✅.** S-3 results (`docs/m3/calibration/syncnet_grid.md`): offset recovered within ~1 frame of each talker's genuine baseline (≈ +1) for ±2/±5/±10. Partial shift: the affected windows read −6/−7 while the others read 0–2. **Audio swap: LSE-C 1.01–1.57 AND stability 0.25–0.75. 240p genuine: LSE-C 1.17–5.39 BUT stability 1.00.** Offset stability separates a bad camera from replaced audio.
- Thresholds: `lse_c_good` 7.0 → 6.0 (GRID genuine mean 5.88), `lse_c_low` 3.0 → 2.0, stability 0.6, bands ≤ 2 / 3–5 / ≥ 6 frames.
- `syncnet_adapter.py` real path: (1) offset bands, (2) per-window desync flags with timestamp + question (one manipulated answer), (3) `av_no_sync` = LSE-C < 2 AND stability < 0.6 → HIGH, (4) low LSE-C with a stable offset → lower confidence only (R1). Legacy fixture path unchanged. Evidence never says "deepfake". On the 66 clips: genuine 6× LOW, silent 6× LOW, 240p 6× LOW (conf 0.20–0.90), swap 6× HIGH, partial 6× HIGH, ±10 12× HIGH, −5 HIGH / +5 MEDIUM, +2 LOW / −2 MEDIUM (GRID's +1 baseline puts −2 at 3 frames ≈ 120 ms, a single-module MEDIUM). Suite 92/92.
- **P-4 done.** `backend/vision/eye_auth.py`: blink onsets from MediaPipe blendshapes (cross-checked with EAR), rate, inter-blink CV, one-sided flags (`too_few_blinks`, `metronomic_blinks`; `high_blink_rate` = stress note, never risk), and `mouth_speech_mismatch_pct` (static closed mouth during speech only). Results: `docs/m3/calibration/eye_auth_grid.md`. Genuine blink rates 27–36/min; **one genuine reader (s4) did not blink in 36 s**, so `min_eye_visible_s` went 60 → 120 s and the cue stays corroborative only. New `frozen_face` manipulation in the test set (now 84 clips): mismatch 24–27 % vs 0 % genuine. RT-BENE is unusable (eye crops only), noted.
- **P-5 done.** `backend/vision/objects.py`: Proctoring-AI person_and_phone logic on cv2.dnn Darknet weights (COCO 0 = person, 67 = phone, NMS 0.5), ≥ 2 consecutive samples needed for an alert, events with timestamps and question ids. COCO 914-image eval (`docs/m3/calibration/objects_coco.md`): **YOLOv3-tiny is too weak** (score 0.6: phone recall 0.07, multi-person 0.15). **Full YOLOv3 at score 0.4:** phone P 0.98 / R 0.50, multi-person P 0.97 / R 0.84, 0 false phones on 200 negatives. Decision D11: full YOLOv3 is the default (download script now includes it by default; `--no-full-yolo` to skip), score 0.4. Cost ~1.6 s/image on CPU → sampling rate is set in N-4. 3 tests.
- **P-6 code done.** `proctoring_adapter.analyze_video()` combines gaze (MediaPipe at `landmarks_fps`, gated on clear SCRFD faces from the identity track) with YOLOv3 objects (`objects_fps`). `compute()` real path: gaze risk = 0.3 × overall off-screen + 0.6 × worst later-question excess over the candidate's own first-answer baseline + head-variance bonus, **capped at 55 (gaze alone ≤ MEDIUM)**. Phone → ≥ 85 and extra person → ≥ 80 (strong). Question-spike evidence names the question and baseline. Steady aversion from the start gets a "common with nervousness" note (R1). 6 tests incl. nervous-steady < question-spike. Legacy fixture path unchanged; all prior proctoring + e2e tests pass.
- **P-3 done** (`docs/m3/calibration/gaze_oep.md`). OEP, 24 × 10 min: normal off-screen median 2.2 %, p90 10 %, max 20.7 %. Cheating raises it in 17/24 (8.3 % vs 4.2 %). By type: reading notes 25 %, talking 8 %, Internet 4 %, phone 3 %, undocumented type 6 34 %. Gaze is real but partial evidence, which confirms the weak-evidence cap and the separate strong phone/person signal.
- **Phase 3 complete (P-1..P-6 ✅).** OEP real-webcam phone check (`calibration/objects_coco.md`): phone visible in 54 % of phone-use frames vs 1.7 % of normal frames, no phantom second person. **`object_persist_samples` 2 → 3** (2 would give ~0.5 false alerts per 30 min at 1 fps).
- **F-1 done.** `backend/modules/deepfake_fusion.py` + `eye_authenticity_result()`. Added optional `ModuleResult.details` to `schemas.py` (additive, default None, Q2 for M5). End-to-end first pass showed one clear finding being averaged away (audio swap: sync HIGH 65 → sub-score LOW 29; frozen face: eye MEDIUM 50 → LOW 14). Added `single_input_floor` 0.75: swap → **MEDIUM 48.8**, frozen → **MEDIUM 37.1**, splice → **HIGH 75**, genuine → **LOW 7.5**. Nervous profiles stay LOW (eye score 5; high blink rate adds 0).
- **F-3 done** (12 tests). **F-4 synthetic part done.** Real-data F-4 needs N-4 first.
- **N-1 done.** `backend/adapters/video_intake.py` rewritten: shared tracks, landmarks computed once for gaze AND eye cues, objects optional, SyncNet over speech, per-step `status_reason`, sha256 cache in `storage/features/` (numbers only; second call is a cache hit). The SHA-256 placeholder is gone; `derive_av_signals()` (used by main.py) now runs the real analysis.
  - **Speed problem:** 130–185 s of CPU for a 36 s clip (identity 50–65 s, landmarks 17–21 s, SyncNet 50–99 s; other jobs were running). → N-4 next.

### 2026-10-17
- **N-4 in progress.** Profiling showed the laptop **on battery** (Balanced plan, i5-1135G7 held at its 2.4 GHz base), and Windows' background-process throttling made timings swing wildly (SCRFD 129 → 3,123 ms/frame on the same frames). **Final timings must be measured on AC power with the Best-performance power mode.**
  - Relative costs (idle CPU, battery): SCRFD detect ~130 ms; + ArcFace ~430 ms; + 3D-68 pose ~300 ms; MediaPipe landmarks ~43 ms; YOLOv3-416 ~950 ms; SyncNet ~1.5 s per second of speech (8 threads ~17 % faster than 4).
  - Rejected: a 5-keypoint yaw shortcut to skip the 3D pose model (corr 0.38 with the 3D pose, p95 error 25°). Too inaccurate for turned-away gating (R1).
  - Applied: `identity_fps` 2 → 1 (re-verified: 6/6 genuine LOW, 6/6 splices HIGH with drift at the exact cut). SyncNet `syncnet_max_seconds_total` 60 → 30 s with torch using all 8 threads. **New per-question SyncNet budget:** with question timings, every answer gets an equal share and at least one window, so one manipulated answer cannot be skipped (test added).
  - Kept `objects_fps` = 1: OEP phone-use episodes are short (median 4 s, p25 2 s), so lower rates would miss them. The 3-in-a-row rule favours precision over recall (R1). Revisit with a full-video episode check on AC power.
- **N-4 SyncNet re-check with the 30 s budget** (30 clips): genuine 6/6 LOW, shift −10 6/6 HIGH, partial 6/6 HIGH, 240p 6/6 LOW, audio swap 5/6 HIGH + 1 MEDIUM. The miss (s3) had stability 0.667: with only ~3 windows stability is coarse (0.33/0.67/1.0). Genuine and 240p were 1.00 in all 96 runs, so **`min_offset_stability` 0.6 → 0.75**; s3 re-scores HIGH 65.
- **F-2 done.** `docs/m3/EVIDENCE_TYPES.md`: every evidence type the M3 modules emit (collected from the code), with severity, meaning, trigger, whether it carries time/question, which ones are confidence-only, and the deepfake fusion rules M5 can quote.
- **N-4 done (CPU).** On AC power a 3-min video took 205 s serially (identity 52, landmarks 60, objects 57, SyncNet 37). Stages 3–5 now run in parallel threads after identity: **158 s (0.89× real time)**, results unchanged. YOLO at 320 was rejected (phone recall 0.50 → 0.40). Further speed needs the MX450 GPU (CUDA torch + onnxruntime-gpu, optional, multi-GB install).
- **N-2 done.** `main.py /api/analyze`: optional `reference_photo` upload, video saved to a temp file → `video_intake.analyze_video` in a worker thread (`run_in_threadpool`) → temp file deleted. Real `behavior_signals` + `eye_authenticity` go into the record, new `video_analysis_note`, `video_analysis_meta` (timings). **This changes M5's file: PR with M5 review.**
- **N-3 done (backend).** `pipeline.run_full_analysis` adds `deepfake_assessment` (report-only, Q1 default). Demo candidates: genuine LOW 9.2, weak LOW 11.6, AI-assisted LOW 19.0 (AI help is not a media fake), **deepfake HIGH 100**, **nervous genuine LOW 11.9**. Overall risk unchanged. The frontend card for the sub-score is not built (UI owner). Suite 121/121.

### Backend API for the frontend (owner request)
- **`backend/api_vision.py`** (router mounted in `main.py` with one `include_router` line): `POST /api/vision/jobs` (video + optional reference photo + optional question windows → 202 job), `GET /api/vision/jobs/{id}` (status, stage, ready-to-show stage label, progress 0–1, queue position, result), `DELETE /api/vision/jobs/{id}`, `GET /api/vision/health` (which models/deps/checks are available). Typed Pydantic contract: `VisionJob`, `VisionResult` {deepfake, identity, sync, eye_authenticity, behavior, merged time-sorted `timeline`, meta}.
- One analysis at a time (CPU-bound), others queue. Upload validation (type, size ≤ 500 MB / 10 MB photo, `questions` schema). Temp video deleted after the job (test-verified). Jobs expire after 1 h, max 50 kept. Failures become `status: error` with the message, never a server crash.
- `video_intake.analyze_video(progress=...)` callback added for stage progress.
- Tests: `tests/test_api_vision.py`, 6 (lifecycle + contract + temp deletion, delete, validation 415/413/422/404, failure → error, health, missing signals). Needs `httpx` (installed in .venv; skips if missing). **Real e2e through the API** on `s1_identity_splice.mp4`: progress queued → identity 5 % → 35/55/75 % → done. Deepfake **HIGH 75** with summary. Timeline `identity_drift` at 18.0 s in Q2. 103 s (CPU shared with F-4).
- `docs/API.md` now documents the full contract plus UI rules for the frontend.
- **F-4 first run caught an R1 violation:** 5 of 22 stressed OEP students scored deepfake HIGH, all through identity "2 people". Diagnosis (subjects 6, 12, 15, 18, 19): **not other people. The same student at another angle** (turned to talk, looking down at notes) formed its own cluster, because the clustering link (0.40) is stricter than the same-person threshold (0.30). Subject 15 also had an unusual opening 10 s, so the rest of the session (cluster 0.341 vs the anchor: same person) hovered around the threshold frame by frame.
  - **Fix 1:** a cluster counts as a different person only if its centroid is < 0.30 vs the anchor (`count_distinct_people`).
  - **Fix 2:** candidate **gallery** = anchor + long clusters confirmed as the same person; each frame uses its best match. An impostor cluster (~0.0–0.14 vs the anchor; LFW impostor max 0.233) never joins.
  - Re-verified: OEP 6/12/15/18/19 → LOW (5–16); GRID splices s1/s4/s20 → HIGH 90 (2 people); genuine → LOW. 2 new tests. F-4 re-run from scratch.
  - Also from the first run: lip-sync UNAVAILABLE on every OEP clip (students barely speak during a written exam: honest "not assessed"). DeepfakeTIMIT ~4 s clips: reference photo → ~15 HIGH, ~16 MEDIUM (weak match 0.30–0.40), 1 LOW at the 30 boundary; clips too short for lip-sync and blink checks.

### 2026-10-19
- **F-4 done** (`docs/m3/calibration/nervous_vs_fake.md`). With the identity fix: **stressed OEP 24/24 LOW (mean 5.4), 0 HIGH → R1 met on real data.** DeepfakeTIMIT 21 HIGH + 10 MEDIUM + 1 LOW (AUC 1.000 vs stressed). **AVLips lip-sync fakes: 0/30 flagged** (half sync MEDIUM → deepfake ~23 LOW), real-vs-fake AUC 0.655. AVLips reals: 24 LOW, 5 MEDIUM, 1 unavailable. Thresholds deliberately not re-tuned on test data. **Known limitation:** SyncNet-trained lip-sync generators evade SyncNet (as LipFD reported) → recommend stretch X-2 (LipFD detector) or document.
- Performance note: AVLips `.tar.xz` cannot seek (6–67 min per clip by random access) → one-pass extraction into `datasets/avlips/eval_sample/`. The laptop slept once mid-run (~22 h gap); the resumable CSV kept all results.
- **Phase 6 (T-1..T-4) done.** `pytest.ini` registers the `vision` marker. `tests/test_vision_e2e.py` runs the full pipeline on 10 manifest expectations (genuine LOW ×3, splice identity + deepfake HIGH, shift −10 sync HIGH, swap sync HIGH, 240p sync LOW, frozen eye MEDIUM, occluded identity LOW) plus a broken-file case. All pass, ~5 min uncached. Full suite 140/140; `-m "not vision"` 129. Docs updated across `docs/` + README. **T-5 (delete datasets) stays gated** until M5 integrates and the branch is merged (Safe-to-delete gate).
- **X-2 feasibility (deferred).** LipFD: **no licence → unusable**. AVAD (MIT code, CVPR 2023, built for AV inconsistency): checkpoint downloaded (163 MB, 41 M params), all `.pth` load with `weights_only=True`, `pca.pkl` references only numpy/sklearn (restricted unpickling feasible). Blockers: weights terms unstated; 2022 stack needs porting (torchaudio has no build for torch 2.14; mel-spectrogram must be reimplemented; unsafe `shell=True` ffmpeg must be replaced); **no reference output to verify a port**. Decision: deferred. Real-time face-swap (the main live-interview attack) is covered via the reference photo (31/32). Wav2Lip-style lip-sync fakes stay a documented limitation (LIMITATIONS.md §8). Kept in gitignored `datasets/vendor_eval/`; `einops` installed in .venv only (not added to requirements).
