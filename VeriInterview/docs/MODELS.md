# Models & Repositories

| Purpose | Repository | License | Status in this build |
|---|---|---|---|
| Resume/JD skill match | https://github.com/srbhr/Resume-Matcher | MIT | Contract re-implemented locally (see `backend/adapters/resume_matcher.py`); swap in real textacy pipeline for production |
| Lip-sync (AV offset/confidence) | https://github.com/joonson/syncnet_python | Code **MIT** (repo LICENSE.md); `syncnet_v2.model` weights are hosted by VGG Oxford, so check their terms | **Real (M3).** Model vendored in `backend/vision/syncnet/` (reproduces the upstream reference exactly), runs on CPU over speech windows. Calibrated on 66 generated GRID clips |
| Gaze / head-pose / person+phone | https://github.com/vardanagarwal/Proctoring-AI | MIT (its landmark model was trained on non-commercial data and is not used here) | **Real (M3), ported.** Eye-ratio rule, head pose and person/phone logic re-implemented on MediaPipe Face Landmarker (Apache-2.0) + YOLOv3 via OpenCV DNN (no TensorFlow/dlib). Validated on OEP exam videos + COCO |
| Voice modulation (eGeMAPSv02) | https://github.com/audeering/opensmile, opensmile-python | **audEERING research license — commercial use requires a separate license, see docs/SECURITY.md** | **Real (M1).** Real LLD extraction, quality-aware baseline scoring; legacy fixture interface retained for synthetic demos. See AUDIO_INTELLIGENCE.md |
| Audio transcription | https://github.com/SYSTRAN/faster-whisper | MIT (review model terms separately) | **Real (M1).** Real CPU Whisper, Silero VAD and word timestamps; uploaded files and bounded live PCM sessions |
| Face identity | https://github.com/deepinsight/insightface | MIT (code); **buffalo_l weights are non-commercial research only** | **Real (M3).** SCRFD + ArcFace on CPU; same-person threshold 0.30 calibrated on LFW (0 % false accepts, 0.4 % false rejects); optional reference-photo match |
| Expected-answer semantic match | https://github.com/UKPLab/sentence-transformers (all-MiniLM-L6-v2) | Apache 2.0 | Tries the real model first, falls back to scikit-learn TF-IDF cosine similarity if unavailable offline |
| Landmarks / iris / head pose | https://github.com/google-ai-edge/mediapipe (Face Landmarker) | Apache-2.0 | **Real (M3).** 478 points + blendshapes; used for gaze, blink and mouth cues |
| Person / phone detection | Darknet YOLOv3 (COCO) via OpenCV DNN | Darknet: public domain | **Real (M3).** Full YOLOv3 at score 0.4 (COCO: phone P 0.98 / R 0.50); the YOLOv3-tiny fallback is too weak |
| Knowledge graph | https://github.com/networkx/networkx | BSD | Used directly (`backend/modules/knowledge_ownership.py::build_claim_graph`) |

## M3 vision models: install and calibration

The four video checks (face identity, lip-sync, gaze/surroundings, blink/mouth cues) now run the real models on uploaded video.
Install with `pip install -r requirements-vision.txt` (CPU torch first, see README), then `python scripts/download_models.py`
(~630 MB into `storage/models/`, SHA-256 pinned). Every threshold and how it was calibrated: `docs/m3/calibration/`.
Without the vision dependencies or weights, the checks report UNAVAILABLE and the rest of the app works unchanged.

## Why adapters (history: the pre-M3 build)

Member 1's audio path now runs real inference. The original explanation below
applies to the remaining vision adapters and legacy synthetic demos.

This prototype was assembled in a text-only development session with no GPU, no camera/
microphone capture, and no guaranteed internet access to download multi-GB pretrained
weights (SyncNet's checkpoint, InsightFace's buffalo_l bundle, dlib's 68-point landmark
model that Proctoring-AI depends on). Rather than fake the whole system end-to-end, every
adapter:

1. Documents the exact repo and script it wraps in its own docstring.
2. Implements the **exact output contract** specified in the Build Spec for that module.
3. Computes that output from structured signals (`speech_features`, `behavior_signals`,
   `identity_signals`, `av_signals`) that stand in for what the real model would emit —
   consistent with the Prototype Scope in the Master Spec ("Use sample/synthetic candidate
   profiles... audio/video **observations**").
4. Names precisely what to swap (one function body) to call the real repo once
   video/audio/GPU are available — no other module needs to change.

This keeps the system's most important property intact end-to-end: **the deterministic
evidence-fusion engine, the risk/confidence/capability scoring, and the explainability
rules all run on real logic, not mocked numbers.**

## Explicitly not used

- No LLM API key is configured (`LLM_PROVIDER=none` in `.env`). Counterfactual follow-up
  questions and expected-concept sets are pre-authored in the synthetic fixtures exactly as
  an LLM question-generation step would have produced them (Master Spec Section 31: the LLM
  only ever generates text, never owns the score). `backend/config.py` exposes
  `LLM_API_KEY`/`LLM_PROVIDER` as the integration point for a real provider.
