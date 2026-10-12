# Models & Repositories

| Purpose | Repository | License | Status in this build |
|---|---|---|---|
| Resume/JD skill match | https://github.com/srbhr/Resume-Matcher | MIT | Contract re-implemented locally (see `backend/adapters/resume_matcher.py`); swap in real textacy pipeline for production |
| Lip-sync (AV offset/confidence) | https://github.com/joonson/syncnet_python | Non-commercial research license (see repo) | Adapter contract implemented; needs pretrained model + GPU + raw video to run for real |
| Gaze / head-pose / person+phone | https://github.com/vardanagarwal/Proctoring-AI | MIT | Adapter contract implemented; needs OpenCV/dlib + webcam video to run for real |
| Voice modulation (eGeMAPSv02) | https://github.com/audeering/opensmile, opensmile-python | **audEERING research license — commercial use requires a separate license, see docs/SECURITY.md** | Adapter contract + heuristic-v1 formula implemented; needs raw audio (.wav) to run for real |
| Face identity | https://github.com/deepinsight/insightface | MIT (code) — model weights have separate non-commercial terms | Adapter contract implemented; needs ONNX buffalo_l models + raw video frames |
| Expected-answer semantic match | https://github.com/UKPLab/sentence-transformers (all-MiniLM-L6-v2) | Apache 2.0 | Tries the real model first, falls back to scikit-learn TF-IDF cosine similarity if unavailable offline |
| Knowledge graph | https://github.com/networkx/networkx | BSD | Used directly (`backend/modules/knowledge_ownership.py::build_claim_graph`) |

## Why adapters, not live model inference, in this build

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
