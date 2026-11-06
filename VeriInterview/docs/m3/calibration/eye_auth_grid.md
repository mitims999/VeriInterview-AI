# Eye/mouth authenticity cues (task P-4)

`backend/vision/eye_auth.py` on generated GRID clips (10 fps landmarks, energy VAD speech).

| talker | genuine: blinks/min | genuine: closed-vs-open EAR | genuine: mouth-speech mismatch | frozen_face (middle third): mismatch |
|---|---|---|---|---|
| s1 | 30.2 | 0.148 | 0.000 | 0.245 |
| s4 | **0.0** (no blink in 36 s) | — | 0.000 | 0.242 |
| s15 | 26.8 | 0.169 | 0.000 | 0.267 |
| AVLips real 578 | 35.8 | 0.238 | 0.000 | — |

- Blink detection (MediaPipe eyeBlink > 0.5) agrees with the geometric EAR: closed frames have 15–24 % of the open-eye EAR.
- **s4 did not blink at all** (blendshape max 0.07, EAR min 0.342 vs median 0.373, checked at 25 fps). Reading suppresses blinking, so 'too few blinks' is weak evidence: judged only over ≥ 120 s of visible eyes, and capped at MEDIUM in the deepfake fusion.
- Mouth-vs-speech: 0 % on every genuine clip, 24–27 % when the video is frozen for a third (some frozen frames have the mouth open, so this undercounts, i.e. errs safe). Threshold 15 % → all frozen flagged, no genuine flagged.
- RT-BENE holds only 60×36 eye crops, which the full-face pipeline cannot use, so it was not used for validation. It can be deleted early.
