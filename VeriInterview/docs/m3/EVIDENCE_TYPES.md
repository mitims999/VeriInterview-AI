# M3 Evidence Types — catalog for M5 (XAI / report)

Every `Evidence` item emitted by the M3 modules. Each item has
`type`, `severity` (`low` | `medium` | `high`), `message` (already plain
English, safe to show as-is), and where known `question_id` and
`timestamp` (seconds from the start of the video).

**Rules for the explanation layer (from the specs and R1):**
- Never turn any of these into a verdict ("cheater", "deepfake confirmed"). They are reasons for human review.
- `severity: low` items are context (what was checked and found normal, or why confidence dropped). Show them collapsed or greyed.
- Items marked **confidence-only** never raise risk. They explain a lower confidence and should be worded that way.
- Gaze, head pose, phone and extra-person items are **behaviour** evidence (`behavior_risk`). They never feed the deepfake score (decision D3).

## Face identity — `components.identity_risk` (feature `face_identity`)

| type | severity | meaning | triggered by | has time / question |
|---|---|---|---|---|
| `identity_drift` | low | Face matched the opening face throughout | no drift | — |
| `identity_drift` | medium / high | Face stopped matching the session's opening face for a stretch | ≥ 1 s of frames below the LFW-calibrated threshold (0.30) | ✅ start, end, question |
| `identity_switch` | high | Two different people each appeared for several seconds as the main person | ≥ 2 identity clusters, each ≥ 3 s | (see drift items) |
| `reference_mismatch` | high | Interview face does not match the candidate's reference photo | similarity < 0.30 (LFW: 0 % false accepts) | — |
| `reference_weak_match` | medium | Only a weak match to the reference photo; old/low-quality photos can cause this | 0.30 – 0.40 | — |
| `reference_match` | low | Matches the reference photo | ≥ 0.40 | — |
| `reference_photo` | low | Photo unusable (no face, too small) or had several faces | upload problem | — |
| `face_lost` | low | No face for X % of the video. **Confidence-only** | > 10 % frames without a face | — |
| `anchor_quality` | low | Opening frames inconsistent, so the identity reference is weak. **Confidence-only** | anchor quality < 0.6 | — |
| `face_identity` | low | Check not possible (no video / no face / model missing) | status ≠ OK | — |

## Lip-sync — `components.sync_risk` (feature `av_sync`)

| type | severity | meaning | triggered by | has time / question |
|---|---|---|---|---|
| `av_offset` | low | Lips and voice aligned | \|offset\| ≤ 2 frames | — |
| `av_offset` | medium / high | Voice out of step with the lips by N frames (~N×40 ms) across the speech | 3–5 / ≥ 6 frames | — |
| `av_offset_segment` | high | One part of the interview was out of sync while the rest was not (one manipulated answer) | a window with \|offset\| ≥ 6 and usable confidence | ✅ start, question |
| `av_no_sync` | high | Lips did not match the audio at any alignment; the audio may not come from the person on camera | SyncNet confidence < 2.0 AND offset unstable (< 75 % agreement) | — |
| `av_offset_unstable` | medium | Offset varied between parts of the interview | stability < 0.75, confidence OK | — |
| `sync_confidence` | low | Weak lip-sync signal, typical of a low-res/compressed camera. **Confidence-only** | SyncNet confidence < 2.0 with a stable offset | — |
| `av_sync` | low | Check not possible (no audio / speech / face / model) | status ≠ OK | — |

## Behaviour (gaze, head pose, surroundings) — `components.behavior_risk` (feature `behavior`)

| type | severity | meaning | triggered by | has time / question |
|---|---|---|---|---|
| `gaze` | medium | During Qn the candidate looked away X % of the time vs Y % in their first answer | +20 points above own baseline | ✅ question |
| `gaze` | low | Looked {direction} from t1 to t2 | off-screen episode ≥ 2 s | ✅ time, question |
| `gaze` | low | Off-centre throughout but consistently so from the first answer; **"common with nervousness"** | high but steady gaze aversion | — |
| `head_pose` | medium / high | Head movement elevated / high | yaw or pitch spread ≥ 8° / ≥ 15° | — |
| `phone` | high | A mobile phone was visible from t1 to t2 | YOLOv3, ≥ 3 consecutive 1-fps samples | ✅ time, question |
| `multi_person` | high | More than one person in frame from t1 to t2 | YOLOv3 persons ≥ 2, ≥ 3 consecutive samples | ✅ time, question |
| `behavior` | low | Nothing notable / check not possible | — | — |

Gaze alone is capped at 55 (≤ MEDIUM). Phone ≥ 85, extra person ≥ 80.

## Deepfake sub-score — `deepfake_fusion.fuse_deepfake()` (feature `deepfake`)

| type | severity | meaning |
|---|---|---|
| `deepfake_summary` | low / medium / high | One-line summary: which checks agree, or that all look genuine; lists checks not assessed |
| `mouth_speech_mismatch` | medium | For X % of the speech the mouth showed no articulation (still or replayed face) |
| `blink_pattern` | medium | Blinks unusually regular (synthesis artifact) |
| `blink_rate` | low | Very few blinks; **reading/concentrating also does this, so weak evidence** |
| `eye_authenticity` | low | Blinking and articulation natural / not assessable |
| `nervousness_note` | low | Frequent blinking is consistent with nervousness; authenticity signals normal (R1) |
| `deepfake` | low | No authenticity check could run |

The deepfake result also carries the top non-low items from identity / lip-sync / eye cues, and
`details = {sub_scores: {identity, sync, eye_authenticity: {score, level, confidence, status}},
inputs_flagged, inputs_high, strong_identity_signal, coverage}` for a per-component breakdown.

**Fusion rules M5 can quote:** HIGH only when ≥ 2 of the three checks are at least MEDIUM, or there is a
strong identity signal (a person switch or a reference-photo mismatch). One clear finding alone gives MEDIUM
("needs review"). Eye cues alone never exceed MEDIUM. Gaze and nervousness are never inputs.
