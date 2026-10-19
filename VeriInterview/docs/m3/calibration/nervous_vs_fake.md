# Nervous vs fake on real data (requirement R1, task F-4)

`scripts/eval_nervous_vs_fake.py`: full pipeline (identity + SyncNet + eye cues → deepfake fusion; YOLO excluded, since it is not a
deepfake input) on public data. Per-clip rows: `nervous_vs_fake.csv` (numbers only).

- **stressed_real**: 24 OEP students during an exam (first 3 min; webcam + separately recorded mic, muxed)
- **dftimit_fake**: 32 DeepfakeTIMIT face-swaps (high quality), each checked against a VidTIMIT photo of the real person whose video it is
- **avlips_real / avlips_fake**: 30 + 30 random AVLips clips (fakes: MakeItTalk, Wav2Lip, TalkLip, SadTalker)

| group | n | LOW | MEDIUM | HIGH | UNAVAILABLE | mean score | identity HIGH | sync HIGH | eye MEDIUM+ |
|---|---|---|---|---|---|---|---|---|---|
| dftimit_fake | 32 | 1 | 10 | 21 | 0 | 62.7 | 21 | 0 | 0 |
| stressed_real | 24 | 24 | 0 | 0 | 0 | 5.4 | 0 | 0 | 0 |
| avlips_real | 30 | 24 | 5 | 0 | 1 | 15.1 | 1 | 2 | 0 |
| avlips_fake | 30 | 30 | 0 | 0 | 0 | 17.5 | 0 | 0 | 0 |
AUC stressed_real vs dftimit_fake: 1.000
AUC stressed_real vs avlips_fake: 0.938
AUC avlips_real vs avlips_fake: 0.655
R1 target - stressed_real scored HIGH: 0 (must be 0)

## Reading

1. **R1 holds on real stressed people: 24/24 LOW, 0 HIGH** (mean 5.4). This was after fixing the identity bug this run exposed:
   in the first pass 5/22 students were HIGH because the same student at another angle formed a second cluster (see M3_PROGRESS 2026-10-17).
2. **Face-swaps are caught: 31/32 flagged** (21 HIGH via reference-photo mismatch, 10 MEDIUM "weak match" 0.30–0.40), AUC 1.000 vs stressed real.
   The clips are ~4 s, too short for lip-sync and blink checks, so this rests on the reference photo (task I-4).
3. **Lip-sync generators are NOT caught: 0/30 AVLips fakes flagged**, real-vs-fake AUC 0.655. About half read as sync MEDIUM,
   but one MEDIUM input gives a deepfake score of ~23 (LOW). This matches LipFD (NeurIPS 2024): these generators are trained against
   SyncNet. The stressed-vs-AVLips AUC (0.938) reflects OEP students barely speaking, not detection, so do not quote it.
4. **5/30 AVLips real clips got MEDIUM** (needs review) from lip-sync; none HIGH. AVLips reals are TV-style clips with their own offsets/cuts.
5. Lip-sync was "not assessed" for every OEP student (written exam, little speech): an honest status, not a failure.

Thresholds were **not** re-tuned on these results (that would fit the test set). Closing gap 3 needs a dedicated lip-forgery detector
(LipFD, stretch task X-2), not lower thresholds.
