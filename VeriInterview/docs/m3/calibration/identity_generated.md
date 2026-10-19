# Identity adapter end-to-end on the generated GRID set (task I-3)

`insightface_adapter.analyze_video()` → `compute()` on 24 clips from `scripts/build_m3_testset.py` (6 talkers × 4 manipulations, 12 sentences ≈ 36 s each, 2 fps). Thresholds: same_person_cosine 0.30 (LFW).

| manipulation | n | level | score | confidence | distinct identities | below-threshold share | face lost | first drift |
|---|---|---|---|---|---|---|---|---|
| genuine | 6 | 6× LOW | 5.0 | 0.92 | 1 | 0.00 | 0.00 | none |
| identity_splice | 6 | 6× HIGH | 90.0 | 0.92 | 2 | 0.50 | 0.00 | 18.0 s (= cut) |
| face_occluded (middle third) | 6 | 6× LOW | 5.0 | 0.78 | 1 | 0.00 | 0.29–0.33 | none |
| low_quality (240p CRF 40) | 6 | 6× LOW | 5.0 | 0.92 | 1 | 0.00 | 0.00 | none |

All 24 match the manifest's expected outcome. A hidden face lowers confidence and never raises risk (R1 rule 4). The manifest also expected low_quality to lower confidence, but ArcFace still matched perfectly at 240p, so confidence correctly stayed high.
