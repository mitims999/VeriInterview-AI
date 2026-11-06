# Identity threshold calibration on LFW (tasks I-1, I-4)

Production pipeline (SCRFD det_size 320 → largest centred face → ArcFace buffalo_l, L2-normalised) on both images of all 6,000 official LFW pairs (`pairs.txt`, 10 folds × 300 same + 300 different). Per-pair cosines: `identity_lfw_scores.csv` (numbers only). Reproduce: `python scripts/calibrate_identity_lfw.py` (~35 min CPU); summary: `--summary-only`.

- Faces found in every image (0 pairs dropped).
- Genuine cosine: mean 0.672, median 0.678, p5 0.498, p1 0.395.
- Impostor cosine: mean 0.004, p99 0.141, p99.9 0.205, **max 0.233**.
- Best balanced-accuracy threshold 0.234 (accuracy 99.87 %).

| threshold | false accept | false reject |
|---|---|---|
| 0.20 | 0.20 % | 0.27 % |
| 0.25 | 0.00 % | 0.27 % |
| **0.30 (chosen)** | **0.00 %** | **0.40 %** |
| 0.35 | 0.00 % | 0.60 % |
| 0.40 | 0.00 % | 1.17 % |

**Chosen:** `same_person_cosine = reference_match_cosine = 0.30`. It sits above the highest impostor score (0.233) with margin, and rejects only 0.4 % of genuine LFW pairs (different photos, years, lighting: harder than one live session, where genuine frames score 0.93–0.96 vs the anchor on GRID). The reference "weak match" band is 0.30–0.40 (+0.10 margin): 1.2 % of genuine LFW pairs land there and get a MEDIUM note that mentions old or low-quality photos, never HIGH.

Real-data checks: GRID s1→s4 and s1→s2 splices drop to −0.10/−0.01 at the cut. DeepfakeTIMIT fake (fram1 face on fadg0 video) vs a real fadg0 photo scores 0.223 → mismatch.
