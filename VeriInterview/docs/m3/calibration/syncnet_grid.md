# SyncNet calibration on the generated GRID set (tasks S-3, S-4)

`scripts/calibrate_syncnet.py` ran `syncnet_runner.analyze` on 66 clips (6 talkers × 11 sync-relevant manipulations, ~36 s each, up to 60 s of speech per clip). Raw rows: `syncnet_grid.csv` (numbers only). Reproduce: `python scripts/build_m3_testset.py && python scripts/calibrate_syncnet.py`.

## Measurements

| clip type | n | offset (abs, mean) | offset - talker baseline (mean) | stability (mean) | LSE-C (mean, min) |
|---|---|---|---|---|---|
| audio_swap | 6 | 6.5 | -1.7 | 0.47 | 1.24, 1.01 |
| av_shift frames=-10 | 6 | 11.2 | +10.0 | 1.00 | 6.12, 5.02 |
| av_shift frames=-2 | 6 | 3.3 | +2.2 | 1.00 | 5.91, 4.52 |
| av_shift frames=-5 | 6 | 6.2 | +5.0 | 1.00 | 5.87, 4.50 |
| av_shift frames=10 | 6 | 8.7 | -9.8 | 0.83 | 6.15, 4.59 |
| av_shift frames=2 | 6 | 0.5 | -1.7 | 0.92 | 5.86, 4.47 |
| av_shift frames=5 | 6 | 3.7 | -4.8 | 0.96 | 5.94, 4.49 |
| av_shift_partial | 6 | 4.3 | -4.8 | 0.54 | 5.04, 3.53 |
| genuine | 6 | 1.2 | +0.0 | 1.00 | 5.88, 4.54 |
| low_quality | 6 | 1.0 | -0.2 | 1.00 | 2.72, 1.17 |
| silent_segment | 6 | 1.2 | +0.0 | 0.96 | 5.93, 4.94 |

- Shifts are recovered to within ~1 frame, relative to each talker's genuine baseline (GRID genuine baseline ≈ +1 frame). Sign: audio delayed → negative offset.
- Audio swap: LSE-C 1.01–1.57 **and** offset stability 0.25–0.75. Low-quality 240p genuine: LSE-C 1.17–5.39 **but** stability 1.00 → stability separates the two.
- Partial shift: the windows inside the manipulated third read −6/−7 frames while the others read 0–2.

## Thresholds chosen (config.SyncThresholds)

bonafide_offset_frames 2, high_offset_frames 6, lse_c_good 6.0 (GRID genuine mean 5.88), lse_c_low 2.0, min_offset_stability 0.6.

## Adapter outcomes on the same 66 clips (`syncnet_adapter.compute`)

| clip type | outcomes (6 talkers) |
|---|---|
| genuine | 6× LOW (10–15) |
| silent_segment | 6× LOW |
| low_quality 240p | 6× LOW, confidence 0.20–0.90 (not risk; R1) |
| audio_swap | 6× HIGH (65–90) |
| av_shift_partial | 6× HIGH (61), the manipulated windows flagged with timestamps |
| shift ±10 | 12× HIGH |
| shift −5 / +5 | 6× HIGH / 6× MEDIUM |
| shift +2 / −2 | 6× LOW / 6× MEDIUM |

The ±2 asymmetry comes from GRID's +1 baseline: −2 lands at absolute 3 frames (~120 ms, perceptible), so it gets a single-module MEDIUM. One module can never set the final risk alone.
