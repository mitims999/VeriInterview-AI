# VAD calibration on GRID (task G-3)

Ground truth: GRID Sheffield word alignments (first to last non-silence word). 100 random sentences × 6 talkers (s1, s2, s3, s4, s15, s20). Reproduce with `python scripts/eval_vad_grid.py --n 100` (random seed 0).

Chosen setting: `detect_speech(rel_threshold=0.55)` (default). Sweep on 240 clips: 0.35 → acc 0.866 / prec 0.842; 0.50 → 0.927 / 0.934; 0.60 → 0.940 / 0.964; 0.70 → 0.924 / 0.984 (recall drops to 0.889). 0.55 favours precision, because false "speech" in silent frames could later create false mouth-vs-speech flags against genuine candidates (R1).

| talker | n | accuracy | recall | precision | median start err (s) | median end err (s) | missed |
|---|---|---|---|---|---|---|---|
| s1 | 100 | 0.939 | 0.932 | 0.956 | 0.035 | 0.030 | 0 |
| s2 | 100 | 0.955 | 0.944 | 0.986 | 0.020 | 0.050 | 0 |
| s3 | 100 | 0.965 | 0.974 | 0.978 | 0.030 | 0.040 | 0 |
| s4 | 100 | 0.939 | 0.972 | 0.930 | 0.010 | 0.105 | 0 |
| s15 | 100 | 0.897 | 0.980 | 0.888 | 0.030 | 0.100 | 0 |
| s20 | 100 | 0.919 | 0.924 | 0.941 | 0.025 | 0.150 | 0 |
| ALL | 600 | 0.936 | 0.954 | 0.947 | 0.020 | 0.060 | 0 |
