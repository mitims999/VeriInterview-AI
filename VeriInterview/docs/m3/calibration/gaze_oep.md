# Gaze on OEP online-exam proctoring (task P-3)

`scripts/eval_gaze_oep.py`: first 10 min of each of the 24 OEP webcam videos at 3 fps, MediaPipe landmarks → `gaze.frame_state` → 0.5 s debounce. Compared with the labelled cheating intervals (`gt.txt`). Per-subject rows: `gaze_oep.csv`.

| subject | type | minutes | coverage | off-screen % no-cheat | off-screen % cheating | by cheat type (off %) |
|---|---|---|---|---|---|---|
| subject1 | actor | 10.0 | 0.80 | 10.1 | 36.9 | 1: 93, 2: 20, 3: 16, 5: 2 |
| subject2 | actor | 10.0 | 1.00 | 1.0 | 13.9 | 1: 4, 3: 10, 6: 40 |
| subject3 | actor | 10.0 | 0.99 | 0.8 | 2.3 | 1: 11, 3: 0, 6: 0 |
| subject4 | actor | 10.0 | 0.90 | 9.8 | 3.2 | 1: 100, 3: 2 |
| subject5 | actor | 10.0 | 0.97 | 2.3 | 16.3 | 1: 53, 2: 0, 3: 6, 6: 27 |
| subject6 | actor | 10.0 | 0.92 | 15.1 | 10.3 | 2: 77, 3: 2, 5: 12, 6: 50 |
| subject7 | actor | 10.0 | 0.89 | 3.4 | 31.8 | 1: 42, 3: 0, 6: 65 |
| subject8 | actor | 10.0 | 0.96 | 0.0 | 1.8 | 3: 2 |
| subject9 | actor | 10.0 | 0.96 | 2.8 | 11.5 | 1: 22, 3: 0, 5: 0, 6: 19 |
| subject10 | real_exam | 10.0 | 0.81 | 9.6 | 5.7 | 1: 0, 2: 6 |
| subject11 | real_exam | 10.0 | 0.48 | 1.2 | 0.0 | 2: 0 |
| subject12 | real_exam | 10.0 | 0.66 | 2.2 | 0.0 | 2: 0 |
| subject13 | real_exam | 10.0 | 0.90 | 1.2 | 2.6 | 1: 0, 2: 3 |
| subject14 | real_exam | 10.0 | 0.98 | 0.0 | 0.9 | 1: 0, 2: 1 |
| subject15 | real_exam | 10.0 | 0.95 | 0.1 | 10.2 | 2: 10 |
| subject16 | real_exam | 10.0 | 0.79 | 20.7 | 5.0 | 1: 17, 2: 4 |
| subject17 | actor | 10.0 | 0.97 | 1.0 | 2.4 | 1: 11, 2: 0 |
| subject18 | real_exam | 10.0 | 0.95 | 1.2 | 1.7 | 2: 2 |
| subject19 | real_exam | 10.0 | 0.60 | 8.3 | 4.5 | 2: 4 |
| subject20 | actor | 10.0 | 0.97 | 1.8 | 5.6 | 1: 0, 2: 14 |
| subject21 | actor | 10.0 | 0.86 | 3.3 | 4.1 | 1: 10, 2: 1 |
| subject22 | actor | 10.0 | 1.00 | 2.2 | 19.2 | 1: 23, 2: 0 |
| subject23 | actor | 10.0 | 0.80 | 0.3 | 0.8 | 1: 0, 2: 2 |
| subject24 | actor | 10.0 | 0.85 | 2.3 | 8.0 | 1: 36, 2: 6, 5: 0 |

Mean over subjects: off-screen 4.2% when NOT cheating vs 8.3% when cheating. Subjects with higher off-screen while cheating: 17/24
By cheat type (mean off-screen %): type 1: 25% (n=17), type 2: 8% (n=18), type 3: 4% (n=9), type 5: 3% (n=4), type 6: 34% (n=6)
Normal (non-cheating) off-screen rate across subjects: median 2.2%, p90 10.0%, max 20.7%

## Reading

- Normal exam behaviour: median 2.2 % off-screen, p90 10 %, max 20.7 %. This sets the scale for 'normal' and supports the +20-point excess-over-baseline trigger for flagging a question.
- Cheating raises off-screen time in 17/24 subjects (mean 8.3 % vs 4.2 %). **Reading from a book/notes (type 1) is the clear case** (mean 25 %, up to 93–100 %). Talking to someone (2), the Internet (3, eyes stay on the screen) and phone use (5) barely change gaze.
- So gaze is real but partial evidence: it catches 'reading answers off-screen' but not cheating that keeps the eyes on screen. This is why it is weak evidence capped at MEDIUM, and why the phone/person detector (P-5) is a separate strong signal.
- Coverage (frames with landmarks) 0.48–1.00; low coverage = the student often outside the camera's view or turned far away. It lowers confidence, not risk.
