"""
Evaluates the gaze analyzer (backend.vision.gaze) on the OEP online-exam
proctoring dataset (M3 task P-3). Compares the share of off-screen frames
inside vs outside the labelled cheating intervals, per cheat type and per
subject. Writes docs/m3/calibration/gaze_oep.csv (numbers only). Resumable.

    python scripts/eval_gaze_oep.py [--fps 3] [--max-minutes 10] [--summary-only]

Needs datasets/oep/extracted/<subject>/{<user>1.avi, gt.txt} (webcam video +
labels; see docs/m3/M3_DATA.md). gt.txt rows: "MMSS MMSS type". Types (paper):
1 book/notes, 2 talk to a person in the room, 3 Internet, 4 phone call, 5 phone
use; 6 appears in the labels but is undocumented.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OEP = os.path.join(ROOT, "datasets", "oep", "extracted")
OUT = os.path.join(ROOT, "docs", "m3", "calibration", "gaze_oep.csv")
ACTORS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 20, 21, 22, 23, 24}


def read_gt(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            to_s = lambda x: int(x[:-2] or 0) * 60 + int(x[-2:])
            rows.append((to_s(p[0]), to_s(p[1]), int(p[2])))
    return rows


def label_at(t, gt):
    for s, e, k in gt:
        if s <= t <= e:
            return k
    return 0


def summarize():
    rows = list(csv.DictReader(open(OUT)))
    print("| subject | type | minutes | coverage | off-screen % no-cheat | off-screen % cheating | by cheat type (off %) |")
    print("|---|---|---|---|---|---|---|")
    agg = {"no": [], "cheat": [], "by_type": {}}
    for r in rows:
        bt = json.loads(r["off_by_type"])
        print(f"| {r['subject']} | {r['kind']} | {float(r['minutes']):.1f} | {float(r['coverage']):.2f} | "
              f"{float(r['off_nocheat']) * 100:.1f} | {float(r['off_cheat']) * 100 if r['off_cheat'] else float('nan'):.1f} | "
              f"{', '.join(f'{k}: {v * 100:.0f}' for k, v in sorted(bt.items()))} |")
        agg["no"].append(float(r["off_nocheat"]))
        if r["off_cheat"]:
            agg["cheat"].append(float(r["off_cheat"]))
        for k, v in bt.items():
            agg["by_type"].setdefault(k, []).append(v)
    print(f"\nMean over subjects: off-screen {np.mean(agg['no']) * 100:.1f}% when NOT cheating vs "
          f"{np.mean(agg['cheat']) * 100:.1f}% when cheating. Subjects with higher off-screen while cheating: "
          f"{sum(1 for r in rows if r['off_cheat'] and float(r['off_cheat']) > float(r['off_nocheat']))}/{sum(1 for r in rows if r['off_cheat'])}")
    print("By cheat type (mean off-screen %): " + ", ".join(f"type {k}: {np.mean(v) * 100:.0f}% (n={len(v)})"
                                                       for k, v in sorted(agg["by_type"].items())))
    nc = np.array(agg["no"])
    print(f"Normal (non-cheating) off-screen rate across subjects: median {np.median(nc) * 100:.1f}%, "
          f"p90 {np.percentile(nc, 90) * 100:.1f}%, max {nc.max() * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-minutes", type=float, default=10.0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summarize()
        return
    from backend.vision import gaze, landmarks, media
    done = {r["subject"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["subject", "kind", "minutes", "coverage", "off_nocheat", "off_cheat", "off_by_type", "head_variance", "seconds"]
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for d in sorted(glob.glob(os.path.join(OEP, "subject*")), key=lambda x: int(x.split("subject")[-1])):
            subj = os.path.basename(d)
            vids = glob.glob(os.path.join(d, "*1.avi"))
            if subj in done or not vids or not os.path.exists(os.path.join(d, "gt.txt")):
                continue
            t0 = time.time()
            gt = read_gt(os.path.join(d, "gt.txt"))
            stream = [(fr.t, o, fr.image.shape[:2]) for fr, o in
                      landmarks.iter_landmarks(media.iter_frames(vids[0], fps=args.fps, max_width=640, end=args.max_minutes * 60))]
            states = gaze.debounce([gaze.frame_state(o, size) for t, o, size in stream if o is not None],
                                   gaze.GAZE_THRESHOLDS.min_off_screen_s, args.fps)
            lab = np.array([label_at(s["t"], gt) for s in states])
            off = np.array([s["off"] for s in states])
            by_type = {str(k): round(float(off[lab == k].mean()), 3) for k in sorted(set(lab[lab > 0]))}
            summ = gaze.analyze(stream, args.fps)
            w.writerow({"subject": subj, "kind": "actor" if int(subj[7:]) in ACTORS else "real_exam",
                        "minutes": round(len(stream) / args.fps / 60, 2), "coverage": round(len(states) / max(1, len(stream)), 3),
                        "off_nocheat": round(float(off[lab == 0].mean()), 3) if (lab == 0).any() else "",
                        "off_cheat": round(float(off[lab > 0].mean()), 3) if (lab > 0).any() else "",
                        "off_by_type": json.dumps(by_type), "head_variance": summ.get("head_pose_variance"),
                        "seconds": round(time.time() - t0, 1)})
            f.flush()
            print(f"{subj}: {time.time() - t0:.0f}s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
