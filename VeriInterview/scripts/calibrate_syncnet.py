"""
SyncNet calibration on the generated GRID test set (M3 task S-3).

Runs backend.vision.syncnet_runner.analyze on every sync-relevant clip in
datasets/m3_generated/manifest.csv and appends one row per clip to
docs/m3/calibration/syncnet_grid.csv (numbers only). Resumable.

    python scripts/build_m3_testset.py      # first
    python scripts/calibrate_syncnet.py [--summary-only]
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GEN = os.path.join(ROOT, "datasets", "m3_generated")
OUT = os.path.join(ROOT, "docs", "m3", "calibration", "syncnet_grid.csv")
SYNC_CLIPS = {"genuine", "av_shift", "av_shift_partial", "audio_swap", "silent_segment", "low_quality"}
FIELDS = ["clip", "talker", "manipulation", "params", "status", "av_offset_frames", "offset_stability", "lse_c",
          "lse_d", "speech_s", "per_segment_offsets", "per_segment_lse_c", "seconds"]


def summarize():
    import numpy as np
    rows = list(csv.DictReader(open(OUT)))
    base = {r["talker"]: int(r["av_offset_frames"]) for r in rows if r["manipulation"] == "genuine" and r["av_offset_frames"]}
    print("| clip type | n | offset (abs, mean) | offset - talker baseline (mean) | stability (mean) | LSE-C (mean, min) |")
    print("|---|---|---|---|---|---|")
    groups = {}
    for r in rows:
        key = r["manipulation"] + (f" {r['params']}" if r["manipulation"] == "av_shift" else "")
        groups.setdefault(key, []).append(r)
    for k in sorted(groups):
        g = [r for r in groups[k] if r["av_offset_frames"]]
        if not g:
            print(f"| {k} | {len(groups[k])} | status: {groups[k][0]['status']} | | | |")
            continue
        off = np.array([int(r["av_offset_frames"]) for r in g])
        rel = np.array([int(r["av_offset_frames"]) - base.get(r["talker"], 0) for r in g])
        st = np.array([float(r["offset_stability"]) for r in g])
        lc = np.array([float(r["lse_c"]) for r in g])
        print(f"| {k} | {len(g)} | {np.mean(np.abs(off)):.1f} | {rel.mean():+.1f} | {st.mean():.2f} | {lc.mean():.2f}, {lc.min():.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summarize()
        return
    from backend.vision import face_track, syncnet_runner
    app = face_track.get_face_app()
    manifest = [r for r in csv.DictReader(open(os.path.join(GEN, "manifest.csv"))) if r["manipulation"] in SYNC_CLIPS]
    done = {r["clip"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for m in manifest:
            if m["clip"] in done:
                continue
            t0 = time.time()
            s = syncnet_runner.analyze(os.path.join(GEN, m["clip"]), app=app)
            segs = [x for x in s.get("per_segment", []) if "offset" in x]
            w.writerow({"clip": m["clip"], "talker": m["talker"], "manipulation": m["manipulation"], "params": m["params"],
                        "status": s.get("status_reason") or "ok", "av_offset_frames": s.get("av_offset_frames", ""),
                        "offset_stability": s.get("offset_stability", ""), "lse_c": s.get("lse_c", ""),
                        "lse_d": s.get("lse_d", ""), "speech_s": s.get("speech_seconds_analyzed", ""),
                        "per_segment_offsets": json.dumps([x["offset"] for x in segs]),
                        "per_segment_lse_c": json.dumps([x["lse_c"] for x in segs]),
                        "seconds": round(time.time() - t0, 1)})
            f.flush()
            print(f"{m['clip']}: offset={s.get('av_offset_frames')} lse_c={s.get('lse_c')} ({time.time() - t0:.0f}s)", flush=True)
    summarize()


if __name__ == "__main__":
    main()
