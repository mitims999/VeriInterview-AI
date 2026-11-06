"""
Phone / person detection on real webcam exam footage (M3 task P-5/P-6 check).

For every OEP subject with labelled phone-use intervals (type 5), samples
frames at 1 fps inside those intervals plus the same number of frames from
non-cheating time, and reports how often backend.vision.objects.detect sees a
phone / a second person. Prints a table (numbers only).

    python scripts/eval_objects_oep.py [--per-subject 40]
"""
import argparse
import glob
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.eval_gaze_oep import OEP, label_at, read_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=40)
    args = ap.parse_args()
    from backend.vision import media, objects
    random.seed(0)
    print("| subject | phone-use frames | phone detected | extra person | normal frames | phone detected | extra person |")
    print("|---|---|---|---|---|---|---|")
    tot = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "pm": 0, "nm": 0}
    for d in sorted(glob.glob(os.path.join(OEP, "subject*")), key=lambda x: int(x.split("subject")[-1])):
        gt = read_gt(os.path.join(d, "gt.txt"))
        phone_iv = [(s, e) for s, e, k in gt if k == 5]
        vids = glob.glob(os.path.join(d, "*1.avi"))
        if not phone_iv or not vids:
            continue
        info = media.probe(vids[0])
        pos = [t for s, e in phone_iv for t in range(s, e + 1) if t < info.duration]
        neg = [t for t in range(0, int(info.duration)) if label_at(t, gt) == 0]
        pos = random.sample(pos, min(args.per_subject, len(pos)))
        neg = random.sample(neg, min(len(pos), len(neg)))

        def rate(times):
            ph = mp = 0
            for t in times:
                f = next(media.iter_frames(vids[0], start=float(t), end=float(t) + 0.2, info=info))
                dets = objects.detect(f.image)
                ph += any(x["cls"] == objects.PHONE for x in dets)
                mp += sum(x["cls"] == objects.PERSON for x in dets) >= 2
            return ph, mp

        pp, pm = rate(pos)
        np_, nm = rate(neg)
        tot["pp"] += pp; tot["pn"] += len(pos); tot["np"] += np_; tot["nn"] += len(neg); tot["pm"] += pm; tot["nm"] += nm
        print(f"| {os.path.basename(d)} | {len(pos)} | {pp / max(1, len(pos)) * 100:.0f}% | {pm / max(1, len(pos)) * 100:.0f}% | "
              f"{len(neg)} | {np_ / max(1, len(neg)) * 100:.0f}% | {nm / max(1, len(neg)) * 100:.0f}% |", flush=True)
    print(f"\nPhone visible in {tot['pp']}/{tot['pn']} phone-use frames ({tot['pp'] / max(1, tot['pn']) * 100:.0f}%) "
          f"vs {tot['np']}/{tot['nn']} normal frames ({tot['np'] / max(1, tot['nn']) * 100:.1f}%). "
          f"Extra person: {tot['pm']} in phone-use frames, {tot['nm']} in normal frames.")


if __name__ == "__main__":
    main()
