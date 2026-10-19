"""
Calibrates the ArcFace "same person" cosine threshold on LFW (M3 task I-1).

Runs the production face pipeline (backend.vision.face_track: SCRFD detect,
largest face, ArcFace embedding) on both images of each official LFW pair
(pairs.txt, 10 folds x 300 same + 300 different) and writes per-pair cosine
scores to docs/m3/calibration/identity_lfw_scores.csv (numbers only: no
images, no embeddings). Then prints genuine/impostor statistics and the
threshold at several false-accept rates.

    python scripts/calibrate_identity_lfw.py [--limit-folds 10]

Needs datasets/lfw/lfw_home/{pairs.txt,lfw_funneled/} (see docs/m3/M3_DATA.md).
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.vision import face_track as ft  # noqa: E402
from backend.vision.media import Frame  # noqa: E402

LFW = os.path.join(ROOT, "datasets", "lfw", "lfw_home")
OUT = os.path.join(ROOT, "docs", "m3", "calibration", "identity_lfw_scores.csv")


def img_path(name, idx):
    return os.path.join(LFW, "lfw_funneled", name, f"{name}_{int(idx):04d}.jpg")


def read_pairs(limit_folds):
    lines = open(os.path.join(LFW, "pairs.txt")).read().splitlines()
    folds, per = (int(x) for x in lines[0].split())
    pairs, i = [], 1
    for fold in range(folds):
        for same in (1, 0):
            for _ in range(per):
                p = lines[i].split()
                i += 1
                if fold >= limit_folds:
                    continue
                if same:
                    pairs.append((fold, 1, img_path(p[0], p[1]), img_path(p[0], p[2])))
                else:
                    pairs.append((fold, 0, img_path(p[0], p[1]), img_path(p[2], p[3])))
    return pairs


def summarize(rows):
    g = np.array([float(r["cosine"]) for r in rows if int(r["same"]) == 1 and r["cosine"] != ""])
    im = np.array([float(r["cosine"]) for r in rows if int(r["same"]) == 0 and r["cosine"] != ""])
    print(f"pairs scored: genuine={g.size} impostor={im.size} (no face in either image: "
          f"{sum(r['cosine'] == '' for r in rows)})")
    print(f"genuine  cosine: mean {g.mean():.3f}  p1 {np.percentile(g, 1):.3f}  p5 {np.percentile(g, 5):.3f}  median {np.median(g):.3f}")
    print(f"impostor cosine: mean {im.mean():.3f}  p99 {np.percentile(im, 99):.3f}  p99.9 {np.percentile(im, 99.9):.3f}  max {im.max():.3f}")
    print("| threshold | false accept (impostor >= thr) | false reject (genuine < thr) |")
    print("|---|---|---|")
    for thr in (0.20, 0.25, 0.28, 0.30, 0.35, 0.40):
        print(f"| {thr:.2f} | {np.mean(im >= thr):.4f} | {np.mean(g < thr):.4f} |")
    best = max(np.linspace(0, 1, 1001), key=lambda t: np.mean(g >= t) / 2 + np.mean(im < t) / 2)
    print(f"best balanced-accuracy threshold: {best:.3f} "
          f"(accuracy {(np.mean(g >= best) + np.mean(im < best)) / 2:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-folds", type=int, default=10)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        summarize(list(csv.DictReader(open(OUT))))
        return

    app = ft.get_face_app()
    cache = {}

    def embed(path):
        if path not in cache:
            img = cv2.imread(path)
            faces = ft.detect(Frame(0, 0.0, img), app, embed=True) if img is not None else []
            # LFW images are centred on the labelled person: take the face nearest the centre
            if faces:
                cx, cy = img.shape[1] / 2, img.shape[0] / 2
                faces.sort(key=lambda f: (f.bbox[0] + f.bbox[2] - 2 * cx) ** 2 + (f.bbox[1] + f.bbox[3] - 2 * cy) ** 2)
            cache[path] = faces[0].embedding if faces else None
        return cache[path]

    pairs = read_pairs(args.limit_folds)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # resumable: rows are appended as they are scored, so an interrupted run continues where it stopped
    done = sum(1 for _ in open(OUT)) - 1 if os.path.exists(OUT) else 0
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "fold", "same", "cosine"])
        if done <= 0:
            done = 0
            w.writeheader()
        for k, (fold, same, a, b) in enumerate(pairs):
            if k < done:
                continue
            ea, eb = embed(a), embed(b)
            cos = "" if ea is None or eb is None else round(float(np.dot(ea, eb)), 4)
            w.writerow({"pair": k, "fold": fold, "same": same, "cosine": cos})
            if k % 100 == 0:
                f.flush()
                print(f"{k}/{len(pairs)}", flush=True)
    summarize(list(csv.DictReader(open(OUT))))


if __name__ == "__main__":
    main()
