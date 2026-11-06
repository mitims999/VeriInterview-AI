"""
Evaluates backend.vision.media.detect_speech against GRID word alignments
(M3 task G-3). Ground truth = first to last non-silence word per sentence.

    python scripts/eval_vad_grid.py [--n 100] [--kwargs '{"rel_threshold": 0.55}']

Needs datasets/grid/{video,align_shef}/ (see docs/m3/M3_DATA.md). Use the
Sheffield alignments: the Zenodo alignments.zip has mismatched talker folders
(s15/s20 names do not match the videos). GRID .align units are 1/25000 s.
Prints a per-talker table. No media is written outside a temp dir.
"""
import argparse
import collections
import json
import os
import random
import sys
import tarfile
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.vision import media  # noqa: E402

TALKERS = ["s1", "s2", "s3", "s4", "s15", "s20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="clips per talker")
    ap.add_argument("--kwargs", default="{}", help="JSON kwargs for detect_speech")
    args = ap.parse_args()
    kw = json.loads(args.kwargs)

    grid = os.path.join(ROOT, "datasets", "grid")
    align = {}
    for t in TALKERS:
        tf = tarfile.open(os.path.join(grid, "align_shef", f"{t}.tar"))
        for m in tf.getmembers():
            if m.name.endswith(".align"):
                align[(t, os.path.basename(m.name)[:-6])] = tf.extractfile(m).read().decode()

    random.seed(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in TALKERS:
            z = zipfile.ZipFile(os.path.join(grid, "video", f"{t}.mpg_vcd.zip"))
            for name in random.sample([n for n in z.namelist() if n.endswith(".mpg")], args.n):
                stem = os.path.basename(name)[:-4]
                words = [l.split() for l in align.get((t, stem), "").splitlines() if l.strip()]
                spoken = [(int(a) / 25000, int(b) / 25000) for a, b, w in words if w not in ("sil", "sp")]
                if not spoken:
                    continue
                gt = (spoken[0][0], spoken[-1][1])
                path = os.path.join(tmp, f"{t}_{stem}.mpg")
                with open(path, "wb") as f:
                    f.write(z.read(name))
                audio = media.load_audio(path)
                segs = media.detect_speech(audio, **kw)
                os.remove(path)
                ts = np.arange(0, audio.size / 16000, 0.01)
                g = (ts >= gt[0]) & (ts < gt[1])
                p = np.zeros_like(g)
                for s in segs:
                    p |= (ts >= s.start) & (ts < s.end)
                est = (segs[0].start, segs[-1].end) if segs else (np.nan, np.nan)
                rows.append({"talker": t, "acc": (g == p).mean(), "recall": (g & p).sum() / max(1, g.sum()),
                             "precision": (g & p).sum() / max(1, p.sum()),
                             "start_err": abs(est[0] - gt[0]), "end_err": abs(est[1] - gt[1]), "missed": not segs})

    by = collections.defaultdict(list)
    for r in rows:
        by[r["talker"]].append(r)
    print(f"kwargs={kw}")
    print("| talker | n | accuracy | recall | precision | median start err (s) | median end err (s) | missed |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in list(by.items()) + [("ALL", rows)]:
        m = lambda f: np.nanmean([r[f] for r in v])
        md = lambda f: np.nanmedian([r[f] for r in v])
        print(f"| {k} | {len(v)} | {m('acc'):.3f} | {m('recall'):.3f} | {m('precision'):.3f} | "
              f"{md('start_err'):.3f} | {md('end_err'):.3f} | {sum(r['missed'] for r in v)} |")


if __name__ == "__main__":
    main()
