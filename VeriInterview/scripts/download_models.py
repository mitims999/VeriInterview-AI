"""
Downloads the pretrained weights used by the M3 vision adapters into
storage/models/ (gitignored — weights are never committed).

    python scripts/download_models.py            # default set (~630 MB, incl. full YOLOv3)
    python scripts/download_models.py --no-full-yolo  # skip full YOLOv3 (248 MB); falls back to YOLOv3-tiny (weak, see P-5)
    python scripts/download_models.py --check      # verify only, no downloads

Re-running is a no-op once every file is present and its SHA-256 matches.
Downloads resume after a dropped connection. Standard library only.

| Model | Used by | License |
|---|---|---|
| SyncNet v2 (joonson/syncnet_python, VGG Oxford) | lip-sync | code MIT; check weight terms with VGG |
| InsightFace buffalo_l (SCRFD-10GF + ArcFace R50) | face detect + identity | **non-commercial research only** |
| MediaPipe Face Landmarker (float16 v1) | landmarks / iris / head pose | Apache-2.0 |
| YOLOv3 (default) + YOLOv3-tiny fallback (Darknet, COCO) | person / phone | Darknet: public domain |
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "models")

# Sizes and SHA-256 hashes pinned 2026-10-14 (all files, including full YOLOv3).
MODELS = [
    {"name": "syncnet_v2.model", "dest": "syncnet/syncnet_v2.model",
     "url": "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
     "size": 54573114, "sha256": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"},
    {"name": "syncnet example.avi (reference clip for the S-1 reproduction test)", "dest": "syncnet/example.avi",
     "url": "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi",
     "size": 1111536, "sha256": "1724121b3a50141cfc9e77c034f7de914024d015b92762ae12d3e1a7151b8621"},
    {"name": "face_landmarker.task", "dest": "mediapipe/face_landmarker.task",
     "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     "size": 3758596, "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"},
    {"name": "insightface buffalo_l", "dest": "insightface/models/buffalo_l.zip",
     "url": "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
     "size": 288621354, "sha256": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", "unzip_to": "insightface/models/buffalo_l"},
    {"name": "yolov3-tiny.cfg", "dest": "yolo/yolov3-tiny.cfg",
     "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
     "size": 1915, "sha256": "84eb7a675ef87c906019ff5a6e0effe275d175adb75100dcb47f0727917dc2c7"},
    {"name": "yolov3-tiny.weights", "dest": "yolo/yolov3-tiny.weights",
     "url": "https://pjreddie.com/media/files/yolov3-tiny.weights",
     "size": 35434956, "sha256": "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c"},
]
FULL_YOLO = [
    {"name": "yolov3.cfg", "dest": "yolo/yolov3.cfg",
     "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
     "size": None, "sha256": "22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953"},
    {"name": "yolov3.weights", "dest": "yolo/yolov3.weights",
     "url": "https://pjreddie.com/media/files/yolov3.weights",
     "size": 248007048, "sha256": "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"},
]

CHUNK = 1 << 20


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _complete(m: dict, path: str) -> bool:
    if not os.path.exists(path):
        return False
    if m["size"] and os.path.getsize(path) != m["size"]:
        return False
    return not m["sha256"] or _sha256(path) == m["sha256"]


def _download(url: str, path: str, size: int | None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    for attempt in range(1, 11):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "VeriInterview-model-fetch/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while block := r.read(CHUNK):
                        f.write(block)
            if size and os.path.getsize(part) != size:
                raise IOError(f"size {os.path.getsize(part)} != expected {size}")
            os.replace(part, path)
            return
        except Exception as e:  # network drops are common; resume from .part
            print(f"    retry {attempt}/10 after error: {e}")
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"failed to download {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-full-yolo", action="store_true", help="skip full YOLOv3 (248 MB); YOLOv3-tiny is used instead (much lower recall)")
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 if anything is missing")
    args = ap.parse_args()

    models = MODELS + ([] if args.no_full_yolo else FULL_YOLO)
    missing = []
    for m in models:
        path = os.path.join(ROOT, m["dest"])
        if _complete(m, path):
            print(f"[ok]   {m['name']}")
        elif args.check:
            print(f"[MISSING] {m['name']}  -> {path}")
            missing.append(m["name"])
            continue
        else:
            print(f"[get]  {m['name']}  <- {m['url']}")
            _download(m["url"], path, m["size"])
            digest = _sha256(path)
            if m["sha256"] and digest != m["sha256"]:
                os.remove(path)
                raise SystemExit(f"SHA-256 mismatch for {m['name']} — file deleted, re-run to retry")
            print(f"       sha256 {digest}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        if m.get("unzip_to"):
            target = os.path.join(ROOT, m["unzip_to"])
            if os.path.exists(path) and not (os.path.isdir(target) and os.listdir(target)):
                with zipfile.ZipFile(path) as z:
                    for name in z.namelist():  # refuse path traversal in archive members
                        if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
                            raise SystemExit(f"unsafe path in {path}: {name}")
                    z.extractall(target)
                print(f"       extracted -> {target}")
    if missing:
        print(f"\n{len(missing)} model file(s) missing. Run: python scripts/download_models.py")
        return 1
    print(f"\nAll model files present in {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
