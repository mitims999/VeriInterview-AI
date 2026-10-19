"""
Requirement R1 / task F-4 (dataset part): does the Deepfake Sub-Score separate
real-but-stressed people from real fakes?

Groups (all public data, see docs/m3/M3_DATA.md):
  stressed_real  OEP students under exam pressure (webcam .avi + separate .wav, muxed, first --oep-seconds)
  avlips_real    AVLips real talking-head clips
  avlips_fake    AVLips lip-sync fakes (MakeItTalk / Wav2Lip / TalkLip / SadTalker)
  dftimit_fake   DeepfakeTIMIT face-swaps (fram1's face on fadg0's video etc.), checked against a
                 reference photo of the REAL person whose video it is (VidTIMIT frame)

Full pipeline per clip: video_intake.analyze_video (identity + SyncNet + eye cues; no YOLO,
which is not a deepfake input) -> adapters -> deepfake_fusion. Rows are appended to
docs/m3/calibration/nervous_vs_fake.csv (numbers only). Resumable.

    python scripts/eval_nervous_vs_fake.py [--oep-seconds 180] [--n-avlips 30] [--summary-only]
"""
import argparse
import csv
import glob
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "docs", "m3", "calibration", "nervous_vs_fake.csv")
FIELDS = ["group", "clip", "deepfake_level", "deepfake_score", "deepfake_conf", "identity", "sync", "eye",
          "av_offset", "lse_c", "stability", "blink_rate", "mouth_mismatch", "ref_sim", "seconds"]


def ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def oep_items(seconds):
    for d in sorted(glob.glob(os.path.join(D, "oep", "extracted", "subject*")), key=lambda x: int(x.split("subject")[-1])):
        vids, wavs = glob.glob(os.path.join(d, "*1.avi")), glob.glob(os.path.join(d, "*.wav"))
        if vids and wavs:
            yield "stressed_real", os.path.basename(d), {"video": vids[0], "wav": wavs[0], "seconds": seconds}


AVLIPS_SAMPLE = os.path.join(D, "avlips", "eval_sample")
_AVLIPS_N = {"n": 30}


def extract_avlips_sample():
    """One sequential pass over the .tar.xz pulling every sampled member (seed 0)."""
    wanted = {spec["tar_member"] for _, _, spec in avlips_items(_AVLIPS_N["n"])}
    os.makedirs(AVLIPS_SAMPLE, exist_ok=True)
    missing = {m for m in wanted if not os.path.exists(os.path.join(AVLIPS_SAMPLE, m.replace("/", "__")))}
    if not missing:
        return
    print(f"extracting {len(missing)} AVLips clips in one pass...", flush=True)
    with tarfile.open(os.path.join(D, "avlips", "AVLips v1.0.tar.xz"), "r|xz") as tar:   # streaming mode
        for m in tar:
            if m.name in missing:
                with tar.extractfile(m) as src, open(os.path.join(AVLIPS_SAMPLE, m.name.replace("/", "__")), "wb") as dst:
                    dst.write(src.read())
                missing.discard(m.name)
                if not missing:
                    break


def avlips_items(n):
    tar = tarfile.open(os.path.join(D, "avlips", "AVLips v1.0.tar.xz"))
    names = tar.getnames()
    random.seed(0)
    for label, group in (("0_real", "avlips_real"), ("1_fake", "avlips_fake")):
        mp4 = sorted(x for x in names if x.startswith(f"AVLips/{label}/") and x.endswith(".mp4"))
        for x in random.sample(mp4, n):
            yield group, os.path.basename(x), {"tar_member": x}


def dftimit_items():
    tar = tarfile.open(os.path.join(D, "deepfaketimit", "DeepfakeTIMIT.tar.gz"))
    seen = set()
    for x in sorted(tar.getnames()):
        if "/higher_quality/" in x and x.endswith(".avi"):
            person = x.split("/")[2]
            if person not in seen:                         # one clip per target person
                seen.add(person)
                yield "dftimit_fake", f"{person}/{os.path.basename(x)}", {"dftimit_member": x, "person": person}


def vidtimit_photo(person):
    import cv2
    import numpy as np
    z = zipfile.ZipFile(os.path.join(D, "vidtimit", f"{person}.zip"))
    frames = sorted(n for n in z.namelist() if "/video/sx" in n and not n.endswith("/"))
    return cv2.imencode(".png", cv2.imdecode(np.frombuffer(z.read(frames[len(frames) // 2]), np.uint8), cv2.IMREAD_COLOR))[1].tobytes()


def prepare(spec, tmp):
    """Return (path to a playable clip with audio, reference photo bytes or None)."""
    if "video" in spec:
        out = os.path.join(tmp, "oep.mp4")
        subprocess.run([ff(), "-v", "error", "-y", "-i", spec["video"], "-i", spec["wav"], "-t", str(spec["seconds"]),
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-ar", "16000", out], check=True)
        return out, None
    if "tar_member" in spec:
        # .tar.xz cannot seek: random member access re-decompresses from the start (6-67 min per clip).
        # Extract all sampled members in ONE sequential pass into datasets/avlips/eval_sample/ first.
        cached = os.path.join(AVLIPS_SAMPLE, spec["tar_member"].replace("/", "__"))
        if not os.path.exists(cached):
            extract_avlips_sample()
        return cached, None
    tar = tarfile.open(os.path.join(D, "deepfaketimit", "DeepfakeTIMIT.tar.gz"))
    out = os.path.join(tmp, "df.avi")
    with tar.extractfile(spec["dftimit_member"]) as src, open(out, "wb") as dst:
        dst.write(src.read())
    wav = spec["dftimit_member"][:-4] + ".wav"
    if wav in tar.getnames():
        wav_out = os.path.join(tmp, "df.wav")
        with tar.extractfile(wav) as src, open(wav_out, "wb") as dst:
            dst.write(src.read())
        muxed = os.path.join(tmp, "df.mp4")
        subprocess.run([ff(), "-v", "error", "-y", "-i", out, "-i", wav_out, "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-ar", "16000", "-shortest", muxed], check=True)
        out = muxed
    return out, vidtimit_photo(spec["person"])


def summarize():
    import numpy as np
    rows = list(csv.DictReader(open(OUT)))
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    print("| group | n | LOW | MEDIUM | HIGH | UNAVAILABLE | mean score | identity HIGH | sync HIGH | eye MEDIUM+ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g, rs in groups.items():
        lv = [r["deepfake_level"] for r in rs]
        sc = [float(r["deepfake_score"]) for r in rs]
        print(f"| {g} | {len(rs)} | {lv.count('LOW')} | {lv.count('MEDIUM')} | {lv.count('HIGH')} | {lv.count('UNAVAILABLE')} | "
              f"{np.mean(sc):.1f} | {sum(r['identity'] == 'HIGH' for r in rs)} | {sum(r['sync'] == 'HIGH' for r in rs)} | "
              f"{sum(r['eye'] in ('MEDIUM', 'HIGH') for r in rs)} |")

    def auc(pos, neg):
        pos, neg = np.array(pos), np.array(neg)
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

    stressed = [float(r["deepfake_score"]) for r in groups.get("stressed_real", [])]
    for fake_group in ("dftimit_fake", "avlips_fake"):
        fakes = [float(r["deepfake_score"]) for r in groups.get(fake_group, [])]
        if stressed and fakes:
            print(f"AUC stressed_real vs {fake_group}: {auc(fakes, stressed):.3f}")
    if groups.get("avlips_real") and groups.get("avlips_fake"):
        print(f"AUC avlips_real vs avlips_fake: {auc([float(r['deepfake_score']) for r in groups['avlips_fake']], [float(r['deepfake_score']) for r in groups['avlips_real']]):.3f}")
    print(f"R1 target - stressed_real scored HIGH: {sum(r['deepfake_level'] == 'HIGH' for r in groups.get('stressed_real', []))} (must be 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oep-seconds", type=int, default=180)
    ap.add_argument("--n-avlips", type=int, default=30)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        summarize()
        return
    from backend.adapters import insightface_adapter as ia, syncnet_adapter as sa, video_intake as vi
    from backend.modules.deepfake_fusion import fuse_deepfake
    done = {(r["group"], r["clip"]) for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        _AVLIPS_N["n"] = args.n_avlips
        extract_avlips_sample()
        items = list(dftimit_items()) + list(oep_items(args.oep_seconds)) + list(avlips_items(args.n_avlips))
        for group, clip, spec in items:
            if (group, clip) in done:
                continue
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                path, photo = prepare(spec, tmp)
                r = vi.analyze_video(path, reference_photo=photo, use_cache=False, run_objects=False)
            idm, sm = ia.compute(r["identity_signals"]), sa.compute(r["av_signals"])
            df = fuse_deepfake(idm, sm, r["eye_authenticity"])
            sub, av, eye = df.details["sub_scores"], r["av_signals"], r["eye_authenticity"]
            w.writerow({"group": group, "clip": clip, "deepfake_level": df.level, "deepfake_score": df.score,
                        "deepfake_conf": df.confidence, "identity": sub["identity"]["level"], "sync": sub["sync"]["level"],
                        "eye": sub["eye_authenticity"]["level"], "av_offset": av.get("av_offset_frames", ""),
                        "lse_c": av.get("lse_c", ""), "stability": av.get("offset_stability", ""),
                        "blink_rate": eye.get("blink_rate_per_min", ""), "mouth_mismatch": eye.get("mouth_speech_mismatch_pct", ""),
                        "ref_sim": r["identity_signals"].get("reference_similarity", ""), "seconds": round(time.time() - t0, 1)})
            f.flush()
            print(f"{group} {clip}: {df.level} {df.score} (id {sub['identity']['level']}, sync {sub['sync']['level']}, "
                  f"eye {sub['eye_authenticity']['level']}) {time.time() - t0:.0f}s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
