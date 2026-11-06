"""
Builds the controlled M3 test set from public GRID clips (CC BY 4.0) — see
docs/m3/M3_DATA.md section 4. Output: datasets/m3_generated/*.mp4 (gitignored,
real faces) + manifest.csv (clip, talker, manipulation, parameters, expected).

    python scripts/build_m3_testset.py [--talkers s1 s2 s3 s4 s15 s20] [--sentences 12]

Each "interview" = N consecutive GRID sentences of one talker (~3 s each),
re-encoded to H.264 25 fps + AAC 16 kHz. Manipulations:
  genuine            unchanged
  av_shift_{+-2,+-5,+-10}   whole audio track shifted by N frames (positive = audio delayed)
  av_shift_partial   middle third of the audio shifted by +8 frames (one manipulated "answer")
  audio_swap         video of talker A with audio of talker B (same sentence count)
  identity_splice    first half talker A, second half talker B (video AND audio)
  silent_segment     middle third of the audio replaced by silence
  face_occluded      face region blacked out for the middle third
  frozen_face        video frozen for the middle third while the audio keeps playing (replayed/still face)
  low_quality        240p, CRF 40
Only ffmpeg (bundled with imageio-ffmpeg) is used.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "datasets", "grid", "video")
OUT = os.path.join(ROOT, "datasets", "m3_generated")


def ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args):
    subprocess.run([ff(), "-v", "error", "-y"] + args, check=True)


def interview(talker, n, offset, tmp):
    """Concatenate n GRID sentences (starting at `offset`) into one H.264/AAC clip."""
    z = zipfile.ZipFile(os.path.join(GRID, f"{talker}.mpg_vcd.zip"))
    names = sorted(x for x in z.namelist() if x.endswith(".mpg"))[offset:offset + n]
    parts = []
    for x in names:
        p = os.path.join(tmp, f"{talker}_{os.path.basename(x)}")
        with open(p, "wb") as f:
            f.write(z.read(x))
        parts.append(p)
    lst = os.path.join(tmp, f"{talker}_{offset}.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    out = os.path.join(tmp, f"{talker}_{offset}_base.mp4")
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "18", "-r", "25",
         "-c:a", "aac", "-ar", "16000", "-ac", "1", out])
    return out


def duration(path):
    sys.path.insert(0, ROOT)
    from backend.vision.media import probe
    return probe(path).duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talkers", nargs="+", default=["s1", "s2", "s3", "s4", "s15", "s20"])
    ap.add_argument("--sentences", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []

    def add(clip, talker, manip, params, expected):
        rows.append({"clip": os.path.basename(clip), "talker": talker, "manipulation": manip,
                     "params": params, "expected": expected})
        print(f"  {os.path.basename(clip)}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        talkers = args.talkers
        for i, t in enumerate(talkers):
            other = talkers[(i + 1) % len(talkers)]
            print(f"{t} (partner {other})", flush=True)
            base = interview(t, args.sentences, 0, tmp)
            alt = interview(other, args.sentences, 0, tmp)
            d = duration(base)
            third = d / 3

            g = os.path.join(OUT, f"{t}_genuine.mp4")
            run(["-i", base, "-c", "copy", g])
            add(g, t, "genuine", "", "identity LOW; sync LOW; deepfake LOW")

            for k in (-10, -5, -2, 2, 5, 10):
                o = os.path.join(OUT, f"{t}_av_shift{k:+d}.mp4")
                run(["-i", base, "-itsoffset", f"{k / 25:.3f}", "-i", base, "-map", "0:v", "-map", "1:a",
                     "-c:v", "copy", "-c:a", "aac", "-shortest", o])
                exp = "sync LOW" if abs(k) <= 2 else ("sync MEDIUM+" if abs(k) <= 5 else "sync HIGH")
                add(o, t, "av_shift", f"frames={k}", exp)

            o = os.path.join(OUT, f"{t}_av_shift_partial.mp4")
            run(["-i", base, "-filter_complex",
                 f"[0:a]atrim=0:{third:.3f},asetpts=PTS-STARTPTS[a1];"
                 f"[0:a]atrim={third:.3f}:{2 * third:.3f},asetpts=PTS-STARTPTS,adelay=320[a2];"
                 f"[0:a]atrim={2 * third + 0.32:.3f},asetpts=PTS-STARTPTS[a3];[a1][a2][a3]concat=n=3:v=0:a=1[a]",
                 "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", o])
            add(o, t, "av_shift_partial", f"frames=+8 from {third:.1f}s to {2 * third:.1f}s",
                "middle segments offset; others normal")

            o = os.path.join(OUT, f"{t}_audio_swap.mp4")
            run(["-i", base, "-i", alt, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", o])
            add(o, t, "audio_swap", f"audio={other}", "sync MEDIUM/HIGH (low LSE-C / unstable offset); identity LOW")

            o = os.path.join(OUT, f"{t}_identity_splice.mp4")
            half = d / 2
            run(["-i", base, "-i", alt, "-filter_complex",
                 f"[0:v]trim=0:{half:.3f},setpts=PTS-STARTPTS[v1];[1:v]trim=0:{half:.3f},setpts=PTS-STARTPTS[v2];"
                 f"[0:a]atrim=0:{half:.3f},asetpts=PTS-STARTPTS[a1];[1:a]atrim=0:{half:.3f},asetpts=PTS-STARTPTS[a2];"
                 f"[v1][a1][v2][a2]concat=n=2:v=1:a=1[v][a]",
                 "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", o])
            add(o, t, "identity_splice", f"{t}->{other} at {half:.1f}s", "identity HIGH (2 identities, drift at cut); sync LOW")

            o = os.path.join(OUT, f"{t}_silent_segment.mp4")
            run(["-i", base, "-af", f"volume=enable='between(t,{third:.3f},{2 * third:.3f})':volume=0",
                 "-c:v", "copy", "-c:a", "aac", o])
            add(o, t, "silent_segment", f"silence {third:.1f}-{2 * third:.1f}s", "silent part not scored; sync LOW")

            o = os.path.join(OUT, f"{t}_face_occluded.mp4")
            run(["-i", base, "-vf", f"drawbox=x=0:y=0:w=iw:h=ih*0.75:color=black:t=fill:enable='between(t,{third:.3f},{2 * third:.3f})'",
                 "-c:v", "libx264", "-crf", "18", "-c:a", "copy", o])
            add(o, t, "face_occluded", f"face hidden {third:.1f}-{2 * third:.1f}s", "face_lost up, confidence down, risk NOT up")

            o = os.path.join(OUT, f"{t}_frozen_face.mp4")
            f1, f2 = int(third * 25), int(2 * third * 25)
            run(["-i", base, "-filter_complex", f"[0:v]split[a][b];[a][b]freezeframes=first={f1}:last={f2}:replace={f1}[v]",
                 "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", o])
            add(o, t, "frozen_face", f"video frozen {third:.1f}-{2 * third:.1f}s, audio continues",
                "mouth_speech_mismatch up; sync unstable in frozen part")

            o = os.path.join(OUT, f"{t}_low_quality.mp4")
            run(["-i", base, "-vf", "scale=-2:240", "-c:v", "libx264", "-crf", "40", "-c:a", "aac", "-b:a", "32k", o])
            add(o, t, "low_quality", "240p CRF40", "confidence down; outcome unchanged")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "talker", "manipulation", "params", "expected"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {OUT}")


if __name__ == "__main__":
    main()
