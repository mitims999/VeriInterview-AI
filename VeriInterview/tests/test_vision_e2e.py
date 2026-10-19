"""
Model-backed end-to-end M3 tests (task T-2): the full video pipeline
(video_intake.analyze_video -> adapters -> deepfake_fusion) on the generated GRID
test clips, against the committed expectations in tests/data/m3_vision_manifest.csv.

Marked `vision`; skipped automatically when the clips (datasets/m3_generated, built by
scripts/build_m3_testset.py from public GRID data) or the model weights are absent,
so the suite keeps passing after datasets/ is deleted (task T-5). The manifest holds
only clip names and expected levels, no media.

Run only these:  pytest -m vision
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = os.path.dirname(os.path.dirname(__file__))
CLIPS = os.path.join(ROOT, "datasets", "m3_generated")
MANIFEST = os.path.join(os.path.dirname(__file__), "data", "m3_vision_manifest.csv")
ROWS = list(csv.DictReader(open(MANIFEST)))

pytestmark = pytest.mark.vision


def _models_ready():
    try:
        from backend.api_vision import health
        return health().ready
    except Exception:
        return False


_cache: dict = {}


def _analyse(clip):
    if clip not in _cache:
        from backend.adapters import video_intake
        from backend.api_vision import build_result
        path = os.path.join(CLIPS, clip)
        _cache[clip] = build_result(video_intake.analyze_video(path, run_objects=False))
    return _cache[clip]


@pytest.mark.parametrize("row", ROWS, ids=[f"{r['clip']}-{r['check']}" for r in ROWS])
def test_clip_meets_expected_level(row):
    if not os.path.exists(os.path.join(CLIPS, row["clip"])):
        pytest.skip("generated test clips not present (run scripts/build_m3_testset.py)")
    if not _models_ready():
        pytest.skip("vision models / dependencies not installed")
    result = _analyse(row["clip"])
    got = getattr(result, row["check"])
    if row["expected"] == "MEDIUM":
        # MEDIUM = "flagged for review": weak cues may legitimately land anywhere in MEDIUM, never HIGH
        assert got.level == "MEDIUM", (row, got.score, [e.message for e in got.evidence])
    else:
        assert got.level == row["expected"], (row, got.score, [e.message for e in got.evidence])


def test_bad_media_becomes_insufficient_evidence_not_an_exception(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from backend.adapters import video_intake
    from backend.api_vision import build_result
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"definitely not a video")
    res = build_result(video_intake.analyze_video(str(bad), use_cache=False))
    for m in (res.identity, res.sync, res.behavior, res.deepfake):
        assert m.status in ("INSUFFICIENT_EVIDENCE", "UNAVAILABLE") and m.score == 0
