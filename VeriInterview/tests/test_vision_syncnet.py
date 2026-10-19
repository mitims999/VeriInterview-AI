"""
Tests for the vendored SyncNet scorer (backend/vision/syncnet, M3 task S-1).

The MFCC / distance-sweep tests need only numpy + torch + python_speech_features.
The reproduction test runs the real pretrained model on the upstream reference
clip (storage/models/syncnet/example.avi) and skips if the weights are missing.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("python_speech_features")

from backend.config import MODELS_DIR  # noqa: E402
from backend.vision import syncnet  # noqa: E402


def test_mfcc_is_13_coeffs_at_100hz_and_float_input_is_rescaled():
    t = np.arange(16000) / 16000.0
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    m_float = syncnet.mfcc(tone)
    m_scaled = syncnet.mfcc(tone.astype(np.float64) * 32768.0)   # already int16-scale -> not rescaled again
    assert m_float.shape[0] == 13 and abs(m_float.shape[1] - 100) <= 1
    np.testing.assert_allclose(m_float, m_scaled, rtol=1e-6, atol=1e-6)


def test_distance_sweep_finds_a_known_shift():
    rng = np.random.default_rng(0)
    feats = torch.from_numpy(rng.standard_normal((60, 16)).astype(np.float32))
    shifted = torch.roll(feats, shifts=3, dims=0)        # "audio" feature i == video feature i-3
    dists = syncnet._calc_pdist(feats, shifted, vshift=5)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    assert int(5 - torch.argmin(mdist)) == -3


def test_too_short_clip_is_rejected():
    if not os.path.exists(os.path.join(MODELS_DIR, "syncnet", "syncnet_v2.model")):
        pytest.skip("SyncNet weights not downloaded")
    with pytest.raises(ValueError):
        syncnet.score(np.zeros((4, 224, 224, 3), np.uint8), np.zeros(16000, np.float32))


def test_reproduces_upstream_reference_offset_on_example_clip():
    clip = os.path.join(MODELS_DIR, "syncnet", "example.avi")
    if not os.path.exists(clip) or not os.path.exists(os.path.join(MODELS_DIR, "syncnet", "syncnet_v2.model")):
        pytest.skip("SyncNet weights / example clip not downloaded")
    pytest.importorskip("imageio_ffmpeg")
    from backend.vision import media
    crops = np.stack([f.image for f in media.iter_frames(clip, end=5.0)])     # first 125 frames
    audio = media.load_audio(clip, end=5.0)
    s = syncnet.score(crops, audio)
    # full clip reproduces the README exactly (offset 3, min dist 5.353, conf 10.021);
    # the 5 s excerpt keeps the same offset with high confidence
    assert s.offset == 3
    assert s.confidence > 8.0 and s.min_dist < 6.5
