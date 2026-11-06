"""
SyncNet lip-sync scoring, adapted from joonson/syncnet_python
(SyncNetInstance.evaluate, MIT, Copyright (c) 2016-present Joon Son Chung,
see LICENSE.md). `model.py` is the original network, vendored verbatim.

Changes from upstream (M3 task S-1):
- Works on in-memory arrays (224x224 BGR face crops at 25 fps + 16 kHz
  audio) instead of shelling out to ffmpeg and writing JPEGs to a temp dir.
- Returns LSE-D (min mean distance) and the framewise confidence as well as
  offset/confidence. LSE-C is upstream's `conf`.
- Weights load with torch.load(weights_only=True) only (no pickled code).

Maths is unchanged: 13 MFCCs at 100 Hz from int16-scale audio, 4 MFCC frames
per video frame, 5-frame video windows vs 20-step audio windows, a +/-vshift
pairwise-distance sweep, offset = vshift - argmin(mean distance),
confidence = median - min of the mean distances.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.config import MODELS_DIR
from backend.vision.face_track import ModelUnavailable

FPS = 25
SR = 16000
CROP = 224

_MODEL = None
_LOCK = threading.Lock()


@dataclass
class SyncScore:
    offset: int              # frames; positive = audio leads video (upstream convention)
    confidence: float        # LSE-C (upstream "Confidence")
    min_dist: float          # LSE-D (upstream "Min dist")
    framewise_conf: np.ndarray   # median-filtered per-frame confidence at the best offset
    n_frames: int


def get_model(device: Optional[str] = None):
    """Lazy SyncNet model on CPU (or `device`). Raises ModelUnavailable if torch/weights are missing."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            path = os.path.join(MODELS_DIR, "syncnet", "syncnet_v2.model")
            if not os.path.exists(path):
                raise ModelUnavailable("SyncNet weights missing: run python scripts/download_models.py")
            try:
                import torch
                from backend.vision.syncnet.model import S
            except ImportError as e:
                raise ModelUnavailable("torch not installed: pip install -r requirements-vision.txt") from e
            device = device or "cpu"
            torch.set_num_threads(max(1, os.cpu_count() or 1))   # N-4: 8 threads ~17% faster than torch's default 4 here
            net = S(num_layers_in_fc_layers=1024).to(device)
            state = torch.load(path, map_location="cpu", weights_only=True)
            own = net.state_dict()
            for name, param in state.items():
                own[name].copy_(param)
            net.eval()
            _MODEL = (net, device)
        return _MODEL


def mfcc(audio: np.ndarray) -> np.ndarray:
    """13 x T MFCC matrix at 100 Hz, computed exactly as upstream (int16-scale input)."""
    import python_speech_features
    x = audio.astype(np.float64)
    if np.abs(x).max() <= 1.0:            # float audio from media.load_audio -> int16 scale
        x = x * 32768.0
    return np.asarray(python_speech_features.mfcc(x, SR)).T


def _calc_pdist(feat1, feat2, vshift: int):
    import torch
    win = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    return [torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win, 1), feat2p[i:i + win, :])
            for i in range(len(feat1))]


def score(crops: np.ndarray, audio: np.ndarray, vshift: int = 15, batch_size: int = 20) -> SyncScore:
    """Lip-sync score for one aligned clip.

    crops: T x 224 x 224 x 3 uint8 BGR face crops at 25 fps.
    audio: the matching 16 kHz mono audio (float in [-1, 1] or int16 scale).
    """
    import torch
    from scipy import signal

    net, device = get_model()
    cc = mfcc(audio)
    n = min(len(crops), cc.shape[1] // 4)
    if n < 5 + 1:
        raise ValueError(f"clip too short for SyncNet ({n} frames)")
    last = n - 5

    # upstream layout: 1 x 3 x T x H x W video, 1 x 1 x 13 x T audio, raw (unnormalised) values
    imtv = torch.from_numpy(np.ascontiguousarray(crops[:n].transpose(3, 0, 1, 2)[None]).astype(np.float32))
    cct = torch.from_numpy(cc[None, None].astype(np.float32))
    im_feat, cc_feat = [], []
    with torch.no_grad():
        for i in range(0, last, batch_size):
            idx = range(i, min(last, i + batch_size))
            im_in = torch.cat([imtv[:, :, v:v + 5] for v in idx], 0)
            cc_in = torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in idx], 0)
            im_feat.append(net.forward_lip(im_in.to(device)).cpu())
            cc_feat.append(net.forward_aud(cc_in.to(device)).cpu())
    im_feat, cc_feat = torch.cat(im_feat, 0), torch.cat(cc_feat, 0)

    dists = _calc_pdist(im_feat, cc_feat, vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    fdist = np.stack([d[minidx].numpy() for d in dists])
    fconf = torch.median(mdist).numpy() - fdist
    return SyncScore(offset=int(vshift - minidx), confidence=float(torch.median(mdist) - minval),
                     min_dist=float(minval), framewise_conf=signal.medfilt(fconf, kernel_size=9), n_frames=n)
