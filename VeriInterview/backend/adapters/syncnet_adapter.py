"""
Adapter for joonson/syncnet_python (https://github.com/joonson/syncnet_python).

Build Spec Section 3.2: run `run_pipeline.py` then `run_syncnet.py` on each
interview video segment, capture native AV-offset + confidence outputs, map to
`sync_risk_score`.

INTEGRATION NOTE: SyncNet requires a pretrained model download (`download_model.sh`)
and a CUDA-capable environment for practical per-frame inference; it operates on
raw video segments which are not available in this text-only prototype session.
This adapter exposes the exact contract SyncNet's outputs are converted to
(av_offset_frames, confidence -> sync_risk_score) and computes it from the
per-segment `av_offset_frames` / `sync_confidence` values recorded in the
synthetic fixtures (data/synthetic/*.json -> "av_signals"), standing in for
what `run_syncnet.py` would report for real media. Swap `compute()` for a
subprocess call into `vendor/syncnet_python/run_syncnet.py` when real video is
available.
"""
from __future__ import annotations
from backend.schemas import ModuleResult, Evidence, Quality


def compute(av_signals: dict | None) -> ModuleResult:
    if not av_signals:
        return ModuleResult(
            feature="av_sync", score=0, confidence=0.0, level="UNAVAILABLE",
            status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="av_sync", severity="low", message="No video segment available for lip-sync analysis.")],
            quality=Quality(data_quality=0.0, baseline_sufficient=False),
        )

    offset = abs(av_signals.get("av_offset_frames", 0))
    sync_confidence = av_signals.get("sync_confidence", 0.7)  # SyncNet's own confidence, 0-1

    # SyncNet bonafide range is roughly within +-1 to 2 frames at good confidence.
    if offset <= 2:
        risk = 5 + offset * 5
    elif offset <= 5:
        risk = 25 + (offset - 2) * 12
    else:
        risk = min(100, 60 + (offset - 5) * 8)

    # low SyncNet confidence widens uncertainty rather than raising risk outright
    reported_confidence = max(0.2, min(0.95, sync_confidence))

    evidence = []
    if offset <= 2:
        evidence.append(Evidence(type="av_offset", severity="low", message=f"AV offset of {offset} frame(s) within normal bonafide range."))
    elif offset <= 5:
        evidence.append(Evidence(type="av_offset", severity="medium", message=f"AV offset of {offset} frames is above typical bonafide range."))
    else:
        evidence.append(Evidence(type="av_offset", severity="high", message=f"AV offset of {offset} frames is well outside bonafide range; audio/video desync detected."))

    if sync_confidence < 0.5:
        evidence.append(Evidence(type="sync_confidence", severity="medium", message=f"SyncNet confidence low ({sync_confidence:.2f}); result should be treated cautiously."))

    level = "LOW" if risk <= 30 else ("MEDIUM" if risk <= 60 else "HIGH")
    return ModuleResult(
        feature="av_sync", score=round(risk, 1), confidence=round(reported_confidence, 2), level=level,
        evidence=evidence, quality=Quality(data_quality=sync_confidence, baseline_sufficient=True),
    )
