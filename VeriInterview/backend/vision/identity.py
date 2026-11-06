"""
Face-identity consistency over an interview video (M3 tasks I-1, I-2).

Builds the `identity_signals` dict consumed by adapters/insightface_adapter.py:

1. Candidate timeline: in every sampled frame, the largest face that passed
   quality gating (face_track.assess_track_quality). Taking the largest face
   per frame, not one tracker track, means a person swapped in mid-interview
   is still followed even if the tracker starts a new track.
2. Enrollment anchor (I-1): mean ArcFace embedding of the good frames in the
   first `enrollment_seconds`, falling back to the sharpest, most confident
   frames anywhere if the opening is unusable. `anchor_quality` = how
   consistent those enrollment frames are with each other.
3. Similarity trace vs the candidate's gallery (anchor + long clusters confirmed
   as the same person, so an unusual opening does not skew the session): robust
   minimum (p5), mean, and the share of frames below the same-person threshold.
4. Identity clusters (I-2): average-linkage clustering of all candidate
   embeddings. A cluster is a distinct PERSON only if it lasts >=
   `min_identity_presence_s` AND its mean face is below the same-person threshold
   against the anchor (the same face at another angle can form its own cluster). Drift events = runs of >= 1 s below the threshold, with
   timestamps and question ids for the evidence timeline.

Frames that fail quality gating are skipped, never scored as drift (R1 rule 4).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from backend.config import IDENTITY_THRESHOLDS
from backend.vision.face_track import FaceObs, TrackResult
from backend.vision.media import QuestionSegment, question_at


def candidate_timeline(result: TrackResult) -> list[FaceObs]:
    """Largest good-quality face with an embedding in each sampled frame, in time order."""
    best: dict[int, FaceObs] = {}
    for tr in result.tracks:
        for o in tr.obs:
            if not o.quality_ok or o.embedding is None:
                continue
            cur = best.get(o.frame_index)
            if cur is None or o.size > cur.size:
                best[o.frame_index] = o
    return [best[k] for k in sorted(best)]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_anchor(timeline: Sequence[FaceObs]) -> tuple[Optional[np.ndarray], float, str]:
    """Return (anchor embedding, anchor_quality 0-1, method)."""
    cfg = IDENTITY_THRESHOLDS
    if not timeline:
        return None, 0.0, "none"
    t0 = timeline[0].t
    window = [o for o in timeline if o.t - t0 <= cfg.enrollment_seconds]
    method = "enrollment_window"
    if len(window) < cfg.enrollment_min_frames:
        method = "best_frames_fallback"
        window = sorted(timeline, key=lambda o: -(o.det_score * min(1.0, o.sharpness / 100.0)))[: max(cfg.enrollment_min_frames, 15)]
    if len(window) < 3:
        return None, 0.0, "insufficient"
    embs = np.stack([o.embedding for o in window])
    anchor = _normalize(embs.mean(axis=0))
    quality = float(np.clip(np.median(embs @ anchor), 0.0, 1.0))
    return anchor, quality, method


def cluster_identities(timeline: Sequence[FaceObs], sample_fps: float) -> list[dict]:
    """Average-linkage cosine clustering. Returns clusters sorted by presence, with time spans."""
    if len(timeline) < 2:
        return [{"n": len(timeline), "seconds": len(timeline) / sample_fps, "start": timeline[0].t, "end": timeline[-1].t,
                 "centroid": timeline[0].embedding}] if timeline else []
    from scipy.cluster.hierarchy import fcluster, linkage
    embs = np.stack([o.embedding for o in timeline])
    z = linkage(embs, method="average", metric="cosine")
    labels = fcluster(z, t=1.0 - IDENTITY_THRESHOLDS.cluster_cosine, criterion="distance")
    out = []
    for lab in np.unique(labels):
        members = [(o, l) for o, l in zip(timeline, labels) if l == lab]
        ts = [o.t for o, _ in members]
        centroid = _normalize(np.mean([o.embedding for o, _ in members], axis=0))
        out.append({"n": len(ts), "seconds": len(ts) / sample_fps, "start": min(ts), "end": max(ts), "centroid": centroid})
    return sorted(out, key=lambda c: -c["n"])


def count_distinct_people(clusters: list[dict], anchor: np.ndarray) -> int:
    """1 (the candidate) + clusters that are a DIFFERENT person.

    A cluster counts only if it lasts >= min_identity_presence_s AND its mean face is
    below the LFW-calibrated same-person threshold against the anchor. The clustering
    link is stricter than the same-person threshold, so the same face at another angle
    (turned to talk, looking down at notes) can form its own cluster. Without the
    centroid check that would be miscounted as a second person (found on OEP, task F-4).
    """
    cfg = IDENTITY_THRESHOLDS
    others = [c for c in clusters if c["seconds"] >= cfg.min_identity_presence_s
              and float(c["centroid"] @ anchor) < cfg.same_person_cosine]
    return 1 + len(others)


def drift_events(timeline: Sequence[FaceObs], sims: np.ndarray, sample_fps: float,
                 questions: Sequence[QuestionSegment] = (), min_duration_s: float = 1.0) -> list[dict]:
    """Runs of consecutive below-threshold frames lasting >= min_duration_s."""
    thr = IDENTITY_THRESHOLDS.same_person_cosine
    events, run = [], []
    for o, s in list(zip(timeline, sims)) + [(None, 1.0)]:
        if o is not None and s < thr:
            run.append((o, s))
            continue
        if run and (run[-1][0].t - run[0][0].t + 1.0 / sample_fps) >= min_duration_s:
            worst = min(run, key=lambda x: x[1])
            events.append({"t": round(run[0][0].t, 2), "t_end": round(run[-1][0].t, 2),
                           "question_id": question_at(run[0][0].t, questions) if questions else None,
                           "similarity": round(float(worst[1]), 3)})
        run = []
    return events


def compare_reference(timeline: Sequence[FaceObs], anchor: np.ndarray, reference: dict) -> dict:
    """Session face vs an uploaded reference photo (task I-4).

    Catches a face that is consistent all session but is not the registered
    candidate (proxy or full-session face-swap), which the within-session
    anchor cannot see. `reference` is face_track.embed_photo() output.
    """
    if reference.get("error"):
        return {"reference_status": reference["error"]}
    ref = reference["embedding"]
    frame_sims = np.stack([o.embedding for o in timeline]) @ ref
    return {
        "reference_status": "ok",
        "reference_similarity": round(float(anchor @ ref), 3),
        "reference_match_pct": round(float(np.mean(frame_sims >= IDENTITY_THRESHOLDS.reference_match_cosine)), 3),
        "reference_extra_faces": int(reference.get("faces", 1)) - 1,
    }


def analyze(result: TrackResult, questions: Sequence[QuestionSegment] = (), reference: Optional[dict] = None) -> dict:
    """identity_signals dict (legacy keys kept for adapters/fixtures; see M3_TASKS.md section 5).

    `reference`: optional face_track.embed_photo() result for the registered candidate's photo.
    """
    cfg = IDENTITY_THRESHOLDS
    base = {"face_lost_pct": round(result.face_lost_pct, 3), "multi_face_pct": round(result.multi_face_pct, 3),
            "frames_sampled": result.frames_sampled}
    timeline = candidate_timeline(result)
    if result.frames_sampled == 0 or result.frames_with_face == 0:
        return {**base, "status_reason": "no_face"}
    anchor, anchor_quality, method = build_anchor(timeline)
    if anchor is None:
        return {**base, "status_reason": "insufficient_good_frames", "good_frames": len(timeline)}

    clusters = cluster_identities(timeline, result.sample_fps)
    distinct = count_distinct_people(clusters, anchor)
    # Candidate gallery: the opening anchor + long clusters confirmed as the SAME person
    # (centroid >= same-person threshold vs the anchor). Compare each frame with its best match,
    # so an unusual opening (lighting / pose) does not make the rest of the session "not match".
    # An impostor's cluster scores ~0 vs the anchor (LFW impostor max 0.233), so it never joins.
    gallery = [anchor] + [c["centroid"] for c in clusters if c["seconds"] >= cfg.min_identity_presence_s
                          and float(c["centroid"] @ anchor) >= cfg.same_person_cosine]
    sims = (np.stack([o.embedding for o in timeline]) @ np.stack(gallery).T).max(axis=1)
    signals = {
        **base,
        "min_frame_similarity_to_anchor": round(float(np.percentile(sims, 5)), 3),   # robust min (p5)
        "mean_similarity": round(float(sims.mean()), 3),
        "below_threshold_pct": round(float(np.mean(sims < cfg.same_person_cosine)), 3),
        "distinct_identities": distinct,
        "anchor_quality": round(anchor_quality, 3),
        "anchor_method": method,
        "gallery_size": len(gallery),
        "good_frames": len(timeline),
        "good_frame_pct": round(len(timeline) / result.frames_sampled, 3),
        "drift_events": drift_events(timeline, sims, result.sample_fps, questions),
        "clusters": [{**{k: round(v, 2) if isinstance(v, float) else v for k, v in c.items() if k != "centroid"},
                      "similarity_to_anchor": round(float(c["centroid"] @ anchor), 3)} for c in clusters[:5]],
        "status_reason": None,
    }
    if reference is not None:
        signals.update(compare_reference(timeline, anchor, reference))
    if questions:
        per_q = []
        for q in questions:
            qs = [s for o, s in zip(timeline, sims) if q.start <= o.t < q.end]
            if qs:
                per_q.append({"question_id": q.question_id, "frames": len(qs),
                              "mean_similarity": round(float(np.mean(qs)), 3),
                              "below_threshold_pct": round(float(np.mean(np.array(qs) < cfg.same_person_cosine)), 3)})
        signals["per_question"] = per_q
    return signals
