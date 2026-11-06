"""
Deepfake Sub-Score fusion (M3 tasks F-3, F-4 synthetic part).

F-3: golden rule, corroboration, strong identity signals, unavailable inputs.
F-4 / R1: nervous-but-genuine profiles must be LOW. Fakes must be HIGH whether
or not the person also looks nervous. Eye cues alone never exceed MEDIUM.
Inputs are built with the real adapters' compute() from signal dicts, so the
whole adapter -> fusion chain is exercised.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.adapters import insightface_adapter as ia, syncnet_adapter as sa
from backend.modules.deepfake_fusion import fuse_deepfake


# ---- signal builders (values from the M3 calibrations)
def identity(below=0.0, distinct=1, ref=None):
    s = {"face_lost_pct": 0.02, "multi_face_pct": 0.0, "frames_sampled": 240, "min_frame_similarity_to_anchor": 0.9,
         "mean_similarity": 0.95, "below_threshold_pct": below, "distinct_identities": distinct, "anchor_quality": 0.95,
         "good_frames": 230, "good_frame_pct": 0.96, "status_reason": None,
         "drift_events": [{"t": 60.0, "t_end": 120.0, "question_id": "Q3", "similarity": 0.05}] if below > 0.1 else []}
    if ref is not None:
        s.update({"reference_status": "ok", "reference_similarity": ref, "reference_match_pct": 1.0 if ref > 0.4 else 0.0})
    return ia.compute(s)


def sync(offset=1, lse=5.9, stability=1.0, segs=None):
    segs = segs or [{"start": i * 10.0, "end": i * 10.0 + 9, "offset": offset, "lse_c": lse, "n_frames": 230} for i in range(4)]
    return sa.compute({"av_offset_frames": offset, "lse_c": lse, "offset_stability": stability,
                       "speech_seconds_analyzed": 40.0, "per_segment": segs, "status_reason": None})


def eyes(rate=18.0, cv=0.6, mismatch=0.0, visible=300.0, few=False, metronomic=False):
    return {"frames_used": int(visible * 10), "eye_visible_s": visible, "blink_count": int(rate * visible / 60),
            "blink_rate_per_min": rate, "blink_ibi_cv": cv, "blink_judgeable": visible >= 120, "too_few_blinks": few,
            "metronomic_blinks": metronomic, "high_blink_rate": rate > 30, "mouth_speech_mismatch_pct": mismatch}


UNAVAILABLE = ia.compute(None)


# ------------------------------------------------------------------ F-3
def test_genuine_is_low():
    r = fuse_deepfake(identity(), sync(), eyes())
    assert r.feature == "deepfake" and r.level == "LOW" and r.status == "OK"
    assert r.details["sub_scores"]["identity"]["level"] == "LOW"


def test_one_module_high_alone_is_capped_at_medium():
    only_sync = fuse_deepfake(identity(), sync(offset=11), eyes())
    assert only_sync.level in ("LOW", "MEDIUM") and only_sync.score <= 60
    partial_identity = fuse_deepfake(identity(below=0.3), sync(), eyes())     # drift but not a confident switch
    assert partial_identity.level != "HIGH"


def test_two_independent_highs_give_high_with_boost():
    r = fuse_deepfake(identity(below=0.3), sync(offset=11), eyes())
    assert r.level == "HIGH" and r.details["inputs_high"] == 2
    assert "2 independent checks" in r.evidence[0].message


def test_strong_identity_signal_alone_reaches_high():
    switch = fuse_deepfake(identity(below=0.5, distinct=2), sync(), eyes())
    assert switch.level == "HIGH" and switch.details["strong_identity_signal"]
    photo = fuse_deepfake(identity(ref=0.12), sync(), eyes())                # face-swap for the whole session
    assert photo.level == "HIGH" and photo.details["strong_identity_signal"]


def test_unavailable_inputs_redistribute_weight_and_lower_confidence():
    full = fuse_deepfake(identity(), sync(), eyes())
    no_video = fuse_deepfake(UNAVAILABLE, sa.compute(None), None)
    assert no_video.level == "UNAVAILABLE" and no_video.status == "INSUFFICIENT_EVIDENCE"
    only_identity = fuse_deepfake(identity(), sa.compute({"status_reason": "no_speech"}), None)
    assert only_identity.status == "OK" and only_identity.confidence < full.confidence
    assert "Not assessed" in only_identity.evidence[0].message


def test_eye_cues_alone_never_exceed_medium():
    r = fuse_deepfake(identity(), sync(), eyes(mismatch=0.9, few=True, metronomic=True))
    assert r.level in ("LOW", "MEDIUM")
    assert r.details["sub_scores"]["eye_authenticity"]["score"] <= 60


def test_summary_never_claims_a_verdict():
    r = fuse_deepfake(identity(below=0.5, distinct=2), sync(offset=11), eyes(mismatch=0.5))
    assert not any("confirmed" in e.message.lower() or "is a deepfake" in e.message.lower() for e in r.evidence)


# ------------------------------------------------------------------ F-4 / R1
def nervous_genuine():
    # stress: blinks a lot, irregularly; identity and lip-sync are clean. (Gaze aversion / head movement /
    # pauses are behaviour signals and are not inputs here by design.)
    return fuse_deepfake(identity(), sync(lse=4.8), eyes(rate=38.0, cv=0.9))


def test_r1_nervous_genuine_is_low_and_explained():
    r = nervous_genuine()
    assert r.level == "LOW"
    assert any(e.type == "nervousness_note" for e in r.evidence)


def test_r1_nervous_genuine_on_a_poor_webcam_is_still_low():
    r = fuse_deepfake(identity(), sync(lse=1.2, stability=1.0), eyes(rate=40.0, cv=0.8))
    assert r.level == "LOW"


def test_r1_fake_calm_is_high():
    r = fuse_deepfake(identity(below=0.5, distinct=2), sync(offset=-9), eyes(rate=0.5, few=True))
    assert r.level == "HIGH"


def test_r1_fake_nervous_is_still_high():
    # a deepfake operator who also looks nervous: nervousness must not mask the fake
    r = fuse_deepfake(identity(ref=0.1), sync(offset=4, lse=1.1, stability=0.4), eyes(rate=40.0, cv=0.9))
    assert r.level == "HIGH"


def test_r1_nervous_scores_far_below_fakes():
    fakes = [fuse_deepfake(identity(below=0.5, distinct=2), sync(offset=-9), eyes(rate=0.5, few=True)),
             fuse_deepfake(identity(ref=0.1), sync(), eyes()),
             fuse_deepfake(identity(), sync(offset=4, lse=1.1, stability=0.4), eyes(mismatch=0.3))]
    assert all(nervous_genuine().score + 30 < f.score for f in fakes[:2])
    assert nervous_genuine().score < fakes[2].score
