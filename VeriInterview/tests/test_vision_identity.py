"""
Unit tests for backend/vision/identity.py (M3 tasks I-1, I-2).
Synthetic 512-d embeddings: same person = small noise around one direction.
No models, no data.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.vision import identity
from backend.vision.face_track import FaceObs, Track, TrackResult
from backend.vision.media import QuestionSegment

RNG = np.random.default_rng(0)
PERSON_A, PERSON_B = RNG.standard_normal(512), RNG.standard_normal(512)


def _emb(person, noise=0.25):
    v = person + noise * np.linalg.norm(person) / np.sqrt(512) * RNG.standard_normal(512) * 3
    return (v / np.linalg.norm(v)).astype(np.float32)


def _obs(i, fps, person, ok=True, size=120.0):
    return FaceObs(t=i / fps, frame_index=i, bbox=np.array([0, 0, size, size], np.float32), kps=None,
                   det_score=0.9, sharpness=200.0, embedding=_emb(person), quality_ok=ok)


def _result(people, fps=2.0, extra_tracks=()):
    obs = [_obs(i, fps, p) for i, p in enumerate(people)]
    return TrackResult(tracks=[Track(0, obs), *extra_tracks], frames_sampled=len(people), frames_with_face=len(people),
                       frames_multi_face=0, sample_fps=fps, primary_track_id=0)


def test_synthetic_embeddings_behave_like_arcface():
    a1, a2, b = _emb(PERSON_A), _emb(PERSON_A), _emb(PERSON_B)
    assert a1 @ a2 > 0.6 and abs(a1 @ b) < 0.2


def test_genuine_session_is_consistent():
    s = identity.analyze(_result([PERSON_A] * 60))
    assert s["status_reason"] is None
    assert s["below_threshold_pct"] == 0.0
    assert s["distinct_identities"] == 1
    assert s["drift_events"] == []
    assert s["min_frame_similarity_to_anchor"] > 0.6 and s["anchor_method"] == "enrollment_window"


def test_identity_switch_is_detected_with_time_and_question():
    people = [PERSON_A] * 40 + [PERSON_B] * 40          # switch at t = 20 s
    qs = [QuestionSegment("Q1", 0, 15), QuestionSegment("Q2", 15, 40)]
    s = identity.analyze(_result(people), qs)
    assert s["distinct_identities"] == 2
    assert s["below_threshold_pct"] == pytest.approx(0.5, abs=0.01)
    assert len(s["drift_events"]) == 1
    ev = s["drift_events"][0]
    assert ev["t"] == pytest.approx(20.0) and ev["question_id"] == "Q2"
    assert {q["question_id"]: q["below_threshold_pct"] for q in s["per_question"]} == {"Q1": 0.0, "Q2": pytest.approx(0.8)}


def test_brief_other_face_is_not_a_second_identity():
    # 1 second of someone else (2 frames) is below min_identity_presence_s and min drift duration
    s = identity.analyze(_result([PERSON_A] * 30 + [PERSON_B] * 1 + [PERSON_A] * 30))
    assert s["distinct_identities"] == 1 and s["drift_events"] == []


def test_low_quality_frames_are_skipped_not_scored():
    obs = [_obs(i, 2.0, PERSON_A) for i in range(40)] + [_obs(40 + i, 2.0, PERSON_B, ok=False) for i in range(20)]
    s = identity.analyze(TrackResult([Track(0, obs)], 60, 60, 0, 2.0, 0))
    assert s["good_frames"] == 40 and s["below_threshold_pct"] == 0.0 and s["distinct_identities"] == 1


def test_largest_face_per_frame_is_the_candidate():
    # a smaller second face (other person) in every frame must not affect the candidate trace
    other = Track(1, [_obs(i, 2.0, PERSON_B, size=60.0) for i in range(30)])
    s = identity.analyze(_result([PERSON_A] * 30, extra_tracks=[other]))
    assert s["below_threshold_pct"] == 0.0 and s["distinct_identities"] == 1


def test_anchor_falls_back_when_opening_is_unusable():
    obs = [_obs(i, 2.0, PERSON_A, ok=(i >= 30)) for i in range(60)]   # first 15 s unusable
    s = identity.analyze(TrackResult([Track(0, obs)], 60, 60, 0, 2.0, 0))
    assert s["status_reason"] is None and s["below_threshold_pct"] == 0.0


def test_no_face_and_too_few_frames_are_reported_not_scored():
    assert identity.analyze(TrackResult([], 20, 0, 0, 2.0, None))["status_reason"] == "no_face"
    few = TrackResult([Track(0, [_obs(i, 2.0, PERSON_A) for i in range(2)])], 20, 2, 0, 2.0, 0)
    assert identity.analyze(few)["status_reason"] == "insufficient_good_frames"


def test_reference_photo_catches_consistent_but_wrong_person():
    # a face-swap / proxy present for the whole session: consistent within the session...
    impostor_session = _result([PERSON_B] * 60)
    within = identity.analyze(impostor_session)
    assert within["below_threshold_pct"] == 0.0 and within["distinct_identities"] == 1
    # ...but it does not match the registered candidate's reference photo
    s = identity.analyze(impostor_session, reference={"embedding": _emb(PERSON_A), "faces": 1})
    assert s["reference_status"] == "ok"
    assert s["reference_similarity"] < 0.2 and s["reference_match_pct"] == 0.0
    genuine = identity.analyze(_result([PERSON_A] * 60), reference={"embedding": _emb(PERSON_A), "faces": 1})
    assert genuine["reference_similarity"] > 0.6 and genuine["reference_match_pct"] == 1.0


def test_unusable_reference_photo_is_reported_not_scored():
    s = identity.analyze(_result([PERSON_A] * 30), reference={"error": "no_face"})
    assert s["reference_status"] == "no_face" and "reference_similarity" not in s


def test_same_person_at_another_angle_is_not_a_second_person():
    # F-4 finding on OEP: a pose change (turned to talk / looking down) can form its own cluster.
    # Simulate: 20 s of the same person whose embedding drifts (still ~0.5 cosine to the anchor).
    rng = np.random.default_rng(5)
    turned = PERSON_A + 0.9 * np.linalg.norm(PERSON_A) / np.sqrt(512) * rng.standard_normal(512) * 3
    obs = [_obs(i, 2.0, PERSON_A) for i in range(80)] + [_obs(80 + i, 2.0, turned) for i in range(40)]
    s = identity.analyze(TrackResult([Track(0, obs)], 120, 120, 0, 2.0, 0))
    sims = [c["similarity_to_anchor"] for c in s["clusters"]]
    assert s["distinct_identities"] == 1, (sims, s["clusters"])


def test_unusual_opening_does_not_make_the_session_mismatch():
    # F-4 finding (OEP subject 15): the first seconds look different (lighting/pose); the rest of the
    # session is the same person but only ~0.35 similar to that opening anchor.
    rng = np.random.default_rng(9)
    usual = PERSON_A + 0.75 * np.linalg.norm(PERSON_A) / np.sqrt(512) * rng.standard_normal(512) * 3
    odd_opening = [_obs(i, 1.0, PERSON_A) for i in range(10)]
    rest = [_obs(10 + i, 1.0, usual) for i in range(150)]
    s = identity.analyze(TrackResult([Track(0, odd_opening + rest)], 160, 160, 0, 1.0, 0))
    assert s["distinct_identities"] == 1 and s["below_threshold_pct"] < 0.05, s["clusters"]
    # ...while a real switch to another person still shows up
    sw = identity.analyze(TrackResult([Track(0, odd_opening + [_obs(10 + i, 1.0, PERSON_B) for i in range(150)])],
                                      160, 160, 0, 1.0, 0))
    assert sw["distinct_identities"] == 2 and sw["below_threshold_pct"] > 0.8
