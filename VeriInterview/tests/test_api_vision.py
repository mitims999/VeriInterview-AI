"""
API tests for backend/api_vision.py (M3 video-analysis jobs for the frontend).

The heavy video analysis is replaced by a stub returning canned signals, so these
tests check the HTTP contract, job lifecycle, validation and privacy behaviour fast.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend import api_vision  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)

CANNED = {
    "identity_signals": {"face_lost_pct": 0.0, "multi_face_pct": 0.0, "frames_sampled": 180, "min_frame_similarity_to_anchor": 0.0,
                         "mean_similarity": 0.5, "below_threshold_pct": 0.5, "distinct_identities": 2, "anchor_quality": 0.95,
                         "good_frames": 175, "good_frame_pct": 0.97, "status_reason": None,
                         "drift_events": [{"t": 90.0, "t_end": 179.0, "question_id": "Q2", "similarity": -0.05}]},
    "av_signals": {"av_offset_frames": 1, "lse_c": 5.8, "offset_stability": 1.0, "speech_seconds_analyzed": 30.0,
                   "per_segment": [], "status_reason": None},
    "behavior_signals": {"gaze_status": "ok", "gaze_off_screen_pct": 0.05, "head_pose_variance": "normal", "gaze_coverage": 0.95,
                         "baseline_gaze_off_pct": 0.04, "per_question": [], "gaze_events": [],
                         "objects_status": "ok", "multi_person_detected": False, "phone_detected": True,
                         "object_events": [{"type": "phone", "t": 42.0, "t_end": 50.0, "question_id": "Q1"}]},
    "eye_authenticity": {"frames_used": 1790, "eye_visible_s": 179, "blink_rate_per_min": 17, "blink_ibi_cv": 0.6,
                         "too_few_blinks": False, "metronomic_blinks": False, "high_blink_rate": False,
                         "mouth_speech_mismatch_pct": 0.0},
    "meta": {"duration_s": 179.0, "total_s": 1.0, "cache_hit": False},
}


@pytest.fixture
def stub_analysis(monkeypatch):
    calls = {}

    def fake(path, questions=(), reference_photo=None, progress=None, **kw):
        calls.update(path=path, existed=os.path.exists(path), questions=list(questions), photo=reference_photo)
        for stage in ("speech", "identity", "gaze_objects_lipsync", "landmarks_done", "objects_done", "sync_done", "finishing"):
            progress(stage, 0.0)
        return CANNED

    monkeypatch.setattr("backend.adapters.video_intake.analyze_video", fake)
    return calls


def _wait(job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/vision/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_lifecycle_result_contract_and_temp_file_deleted(stub_analysis):
    r = client.post("/api/vision/jobs", files={"video": ("interview.mp4", b"\x00" * 1000, "video/mp4"),
                                               "reference_photo": ("me.jpg", b"\xff\xd8photo", "image/jpeg")},
                    data={"questions": '[{"question_id": "Q1", "start": 0, "end": 60}, {"question_id": "Q2", "start": 60, "end": 179}]'})
    assert r.status_code == 202
    job = r.json()
    assert job["status"] in ("queued", "running", "done") and job["reference_photo"] is True
    done = _wait(job["job_id"])
    assert done["status"] == "done" and done["progress"] == 1.0 and done["stage_label"] == "Done"

    res = done["result"]
    assert set(res) == {"deepfake", "identity", "sync", "eye_authenticity", "behavior", "timeline", "meta"}
    assert res["deepfake"]["feature"] == "deepfake" and res["deepfake"]["level"] == "HIGH"       # identity switch
    assert set(res["deepfake"]["details"]["sub_scores"]) == {"identity", "sync", "eye_authenticity"}
    assert res["behavior"]["level"] == "HIGH"                                                   # phone, behaviour only
    ts = [e["timestamp"] for e in res["timeline"]]
    assert ts == sorted(ts) and 42.0 in ts and 90.0 in ts

    assert stub_analysis["existed"] and not os.path.exists(stub_analysis["path"])               # temp video deleted
    assert [q.question_id for q in stub_analysis["questions"]] == ["Q1", "Q2"]
    assert stub_analysis["photo"] == b"\xff\xd8photo"


def test_delete_removes_finished_job(stub_analysis):
    jid = client.post("/api/vision/jobs", files={"video": ("a.webm", b"\x00" * 10, "video/webm")}).json()["job_id"]
    _wait(jid)
    assert client.delete(f"/api/vision/jobs/{jid}").status_code == 204
    assert client.get(f"/api/vision/jobs/{jid}").status_code == 404


def test_validation_errors():
    assert client.post("/api/vision/jobs", files={"video": ("a.txt", b"x", "text/plain")}).status_code == 415
    assert client.post("/api/vision/jobs", files={"video": ("a.mp4", b"", "video/mp4")}).status_code == 422
    bad_q = client.post("/api/vision/jobs", files={"video": ("a.mp4", b"x", "video/mp4")},
                        data={"questions": '[{"question_id": "Q1", "start": 10, "end": 5}]'})
    assert bad_q.status_code == 422
    bad_photo = client.post("/api/vision/jobs", files={"video": ("a.mp4", b"x", "video/mp4"),
                                                       "reference_photo": ("p.gif", b"x", "image/gif")})
    assert bad_photo.status_code == 415
    assert client.get("/api/vision/jobs/does-not-exist").status_code == 404


def test_analysis_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr("backend.adapters.video_intake.analyze_video", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    jid = client.post("/api/vision/jobs", files={"video": ("a.mp4", b"\x00", "video/mp4")}).json()["job_id"]
    j = _wait(jid)
    assert j["status"] == "error" and "boom" in j["error"] and j["result"] is None


def test_health_reports_models_and_checks():
    h = client.get("/api/vision/health").json()
    assert set(h["checks_available"]) == {"face_identity", "lip_sync", "gaze_and_eye_cues", "phone_and_people"}
    assert isinstance(h["ready"], bool) and "yolov3" in h["models"]


def test_build_result_handles_missing_video_signals():
    res = api_vision.build_result({"meta": {}})
    assert res.deepfake.status == "INSUFFICIENT_EVIDENCE" and res.timeline == []
