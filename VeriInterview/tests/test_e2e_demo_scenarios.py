"""
End-to-end demo tests over the 5 synthetic candidates (Master Spec Section 26/45).
Asserts the robustness properties the demo is designed to prove, not exact score values.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.modules.pipeline import run_full_analysis

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")


def _load(name):
    with open(os.path.join(DATA_DIR, f"{name}.json"), encoding="utf-8") as f:
        return run_full_analysis(json.load(f))


def test_genuine_strong_is_low_risk_high_capability():
    r = _load("genuine")
    assert r["assessment"]["risk_level"] == "LOW"
    assert r["assessment"]["capability_score"] > 40


def test_genuine_weak_is_low_risk_despite_low_capability():
    r = _load("genuine_weak")
    assert r["assessment"]["risk_level"] == "LOW"
    assert r["assessment"]["capability_score"] < 30


def test_ai_assisted_is_at_least_medium_risk():
    r = _load("ai_assisted")
    assert r["assessment"]["risk_level"] in ("MEDIUM", "HIGH")


def test_deepfake_shows_high_av_and_identity_risk():
    r = _load("deepfake")
    comps = r["assessment"]["components"]
    assert comps["identity_risk"]["level"] == "HIGH"
    assert comps["sync_risk"]["level"] == "HIGH"


def test_nervous_genuine_does_not_automatically_become_high_risk():
    r = _load("nervous_genuine")
    assert r["assessment"]["risk_level"] != "HIGH"
    assert r["assessment"]["components"]["speech_behavior_risk"]["level"] == "HIGH"
