"""
Member 5 guardrail tests: the per-component contribution breakdown must sum
exactly to the fused risk_score (so the XAI explanation's numbers add up),
and the explainer must never alter the underlying score.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.modules import evidence_fusion, xai_explainer
from backend.schemas import ModuleResult, Evidence, Quality


def _mod(score, level, confidence=0.8, status="OK", data_quality=1.0, baseline_sufficient=True, evidence=None):
    return ModuleResult(
        feature="x", score=score, confidence=confidence, level=level, status=status,
        quality=Quality(data_quality=data_quality, baseline_sufficient=baseline_sufficient),
        evidence=evidence or [Evidence(type="t", severity="low", message="ok")],
    )


def _fuse(**overrides):
    base = dict(
        identity=_mod(10, "LOW"), sync=_mod(10, "LOW"), behavior=_mod(10, "LOW"),
        speech=_mod(10, "LOW"), knowledge_ownership=_mod(10, "LOW"), profile_consistency=_mod(10, "LOW"),
    )
    base.update(overrides)
    return evidence_fusion.fuse(candidate_name="Test", capability_score=50.0, **base)


def test_contributions_sum_to_risk_score_all_available():
    a = _fuse(identity=_mod(80, "HIGH"), sync=_mod(20, "LOW"))
    assert round(sum(a.component_contributions.values()), 1) == round(a.risk_score, 1)


def test_contributions_sum_to_risk_score_with_corroboration_boost():
    a = _fuse(identity=_mod(90, "HIGH"), sync=_mod(90, "HIGH"), knowledge_ownership=_mod(90, "HIGH"))
    assert round(sum(a.component_contributions.values()), 1) == round(a.risk_score, 1)


def test_contributions_sum_to_risk_score_with_unavailable_module():
    a = _fuse(sync=_mod(0, "UNAVAILABLE", status="UNAVAILABLE"))
    assert round(sum(a.component_contributions.values()), 1) == round(a.risk_score, 1)
    assert "sync_risk" not in a.component_contributions


def test_confidence_penalized_by_low_data_quality():
    high_quality = _fuse()
    low_quality = _fuse(identity=_mod(10, "LOW", data_quality=0.2))
    assert low_quality.confidence < high_quality.confidence


def test_xai_explainer_does_not_change_the_score():
    from backend.modules.deepfake_fusion import fuse_deepfake
    identity = _mod(70, "HIGH")
    sync = _mod(70, "HIGH")
    a = _fuse(identity=identity, sync=sync)
    deepfake = fuse_deepfake(identity, sync, None)
    explanation = xai_explainer.explain(a, deepfake)
    assert explanation.risk_score == a.risk_score
    assert explanation.risk_level == a.risk_level
    assert explanation.confidence == a.confidence
    assert len(explanation.components) == len(a.components)


def test_xai_explainer_tags_unavailable_components():
    from backend.modules.deepfake_fusion import fuse_deepfake
    a = _fuse(behavior=_mod(0, "UNAVAILABLE", status="UNAVAILABLE"))
    deepfake = fuse_deepfake(_mod(10, "LOW"), _mod(10, "LOW"), None)
    explanation = xai_explainer.explain(a, deepfake)
    behavior_row = next(c for c in explanation.components if c.key == "behavior_risk")
    assert behavior_row.tag == "insufficient_evidence"


def test_xai_explainer_every_component_has_evidence_text():
    from backend.modules.deepfake_fusion import fuse_deepfake
    a = _fuse()
    deepfake = fuse_deepfake(_mod(10, "LOW"), _mod(10, "LOW"), None)
    explanation = xai_explainer.explain(a, deepfake)
    for c in explanation.components:
        assert len(c.top_evidence) >= 1
        assert c.top_evidence[0]
