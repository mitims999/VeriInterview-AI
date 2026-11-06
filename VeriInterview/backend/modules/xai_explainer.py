"""
Explainability engine (Member 5 / Feature 6 — Confidence + XAI).

Master Spec Section 27 rule: a report must distinguish a MEASURED FEATURE
from a MODEL INFERENCE from a RECOMMENDED ACTION, and every high/medium
evidence signal needs a reason — never just a bare number.

This module is deliberately dumb: it only reads the numbers and evidence
strings evidence_fusion.py and deepfake_fusion.py already computed and turns
them into a structured, plain-English breakdown. No LLM, no new scoring —
Master Spec Section 31 ("LLM should NOT own the final risk score") applies
transitively here too: the explainer must not be able to change the score,
only describe it.
"""
from __future__ import annotations
from backend.config import FUSION_WEIGHTS
from backend.schemas import (
    RiskAssessment, ModuleResult, ComponentExplanation, AuthenticityExplanation, XAIExplanation,
)

_LABELS = {
    "identity_risk": "Face Identity (InsightFace)",
    "sync_risk": "A/V Lip-Sync (SyncNet)",
    "behavior_risk": "Gaze / Behavior (Proctoring-AI)",
    "speech_behavior_risk": "Voice Modulation (openSMILE)",
    "knowledge_ownership_risk": "Knowledge Ownership (CCST + AI-Answer Detection)",
    "profile_consistency_risk": "Resume / Answer Consistency (+ Keyword Cross-Reference)",
}

_NOMINAL_WEIGHTS = {
    "identity_risk": FUSION_WEIGHTS.identity_risk,
    "sync_risk": FUSION_WEIGHTS.sync_risk,
    "behavior_risk": FUSION_WEIGHTS.behavior_risk,
    "speech_behavior_risk": FUSION_WEIGHTS.speech_behavior_risk,
    "knowledge_ownership_risk": FUSION_WEIGHTS.knowledge_ownership_risk,
    "profile_consistency_risk": FUSION_WEIGHTS.profile_consistency_risk,
}


def _component_row(key: str, mod_dict: dict, contribution: float) -> ComponentExplanation:
    status = mod_dict["status"]
    evidence = mod_dict.get("evidence", [])
    # Evidence worth surfacing in the explanation: non-trivial severities first,
    # falling back to the first "low" message if nothing more severe fired —
    # a component with no findings still needs to say so (never a bare number).
    ranked = sorted(evidence, key=lambda e: {"high": 0, "medium": 1, "low": 2}[e["severity"]])
    top_evidence = [e["message"] for e in ranked[:3]]
    if not top_evidence:
        top_evidence = ["No evidence recorded for this component."]

    if status != "OK":
        tag = "insufficient_evidence"
    elif mod_dict["level"] in ("MEDIUM", "HIGH"):
        tag = "inference"  # the module's score is itself a model inference over measured signals
    else:
        tag = "measured"

    return ComponentExplanation(
        key=key,
        label=_LABELS.get(key, key),
        status=status,
        level=mod_dict["level"],
        raw_score=mod_dict["score"],
        weight=_NOMINAL_WEIGHTS.get(key, 0.0),
        contribution_points=round(contribution, 2),
        confidence=mod_dict["confidence"],
        top_evidence=top_evidence,
        tag=tag,
    )


def _authenticity_row(deepfake: ModuleResult) -> AuthenticityExplanation:
    d = deepfake.model_dump() if hasattr(deepfake, "model_dump") else deepfake
    summary_evidence = next((e["message"] for e in d.get("evidence", []) if e["type"] == "deepfake_summary"), None)
    if summary_evidence is None:
        summary_evidence = d["evidence"][0]["message"] if d.get("evidence") else "No authenticity signal available."
    return AuthenticityExplanation(
        score=d["score"], level=d["level"], status=d["status"],
        summary=summary_evidence, sub_scores=(d.get("details") or {}).get("sub_scores", {}),
    )


def _narrative(assessment: RiskAssessment, components: list[ComponentExplanation]) -> str:
    drivers = sorted(
        [c for c in components if c.status == "OK"],
        key=lambda c: c.contribution_points, reverse=True,
    )[:2]
    if not drivers or assessment.risk_level == "LOW":
        if assessment.risk_level == "LOW":
            return (f"{assessment.candidate_name}'s integrity risk is {assessment.risk_score:.0f}/100 (LOW) — "
                    f"no component independently or jointly supports elevated concern.")
        return f"{assessment.candidate_name}'s integrity risk is {assessment.risk_score:.0f}/100; evidence is too sparse to attribute to specific components."

    driver_text = " and ".join(f"{c.label} ({c.contribution_points:.0f} pts)" for c in drivers)
    corroboration_note = ""
    high_components = [c for c in components if c.status == "OK" and c.level == "HIGH"]
    if len(high_components) >= 2:
        corroboration_note = (f" {len(high_components)} independent signals agree at HIGH, which is why the score "
                               f"is higher than any single component alone would justify.")
    return (f"{assessment.candidate_name}'s integrity risk is {assessment.risk_score:.0f}/100 ({assessment.risk_level}), "
            f"driven mainly by {driver_text}.{corroboration_note}")


def explain(assessment: RiskAssessment, deepfake: ModuleResult) -> XAIExplanation:
    components = [
        _component_row(key, mod_dict, assessment.component_contributions.get(key, 0.0))
        for key, mod_dict in assessment.components.items()
    ]
    narrative = _narrative(assessment, components)
    return XAIExplanation(
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        confidence=assessment.confidence,
        narrative=narrative,
        components=components,
        authenticity=_authenticity_row(deepfake),
        recommended_action=assessment.recommended_action,
    )
