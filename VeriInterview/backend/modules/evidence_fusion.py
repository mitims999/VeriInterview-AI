"""
Evidence Fusion Engine (Master Spec Section 14 / Build Spec Section 4).

THE CENTRAL RULE: no single module decides cheating. This engine is a
deterministic weighted formula — never LLM-controlled (Master Spec Section
31, Build Spec instruction 5). Every module contributes (score, confidence,
evidence[]); this engine combines them into one 0-100 Integrity Risk Score
+ a separate Confidence score, and reports Candidate Capability separately
(never merged into the risk number, per Build Spec Section 4).
"""
from __future__ import annotations
from backend.config import FUSION_WEIGHTS, FUSION_CAPS, RISK_THRESHOLDS
from backend.schemas import ModuleResult, Evidence, RiskAssessment


def _level_for(score: float) -> str:
    if score <= RISK_THRESHOLDS.low_max:
        return "LOW"
    if score <= RISK_THRESHOLDS.medium_max:
        return "MEDIUM"
    return "HIGH"


def fuse(
    candidate_name: str,
    identity: ModuleResult,
    sync: ModuleResult,
    behavior: ModuleResult,
    speech: ModuleResult,
    knowledge_ownership: ModuleResult,
    profile_consistency: ModuleResult,
    capability_score: float,
) -> RiskAssessment:
    components = {
        "identity_risk": identity,
        "sync_risk": sync,
        "behavior_risk": behavior,
        "speech_behavior_risk": speech,
        "knowledge_ownership_risk": knowledge_ownership,
        "profile_consistency_risk": profile_consistency,
    }
    weights = {
        "identity_risk": FUSION_WEIGHTS.identity_risk,
        "sync_risk": FUSION_WEIGHTS.sync_risk,
        "behavior_risk": FUSION_WEIGHTS.behavior_risk,
        "speech_behavior_risk": FUSION_WEIGHTS.speech_behavior_risk,
        "knowledge_ownership_risk": FUSION_WEIGHTS.knowledge_ownership_risk,
        "profile_consistency_risk": FUSION_WEIGHTS.profile_consistency_risk,
    }

    available = {k: v for k, v in components.items() if v.status == "OK"}
    if not available:
        return RiskAssessment(
            candidate_name=candidate_name, risk_score=0, risk_level="LOW", confidence=0.0,
            capability_score=capability_score,
            components={k: v.model_dump() for k, v in components.items()},
            evidence_timeline=[], recommended_action="Insufficient evidence collected — re-run analysis before review.",
        )

    # Redistribute weight of unavailable modules proportionally across available ones.
    total_available_weight = sum(weights[k] for k in available)
    weighted_sum = 0.0
    for k, mod in available.items():
        module_weight = weights[k] / total_available_weight
        # Cap: no single module's contribution can exceed FUSION_CAPS.max_single_module_contribution
        # of the *total* fused score (prevents one noisy signal from dominating).
        capped_weight = min(module_weight, FUSION_CAPS.max_single_module_contribution)
        weighted_sum += capped_weight * mod.score
    # renormalize after capping so scores still land in 0-100
    cap_norm = sum(min(weights[k] / total_available_weight, FUSION_CAPS.max_single_module_contribution) for k in available)
    risk_score = weighted_sum / cap_norm if cap_norm else 0.0

    # Corroboration principle (Master Spec Section 33): one HIGH signal alone
    # must not decide anything, but when MULTIPLE INDEPENDENT modules
    # independently land HIGH, that agreement is itself evidence and modestly
    # increases the fused score (capped so it cannot alone cross a bucket
    # boundary from a single corroborating pair).
    high_count = sum(1 for mod in available.values() if mod.level == "HIGH")
    if high_count >= 2:
        risk_score = min(100.0, risk_score * (1.0 + 0.15 * (high_count - 1)))

    # Confidence: weighted average of module confidences, penalized by missing-module coverage
    coverage_ratio = len(available) / len(components)
    conf_weighted = sum(weights[k] * mod.confidence for k, mod in available.items()) / total_available_weight
    # agreement bonus: if module confidences disagree wildly, reduce overall confidence
    confs = [mod.confidence for mod in available.values()]
    spread_penalty = (max(confs) - min(confs)) * 0.15 if len(confs) > 1 else 0.0
    confidence = max(0.05, min(0.99, conf_weighted * (0.5 + 0.5 * coverage_ratio) - spread_penalty))

    risk_level = _level_for(risk_score)

    timeline: list[Evidence] = []
    for mod in components.values():
        timeline.extend(mod.evidence)
    timeline.sort(key=lambda e: (e.timestamp is None, e.timestamp or 0))

    recommended_action = _recommend(risk_level, risk_score, confidence)

    return RiskAssessment(
        candidate_name=candidate_name,
        risk_score=round(risk_score, 1),
        risk_level=risk_level,
        confidence=round(confidence, 2),
        capability_score=round(capability_score, 1),
        components={k: v.model_dump() for k, v in components.items()},
        evidence_timeline=timeline,
        recommended_action=recommended_action,
    )


def _recommend(level: str, score: float, confidence: float) -> str:
    if level == "LOW":
        return "No targeted verification required. Standard review process applies."
    if level == "MEDIUM" and confidence < 0.5:
        return "Signals are mixed and evidence is incomplete — collect additional data before deciding on verification."
    if level == "MEDIUM":
        return "Review flagged evidence with the interviewer before proceeding; consider one targeted follow-up question."
    # HIGH
    if confidence < 0.5:
        return "High risk score but low confidence — evidence is inconsistent or incomplete. Do not act on this score alone; gather more signal."
    return "Conduct a targeted verification interview focusing on the flagged skills and behavioral evidence before making any decision."
