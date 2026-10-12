"""
Build Spec Section 3.7 — Claim Graph + Counterfactual Stress Test (CCST).

This is the project's central novelty (Master Spec Section 13/32) and is kept
as original code — no repo covers it. Not skipped, not simplified.

Pipeline:
  resume_claims + answers -> claim graph (networkx: claim -> concept edges)
  -> pick a perturbation (one changed assumption, per fixture)
  -> counterfactual follow-up question (already asked in the synthetic
     transcript; in production this would be LLM-generated per Master Spec
     Section 13)
  -> compare candidate's follow-up answer against the concepts a genuine
     expert would need to introduce (adaptation) and the concepts that
     should remain stable (consistency)
  -> Knowledge Ownership Score (KOS) and Causal Adaptation Consistency (CAC)

INTEGRATION NOTE: question *generation* is an LLM job in production
(Master Spec Section 31: LLM never owns the score, only generates text).
No LLM_API_KEY is configured in this environment, so the counterfactual
question/expected-concept sets are pre-authored in the synthetic fixtures
exactly as the interviewer/LLM pipeline would have produced them; the
consistency SCORING below is fully deterministic either way.
"""
from __future__ import annotations
import networkx as nx
from backend.adapters.sentence_embed import concept_coverage
from backend.schemas import ModuleResult, Evidence, Quality


def build_claim_graph(resume_claims: list[dict], counterfactuals: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for c in resume_claims:
        g.add_node(c["skill"], kind="claim")
    for cf in counterfactuals:
        claim_skill = cf.get("skill", "unknown")
        g.add_node(claim_skill, kind="claim")
        for concept in cf.get("expected_new_concepts", []) + cf.get("expected_retained_concepts", []):
            g.add_node(concept, kind="concept")
            g.add_edge(claim_skill, concept, relation="depends_on")
    return g


def evaluate_counterfactuals(counterfactuals: list[dict]) -> ModuleResult:
    """
    counterfactuals: list of dicts with keys:
      skill, perturbed_assumption, followup_question, candidate_followup_answer,
      expected_new_concepts, expected_retained_concepts, original_answer_score (0-100, optional)

    Gating rule: CCST measures whether a candidate's ORIGINAL strong-sounding
    answer reflects genuine ownership, not whether they are a strong engineer
    overall. If the original answer was already weak (low original_answer_score),
    failing the follow-up is expected and consistent — it is a CAPABILITY
    signal, not an integrity signal (a genuinely weak candidate is not
    "faking"). Only a HIGH original score paired with a LOW follow-up
    consistency indicates a fabricated-ownership pattern.
    """
    if not counterfactuals:
        return ModuleResult(
            feature="knowledge_ownership", score=0, confidence=0.0, level="UNAVAILABLE",
            status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(type="knowledge_ownership", severity="low",
                                message="No counterfactual/follow-up probes were run for this interview.")],
            quality=Quality(data_quality=0.0, baseline_sufficient=False),
        )

    per_probe_risk = []
    evidence = []
    for cf in counterfactuals:
        answer = cf.get("candidate_followup_answer", "")
        original_score = cf.get("original_answer_score", 70)  # assume demonstrated if not provided
        new_coverage, missing_new = concept_coverage(answer, cf.get("expected_new_concepts", []))
        retained_coverage, missing_retained = concept_coverage(answer, cf.get("expected_retained_concepts", []))
        cac = (new_coverage + retained_coverage) / 2.0

        if original_score < 50:
            # Candidate never appeared to strongly know this — a weak follow-up is
            # consistent (capability gap), not evidence of fabricated ownership.
            probe_risk = 15.0
            evidence.append(Evidence(
                type="counterfactual", severity="low", question_id=cf.get("followup_question_id"),
                message=f"Candidate's original answer on {cf.get('skill')} was already weak; follow-up consistency "
                        f"is not treated as an integrity signal.",
            ))
        else:
            probe_risk = 100.0 * (1.0 - cac)
            if cac >= 0.75:
                evidence.append(Evidence(
                    type="counterfactual", severity="low", question_id=cf.get("followup_question_id"),
                    message=f"Candidate adapted reasoning consistently when '{cf.get('perturbed_assumption')}' was introduced "
                            f"for {cf.get('skill')}.",
                ))
            elif cac >= 0.45:
                evidence.append(Evidence(
                    type="counterfactual", severity="medium", question_id=cf.get("followup_question_id"),
                    message=f"Candidate's follow-up reasoning on {cf.get('skill')} was only partially consistent "
                            f"after the assumption change (missing: {', '.join(missing_new) or 'none'}).",
                ))
            else:
                evidence.append(Evidence(
                    type="counterfactual", severity="high", question_id=cf.get("followup_question_id"),
                    message=f"Candidate gave a strong original answer on {cf.get('skill')} but could not adapt "
                            f"reasoning after '{cf.get('perturbed_assumption')}' — suggests the original answer may "
                            f"not reflect genuine ownership of the concept.",
                ))
        per_probe_risk.append(probe_risk)

    risk = sum(per_probe_risk) / len(per_probe_risk)
    level = "LOW" if risk <= 30 else ("MEDIUM" if risk <= 60 else "HIGH")

    return ModuleResult(
        feature="knowledge_ownership", score=round(risk, 1), confidence=0.85, level=level,
        evidence=evidence, quality=Quality(data_quality=1.0, baseline_sufficient=True),
    )
