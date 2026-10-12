"""
Shared Pydantic data contracts. Every analysis module returns a ModuleResult
(score, confidence, evidence[]) — never a final decision (Build Spec Section 3, rule 3).
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel

Severity = Literal["low", "medium", "high"]
Level = Literal["LOW", "MEDIUM", "HIGH", "UNAVAILABLE"]


class Evidence(BaseModel):
    type: str
    severity: Severity
    message: str
    question_id: Optional[str] = None
    timestamp: Optional[float] = None


class Quality(BaseModel):
    data_quality: float = 1.0          # 0-1, e.g. audio/video quality
    baseline_sufficient: bool = True


class ModuleResult(BaseModel):
    feature: str
    score: float                        # 0-100 risk (or capability) score
    confidence: float                   # 0-1
    level: Level
    evidence: List[Evidence] = []
    quality: Quality = Quality()
    status: Literal["OK", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"] = "OK"


class ResumeClaim(BaseModel):
    skill: str
    claim_text: str
    claim_strength: Literal["weak", "medium", "strong"]
    source: str = "resume"


class SkillProfile(BaseModel):
    skill_match_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    resume_claims: List[ResumeClaim]


class QAExchange(BaseModel):
    question_id: str
    question_text: str
    skill: str
    difficulty: str
    expected_proficiency: str
    reference_answer: str
    required_concepts: List[str]
    candidate_answer: str
    response_latency_sec: float
    answer_duration_sec: float


class AnswerEvaluation(BaseModel):
    question_id: str
    score: float
    concept_coverage: float
    missing_concepts: List[str]
    incorrect_concepts: List[str]
    confidence: float
    # Member 4 additions — richer answer intelligence
    semantic_depth: float = 0.0        # 0-1: how semantically rich the answer is vs reference
    length_adequacy: float = 0.0       # 0-1: penalises too-short or suspiciously padded answers
    ai_likelihood: float = 0.0         # 0-1: probability this answer was AI-generated / read


class AIAnswerFlag(BaseModel):
    """Per-answer AI-detection signal package (Member 4 / F7)."""
    question_id: str
    ai_likelihood: float               # 0.0 – 1.0
    flags: List[str]                   # human-readable triggered rule names
    severity: Severity
    latency_suspicious: bool = False
    vocab_richness_z: float = 0.0      # z-score vs candidate's own baseline; high z = richer than usual
    sentence_uniformity: float = 0.0   # 0-1, high = suspiciously uniform sentence lengths (AI trait)
    perfect_structure_score: float = 0.0  # 0-1, well-formed list-like structure without natural hesitation
    answer_too_polished: bool = False   # True if multiple signals converge on "too perfect"


class AIDetectionResult(BaseModel):
    """Module-level result from the AI answer detector (Member 4 / F7)."""
    feature: str = "ai_answer_detection"
    overall_ai_likelihood: float       # 0-1, average across flagged answers
    level: Level
    confidence: float
    per_answer_flags: List[AIAnswerFlag]
    evidence: List[Evidence] = []
    status: Literal["OK", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"] = "OK"


class SkillVerdict(BaseModel):
    skill: str
    verdict: Literal[
        "STRONGLY_VERIFIED", "VERIFIED", "PARTIALLY_VERIFIED",
        "WEAKLY_SUPPORTED", "RESUME_INTERVIEW_MISMATCH", "NOT_ASSESSED"
    ]
    evidence_score: float
    questions_used: List[str]


class RiskAssessment(BaseModel):
    candidate_name: str
    risk_score: float
    risk_level: Level
    confidence: float
    capability_score: float
    components: dict
    evidence_timeline: List[Evidence]
    recommended_action: str
