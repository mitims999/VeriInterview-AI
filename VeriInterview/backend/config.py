"""
Central configuration for VeriInterview.

All risk weights and thresholds live here (never hard-coded inline in modules),
per the master spec's "keep all risk weights and thresholds configurable" rule.
"""
import os
from pydantic import BaseModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "veriinterview.db")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")  # "none" -> deterministic fallback, no external call


class FusionWeights(BaseModel):
    """Evidence fusion weights (Build Spec section 4). Must sum to 1.0."""
    identity_risk: float = 0.20          # InsightFace adapter
    sync_risk: float = 0.15              # SyncNet adapter
    behavior_risk: float = 0.15          # Proctoring-AI adapter (gaze/head-pose/person-phone)
    speech_behavior_risk: float = 0.15   # openSMILE adapter
    knowledge_ownership_risk: float = 0.20  # Claim graph / CCST (original module)
    profile_consistency_risk: float = 0.15  # Resume-Matcher + expected-answer match


class FusionCaps(BaseModel):
    """Cap on any single module's contribution to the final score (0-1 scale of its own weight)."""
    max_single_module_contribution: float = 0.40  # no module may alone push risk above 40% of total


class RiskThresholds(BaseModel):
    low_max: int = 30       # 0-30 LOW
    medium_max: int = 60    # 31-60 MEDIUM
    # 61-100 HIGH


class SpeechBehaviorWeights(BaseModel):
    """Voice_Prosody spec Section 3.4 heuristic v1 formula."""
    pitch_deviation: float = 0.25
    loudness_deviation: float = 0.20
    speech_rate_deviation: float = 0.25
    pause_deviation: float = 0.20
    speech_ratio_deviation: float = 0.10


FUSION_WEIGHTS = FusionWeights()
FUSION_CAPS = FusionCaps()
RISK_THRESHOLDS = RiskThresholds()
SPEECH_WEIGHTS = SpeechBehaviorWeights()

# answer-evaluation weighting (semantic similarity vs concept coverage)
ANSWER_EVAL_SEMANTIC_WEIGHT = 0.5
ANSWER_EVAL_CONCEPT_WEIGHT = 0.5

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
