"""
Member 4 — Feature 7: AI Answer Detection (new module).

Detects whether a candidate is reading AI-generated answers (ChatGPT, Gemini,
Copilot, etc.) during their interview, using purely stylometric and
behavioural signals — no external API, no internet required.

Detection Signals
─────────────────
1.  Vocabulary Richness Z-Score
    Each answer's Type-Token Ratio (TTR) is computed.  A z-score is calculated
    against the candidate's OWN baseline across all their answers.  A single
    answer with a much higher TTR than the candidate's baseline (z > 1.8)
    is suspicious — it means one answer sounds measurably more "polished" than
    the others.

2.  Sentence-Length Uniformity
    AI-generated text tends to have suspiciously uniform sentence lengths.
    We compute the coefficient of variation (CV = std/mean) of sentence lengths.
    Low CV (<0.30) → high uniformity → suspicious.

3.  Transition / Hedge Word Ratio
    Natural speech contains fillers, hedges, and discourse markers
    ("I think", "basically", "um", "so", "you know").
    AI text rarely includes these.  A very low ratio (<0.03) is a flag.

4.  Perfect-Structure Score
    AI answers often start with a definition sentence, followed by numbered/
    bulleted points and a summary.  We detect structural patterns like:
    "X is ...", followed by listing with commas or "and" joining ≥3 items.

5.  Latency-Quality Compound Spike
    Imported from response_timing context: if response_latency_sec is ≥ 2x
    the candidate's own average AND the answer scores ≥ 75 on correctness,
    it strongly suggests the candidate looked something up / pasted from AI.

6.  Answer vs. Question Semantic Gap
    AI answers to complex questions often look great in isolation but miss
    the nuance of what was asked.  We detect cases where semantic similarity
    to the question is very low while the answer scores high — suggesting a
    "stock" AI answer that doesn't respond to the specific phrasing.

Each flag contributes partial likelihood.  The final per-answer ai_likelihood
is clamped to [0, 1].  The module-level level is set to:
  LOW    if overall_ai_likelihood < 0.30
  MEDIUM if overall_ai_likelihood < 0.60
  HIGH   otherwise

Design rule: this module NEVER makes a hiring decision.  It surfaces evidence
for human review with explanations for each flag raised.
"""
from __future__ import annotations
import re
import math
from statistics import mean, pstdev
from backend.schemas import (
    QAExchange, AnswerEvaluation, AIAnswerFlag, AIDetectionResult, Evidence
)
from backend.adapters.sentence_embed import semantic_similarity


# ── Configurable thresholds ───────────────────────────────────────────────────
VOCAB_RICHNESS_Z_THRESHOLD = 1.8     # z-score above which TTR is suspicious
SENTENCE_UNIFORMITY_THRESHOLD = 0.30 # CV below this → suspiciously uniform
HEDGE_RATIO_THRESHOLD = 0.03         # fraction of tokens that are hedges
LATENCY_SPIKE_FACTOR = 2.0           # latency must be ≥ X times candidate mean
LATENCY_QUALITY_SCORE_THRESHOLD = 72 # score above which latency spike is suspicious
QUESTION_SEMANTIC_GAP_THRESHOLD = 0.15  # sim to question below this = possible stock answer
POLISHED_FLAG_COUNT_THRESHOLD = 3    # ≥ this many flags → answer_too_polished = True


# ── Hedge / filler / discourse-marker words (natural speech markers) ──────────
HEDGE_WORDS = {
    "i", "think", "basically", "actually", "honestly", "um", "uh", "so",
    "you", "know", "like", "really", "kind", "sort", "maybe", "probably",
    "generally", "typically", "usually", "often", "quite", "fairly", "pretty",
    "perhaps", "i'd", "i've", "we'd", "i'm", "let's", "right", "okay",
    "well", "anyway", "obviously", "essentially", "roughly", "more", "less",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9']+\b", text.lower())


def _type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _sentence_lengths(text: str) -> list[int]:
    sentences = re.split(r"[.!?;]+", text.strip())
    return [len(s.split()) for s in sentences if len(s.split()) >= 2]


def _coefficient_of_variation(lengths: list[int]) -> float:
    """Low CV = uniform sentence lengths (suspicious for AI)."""
    if len(lengths) < 2:
        return 1.0                        # single sentence → not enough to judge
    m = mean(lengths)
    if m == 0:
        return 1.0
    return pstdev(lengths) / m


def _hedge_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    hedge_count = sum(1 for t in tokens if t in HEDGE_WORDS)
    return hedge_count / len(tokens)


def _perfect_structure_score(answer: str) -> float:
    """
    Heuristic to detect AI's characteristic answer structure:
      - Starts with a definitional sentence ("X is …" or "X refers to …")
      - Contains an enumeration of ≥3 items joined by commas or "and"
      - Ends with a summary or conclusion
    Each pattern contributes 0.33 to the 0-1 score.
    """
    score = 0.0
    text = answer.strip()

    # 1. Definitional opener
    definitional = re.match(
        r"^[A-Z][^.!?]{5,80}(?: is | refers to | means | involves | occurs when )",
        text
    )
    if definitional:
        score += 0.33

    # 2. Enumeration (≥3 items separated by commas or "and/or")
    enum_matches = re.findall(r"\b\w[\w\s]{1,20},\s*\w[\w\s]{1,20},\s*(?:and|or)?\s*\w", text)
    if enum_matches:
        score += 0.33

    # 3. Summary / closing phrase
    summary_phrases = [
        "in summary", "in conclusion", "to summarize", "overall",
        "this ensures", "this allows", "this approach", "together,",
        "by doing this", "which results in", "as a result,",
    ]
    text_lower = text.lower()
    if any(p in text_lower for p in summary_phrases):
        score += 0.34

    return round(min(1.0, score), 2)


# ── Per-answer analyser ───────────────────────────────────────────────────────

def _analyse_one(
    qa: QAExchange,
    evaluation: AnswerEvaluation,
    candidate_baseline_ttr: float,
    candidate_ttr_std: float,
    candidate_avg_latency: float,
) -> AIAnswerFlag:
    """
    Runs all 6 detection signals on a single Q&A pair.
    Returns an AIAnswerFlag with the combined likelihood and individual flags.
    """
    answer = qa.candidate_answer
    tokens = _tokenise(answer)
    flags: list[str] = []
    likelihood_components: list[float] = []

    # ── Signal 1: Vocabulary Richness Z-Score ─────────────────────────────────
    ttr = _type_token_ratio(tokens)
    if candidate_ttr_std > 0.0:
        vocab_z = (ttr - candidate_baseline_ttr) / candidate_ttr_std
    else:
        vocab_z = 0.0

    if vocab_z > VOCAB_RICHNESS_Z_THRESHOLD:
        flags.append(f"VOCAB_RICHNESS_HIGH (z={vocab_z:.2f}): vocabulary is significantly richer than candidate's own baseline")
        likelihood_components.append(min(0.35, 0.15 + 0.05 * (vocab_z - VOCAB_RICHNESS_Z_THRESHOLD)))

    # ── Signal 2: Sentence Uniformity ────────────────────────────────────────
    sent_lengths = _sentence_lengths(answer)
    cv = _coefficient_of_variation(sent_lengths)
    uniformity = max(0.0, 1.0 - cv)   # high when CV is low

    if cv < SENTENCE_UNIFORMITY_THRESHOLD and len(sent_lengths) >= 3:
        flags.append(f"SENTENCE_UNIFORMITY_HIGH (cv={cv:.2f}): sentence lengths are unusually uniform -- AI trait")
        likelihood_components.append(0.20)

    # ── Signal 3: Low Hedge / Filler Ratio ────────────────────────────────────
    hedge = _hedge_ratio(tokens)
    if hedge < HEDGE_RATIO_THRESHOLD and len(tokens) >= 15:
        flags.append(f"LOW_HEDGE_RATIO ({hedge:.3f}): almost no natural discourse markers -- atypical for unscripted speech")
        likelihood_components.append(0.20)

    # ── Signal 4: Perfect Structure ──────────────────────────────────────────
    struct_score = _perfect_structure_score(answer)
    if struct_score >= 0.65:
        flags.append(f"PERFECT_STRUCTURE ({struct_score:.2f}): answer follows a textbook definition -> enum -> summary pattern")
        likelihood_components.append(0.25)

    # ── Signal 5: Latency-Quality Compound Spike ─────────────────────────────
    latency_suspicious = False
    if candidate_avg_latency > 0 and qa.response_latency_sec >= candidate_avg_latency * LATENCY_SPIKE_FACTOR:
        if evaluation.score >= LATENCY_QUALITY_SCORE_THRESHOLD:
            latency_suspicious = True
            flags.append(
                f"LATENCY_QUALITY_SPIKE (latency={qa.response_latency_sec:.1f}s, "
                f"avg={candidate_avg_latency:.1f}s, score={evaluation.score}): "
                f"unusually long pause before a high-quality answer -- possible lookup"
            )
            likelihood_components.append(0.35)

    # ── Signal 6: Semantic Gap to Question ────────────────────────────────────
    q_similarity = semantic_similarity(answer, qa.question_text)
    if q_similarity < QUESTION_SEMANTIC_GAP_THRESHOLD and evaluation.score >= 60:
        flags.append(
            f"QUESTION_SEMANTIC_GAP (sim={q_similarity:.2f}): answer scores well on the reference "
            f"but is semantically distant from the actual question -- possible stock/generic AI answer"
        )
        likelihood_components.append(0.20)

    # ── Combine ───────────────────────────────────────────────────────────────
    # Probabilities are partially independent, so we use complement product:
    # P(at least one true) = 1 - prod(1 - p_i)  -- then we cap at 0.95
    complement = 1.0
    for p in likelihood_components:
        complement *= (1.0 - p)
    ai_likelihood = round(min(0.95, 1.0 - complement), 3) if likelihood_components else 0.0

    # Severity
    if ai_likelihood >= 0.60:
        severity = "high"
    elif ai_likelihood >= 0.30:
        severity = "medium"
    else:
        severity = "low"

    too_polished = len(flags) >= POLISHED_FLAG_COUNT_THRESHOLD

    return AIAnswerFlag(
        question_id=qa.question_id,
        ai_likelihood=ai_likelihood,
        flags=flags,
        severity=severity,
        latency_suspicious=latency_suspicious,
        vocab_richness_z=round(vocab_z, 3),
        sentence_uniformity=round(uniformity, 3),
        perfect_structure_score=struct_score,
        answer_too_polished=too_polished,
    )


# ── Module entry-point ────────────────────────────────────────────────────────

def run(
    qa_list: list[QAExchange],
    evaluations: list[AnswerEvaluation],
) -> AIDetectionResult:
    """
    Runs AI-answer detection across all Q&A pairs.

    Parameters
    ----------
    qa_list     : QAExchange list (same order as evaluations)
    evaluations : AnswerEvaluation list (from answer_evaluation.evaluate_all)

    Returns
    -------
    AIDetectionResult with per-answer flags and module-level risk level.
    Also mutates each AnswerEvaluation.ai_likelihood in-place so downstream
    modules (pipeline, XAI) have it without a separate lookup.
    """
    if len(qa_list) < 2:
        return AIDetectionResult(
            overall_ai_likelihood=0.0,
            level="UNAVAILABLE",
            confidence=0.0,
            per_answer_flags=[],
            evidence=[Evidence(
                type="ai_detection",
                severity="low",
                message="Insufficient Q&A pairs to establish a candidate baseline for AI detection (need >= 2).",
            )],
            status="INSUFFICIENT_EVIDENCE",
        )

    eval_by_qid = {e.question_id: e for e in evaluations}

    # ── Compute candidate-level baselines ─────────────────────────────────────
    ttrs = [_type_token_ratio(_tokenise(qa.candidate_answer)) for qa in qa_list if qa.candidate_answer.strip()]
    baseline_ttr = mean(ttrs) if ttrs else 0.5
    ttr_std = pstdev(ttrs) if len(ttrs) > 1 else 0.1
    avg_latency = mean([qa.response_latency_sec for qa in qa_list])

    # ── Per-answer analysis ───────────────────────────────────────────────────
    per_flags: list[AIAnswerFlag] = []
    for qa in qa_list:
        ev = eval_by_qid.get(qa.question_id)
        if ev is None:
            continue
        flag = _analyse_one(qa, ev, baseline_ttr, ttr_std, avg_latency)
        per_flags.append(flag)
        ev.ai_likelihood = flag.ai_likelihood        # mutate in-place

    if not per_flags:
        return AIDetectionResult(
            overall_ai_likelihood=0.0,
            level="UNAVAILABLE",
            confidence=0.0,
            per_answer_flags=[],
            status="INSUFFICIENT_EVIDENCE",
        )

    # ── Module-level likelihood and level ─────────────────────────────────────
    likelihoods = [f.ai_likelihood for f in per_flags]
    # Weight by severity: high-likelihood answers should pull the average up
    weighted = sum(l ** 1.5 for l in likelihoods)        # mild non-linearity
    overall = round(min(0.98, weighted / max(1, len(likelihoods))), 3)

    if overall >= 0.60:
        level = "HIGH"
    elif overall >= 0.30:
        level = "MEDIUM"
    else:
        level = "LOW"

    # ── Build evidence list (one entry per suspicious answer) ─────────────────
    evidence: list[Evidence] = []
    for flag in per_flags:
        if flag.ai_likelihood >= 0.30:
            evidence.append(Evidence(
                type="ai_answer_detection",
                severity=flag.severity,
                question_id=flag.question_id,
                message=(
                    f"Q{flag.question_id}: AI-likelihood={flag.ai_likelihood:.2f}. "
                    + ("Answer flagged as 'too polished' (multiple signals converge). " if flag.answer_too_polished else "")
                    + "Triggered: " + "; ".join(flag.flags[:2])  # top 2 flags for readability
                    + ("..." if len(flag.flags) > 2 else ".")
                ),
            ))

    if not evidence:
        evidence.append(Evidence(
            type="ai_answer_detection",
            severity="low",
            message="No strong AI-answer indicators detected. Candidate answers appear consistent with natural speech patterns.",
        ))

    # Confidence is higher when we have more answers to compare against
    confidence = min(0.90, 0.50 + 0.08 * len(qa_list))

    return AIDetectionResult(
        overall_ai_likelihood=overall,
        level=level,
        confidence=round(confidence, 2),
        per_answer_flags=per_flags,
        evidence=evidence,
        status="OK",
    )
