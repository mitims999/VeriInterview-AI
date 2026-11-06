"""
Member 4 — Feature 9: Response Timing / Prompting Anomalies (upgraded).

Original logic retained. Additions:
  - Candidate's baseline is now computed with outlier exclusion (IQR-trimmed)
    so one genuine long answer doesn't warp the baseline.
  - Burst-silence pattern: very fast answers (<0.8 s) on hard questions
    flagged as "pre-typed" evidence.
  - read_answer_suspected composite: latency spike + quality jump + fast
    delivery (the full classic "look-it-up" or "copy-paste" pattern).
  - Confidence now scales with sample size (≥4 QA pairs → 0.85).
"""
from __future__ import annotations
from statistics import mean, pstdev, median
from backend.schemas import QAExchange, AnswerEvaluation, ModuleResult, Evidence, Quality


# ── Configurable thresholds ───────────────────────────────────────────────────
LATENCY_Z_THRESHOLD = 1.8          # z-score for latency spike
QUALITY_SPIKE_SCORE = 75.0         # answer score above which latency spike is suspicious
FAST_ANSWER_SEC = 0.8              # suspiciously fast response on any question
FAST_HARD_ANSWER_SEC = 1.2         # fast response on intermediate/advanced question
BURST_SCORE_THRESHOLD = 70.0       # suspicious fast + high quality


def _iqr_trimmed_stats(values: list[float]) -> tuple[float, float]:
    """
    Returns (mean, std) after removing values outside [Q1-1.5*IQR, Q3+1.5*IQR].
    Falls back to raw stats if fewer than 4 values.
    """
    if len(values) < 4:
        m = mean(values)
        s = pstdev(values) if len(values) > 1 else 1.0
        return m, s or 1.0

    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    trimmed = [v for v in values if lo <= v <= hi] or values
    m = mean(trimmed)
    s = pstdev(trimmed) if len(trimmed) > 1 else 1.0
    return m, s or 1.0


def analyze_timing(
    qa_list: list[QAExchange],
    evaluations: list[AnswerEvaluation],
) -> ModuleResult:
    """
    Analyses response latency patterns across the interview.

    Signals detected:
    ─────────────────
    1. LATENCY_QUALITY_SPIKE  — high latency + high score (look-up suspected)
    2. LATENCY_SPIKE          — high latency alone (hesitation)
    3. BURST_PRETYPED         — very fast answer + high score (pre-typed suspected)
    4. FAST_HARD_QUESTION     — fast answer on a hard question (unexpected fluency)

    Signal 1 and 3 together produce read_answer_suspected=True in the Evidence
    extra field (carried as part of the message for now).
    """
    if len(qa_list) < 2:
        return ModuleResult(
            feature="response_timing", score=0, confidence=0.0, level="UNAVAILABLE",
            status="INSUFFICIENT_EVIDENCE",
            evidence=[Evidence(
                type="timing", severity="low",
                message="Not enough answered questions to assess timing patterns.",
            )],
        )

    eval_by_q = {e.question_id: e for e in evaluations}
    latencies = [qa.response_latency_sec for qa in qa_list]
    lm, ls = _iqr_trimmed_stats(latencies)

    evidence: list[Evidence] = []
    risk_points: list[float] = []
    read_suspected_count = 0

    for qa in qa_list:
        z = (qa.response_latency_sec - lm) / ls
        ev = eval_by_q.get(qa.question_id)
        score = ev.score if ev else 0.0
        difficulty = qa.difficulty.lower()

        spike = z > LATENCY_Z_THRESHOLD
        quality_jump = score >= QUALITY_SPIKE_SCORE
        very_fast = qa.response_latency_sec < FAST_ANSWER_SEC
        fast_hard = (
            qa.response_latency_sec < FAST_HARD_ANSWER_SEC
            and difficulty in ("intermediate", "advanced")
        )

        if spike and quality_jump:
            # Full latency + quality compound — strongest AI/assisted signal
            risk_points.append(88.0)
            read_suspected_count += 1
            evidence.append(Evidence(
                type="latency_quality_spike", severity="high",
                question_id=qa.question_id,
                message=(
                    f"[READ_ANSWER_SUSPECTED] {qa.question_id}: "
                    f"latency {qa.response_latency_sec:.1f}s (z={z:.1f}×σ above candidate mean {lm:.1f}s) "
                    f"followed by a high-quality answer (score {score:.0f}). "
                    f"Classic pattern for looking up or reading an AI-generated response."
                ),
            ))

        elif very_fast and quality_jump:
            # Suspiciously fast + good → pre-typed / pre-read
            risk_points.append(70.0)
            read_suspected_count += 1
            evidence.append(Evidence(
                type="burst_pretyped", severity="high",
                question_id=qa.question_id,
                message=(
                    f"[READ_ANSWER_SUSPECTED] {qa.question_id}: "
                    f"response came in {qa.response_latency_sec:.1f}s (unusually fast) "
                    f"with a high answer score ({score:.0f}). "
                    f"May indicate a pre-typed or pre-read answer."
                ),
            ))

        elif spike:
            risk_points.append(42.0)
            evidence.append(Evidence(
                type="latency_spike", severity="medium",
                question_id=qa.question_id,
                message=(
                    f"{qa.question_id}: latency {qa.response_latency_sec:.1f}s "
                    f"(z={z:.1f}) is well above this candidate's own average ({lm:.1f}s). "
                    f"Could indicate hesitation, external lookup, or technical difficulty."
                ),
            ))

        elif fast_hard:
            risk_points.append(30.0)
            evidence.append(Evidence(
                type="fast_hard_question", severity="medium",
                question_id=qa.question_id,
                message=(
                    f"{qa.question_id} (difficulty={qa.difficulty}): answered in "
                    f"{qa.response_latency_sec:.1f}s — unusually fast for a {difficulty} question. "
                    f"May indicate familiarity or pre-prepared answer."
                ),
            ))

        else:
            risk_points.append(5.0)

    score_val = sum(risk_points) / len(risk_points)
    level = "LOW" if score_val <= 30 else ("MEDIUM" if score_val <= 60 else "HIGH")

    if not any(e.type in ("latency_quality_spike", "burst_pretyped", "latency_spike", "fast_hard_question") for e in evidence):
        evidence.append(Evidence(
            type="timing", severity="low",
            message="Response latencies were consistent throughout the interview. No timing anomalies detected.",
        ))

    if read_suspected_count >= 2:
        evidence.insert(0, Evidence(
            type="read_answer_pattern", severity="high",
            message=(
                f"READ_ANSWER PATTERN DETECTED across {read_suspected_count} question(s): "
                f"multiple latency-quality spikes or burst-pretyped signals. "
                f"Strongly suggest cross-referencing with AI detection module."
            ),
        ))

    # Confidence scales with sample size
    confidence = min(0.88, 0.55 + 0.08 * len(qa_list))

    return ModuleResult(
        feature="response_timing",
        score=round(score_val, 1),
        confidence=round(confidence, 2),
        level=level,
        evidence=evidence,
        quality=Quality(data_quality=1.0, baseline_sufficient=len(qa_list) >= 3),
    )
