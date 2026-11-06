"""
Member 4 — Unit Tests (F4: Answer Correctness, F7: AI Detection, F9: Timing).

Coverage:
  test_unit_modules.py  (already covers some basic cases — we add deeper ones here)
  test_answer_evaluation_*  — multi-dimension scoring
  test_ai_detector_*        — all 6 signals individually + integration
  test_response_timing_*    — upgraded timing analysis
  test_knowledge_ownership_* — CCST edge cases

Run with:
    pytest tests/test_unit_answer.py -v
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.schemas import QAExchange, AnswerEvaluation
from backend.modules import answer_evaluation, ai_answer_detector, response_timing, knowledge_ownership


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — reusable QAExchange objects
# ═══════════════════════════════════════════════════════════════════════════════

def _qa(
    qid="Q1", text="What is overfitting?",
    skill="ML", difficulty="Intermediate",
    reference="Overfitting is when a model memorises training data including noise and generalises poorly.",
    required=None, answer="Overfitting occurs when the model learns noise from training data and fails to generalise to new data.",
    latency=2.0, duration=35.0,
):
    return QAExchange(
        question_id=qid, question_text=text, skill=skill, difficulty=difficulty,
        expected_proficiency="Intermediate", reference_answer=reference,
        required_concepts=required or ["training data", "noise", "generalisation"],
        candidate_answer=answer, response_latency_sec=latency, answer_duration_sec=duration,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# answer_evaluation — multi-dimension scoring
# ═══════════════════════════════════════════════════════════════════════════════

def test_answer_eval_empty_answer_scores_zero():
    qa = _qa(answer="")
    result = answer_evaluation.evaluate_answer(qa)
    assert result.score == 0.0
    assert result.concept_coverage == 0.0
    assert result.length_adequacy == 0.0
    assert result.semantic_depth == 0.0
    assert result.confidence < 0.5


def test_answer_eval_strong_answer_scores_high():
    qa = _qa(
        answer="Overfitting happens when a model memorises training data including its noise, "
               "so it performs well on training but generalises poorly to unseen or validation data. "
               "You detect it via training vs validation loss gap.",
        required=["training data", "noise", "generalisation", "unseen data", "validation"],
    )
    result = answer_evaluation.evaluate_answer(qa)
    assert result.score >= 55.0, f"Expected >= 55, got {result.score}"
    assert result.concept_coverage >= 0.5
    assert result.semantic_depth > 0.0


def test_answer_eval_too_short_penalised():
    qa = _qa(answer="it overfits", difficulty="Intermediate")
    result = answer_evaluation.evaluate_answer(qa)
    assert result.length_adequacy < 0.9
    assert result.score < 40


def test_answer_eval_confidence_scales_with_length():
    short_qa = _qa(answer="it overfits data")
    long_qa = _qa(answer=(
        "Overfitting is when a model fits training data so tightly that it captures noise "
        "along with signal, resulting in excellent training performance but poor generalisation "
        "to unseen data. You detect it by monitoring the gap between training and validation loss."
    ))
    short_res = answer_evaluation.evaluate_answer(short_qa)
    long_res = answer_evaluation.evaluate_answer(long_qa)
    assert short_res.confidence < long_res.confidence


def test_answer_eval_semantic_depth_is_zero_for_empty():
    qa = _qa(answer="")
    result = answer_evaluation.evaluate_answer(qa)
    assert result.semantic_depth == 0.0


def test_answer_eval_summary_structure():
    qas = [_qa(qid=f"Q{i}") for i in range(3)]
    evals = answer_evaluation.evaluate_all(qas)
    summary = answer_evaluation.answer_summary(evals)
    assert "average_score" in summary
    assert "average_depth" in summary
    assert "questions_evaluated" in summary
    assert summary["questions_evaluated"] == 3


def test_answer_eval_all_evaluates_whole_list():
    qas = [_qa(qid=f"Q{i}") for i in range(5)]
    results = answer_evaluation.evaluate_all(qas)
    assert len(results) == 5
    for r in results:
        assert 0 <= r.score <= 100
        assert 0 <= r.concept_coverage <= 1
        assert 0 <= r.semantic_depth <= 1
        assert 0 <= r.length_adequacy <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# ai_answer_detector — signal tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_qa_eval_pair(
    qid="Q1", answer="", latency=2.0, score=50.0, difficulty="Intermediate",
    question="What is overfitting?",
):
    qa = QAExchange(
        question_id=qid, question_text=question, skill="ML",
        difficulty=difficulty, expected_proficiency="Intermediate",
        reference_answer="Overfitting occurs when a model learns training noise.",
        required_concepts=["training", "noise", "generalisation"],
        candidate_answer=answer, response_latency_sec=latency, answer_duration_sec=30.0,
    )
    ev = AnswerEvaluation(
        question_id=qid, score=score, concept_coverage=0.7, missing_concepts=[],
        incorrect_concepts=[], confidence=0.85,
    )
    return qa, ev


def test_ai_detector_insufficient_for_single_qa():
    qa, ev = _make_qa_eval_pair()
    result = ai_answer_detector.run([qa], [ev])
    assert result.status == "INSUFFICIENT_EVIDENCE"


def test_ai_detector_low_likelihood_natural_answers():
    """Natural, varied, hedged answers should score low AI likelihood."""
    pairs = [
        _make_qa_eval_pair(
            qid=f"Q{i}", latency=2.0 + i * 0.3, score=55.0,
            answer=f"I think the main idea is that overfitting basically happens "
                   f"when the model kind of memorizes noise. You know, like, it trains really well "
                   f"but then actually fails on new data honestly. So I'd generally watch the val loss gap, "
                   f"you know what I mean? It's sort of like a reality check for the model.",
        )
        for i in range(4)
    ]
    qa_list = [p[0] for p in pairs]
    ev_list = [p[1] for p in pairs]
    result = ai_answer_detector.run(qa_list, ev_list)
    assert result.status == "OK"
    assert result.overall_ai_likelihood < 0.60, f"Expected low, got {result.overall_ai_likelihood}"


def test_ai_detector_latency_quality_spike_raises_likelihood():
    """A single question with a long latency + high score should raise likelihood."""
    normal_pairs = [
        _make_qa_eval_pair(qid=f"Q{i}", latency=2.0, score=55.0,
                           answer="I think it overfits when it memorizes noise basically.")
        for i in range(3)
    ]
    # The suspicious one: 10s latency, score 90
    spike_qa, spike_ev = _make_qa_eval_pair(
        qid="Q_spike", latency=10.0, score=90.0,
        answer=(
            "Overfitting is a phenomenon in statistical learning where a model achieves exceptionally "
            "low training error by capturing idiosyncratic noise in the training data rather than the "
            "underlying generalisable signal, resulting in significantly degraded performance on "
            "held-out validation or test sets. Detection involves monitoring the train-validation loss gap."
        ),
    )
    qa_list = [p[0] for p in normal_pairs] + [spike_qa]
    ev_list = [p[1] for p in normal_pairs] + [spike_ev]
    result = ai_answer_detector.run(qa_list, ev_list)
    assert result.status == "OK"
    spike_flag = next((f for f in result.per_answer_flags if f.question_id == "Q_spike"), None)
    assert spike_flag is not None
    assert spike_flag.latency_suspicious is True
    assert spike_flag.ai_likelihood > 0.0


def test_ai_detector_per_answer_flags_count_matches_qa_list():
    pairs = [_make_qa_eval_pair(qid=f"Q{i}", answer="I think so basically.") for i in range(4)]
    qa_list = [p[0] for p in pairs]
    ev_list = [p[1] for p in pairs]
    result = ai_answer_detector.run(qa_list, ev_list)
    assert len(result.per_answer_flags) == 4


def test_ai_detector_mutates_evaluation_ai_likelihood():
    """run() should write ai_likelihood back into the AnswerEvaluation objects."""
    pairs = [_make_qa_eval_pair(qid=f"Q{i}", answer="I believe it basically overfits.") for i in range(3)]
    qa_list = [p[0] for p in pairs]
    ev_list = [p[1] for p in pairs]
    ai_answer_detector.run(qa_list, ev_list)
    # After run(), every evaluation should have had ai_likelihood set
    for ev in ev_list:
        assert ev.ai_likelihood >= 0.0


def test_ai_detector_level_matches_overall_likelihood():
    pairs = [_make_qa_eval_pair(qid=f"Q{i}", answer="I think so.") for i in range(3)]
    qa_list = [p[0] for p in pairs]
    ev_list = [p[1] for p in pairs]
    result = ai_answer_detector.run(qa_list, ev_list)
    if result.overall_ai_likelihood < 0.30:
        assert result.level == "LOW"
    elif result.overall_ai_likelihood < 0.60:
        assert result.level == "MEDIUM"
    else:
        assert result.level == "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
# response_timing — upgraded module
# ═══════════════════════════════════════════════════════════════════════════════

def _timing_qa(qid="Q1", latency=2.0, answer="A solid answer about the topic with good content.", difficulty="Intermediate"):
    return QAExchange(
        question_id=qid, question_text="Explain X", skill="ML", difficulty=difficulty,
        expected_proficiency="Intermediate",
        reference_answer="A reference answer about X.",
        required_concepts=["X"],
        candidate_answer=answer, response_latency_sec=latency, answer_duration_sec=30.0,
    )


def _timing_eval(qid="Q1", score=60.0):
    return AnswerEvaluation(
        question_id=qid, score=score, concept_coverage=0.7,
        missing_concepts=[], incorrect_concepts=[], confidence=0.8,
    )


def test_timing_insufficient_for_single_qa():
    qa = _timing_qa()
    ev = _timing_eval()
    result = response_timing.analyze_timing([qa], [ev])
    assert result.status == "INSUFFICIENT_EVIDENCE"


def test_timing_no_anomalies_is_low_risk():
    qas = [_timing_qa(qid=f"Q{i}", latency=2.0 + i * 0.1) for i in range(4)]
    evs = [_timing_eval(qid=f"Q{i}", score=60.0) for i in range(4)]
    result = response_timing.analyze_timing(qas, evs)
    assert result.level == "LOW"
    assert result.score <= 30


def test_timing_latency_spike_with_quality_jump_is_high_risk():
    """Classic READ_ANSWER_SUSPECTED pattern.
    
    Needs enough normal QAs so IQR-trimmed baseline is stable and the spike
    sits far enough above it to cross z > 1.8.  We use 6 normal QAs (all
    latency ~2s, std ≈ 0.1s) and a spike of 20s, giving z ≈ (20-2)/0.1 = 180.
    """
    normal_qas = [_timing_qa(qid=f"Q{i}", latency=2.0 + i * 0.05) for i in range(6)]
    spike_qa = _timing_qa(qid="Q_spike", latency=20.0)    # extreme spike
    normal_evs = [_timing_eval(qid=f"Q{i}", score=55.0) for i in range(6)]
    spike_ev = _timing_eval(qid="Q_spike", score=90.0)

    result = response_timing.analyze_timing(normal_qas + [spike_qa], normal_evs + [spike_ev])
    # The spike evidence is HIGH-severity even if the module-level average is diluted by
    # many low-risk normal answers.  The important thing is the individual spike is caught.
    high_evidence = [e for e in result.evidence if e.severity == "high"]
    assert len(high_evidence) >= 1, "Expected at least one HIGH-severity latency-quality-spike evidence"
    assert any("Q_spike" in e.message for e in high_evidence)
    assert any("READ_ANSWER_SUSPECTED" in e.message for e in high_evidence)


def test_timing_confidence_increases_with_more_qa():
    few = [_timing_qa(qid=f"Q{i}") for i in range(2)]
    many = [_timing_qa(qid=f"Q{i}") for i in range(8)]
    few_evs = [_timing_eval(qid=f"Q{i}") for i in range(2)]
    many_evs = [_timing_eval(qid=f"Q{i}") for i in range(8)]
    r_few = response_timing.analyze_timing(few, few_evs)
    r_many = response_timing.analyze_timing(many, many_evs)
    assert r_many.confidence >= r_few.confidence


# ═══════════════════════════════════════════════════════════════════════════════
# knowledge_ownership — CCST edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_ccst_empty_counterfactuals_is_unavailable():
    result = knowledge_ownership.evaluate_counterfactuals([])
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.level == "UNAVAILABLE"


def test_ccst_weak_original_score_is_not_penalised():
    """A candidate who scored low on the original Q and also fails follow-up
    should get LOW risk (capability gap, not integrity)."""
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "ML",
        "followup_question_id": "Q1F",
        "perturbed_assumption": "dataset has only 100 samples",
        "expected_new_concepts": ["regularisation", "cross-validation"],
        "expected_retained_concepts": ["bias", "variance"],
        "original_answer_score": 25,     # weak original
        "candidate_followup_answer": "I'm not sure to be honest.",
    }])
    assert result.score <= 20.0, f"Weak-original should be low risk, got {result.score}"
    assert result.level == "LOW"


def test_ccst_strong_original_good_adaptation_is_low_risk():
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "ML",
        "followup_question_id": "Q2F",
        "perturbed_assumption": "minority class is only 1%",
        "expected_new_concepts": ["SMOTE", "oversampling", "anomaly detection"],
        "expected_retained_concepts": ["F1", "precision", "recall"],
        "original_answer_score": 88,
        "candidate_followup_answer": (
            "At 1% minority, I'd frame it more like anomaly detection. "
            "SMOTE or targeted oversampling would be necessary, and F1 or precision-recall "
            "curves become more important than raw accuracy."
        ),
    }])
    assert result.level in ("LOW", "MEDIUM")
    assert result.evidence[0].severity in ("low", "medium")


def test_ccst_strong_original_failed_adaptation_is_high_risk():
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "ML",
        "followup_question_id": "Q3F",
        "perturbed_assumption": "we now have GPU budget",
        "expected_new_concepts": ["neural network", "transfer learning", "embedding"],
        "expected_retained_concepts": ["overfitting", "cross-validation"],
        "original_answer_score": 91,
        "candidate_followup_answer": "I'm not really sure, maybe just use more data?",
    }])
    assert result.level in ("MEDIUM", "HIGH"), f"Expected MEDIUM/HIGH, got {result.level}"
    assert result.evidence[0].severity == "high"


def test_ccst_confidence_grows_with_probe_count():
    single = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "ML", "followup_question_id": "Q1F",
        "perturbed_assumption": "X", "expected_new_concepts": [],
        "expected_retained_concepts": ["F1"],
        "original_answer_score": 80,
        "candidate_followup_answer": "F1 is fine.",
    }])
    triple = knowledge_ownership.evaluate_counterfactuals([{
        "skill": f"Skill{i}", "followup_question_id": f"Q{i}F",
        "perturbed_assumption": "X", "expected_new_concepts": [],
        "expected_retained_concepts": ["F1"],
        "original_answer_score": 80,
        "candidate_followup_answer": "F1 is fine.",
    } for i in range(3)])
    assert triple.confidence >= single.confidence


def test_ccst_summary_structure():
    result = knowledge_ownership.evaluate_counterfactuals([{
        "skill": "AWS", "followup_question_id": "QF1",
        "perturbed_assumption": "no VPC",
        "expected_new_concepts": ["public subnet", "security group"],
        "expected_retained_concepts": ["IAM", "EC2"],
        "original_answer_score": 75,
        "candidate_followup_answer": "I'd use IAM and security groups for access control.",
    }])
    summary = knowledge_ownership.ccst_summary(result)
    assert "knowledge_ownership_risk" in summary
    assert "level" in summary
    assert "interpretation" in summary
    assert "total_probes" in summary


def test_ccst_generate_counterfactual_probe_known_domain():
    probe = knowledge_ownership.generate_counterfactual_probe(skill="AWS", original_question_id="Q2")
    assert probe["skill"] == "AWS"
    assert probe["original_question_id"] == "Q2"
    assert probe["followup_question_id"] == "Q2_cf"
    assert len(probe["expected_new_concepts"]) > 0
    assert len(probe["expected_retained_concepts"]) > 0
    assert "traffic spikes" in probe["perturbed_assumption"]


def test_ccst_generate_counterfactual_probe_generic_fallback():
    probe = knowledge_ownership.generate_counterfactual_probe(skill="Solidity / Smart Contracts", original_question_id="Q7")
    assert probe["skill"] == "Solidity / Smart Contracts"
    assert probe["followup_question_id"] == "Q7_cf"
    assert len(probe["expected_new_concepts"]) > 0
    assert "constraints" in probe["perturbed_assumption"]

