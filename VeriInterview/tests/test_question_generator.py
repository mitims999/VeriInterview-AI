"""Unit tests for Member 5's question_generator.py (no existing coverage before this)."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.modules import question_generator
from backend.schemas import ResumeClaim


def test_generates_one_question_per_distinct_skill():
    claims = [
        ResumeClaim(skill="AWS", claim_text="Designed and deployed AWS architecture.", claim_strength="strong"),
        ResumeClaim(skill="Machine Learning", claim_text="Familiar with ML.", claim_strength="weak"),
    ]
    questions = question_generator.generate_questions(claims)
    assert len(questions) == 2
    assert {q["skill"] for q in questions} == {"AWS", "Machine Learning"}


def test_strongest_claims_come_first():
    claims = [
        ResumeClaim(skill="Docker", claim_text="Familiar with Docker.", claim_strength="weak"),
        ResumeClaim(skill="Kubernetes", claim_text="Led migration to Kubernetes.", claim_strength="strong"),
    ]
    questions = question_generator.generate_questions(claims)
    assert questions[0]["skill"] == "Kubernetes"


def test_every_draft_is_marked_ai_generated_and_editable():
    claims = [ResumeClaim(skill="Python", claim_text="Used Python.", claim_strength="medium")]
    questions = question_generator.generate_questions(claims)
    assert questions[0]["ai_generated"] is True
    assert questions[0]["question_text"]
    assert questions[0]["reference_answer"]


def test_unknown_skill_falls_back_to_generic_template():
    claims = [ResumeClaim(skill="Rust", claim_text="Used Rust.", claim_strength="medium")]
    questions = question_generator.generate_questions(claims)
    assert "Rust" in questions[0]["question_text"]


def test_duplicate_skill_only_generates_once():
    claims = [
        ResumeClaim(skill="AWS", claim_text="Familiar with AWS.", claim_strength="weak"),
        ResumeClaim(skill="AWS", claim_text="Deployed AWS architecture.", claim_strength="strong"),
    ]
    questions = question_generator.generate_questions(claims)
    assert len(questions) == 1
    assert questions[0]["difficulty"] == "Advanced"  # kept the strongest claim's tier


def test_generate_counterfactual_produces_followup_with_split_concepts():
    question = {
        "question_id": "Q1", "skill": "Machine Learning",
        "question_text": "How would you handle an imbalanced dataset?",
        "required_concepts": ["resampling", "class weight", "F1", "precision"],
    }
    cf = question_generator.generate_counterfactual(question)
    assert cf["skill"] == "Machine Learning"
    assert cf["followup_question_id"] == "Q1F"
    assert cf["expected_new_concepts"]
    assert cf["expected_retained_concepts"]


def test_export_to_ideal_answers_format_round_trips_through_intake_parser():
    from backend.modules import intake_parser
    claims = [ResumeClaim(skill="AWS", claim_text="Designed AWS systems.", claim_strength="strong")]
    questions = question_generator.generate_questions(claims)
    text = question_generator.export_to_ideal_answers_format(questions)
    parsed = intake_parser.parse_ideal_answers(text)
    assert "Q1" in parsed
    assert parsed["Q1"]["skill"] == "AWS"
    assert parsed["Q1"]["question_text"] == questions[0]["question_text"]
