"""
SQLAlchemy models for the interviewer-facing setup flow (Master Spec Section
21). Kept intentionally simple for the hackathon prototype (SQLite); the
entity names mirror the spec's core-entity list so a migration to Postgres
only needs a driver swap, not a schema redesign.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    interviews = relationship("Interview", back_populates="candidate")


class Interview(Base):
    __tablename__ = "interviews"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    job_description = Column(Text, default="")
    resume_text = Column(Text, default="")
    resume_claims = Column(JSON, default=list)   # list[{skill, claim_text, claim_strength}]
    status = Column(String, default="DRAFT")     # DRAFT | READY | STARTED | COMPLETED
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="interviews")
    questions = relationship("Question", back_populates="interview", cascade="all, delete-orphan")


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"))
    question_text = Column(Text, nullable=False)
    skill = Column(String, default="")
    difficulty = Column(String, default="Basic")
    expected_proficiency = Column(String, default="Intermediate")
    reference_answer = Column(Text, default="")
    required_concepts = Column(JSON, default=list)
    ai_generated = Column(Boolean, default=False)
    order_index = Column(Integer, default=0)

    interview = relationship("Interview", back_populates="questions")
    answer = relationship("InterviewAnswer", back_populates="question", uselist=False)


class InterviewAnswer(Base):
    __tablename__ = "interview_answers"
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("questions.id"))
    transcript = Column(Text, default="")
    response_latency_sec = Column(Float, default=0.0)
    answer_duration_sec = Column(Float, default=0.0)

    question = relationship("Question", back_populates="answer")


class RiskAssessmentRecord(Base):
    __tablename__ = "risk_assessments"
    id = Column(Integer, primary_key=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"))
    result_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
