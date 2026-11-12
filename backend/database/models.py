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


class Organization(Base):
    """A tenant. Every interviewer belongs to exactly one; interviews/candidates are scoped to it."""
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True, index=True)
    is_demo = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    users = relationship("User", back_populates="organization")


class User(Base):
    """An interviewer account. `password_hash` is a bcrypt hash; plaintext is never stored."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    email = Column(String, nullable=False, unique=True, index=True)   # stored lower-case
    full_name = Column(String, default="")
    password_hash = Column(String, nullable=False)
    role = Column(String, default="interviewer")                      # interviewer | admin
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    organization = relationship("Organization", back_populates="users")


class AuthSession(Base):
    """Server-side session. Only sha256(token) is stored, so a database leak cannot be replayed."""
    __tablename__ = "auth_sessions"
    id = Column(Integer, primary_key=True)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    user = relationship("User")


class SavedReport(Base):
    """An analysis the interviewer ran from the workspace (upload review or smart analysis), kept per
    organization so past evaluations can be reopened. `result_json` is exactly what the API returned."""
    __tablename__ = "saved_reports"
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    candidate_name = Column(String, nullable=False)
    mode = Column(String, default="upload")          # upload | smart
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=True)
    result_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    # Nullable on purpose: pre-auth rows have no owner and stay untouched (claim them with
    # scripts/manage_auth.py claim-legacy). Rows without an organization are invisible to the API.
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    interviews = relationship("Interview", back_populates="candidate")


class Interview(Base):
    __tablename__ = "interviews"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
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
