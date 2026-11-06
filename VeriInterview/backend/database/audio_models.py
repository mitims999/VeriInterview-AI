"""Additive Member 1 storage; existing tables and migrations are untouched."""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, JSON
from backend.database.models import Base


class AudioAnalysisRecord(Base):
    __tablename__ = "audio_analyses"
    id = Column(String, primary_key=True)
    candidate_id = Column(String, nullable=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"), nullable=True, index=True)
    result_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
