"""
FastAPI dependencies: who is calling, and what may they touch.

Policy (also tabulated in docs/SECURITY.md):
  * public:                 /api/health, /api/auth/login, /api/auth/demo
  * authenticated:          everything else under /api (any org user, including the demo user)
  * authenticated+writable: anything that creates/changes/deletes data or stores uploads
                            (`require_writable` -> 403 for the demo account)
Cross-organization access answers 404, never 403, so record existence is not disclosed.
"""
from __future__ import annotations
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, WebSocketException
from sqlalchemy.orm import Session
from starlette.requests import HTTPConnection

from backend.auth.security import hash_token
from backend.config import SESSION_COOKIE_NAME, SESSION_IDLE_MINUTES
from backend.database import models
from backend.database.session import get_db

_TOUCH_EVERY = timedelta(minutes=1)   # do not write last_seen on every request


def _now() -> datetime:
    return datetime.utcnow()


def client_ip(conn: HTTPConnection) -> str:
    # Behind Render/Vercel the real client is in X-Forwarded-For (first hop). Only used for rate
    # limiting, never for authorization.
    fwd = conn.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return conn.client.host if conn.client else "unknown"


def user_for_token(db: Session, token: str | None) -> models.User | None:
    if not token:
        return None
    sess = db.query(models.AuthSession).filter(models.AuthSession.token_hash == hash_token(token)).first()
    if sess is None:
        return None
    now = _now()
    if sess.expires_at <= now or now - sess.last_seen_at > timedelta(minutes=SESSION_IDLE_MINUTES):
        db.delete(sess)
        db.commit()
        return None
    user = sess.user
    if user is None or not user.is_active:
        return None
    if now - sess.last_seen_at > _TOUCH_EVERY:
        sess.last_seen_at = now
        db.commit()
    return user


def _unauthorized(conn: HTTPConnection):
    if conn.scope.get("type") == "websocket":
        raise WebSocketException(code=1008, reason="Authentication required")
    raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Cookie"})


def require_user(conn: HTTPConnection, db: Session = Depends(get_db)) -> models.User:
    user = user_for_token(db, conn.cookies.get(SESSION_COOKIE_NAME))
    if user is None:
        _unauthorized(conn)
    return user


def is_demo(user: models.User) -> bool:
    return bool(user.organization and user.organization.is_demo)


def require_writable(user: models.User = Depends(require_user)) -> models.User:
    if is_demo(user):
        raise HTTPException(403, "This action is disabled in demo mode.")
    return user


# ---- organization-scoped lookups ------------------------------------------------------------

def scoped_interview(db: Session, interview_id: int, user: models.User) -> models.Interview:
    row = (db.query(models.Interview)
           .filter(models.Interview.id == interview_id, models.Interview.organization_id == user.organization_id)
           .first())
    if row is None:
        raise HTTPException(404, "Interview not found")
    return row


def scoped_candidate(db: Session, candidate_id: int, user: models.User) -> models.Candidate:
    row = (db.query(models.Candidate)
           .filter(models.Candidate.id == candidate_id, models.Candidate.organization_id == user.organization_id)
           .first())
    if row is None:
        raise HTTPException(404, "Candidate not found")
    return row


def scoped_question(db: Session, question_id: int, user: models.User) -> models.Question:
    row = (db.query(models.Question).join(models.Interview, models.Question.interview_id == models.Interview.id)
           .filter(models.Question.id == question_id, models.Interview.organization_id == user.organization_id)
           .first())
    if row is None:
        raise HTTPException(404, "Question not found")
    return row
