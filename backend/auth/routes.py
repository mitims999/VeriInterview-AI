"""/api/auth/* - login, logout, current user, demo entry."""
from __future__ import annotations
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth import provision
from backend.auth.deps import client_ip, require_user
from backend.auth.security import (burn_password_check, hash_token, limiter, new_session_token, normalize_email,
                                   valid_email, verify_password)
from backend.config import (DEMO_USER_EMAIL, SESSION_COOKIE_NAME, SESSION_COOKIE_SAMESITE, SESSION_COOKIE_SECURE,
                            SESSION_TTL_HOURS)
from backend.database import models
from backend.database.session import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])

_GENERIC_FAIL = "Invalid email or password."


class LoginBody(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


def user_payload(user: models.User) -> dict:
    org = user.organization
    return {
        "id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role,
        "organization": {"id": org.id, "name": org.name, "slug": org.slug},
        "is_demo": bool(org.is_demo),
    }


def _start_session(db: Session, user: models.User, response: Response) -> None:
    token = new_session_token()
    now = datetime.utcnow()
    db.add(models.AuthSession(token_hash=hash_token(token), user_id=user.id, created_at=now, last_seen_at=now,
                              expires_at=now + timedelta(hours=SESSION_TTL_HOURS)))
    user.last_login_at = now
    # opportunistic cleanup of this user's dead sessions
    db.query(models.AuthSession).filter(models.AuthSession.user_id == user.id,
                                        models.AuthSession.expires_at <= now).delete()
    db.commit()
    samesite = SESSION_COOKIE_SAMESITE.lower()
    response.set_cookie(SESSION_COOKIE_NAME, token, max_age=SESSION_TTL_HOURS * 3600, httponly=True,
                        secure=SESSION_COOKIE_SECURE or samesite == "none", samesite=samesite, path="/")


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response, db: Session = Depends(get_db)):
    email = normalize_email(body.email)
    ip = client_ip(request)
    keys = [f"{email}|{ip}", f"ip|{ip}"]
    # per-(email,ip) uses the configured limit; per-ip is looser so a shared office NAT is not locked out fast
    wait = max(limiter.retry_after(keys[0]), limiter.retry_after(keys[1], limit=20))
    if wait:
        raise HTTPException(429, "Too many failed sign-in attempts. Try again later.",
                            headers={"Retry-After": str(wait)})

    user = db.query(models.User).filter(models.User.email == email).first() if valid_email(email) else None
    ok = False
    if user is not None and user.is_active and not user.organization.is_demo and user.email != DEMO_USER_EMAIL:
        ok = verify_password(body.password, user.password_hash)
    else:
        burn_password_check(body.password)
    if not ok:
        limiter.record_failure(keys[0])
        limiter.record_failure(keys[1])
        raise HTTPException(401, _GENERIC_FAIL)
    limiter.reset(keys[0])
    _start_session(db, user, response)
    return {"user": user_payload(user)}


@router.post("/demo")
def demo_login(request: Request, response: Response, db: Session = Depends(get_db)):
    """Passwordless entry to the isolated, read-only demo organization (fictional data only)."""
    ip = client_ip(request)
    key = f"demo|{ip}"
    wait = limiter.retry_after(key, limit=30, window=600)    # 30 demo sessions / 10 min / ip
    if wait:
        raise HTTPException(429, "Too many demo sessions from this address.", headers={"Retry-After": str(wait)})
    limiter.record_failure(key, window=600)
    user = provision.ensure_demo_environment(db)
    _start_session(db, user, response)
    return {"user": user_payload(user)}


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        db.query(models.AuthSession).filter(models.AuthSession.token_hash == hash_token(token)).delete()
        db.commit()
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: models.User = Depends(require_user)):
    return {"user": user_payload(user)}
