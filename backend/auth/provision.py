"""Create organizations/users and the demo environment. Used by startup, the CLI and the tests."""
from __future__ import annotations
import re

from sqlalchemy.orm import Session

from backend.auth.security import hash_password, normalize_email, password_problem, unusable_password_hash, valid_email
from backend.config import (BOOTSTRAP_ADMIN_EMAIL, BOOTSTRAP_ADMIN_PASSWORD, BOOTSTRAP_ORG_NAME, DEMO_ORG_SLUG,
                            DEMO_USER_EMAIL)
from backend.database import models


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"


def get_or_create_org(db: Session, name: str, *, is_demo: bool = False, slug: str | None = None) -> models.Organization:
    slug = slug or slugify(name)
    org = db.query(models.Organization).filter(models.Organization.slug == slug).first()
    if org is None:
        org = models.Organization(name=name, slug=slug, is_demo=is_demo)
        db.add(org)
        db.commit()
        db.refresh(org)
    return org


def create_user(db: Session, org: models.Organization, email: str, password: str, full_name: str = "",
                role: str = "interviewer") -> models.User:
    email = normalize_email(email)
    if not valid_email(email):
        raise ValueError("Invalid email address.")
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    if role not in ("interviewer", "admin"):
        raise ValueError("role must be 'interviewer' or 'admin'")
    if db.query(models.User).filter(models.User.email == email).first():
        raise ValueError("A user with that email already exists.")
    user = models.User(organization_id=org.id, email=email, full_name=full_name, role=role,
                       password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def ensure_demo_environment(db: Session) -> models.User:
    """Idempotent: the demo org and its password-less user. Demo content is the bundled fictional
    synthetic dataset (data/synthetic), served read-only; no real rows are ever shared with it."""
    org = get_or_create_org(db, "VeriInterview Demo (fictional data)", is_demo=True, slug=DEMO_ORG_SLUG)
    user = db.query(models.User).filter(models.User.email == DEMO_USER_EMAIL).first()
    if user is None:
        user = models.User(organization_id=org.id, email=DEMO_USER_EMAIL, full_name="Demo Interviewer",
                           role="interviewer", password_hash=unusable_password_hash())
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def bootstrap_admin_from_env(db: Session) -> bool:
    """Create the first org and admin from BOOTSTRAP_* env vars (all three required). Never overwrites."""
    if not (BOOTSTRAP_ORG_NAME and BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD):
        return False
    email = normalize_email(BOOTSTRAP_ADMIN_EMAIL)
    if db.query(models.User).filter(models.User.email == email).first():
        return False
    org = get_or_create_org(db, BOOTSTRAP_ORG_NAME)
    create_user(db, org, email, BOOTSTRAP_ADMIN_PASSWORD, full_name="Administrator", role="admin")
    return True
