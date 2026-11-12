"""Password hashing, session tokens and the login rate limiter. No framework imports here."""
from __future__ import annotations
import hashlib
import re
import secrets
import threading
import time
from collections import defaultdict, deque

import bcrypt

from backend.config import LOGIN_MAX_FAILURES, LOGIN_WINDOW_SECONDS, MIN_PASSWORD_LENGTH

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email)) and len(email) <= 254


def password_problem(password: str) -> str | None:
    """Return a human-readable reason a password is unacceptable, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password.encode("utf-8")) > 72:   # bcrypt silently truncates beyond 72 bytes - refuse instead
        return "Password must be at most 72 bytes."
    return None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


# A real bcrypt hash of a random value, verified against when the email is unknown so that
# "no such user" and "wrong password" cost the same time (no account enumeration by timing).
_DUMMY_HASH = hash_password(secrets.token_urlsafe(24))


def burn_password_check(password: str) -> None:
    verify_password(password, _DUMMY_HASH)


def unusable_password_hash() -> str:
    """A hash of 32 random bytes nobody ever sees - the account cannot be logged into with a password."""
    return hash_password(secrets.token_urlsafe(32))


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LoginRateLimiter:
    """
    Sliding-window failure counter keyed by arbitrary strings (we use `email|ip` and `ip`).
    In-memory: correct for a single process, resets on restart. Behind several workers, move it
    to a shared store (documented in docs/SECURITY.md).
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float, window: int) -> deque[float]:
        q = self._events[key]
        while q and now - q[0] > window:
            q.popleft()
        return q

    def retry_after(self, key: str, limit: int = LOGIN_MAX_FAILURES, window: int = LOGIN_WINDOW_SECONDS) -> int:
        """Seconds until another attempt is allowed (0 = allowed now)."""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now, window)
            if len(q) >= limit:
                return max(1, int(window - (now - q[0])) + 1)
            return 0

    def record_failure(self, key: str, window: int = LOGIN_WINDOW_SECONDS) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now, window).append(now)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._events.clear()
            else:
                self._events.pop(key, None)


limiter = LoginRateLimiter()
