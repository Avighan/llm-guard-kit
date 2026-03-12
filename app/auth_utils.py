"""
Authentication utilities for llm-guard-kit SaaS.

Provides:
  • Password hashing / verification (passlib bcrypt)
  • JWT creation / decoding (PyJWT HS256, 30-day expiry)
  • FastAPI dependencies: get_current_user, get_admin_user, get_optional_user
  • API-key resolution (Bearer sk_... → User)
  • Fernet symmetric encryption for stored LLM / SMTP secrets
  • Google OAuth URL builder + code exchange
"""

import base64
import hashlib
import os
import time
import uuid
from typing import Optional

import bcrypt
import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session

from app.database import get_db
from app import models

# ── Bcrypt ────────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── JWT ───────────────────────────────────────────────────────────────────────

JWT_SECRET    = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_S  = 30 * 24 * 3600   # 30 days


def create_access_token(user_id: int, email: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "iat":   int(time.time()),
        "exp":   int(time.time()) + JWT_EXPIRY_S,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ── Fernet encryption for stored secrets ─────────────────────────────────────

def _fernet_key() -> bytes:
    """Derive a 32-byte Fernet key from JWT_SECRET (SHA-256 + base64url)."""
    digest = hashlib.sha256(JWT_SECRET.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_config(data: dict) -> str:
    import json
    f = Fernet(_fernet_key())
    return f.encrypt(json.dumps(data).encode()).decode()


def decrypt_config(enc: str) -> dict:
    import json
    try:
        f = Fernet(_fernet_key())
        return json.loads(f.decrypt(enc.encode()).decode())
    except Exception:
        return {}


# ── API-key generation helpers ────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (full_key, prefix, key_hash).
    full_key  = "sk_" + 40 random hex chars  (shown once)
    prefix    = first 12 chars of full_key    (stored in plain, used for lookup)
    key_hash  = bcrypt hash of full_key       (stored, used for verification)
    """
    secret     = uuid.uuid4().hex + uuid.uuid4().hex   # 64 hex chars
    full_key   = "sk_" + secret[:40]
    prefix     = full_key[:12]
    key_hash   = hash_password(full_key)
    return full_key, prefix, key_hash


def verify_api_key(full_key: str, stored_hash: str) -> bool:
    return verify_password(full_key, stored_hash)


# ── FastAPI bearer-token extractor ────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _resolve_bearer(
    credentials: Optional[HTTPAuthorizationCredentials],
    db: Session,
) -> Optional[models.User]:
    """
    Accepts either:
      • JWT token  (Bearer eyJ...)  → decode and look up user by id
      • API key    (Bearer sk_...)  → look up ApiKey by prefix, verify hash
    Returns None if no credentials supplied (for optional-auth routes).
    """
    if credentials is None:
        return None

    token = credentials.credentials

    # ── API key path ──────────────────────────────────────────────────────────
    if token.startswith("sk_"):
        prefix = token[:12]
        key_row = (
            db.query(models.ApiKey)
            .filter(models.ApiKey.key_prefix == prefix, models.ApiKey.is_active == True)  # noqa: E712
            .first()
        )
        if key_row is None or not verify_api_key(token, key_row.key_hash):
            return None
        # Increment request counter (fire-and-forget; ignore errors)
        try:
            key_row.request_count += 1
            db.commit()
        except Exception:
            db.rollback()
        return db.query(models.User).filter(models.User.id == key_row.user_id).first()

    # ── JWT path ──────────────────────────────────────────────────────────────
    payload = decode_token(token)
    user_id = int(payload.get("sub", 0))
    return db.query(models.User).filter(models.User.id == user_id, models.User.is_active == True).first()  # noqa: E712


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    """Returns User or None — for routes that optionally log chains."""
    return _resolve_bearer(credentials, db)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> models.User:
    """Returns authenticated User or raises 401."""
    user = _resolve_bearer(credentials, db)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def get_admin_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    """Returns User only if is_admin=True, else raises 403."""
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


# ── Monthly usage counter reset ───────────────────────────────────────────────

def maybe_reset_monthly_counter(user: models.User, db: Session) -> None:
    """Reset chains_this_month if we've crossed into a new calendar month."""
    now = time.gmtime()
    reset = time.gmtime(user.month_reset_at)
    if now.tm_year != reset.tm_year or now.tm_mon != reset.tm_mon:
        user.chains_this_month = 0
        user.month_reset_at = int(time.time())
        db.commit()


# ── Google OAuth ──────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
_GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def get_google_auth_url(redirect_uri: str) -> str:
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{_GOOGLE_AUTH_URL}?{qs}"


def exchange_google_code(code: str, redirect_uri: str) -> dict:
    """Exchange OAuth code for user info dict (email, name, google_id)."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")

    with httpx.Client() as client:
        token_resp = client.post(_GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        })
        token_resp.raise_for_status()
        token_data = token_resp.json()

        userinfo_resp = client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        userinfo_resp.raise_for_status()
        info = userinfo_resp.json()

    return {
        "email":     info.get("email", ""),
        "name":      info.get("name", ""),
        "google_id": info.get("sub", ""),
    }
