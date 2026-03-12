"""
/auth/* endpoints — registration, login, Google OAuth, profile.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.auth_utils import (
    hash_password, verify_password,
    create_access_token,
    get_current_user,
    get_google_auth_url, exchange_google_code,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    str
    password: str
    name:     str = ""


class LoginRequest(BaseModel):
    email:    str
    password: str


class GoogleCallbackRequest(BaseModel):
    code:         str
    redirect_uri: str


class ProfileRequest(BaseModel):
    name: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_first_user(db: Session) -> bool:
    return db.query(models.User).count() == 0


def _create_token_response(user: models.User) -> dict:
    return {"access_token": create_access_token(user.id, user.email)}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account. First registered user becomes admin."""
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = models.User(
        email         = req.email.lower().strip(),
        name          = req.name or req.email.split("@")[0],
        password_hash = hash_password(req.password),
        is_admin      = _is_first_user(db),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create default onboarding state
    db.add(models.OnboardingState(user_id=user.id))
    db.commit()

    return _create_token_response(user)


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(
        models.User.email     == req.email.lower().strip(),
        models.User.is_active == True,  # noqa: E712
    ).first()
    if user is None or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _create_token_response(user)


@router.get("/google")
def google_login(redirect_uri: str, request: Request):
    """Redirect browser to Google OAuth consent screen."""
    return RedirectResponse(url=get_google_auth_url(redirect_uri))


@router.post("/google/callback")
def google_callback(req: GoogleCallbackRequest, db: Session = Depends(get_db)):
    """Exchange Google OAuth code for JWT."""
    info = exchange_google_code(req.code, req.redirect_uri)

    user = db.query(models.User).filter(models.User.google_id == info["google_id"]).first()
    if user is None:
        # Try by email (user may have registered with password before)
        user = db.query(models.User).filter(models.User.email == info["email"]).first()

    if user is None:
        user = models.User(
            email     = info["email"],
            name      = info["name"],
            google_id = info["google_id"],
            is_admin  = _is_first_user(db),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(models.OnboardingState(user_id=user.id))
        db.commit()
    elif user.google_id is None:
        user.google_id = info["google_id"]
        if not user.name:
            user.name = info["name"]
        db.commit()

    return _create_token_response(user)


@router.get("/me")
def me(current_user: models.User = Depends(get_current_user)):
    return {
        "id":                 current_user.id,
        "email":              current_user.email,
        "name":               current_user.name,
        "plan":               current_user.plan,
        "is_admin":           current_user.is_admin,
        "chains_this_month":  current_user.chains_this_month,
        "created_at":         current_user.created_at,
    }


@router.post("/profile")
def update_profile(
    req: ProfileRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user.name = req.name.strip()
    db.commit()
    return {"status": "ok"}


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password:     str


@router.put("/password")
def change_password(
    req: PasswordChangeRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password. Requires current password for verification."""
    if not current_user.password_hash:
        raise HTTPException(status_code=400, detail="Account uses OAuth login — no password to change")
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"status": "ok"}
