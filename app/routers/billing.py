"""
/billing/* endpoints — Stripe checkout + portal (or stubs when not configured).
"""

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models
from app.auth_utils import get_current_user

router = APIRouter(prefix="/billing", tags=["billing"])

STRIPE_SECRET_KEY    = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRO_PRICE_ID  = os.getenv("STRIPE_PRO_PRICE_ID", "")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:3000")


@router.post("/checkout")
def create_checkout(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a Stripe Checkout session for the Pro plan.
    Returns {checkout_url} — frontend redirects to it.
    Falls back to a stub URL when STRIPE_SECRET_KEY is not configured.
    """
    if not STRIPE_SECRET_KEY:
        # Dev/demo stub
        return {"checkout_url": f"{FRONTEND_URL}/app/settings?upgraded=1"}

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        # Upsert Stripe customer
        customer_id = None
        if hasattr(current_user, "stripe_customer_id") and current_user.stripe_customer_id:
            customer_id = current_user.stripe_customer_id
        else:
            customer = stripe.Customer.create(email=current_user.email, name=current_user.name)
            customer_id = customer.id
            # Persist (best-effort; column may not exist yet)
            try:
                current_user.stripe_customer_id = customer_id
                db.commit()
            except Exception:
                db.rollback()

        session = stripe.checkout.Session.create(
            customer    = customer_id,
            mode        = "subscription",
            line_items  = [{"price": STRIPE_PRO_PRICE_ID, "quantity": 1}],
            success_url = f"{FRONTEND_URL}/app/settings?upgraded=1",
            cancel_url  = f"{FRONTEND_URL}/app/settings",
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")


@router.get("/portal")
def billing_portal(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a Stripe customer portal URL, or stub."""
    if not STRIPE_SECRET_KEY:
        return {"portal_url": f"{FRONTEND_URL}/app/settings"}

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        customer_id    = getattr(current_user, "stripe_customer_id", None)
        if not customer_id:
            raise HTTPException(status_code=400, detail="No billing account found")
        session = stripe.billing_portal.Session.create(
            customer   = customer_id,
            return_url = f"{FRONTEND_URL}/app/settings",
        )
        return {"portal_url": session.url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")
