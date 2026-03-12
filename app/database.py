"""
SQLAlchemy database setup.

DATABASE_URL priority:
  1. QPPG_DB env var  (e.g. postgresql://user:pw@host/db)
  2. Default: sqlite:///./qppg.db  (local development)
"""
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("QPPG_DB", "sqlite:///./qppg.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Create all tables if they don't exist, and add any new columns (called at app startup)."""
    from app import models  # noqa: F401 — ensure models are registered
    Base.metadata.create_all(bind=engine)

    # SQLite does not support ALTER TABLE ADD COLUMN IF NOT EXISTS, so we do it manually.
    if DATABASE_URL.startswith("sqlite"):
        with engine.connect() as conn:
            _add_column_if_missing(conn, "users", "stripe_customer_id", "VARCHAR(128)")

def _add_column_if_missing(conn, table: str, column: str, col_type: str) -> None:
    """Adds a column to a SQLite table if it doesn't already exist."""
    from sqlalchemy import text
    try:
        result = conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {row[1] for row in result.fetchall()}
        if column not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
            conn.commit()
    except Exception:
        pass
