"""SQLite persistence for predictions and reviewed feedback."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Generator

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base for SmartWaste database models."""


class PredictionRecord(Base):
    """Non-sensitive metadata for one inference request."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    predicted_class: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FeedbackRecord(Base):
    """User correction awaiting or having completed review."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    corrected_label: Mapped[str] = mapped_column(String(32))
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    accepted_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    """Create the engine, initialize tables, and return a session factory."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@lru_cache
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    """Return a cached process-wide session factory."""
    return create_session_factory(database_url)


def get_db(database_url: str) -> Generator[Session, None, None]:
    """Yield and close a database session for a request."""
    session = get_session_factory(database_url)()
    try:
        yield session
    finally:
        session.close()
