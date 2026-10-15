from datetime import datetime, timezone

import config

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = config.get("DATABASE_URL") or "sqlite:///dev.db"

# Providers hand out "postgres://" or "postgresql://"; pin the psycopg2 driver
for prefix in ("postgres://", "postgresql://"):
    if DATABASE_URL.startswith(prefix):
        DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len(prefix):]
        break

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow():
    return datetime.now(timezone.utc)


def get_session():
    """FastAPI dependency: one session per request."""
    with SessionLocal() as session:
        yield session


class Base(DeclarativeBase):
    pass


class Admin(Base):
    __tablename__ = "admins"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120), default="Admin")
    role: Mapped[str] = mapped_column(String(20), default="admin")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KBEntry(Base):
    """The live, approved knowledge base."""

    __tablename__ = "kb_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    symptom: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    phrases: Mapped[list] = mapped_column(JSON, default=list)
    recommendation: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserInput(Base):
    """Every query (and feedback) the assistant sees, for monitoring."""

    __tablename__ = "user_inputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10))  # voice | chat | feedback
    outcome: Mapped[str] = mapped_column(String(20), index=True)  # exact | fuzzy | miss | helpful | unhelpful
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_symptom: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Candidate(Base):
    """A de-duplicated user input waiting for admin review."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    text_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    example_text: Mapped[str] = mapped_column(Text)
    count: Mapped[int] = mapped_column(Integer, default=1)
    best_score: Mapped[float] = mapped_column(Float, default=0.0)
    suggested_symptom: Mapped[str | None] = mapped_column(String(120), nullable=True)
    suggested_phrase: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str] = mapped_column(String(20), default="miss")  # miss | low_confidence | unhelpful
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending | approved | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class KBAudit(Base):
    """Before/after snapshots of every KB change, so any change can be reverted."""

    __tablename__ = "kb_audit"

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(20))  # create | add_phrase | update | revert
    before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    admin_email: Mapped[str] = mapped_column(String(255))
    candidate_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


def init_db():
    Base.metadata.create_all(engine)
