"""Database models and session handling (SQLite by default, PostgreSQL via VIRALCUT_DATABASE_URL)."""

from __future__ import annotations

import datetime as dt
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import get_settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def new_id() -> str:
    """128-bit unguessable identifier. Job ids double as capability tokens (there are no accounts yet)."""
    return secrets.token_hex(16)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # uploaded -> queued -> processing -> completed | failed | canceled
    status: Mapped[str] = mapped_column(String(16), default="uploaded", index=True)
    stage: Mapped[str] = mapped_column(String(32), default="")
    stage_progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stages: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    original_filename: Mapped[str] = mapped_column(String(255), default="video")
    source_ext: Mapped[str] = mapped_column(String(8), default=".mp4")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    media: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    worker_id: Mapped[str] = mapped_column(String(64), default="")
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    queued_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    clips: Mapped[list[Clip]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Clip.rank"
    )
    logs: Mapped[list[JobLog]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobLog.id"
    )


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    # pending -> rendering -> ready | failed ; queued = waiting for a re-render requested from the studio
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float, default=0.0)  # rendered duration (after smart trimming)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    hook: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    titles: Mapped[list[str]] = mapped_column(JSON, default=list)
    caption: Mapped[str] = mapped_column(Text, default="")
    copy_source: Mapped[str] = mapped_column(String(16), default="extractive")  # extractive | llm
    language: Mapped[str] = mapped_column(String(8), default="")
    transcript: Mapped[str] = mapped_column(Text, default="")
    # user-editable state
    captions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # caption options for this clip
    pages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # caption pages (output timeline)
    text_edits: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # source-time keyed word corrections
    framing_mode: Mapped[str] = mapped_column(String(16), default="auto")
    smart_trim: Mapped[bool] = mapped_column(Boolean, default=True)
    # render results
    framing: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # per-clip editing choices: {"style", "intensity", "variant": {...}, "recommendation": {...}}
    edit: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=dict)
    qa: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    worker_id: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    job: Mapped[Job] = relationship(back_populates="clips")
    metrics: Mapped[list[ClipMetric]] = relationship(back_populates="clip", cascade="all, delete-orphan")


class JobLog(Base):
    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    level: Mapped[str] = mapped_column(String(8), default="info")
    stage: Mapped[str] = mapped_column(String(32), default="")
    message: Mapped[str] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="logs")


class ClipMetric(Base):
    """Real-world performance reported by the user for a published clip.

    Stored next to the feature vector that produced the score so the weights can later be
    calibrated against observed outcomes (see docs/SCORING.md). Nothing here is ever fabricated.
    """

    __tablename__ = "clip_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    platform: Mapped[str] = mapped_column(String(16))
    reported_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # snapshot of score inputs

    clip: Mapped[Clip] = relationship(back_populates="metrics")


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().db_url
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

        Base.metadata.create_all(_engine)
        _add_missing_columns(_engine)
        _SessionLocal = sessionmaker(_engine, expire_on_commit=False)
    return _engine


# Columns added after the first release: (table, column, SQL type). create_all() does not alter
# existing tables, so a database created by an older version gets them here (nullable, no data
# rewrite), on SQLite and PostgreSQL alike.
_LATER_COLUMNS = [("clips", "edit", "JSON")]


def _add_missing_columns(engine: Engine) -> None:
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    for table, column, sql_type in _LATER_COLUMNS:
        if column not in {c["name"] for c in insp.get_columns(table)}:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


def reset_engine() -> None:
    """Drop the cached engine (used by tests that switch data directories)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    s = _SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
