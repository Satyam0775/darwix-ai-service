# app/db/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from loguru import logger

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./darwix.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error(f"DB session error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables on startup."""
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")