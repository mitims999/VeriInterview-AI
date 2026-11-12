from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.config import DB_PATH, STORAGE_DIR
from backend.database.models import Base

os.makedirs(STORAGE_DIR, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    # Ensure every model module is registered before create_all (audio table lives in its own module).
    import backend.database.audio_models  # noqa: F401
    from backend.database.migrations import migrate_up
    Base.metadata.create_all(engine)
    migrate_up(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
