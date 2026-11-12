"""
Safe, idempotent, reversible schema changes for the organization/auth layer.

`Base.metadata.create_all` only creates *missing tables*; it never alters an existing one. The
auth work adds nullable columns to tables that may already hold real data, so those are added
here with plain `ALTER TABLE ... ADD COLUMN` (nullable, no default rewrite, existing rows keep
NULL). Nothing is dropped or rewritten by `migrate_up`.

`migrate_down` removes only the columns this module added (SQLite >= 3.35 `DROP COLUMN`); it
leaves the new tables in place so no accounts are lost - drop those by hand if truly needed.
"""
from __future__ import annotations
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

# (table, column, DDL type/constraint)
ADDED_COLUMNS: list[tuple[str, str, str]] = [
    ("candidates", "organization_id", "INTEGER REFERENCES organizations(id)"),
    ("interviews", "organization_id", "INTEGER REFERENCES organizations(id)"),
    ("interviews", "created_by_user_id", "INTEGER REFERENCES users(id)"),
    ("audio_analyses", "organization_id", "INTEGER REFERENCES organizations(id)"),
]
ADDED_INDEXES: list[tuple[str, str, str]] = [
    ("ix_candidates_organization_id", "candidates", "organization_id"),
    ("ix_interviews_organization_id", "interviews", "organization_id"),
    ("ix_audio_analyses_organization_id", "audio_analyses", "organization_id"),
]


def _columns(engine: Engine, table: str) -> set[str]:
    insp = inspect(engine)
    return {c["name"] for c in insp.get_columns(table)} if insp.has_table(table) else set()


def migrate_up(engine: Engine) -> list[str]:
    """Add any missing columns/indexes. Returns a description of each change made (empty if none)."""
    changes: list[str] = []
    with engine.begin() as conn:
        for table, column, ddl in ADDED_COLUMNS:
            cols = _columns(engine, table)
            if cols and column not in cols:
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))
                changes.append(f"{table}.{column}")
        for name, table, column in ADDED_INDEXES:
            if column in _columns(engine, table):
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ("{column}")'))
    return changes


def migrate_down(engine: Engine) -> list[str]:
    """Reverse `migrate_up` for the added columns/indexes (tables are kept)."""
    changes: list[str] = []
    with engine.begin() as conn:
        for name, _table, _column in ADDED_INDEXES:
            conn.execute(text(f'DROP INDEX IF EXISTS "{name}"'))
        for table, column, _ddl in reversed(ADDED_COLUMNS):
            if column in _columns(engine, table):
                conn.execute(text(f'ALTER TABLE "{table}" DROP COLUMN "{column}"'))
                changes.append(f"{table}.{column}")
    return changes
