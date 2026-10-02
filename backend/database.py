"""
database.py
-----------
SQLite database layer for storing and retrieving analysis results.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "deepguard.db"


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────
def init_db() -> None:
    """Create tables if they don't already exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                filename                 TEXT    NOT NULL,
                visual_score             REAL    NOT NULL,
                audio_score              REAL    NOT NULL,
                lip_sync_score           REAL    NOT NULL,
                temporal_score           REAL    NOT NULL,
                real_probability         REAL    NOT NULL,
                ai_generated_probability REAL    NOT NULL,
                prediction               TEXT    NOT NULL,
                created_at               TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    logger.info("Database initialised at %s", DB_PATH)


# ─────────────────────────────────────────────
# CRUD operations
# ─────────────────────────────────────────────
def save_analysis(
    filename: str,
    visual_score: float,
    audio_score: float,
    lip_sync_score: float,
    temporal_score: float,
    real_probability: float,
    ai_generated_probability: float,
    prediction: str,
) -> int:
    """Insert a new analysis record and return its id."""
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses
                (filename, visual_score, audio_score, lip_sync_score,
                 temporal_score, real_probability, ai_generated_probability,
                 prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (filename, visual_score, audio_score, lip_sync_score,
             temporal_score, real_probability, ai_generated_probability,
             prediction),
        )
        conn.commit()
        return cursor.lastrowid


def get_all_analyses() -> list[dict]:
    """Return all analyses ordered by most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int) -> Optional[dict]:
    """Return a single analysis by id, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_analysis(analysis_id: int) -> bool:
    """Delete an analysis by id. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM analyses WHERE id = ?", (analysis_id,)
        )
        conn.commit()
    return cursor.rowcount > 0
