"""SQLite persistence layer for the AI reading agent prototype."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).with_name("reading_agent.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create required tables if they do not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                literature_excerpt TEXT,
                task_type TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('student', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reading_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                week_node TEXT NOT NULL,
                literature_title TEXT NOT NULL,
                research_question TEXT,
                theoretical_framework TEXT,
                methodology TEXT,
                evidence_chain TEXT,
                limitations_contributions TEXT,
                transfer_reflection TEXT,
                open_questions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.commit()


def create_session(student_id: str, week_node: str, literature_title: str, literature_excerpt: str, task_type: str) -> int:
    """Create a reading session and return its id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (student_id, week_node, literature_title, literature_excerpt, task_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, week_node, literature_title, literature_excerpt, task_type),
        )
        conn.commit()
        return int(cursor.lastrowid)


def add_message(session_id: int, role: str, content: str) -> None:
    """Persist a chat message."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def get_messages(session_id: int) -> list[sqlite3.Row]:
    """Return all messages for a session in chronological order."""
    with get_connection() as conn:
        return list(
            conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, id",
                (session_id,),
            )
        )


def get_sessions() -> list[sqlite3.Row]:
    """Return all sessions, newest first."""
    with get_connection() as conn:
        return list(conn.execute("SELECT * FROM sessions ORDER BY created_at DESC, id DESC"))


def upsert_reading_note(
    session_id: int,
    student_id: str,
    week_node: str,
    literature_title: str,
    research_question: str,
    theoretical_framework: str,
    methodology: str,
    evidence_chain: str,
    limitations_contributions: str,
    transfer_reflection: str,
    open_questions: str,
) -> None:
    """Create or update the reading note for one session."""
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM reading_notes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE reading_notes
                SET research_question = ?, theoretical_framework = ?, methodology = ?,
                    evidence_chain = ?, limitations_contributions = ?, transfer_reflection = ?,
                    open_questions = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (
                    research_question,
                    theoretical_framework,
                    methodology,
                    evidence_chain,
                    limitations_contributions,
                    transfer_reflection,
                    open_questions,
                    session_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO reading_notes (
                    session_id, student_id, week_node, literature_title, research_question,
                    theoretical_framework, methodology, evidence_chain,
                    limitations_contributions, transfer_reflection, open_questions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    week_node,
                    literature_title,
                    research_question,
                    theoretical_framework,
                    methodology,
                    evidence_chain,
                    limitations_contributions,
                    transfer_reflection,
                    open_questions,
                ),
            )
        conn.commit()


def get_reading_note(session_id: int) -> sqlite3.Row | None:
    """Return the note for a session, if present."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM reading_notes WHERE session_id = ?",
            (session_id,),
        ).fetchone()


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    """Convert SQLite rows into serializable dictionaries."""
    return [dict(row) for row in rows]
